/* ============================================================
   Victoria Line Motion Lab — app logic
   ------------------------------------------------------------
   Key fixes vs. the original:
   - Rendering is driven by a self-rescheduling requestAnimationFrame
     loop (max one redraw per display frame), not by every sensor
     event. The old "rAF is unreliable on iPhone" comment was a
     misdiagnosis of a loop that stopped rescheduling itself.
   - Canvas sizing never writes inline styles (the old code pinned
     the canvas at its first width forever) and uses clientWidth/
     clientHeight consistently for both the bitmap and drawing.
   - Timestamps use the monotonic performance.now() clock internally;
     exports still contain epoch milliseconds, so the JSON format is
     backwards-compatible with the original.
   - The devicemotion listener is attached only while recording, so
     iOS can power the sensors down when idle (battery).
   - Window lookups use binary search and the draw path allocates
     nothing per frame (no GC hitches on older iPhones).
   - Native ctx.clip() replaces ~80 lines of hand-rolled
     Cohen–Sutherland clipping and the ctx.__lastPoint hack.
   - Saving uses the iOS share sheet (Web Share API) when available,
     falling back to a classic download elsewhere.

   ML Integration:
   - Phase 1: Post-hoc platform prediction after recording
   - Phase 2: Live streaming prediction during recording (~every 5 seconds)
   ============================================================ */

"use strict";

// Dynamic imports for ML modules (loaded asynchronously to avoid blocking startup)
let mlModules = null;
async function loadMLModules() {
  if (mlModules) return mlModules;
  const [features, classifier, trainingSet] = await Promise.all([
    import("./features.js"),
    import("./classifier.js"),
    import("./training-set.js"),
  ]);
  mlModules = { features, classifier, trainingSet };
  return mlModules;
}

(() => {
  // ---------- Configuration ----------------------------------------------

  const TIME_WINDOW_MS = 8000;     // visible time span of the charts
  const MAX_BUFFER_POINTS = 20000; // raw history retained for export
  const TRIM_CHUNK = 2000;         // trim this many at once (amortised, not per-event)
  const STATS_UPDATE_MS = 150;     // throttle for the text counters
  const SENSOR_WATCHDOG_MS = 2500; // "no data arriving" warning delay
  const GRAVITY = 9.81;            // m/s² per g

  // Series definitions — colours kept in sync with style.css variables.
  const ACCEL_CHART = {
    yMin: -4, yMax: 4, step: 1,
    label: (v) => `${v.toFixed(1)} g`,
    series: [
      { color: "#ff375f", accessor: (e) => e.ax / GRAVITY },
      { color: "#32d74b", accessor: (e) => e.ay / GRAVITY },
      { color: "#64d2ff", accessor: (e) => e.az / GRAVITY },
    ],
  };

  const GYRO_CHART = {
    yMin: -360, yMax: 360, step: 90,
    label: (v) => `${v.toFixed(0)}\u00B0/s`,
    series: [
      { color: "#ffd60a", accessor: (e) => e.rotationAlpha },
      { color: "#ff9f0a", accessor: (e) => e.rotationBeta },
      { color: "#bf5af2", accessor: (e) => e.rotationGamma },
    ],
  };

  // ---------- DOM lookup (fail fast, visibly) -----------------------------

  function requireElement(id) {
    const el = document.getElementById(id);
    if (!el) throw new Error(`Required element #${id} is missing from index.html`);
    return el;
  }

  let ui;
  try {
    ui = {
      sensorBtn: requireElement("sensorBtn"),
      recordBtn: requireElement("recordBtn"),
      clearBtn: requireElement("clearBtn"),
      generateTestDataBtn: requireElement("generateTestDataBtn"),
      backupBtn: requireElement("backupBtn"),
      restoreBtn: requireElement("restoreBtn"),
      historyCard: requireElement("recordingsList").parentElement,
      recordingsList: requireElement("recordingsList"),
      toggleHistoryBtn: requireElement("toggleHistoryBtn"),
      sensorStatus: requireElement("sensorStatus"),
      sessionState: requireElement("sessionState"),
      sampleCount: requireElement("sampleCount"),
      sampleRate: requireElement("sampleRate"),
      duration: requireElement("duration"),
      accelCanvas: requireElement("accelChart"),
      gyroCanvas: requireElement("gyroChart"),
      trainingStatus: requireElement("trainingStatus"),
      liveForcastCard: requireElement("liveForcastCard"),
      forecastPlatform: requireElement("forecastPlatform"),
      forecastConfidence: requireElement("forecastConfidence"),
      forecastNote: requireElement("forecastNote"),
    };
  } catch (err) {
    console.error("[MotionLab] Startup failed:", err);
    const msg = document.createElement("p");
    msg.className = "noscript-message";
    msg.textContent = `App failed to start: ${err.message}`;
    document.body.prepend(msg);
    return; // nothing else can work without the UI
  }

  // alpha:false lets Safari skip compositing an alpha channel for the charts.
  const accelCtx = ui.accelCanvas.getContext("2d", { alpha: false });
  const gyroCtx = ui.gyroCanvas.getContext("2d", { alpha: false });
  if (!accelCtx || !gyroCtx) {
    console.error("[MotionLab] Canvas 2D context unavailable.");
    setSessionState("Charts unavailable: canvas not supported");
    return;
  }

  // ---------- State --------------------------------------------------------

  const state = {
    recording: false,
    data: [],            // { time, ax, ay, az, rotationAlpha, rotationBeta, rotationGamma }
    startTime: null,     // performance.now() ms at recording start
    dataTrimmed: false,  // true once the rolling buffer has dropped old samples
    unsaved: false,      // there is a finished recording not yet exported
    rafId: null,         // current requestAnimationFrame handle
    lastStatsAt: 0,      // last counter-text update (rAF timestamp)
    watchdogId: null,    // "no sensor data" timer
    epochBase: Date.now() - performance.now(),

    // ML state (Phase 1 & 2)
    trainSet: null,           // TrainingSet instance (loaded on init)
    classifier: null,         // cached KNNClassifier (Phase 2 live predictions)
    classifierBuiltFor: -1,   // trainSet.count() the cached classifier was built from
    lastPredictionAt: 0,      // timestamp of last live prediction (Phase 2)
    lastLivePrediction: null, // { label, confidence, neighbors } from Phase 2
  };

  // ---------- Small helpers ------------------------------------------------

  function setSessionState(text) {
    ui.sessionState.textContent = text;
  }

  function setPill(kind, text) {
    // kind: "pending" | "granted" | "denied" — styled in style.css
    ui.sensorStatus.textContent = text;
    ui.sensorStatus.className = `status-pill status-pill--${kind}`;
  }

  // Haptic tick. Android browsers vibrate; iOS Safari has no Vibration API,
  // so this is a deliberate no-op there (iOS exposes no web haptics today).
  function hapticTick(ms) {
    try {
      navigator.vibrate?.(ms);
    } catch {
      /* ignore — purely cosmetic */
    }
  }

  // First index whose .time >= t (data is sorted; performance.now is monotonic).
  function firstIndexAtOrAfter(arr, t) {
    let lo = 0;
    let hi = arr.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (arr[mid].time < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  }

  // Align odd line widths to the pixel grid for crisp 1px rules.
  function crisp(v) {
    return Math.round(v) + 0.5;
  }

  function timestampForFilename() {
    const d = new Date();
    const p = (n) => String(n).padStart(2, "0");
    return `${d.getFullYear()}-${p(d.getMonth() + 1)}-${p(d.getDate())}_` +
           `${p(d.getHours())}-${p(d.getMinutes())}-${p(d.getSeconds())}`;
  }

  // ---------- Canvas sizing ------------------------------------------------

  function resizeCanvas(canvas, ctx) {
    const dpr = window.devicePixelRatio || 1;

    // clientWidth/Height = CSS content box (excludes the 1px border), which is
    // exactly the area the bitmap is displayed in. The original mixed
    // offsetWidth (resize) with clientWidth (draw), skewing the scale slightly.
    const cssWidth = Math.max(1, canvas.clientWidth);
    const cssHeight = Math.max(1, canvas.clientHeight);

    const pixelWidth = Math.round(cssWidth * dpr);
    const pixelHeight = Math.round(cssHeight * dpr);

    if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
      canvas.width = pixelWidth;
      canvas.height = pixelHeight;
      // IMPORTANT: never set canvas.style.width/height here. The original did,
      // which overrode the CSS `width:100%` and froze the layout size forever.
    }

    // Draw in CSS-pixel coordinates over a high-resolution backing store.
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  }

  function resizeAllCanvases() {
    resizeCanvas(ui.accelCanvas, accelCtx);
    resizeCanvas(ui.gyroCanvas, gyroCtx);
    scheduleRender();
  }

  function setupResizeHandling() {
    if (typeof ResizeObserver !== "undefined") {
      // Covers rotation, split view, URL-bar changes — anything that resizes
      // the canvases — without the stacked setTimeout hacks of the original.
      const observer = new ResizeObserver(() => resizeAllCanvases());
      observer.observe(ui.accelCanvas);
      observer.observe(ui.gyroCanvas);
    } else {
      // Fallback for very old browsers.
      let pendingResize = null;
      const onResize = () => {
        if (pendingResize !== null) clearTimeout(pendingResize); // no stacking
        pendingResize = setTimeout(() => {
          pendingResize = null;
          resizeAllCanvases();
        }, 120);
      };
      window.addEventListener("resize", onResize);
      window.addEventListener("orientationchange", onResize);
    }
  }

  // ---------- ML Platform Prediction (Phase 1 & 2) -------------------------

  /**
   * Get the (lazily created) training set, loading any persisted examples.
   */
  async function ensureTrainSet() {
    if (!state.trainSet) {
      const ml = await loadMLModules();
      state.trainSet = new ml.trainingSet.TrainingSet();
      await state.trainSet.load();
    }
    return state.trainSet;
  }

  /**
   * Build a fresh classifier from the current training set.
   * Examples without a feature vector (old-format imports) are skipped.
   */
  async function buildClassifier() {
    const ml = await loadMLModules();
    const trainSet = await ensureTrainSet();
    const clf = new ml.classifier.KNNClassifier(5);
    for (const ex of trainSet.examples) {
      if (!ex.features || typeof ex.features !== "object") continue;
      clf.addTrainingExample(ex.features, ex.label, {
        recordingId: ex.recordingId,
        timestamp: ex.timestamp,
      });
    }
    return clf;
  }

  /**
   * Generate fake training data for testing (10 examples: 5 left, 5 right).
   * Useful for validating Phase 1 & 2 without needing real motion data.
   */
  async function generateFakeTrainingData() {
    try {
      ui.generateTestDataBtn.disabled = true;

      const ml = await loadMLModules();
      await ensureTrainSet();

      // Re-tapping the button replaces previous FAKE examples instead of
      // stacking duplicates. Real labeled trips are left untouched.
      for (const ex of [...state.trainSet.examples]) {
        if (ex.notes && ex.notes.startsWith("Fake")) {
          state.trainSet.remove(ex.id);
        }
      }

      // Each fake trip mimics the physics of taking a fork IN THE WORLD
      // FRAME — a sustained yaw rotation about the world's vertical axis
      // plus a sustained lateral push — then expresses it in a RANDOM
      // device orientation per example. That makes the world-frame yaw the
      // only signal consistent within a class, which is exactly the
      // property a real phone (held however) has on a real train. The
      // classifier's Fisher weighting discovers this on its own.

      // Random orthonormal basis; rows map world vectors to device coords.
      const randomOrientation = () => {
        const rand = () => Math.random() * 2 - 1;
        let r1, n1;
        do {
          r1 = [rand(), rand(), rand()];
          n1 = Math.hypot(...r1);
        } while (n1 < 0.1);
        r1 = r1.map((v) => v / n1);

        let r2, n2;
        do {
          const t = [rand(), rand(), rand()];
          const d = t[0] * r1[0] + t[1] * r1[1] + t[2] * r1[2];
          r2 = t.map((v, i) => v - d * r1[i]);
          n2 = Math.hypot(...r2);
        } while (n2 < 0.1);
        r2 = r2.map((v) => v / n2);

        const r3 = [
          r1[1] * r2[2] - r1[2] * r2[1],
          r1[2] * r2[0] - r1[0] * r2[2],
          r1[0] * r2[1] - r1[1] * r2[0],
        ];
        return [r1, r2, r3];
      };
      const toDevice = (R, w) => [
        R[0][0] * w[0] + R[0][1] * w[1] + R[0][2] * w[2],
        R[1][0] * w[0] + R[1][1] * w[1] + R[1][2] * w[2],
        R[2][0] * w[0] + R[2][1] * w[1] + R[2][2] * w[2],
      ];

      const makeFakeTrip = (direction, turnRateDegPerSec) => {
        const sign = direction === "left" ? 1 : -1;
        const R = randomOrientation();
        // World frame: up = +z. Left fork = anticlockwise yaw (positive
        // about up, right-hand rule) + lateral push; gravity reaction +9.81.
        const rotD = toDevice(R, [0, 0, sign * turnRateDegPerSec]); // (ωx, ωy, ωz)
        const accD = toDevice(R, [sign * 0.3, 0, 0]);
        const gD = toDevice(R, [0, 0, 9.81]);
        const noise = (scale) => (Math.random() - 0.5) * 2 * scale;
        return Array.from({ length: 3000 }, () => ({
          ax: accD[0] + noise(1),
          ay: accD[1] + noise(1),
          az: accD[2] + noise(1),
          // rotationRate axis mapping: alpha = about z, beta = x, gamma = y
          rotationAlpha: rotD[2] + noise(15),
          rotationBeta: rotD[0] + noise(15),
          rotationGamma: rotD[1] + noise(15),
          gx: gD[0] + noise(0.05),
          gy: gD[1] + noise(0.05),
          gz: gD[2] + noise(0.05),
        }));
      };

      for (let i = 0; i < 5; i++) {
        const turnRate = 30 + i * 4; // 30–46 °/s, varied per example
        const left = ml.features.extractFeatures(makeFakeTrip("left", turnRate), 60);
        state.trainSet.add(left, "left", { notes: `Fake left ${i}` });
        const right = ml.features.extractFeatures(makeFakeTrip("right", turnRate), 60);
        state.trainSet.add(right, "right", { notes: `Fake right ${i}` });
      }

      await state.trainSet.save();

      console.log("[ML] Generated 10 fake training examples");
      alert(
        "✓ Created 10 fake training examples (5 left, 5 right).\n\n" +
          "To test: record ~10 seconds while steadily turning the phone " +
          "(or yourself, in a swivel chair) in one direction — ANY phone " +
          "orientation works.\n\n" +
          "Turn anticlockwise (seen from above) → LEFT\n" +
          "Turn clockwise → RIGHT\n\n" +
          "(Some devices report mirrored signs — then the labels swap. " +
          "What matters is the two directions give opposite, consistent " +
          "answers.)"
      );

      await updateTrainingStatus();
    } catch (err) {
      console.error("[ML] Failed to generate fake data:", err);
      alert(`Failed: ${err.message}`);
    } finally {
      ui.generateTestDataBtn.disabled = false;
    }
  }

  /**
   * Update the training status text to show current dataset size.
   */
  async function updateTrainingStatus() {
    if (!state.trainSet) return;
    const stats = state.trainSet.getStats();
    const { leftCount, rightCount, isReadyForTraining, totalExamples } = stats;
    const text = `Training data: ${leftCount}L / ${rightCount}R${
      isReadyForTraining ? " ✓ Ready to predict" : " (need 5+ of each)"
    }`;
    ui.trainingStatus.textContent = text;
    ui.trainingStatus.classList.remove("hidden");

    // Enable backup button if there's any training data to back up
    ui.backupBtn.disabled = totalExamples === 0;

    // Show/update recordings history if there's training data
    if (totalExamples > 0) {
      ui.historyCard.classList.remove("hidden");
      renderRecordingsList();
    } else {
      ui.historyCard.classList.add("hidden");
    }
  }

  function renderRecordingsList() {
    if (!state.trainSet || state.trainSet.count() === 0) {
      ui.recordingsList.innerHTML = "<p style='text-align: center; color: var(--text-muted); margin: 0;'>No recordings yet</p>";
      return;
    }

    // Sort by timestamp, newest first
    const sorted = [...state.trainSet.examples].sort((a, b) => b.timestamp - a.timestamp);

    ui.recordingsList.innerHTML = sorted.map((ex, idx) => {
      const date = new Date(ex.timestamp);
      const timeStr = date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
      const dateStr = date.toLocaleDateString([], { month: "short", day: "numeric" });
      const durationStr = ex.duration ? `${ex.duration}s` : "—";
      const labelClass = ex.label === "left" ? "recording-label-left" : "recording-label-right";
      const labelEmoji = ex.label === "left" ? "←" : "→";

      return `
        <div class="recording-item">
          <div class="recording-info">
            <div class="recording-label ${labelClass}">
              ${labelEmoji} ${ex.label.toUpperCase()}
            </div>
            <div class="recording-timestamp">${dateStr} at ${timeStr}</div>
          </div>
          <div class="recording-duration">${durationStr}</div>
          <button type="button" class="recording-delete" data-id="${ex.id}" title="Delete this recording">
            ✕
          </button>
        </div>
      `;
    }).join("");

    // Wire up delete buttons
    ui.recordingsList.querySelectorAll(".recording-delete").forEach(btn => {
      btn.addEventListener("click", (e) => {
        const id = e.target.dataset.id;
        if (window.confirm("Delete this recording?")) {
          state.trainSet.remove(id);
          state.trainSet.save().then(() => {
            updateTrainingStatus();
          });
        }
      });
    });
  }

  /**
   * Phase 2: Make a live prediction from the last N seconds of data.
   * Called every 5 seconds during recording.
   */
  async function makeLivePrediction() {
    if (!state.trainSet || state.trainSet.count() < 5) return;
    if (state.data.length < 30) return; // need at least 0.5s of data

    try {
      const ml = await loadMLModules();

      // Get the last 10 seconds of data
      const windowMs = 10000;
      const latest = state.data[state.data.length - 1]?.time ?? 0;
      const windowStart = latest - windowMs;
      const idx = firstIndexAtOrAfter(state.data, windowStart);
      const window = state.data.slice(idx);

      if (window.length < 30) return;

      // Extract features
      const features = ml.features.extractFeatures(window, 60);

      // Build the classifier once per recording; rebuild only if the
      // training set has changed since (labelling happens between trips).
      if (!state.classifier || state.classifierBuiltFor !== state.trainSet.count()) {
        state.classifier = await buildClassifier();
        state.classifierBuiltFor = state.trainSet.count();
      }

      // Predict
      const prediction = state.classifier.predict(features);
      if (!prediction.label) return; // no usable training examples
      state.lastLivePrediction = prediction;

      // Update UI (colour via class — the CSP forbids style attributes)
      ui.forecastPlatform.textContent = prediction.label.toUpperCase();
      ui.forecastPlatform.classList.toggle("platform-left", prediction.label === "left");
      ui.forecastPlatform.classList.toggle("platform-right", prediction.label === "right");
      ui.forecastConfidence.textContent = `Confidence: ${(
        prediction.confidence * 100
      ).toFixed(0)}%`;
      ui.forecastNote.textContent = `${prediction.rawVotes.left}/${
        prediction.rawVotes.right
      } neighbors`;
    } catch (err) {
      console.warn("[ML] Live prediction failed:", err);
    }
  }

  /**
   * Phase 1: Post-hoc prediction after recording stops.
   */
  /**
   * Show a modal to label the current recording.
   */
  async function showLabelingDialog() {
    return new Promise((resolve) => {
      const modal = document.createElement("div");
      modal.className = "modal-overlay";

      const content = document.createElement("div");
      content.className = "modal-content";

      content.innerHTML = `
        <h2 class="modal-title">Label This Trip</h2>
        <p class="modal-subtitle">Which platform did you arrive at?</p>
        <div class="modal-buttons">
          <button class="btn btn-primary modal-btn" data-answer="left">← LEFT</button>
          <button class="btn btn-primary modal-btn" data-answer="right">RIGHT →</button>
        </div>
        <button class="btn btn-outline modal-btn-full" data-answer="skip">Skip</button>
      `;

      modal.appendChild(content);
      document.body.appendChild(modal);

      content.querySelectorAll("button").forEach((btn) => {
        btn.addEventListener("click", () => {
          const answer = btn.dataset.answer;
          modal.remove();
          resolve(answer);
        });
      });

      // Close on background tap
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          modal.remove();
          resolve("skip");
        }
      });
    });
  }

  /**
   * Save the current recording to the training set.
   */
  async function labelAndSaveToTrainingSet(label) {
    if (label === "skip" || !["left", "right"].includes(label)) return;

    try {
      const ml = await loadMLModules();
      await ensureTrainSet();

      // Extract features from the recording (lightweight storage)
      const features = ml.features.extractFeatures(state.data, 60);

      // Add features + full raw motion data to training set
      const durationSec = state.data.length > 0 ? (state.data[state.data.length - 1].time / 1000).toFixed(1) : "0";
      const rawMotionData = state.data.map((s) => ({
        time: Math.round(state.epochBase + s.time),
        ax: s.ax,
        ay: s.ay,
        az: s.az,
        rotationAlpha: s.rotationAlpha,
        rotationBeta: s.rotationBeta,
        rotationGamma: s.rotationGamma,
        gx: s.gx ?? 0,
        gy: s.gy ?? 0,
        gz: s.gz ?? 0,
      }));

      state.trainSet.add(features, label, {
        recordingId: ml.trainingSet.generateRecordingId(),
        timestamp: Date.now(),
        sampleCount: state.data.length,
        duration: parseFloat(durationSec),
        rawMotionData, // full raw motion data for this recording
        notes: `User-labeled ${label} platform`,
      });

      // Save to persistent storage
      await state.trainSet.save();

      // Auto-backup (silent): store a copy in localStorage and trigger an
      // invisible auto-export that the user can download or restore later
      await state.trainSet.autoBackup();

      // Update UI
      const stats = state.trainSet.getStats();
      alert(
        `✓ Saved to training set!\n\nTotal: ${stats.totalExamples} examples\nLeft: ${stats.leftCount} | Right: ${stats.rightCount}\n\nBackup saved automatically. You can download it anytime via the "⬇️ Backup Data" button.`
      );

      await updateTrainingStatus();
    } catch (err) {
      console.error("[ML] Save to training set failed:", err);
      alert(`Failed to save: ${err.message}`);
    }
  }

  // ---------- Render loop --------------------------------------------------

  // At most one redraw per display frame. While recording, the loop keeps
  // itself alive; when idle it runs once per scheduleRender() call.
  function scheduleRender() {
    if (state.rafId === null && !document.hidden) {
      state.rafId = requestAnimationFrame(renderFrame);
    }
  }

  function renderFrame(now) {
    state.rafId = null;

    try {
      renderChart(accelCtx, ACCEL_CHART);
      renderChart(gyroCtx, GYRO_CHART);
    } catch (err) {
      console.error("[MotionLab] Render error:", err);
    }

    // Text counters are throttled — updating DOM text at sensor rate is
    // pointless work for the layout engine.
    if (now - state.lastStatsAt > STATS_UPDATE_MS) {
      state.lastStatsAt = now;
      updateSessionInfo();
    }

    // Phase 2: Live prediction every ~5 seconds during recording
    if (state.recording && now - state.lastPredictionAt > 5000) {
      state.lastPredictionAt = now;
      makeLivePrediction(); // async, doesn't block render
    }

    if (state.recording) scheduleRender();
  }

  function renderChart(ctx, chart) {
    const width = ctx.canvas.clientWidth;
    const height = ctx.canvas.clientHeight;
    if (width < 2 || height < 2) return;

    drawAxes(ctx, width, height, chart);

    const data = state.data;
    if (data.length < 2) return;

    const latest = data[data.length - 1].time;
    const windowStart = latest - TIME_WINDOW_MS;

    // While the recording is shorter than the window the trace grows from the
    // left edge; afterwards it scrolls (same behaviour as the original).
    const xOrigin = data[0].time > windowStart ? data[0].time : windowStart;

    // Include one sample before the window so the trace enters from off-screen
    // left; ctx.clip() trims it at the edge (replaces the old interpolation).
    let firstIndex = firstIndexAtOrAfter(data, windowStart);
    if (firstIndex > 0) firstIndex -= 1;

    const plotWidth = Math.max(2, width - 2); // 1px inset on each side
    const range = chart.yMax - chart.yMin;
    const toY = (v) => height - ((v - chart.yMin) / range) * height;
    const toX = (t) => 1 + ((t - xOrigin) / TIME_WINDOW_MS) * (plotWidth - 1);

    ctx.save();

    // Native clipping: segments that leave the plot are cut off correctly,
    // with none of the top/bottom "smearing" the original hand-rolled
    // clipper was written to avoid.
    ctx.beginPath();
    ctx.rect(1, 0, plotWidth, height);
    ctx.clip();

    ctx.lineWidth = 1.5; // a touch bolder than 1px — clearer on 3x displays
    ctx.lineJoin = "round";
    ctx.lineCap = "round";

    for (const series of chart.series) {
      strokeSeries(ctx, data, firstIndex, toX, toY, series);
    }

    ctx.restore();
  }

  function strokeSeries(ctx, data, firstIndex, toX, toY, series) {
    ctx.strokeStyle = series.color;
    ctx.beginPath();

    let started = false;
    let lastX = -Infinity;
    const lastIndex = data.length - 1;

    for (let i = firstIndex; i <= lastIndex; i += 1) {
      const value = series.accessor(data[i]);

      if (!Number.isFinite(value)) {
        // Bad sample: break the line rather than drawing garbage.
        started = false;
        continue;
      }

      const x = toX(data[i].time);

      // Thin samples that land on the same horizontal pixel (always keep the
      // newest one) — same effect as the original per-column de-duplication,
      // with no per-frame array allocation.
      if (started && x - lastX < 0.5 && i < lastIndex) continue;

      const y = toY(value);
      if (started) {
        ctx.lineTo(x, y);
      } else {
        ctx.moveTo(x, y);
        started = true;
      }
      lastX = x;
    }

    ctx.stroke();
  }

  function drawAxes(ctx, width, height, chart) {
    // alpha:false context — the opaque fill is the clear.
    ctx.fillStyle = "#05060a";
    ctx.fillRect(0, 0, width, height);

    const range = chart.yMax - chart.yMin;
    const toY = (v) => height - ((v - chart.yMin) / range) * height;

    ctx.save();
    ctx.strokeStyle = "#242838";
    ctx.fillStyle = "#9aa0b5";
    ctx.lineWidth = 1;
    // Constant CSS-pixel size. The original shrank this by devicePixelRatio
    // (7.5px on every modern iPhone) even though the context is already
    // DPR-scaled — labels were illegibly small on exactly the target devices.
    ctx.font = "10px -apple-system, BlinkMacSystemFont, system-ui, sans-serif";
    ctx.textAlign = "left";
    ctx.textBaseline = "middle";

    for (let value = chart.yMin; value <= chart.yMax + 1e-9; value += chart.step) {
      const y = crisp(toY(value));

      ctx.beginPath();
      ctx.moveTo(0, y);
      ctx.lineTo(width, y);
      ctx.stroke();

      // Clamp so the top/bottom labels aren't half-cut by the canvas edge.
      const labelY = Math.max(8, Math.min(height - 8, y));
      ctx.fillText(chart.label(value), 6, labelY);
    }

    // Stronger zero line.
    const zeroY = toY(0);
    if (zeroY >= 0 && zeroY <= height) {
      ctx.strokeStyle = "#3a415a";
      ctx.beginPath();
      ctx.moveTo(0, crisp(zeroY));
      ctx.lineTo(width, crisp(zeroY));
      ctx.stroke();
    }

    ctx.restore();
  }

  // ---------- Session info -------------------------------------------------

  function updateSessionInfo() {
    const data = state.data;

    ui.sampleCount.textContent = state.dataTrimmed
      ? `${data.length} (oldest trimmed)`
      : String(data.length);

    // Live sample rate over the most recent second of data.
    let rateText = "\u2014";
    if (data.length >= 2) {
      const latest = data[data.length - 1].time;
      const i = firstIndexAtOrAfter(data, latest - 1000);
      const n = data.length - i;
      if (n >= 2) {
        const span = latest - data[i].time;
        if (span > 0) rateText = `${((n - 1) * 1000 / span).toFixed(0)} Hz`;
      }
    }
    ui.sampleRate.textContent = rateText;

    if (!state.startTime || data.length === 0) {
      ui.duration.textContent = "0.0 s";
      return;
    }
    const seconds = (data[data.length - 1].time - state.startTime) / 1000;
    ui.duration.textContent = `${seconds.toFixed(1)} s`;
  }

  // ---------- Motion capture -----------------------------------------------

  function handleMotion(event) {
    if (!state.recording) return; // safety; listener is removed on stop anyway

    const acc = event.acceleration;
    const rot = event.rotationRate; // iOS reports °/s (per spec)
    const aig = event.accelerationIncludingGravity;

    state.data.push({
      time: performance.now(), // monotonic — immune to clock changes
      ax: acc?.x ?? 0,
      ay: acc?.y ?? 0,
      az: acc?.z ?? 0,
      rotationAlpha: rot?.alpha ?? 0,
      rotationBeta: rot?.beta ?? 0,
      rotationGamma: rot?.gamma ?? 0,
      // Gravity estimate in device coordinates: (accel incl. gravity) −
      // (linear accel) is the OS's own sensor-fusion gravity vector. It
      // tells the feature extractor which way is DOWN, which makes the
      // turn-direction features orientation-invariant. The sign convention
      // differs between platforms, but it's consistent per device — which
      // is all the classifier needs.
      gx: (aig?.x ?? 0) - (acc?.x ?? 0),
      gy: (aig?.y ?? 0) - (acc?.y ?? 0),
      gz: (aig?.z ?? 0) - (acc?.z ?? 0),
    });

    // First sample arrived — cancel the "no data" watchdog.
    if (state.watchdogId !== null) {
      clearTimeout(state.watchdogId);
      state.watchdogId = null;
    }

    // Amortised trim: the original spliced one element per event once full,
    // shifting the entire 20k array ~60 times a second.
    if (state.data.length > MAX_BUFFER_POINTS + TRIM_CHUNK) {
      state.data.splice(0, state.data.length - MAX_BUFFER_POINTS);
      state.dataTrimmed = true;
    }

    scheduleRender(); // idempotent — keeps the loop alive if it ever stopped
  }

  // ---------- Permission flow ----------------------------------------------

  ui.sensorBtn.addEventListener("click", async () => {
    try {
      if (typeof DeviceMotionEvent === "undefined") {
        setPill("denied", "Motion sensors: not supported in this browser");
        setSessionState("Open this page on a phone");
        return;
      }

      // iOS 13+ requires an explicit permission request from a user gesture.
      if (typeof DeviceMotionEvent.requestPermission === "function") {
        const response = await DeviceMotionEvent.requestPermission();
        if (response !== "granted") {
          setPill("denied", "Motion permission: denied");
          // iOS remembers the denial — explain how to get re-prompted.
          setSessionState("Denied \u2014 quit Safari and reopen to be asked again");
          return;
        }
      }
      // Browsers without requestPermission (Android, desktop) need no prompt.

      setPill("granted", "Motion permission: granted");
      setSessionState("Sensors enabled");
      ui.recordBtn.disabled = false;
      ui.sensorBtn.disabled = true;
      ui.sensorBtn.textContent = "Sensors Enabled";
      hapticTick(10);
    } catch (err) {
      console.error("[MotionLab] Motion permission request failed:", err);
      setPill("denied", "Motion permission: error");
      setSessionState(
        err?.name === "NotAllowedError"
          ? "The tap must come directly from you \u2014 try again"
          : "Error requesting permission \u2014 see console"
      );
    }
  });

  // ---------- Recording ----------------------------------------------------

  function startRecording() {
    // Guard against silently wiping a finished-but-unsaved take.
    if (state.unsaved && state.data.length &&
        !window.confirm("Start a new recording? The unsaved one will be discarded.")) {
      return;
    }

    state.data = [];
    state.dataTrimmed = false;
    state.unsaved = false;
    state.startTime = performance.now();
    state.recording = true;
    state.lastPredictionAt = 0;
    state.lastLivePrediction = null;

    // Attach only while recording: a permanently-attached devicemotion
    // listener keeps iOS sensors powered and drains the battery even idle.
    window.addEventListener("devicemotion", handleMotion, { passive: true });

    // If nothing arrives, tell the user instead of recording silence.
    state.watchdogId = setTimeout(() => {
      state.watchdogId = null;
      if (state.recording && state.data.length === 0) {
        setSessionState("No motion data \u2014 use a phone over HTTPS");
      }
    }, SENSOR_WATCHDOG_MS);

    ui.recordBtn.textContent = "Stop Recording";
    ui.recordBtn.classList.remove("btn-secondary");
    ui.recordBtn.classList.add("btn-danger");
    ui.sessionState.classList.add("recording"); // pulsing red dot (CSS)
    setSessionState("Recording\u2026");
    ui.clearBtn.disabled = true;

    // Show live forecast card (Phase 2), reset to its waiting state
    ui.liveForcastCard.classList.remove("hidden");
    ui.forecastPlatform.textContent = "\u2014";
    ui.forecastPlatform.classList.remove("platform-left", "platform-right");
    ui.forecastConfidence.textContent = "Confidence: \u2014";
    ui.forecastNote.textContent = "Waiting for data\u2026";

    hapticTick(15);
    updateSessionInfo();
    scheduleRender();
  }

  async function stopRecording() {
    state.recording = false;
    window.removeEventListener("devicemotion", handleMotion);

    if (state.watchdogId !== null) {
      clearTimeout(state.watchdogId);
      state.watchdogId = null;
    }

    ui.recordBtn.textContent = "Start Recording";
    ui.recordBtn.classList.remove("btn-danger");
    ui.recordBtn.classList.add("btn-secondary");
    ui.sessionState.classList.remove("recording");

    // Hide live forecast card (Phase 2)
    ui.liveForcastCard.classList.add("hidden");

    const hasData = state.data.length > 0;
    setSessionState(hasData ? "Recorded" : "Idle");
    ui.clearBtn.disabled = !hasData;

    // Update predict/label button states
    await updateTrainingStatus();

    hapticTick(8);
    updateSessionInfo();
    scheduleRender();

    // Offer to label the recording for training. The modal has its own
    // Skip button (and tap-outside-to-skip), so no extra confirm needed.
    if (hasData && state.trainSet) {
      const label = await showLabelingDialog();
      if (label && label !== "skip") {
        await labelAndSaveToTrainingSet(label);
      }
    }
  }

  ui.recordBtn.addEventListener("click", () => {
    if (state.recording) stopRecording();
    else startRecording();
  });

  ui.generateTestDataBtn.addEventListener("click", () => {
    generateFakeTrainingData();
  });

  // ---------- Training data backup / restore --------------------------------

  ui.backupBtn.addEventListener("click", async () => {
    try {
      await ensureTrainSet();
      state.trainSet.triggerDownload();
      alert("✓ Backup downloaded. You can save it to OneDrive or email to yourself.");
    } catch (err) {
      alert(`Backup failed: ${err.message}`);
    }
  });

  ui.restoreBtn.addEventListener("click", async () => {
    const fileInput = document.createElement("input");
    fileInput.type = "file";
    fileInput.accept = ".json";
    fileInput.onchange = async (event) => {
      const file = event.target.files?.[0];
      if (!file) return;

      try {
        const text = await file.text();
        const json = JSON.parse(text);

        await ensureTrainSet();
        const imported = new (await loadMLModules()).trainingSet.TrainingSet();
        imported.fromJSON(json);

        const merged = state.trainSet.mergeFrom(imported);
        await state.trainSet.save();

        await updateTrainingStatus();

        const stats = state.trainSet.getStats();
        alert(
          `✓ Restored!\n\nMerged ${merged} new examples.\n\nTotal now: ${stats.totalExamples} (${stats.leftCount}L / ${stats.rightCount}R)`
        );
      } catch (err) {
        alert(`Restore failed: ${err.message}`);
      }
    };
    fileInput.click();
  });

  // Toggle history card collapse
  ui.toggleHistoryBtn.addEventListener("click", () => {
    const isCollapsed = ui.recordingsList.classList.contains("hidden");
    if (isCollapsed) {
      ui.recordingsList.classList.remove("hidden");
      ui.toggleHistoryBtn.textContent = "−";
    } else {
      ui.recordingsList.classList.add("hidden");
      ui.toggleHistoryBtn.textContent = "+";
    }
  });


  // ---------- Clear --------------------------------------------------------

  ui.clearBtn.addEventListener("click", () => {
    if (state.unsaved && state.data.length &&
        !window.confirm("Discard this unsaved recording?")) {
      return;
    }

    state.data = [];
    state.startTime = null;
    state.dataTrimmed = false;
    state.unsaved = false;
    setSessionState("Idle");
    ui.clearBtn.disabled = true;
    updateSessionInfo();
    scheduleRender();
  });

  // ---------- Lifecycle ----------------------------------------------------

  // Stop rendering while hidden (battery); refresh on return. Note: iOS pauses
  // JS in background tabs, so a recording spanning a backgrounding will simply
  // contain a time gap — the data itself stays valid.
  document.addEventListener("visibilitychange", () => {
    if (document.hidden) {
      if (state.rafId !== null) {
        cancelAnimationFrame(state.rafId);
        state.rafId = null;
      }
    } else {
      resizeAllCanvases();
      scheduleRender();
    }
  });

  // Warn before losing an unsaved recording. (Best-effort: iOS Safari often
  // skips this prompt, but desktop and Android honour it.)
  window.addEventListener("beforeunload", (e) => {
    if (state.unsaved && state.data.length) {
      e.preventDefault();
      e.returnValue = "";
    }
  });

  // Optional offline support — registers sw.js if it exists; fails quietly
  // (with an info log) if the file hasn't been added to the repo.
  function registerServiceWorker() {
    if ("serviceWorker" in navigator && window.isSecureContext) {
      window.addEventListener("load", () => {
        navigator.serviceWorker.register("./sw.js").catch((err) => {
          console.info("[MotionLab] Offline mode unavailable (no service worker):", err.message);
        });
      });
    }
  }

  // ---------- Init ---------------------------------------------------------

  async function init() {
    if (typeof DeviceMotionEvent === "undefined") {
      setPill("denied", "Motion sensors: not supported here");
      ui.sensorBtn.disabled = true;
      setSessionState("Open this page on a phone");
    } else if (!window.isSecureContext) {
      // The permission API silently fails over plain HTTP — say so up front.
      setPill("denied", "HTTPS required for motion sensors");
      ui.sensorBtn.disabled = true;
      setSessionState("Serve this page over HTTPS");
    }

    registerServiceWorker();
    setupResizeHandling();
    updateSessionInfo();
    resizeAllCanvases(); // also performs the first render

    // Load training set asynchronously (Phase 1 & 2)
    try {
      await ensureTrainSet();
      console.log(
        `[ML] Loaded training set: ${state.trainSet.count()} examples`
      );
      await updateTrainingStatus();
    } catch (err) {
      console.warn("[ML] Failed to load training set:", err);
    }
  }

  init();
})();
