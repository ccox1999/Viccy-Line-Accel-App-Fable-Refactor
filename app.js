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
   ============================================================ */

"use strict";

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
      saveBtn: requireElement("saveBtn"),
      clearBtn: requireElement("clearBtn"),
      sensorStatus: requireElement("sensorStatus"),
      sessionState: requireElement("sessionState"),
      sampleCount: requireElement("sampleCount"),
      sampleRate: requireElement("sampleRate"),
      duration: requireElement("duration"),
      accelCanvas: requireElement("accelChart"),
      gyroCanvas: requireElement("gyroChart"),
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
    // Snapshot so exports can convert monotonic times back to epoch ms
    // (keeps the JSON format identical to the original app's output).
    epochBase: Date.now() - performance.now(),
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

    state.data.push({
      time: performance.now(), // monotonic — immune to clock changes
      ax: acc?.x ?? 0,
      ay: acc?.y ?? 0,
      az: acc?.z ?? 0,
      rotationAlpha: rot?.alpha ?? 0,
      rotationBeta: rot?.beta ?? 0,
      rotationGamma: rot?.gamma ?? 0,
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
    ui.saveBtn.disabled = true;
    ui.clearBtn.disabled = true;

    hapticTick(15);
    updateSessionInfo();
    scheduleRender();
  }

  function stopRecording() {
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

    const hasData = state.data.length > 0;
    state.unsaved = hasData;
    setSessionState(hasData ? "Recorded \u2014 not saved yet" : "Idle");
    ui.saveBtn.disabled = !hasData;
    ui.clearBtn.disabled = !hasData;

    hapticTick(8);
    updateSessionInfo();
    scheduleRender();
  }

  ui.recordBtn.addEventListener("click", () => {
    if (state.recording) stopRecording();
    else startRecording();
  });

  // ---------- Save / export ------------------------------------------------

  ui.saveBtn.addEventListener("click", async () => {
    if (!state.data.length) return;
    ui.saveBtn.disabled = true; // block double-taps while the share sheet is up

    try {
      // Same schema as the original export: a flat array with epoch-ms times.
      // Compact JSON: ~60% smaller than pretty-printed for 20k samples.
      const samples = state.data.map((s) => ({
        time: Math.round(state.epochBase + s.time),
        ax: s.ax,
        ay: s.ay,
        az: s.az,
        rotationAlpha: s.rotationAlpha,
        rotationBeta: s.rotationBeta,
        rotationGamma: s.rotationGamma,
      }));
      const json = JSON.stringify(samples);
      const fileName = `motion-recording-${timestampForFilename()}.json`;

      // Preferred path on iPhone: the native share sheet (Files, AirDrop,
      // Mail…). <a download> is unreliable in iOS Safari and broken in
      // standalone home-screen apps.
      const file = new File([json], fileName, { type: "application/json" });
      if (navigator.canShare?.({ files: [file] })) {
        try {
          await navigator.share({ files: [file], title: "Motion recording" });
          state.unsaved = false;
          setSessionState("Saved");
          return;
        } catch (err) {
          if (err?.name === "AbortError") return; // user closed the sheet
          console.warn("[MotionLab] Share failed, falling back to download:", err);
        }
      }

      // Fallback: classic download.
      const url = URL.createObjectURL(new Blob([json], { type: "application/json" }));
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = fileName;
      anchor.rel = "noopener";
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      // Revoking immediately after click() can race the download in some
      // browsers — give it a moment.
      setTimeout(() => URL.revokeObjectURL(url), 10000);

      state.unsaved = false;
      setSessionState("Saved");
    } catch (err) {
      console.error("[MotionLab] Save failed:", err);
      setSessionState("Save failed \u2014 see console");
    } finally {
      ui.saveBtn.disabled = state.data.length === 0;
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
    ui.saveBtn.disabled = true;
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

  function init() {
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
  }

  init();
})();
