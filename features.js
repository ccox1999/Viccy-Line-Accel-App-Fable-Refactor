/* ============================================================
   Victoria Line Motion Lab — feature extraction

   Extracts ML-ready features from raw motion data.

   Key insight: a turn at Brixton junction produces:
   - Lateral acceleration spike (Y-axis bias)
   - Yaw rotation (alpha) peak
   - Possible roll component (gamma)

   We extract time-domain, frequency-domain, and statistical
   features that capture magnitude, asymmetry, and timing of
   acceleration changes.
   ============================================================ */

"use strict";

/**
 * Feature-extraction version. Bumped whenever the MEANING of the feature vector
 * changes:
 *   v2 — whole-recording aggregates → a fork-localised window chosen by the
 *        largest |mean world-yaw|.
 *   v3 — that window is now ROUTE-ANCHORED to the Brixton terminus stop (see
 *        extractForkFeatures), falling back to the v2 yaw-amplitude selector
 *        only when no clean terminal stop is present.
 *   v4 — adds APPROACH-SHAPE features (approach_*) over the last 30 s before
 *        the terminal stop. Analysis of the first 11 real trips showed the
 *        v3 aggregates cannot separate left from right (LOOCV at or below the
 *        6/11 majority baseline): a crossover onto a parallel road is an
 *        S-bend with ~zero NET heading change, so world_yaw_mean averages the
 *        fork away by construction. What CAN distinguish the roads is the
 *        TIME STRUCTURE — which way the S swings first, when, and how much
 *        switch-clatter roughness each segment of the approach carries —
 *        which is exactly what the approach_* features capture.
 * Stored examples keep their raw motion data (IndexedDB via raw-store.js), so
 * TrainingSet.reprocessFeatures can transparently re-extract them to the
 * current definition instead of silently mixing two incompatible feature
 * scales in one training set.
 */
export const FEATURE_VERSION = 4;

/** Seconds of data the fork-localised feature window spans (see extractForkFeatures). */
export const FORK_WINDOW_SEC = 10;

/** Seconds of pre-stop data the approach-shape features describe. */
export const APPROACH_WINDOW_SEC = 30;

/** Seconds of per-second history kept in a recording's approach profile. */
export const PROFILE_SEC = 60;

/**
 * Feature keys that are orientation-invariant by construction: computed from
 * gravity-projected quantities (world yaw, vertical/horizontal split) or from
 * rotation/acceleration MAGNITUDES, none of which change when the phone is
 * held differently. The classifier trains ONLY on these — device-frame
 * features (per-axis means, skews, jerks…) encode how the phone was carried,
 * and on a small personal dataset a model will happily learn carry habits as
 * a proxy for the label. That failure mode is excluded structurally here,
 * not left to feature weighting.
 */
export const ORIENTATION_INVARIANT_KEYS = [
  // v3 aggregates (fork window)
  "world_yaw_mean", "world_yaw_peak", "vert_accel_rms", "horiz_accel_rms",
  "accel_mag_rms", "accel_mag_peak", "accel_mag_mean", "accel_rms_total",
  "energy_total",
  "fft_bin_0", "fft_bin_1", "fft_bin_2", "fft_bin_3",
  "fft_bin_4", "fft_bin_5", "fft_bin_6", "fft_bin_7",
  "spectral_entropy",
  // v4 approach-shape features (last APPROACH_WINDOW_SEC before the stop)
  "approach_yaw_early", "approach_yaw_mid", "approach_yaw_late",
  "approach_yaw_net", "approach_yaw_trend", "approach_yaw_swing",
  "approach_yaw_lead",
  "approach_vert_rms_early", "approach_vert_rms_mid", "approach_vert_rms_late",
];

/**
 * Physical ceiling (deg/s) for a TRAIN's yaw rate, used only to score candidate
 * fork windows. Even a tight rail crossover at terminus speed yaws at ~20 °/s;
 * the phone swinging in the hand hits 100–400 °/s (and a flick-and-return
 * gesture carries more total rotation than the gentle fork, so it fools an
 * unclipped windowed mean). Clipping each sample to this ceiling before scoring
 * keeps all real train signal while defanging hand jerks, so the window search
 * locks onto the sustained fork rotation rather than the edge of a jerk.
 */
const FORK_YAW_SELECT_CLIP = 30;

/**
 * Route-anchor thresholds for locating the fork window (see extractForkFeatures).
 *
 * The journey is a fixed Stockwell→Brixton run that ALWAYS ends by stopping at
 * the Brixton terminus, and the fork is the last sustained turn on the approach.
 * So the fork window is the one ending where the train makes its final stop —
 * found by POSITION in the journey, not by yaw amplitude. That is robust to the
 * two things that defeat amplitude-based selection: hand jerks (100× the gentle
 * fork yaw) and the large route curve that BOTH platforms share.
 *
 * STATIONARY_ACCEL_RMS — horizontal-accel RMS (m/s², gravity-projected) below
 *   which the train is treated as stopped. On real data the moving train sits at
 *   ~0.4–0.9 and the terminus dwell at ~0.09, so 0.2 separates them cleanly.
 *   Device/calibration dependent — retune if the accelerometer scale differs.
 * MIN_TERMINAL_STOP_SEC — a qualifying terminus stop must be at least this long,
 *   so a brief between-stations crawl cannot masquerade as the final arrival.
 */
const STATIONARY_ACCEL_RMS = 0.2;
const MIN_TERMINAL_STOP_SEC = 8;

/**
 * Extract features from a motion data window.
 *
 * @param {Array} motionWindow - [{time, ax, ay, az, rotationAlpha, rotationBeta, rotationGamma}, ...]
 * @param {number} sampleRateHz - sampling rate (typically 60 Hz on iPhone)
 * @returns {Object} feature vector with ~25 features, all numeric and finite
 *
 * Features are normalized (mostly 0–1 or log-scaled) so they're comparable
 * across different recordings and suitable for k-NN classification.
 */
export function extractFeatures(motionWindow, sampleRateHz = 60) {
  if (!motionWindow || motionWindow.length < 2) {
    return getNullFeatures();
  }

  // Safety: clamp and validate all inputs.
  // gx/gy/gz: gravity vector in device coords (may be absent in old data).
  const clean = motionWindow.map(s => ({
    ax: clamp(s.ax, -50, 50),
    ay: clamp(s.ay, -50, 50),
    az: clamp(s.az, -50, 50),
    rotationAlpha: clamp(s.rotationAlpha, -360, 360),
    rotationBeta: clamp(s.rotationBeta, -360, 360),
    rotationGamma: clamp(s.rotationGamma, -360, 360),
    gx: clamp(s.gx ?? 0, -30, 30),
    gy: clamp(s.gy ?? 0, -30, 30),
    gz: clamp(s.gz ?? 0, -30, 30),
  }));

  const n = clean.length;
  const dt = 1 / sampleRateHz;

  // NOTE: An earlier version reconstructed a full 3D trajectory here by
  // integrating the gyro into a rotation matrix and double-integrating
  // acceleration into position. That was removed because (1) the matrix
  // update was incorrect, corrupting the rotation after the first step, and
  // (2) even done correctly, doubly-integrated accelerometer data is a
  // random walk dominated by drift, so "trajectory range/curvature" features
  // carry noise, not signal. The orientation-invariant left/right signal is
  // captured directly and correctly by the WORLD-FRAME features below
  // (per-sample gravity projection — no integration, no drift).

  // ============ TIME-DOMAIN ACCELERATION FEATURES ============

  // Per-axis RMS (root mean square — overall magnitude)
  const accelRmsX = rms(clean.map(s => s.ax));
  const accelRmsY = rms(clean.map(s => s.ay));
  const accelRmsZ = rms(clean.map(s => s.az));
  const accelRmsTotal = rms([
    ...clean.map(s => s.ax),
    ...clean.map(s => s.ay),
    ...clean.map(s => s.az),
  ]);

  // Per-axis peak magnitude
  const accelPeakX = maxVal(clean.map(s => Math.abs(s.ax)));
  const accelPeakY = maxVal(clean.map(s => Math.abs(s.ay)));
  const accelPeakZ = maxVal(clean.map(s => Math.abs(s.az)));

  // Vector magnitude of acceleration (distance from origin in 3D space)
  const accelMagnitudes = clean.map(s =>
    Math.sqrt(s.ax ** 2 + s.ay ** 2 + s.az ** 2)
  );
  const accelMagRms = rms(accelMagnitudes);
  const accelMagPeak = maxVal(accelMagnitudes);
  const accelMagMean = mean(accelMagnitudes);

  // ============ SIGNED MEANS (directional — the left/right signal) ============

  // Sustained centripetal acceleration through a curve and sustained yaw
  // rate through a turn are SIGNED, DIRECTIONAL signals. Every other
  // aggregate here (RMS, peak, energy, FFT magnitude) is sign-blind, so
  // these means carry most of the left-vs-right information.
  const accelMeanX = mean(clean.map(s => s.ax));
  const accelMeanY = mean(clean.map(s => s.ay));
  const accelMeanZ = mean(clean.map(s => s.az));
  const rotMeanAlpha = mean(clean.map(s => s.rotationAlpha));
  const rotMeanBeta = mean(clean.map(s => s.rotationBeta));
  const rotMeanGamma = mean(clean.map(s => s.rotationGamma));

  // ============ WORLD-FRAME (ORIENTATION-INVARIANT) FEATURES ============

  // The per-axis features in this file live in DEVICE coordinates, so they
  // change whenever the phone is held differently. The gravity vector
  // (gx/gy/gz) tells us which way is down in device coordinates at every
  // sample, which lets us reconstruct the physics regardless of orientation:
  //
  //  - WORLD YAW: project the rotation-rate vector onto the vertical axis.
  //    A train taking the left fork rotates everything inside it about the
  //    world's vertical — the phone included, however it's held. This is
  //    THE physical left-vs-right signal. (Note rotationRate axis mapping:
  //    alpha is rotation about device z, beta about x, gamma about y, so
  //    the rotation vector in (x,y,z) order is (beta, gamma, alpha).)
  //
  //  - VERTICAL / HORIZONTAL ACCEL SPLIT: track roughness is vertical;
  //    braking and cornering are horizontal — same split in any orientation.
  //
  // Samples without a plausible gravity magnitude (~9.81) are skipped, so
  // old recordings that lack gx/gy/gz degrade gracefully to zeros.
  let yawSum = 0;
  let yawPeak = 0; // signed value with the largest magnitude
  let vertSqSum = 0;
  let horizSqSum = 0;
  let validG = 0;
  for (const s of clean) {
    const gMag = Math.sqrt(s.gx * s.gx + s.gy * s.gy + s.gz * s.gz);
    if (gMag < 4 || gMag > 20) continue; // no usable gravity estimate
    validG += 1;

    // Vertical unit axis in device coords (sign convention is per-device
    // consistent, which is all that matters for classification).
    const ux = s.gx / gMag, uy = s.gy / gMag, uz = s.gz / gMag;

    const yaw = s.rotationBeta * ux + s.rotationGamma * uy + s.rotationAlpha * uz;
    yawSum += yaw;
    if (Math.abs(yaw) > Math.abs(yawPeak)) yawPeak = yaw;

    const vert = s.ax * ux + s.ay * uy + s.az * uz;
    vertSqSum += vert * vert;
    const hx = s.ax - vert * ux;
    const hy = s.ay - vert * uy;
    const hz = s.az - vert * uz;
    horizSqSum += hx * hx + hy * hy + hz * hz;
  }
  const worldYawMean = validG > 0 ? yawSum / validG : 0;
  const worldYawPeak = yawPeak;
  const vertAccelRms = validG > 0 ? Math.sqrt(vertSqSum / validG) : 0;
  const horizAccelRms = validG > 0 ? Math.sqrt(horizSqSum / validG) : 0;

  // ============ TIME-DOMAIN ROTATION FEATURES ============

  // Per-axis RMS rotation
  const rotAlphaRms = rms(clean.map(s => s.rotationAlpha));
  const rotBetaRms = rms(clean.map(s => s.rotationBeta));
  const rotGammaRms = rms(clean.map(s => s.rotationGamma));

  // Per-axis peak rotation
  const rotAlphaPeak = maxVal(clean.map(s => Math.abs(s.rotationAlpha)));
  const rotBetaPeak = maxVal(clean.map(s => Math.abs(s.rotationBeta)));
  const rotGammaPeak = maxVal(clean.map(s => Math.abs(s.rotationGamma)));

  // ============ JERK (derivative of acceleration) ============

  // Jerk measures how quickly acceleration changes — sharp turns produce high jerk.
  const jerkX = computeJerk(clean.map(s => s.ax), dt);
  const jerkY = computeJerk(clean.map(s => s.ay), dt);
  const jerkZ = computeJerk(clean.map(s => s.az), dt);

  // ============ ASYMMETRY / SKEWNESS ============

  // Skewness detects directionality: left turn skews Y-axis left, right turn skews right.
  const accelXSkew = skewness(clean.map(s => s.ax));
  const accelYSkew = skewness(clean.map(s => s.ay));
  const accelZSkew = skewness(clean.map(s => s.az));

  // ============ ENERGY (integral of squared acceleration) ============

  const energyX = sum(clean.map(s => s.ax ** 2)) * dt;
  const energyY = sum(clean.map(s => s.ay ** 2)) * dt;
  const energyZ = sum(clean.map(s => s.az ** 2)) * dt;
  const energyTotal = energyX + energyY + energyZ;

  // ============ FREQUENCY-DOMAIN FEATURES (FFT) ============

  // Remove the DC offset first — accel magnitude has a large constant
  // component that would otherwise swamp bin 0. The mean is already its
  // own feature (accel_mag_mean), so only the oscillation matters here.
  const centeredMagnitudes = accelMagnitudes.map((v) => v - accelMagMean);
  const fftMags = computeFFT(centeredMagnitudes, sampleRateHz, 64);
  const fftFeatures = fftMags.slice(0, 8); // 8 FFT bins

  // Spectral entropy: how "noisy" vs "tonal" the vibration is.
  // Low entropy = clean tones (e.g., a resonant turn). High entropy = noisy random vibration.
  const spectralEntropy = computeSpectralEntropy(fftMags);

  // ============ COMPILE FEATURE VECTOR ============

  const features = {
    // Signed means: directional bias (THE left-vs-right discriminators)
    accel_mean_x: accelMeanX,
    accel_mean_y: accelMeanY,
    accel_mean_z: accelMeanZ,
    rot_mean_alpha: rotMeanAlpha,
    rot_mean_beta: rotMeanBeta,
    rot_mean_gamma: rotMeanGamma,

    // World-frame (orientation-invariant): signed heading-change rate +
    // vertical/horizontal acceleration split. This is the physically sound
    // left/right discriminator — computed by projecting each sample onto the
    // gravity vector, with no integration and therefore no drift.
    world_yaw_mean: worldYawMean,
    world_yaw_peak: worldYawPeak,
    vert_accel_rms: vertAccelRms,
    horiz_accel_rms: horizAccelRms,

    // Acceleration: magnitude (overall intensity)
    accel_rms_x: accelRmsX,
    accel_rms_y: accelRmsY,
    accel_rms_z: accelRmsZ,
    accel_rms_total: accelRmsTotal,
    accel_peak_x: accelPeakX,
    accel_peak_y: accelPeakY,
    accel_peak_z: accelPeakZ,
    accel_mag_rms: accelMagRms,
    accel_mag_peak: accelMagPeak,
    accel_mag_mean: accelMagMean,

    // Rotation: angular velocity
    rot_alpha_rms: rotAlphaRms,
    rot_beta_rms: rotBetaRms,
    rot_gamma_rms: rotGammaRms,
    rot_alpha_peak: rotAlphaPeak,
    rot_beta_peak: rotBetaPeak,
    rot_gamma_peak: rotGammaPeak,

    // Jerk: rate of acceleration change
    jerk_x: jerkX,
    jerk_y: jerkY,
    jerk_z: jerkZ,

    // Asymmetry: directional bias (key for left vs. right detection)
    accel_x_skew: accelXSkew,
    accel_y_skew: accelYSkew,
    accel_z_skew: accelZSkew,

    // Energy: total work done by acceleration
    energy_x: energyX,
    energy_y: energyY,
    energy_z: energyZ,
    energy_total: energyTotal,

    // Frequency: FFT magnitude bins (captures resonant frequencies)
    fft_bin_0: fftFeatures[0] ?? 0,
    fft_bin_1: fftFeatures[1] ?? 0,
    fft_bin_2: fftFeatures[2] ?? 0,
    fft_bin_3: fftFeatures[3] ?? 0,
    fft_bin_4: fftFeatures[4] ?? 0,
    fft_bin_5: fftFeatures[5] ?? 0,
    fft_bin_6: fftFeatures[6] ?? 0,
    fft_bin_7: fftFeatures[7] ?? 0,

    // Spectral entropy: signal complexity
    spectral_entropy: spectralEntropy,
  };

  // Validate: all features must be finite numbers
  for (const [key, val] of Object.entries(features)) {
    if (!Number.isFinite(val)) {
      console.warn(`[Features] Non-finite feature ${key} = ${val}; clamping to 0`);
      features[key] = 0;
    }
  }

  return features;
}

/**
 * Extract features from the most fork-like WINDOW of a recording, not the whole
 * thing — the single most important thing this file does for accuracy.
 *
 * The left/right discriminator is a brief, gentle, SUSTAINED yaw about the world
 * vertical as the train takes the scissors crossover near Brixton: only a few
 * seconds out of a journey that can run minutes. Two facts make whole-recording
 * aggregates the wrong tool for it:
 *   - the fork yaw rate is small (order a few °/s at crossover speed), so a
 *     `world_yaw_mean` averaged over the entire trip dilutes it ~30× toward
 *     zero, and
 *   - the phone moving in the hand produces yaw that is far LARGER in amplitude
 *     but ~zero-mean (a gesture rotates out and back), so it dominates the peak
 *     while contributing nothing to a windowed mean.
 *
 * PRIMARY (v3) — ROUTE ANCHOR. The route is fixed: every recording is a
 * Stockwell→Brixton run that ends by stopping at the Brixton terminus, and the
 * fork is the last sustained turn on the approach. So the fork window is simply
 * the one ENDING where the train comes to its final stop. We detect that stop
 * from the orientation-invariant horizontal-acceleration level (gravity-
 * projected) and take the window just before it. Anchoring by POSITION in the
 * journey — not by yaw amplitude — is robust to hand jerks (which can be 100×
 * the fork yaw) and to the large route curve that both platforms share.
 *
 * FALLBACK (v2 behaviour) — when there is no clean terminal stop (recording cut
 * short, or only mid-route signal holds), slide a fixed-length window and score
 * each position by the MAGNITUDE of its mean world-frame yaw rate, then extract
 * from the best-scoring window. Selecting by |mean yaw| concentrates the signal
 * while preserving its SIGN (left vs right). The yaw is CLIPPED to the train-yaw
 * ceiling for scoring only (features still come from the RAW slice) so a brief
 * high-amplitude hand jerk cannot outscore the real, gentler fork rotation.
 *
 * Using the SAME function for the saved training example and for the live
 * forecast keeps their feature distributions matched. (Training on
 * whole-recording features while predicting on a trailing 10 s window — the old
 * behaviour — was a silent train/inference mismatch: length-dependent features
 * such as energy and the diluted means simply did not line up.)
 *
 * Degrades gracefully: recordings shorter than one window, or old recordings
 * with no usable gravity vector, fall back to whole-recording extraction.
 *
 * @param {Array} motionData - full recording
 * @param {number} sampleRateHz - sampling rate
 * @param {number} windowSec - window length in seconds
 * @returns {Object} feature vector (same shape as extractFeatures)
 */
export function extractForkFeatures(motionData, sampleRateHz = 60, windowSec = FORK_WINDOW_SEC) {
  if (!motionData || motionData.length < 2) return getNullFeatures();

  const n = motionData.length;
  const w = Math.round(windowSec * sampleRateHz);
  if (w < 2 || n <= w) {
    return { ...getNullApproachFeatures(), ...extractFeatures(motionData, sampleRateHz) };
  }

  const pre = motionPrefixes(motionData);

  // Not enough gravity samples anywhere (old data) → whole-recording fallback.
  if (pre.prefValid[n] < w * 0.5) {
    return { ...getNullApproachFeatures(), ...extractFeatures(motionData, sampleRateHz) };
  }

  // PRIMARY: route-anchored window ending at the Brixton terminus stop. When
  // there is no terminal stop yet (live forecast mid-trip, cut-short
  // recording), the approach features describe the trailing end of the data
  // so far — which converges on the anchored definition as the train
  // actually arrives.
  const anchor = anchoredForkWindow(pre.prefHorizSq, pre.prefValid, n, w, sampleRateHz);
  const cease = anchor ? anchor.cease : n;
  const approach = approachFeatures(pre, cease, sampleRateHz);

  if (anchor) {
    return {
      ...approach,
      ...extractFeatures(motionData.slice(anchor.start, anchor.start + w), sampleRateHz),
    };
  }

  // FALLBACK: largest |mean clipped world-yaw| window (the v2 selector).
  const stride = Math.max(1, Math.round(sampleRateHz / 4)); // ~0.25 s steps
  const minValid = w * 0.5; // window must be at least half gravity-valid
  let bestStart = -1;
  let bestScore = -1;
  for (let start = 0; start + w <= n; start += stride) {
    const sv = pre.prefValid[start + w] - pre.prefValid[start];
    if (sv < minValid) continue;
    const score = Math.abs((pre.prefYaw[start + w] - pre.prefYaw[start]) / sv);
    if (score > bestScore) {
      bestScore = score;
      bestStart = start;
    }
  }

  if (bestStart < 0) {
    return { ...approach, ...extractFeatures(motionData, sampleRateHz) };
  }
  return {
    ...approach,
    ...extractFeatures(motionData.slice(bestStart, bestStart + w), sampleRateHz),
  };
}

/**
 * Per-sample, orientation-invariant prefix sums (any window then costs O(1)):
 *  - prefYaw:     CLIPPED world-frame yaw rate. Projection mirrors
 *                 extractFeatures exactly: rotation-rate vector
 *                 (beta, gamma, alpha) dotted with the gravity unit vector;
 *                 clip caps hand jerks (see FORK_YAW_SELECT_CLIP).
 *  - prefHorizSq: squared horizontal acceleration (component ⟂ gravity), for
 *                 terminal-stop detection — braking/cornering live here.
 *  - prefVertSq:  squared vertical acceleration (component ∥ gravity) —
 *                 track roughness / switch clatter lives here.
 *  - prefValid:   gravity-usable sample count, so old data degrades to zeros.
 */
function motionPrefixes(motionData) {
  const n = motionData.length;
  const prefYaw = new Float64Array(n + 1);
  const prefHorizSq = new Float64Array(n + 1);
  const prefVertSq = new Float64Array(n + 1);
  const prefValid = new Int32Array(n + 1);
  for (let i = 0; i < n; i++) {
    const s = motionData[i];
    const gx = s.gx ?? 0, gy = s.gy ?? 0, gz = s.gz ?? 0;
    const gMag = Math.sqrt(gx * gx + gy * gy + gz * gz);
    let yaw = 0, horizSq = 0, vertSq = 0, valid = 0;
    if (gMag >= 4 && gMag <= 20) {
      const ux = gx / gMag, uy = gy / gMag, uz = gz / gMag;
      const rawYaw =
        (s.rotationBeta ?? 0) * ux +
        (s.rotationGamma ?? 0) * uy +
        (s.rotationAlpha ?? 0) * uz;
      yaw = clamp(rawYaw, -FORK_YAW_SELECT_CLIP, FORK_YAW_SELECT_CLIP);
      const ax = s.ax ?? 0, ay = s.ay ?? 0, az = s.az ?? 0;
      const vert = ax * ux + ay * uy + az * uz;
      const hx = ax - vert * ux, hy = ay - vert * uy, hz = az - vert * uz;
      horizSq = hx * hx + hy * hy + hz * hz;
      vertSq = vert * vert;
      valid = 1;
    }
    prefYaw[i + 1] = prefYaw[i] + yaw;
    prefHorizSq[i + 1] = prefHorizSq[i] + horizSq;
    prefVertSq[i + 1] = prefVertSq[i] + vertSq;
    prefValid[i + 1] = prefValid[i] + valid;
  }
  return { n, prefYaw, prefHorizSq, prefVertSq, prefValid };
}

/**
 * Mean clipped world-yaw over [a, b), or null when too few gravity-valid
 * samples to judge (returned as null so callers can SKIP the span rather
 * than treat it as zero yaw).
 */
function spanYaw(pre, a, b) {
  const v = pre.prefValid[b] - pre.prefValid[a];
  if (b <= a || v < (b - a) * 0.5) return null;
  return (pre.prefYaw[b] - pre.prefYaw[a]) / v;
}

/** Vertical-accel RMS over [a, b), or 0 when unjudgeable. */
function spanVertRms(pre, a, b) {
  const v = pre.prefValid[b] - pre.prefValid[a];
  if (b <= a || v < (b - a) * 0.5) return 0;
  return Math.sqrt((pre.prefVertSq[b] - pre.prefVertSq[a]) / v);
}

/**
 * APPROACH-SHAPE features: the left/right discriminators that survive the
 * physics of a parallel-road junction.
 *
 * A crossover to a parallel platform road is an S-BEND: the train yaws one
 * way onto the diverging road, then yaws back to run parallel. Net heading
 * change ≈ 0, which is why the v3 windowed yaw MEAN carries no signal (the
 * first 11 real trips confirmed this empirically). The information is in the
 * shape and timing of the yaw transient, and in WHERE the switch-clatter
 * roughness happens — so these features slice the last APPROACH_WINDOW_SEC
 * before the stop into thirds and describe the sequence:
 *
 *  - approach_yaw_early/mid/late: mean clipped world-yaw per 10 s third
 *    (°/s, signed) — a coarse sampled version of the yaw-vs-time curve.
 *  - approach_yaw_net: mean over the whole approach (net heading tendency).
 *  - approach_yaw_trend: OLS slope of the per-second yaw over the last 20 s
 *    (°/s per s, signed). An S ridden left-first has rising yaw; ridden
 *    right-first, falling. This is THE swing-order feature.
 *  - approach_yaw_swing: max − min of per-second yaw over the last 20 s —
 *    how much S-bend there is at all (a straight platform road ≈ 0, which
 *    itself separates the roads when only one involves a crossover).
 *  - approach_yaw_lead: the per-second yaw value at whichever extreme
 *    (max or min) happens EARLIER — "which way did it swerve first", signed.
 *  - approach_vert_rms_early/mid/late: gravity-projected vertical
 *    acceleration RMS per third — switch clatter is a roughness burst, and
 *    its position in the approach differs between roads.
 *
 * All quantities are gravity-projected or magnitudes → orientation-invariant.
 * Per-second means (not raw samples) keep hand jerks from dominating; the
 * ±FORK_YAW_SELECT_CLIP clip caps what any single flick can contribute.
 *
 * @param {Object} pre - motionPrefixes() result
 * @param {number} cease - sample index where the train comes to rest
 *                         (end of data when no terminal stop was found)
 * @param {number} sampleRateHz
 * @returns {Object} the approach_* feature block
 */
function approachFeatures(pre, cease, sampleRateHz) {
  const f = getNullApproachFeatures();
  const hz = Math.max(1, Math.round(sampleRateHz));
  const end = Math.max(0, Math.min(cease, pre.n));

  // Thirds of the approach window, clamped to the data that exists.
  const third = 10 * hz;
  const late = spanYaw(pre, Math.max(0, end - third), end);
  const mid = spanYaw(pre, Math.max(0, end - 2 * third), Math.max(0, end - third));
  const early = spanYaw(pre, Math.max(0, end - 3 * third), Math.max(0, end - 2 * third));
  const net = spanYaw(pre, Math.max(0, end - 3 * third), end);
  if (late !== null) f.approach_yaw_late = late;
  if (mid !== null) f.approach_yaw_mid = mid;
  if (early !== null) f.approach_yaw_early = early;
  if (net !== null) f.approach_yaw_net = net;

  f.approach_vert_rms_late = spanVertRms(pre, Math.max(0, end - third), end);
  f.approach_vert_rms_mid = spanVertRms(pre, Math.max(0, end - 2 * third), Math.max(0, end - third));
  f.approach_vert_rms_early = spanVertRms(pre, Math.max(0, end - 3 * third), Math.max(0, end - 2 * third));

  // Per-second yaw series over the last 20 s (the fork itself is well inside
  // this) for the shape features.
  const shapeSec = 20;
  const series = []; // [secondsBeforeRest (negative→0), yaw]
  for (let s = shapeSec; s >= 1; s--) {
    const a = end - s * hz;
    const b = end - (s - 1) * hz;
    if (a < 0) continue;
    const y = spanYaw(pre, a, b);
    if (y !== null) series.push([-(s - 0.5), y]);
  }
  if (series.length >= 5) {
    // OLS slope of yaw vs time.
    const mt = mean(series.map((p) => p[0]));
    const my = mean(series.map((p) => p[1]));
    let num = 0, den = 0;
    for (const [t, y] of series) {
      num += (t - mt) * (y - my);
      den += (t - mt) * (t - mt);
    }
    f.approach_yaw_trend = den > 0 ? num / den : 0;

    let maxI = 0, minI = 0;
    for (let i = 1; i < series.length; i++) {
      if (series[i][1] > series[maxI][1]) maxI = i;
      if (series[i][1] < series[minI][1]) minI = i;
    }
    f.approach_yaw_swing = series[maxI][1] - series[minI][1];
    f.approach_yaw_lead = series[Math.min(maxI, minI)][1];
  }

  for (const key in f) {
    if (!Number.isFinite(f[key])) f[key] = 0;
  }
  return f;
}

/** Zeroed approach-feature block (short/old/gravity-less recordings). */
function getNullApproachFeatures() {
  return {
    approach_yaw_early: 0,
    approach_yaw_mid: 0,
    approach_yaw_late: 0,
    approach_yaw_net: 0,
    approach_yaw_trend: 0,
    approach_yaw_swing: 0,
    approach_yaw_lead: 0,
    approach_vert_rms_early: 0,
    approach_vert_rms_mid: 0,
    approach_vert_rms_late: 0,
  };
}

/**
 * Compact per-second summary of the arrival: clipped world-yaw, vertical
 * (roughness) and horizontal (speed proxy) acceleration RMS for each of the
 * last PROFILE_SEC seconds before the train came to rest, plus where the
 * rest point was. ~a few hundred bytes per trip, stored alongside the
 * features in localStorage.
 *
 * WHY: the first 10 real trips lost their raw sensor data to a localStorage
 * quota fallback, leaving nothing to re-analyse when the v3 features turned
 * out to carry no left/right signal. This profile is the belt to IndexedDB's
 * braces: even if raw data is ever lost again, every future trip keeps a
 * time-resolved picture of its approach that new window/shape ideas can be
 * tested against offline.
 *
 * @param {Array} motionData - full recording
 * @param {number} sampleRateHz
 * @returns {Object|null} { version, anchored, restSec, yaw[], vertRms[], horizRms[] }
 */
export function computeApproachProfile(motionData, sampleRateHz = 60) {
  if (!motionData || motionData.length < 2) return null;
  const hz = Math.max(1, Math.round(sampleRateHz));
  const n = motionData.length;
  const w = Math.round(FORK_WINDOW_SEC * hz);
  const pre = motionPrefixes(motionData);
  if (pre.prefValid[n] < Math.min(n, w) * 0.5) return null; // gravity-less (old) data

  const anchor = n > w ? anchoredForkWindow(pre.prefHorizSq, pre.prefValid, n, w, hz) : null;
  const cease = anchor ? anchor.cease : n;

  const seconds = Math.min(PROFILE_SEC, Math.floor(cease / hz));
  const yaw = [], vertRms = [], horizRms = [];
  const round3 = (x) => Math.round(x * 1000) / 1000;
  for (let s = seconds; s >= 1; s--) {
    const a = cease - s * hz;
    const b = cease - (s - 1) * hz;
    const y = spanYaw(pre, a, b);
    const v = pre.prefValid[b] - pre.prefValid[a];
    yaw.push(y === null ? null : round3(y));
    vertRms.push(v > 0 ? round3(Math.sqrt((pre.prefVertSq[b] - pre.prefVertSq[a]) / v)) : null);
    horizRms.push(v > 0 ? round3(Math.sqrt((pre.prefHorizSq[b] - pre.prefHorizSq[a]) / v)) : null);
  }

  return {
    version: 1,
    anchored: !!anchor,          // false → restSec is just the end of the data
    restSec: round3(cease / hz), // rest point, seconds from recording start
    yaw,                         // °/s, clipped ±FORK_YAW_SELECT_CLIP, oldest first
    vertRms,                     // m/s², roughness
    horizRms,                    // m/s², braking/cornering (speed proxy)
  };
}

/**
 * Locate the fork window by anchoring to the train's final stop at Brixton.
 *
 * Returns `{start, cease}` — the start index of a `w`-sample window ending
 * where the train comes to rest, and that rest index itself — or null if no
 * clean terminal stop is found (the caller then falls back to the
 * yaw-amplitude selector).
 *
 * A "stop" is a stretch of at least MIN_TERMINAL_STOP_SEC whose mean horizontal
 * acceleration sits below STATIONARY_ACCEL_RMS. We scan from the END so we lock
 * onto the LAST stop (the terminus), not a mid-route signal hold, then walk back
 * to where motion actually ceased — the window's end. The window immediately
 * before the stop must itself contain motion; if it is quiet too, this stop
 * follows another stop (e.g. standing on the platform after alighting) and the
 * real approach is earlier, so we keep scanning.
 *
 * @param {Float64Array} prefHorizSq - prefix sums of squared horizontal accel
 * @param {Int32Array} prefValid - prefix sums of gravity-valid samples
 * @param {number} n - sample count
 * @param {number} w - window length in samples
 * @param {number} sampleRateHz - sampling rate
 * @returns {{start: number, cease: number}|null}
 */
function anchoredForkWindow(prefHorizSq, prefValid, n, w, sampleRateHz) {
  const minStop = Math.max(2, Math.round(MIN_TERMINAL_STOP_SEC * sampleRateHz));
  if (n < minStop) return null;

  // RMS of horizontal accel over [a, b); Infinity if too little gravity to judge.
  const blockRms = (a, b) => {
    const cnt = prefValid[b] - prefValid[a];
    if (cnt < (b - a) * 0.5) return Infinity;
    return Math.sqrt((prefHorizSq[b] - prefHorizSq[a]) / cnt);
  };
  // True if a stop of length minStop begins at c.
  const isStop = (c) =>
    c + minStop <= n && blockRms(c, c + minStop) < STATIONARY_ACCEL_RMS;

  let c = n - minStop;
  while (c >= 0) {
    if (!isStop(c)) { c--; continue; }
    // Walk down to the onset of this stationary stretch = the moment of rest.
    let cease = c;
    while (cease - 1 >= 0 && isStop(cease - 1)) cease--;
    const winStart = cease - w;
    if (winStart >= 0 && blockRms(winStart, cease) >= STATIONARY_ACCEL_RMS) {
      return { start: winStart, cease };
    }
    c = cease - 1; // this stop is preceded by quiet (or runs off the start) — look earlier
  }
  return null;
}

/**
 * Extract features from a complete recording (post-hoc, e.g., after "Save").
 *
 * Splits the recording into overlapping windows, extracts features from each,
 * and returns an array of feature vectors. This allows the classifier to make
 * predictions on windowed data (e.g., "based on the last 5 seconds, I predict LEFT").
 *
 * @param {Array} motionData - full recording
 * @param {number} windowLengthMs - size of each analysis window (default 5000ms)
 * @param {number} strideMs - overlap stride (default 2500ms, 50% overlap)
 * @param {number} sampleRateHz - sampling rate
 * @returns {Array<Object>} list of feature vectors, each with a .windowEndIndex
 */
export function extractFeaturesWindowed(
  motionData,
  windowLengthMs = 5000,
  strideMs = 2500,
  sampleRateHz = 60
) {
  if (!motionData || motionData.length < 2) return [];

  const samplesPerWindow = Math.round((windowLengthMs / 1000) * sampleRateHz);
  const samplesPerStride = Math.round((strideMs / 1000) * sampleRateHz);

  const results = [];
  let startIdx = 0;

  while (startIdx + samplesPerWindow <= motionData.length) {
    const window = motionData.slice(startIdx, startIdx + samplesPerWindow);

    // Window position is returned ALONGSIDE the features, never mixed into
    // them — a raw epoch timestamp inside the feature vector would dominate
    // every distance calculation the classifier makes.
    results.push({
      features: extractFeatures(window, sampleRateHz),
      windowEndIndex: startIdx + samplesPerWindow,
      windowEndTime: window[window.length - 1]?.time ?? 0,
    });
    startIdx += samplesPerStride;
  }

  return results;
}

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/**
 * Largest element of an array, via a loop.
 *
 * Deliberately NOT `Math.max(...arr)`: that spreads every element as a
 * function argument, and a full recording window can hold tens of thousands
 * of samples — enough to exceed the engine's argument-count limit (a hard
 * RangeError on Safari/JSC) on a long trip. Returns 0 for an empty array.
 */
function maxVal(arr) {
  let m = -Infinity;
  for (let i = 0; i < arr.length; i++) {
    if (arr[i] > m) m = arr[i];
  }
  return m === -Infinity ? 0 : m;
}

/**
 * Clamped RMS (root mean square).
 * Avoids huge values from outliers while remaining proportional.
 */
function rms(arr) {
  if (arr.length === 0) return 0;
  const sumOfSquares = arr.reduce((sum, x) => sum + x * x, 0);
  return Math.sqrt(sumOfSquares / arr.length);
}

/**
 * Arithmetic mean.
 */
function mean(arr) {
  return arr.length === 0 ? 0 : sum(arr) / arr.length;
}

/**
 * Sum of array elements.
 */
function sum(arr) {
  return arr.reduce((a, b) => a + b, 0);
}

/**
 * Clamp a value to [min, max].
 */
function clamp(x, min, max) {
  return Math.max(min, Math.min(max, x));
}

/**
 * Skewness: measure of asymmetry.
 * Positive skew: tail on the right (more positive extremes).
 * Negative skew: tail on the left (more negative extremes).
 *
 * Key for left vs. right detection: a right turn skews Y-axis positive,
 * a left turn skews it negative.
 */
function skewness(arr) {
  if (arr.length < 3) return 0;
  const m = mean(arr);
  const variance = mean(arr.map(x => (x - m) ** 2));
  const sigma = Math.sqrt(variance);
  if (sigma === 0) return 0;
  const thirdMoment = mean(arr.map(x => (x - m) ** 3));
  return thirdMoment / (sigma ** 3);
}

/**
 * Jerk: derivative of acceleration (d²x/dt²).
 * High jerk indicates rapid direction changes (turns).
 */
function computeJerk(acceleration, dt) {
  if (acceleration.length < 2) return 0;
  let sumSqJerk = 0;
  for (let i = 1; i < acceleration.length; i++) {
    const jerk = (acceleration[i] - acceleration[i - 1]) / dt;
    sumSqJerk += jerk * jerk;
  }
  return Math.sqrt(sumSqJerk / (acceleration.length - 1));
}

/**
 * Averaged magnitude spectrum (Welch-style periodogram).
 *
 * A single 64-sample FFT only describes the first ~1 second of a recording.
 * Instead, FFT several 64-sample chunks spread evenly across the WHOLE
 * signal and average their magnitude spectra. This captures the dominant
 * vibration frequencies of the entire window and is far less noisy than
 * any single chunk.
 *
 * @param {Array<number>} signal - input samples (any length)
 * @param {number} sampleRateHz - sampling rate (unused, kept for API clarity)
 * @param {number} fftSize - chunk size (power of 2)
 * @returns {Array<number>} averaged magnitude spectrum, fftSize/2 bins
 */
function computeFFT(signal, sampleRateHz, fftSize = 64) {
  const half = fftSize / 2;
  if (!Array.isArray(signal) || signal.length === 0) {
    return Array(half).fill(0);
  }

  // Signal shorter than one chunk: zero-pad a single FFT.
  if (signal.length < fftSize) {
    const padded = [...signal, ...Array(fftSize - signal.length).fill(0)];
    return magnitudeSpectrum(padded, fftSize);
  }

  // Spread up to 16 chunks evenly across the signal so the spectrum
  // represents the whole window, not just its start.
  const chunkCount = Math.min(16, Math.floor(signal.length / fftSize));
  const stride = Math.floor(signal.length / chunkCount);

  const sums = new Array(half).fill(0);
  for (let c = 0; c < chunkCount; c += 1) {
    const start = Math.min(c * stride, signal.length - fftSize);
    const mags = magnitudeSpectrum(signal.slice(start, start + fftSize), fftSize);
    for (let i = 0; i < half; i += 1) sums[i] += mags[i];
  }
  return sums.map((s) => s / chunkCount);
}

/**
 * Magnitude spectrum of one chunk (first half of the FFT output —
 * the second half mirrors it for real-valued input).
 */
function magnitudeSpectrum(chunk, fftSize) {
  const fft = simpleFFT(chunk);
  const half = fftSize / 2;
  const mags = new Array(half);
  for (let i = 0; i < half; i += 1) {
    mags[i] = Math.sqrt(fft[i].real ** 2 + fft[i].imag ** 2) / fftSize;
  }
  return mags;
}

/**
 * Cooley-Tukey FFT (radix-2, recursive).
 * Input length must be a power of 2 (callers guarantee this).
 */
function simpleFFT(x) {
  const n = x.length;
  if (n <= 1) return x.map(v => ({ real: Number.isFinite(v) ? v : 0, imag: 0 }));

  // Divide into even and odd indices
  const even = simpleFFT(x.filter((_, i) => i % 2 === 0));
  const odd = simpleFFT(x.filter((_, i) => i % 2 === 1));

  const T = [];
  for (let k = 0; k < n / 2; k++) {
    const t = {
      real: Math.cos(-2 * Math.PI * k / n),
      imag: Math.sin(-2 * Math.PI * k / n),
    };
    T[k] = complexMul(t, odd[k]);
  }

  return [
    ...even.map((e, k) => complexAdd(e, T[k])),
    ...even.map((e, k) => complexSub(e, T[k])),
  ];
}

function complexAdd(a, b) {
  return { real: a.real + b.real, imag: a.imag + b.imag };
}

function complexSub(a, b) {
  return { real: a.real - b.real, imag: a.imag - b.imag };
}

function complexMul(a, b) {
  return {
    real: a.real * b.real - a.imag * b.imag,
    imag: a.real * b.imag + a.imag * b.real,
  };
}

/**
 * Spectral entropy: measure of disorder in the frequency domain.
 * Normalizes the magnitude spectrum and treats it as a probability distribution.
 *
 * Low entropy: clean, tonal signal (like a resonance from a turn).
 * High entropy: noisy, broadband signal (like random vibration).
 */
function computeSpectralEntropy(magnitudes) {
  if (magnitudes.length === 0) return 0;
  const total = sum(magnitudes);
  if (total === 0) return 0;

  // Normalize to probability distribution
  const probs = magnitudes.map(m => m / total);

  // Shannon entropy: -sum(p log p)
  let entropy = 0;
  for (const p of probs) {
    if (p > 0) {
      entropy -= p * Math.log2(p);
    }
  }

  // Normalize by max entropy (uniform distribution)
  const maxEntropy = Math.log2(magnitudes.length);
  return maxEntropy > 0 ? entropy / maxEntropy : 0;
}

/**
 * Return a feature vector filled with zeros (for empty/invalid data).
 * Includes the approach_* block so every extractForkFeatures return path
 * has the same key set.
 */
function getNullFeatures() {
  const keys = [
    "accel_mean_x", "accel_mean_y", "accel_mean_z",
    "rot_mean_alpha", "rot_mean_beta", "rot_mean_gamma",
    "world_yaw_mean", "world_yaw_peak", "vert_accel_rms", "horiz_accel_rms",
    "accel_rms_x", "accel_rms_y", "accel_rms_z", "accel_rms_total",
    "accel_peak_x", "accel_peak_y", "accel_peak_z",
    "accel_mag_rms", "accel_mag_peak", "accel_mag_mean",
    "rot_alpha_rms", "rot_beta_rms", "rot_gamma_rms",
    "rot_alpha_peak", "rot_beta_peak", "rot_gamma_peak",
    "jerk_x", "jerk_y", "jerk_z",
    "accel_x_skew", "accel_y_skew", "accel_z_skew",
    "energy_x", "energy_y", "energy_z", "energy_total",
    "fft_bin_0", "fft_bin_1", "fft_bin_2", "fft_bin_3",
    "fft_bin_4", "fft_bin_5", "fft_bin_6", "fft_bin_7",
    "spectral_entropy",
  ];
  return { ...getNullApproachFeatures(), ...Object.fromEntries(keys.map(k => [k, 0])) };
}

/**
 * Euclidean distance between two feature vectors.
 * Used by k-NN classifier for finding nearest neighbors.
 */
export function euclideanDistance(feat1, feat2) {
  let sumSqDiff = 0;
  for (const key in feat1) {
    const v1 = feat1[key];
    if (!Number.isFinite(v1)) continue; // skip any non-numeric metadata
    const v2 = Number.isFinite(feat2[key]) ? feat2[key] : 0;
    sumSqDiff += (v1 - v2) ** 2;
  }
  return Math.sqrt(sumSqDiff);
}

/**
 * Normalize a feature vector to [0, 1] range.
 * Useful for ensuring all features have equal weight in distance calculations.
 *
 * Uses min-max normalization: (x - min) / (max - min)
 * Requires a reference set of training features to compute min/max per feature.
 */
export function normalizeFeatures(features, stats) {
  const normalized = { ...features };
  for (const key in features) {
    const stat = stats[key];
    if (stat && stat.min !== undefined && stat.max !== undefined) {
      const range = stat.max - stat.min;
      if (range > 0) {
        normalized[key] = (features[key] - stat.min) / range;
      }
    }
  }
  return normalized;
}

/**
 * Compute statistics (min, max, mean, std dev) for each feature
 * across a set of feature vectors.
 *
 * Useful for normalization and understanding the range of each feature.
 */
export function computeFeatureStats(featureVectors) {
  if (featureVectors.length === 0) return {};

  const stats = {};

  // Collect all unique keys
  const allKeys = new Set();
  for (const features of featureVectors) {
    for (const key in features) {
      allKeys.add(key);
    }
  }

  // Compute stats for each key
  for (const key of allKeys) {
    const values = featureVectors.map(f => f[key] ?? 0).filter(Number.isFinite);
    if (values.length === 0) continue;

    const min = Math.min(...values);
    const max = Math.max(...values);
    const meanVal = sum(values) / values.length;
    const variance = sum(values.map(v => (v - meanVal) ** 2)) / values.length;
    const stdDev = Math.sqrt(variance);

    stats[key] = { min, max, mean: meanVal, stdDev };
  }

  return stats;
}
