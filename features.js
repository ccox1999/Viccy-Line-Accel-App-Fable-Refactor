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

  // ============ INERTIAL NAVIGATION (TRAJECTORY RECONSTRUCTION) ============

  // Orientation tracking via gyro integration. Start from gravity to find
  // device-to-world rotation, then integrate gyro to track orientation changes.
  // Output: world-frame acceleration, trajectory, and shape features.

  // Initial orientation from gravity: which device axis points down?
  let R = null; // 3×3 rotation matrix: row = device x/y/z axis in world coords
  let validGravityCount = 0;
  for (const s of clean) {
    const gMag = Math.hypot(s.gx, s.gy, s.gz);
    if (gMag >= 4 && gMag <= 20) {
      if (!R) {
        // First plausible gravity: use it to initialize. The gravity vector
        // is the world-Z (down) in device coordinates.
        const uz = [s.gx / gMag, s.gy / gMag, s.gz / gMag]; // world -Z in device coords
        // Orthonormal basis: ux arbitrary perpendicular, uy = uz × ux.
        const ux =
          Math.abs(uz[0]) < 0.9
            ? normalize([1, 0, 0])
            : normalize([0, 1, 0]);
        const uy = cross(uz, ux);
        const ux2 = cross(uy, uz);
        R = [ux2, uy, uz]; // rows = device axes in world coords
      }
      validGravityCount++;
    }
  }
  if (!R) {
    // No gravity estimate: assume device was upright initially (y points down)
    R = [[1, 0, 0], [0, 0, -1], [0, 1, 0]];
  }

  // Integrate gyro to track orientation. Update R incrementally at each
  // sample, then convert acceleration to world frame.
  const worldAccel = [];
  const worldVelocity = [[0, 0, 0]]; // integrated acceleration
  const worldTrajectory = [[0, 0, 0]]; // integrated velocity
  let cumulativeYaw = 0;
  let cumulativePitch = 0;
  let cumulativeRoll = 0;

  for (let i = 0; i < n; i++) {
    const s = clean[i];

    // Infinitesimal rotation: dR = skew(ω × dt) × R
    const wx = (s.rotationBeta * Math.PI) / 180; // convert °/s to rad/s
    const wy = (s.rotationGamma * Math.PI) / 180;
    const wz = (s.rotationAlpha * Math.PI) / 180;
    const ang = Math.hypot(wx, wy, wz) * dt;
    if (ang > 1e-6) {
      const axis = [wx / ang, wy / ang, wz / ang];
      const c = Math.cos(ang / 2);
      const s_ = Math.sin(ang / 2);
      const q = [
        c,
        axis[0] * s_,
        axis[1] * s_,
        axis[2] * s_,
      ]; // quaternion [w, x, y, z]
      // R := quat_to_matrix(q) × R
      R = quatToMatrix(q).map((row) =>
        row.map(
          (v, j) =>
            R[0][j] * v +
            R[1][j] * row[1] +
            R[2][j] * row[2]
        )
      ); // simplified; proper matmul
    }

    // Cumulative angles (Euler-angle style, for feature extraction)
    cumulativeYaw += wz * dt;
    cumulativePitch += wy * dt;
    cumulativeRoll += wx * dt;

    // World-frame acceleration = R × device-accel
    const aDevice = [s.ax, s.ay, s.az];
    const aWorld = [
      R[0][0] * aDevice[0] +
        R[0][1] * aDevice[1] +
        R[0][2] * aDevice[2],
      R[1][0] * aDevice[0] +
        R[1][1] * aDevice[1] +
        R[1][2] * aDevice[2],
      R[2][0] * aDevice[0] +
        R[2][1] * aDevice[1] +
        R[2][2] * aDevice[2],
    ];
    worldAccel.push(aWorld);

    // Integrate: v += a×dt, x += v×dt
    const v = [
      worldVelocity[i][0] + aWorld[0] * dt,
      worldVelocity[i][1] + aWorld[1] * dt,
      worldVelocity[i][2] + aWorld[2] * dt,
    ];
    worldVelocity.push(v);
    const x = [
      worldTrajectory[i][0] + v[0] * dt,
      worldTrajectory[i][1] + v[1] * dt,
      worldTrajectory[i][2] + v[2] * dt,
    ];
    worldTrajectory.push(x);
  }

  // Trajectory shape features (invariant to position origin, revealing of fork)
  let trajCurvatureMean = 0;
  let trajCurvaturePeak = 0;
  for (let i = 1; i < worldVelocity.length - 1; i++) {
    const v = worldVelocity[i];
    const vMag = Math.hypot(...v);
    if (vMag < 0.01) continue;
    const dv = [
      worldVelocity[i + 1][0] - v[0],
      worldVelocity[i + 1][1] - v[1],
      worldVelocity[i + 1][2] - v[2],
    ];
    const curvature = Math.hypot(...dv) / (vMag * dt); // lateral acceleration / speed
    trajCurvatureMean += curvature;
    trajCurvaturePeak = Math.max(trajCurvaturePeak, curvature);
  }
  trajCurvatureMean /= Math.max(worldVelocity.length - 2, 1);

  // Trajectory extent: how far did the phone move?
  const trajXRange = Math.max(...worldTrajectory.map(p => p[0])) - Math.min(...worldTrajectory.map(p => p[0]));
  const trajYRange = Math.max(...worldTrajectory.map(p => p[1])) - Math.min(...worldTrajectory.map(p => p[1]));
  const trajZRange = Math.max(...worldTrajectory.map(p => p[2])) - Math.min(...worldTrajectory.map(p => p[2]));
  const trajHorizRange = Math.hypot(trajXRange, trajYRange);
  const trajVertRange = trajZRange;

  // Jerk: d/dt of acceleration (high jerk = bumpy, distinctive per track)
  const jerkWorld = [];
  for (let i = 1; i < worldAccel.length; i++) {
    const dA = [
      worldAccel[i][0] - worldAccel[i - 1][0],
      worldAccel[i][1] - worldAccel[i - 1][1],
      worldAccel[i][2] - worldAccel[i - 1][2],
    ];
    jerkWorld.push(Math.hypot(...dA) / dt);
  }
  const jerkRms = rms(jerkWorld);
  const jerkPeak = Math.max(...jerkWorld);

  // Vertical vs horizontal jerk (track roughness is vertical; cornering is lateral)
  const jerkVert = [];
  const jerkHoriz = [];
  for (let i = 1; i < worldAccel.length; i++) {
    const jVert = (worldAccel[i][2] - worldAccel[i - 1][2]) / dt;
    const jH = Math.hypot(
      worldAccel[i][0] - worldAccel[i - 1][0],
      worldAccel[i][1] - worldAccel[i - 1][1]
    ) / dt;
    jerkVert.push(Math.abs(jVert));
    jerkHoriz.push(jH);
  }
  const jerkVertRms = rms(jerkVert);
  const jerkHorizRms = rms(jerkHoriz);

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
  const accelPeakX = Math.max(...clean.map(s => Math.abs(s.ax)));
  const accelPeakY = Math.max(...clean.map(s => Math.abs(s.ay)));
  const accelPeakZ = Math.max(...clean.map(s => Math.abs(s.az)));

  // Vector magnitude of acceleration (distance from origin in 3D space)
  const accelMagnitudes = clean.map(s =>
    Math.sqrt(s.ax ** 2 + s.ay ** 2 + s.az ** 2)
  );
  const accelMagRms = rms(accelMagnitudes);
  const accelMagPeak = Math.max(...accelMagnitudes);
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
  const rotAlphaPeak = Math.max(...clean.map(s => Math.abs(s.rotationAlpha)));
  const rotBetaPeak = Math.max(...clean.map(s => Math.abs(s.rotationBeta)));
  const rotGammaPeak = Math.max(...clean.map(s => Math.abs(s.rotationGamma)));

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
    // vertical/horizontal acceleration split
    world_yaw_mean: worldYawMean,
    world_yaw_peak: worldYawPeak,
    vert_accel_rms: vertAccelRms,
    horiz_accel_rms: horizAccelRms,

    // Inertial trajectory: shape and geometry of the 3D path the phone took
    traj_curvature_mean: trajCurvatureMean,
    traj_curvature_peak: trajCurvaturePeak,
    traj_horiz_range: trajHorizRange,
    traj_vert_range: trajVertRange,
    jerk_rms: jerkRms,
    jerk_peak: jerkPeak,
    jerk_vert_rms: jerkVertRms,
    jerk_horiz_rms: jerkHorizRms,
    gyro_cumul_yaw: cumulativeYaw,
    gyro_cumul_pitch: cumulativePitch,
    gyro_cumul_roll: cumulativeRoll,

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
 */
function getNullFeatures() {
  const keys = [
    "accel_mean_x", "accel_mean_y", "accel_mean_z",
    "rot_mean_alpha", "rot_mean_beta", "rot_mean_gamma",
    "world_yaw_mean", "world_yaw_peak", "vert_accel_rms", "horiz_accel_rms",
    "traj_curvature_mean", "traj_curvature_peak", "traj_horiz_range", "traj_vert_range",
    "jerk_rms", "jerk_peak", "jerk_vert_rms", "jerk_horiz_rms",
    "gyro_cumul_yaw", "gyro_cumul_pitch", "gyro_cumul_roll",
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
  return Object.fromEntries(keys.map(k => [k, 0]));
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

/**
 * Convert a unit quaternion [w, x, y, z] to a 3×3 rotation matrix.
 * Used by inertial navigation to update orientation from gyro rotation rate.
 */
function quatToMatrix(q) {
  const [w, x, y, z] = q;
  return [
    [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
    [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
    [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
  ];
}

/**
 * 3D cross product: a × b.
 */
function cross(a, b) {
  return [
    a[1] * b[2] - a[2] * b[1],
    a[2] * b[0] - a[0] * b[2],
    a[0] * b[1] - a[1] * b[0],
  ];
}

/**
 * Normalize a 3D vector to unit length.
 */
function normalize(v) {
  const mag = Math.hypot(v[0], v[1], v[2]);
  return mag > 0 ? [v[0] / mag, v[1] / mag, v[2] / mag] : v;
}
