/* ============================================================
   Fork engine — deterministic Stockwell -> Brixton platform predictor
   ------------------------------------------------------------
   A physics-derived alternative to the learned kNN/logreg path. It does
   not search a feature space: it measures one quantity with a known
   mechanism and thresholds it.

   MECHANISM
   The train must cross the points to reach its platform, and the points
   sit upstream of the platform mouth, so the deciding evidence is
   recorded before a passenger can see which side they are arriving on.
   Signed yaw rate (rotation-rate vector projected onto gravity, so it is
   orientation-invariant by construction — ORIENTATION_INVARIANT hard
   requirement satisfied) integrated over a window before arrival
   separates the classes.

   Measured on 15 labelled trips (9L/6R), LOOCV with the threshold refit
   in every fold:
       window [arr-26, arr-10]   AUC 0.98   <- final call
       window [arr-22, arr-16]   AUC 0.91   <- pre-visibility verdict
   The wide-window S-bend cancellation documented in CLAUDE.md is real
   and is why the window is bounded: at [arr-34, arr-4] AUC falls to
   0.67. Separation does NOT invert under +/-4 s of window shift (left
   stays positive, right negative), so this is not a single S-lobe
   artefact.

   ROUTE PRIOR (why triggering works at all)
   Sensors cannot distinguish the final approach from an intermediate
   stop or a Brixton pre-platform hold. Elapsed time can: over 15 trips
   the fastest journey was 93.1 s and every mid-journey stop began
   between 37 s and 94 s. So a stop starting before MIN_JOURNEY is never
   the arrival. This is drift-free, unlike dead reckoning, which was
   tried and failed badly (58-753 m estimated for a ~1.4 km route: a
   hand-held phone has no heading reference without a magnetometer).

   SIGN SELF-CALIBRATION
   gx/gy/gz sign conventions differ between platforms (see the comment in
   app.js handleMotion). Rather than hard-code a sign fitted to one
   device, calibrate() derives it from the user's own labelled examples,
   so a device change cannot silently invert every prediction.
   ============================================================ */

"use strict";

export const FORK_ENGINE_VERSION = 1;

// Windows are relative to the arrival instant, in seconds before it.
export const WIN_FINAL = [26.0, 10.0];
export const WIN_EARLY = [22.0, 16.0];

// Route prior, Stockwell -> Brixton. See header.
export const MIN_JOURNEY = 85.0;   // s; a stop before this is never arrival
export const AUTO_ARM_AT = 75.0;   // s; final approach becomes plausible

const GRAV_TAU = 5.0;              // s, gravity lowpass
const VIB_WIN = 2.0;               // s, vibration RMS window
const BRAKE_TAU = 2.0;             // s, horizontal-accel lowpass
const WARMUP = 15.0;               // s before adaptive thresholds are trusted

// Defaults from the 15-trip fit; calibrate() overrides scale/threshold/sign.
const DEFAULT_CAL = { sign: 1, thrEarly: 0.255, scaleEarly: 1.27,
                      thrFinal: 0.0, scaleFinal: 1.9 };

function median(xs) {
  if (!xs.length) return 0;
  const s = [...xs].sort((a, b) => a - b);
  const n = s.length;
  return n % 2 ? s[(n - 1) / 2] : 0.5 * (s[n / 2 - 1] + s[n / 2]);
}

/**
 * Signed yaw integrated over [t_hi, t_lo] seconds before `endT`.
 * Orientation-invariant: the rotation vector is projected onto the
 * device's own gravity estimate, so how the phone is held cancels.
 *
 * @param {Array} samples  motion samples: {time(ms), ax,ay,az, gx,gy,gz,
 *                         rotationAlpha,rotationBeta,rotationGamma}
 * @param {number} endT    reference instant, ms
 * @param {number} backHi  window start, seconds before endT (larger)
 * @param {number} backLo  window end,   seconds before endT (smaller)
 * @returns {number} degrees; positive = one side, negative = the other
 */
export function yawIntegral(samples, endT, backHi, backLo) {
  if (!samples || samples.length < 2) return 0;
  const hi = endT - backHi * 1000;
  const lo = endT - backLo * 1000;

  // Gravity must be tracked from before the window so the lowpass has
  // settled; start the filter GRAV_TAU*3 earlier than the window.
  const warm = hi - GRAV_TAU * 3000;
  let g = null, prevT = null, prevYaw = 0, acc = 0;

  // Binary-search the start rather than scanning from 0: this runs on the
  // sensor thread ~1x/second against a buffer that grows to ~9000 samples.
  let loIdx = 0, hiIdx = samples.length - 1, start = samples.length;
  while (loIdx <= hiIdx) {
    const mid = (loIdx + hiIdx) >> 1;
    if (samples[mid].time >= warm) { start = mid; hiIdx = mid - 1; }
    else loIdx = mid + 1;
  }

  for (let i = start; i < samples.length; i++) {
    const s = samples[i];
    if (s.time > lo) break;

    const gv = [s.gx || 0, s.gy || 0, s.gz || 0];
    if (!g) {
      g = gv.slice();
      prevT = s.time;
    }
    const dt = Math.max((s.time - prevT) / 1000, 0);
    const a = Math.min(dt / GRAV_TAU, 1);
    for (let k = 0; k < 3; k++) g[k] += a * (gv[k] - g[k]);

    const gn = Math.hypot(g[0], g[1], g[2]) || 1;
    // W3C devicemotion axes: beta about x, gamma about y, alpha about z.
    const yaw = ((s.rotationBeta || 0) * g[0] +
                 (s.rotationGamma || 0) * g[1] +
                 (s.rotationAlpha || 0) * g[2]) / gn;

    if (s.time >= hi && prevT !== null && dt > 0) {
      acc += 0.5 * (yaw + prevYaw) * dt;   // trapezoid
    }
    prevT = s.time;
    prevYaw = yaw;
  }
  // Negated so that positive = LEFT under the reference device's sign
  // convention; calibrate() can flip this per device.
  return -acc;
}

/**
 * Derive sign and scale from the user's own labelled examples, so a
 * device whose gravity axes are reported with the opposite sign cannot
 * silently invert every prediction.
 *
 * @param {Array} examples  training examples with rawMotionData + label
 * @returns {object} calibration, or DEFAULT_CAL if there is not enough data
 */
export function calibrate(examples) {
  const L = [], R = [];
  for (const ex of examples || []) {
    const raw = ex.rawMotionData;
    if (!raw || raw.length < 600) continue;
    const end = raw[raw.length - 1].time;         // user stops at arrival
    const v = yawIntegral(raw, end, WIN_FINAL[0], WIN_FINAL[1]);
    (ex.label === "left" ? L : R).push(v);
  }
  if (L.length < 3 || R.length < 3) return { ...DEFAULT_CAL, n: L.length + R.length };

  const mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
  const mL = mean(L), mR = mean(R);
  const sd = Math.sqrt(0.5 * (variance(L) + variance(R))) || 1;
  const sign = mL >= mR ? 1 : -1;                 // self-calibrating
  return {
    sign,
    thrFinal: (mL + mR) / 2,
    scaleFinal: Math.max(sd, 0.4),
    thrEarly: DEFAULT_CAL.thrEarly * sign,
    scaleEarly: DEFAULT_CAL.scaleEarly,
    n: L.length + R.length,
    separation: Math.abs(mL - mR) / sd,
  };
}

function variance(a) {
  if (a.length < 2) return 0;
  const m = a.reduce((x, y) => x + y, 0) / a.length;
  return a.reduce((s, x) => s + (x - m) * (x - m), 0) / (a.length - 1);
}

function logistic(x, thr, scale, sign) {
  return 1 / (1 + Math.exp(-(sign * (x - thr)) / scale));
}

/**
 * Authoritative verdict for a COMPLETED recording.
 *
 * Only valid once the user has tapped Stop, because it trusts the
 * recording's own endpoint as the arrival instant — the user's protocol is
 * to stop within ~1 s of the train halting at Brixton. A live, in-progress
 * recording genuinely cannot know it has arrived (see the arrival-detection
 * note in ForkEngine.update), which is the same live/complete split the
 * app's `recordingComplete` flag already encodes.
 *
 * @param {Array} samples  full recording
 * @param {object} cal     calibration from calibrate()
 * @returns {object|null}  {score, pLeft, prediction} or null if too short
 */
export function finalVerdict(samples, cal) {
  const c = cal || DEFAULT_CAL;
  if (!samples || samples.length < 2) return null;
  const end = samples[samples.length - 1].time;
  const durSec = (end - samples[0].time) / 1000;
  if (durSec < WIN_FINAL[0]) return null;          // not enough approach
  const score = yawIntegral(samples, end, WIN_FINAL[0], WIN_FINAL[1]);
  const p = logistic(score, c.thrFinal, c.scaleFinal, c.sign);
  return {
    score,
    pLeft: p,
    pRight: 1 - p,
    prediction: p > 0.5 ? "left" : "right",
    shortJourney: durSec < MIN_JOURNEY,            // caller may want to warn
  };
}

/**
 * Streaming predictor. Feed it samples as they arrive; ask it for the
 * current verdict. All state is causal — nothing looks ahead.
 */
export class ForkEngine {
  constructor(cal) {
    this.cal = cal || DEFAULT_CAL;
    this.t0 = null;
    this.tLast = null;
    this.g = null;
    this.aH = [0, 0, 0];
    this.vib = [];        // {t, mag}
    this.rmsSamples = []; // 1 Hz, for the adaptive quiet threshold
    this.rmsNext = 0;
    this.armed = false;
    this.quietStart = null;
    this.arrivalAt = null;   // ms, confirmed arrival instant
  }

  /**
   * @param {object} s   one motion sample
   * @param {Array} all  the full sample buffer (for window integrals)
   * @returns {object} {phase, pLeft, prediction, elapsed, armed}
   */
  update(s, all, computeVerdict = true) {
    const t = s.time;
    if (this.t0 === null) { this.t0 = t; this.tLast = t; }
    const dt = Math.max((t - this.tLast) / 1000, 0);
    this.tLast = t;
    const elapsed = (t - this.t0) / 1000;

    // gravity + horizontal acceleration
    const gv = [s.gx || 0, s.gy || 0, s.gz || 0];
    if (!this.g) this.g = gv.slice();
    const ag = Math.min(dt / GRAV_TAU, 1);
    for (let k = 0; k < 3; k++) this.g[k] += ag * (gv[k] - this.g[k]);
    const gn = Math.hypot(...this.g) || 1;
    const ghat = this.g.map((c) => c / gn);
    const acc = [s.ax || 0, s.ay || 0, s.az || 0];
    const adg = acc[0] * ghat[0] + acc[1] * ghat[1] + acc[2] * ghat[2];
    const ah = acc.map((a, i) => a - adg * ghat[i]);
    const ab = Math.min(dt / BRAKE_TAU, 1);
    for (let k = 0; k < 3; k++) this.aH[k] += ab * (ah[k] - this.aH[k]);

    // vibration RMS over a trailing window -> quiet detection
    const amag = Math.hypot(...acc);
    this.vib.push({ t, mag: amag });
    while (this.vib.length && this.vib[0].t < t - VIB_WIN * 1000) this.vib.shift();
    const vals = this.vib.map((v) => v.mag);
    const mean = vals.reduce((a, b) => a + b, 0) / vals.length;
    const rms = Math.sqrt(vals.reduce((a, b) => a + (b - mean) * (b - mean), 0) / vals.length);
    if (elapsed >= this.rmsNext && elapsed >= 5) {
      this.rmsSamples.push(rms);
      this.rmsNext = elapsed + 1;
    }
    const med = (this.rmsSamples.length && elapsed >= WARMUP) ? median(this.rmsSamples) : 0.25;
    const quietThr = Math.min(Math.max(0.45 * med, 0.04), 0.15);

    if (!this.armed && elapsed >= AUTO_ARM_AT) this.armed = true;

    // Arrival detection, ONLY once the route prior says the destination is
    // reachable: before MIN_JOURNEY a quiet stretch is a signal halt or a
    // Brixton pre-platform hold, never the arrival.
    //
    // NOTE: this rarely fires live, and that is expected. The user stops the
    // recording within ~1 s of the train halting, so a long stationary dwell
    // never exists — the same constraint that forced the app's own anchor
    // away from requiring stillness (CLAUDE.md, v5). The authoritative answer
    // therefore comes from finalVerdict() once recording is complete; this
    // path only helps when the user happens to leave it running.
    if (rms < quietThr) {
      if (this.quietStart === null) this.quietStart = t;
      const qDur = (t - this.quietStart) / 1000;
      const qElapsed = (this.quietStart - this.t0) / 1000;
      if (qDur >= 1.5 && qElapsed >= MIN_JOURNEY && this.arrivalAt === null) {
        this.arrivalAt = this.quietStart;
      }
    } else {
      this.quietStart = null;
      // motion resumed: whatever that quiet was, it was not the arrival
      this.arrivalAt = null;
    }

    // ---- verdict
    // The state machine above must see every sample, but the window integral
    // is O(window) and only worth running once per UI update — callers feeding
    // a backlog pass computeVerdict=false for all but the last sample.
    if (!computeVerdict) return null;

    let phase, score, p;
    if (this.arrivalAt !== null) {
      phase = "final";
      score = yawIntegral(all, this.arrivalAt, WIN_FINAL[0], WIN_FINAL[1]);
      p = logistic(score, this.cal.thrFinal, this.cal.scaleFinal, this.cal.sign);
    } else if (this.armed) {
      // Pre-visibility verdict: the trailing window reproduces
      // [arr-22, arr-16] at the moment the platform comes into view.
      phase = "early";
      score = yawIntegral(all, t, WIN_EARLY[0] - WIN_EARLY[1], 0);
      // NOTE: holding the last "informative" reading (evidence floor) was
      // tried to damp the display flipping. It roughly halves the flips but
      // costs real accuracy earlier in the approach — 18 s out fell 87% -> 73%
      // and 20 s out 80% -> 60%, because a pre-fork reading gets held into the
      // decision window. Earliness is the whole point of this path, so the
      // flicker is accepted and the note text marks the reading provisional.
      p = logistic(score, this.cal.thrEarly, this.cal.scaleEarly, this.cal.sign);
      p = Math.min(Math.max(p, 0.15), 0.85);   // provisional, cap confidence
    } else {
      phase = "waiting";
      score = 0;
      p = 0.5;
    }
    return {
      phase,
      score,
      pLeft: p,
      pRight: 1 - p,
      prediction: p > 0.5 ? "left" : "right",
      elapsed,
      armed: this.armed,
    };
  }
}
