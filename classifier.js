/* ============================================================
   Victoria Line Motion Lab — k-NN classifier

   Predicts platform direction (left vs. right) based on
   vibration signatures using k-nearest neighbors.

   k-NN is ideal here:
   - No explicit training phase (just store examples)
   - Interpretable (neighbors are actual recordings)
   - Fast inference (a few hundred comparisons)
   - Works well with small datasets (50–100 examples)
   ============================================================ */

"use strict";

/**
 * k-Nearest Neighbors classifier for binary classification.
 *
 * Distances are computed over Z-SCORE NORMALIZED features: each feature is
 * rescaled to (value - mean) / stdDev using statistics from the training
 * set. Without this, large-magnitude features (energy ~hundreds) drown out
 * the small directional features (skewness ~±1) that actually distinguish
 * a left turn from a right turn.
 *
 * Usage:
 *   const clf = new KNNClassifier(5);
 *   clf.addTrainingExample(featureVector, "left");
 *   clf.addTrainingExample(featureVector, "right");
 *   const pred = clf.predict(newFeatureVector);
 *   // pred = { label: "left", confidence: 0.87, neighbors: [...] }
 */
export class KNNClassifier {
  constructor(k = 5, minConfidence = 0) {
    this.k = k;
    // Predictions at or below this fraction of the vote are reported as
    // label:null (undecided) rather than a near-coin-flip guess. 0 disables.
    this.minConfidence = minConfidence;
    this.examples = []; // [{features, label, recordingId, timestamp}, ...]
    this._normCache = null; // { stats, vectors } — rebuilt lazily after changes
  }

  /**
   * Add a training example.
   *
   * @param {Object} features - feature vector from extractFeatures()
   * @param {string} label - "left" or "right"
   * @param {Object} metadata - optional {recordingId, timestamp, ...}
   */
  addTrainingExample(features, label, metadata = {}) {
    if (!["left", "right"].includes(label)) {
      throw new Error(`Invalid label: ${label}. Must be "left" or "right".`);
    }

    if (!features || typeof features !== "object") {
      throw new Error("Features must be a non-null object.");
    }

    this.examples.push({
      features: { ...features },
      label,
      ...metadata,
    });
    this._normCache = null; // stats are stale now
  }

  /**
   * Remove all training examples (reset).
   */
  clear() {
    this.examples = [];
    this._normCache = null;
  }

  /**
   * Get the number of training examples.
   */
  count() {
    return this.examples.length;
  }

  /**
   * Get counts by label.
   */
  countByLabel() {
    const counts = { left: 0, right: 0 };
    for (const ex of this.examples) {
      if (ex.label in counts) counts[ex.label]++;
    }
    return counts;
  }

  /**
   * Predict the label for a new feature vector.
   *
   * Returns the majority label among the k nearest neighbors.
   *
   * @param {Object} features - feature vector to classify
   * @returns {Object} {
   *   label: "left" | "right",
   *   confidence: 0.0–1.0,
   *   neighbors: [{ features, label, distance, ... }, ...],
   *   rawVotes: { left: count, right: count }
   * }
   */
  predict(features) {
    if (this.examples.length === 0) {
      return {
        label: null,
        confidence: 0,
        neighbors: [],
        rawVotes: { left: 0, right: 0 },
        error: "No training examples",
      };
    }

    if (this.k > this.examples.length) {
      console.warn(
        `[KNN] k=${this.k} is larger than training set (${this.examples.length}); capping to training set size`
      );
    }

    // 1. Find k nearest neighbors
    const neighbors = this.findNearestNeighbors(features, this.k);

    // 2. Majority vote
    const votes = { left: 0, right: 0 };
    for (const neighbor of neighbors) {
      votes[neighbor.label]++;
    }

    // 3. Determine winner and confidence
    const leftVotes = votes.left;
    const rightVotes = votes.right;
    const totalVotes = leftVotes + rightVotes;

    let label, confidence;
    if (leftVotes > rightVotes) {
      label = "left";
      confidence = leftVotes / totalVotes;
    } else if (rightVotes > leftVotes) {
      label = "right";
      confidence = rightVotes / totalVotes;
    } else {
      // Tie: use distance-weighted vote (closer neighbors have higher weight)
      const leftDist = sum(
        neighbors.filter(n => n.label === "left").map(n => 1 / (1 + n.distance))
      );
      const rightDist = sum(
        neighbors.filter(n => n.label === "right").map(n => 1 / (1 + n.distance))
      );
      label = leftDist > rightDist ? "left" : "right";
      confidence = Math.max(leftDist, rightDist) / (leftDist + rightDist || 1);
    }

    // Suppress low-confidence guesses (e.g. a 3/2 split on k=5 → 0.6).
    if (this.minConfidence > 0 && confidence < this.minConfidence) {
      return {
        label: null,
        confidence,
        neighbors,
        rawVotes: votes,
        error: `Low confidence (${(confidence * 100).toFixed(0)}%)`,
      };
    }

    return {
      label,
      confidence,
      neighbors,
      rawVotes: votes,
    };
  }

  /**
   * Continuous left/right probability via distance-weighted (soft) k-NN.
   *
   * predict() returns a discrete majority vote quantised to 1/k steps — with
   * k=5 the only possible confidences are 60/80/100%, so a live readout built
   * on it jumps in 20-point steps and never moves smoothly. This returns a
   * smooth probability instead: each of the nearest neighbours votes with a
   * Gaussian weight on its distance, and the bandwidth is the median
   * neighbour distance so the estimate is scale-adaptive (independent of the
   * absolute feature-distance scale, which varies between datasets).
   *
   * @param {Object} features - feature vector to classify
   * @returns {Object} { pLeft, pRight, neighbors }
   */
  predictProbability(features) {
    if (this.examples.length === 0) {
      return { pLeft: 0.5, pRight: 0.5, neighbors: [], error: "No training examples" };
    }

    // Use more neighbours than the discrete k so the soft estimate is stable.
    const n = Math.min(this.examples.length, Math.max(this.k, 12));
    const neighbors = this.findNearestNeighbors(features, n);

    // Median-distance bandwidth → scale-adaptive, smooth kernel.
    const sorted = neighbors.map((nb) => nb.distance).sort((a, b) => a - b);
    const median = sorted[Math.floor(sorted.length / 2)] || 0;
    const bw = median > 1e-6 ? median : 1;

    let wLeft = 0;
    let wRight = 0;
    for (const nb of neighbors) {
      const w = Math.exp(-(nb.distance * nb.distance) / (2 * bw * bw));
      if (nb.label === "left") wLeft += w;
      else if (nb.label === "right") wRight += w;
    }

    const total = wLeft + wRight;
    if (total <= 0) return { pLeft: 0.5, pRight: 0.5, neighbors };
    return { pLeft: wLeft / total, pRight: wRight / total, neighbors };
  }

  /**
   * Find the k nearest neighbors to a feature vector.
   * Distances are computed in z-score-normalized space.
   *
   * @param {Object} features - query feature vector
   * @param {number} k - number of neighbors to return
   * @returns {Array} sorted by distance (nearest first), each with distance
   */
  findNearestNeighbors(features, k = this.k) {
    const { stats, weights, vectors } = this._ensureNormCache();
    const query = this._normalize(features, stats);
    for (const key in query) query[key] *= weights[key] ?? 1;

    const distances = this.examples.map((ex, idx) => {
      const v = vectors[idx];
      let sumSq = 0;
      for (const key in query) {
        const d = query[key] - (v[key] ?? 0);
        sumSq += d * d;
      }
      return { ...ex, distance: Math.sqrt(sumSq), index: idx };
    });

    // Sort by distance (nearest first)
    distances.sort((a, b) => a.distance - b.distance);

    // Return top k
    return distances.slice(0, Math.min(k, distances.length));
  }

  /**
   * Build (or reuse) per-feature mean/stdDev over the training set, plus
   * the normalized training vectors. Invalidated on add/clear.
   */
  _ensureNormCache() {
    if (this._normCache) return this._normCache;

    // Collect every numeric feature key seen in training data.
    const keys = new Set();
    for (const ex of this.examples) {
      for (const key in ex.features) {
        if (Number.isFinite(ex.features[key])) keys.add(key);
      }
    }

    const n = this.examples.length;
    const stats = {};
    for (const key of keys) {
      let total = 0;
      for (const ex of this.examples) total += ex.features[key] ?? 0;
      const mean = total / n;
      let sumSq = 0;
      for (const ex of this.examples) {
        const d = (ex.features[key] ?? 0) - mean;
        sumSq += d * d;
      }
      // Floor the stdDev so a near-constant feature (tiny variance from a
      // couple of noisy samples) can't be amplified into a dominant axis
      // when normalized. Zero variance still maps to 0 in _normalize().
      stats[key] = { mean, stdDev: Math.max(0.01, Math.sqrt(sumSq / n)) };
    }

    const vectors = this.examples.map((ex) => this._normalize(ex.features, stats));

    // Supervised feature weighting (Fisher-style): a feature whose left
    // and right class distributions are well separated gets more say in
    // the distance; a feature that is the same noise in both classes gets
    // less. This is what lets the classifier ignore orientation-dependent
    // device-frame features when the phone is held differently between
    // trips, and lean on the orientation-invariant ones instead — without
    // any of that knowledge being hardcoded.
    const weights = {};
    const leftIdx = [];
    const rightIdx = [];
    this.examples.forEach((ex, i) => {
      if (ex.label === "left") leftIdx.push(i);
      else if (ex.label === "right") rightIdx.push(i);
    });

    // Fisher weighting needs a few examples of EACH class to estimate
    // within-class spread; with 1–2 it just chases that one point's noise.
    // Below the threshold, fall back to uniform weights (plain z-scored k-NN).
    if (leftIdx.length >= 3 && rightIdx.length >= 3) {
      const classStats = (idxs, key) => {
        let total = 0;
        for (const i of idxs) total += vectors[i][key] ?? 0;
        const mean = total / idxs.length;
        let sumSq = 0;
        for (const i of idxs) {
          const d = (vectors[i][key] ?? 0) - mean;
          sumSq += d * d;
        }
        return { mean, stdDev: Math.sqrt(sumSq / idxs.length) };
      };

      for (const key in stats) {
        const L = classStats(leftIdx, key);
        const R = classStats(rightIdx, key);
        // +0.5 (in z-score units) regularises against tiny-sample flukes.
        weights[key] = Math.abs(L.mean - R.mean) / (L.stdDev + R.stdDev + 0.5);
      }

      // Rescale to mean 1 and cap, so no single feature can fully
      // dominate and the overall distance scale stays interpretable.
      const vals = Object.values(weights);
      const meanW = vals.reduce((a, b) => a + b, 0) / (vals.length || 1);
      if (meanW > 0) {
        for (const key in weights) {
          weights[key] = Math.min(weights[key] / meanW, 5);
        }
      } else {
        for (const key in weights) weights[key] = 1;
      }
    } else {
      // Too few examples per class to weight reliably — uniform weights.
      for (const key in stats) weights[key] = 1;
    }

    // Bake the weights into the cached training vectors so per-prediction
    // work stays a plain Euclidean loop.
    for (const v of vectors) {
      for (const key in v) v[key] *= weights[key] ?? 1;
    }

    this._normCache = { stats, weights, vectors };
    return this._normCache;
  }

  /**
   * Z-score normalize a feature vector against training-set stats.
   * Features with zero variance carry no information and map to 0.
   */
  _normalize(features, stats) {
    const out = {};
    for (const key in stats) {
      const { mean, stdDev } = stats[key];
      const v = Number.isFinite(features[key]) ? features[key] : mean;
      out[key] = stdDev > 0 ? (v - mean) / stdDev : 0;
    }
    return out;
  }

  /**
   * Serialize to JSON for storage/export.
   */
  toJSON() {
    return {
      k: this.k,
      examples: this.examples.map(ex => ({
        features: ex.features,
        label: ex.label,
        recordingId: ex.recordingId,
        timestamp: ex.timestamp,
      })),
    };
  }

  /**
   * Deserialize from JSON.
   */
  static fromJSON(json) {
    const clf = new KNNClassifier(json.k);
    for (const ex of json.examples || []) {
      clf.addTrainingExample(ex.features, ex.label, {
        recordingId: ex.recordingId,
        timestamp: ex.timestamp,
      });
    }
    return clf;
  }
}

/**
 * L2-regularised logistic regression (binary: left = 1, right = 0).
 *
 * Better suited than k-NN to THIS problem: the decision is near-linear and
 * sign-driven (world-frame yaw sign ≈ turn direction), the feature space is
 * high-dimensional (~46) with mostly irrelevant, sign-blind features, and the
 * dataset is tiny. LR learns a near-zero weight for the irrelevant features
 * (so it doesn't suffer k-NN's curse of dimensionality) and outputs a
 * calibrated probability directly — ideal for the live percentage readout.
 *
 * Exposes the SAME interface as KNNClassifier (addTrainingExample / predict /
 * predictProbability / count) so the two are interchangeable.
 */
export class LogisticRegression {
  constructor({ l2 = 1.0, lr = 0.2, iters = 800, minConfidence = 0 } = {}) {
    this.l2 = l2;
    this.lr = lr;
    this.iters = iters;
    this.minConfidence = minConfidence;
    this.examples = [];
    this.keys = null;
    this.mean = {};
    this.std = {};
    this.w = {};
    this.b = 0;
    this._trained = false;
  }

  addTrainingExample(features, label, metadata = {}) {
    if (!["left", "right"].includes(label)) {
      throw new Error(`Invalid label: ${label}. Must be "left" or "right".`);
    }
    this.examples.push({ features: { ...features }, label, ...metadata });
    this._trained = false;
  }

  count() {
    return this.examples.length;
  }

  countByLabel() {
    const counts = { left: 0, right: 0 };
    for (const ex of this.examples) if (ex.label in counts) counts[ex.label]++;
    return counts;
  }

  _ensureTrained() {
    if (this._trained) return;
    const ex = this.examples;
    const n = ex.length;

    // Feature keys = every finite numeric feature seen in training.
    const keys = new Set();
    for (const e of ex) {
      for (const k in e.features) if (Number.isFinite(e.features[k])) keys.add(k);
    }
    this.keys = [...keys];

    // Standardise each feature (z-score) so the L2 penalty is even-handed and
    // gradient descent is well conditioned.
    for (const k of this.keys) {
      let s = 0;
      for (const e of ex) s += e.features[k] ?? 0;
      const m = s / n;
      let sq = 0;
      for (const e of ex) {
        const d = (e.features[k] ?? 0) - m;
        sq += d * d;
      }
      this.mean[k] = m;
      this.std[k] = Math.max(1e-6, Math.sqrt(sq / n));
    }

    const y = ex.map((e) => (e.label === "left" ? 1 : 0));
    const X = ex.map((e) => {
      const v = {};
      for (const k of this.keys) v[k] = ((e.features[k] ?? this.mean[k]) - this.mean[k]) / this.std[k];
      return v;
    });

    for (const k of this.keys) this.w[k] = 0;
    this.b = 0;

    // Batch gradient descent on mean logistic loss + L2 (bias unregularised).
    for (let it = 0; it < this.iters; it++) {
      const gw = {};
      for (const k of this.keys) gw[k] = 0;
      let gb = 0;
      for (let i = 0; i < n; i++) {
        let z = this.b;
        for (const k of this.keys) z += this.w[k] * X[i][k];
        const p = _sigmoid(z);
        const err = p - y[i];
        for (const k of this.keys) gw[k] += err * X[i][k];
        gb += err;
      }
      for (const k of this.keys) {
        const grad = gw[k] / n + (this.l2 * this.w[k]) / n;
        this.w[k] -= this.lr * grad;
      }
      this.b -= this.lr * (gb / n);
    }

    this._trained = true;
  }

  _probLeft(features) {
    this._ensureTrained();
    let z = this.b;
    for (const k of this.keys) {
      const raw = Number.isFinite(features[k]) ? features[k] : this.mean[k];
      z += this.w[k] * ((raw - this.mean[k]) / this.std[k]);
    }
    return _sigmoid(z);
  }

  predictProbability(features) {
    if (this.examples.length === 0) {
      return { pLeft: 0.5, pRight: 0.5, neighbors: [], error: "No training examples" };
    }
    const p = this._probLeft(features);
    return { pLeft: p, pRight: 1 - p, neighbors: [] };
  }

  predict(features) {
    if (this.examples.length === 0) {
      return { label: null, confidence: 0, neighbors: [], rawVotes: { left: 0, right: 0 }, error: "No training examples" };
    }
    const p = this._probLeft(features);
    const label = p >= 0.5 ? "left" : "right";
    const confidence = Math.max(p, 1 - p);
    if (this.minConfidence > 0 && confidence < this.minConfidence) {
      return { label: null, confidence, neighbors: [], rawVotes: { left: 0, right: 0 } };
    }
    return {
      label,
      confidence,
      neighbors: [],
      rawVotes: { left: p >= 0.5 ? 1 : 0, right: p < 0.5 ? 1 : 0 },
    };
  }
}

function _sigmoid(z) {
  if (z < -30) return 0;
  if (z > 30) return 1;
  return 1 / (1 + Math.exp(-z));
}

/**
 * Adaptive k for k-NN: the √n rule of thumb, forced odd (no ties) and clamped
 * to a sensible range. Replaces the old fixed k=5 — too large for a handful of
 * trips (washes out local structure), too small for a big set.
 */
export function adaptiveK(n) {
  let k = Math.round(Math.sqrt(Math.max(1, n)));
  if (k % 2 === 0) k += 1;
  return Math.max(3, Math.min(15, Math.min(k, Math.max(1, n))));
}

/**
 * Mean held-out accuracy over several random splits, for any model exposing
 * predict(features).label. makeModel(trainRows) returns a trained model.
 */
function _accuracyCV(data, makeModel, repeats = 12, testFraction = 0.3) {
  let total = 0;
  let valid = 0;
  for (let r = 0; r < repeats; r++) {
    const shuffled = [...data];
    for (let i = shuffled.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
    }
    const splitIdx = Math.ceil(shuffled.length * (1 - testFraction));
    const train = shuffled.slice(0, splitIdx);
    const test = shuffled.slice(splitIdx);
    if (train.length < 2 || test.length < 1) continue;
    // Need both classes in train for a meaningful fit.
    if (!train.some((d) => d.label === "left") || !train.some((d) => d.label === "right")) continue;

    const model = makeModel(train);
    let correct = 0;
    for (const t of test) if (model.predict(t.features).label === t.label) correct++;
    total += correct / test.length;
    valid++;
  }
  return valid > 0 ? total / valid : null;
}

/** Copy only the listed feature keys from a vector (drops the rest). */
function _pruneTo(features, keySet) {
  const out = {};
  for (const k of keySet) if (k in features) out[k] = features[k];
  return out;
}

/**
 * Rank features by class separation (a Fisher / effect-size score in z-scored
 * space, regularised so a near-constant feature can't rank high on a fluke).
 * Most of the ~46 features are sign-blind (energy, RMS, FFT…) and carry no
 * left/right information; this floats the few that do (world-frame yaw, signed
 * means, skewness) to the top. Computed from whatever rows it's given, so when
 * called inside a CV fold it never sees the held-out data.
 */
function rankFeatures(rows) {
  const keys = new Set();
  for (const r of rows) for (const k in r.features) if (Number.isFinite(r.features[k])) keys.add(k);
  const keyArr = [...keys];
  const n = rows.length || 1;

  const mean = {};
  const std = {};
  for (const k of keyArr) {
    let s = 0;
    for (const r of rows) s += r.features[k] ?? 0;
    const m = s / n;
    let sq = 0;
    for (const r of rows) {
      const d = (r.features[k] ?? 0) - m;
      sq += d * d;
    }
    mean[k] = m;
    std[k] = Math.max(1e-6, Math.sqrt(sq / n));
  }
  const z = (r, k) => ((r.features[k] ?? mean[k]) - mean[k]) / std[k];

  const L = rows.filter((r) => r.label === "left");
  const R = rows.filter((r) => r.label === "right");
  const classStat = (grp, k) => {
    if (grp.length === 0) return { m: 0, sd: 0 };
    let s = 0;
    for (const r of grp) s += z(r, k);
    const m = s / grp.length;
    let sq = 0;
    for (const r of grp) {
      const d = z(r, k) - m;
      sq += d * d;
    }
    return { m, sd: Math.sqrt(sq / grp.length) };
  };

  return keyArr
    .map((k) => {
      const a = classStat(L, k);
      const b = classStat(R, k);
      return [k, Math.abs(a.m - b.m) / (a.sd + b.sd + 0.5)];
    })
    .sort((x, y) => y[1] - x[1])
    .map((s) => s[0]);
}

/**
 * Wraps a classifier so every input is pruned to a fixed feature subset before
 * predicting. Keeps the inner model honest about which features it was trained
 * on (essential for k-NN, whose distance otherwise sums over the query's keys).
 */
class FeatureSubsetModel {
  constructor(inner, keys) {
    this.inner = inner;
    this.keys = keys;
    this._keySet = new Set(keys);
  }
  predict(features) {
    return this.inner.predict(_pruneTo(features, this._keySet));
  }
  predictProbability(features) {
    return this.inner.predictProbability(_pruneTo(features, this._keySet));
  }
  count() {
    return this.inner.count();
  }
}

/**
 * Model selection: search over {k-NN, logistic regression} × {how many of the
 * top-ranked features to keep}, estimating each combination by repeated
 * cross-validation on the user's OWN data (feature ranking done inside each
 * fold — no leakage). Returns the winner trained on all examples.
 *
 * The all-feature k-NN is the baseline; a different model/feature-count is only
 * adopted when it beats that baseline by a clear margin (>2pp). So this can
 * never quietly regress — it only upgrades when a leaner or linear model is
 * genuinely better on this data.
 *
 * @returns {{ model, name, accuracy, featureCount, baselineAccuracy }}
 */
export function selectBestModel(examples, minConfidence = 0.6) {
  // Only train on well-formed examples: a stray label (hand-edited backup,
  // future import format) would make addTrainingExample throw and take the
  // whole live forecast down with it.
  const usable = (examples || []).filter(
    (e) =>
      e.features &&
      typeof e.features === "object" &&
      !Array.isArray(e.features) &&
      (e.label === "left" || e.label === "right")
  );
  const data = usable.map((e) => ({
    features: e.features,
    label: e.label,
    recordingId: e.recordingId,
    timestamp: e.timestamp,
  }));

  // K = Infinity → keep all features.
  const buildKNN = (rows, K, mc = 0) => {
    const keys = K === Infinity ? rankFeatures(rows) : rankFeatures(rows).slice(0, K);
    const keySet = new Set(keys);
    const inner = new KNNClassifier(adaptiveK(rows.length), mc);
    for (const r of rows) {
      inner.addTrainingExample(_pruneTo(r.features, keySet), r.label, {
        recordingId: r.recordingId,
        timestamp: r.timestamp,
      });
    }
    return new FeatureSubsetModel(inner, keys);
  };
  const buildLR = (rows, K, mc = 0) => {
    const keys = K === Infinity ? rankFeatures(rows) : rankFeatures(rows).slice(0, K);
    const keySet = new Set(keys);
    const inner = new LogisticRegression({ minConfidence: mc });
    for (const r of rows) inner.addTrainingExample(_pruneTo(r.features, keySet), r.label);
    return new FeatureSubsetModel(inner, keys);
  };

  const counts = { left: 0, right: 0 };
  for (const d of data) if (d.label in counts) counts[d.label]++;
  const totalFeatures = rankFeatures(data).length;

  // Too little data to trust any comparison — use interpretable all-feature k-NN.
  if (counts.left < 5 || counts.right < 5) {
    return {
      model: buildKNN(data, Infinity, minConfidence),
      name: "knn",
      accuracy: null,
      featureCount: totalFeatures,
      baselineAccuracy: null,
    };
  }

  // Baseline: all-feature k-NN (the safe default we must clearly beat to switch).
  const baselineAccuracy = _accuracyCV(data, (rows) => buildKNN(rows, Infinity, 0));

  // Candidate feature counts to try (top-K by class separation).
  const candidateKs = [...new Set([6, 10, 16, totalFeatures])]
    .filter((k) => k >= 3 && k <= totalFeatures)
    .sort((a, b) => a - b);

  let bestAlt = null; // best non-baseline combination
  for (const K of candidateKs) {
    for (const cand of [
      { name: "knn", build: buildKNN },
      { name: "logreg", build: buildLR },
    ]) {
      const acc = _accuracyCV(data, (rows) => cand.build(rows, K, 0));
      if (acc != null && (!bestAlt || acc > bestAlt.acc)) {
        bestAlt = { name: cand.name, K, acc, build: cand.build };
      }
    }
  }

  const switchOver =
    bestAlt != null && baselineAccuracy != null && bestAlt.acc > baselineAccuracy + 0.02;

  if (switchOver) {
    return {
      model: bestAlt.build(data, bestAlt.K, minConfidence),
      name: bestAlt.name,
      accuracy: bestAlt.acc,
      featureCount: bestAlt.K,
      baselineAccuracy,
    };
  }
  return {
    model: buildKNN(data, Infinity, minConfidence),
    name: "knn",
    accuracy: baselineAccuracy,
    featureCount: totalFeatures,
    baselineAccuracy,
  };
}

/**
 * Cross-validation helper: evaluate classifier on a held-out test set.
 *
 * Splits data into training and test sets, trains the classifier,
 * and reports accuracy and per-class metrics.
 *
 * @param {Array<{features, label}>} data - labeled examples
 * @param {number} k - k-NN parameter
 * @param {number} testFraction - fraction to hold out for testing (default 0.2)
 * @returns {Object} metrics: {accuracy, leftPrecision, leftRecall, rightPrecision, rightRecall, ...}
 */
export function evaluateCrossValidation(
  data,
  k = 5,
  testFraction = 0.2
) {
  if (data.length < 5) {
    return { error: "Not enough examples for cross-validation" };
  }

  // Shuffle (unbiased Fisher-Yates) and split. A `.sort(() => Math.random()
  // - 0.5)` shuffle is NOT uniform — comparator-based shuffles leave elements
  // correlated with their start position, which skews the held-out split and
  // therefore the accuracy estimate.
  const shuffled = [...data];
  for (let i = shuffled.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
  }
  const splitIdx = Math.ceil(shuffled.length * (1 - testFraction));
  const trainData = shuffled.slice(0, splitIdx);
  const testData = shuffled.slice(splitIdx);

  // Train classifier
  const clf = new KNNClassifier(k);
  for (const ex of trainData) {
    clf.addTrainingExample(ex.features, ex.label);
  }

  // Evaluate on test set
  const predictions = testData.map(ex => {
    const pred = clf.predict(ex.features);
    return {
      true: ex.label,
      pred: pred.label,
      confidence: pred.confidence,
    };
  });

  // Metrics
  const cm = confusionMatrix(predictions);
  const metrics = {
    accuracy: (cm.tp + cm.tn) / (cm.tp + cm.tn + cm.fp + cm.fn),
    leftPrecision: cm.tp / (cm.tp + cm.fp || 1),
    leftRecall: cm.tp / (cm.tp + cm.fn || 1),
    rightPrecision: cm.tn / (cm.tn + cm.fn || 1),
    rightRecall: cm.tn / (cm.tn + cm.fp || 1),
    confusionMatrix: cm,
    testSetSize: testData.length,
    trainSetSize: trainData.length,
  };

  return metrics;
}

/**
 * Compute confusion matrix.
 * Assumes "left" is the positive class.
 */
function confusionMatrix(predictions) {
  let tp = 0, tn = 0, fp = 0, fn = 0;
  for (const p of predictions) {
    if (p.true === "left" && p.pred === "left") tp++;
    else if (p.true === "right" && p.pred === "right") tn++;
    else if (p.true === "right" && p.pred === "left") fp++;
    else if (p.true === "left" && p.pred === "right") fn++;
  }
  return { tp, tn, fp, fn };
}

/**
 * Sum array elements.
 */
function sum(arr) {
  return arr.reduce((a, b) => a + b, 0);
}
