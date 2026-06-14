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
