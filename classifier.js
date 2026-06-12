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

import { euclideanDistance } from "./features.js";

/**
 * k-Nearest Neighbors classifier for binary classification.
 *
 * Usage:
 *   const clf = new KNNClassifier(k = 5);
 *   clf.addTrainingExample(featureVector, "left");
 *   clf.addTrainingExample(featureVector, "right");
 *   const pred = clf.predict(newFeatureVector);
 *   // pred = { label: "left", confidence: 0.87, neighbors: [...] }
 */
export class KNNClassifier {
  constructor(k = 5) {
    this.k = k;
    this.examples = []; // [{features, label, recordingId, timestamp}, ...]
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
  }

  /**
   * Remove all training examples (reset).
   */
  clear() {
    this.examples = [];
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

    return {
      label,
      confidence,
      neighbors,
      rawVotes: votes,
    };
  }

  /**
   * Find the k nearest neighbors to a feature vector.
   *
   * @param {Object} features - query feature vector
   * @param {number} k - number of neighbors to return
   * @returns {Array} sorted by distance (nearest first), each with distance
   */
  findNearestNeighbors(features, k = this.k) {
    const distances = this.examples.map((ex, idx) => ({
      ...ex,
      distance: euclideanDistance(features, ex.features),
      index: idx,
    }));

    // Sort by distance (nearest first)
    distances.sort((a, b) => a.distance - b.distance);

    // Return top k
    return distances.slice(0, Math.min(k, distances.length));
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

  // Shuffle and split
  const shuffled = [...data].sort(() => Math.random() - 0.5);
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
