/* ============================================================
   Victoria Line Motion Lab — training set manager

   Manages the labeled training dataset:
   - localStorage persistence (feature vectors are tiny — ~35 floats
     per example — so even hundreds of trips fit comfortably)
   - Add/remove examples
   - Export/import JSON
   - Metadata tracking (recording ID, timestamp, notes)

   Each example stores the EXTRACTED FEATURES of a trip, not the raw
   motion samples. Features are all the k-NN classifier needs, and raw
   recordings can always be saved separately via the app's normal
   "Save Recording" export.
   ============================================================ */

"use strict";

/**
 * Training set manager.
 *
 * Usage:
 *   const trainSet = new TrainingSet();
 *   await trainSet.load();                      // from localStorage
 *   trainSet.add(features, "left", { ... });    // features from extractFeatures()
 *   await trainSet.save();
 */
export class TrainingSet {
  constructor(storageKey = "victoria-line-training-set") {
    this.storageKey = storageKey;
    this.examples = []; // [{id, features, label, timestamp, recordingId, notes}, ...]
    this.nextId = 1;
  }

  /**
   * Load training set from localStorage.
   */
  async load() {
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        this.fromJSON(JSON.parse(stored));
        console.log(`[TrainingSet] Loaded ${this.examples.length} examples from localStorage`);
        return;
      }
    } catch (err) {
      console.warn("[TrainingSet] localStorage load failed:", err);
    }

    console.log("[TrainingSet] Starting with empty training set");
  }

  /**
   * Save training set to localStorage.
   */
  async save() {
    const jsonStr = JSON.stringify(this.toJSON());
    const sizeKB = new Blob([jsonStr]).size / 1024;

    try {
      localStorage.setItem(this.storageKey, jsonStr);
      console.log(`[TrainingSet] Saved ${this.examples.length} examples to localStorage (${sizeKB.toFixed(1)} KB)`);
    } catch (err) {
      console.error("[TrainingSet] localStorage save failed:", err);
      throw err;
    }
  }

  /**
   * Add a new training example.
   *
   * @param {Object} features - feature vector from extractFeatures()
   * @param {string} label - "left" or "right"
   * @param {Object} metadata - optional {recordingId, timestamp, notes}
   * @returns {string} unique example ID
   */
  add(features, label, metadata = {}) {
    if (!["left", "right"].includes(label)) {
      throw new Error(`Invalid label: ${label}. Must be "left" or "right".`);
    }

    if (Array.isArray(features) || features === null || typeof features !== "object") {
      throw new Error(
        "Pass a FEATURES object (from extractFeatures), not raw motion data."
      );
    }

    const id = `example-${this.nextId++}`;

    this.examples.push({
      id,
      features: { ...features },
      label,
      timestamp: metadata.timestamp || Date.now(),
      recordingId: metadata.recordingId || id,
      notes: metadata.notes ?? "",
    });

    return id;
  }

  /**
   * Remove an example by ID.
   */
  remove(id) {
    const idx = this.examples.findIndex(ex => ex.id === id);
    if (idx >= 0) {
      this.examples.splice(idx, 1);
      return true;
    }
    return false;
  }

  /**
   * Get an example by ID.
   */
  get(id) {
    return this.examples.find(ex => ex.id === id);
  }

  /**
   * Get all examples with a given label.
   */
  getByLabel(label) {
    return this.examples.filter(ex => ex.label === label);
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
   * Get the total number of examples.
   */
  count() {
    return this.examples.length;
  }

  /**
   * Check if we have enough examples to train (at least a few of each class).
   */
  isReadyForTraining(minPerClass = 5) {
    const counts = this.countByLabel();
    return counts.left >= minPerClass && counts.right >= minPerClass;
  }

  /**
   * Export training set as JSON (portable format).
   */
  toJSON() {
    return {
      version: 2, // v2: examples carry `features`, not raw `motionData`
      storageKey: this.storageKey,
      exportTimestamp: Date.now(),
      examples: this.examples.map(ex => ({
        id: ex.id,
        features: ex.features,
        label: ex.label,
        timestamp: ex.timestamp,
        recordingId: ex.recordingId,
        notes: ex.notes,
      })),
    };
  }

  /**
   * Import training set from JSON.
   * Examples without a features object (e.g. from an old v1 export that
   * stored raw motion data) are skipped with a warning — they can't be
   * used for prediction.
   */
  fromJSON(json) {
    if (!json.examples || !Array.isArray(json.examples)) {
      throw new Error("Invalid training set JSON format");
    }

    const usable = [];
    let skipped = 0;
    for (const ex of json.examples) {
      if (ex.features && typeof ex.features === "object" && !Array.isArray(ex.features)) {
        usable.push({
          id: ex.id,
          features: ex.features,
          label: ex.label,
          timestamp: ex.timestamp,
          recordingId: ex.recordingId,
          notes: ex.notes ?? "",
        });
      } else {
        skipped++;
      }
    }
    if (skipped > 0) {
      console.warn(`[TrainingSet] Skipped ${skipped} example(s) with no feature vector (old format)`);
    }

    this.examples = usable;

    // Resume ID numbering after the highest existing ID (0 when empty).
    const maxId = this.examples.reduce((max, ex) => {
      const match = /example-(\d+)/.exec(ex.id ?? "");
      return match ? Math.max(max, parseInt(match[1], 10)) : max;
    }, 0);
    this.nextId = maxId + 1;
  }

  /**
   * Clear all training data (destructive).
   */
  async clear() {
    this.examples = [];
    this.nextId = 1;

    try {
      localStorage.removeItem(this.storageKey);
    } catch (err) {
      console.warn("[TrainingSet] Failed to clear localStorage:", err);
    }
  }

  /**
   * Get statistics about the training set.
   */
  getStats() {
    const counts = this.countByLabel();

    const oldest = this.examples.length > 0
      ? Math.min(...this.examples.map(ex => ex.timestamp))
      : null;
    const newest = this.examples.length > 0
      ? Math.max(...this.examples.map(ex => ex.timestamp))
      : null;

    return {
      totalExamples: this.examples.length,
      leftCount: counts.left,
      rightCount: counts.right,
      balance: counts.left > 0 ? (counts.right / counts.left).toFixed(2) : "N/A",
      oldestTimestamp: oldest,
      newestTimestamp: newest,
      isReadyForTraining: this.isReadyForTraining(),
    };
  }
}

/**
 * Helper: create a unique recording ID (for metadata tracking).
 */
export function generateRecordingId() {
  return `recording-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}
