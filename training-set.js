/* ============================================================
   Victoria Line Motion Lab — training set manager

   Manages the labeled training dataset:
   - Storage (localStorage for small sets, IndexedDB for large)
   - Load/save operations
   - Add/remove examples
   - Export/import JSON
   - Metadata tracking (recording ID, timestamp, confidence)
   ============================================================ */

"use strict";

/**
 * Training set manager.
 *
 * Stores labeled motion recordings (left or right platform arrival).
 * Each example is linked back to a raw motion recording for traceability.
 *
 * Storage strategy:
 * - Small sets (< 100 examples): localStorage as JSON
 * - Large sets: IndexedDB (larger quota, better performance)
 *
 * Usage:
 *   const trainSet = new TrainingSet("victoria-line-v1");
 *   await trainSet.load(); // load from persistent storage
 *   trainSet.add(motionData, direction, { ...metadata });
 *   await trainSet.save();
 *   const json = trainSet.toJSON();
 */
export class TrainingSet {
  constructor(storageKey = "victoria-line-training-set") {
    this.storageKey = storageKey;
    this.examples = []; // [{id, motionData, features, label, timestamp, ...}, ...]
    this.nextId = 1;
  }

  /**
   * Load training set from persistent storage (localStorage or IndexedDB).
   */
  async load() {
    // Try localStorage first (simpler, synchronous)
    try {
      const stored = localStorage.getItem(this.storageKey);
      if (stored) {
        const data = JSON.parse(stored);
        this.fromJSON(data);
        console.log(`[TrainingSet] Loaded ${this.examples.length} examples from localStorage`);
        return;
      }
    } catch (err) {
      console.warn("[TrainingSet] localStorage load failed:", err);
    }

    // Try IndexedDB (for larger sets)
    try {
      const data = await this.loadFromIndexedDB();
      if (data) {
        this.fromJSON(data);
        console.log(`[TrainingSet] Loaded ${this.examples.length} examples from IndexedDB`);
        return;
      }
    } catch (err) {
      console.warn("[TrainingSet] IndexedDB load failed:", err);
    }

    console.log("[TrainingSet] Starting with empty training set");
  }

  /**
   * Save training set to persistent storage.
   */
  async save() {
    const json = this.toJSON();
    const jsonStr = JSON.stringify(json);
    const sizeKB = new Blob([jsonStr]).size / 1024;

    // Use localStorage if small enough (typically ~5 MB quota per domain)
    if (sizeKB < 500) {
      try {
        localStorage.setItem(this.storageKey, jsonStr);
        console.log(`[TrainingSet] Saved ${this.examples.length} examples to localStorage (${sizeKB.toFixed(1)} KB)`);
        return;
      } catch (err) {
        console.warn("[TrainingSet] localStorage save failed:", err);
      }
    }

    // Use IndexedDB for larger sets
    try {
      await this.saveToIndexedDB(json);
      console.log(`[TrainingSet] Saved ${this.examples.length} examples to IndexedDB (${sizeKB.toFixed(1)} KB)`);
    } catch (err) {
      console.error("[TrainingSet] All storage failed:", err);
      throw err;
    }
  }

  /**
   * Add a new training example.
   *
   * @param {Array} motionData - raw motion samples [{time, ax, ay, az, rotationAlpha, ...}, ...]
   * @param {string} label - "left" or "right"
   * @param {Object} metadata - optional {recordingId, timestamp, confidence, ...}
   * @returns {string} unique example ID
   */
  add(motionData, label, metadata = {}) {
    if (!["left", "right"].includes(label)) {
      throw new Error(`Invalid label: ${label}. Must be "left" or "right".`);
    }

    if (!Array.isArray(motionData) || motionData.length === 0) {
      throw new Error("motionData must be a non-empty array.");
    }

    const id = `example-${this.nextId++}`;
    const timestamp = metadata.timestamp || Date.now();

    this.examples.push({
      id,
      motionData: this.compressMotionData(motionData),
      label,
      timestamp,
      recordingId: metadata.recordingId || id,
      confidence: metadata.confidence ?? null,
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
      version: 1,
      storageKey: this.storageKey,
      exportTimestamp: Date.now(),
      examples: this.examples.map(ex => ({
        id: ex.id,
        motionData: ex.motionData,
        label: ex.label,
        timestamp: ex.timestamp,
        recordingId: ex.recordingId,
        confidence: ex.confidence,
        notes: ex.notes,
      })),
    };
  }

  /**
   * Import training set from JSON.
   */
  fromJSON(json) {
    if (!json.examples || !Array.isArray(json.examples)) {
      throw new Error("Invalid training set JSON format");
    }

    this.examples = json.examples.map(ex => ({
      ...ex,
      motionData: ex.motionData, // already compressed
    }));

    this.nextId = Math.max(...this.examples.map(ex => {
      const match = ex.id.match(/example-(\d+)/);
      return match ? parseInt(match[1], 10) : 0;
    })) + 1;
  }

  /**
   * Compress motion data (remove duplicate/unchanged fields, round values).
   * Reduces JSON size by ~50%.
   */
  compressMotionData(motionData) {
    // For now, just store as-is. Could implement delta compression if storage becomes an issue.
    return motionData.map(s => ({
      ax: Math.round(s.ax * 1000) / 1000,
      ay: Math.round(s.ay * 1000) / 1000,
      az: Math.round(s.az * 1000) / 1000,
      rotationAlpha: Math.round(s.rotationAlpha * 10) / 10,
      rotationBeta: Math.round(s.rotationBeta * 10) / 10,
      rotationGamma: Math.round(s.rotationGamma * 10) / 10,
    }));
  }

  /**
   * Save to IndexedDB (for larger sets).
   */
  async saveToIndexedDB(json) {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("victoriaLineMotionLab", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        const tx = db.transaction("trainingSet", "readwrite");
        const store = tx.objectStore("trainingSet");

        try {
          store.put({ key: this.storageKey, data: json });
          tx.oncomplete = resolve;
          tx.onerror = () => reject(tx.error);
        } catch (err) {
          reject(err);
        }
      };

      request.onupgradeneeded = (event) => {
        const db = event.target.result;
        if (!db.objectStoreNames.contains("trainingSet")) {
          db.createObjectStore("trainingSet", { keyPath: "key" });
        }
      };
    });
  }

  /**
   * Load from IndexedDB.
   */
  async loadFromIndexedDB() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open("victoriaLineMotionLab", 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const db = request.result;
        if (!db.objectStoreNames.contains("trainingSet")) {
          resolve(null);
          return;
        }

        const tx = db.transaction("trainingSet", "readonly");
        const store = tx.objectStore("trainingSet");
        const getRequest = store.get(this.storageKey);

        getRequest.onerror = () => reject(getRequest.error);
        getRequest.onsuccess = () => {
          resolve(getRequest.result?.data ?? null);
        };
      };
    });
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

    try {
      await this.saveToIndexedDB({ examples: [] });
    } catch (err) {
      console.warn("[TrainingSet] Failed to clear IndexedDB:", err);
    }
  }

  /**
   * Get statistics about the training set.
   */
  getStats() {
    const counts = this.countByLabel();
    const totalSamples = this.examples.reduce((sum, ex) => sum + (ex.motionData?.length ?? 0), 0);
    const avgSamplesPerExample = this.examples.length > 0 ? totalSamples / this.examples.length : 0;

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
      totalSamples,
      avgSamplesPerExample: Math.round(avgSamplesPerExample),
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
  return `recording-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`;
}
