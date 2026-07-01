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
   * Re-extract feature vectors that were produced by an older feature-extractor
   * version, in place, from each example's stored raw motion data.
   *
   * Feature extraction has changed several times in this project (e.g. removing
   * the trajectory block, then localising to the fork window). Without this,
   * old examples keep features on a different scale/definition than new ones,
   * and the classifier silently compares apples to oranges. Because v3 examples
   * carry their full rawMotionData, we can bring every recording up to the
   * current definition transparently on load. Examples whose raw data was
   * dropped (quota fallback, or v2 imports) are left untouched — their stale
   * features are still better than discarding the recording.
   *
   * @param {(motionData:Array, rateHz:number)=>Object} extractFn - extractor (e.g. extractForkFeatures)
   * @param {number} version - the current FEATURE_VERSION
   * @param {number} sampleRateHz
   * @returns {Promise<number>} how many examples were re-extracted
   */
  async reprocessFeatures(extractFn, version, sampleRateHz = 60) {
    let changed = 0;
    for (const ex of this.examples) {
      if (ex.featureVersion === version) continue;
      if (!Array.isArray(ex.rawMotionData) || ex.rawMotionData.length < 2) continue;
      try {
        ex.features = extractFn(ex.rawMotionData, sampleRateHz);
        ex.featureVersion = version;
        changed++;
      } catch (err) {
        console.warn(`[TrainingSet] Reprocess failed for ${ex.id}:`, err);
      }
    }
    if (changed > 0) {
      try {
        await this.save();
      } catch (err) {
        console.warn("[TrainingSet] Save after reprocess failed:", err);
      }
    }
    return changed;
  }

  /**
   * Save training set to localStorage.
   */
  async save() {
    try {
      const jsonStr = JSON.stringify(this.toJSON());
      localStorage.setItem(this.storageKey, jsonStr);
      const sizeKB = new Blob([jsonStr]).size / 1024;
      console.log(`[TrainingSet] Saved ${this.examples.length} examples to localStorage (${sizeKB.toFixed(1)} KB)`);
      return;
    } catch (err) {
      // Quota exceeded: raw motion data is the bulk of the payload. Drop it and
      // retry so the recordings (features + labels) STILL persist across
      // reloads — far better than losing everything. Raw data can always be
      // re-captured or kept via the manual "Backup Data" export.
      if (_isQuotaError(err)) {
        try {
          const lite = this.toJSON();
          for (const ex of lite.examples) delete ex.rawMotionData;
          localStorage.setItem(this.storageKey, JSON.stringify(lite));
          console.warn(
            "[TrainingSet] localStorage quota hit — saved WITHOUT raw motion data so the recordings still persist. Export a backup to keep raw data."
          );
          return;
        } catch (err2) {
          console.error("[TrainingSet] Save failed even without raw data:", err2);
          throw err2;
        }
      }
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
      featureVersion: metadata.featureVersion, // which extractor produced `features`
      label,
      timestamp: metadata.timestamp || Date.now(),
      recordingId: metadata.recordingId || id,
      sampleCount: metadata.sampleCount,
      duration: metadata.duration,
      rawMotionData: metadata.rawMotionData, // full raw timestamped sensor data
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
   * Now includes full raw motion data for each recording.
   */
  toJSON() {
    return {
      version: 3, // v3: examples carry full rawMotionData + features
      storageKey: this.storageKey,
      exportTimestamp: Date.now(),
      examples: this.examples.map(ex => ({
        id: ex.id,
        features: ex.features,
        featureVersion: ex.featureVersion,
        label: ex.label,
        timestamp: ex.timestamp,
        recordingId: ex.recordingId,
        sampleCount: ex.sampleCount,
        duration: ex.duration,
        rawMotionData: ex.rawMotionData, // full recording data
        notes: ex.notes,
      })),
    };
  }

  /**
   * Import training set from JSON.
   * Handles v2 (features only) and v3 (features + rawMotionData) formats.
   * Examples without a features object (v1 format) or without a valid
   * left/right label are skipped with a warning — one bad example in a
   * hand-edited backup must not be able to crash every later classifier
   * build (addTrainingExample throws on labels it doesn't recognise).
   */
  fromJSON(json) {
    if (!json.examples || !Array.isArray(json.examples)) {
      throw new Error("Invalid training set JSON format");
    }

    const usable = [];
    let skipped = 0;
    for (const ex of json.examples) {
      const label = typeof ex.label === "string" ? ex.label.toLowerCase() : "";
      const hasFeatures =
        ex.features && typeof ex.features === "object" && !Array.isArray(ex.features);
      if (hasFeatures && (label === "left" || label === "right")) {
        usable.push({
          id: ex.id,
          features: ex.features,
          featureVersion: ex.featureVersion, // undefined for pre-versioned data
          label,
          timestamp: ex.timestamp,
          recordingId: ex.recordingId,
          sampleCount: ex.sampleCount,
          duration: ex.duration,
          rawMotionData: ex.rawMotionData, // may be undefined for v2 imports
          notes: ex.notes ?? "",
        });
      } else {
        skipped++;
      }
    }
    if (skipped > 0) {
      console.warn(`[TrainingSet] Skipped ${skipped} example(s) with no feature vector or invalid label`);
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

  /**
   * Export training set as a downloadable JSON file blob.
   * Returns {blob, filename} for use with download links.
   */
  exportAsFile() {
    const data = this.toJSON();
    const counts = this.countByLabel();
    data.exportNote = `Victoria Line Motion Lab training data — ${counts.left}L / ${counts.right}R examples`;

    const jsonStr = JSON.stringify(data, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const dateStr = new Date().toISOString().slice(0, 10);
    const filename = `victoria-training-${dateStr}-${counts.left}L${counts.right}R.json`;

    return { blob, filename, data };
  }

  /**
   * Generate a downloadable file and trigger browser download.
   * The file can be manually saved to OneDrive or imported on another device.
   */
  triggerDownload() {
    const { blob, filename } = this.exportAsFile();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`[TrainingSet] Downloaded backup: ${filename}`);
  }

  /**
   * Create a localStorage-based auto-backup (internal copy).
   * Useful as a fallback recovery mechanism.
   */
  async autoBackup() {
    const backupKey = `${this.storageKey}--backup`;
    try {
      // The safety net only needs the feature vectors to rebuild the
      // classifier after an accidental clear. Raw motion data is large and
      // already lives in the primary store (and in the manual "Backup Data"
      // download), so we strip it here to avoid doubling localStorage usage
      // and tripping Safari's ~5 MB quota.
      const json = this.toJSON();
      for (const ex of json.examples) delete ex.rawMotionData;
      localStorage.setItem(backupKey, JSON.stringify(json));
      console.log(`[TrainingSet] Auto-backup saved (${this.examples.length} examples, features only)`);
    } catch (err) {
      console.warn("[TrainingSet] Auto-backup failed:", err);
    }
  }

  /**
   * Restore from auto-backup (recovers from accidental deletion).
   */
  async restoreFromAutoBackup() {
    const backupKey = `${this.storageKey}--backup`;
    try {
      const stored = localStorage.getItem(backupKey);
      if (stored) {
        const json = JSON.parse(stored);
        this.fromJSON(json);
        console.log(`[TrainingSet] Restored from auto-backup: ${this.examples.length} examples`);
        return true;
      }
    } catch (err) {
      console.warn("[TrainingSet] Auto-backup restore failed:", err);
    }
    return false;
  }

  /**
   * Merge another training set into this one (for restoring from exports).
   * Deduplicates by recordingId to avoid double-imports.
   *
   * Incoming examples get FRESH local ids: every device numbers its own
   * examples from example-1, so keeping the imported id would collide with
   * ids already in this set — and get()/remove() (e.g. the history delete
   * button) would silently hit the wrong recording. recordingId, the stable
   * cross-device key, is preserved (or assigned, for very old exports that
   * lack one, so those aren't all deduped against each other as undefined).
   */
  mergeFrom(otherTrainingSet) {
    // Make sure new ids start beyond everything already in this set.
    const maxId = this.examples.reduce((max, ex) => {
      const match = /example-(\d+)/.exec(ex.id ?? "");
      return match ? Math.max(max, parseInt(match[1], 10)) : max;
    }, 0);
    this.nextId = Math.max(this.nextId, maxId + 1);

    const existingIds = new Set(
      this.examples.map(ex => ex.recordingId).filter(id => id != null)
    );
    let merged = 0;

    for (const ex of otherTrainingSet.examples) {
      if (ex.recordingId != null && existingIds.has(ex.recordingId)) continue;
      const id = `example-${this.nextId++}`;
      this.examples.push({ ...ex, id, recordingId: ex.recordingId ?? id });
      if (ex.recordingId != null) existingIds.add(ex.recordingId);
      merged++;
    }

    console.log(`[TrainingSet] Merged ${merged} new examples`);
    return merged;
  }
}

/**
 * Helper: create a unique recording ID (for metadata tracking).
 */
export function generateRecordingId() {
  return `recording-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

/**
 * Detect a localStorage quota-exceeded error across browsers. Safari throws
 * QUOTA_EXCEEDED_ERR (code 22); some older WebKit builds use code 1014 with
 * name "NS_ERROR_DOM_QUOTA_REACHED".
 */
function _isQuotaError(err) {
  return (
    err &&
    (err.name === "QuotaExceededError" ||
      err.name === "NS_ERROR_DOM_QUOTA_REACHED" ||
      err.code === 22 ||
      err.code === 1014)
  );
}
