/* ============================================================
   Victoria Line Motion Lab — raw motion data store (IndexedDB)

   WHY THIS EXISTS: raw motion data is ~3 MB of JSON per trip, and
   localStorage tops out around 5 MB on iOS Safari. The old design
   kept rawMotionData inline in the localStorage training set, so
   from the SECOND real trip onward every save hit the quota and the
   fallback silently stripped raw data from every stored example —
   which is how 10 of the first 11 real recordings lost their raw
   sensor data permanently. IndexedDB has no such practical limit
   (hundreds of MB on an installed PWA), so raw recordings now live
   here, keyed by recordingId, while the small stuff (features,
   labels, metadata) stays in localStorage.

   All methods resolve rather than reject on storage failure and
   report via their return value (null / false), so a browser with
   IndexedDB unavailable (e.g. some private-browsing modes) degrades
   to the old inline-localStorage behaviour instead of breaking the
   recording flow.
   ============================================================ */

"use strict";

const DB_NAME = "motion-lab-raw";
const DB_VERSION = 1;
const STORE = "rawRecordings";

let _dbPromise = null;

function _openDB() {
  if (_dbPromise) return _dbPromise;
  _dbPromise = new Promise((resolve) => {
    let request;
    try {
      request = indexedDB.open(DB_NAME, DB_VERSION);
    } catch (err) {
      console.warn("[RawStore] IndexedDB unavailable:", err);
      resolve(null);
      return;
    }
    request.onupgradeneeded = () => {
      const db = request.result;
      if (!db.objectStoreNames.contains(STORE)) {
        db.createObjectStore(STORE); // key = recordingId (out-of-line)
      }
    };
    request.onsuccess = () => {
      const db = request.result;
      // If another tab upgrades the schema, close so it isn't blocked.
      db.onversionchange = () => db.close();
      resolve(db);
    };
    request.onerror = () => {
      console.warn("[RawStore] IndexedDB open failed:", request.error);
      resolve(null);
    };
    request.onblocked = () => {
      console.warn("[RawStore] IndexedDB open blocked by another connection");
    };
  });
  return _dbPromise;
}

/** Run one operation in a transaction; resolves null/false on any failure. */
async function _withStore(mode, operation, failValue) {
  const db = await _openDB();
  if (!db) return failValue;
  return new Promise((resolve) => {
    let tx;
    try {
      tx = db.transaction(STORE, mode);
    } catch (err) {
      console.warn("[RawStore] Transaction failed to open:", err);
      resolve(failValue);
      return;
    }
    let result = failValue;
    let request;
    try {
      request = operation(tx.objectStore(STORE));
    } catch (err) {
      console.warn("[RawStore] Operation failed:", err);
      resolve(failValue);
      return;
    }
    request.onsuccess = () => {
      result = request.result;
    };
    tx.oncomplete = () => resolve(result);
    tx.onerror = () => {
      console.warn("[RawStore] Transaction error:", tx.error);
      resolve(failValue);
    };
    tx.onabort = () => {
      console.warn("[RawStore] Transaction aborted:", tx.error);
      resolve(failValue);
    };
  });
}

/**
 * Store a recording's raw motion samples under its recordingId.
 * @returns {Promise<boolean>} true if durably written
 */
export async function putRaw(recordingId, rawMotionData) {
  if (!recordingId || !Array.isArray(rawMotionData) || rawMotionData.length === 0) {
    return false;
  }
  const ok = await _withStore(
    "readwrite",
    (store) => store.put(rawMotionData, recordingId),
    undefined
  );
  return ok !== undefined;
}

/**
 * Fetch a recording's raw motion samples.
 * @returns {Promise<Array|null>} samples, or null if absent/unavailable
 */
export async function getRaw(recordingId) {
  if (!recordingId) return null;
  const result = await _withStore("readonly", (store) => store.get(recordingId), null);
  return Array.isArray(result) ? result : null;
}

/** Delete one recording's raw data (no-op if absent). */
export async function deleteRaw(recordingId) {
  if (!recordingId) return false;
  const ok = await _withStore(
    "readwrite",
    (store) => store.delete(recordingId),
    undefined
  );
  return ok !== undefined;
}

/** All stored recordingIds. */
export async function listRawKeys() {
  const keys = await _withStore("readonly", (store) => store.getAllKeys(), null);
  return Array.isArray(keys) ? keys : [];
}

/** Delete everything (used by TrainingSet.clear). */
export async function clearRaw() {
  const ok = await _withStore("readwrite", (store) => store.clear(), undefined);
  return ok !== undefined;
}

/**
 * Ask the browser to treat this origin's storage as persistent so iOS
 * doesn't evict IndexedDB under storage pressure. Safe to call repeatedly;
 * best-effort (user gesture / installed-PWA status decides the outcome).
 */
export async function requestPersistence() {
  try {
    if (navigator.storage?.persist) {
      const granted = await navigator.storage.persist();
      console.log(`[RawStore] Persistent storage ${granted ? "granted" : "not granted"}`);
      return granted;
    }
  } catch (err) {
    console.warn("[RawStore] Persistence request failed:", err);
  }
  return false;
}
