# ML Integration Guide — Victoria Line Motion Lab

This guide shows how to integrate the three new ML modules into the app to add platform-prediction capability.

## Module Architecture

```
features.js       → Extract features from raw motion data
    ↓
training-set.js   → Store labeled training examples persistently
    ↓
classifier.js     → k-NN prediction on new data
```

## Modules Overview

### 1. `features.js` — Feature Extraction

Converts raw motion data (acceleration, rotation) into ML-ready feature vectors.

**Key functions:**

```javascript
import { extractFeatures, extractFeaturesWindowed, euclideanDistance } from "./features.js";

// Extract features from a complete recording
const features = extractFeatures(motionData, sampleRateHz = 60);
// Returns: { accel_rms_x, accel_rms_y, ..., spectral_entropy, ... } (25 features)

// Extract features from overlapping windows (for streaming prediction)
const windowedFeatures = extractFeaturesWindowed(
  motionData,
  windowLengthMs = 5000,
  strideMs = 2500,
  sampleRateHz = 60
);
// Returns: [ { ...features, windowEndIndex, windowEndTime }, ... ]

// Distance metric for k-NN
const dist = euclideanDistance(features1, features2);
```

**Features included:**

- **Acceleration**: RMS, peak, magnitude (per-axis and total)
- **Rotation**: RMS, peak per axis (alpha/beta/gamma)
- **Jerk**: rate of acceleration change (sensitive to turns)
- **Skewness**: directional bias (left turn vs. right turn)
- **Energy**: integral of squared acceleration
- **FFT**: 8 frequency-domain bins (resonances)
- **Spectral entropy**: signal complexity

**Why these features?**

A turn at Brixton junction produces a distinctive vibration signature:
- **Lateral acceleration spike** (Y-axis bias for left/right turn)
- **Yaw rotation peak** (alpha increases sharply)
- **Jerk pulse** (rapid direction change)

These features capture that signature across multiple representations, so the classifier can find the pattern even with recording-to-recording variation.

### 2. `training-set.js` — Training Data Manager

Stores labeled motion recordings and manages persistence.

**Key API:**

```javascript
import { TrainingSet, generateRecordingId } from "./training-set.js";

const trainSet = new TrainingSet("victoria-line-training-set");
await trainSet.load(); // Load from localStorage or IndexedDB

// Add a labeled example
const recordingId = generateRecordingId();
trainSet.add(
  motionData,
  "left",  // or "right"
  {
    recordingId,
    timestamp: Date.now(),
    confidence: null,  // for user feedback later
    notes: "Arrived left platform, live video confirmed"
  }
);

await trainSet.save(); // Persist to storage

// Query
console.log(trainSet.count());           // total examples
console.log(trainSet.countByLabel());    // {left: 25, right:30}
console.log(trainSet.getByLabel("left")); // all left-platform examples

// Export (for backup, version control, sharing)
const json = trainSet.toJSON();
localStorage.setItem("backup", JSON.stringify(json));

// Stats
const stats = trainSet.getStats();
console.log(stats.isReadyForTraining); // true if >= 5 of each class
```

**Storage:**

- Small sets (< 500 KB): localStorage (synchronous, simpler)
- Large sets: IndexedDB (browser's dedicated DB, larger quota)

### 3. `classifier.js` — k-NN Classifier

Makes predictions based on a training set.

**Key API:**

```javascript
import { KNNClassifier, evaluateCrossValidation } from "./classifier.js";

// Create and train a classifier
const clf = new KNNClassifier(k = 5);

// Add examples (from training set)
for (const ex of trainSet.examples) {
  clf.addTrainingExample(ex.features, ex.label, {
    recordingId: ex.recordingId,
    timestamp: ex.timestamp
  });
}

// Predict on new data
const features = extractFeatures(newMotionData, 60);
const prediction = clf.predict(features);
// Returns:
// {
//   label: "left",
//   confidence: 0.87,  // 4 out of 5 neighbors voted left
//   neighbors: [ ... ],  // the 5 nearest examples for debugging
//   rawVotes: { left: 4, right: 1 }
// }

// Evaluate on a held-out test set
const metrics = evaluateCrossValidation(
  trainSet.examples.map(ex => ({ features: ex.features, label: ex.label })),
  k = 5,
  testFraction = 0.2
);
// Returns:
// {
//   accuracy: 0.92,
//   leftPrecision: 0.88,
//   leftRecall: 0.95,
//   rightPrecision: 0.95,
//   rightRecall: 0.88,
//   ...
// }
```

## Integration into app.js

### Phase 1: Post-Hoc Analysis (Easier, validate the approach first)

After user records a trip and clicks "Save", add a "Predict Platform" button:

```javascript
// In app.js, after state.recording is false:

import { extractFeatures, extractFeaturesWindowed } from "./features.js";
import { TrainingSet } from "./training-set.js";
import { KNNClassifier } from "./classifier.js";

// UI: add a button to the controls-card
const predictBtn = document.createElement("button");
predictBtn.textContent = "Analyze & Predict Platform";
predictBtn.className = "btn btn-outline";
predictBtn.disabled = true; // enabled when we have training data
// ... add to DOM

// Handler
predictBtn.addEventListener("click", async () => {
  try {
    predictBtn.disabled = true;

    // Load training data
    const trainSet = new TrainingSet();
    await trainSet.load();
    
    if (trainSet.count() < 10) {
      alert(`Need at least 10 labeled examples. Currently have ${trainSet.count()}.`);
      predictBtn.disabled = false;
      return;
    }

    // Extract features from the most recent recording
    const features = extractFeatures(state.data, 60);

    // Build and run classifier
    const clf = new KNNClassifier(k = 5);
    for (const ex of trainSet.examples) {
      clf.addTrainingExample(ex.features, ex.label, {
        recordingId: ex.recordingId
      });
    }

    const prediction = clf.predict(features);

    // Show result
    const result = document.createElement("div");
    result.className = "card";
    result.innerHTML = `
      <h3>Platform Prediction</h3>
      <p><strong>Platform: ${prediction.label.toUpperCase()}</strong></p>
      <p>Confidence: ${(prediction.confidence * 100).toFixed(0)}%</p>
      <p>${prediction.rawVotes.left}/${prediction.rawVotes.right} neighbors</p>
      <button onclick="this.parentElement.remove()">Close</button>
    `;
    document.querySelector(".app-shell").appendChild(result);

  } catch (err) {
    console.error("[ML] Prediction failed:", err);
    alert(`Prediction failed: ${err.message}`);
  } finally {
    predictBtn.disabled = state.data.length === 0 && trainSet.count() > 0;
  }
});
```

### Phase 2: Real-Time Prediction (Ambitious)

While recording, compute a live prediction every 5 seconds:

```javascript
// In renderFrame(), if recording:

if (state.recording && state.rafId !== null) {
  // Every 5 seconds, make a prediction from the last 10 seconds of data
  if (now - (state.lastPredictionAt ?? 0) > 5000) {
    state.lastPredictionAt = now;
    makeLiveForecasts();
    scheduleRender();
  }
}

async function makeLiveForecasts() {
  // Get last 10 seconds of data
  const windowMs = 10000;
  const latest = state.data[state.data.length - 1]?.time ?? 0;
  const windowStart = latest - windowMs;
  const idx = state.data.findIndex(d => d.time >= windowStart);
  const window = state.data.slice(idx);

  if (window.length < 30) return; // need at least 0.5s of data

  const features = extractFeatures(window, 60);

  // Load classifier (from IndexedDB cache)
  const trainSet = new TrainingSet();
  if (!state.clf) {
    await trainSet.load();
    state.clf = new KNNClassifier(5);
    for (const ex of trainSet.examples) {
      state.clf.addTrainingExample(ex.features, ex.label);
    }
  }

  const prediction = state.clf.predict(features);
  
  // Update UI with live forecast
  updateLiveForecast(prediction);
}

function updateLiveForecast(prediction) {
  const forecastEl = document.querySelector("#liveForecast") || (() => {
    const el = document.createElement("div");
    el.id = "liveForecast";
    el.className = "card";
    document.querySelector(".app-shell").insertBefore(el, document.querySelector(".chart-card"));
    return el;
  })();

  forecastEl.innerHTML = `
    <h3>Live Platform Forecast</h3>
    <p style="font-size: 18px; font-weight: bold; color: ${prediction.label === "left" ? "#ff375f" : "#32d74b"}">
      ${prediction.label.toUpperCase()}
    </p>
    <p>Confidence: ${(prediction.confidence * 100).toFixed(0)}%</p>
  `;
}
```

### Phase 3: Training Data UI

Add a modal dialog for labeling trips:

```javascript
// After recording stops, if we have unsaved data:

if (state.unsaved && state.data.length) {
  showLabelingDialog(state.data); // async
}

async function showLabelingDialog(motionData) {
  const response = await new Promise(resolve => {
    const modal = document.createElement("div");
    modal.className = "modal";
    modal.innerHTML = `
      <div class="modal-content">
        <h2>Label This Trip</h2>
        <p>Which platform did you arrive at?</p>
        <button class="btn btn-primary" onclick="this.dataset.answer='left'">
          ← LEFT
        </button>
        <button class="btn btn-secondary" onclick="this.dataset.answer='right'">
          RIGHT →
        </button>
        <button class="btn btn-outline" onclick="this.dataset.answer='skip'">
          Skip
        </button>
      </div>
    `;
    document.body.appendChild(modal);

    // Button handlers
    modal.querySelectorAll("button").forEach(btn => {
      btn.addEventListener("click", () => {
        resolve(btn.dataset.answer);
        modal.remove();
      });
    });
  });

  if (response === "skip") return;

  // Extract features and store
  const features = extractFeatures(motionData, 60);
  const trainSet = new TrainingSet();
  await trainSet.load();

  trainSet.add(motionData, response, {
    recordingId: generateRecordingId(),
    timestamp: Date.now(),
    notes: `User-labeled ${response} platform`
  });

  await trainSet.save();

  alert(`Saved to training set. Total: ${trainSet.count()}`);
}
```

## Testing Checklist

- [ ] `extractFeatures()` returns 25 finite numeric features
- [ ] `extractFeaturesWindowed()` produces one feature vector per window
- [ ] `TrainingSet.add()` stores examples correctly
- [ ] `TrainingSet.load()` / `save()` work with localStorage
- [ ] `KNNClassifier.predict()` returns correct structure
- [ ] Predictions are consistent (same input → same output)
- [ ] k-NN finds reasonable neighbors (low distances)
- [ ] Cross-validation metrics are in expected ranges (50–95% accuracy with 50+ examples)

## Debugging Tools

```javascript
// In browser console:

// Inspect a feature vector
const { extractFeatures } = await import('./features.js');
const features = extractFeatures(motionData);
console.table(features);

// Check classifier neighbors
const pred = clf.predict(features);
console.table(pred.neighbors.map(n => ({ label: n.label, distance: n.distance })));

// Evaluate on training set
const { evaluateCrossValidation } = await import('./classifier.js');
const metrics = evaluateCrossValidation(trainSet.examples, 5, 0.2);
console.log(metrics);

// Export training set
const json = trainSet.toJSON();
console.log(JSON.stringify(json, null, 2));
```

## Data Collection Strategy

To get a good classifier:

1. **Collect at least 50 examples total** (25+ of each class)
   - Take 10 trips, label each "left" or "right"
   - Repeat for balance

2. **Vary conditions**
   - Different times of day (busy vs. quiet)
   - Different seats (front, middle, back of train)
   - Different phone orientations
   - After maintenance windows (track changes)

3. **Validate predictions**
   - Use cross-validation to estimate real accuracy
   - If < 80%, collect more examples or tune features

4. **Iterate**
   - Add new examples based on failed predictions
   - Re-train classifier after adding batch of examples

## Performance Considerations

- **Feature extraction**: ~50ms per recording (10,000 samples at 60 Hz)
- **k-NN prediction**: O(n×m) where n = training examples, m = features
  - 100 examples, 25 features: ~10ms per prediction
  - 1000 examples: ~100ms per prediction
- **Live prediction**: compute every 5 seconds in background, won't block UI

## Next Steps

1. **Integrate Phase 1** (post-hoc prediction button)
2. **Collect 30–50 labeled examples** from real Victoria Line rides
3. **Evaluate cross-validation** metrics (should see 70–90% accuracy)
4. **Add Phase 2** (live prediction during recording)
5. **Optional**: tune k parameter, add distance weighting, or try other distance metrics

