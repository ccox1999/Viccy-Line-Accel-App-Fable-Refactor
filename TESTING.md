# Testing Guide — Victoria Line Motion Lab ML Integration

Both Phase 1 and Phase 2 are now integrated and ready to test. Here's how to validate the implementation.

## Quick Start

1. **Deploy to HTTPS** (required for DeviceMotion API):
   ```bash
   # GitHub Pages (easiest)
   git push origin main
   # Then visit: https://<username>.github.io/<repo>/
   
   # Or local testing with HTTPS (use a tool like:)
   # - ngrok: ngrok http 3000
   # - localhost.run: (doesn't require signing up)
   # - or any HTTPS tunnel / local server
   ```

2. **Open on iPhone in Safari** (or any HTTPS-capable phone)
3. **Tap "Add to Home Screen"** to install as PWA (optional but recommended)

## Phase 1: Post-Hoc Prediction Testing

**Goal:** Test the predict button after recording a trip.

### Setup (once)

1. **Tap the "🧪 Test Data" button** in the app's controls section.
   It generates 10 fake training examples (5 left, 5 right) — no console needed.

2. You should see:
   - An alert confirming the examples were created
   - Training status: `Training data: 5L / 5R ✓ Ready to predict`
   - "🏷️ Label & Train" and "🔮 Predict Platform" enabled once a recording exists

### Test Steps

1. **Record a short motion**:
   - Tap "Enable Motion Sensors"
   - Tap "Start Recording"
   - Shake your phone or move it around for ~3 seconds
   - Tap "Stop Recording"

2. **Predict the platform**:
   - Tap "🔮 Predict Platform"
   - You should see a result card pop up with:
     - Platform: LEFT or RIGHT
     - Confidence: 60–100%
     - Neighbor votes: e.g., "5/0"
   - This is Phase 1 working! ✓

3. **Close the result and try again** with different motions to see confidence change

### Validation Checklist

- [ ] "Predict Platform" button is disabled before recording (or if training set is empty)
- [ ] "Predict Platform" button is enabled after recording stops (with training data)
- [ ] Prediction result shows within 1–2 seconds
- [ ] Result card shows platform, confidence, and neighbor breakdown
- [ ] Predictions are consistent (same input → same output)
- [ ] Different motions show different confidence levels

---

## Phase 2: Live Streaming Prediction Testing

**Goal:** Test the live forecast card that updates during recording.

### Test Steps

1. **Start a recording**:
   - Tap "Start Recording"
   - You should see a new "Live Platform Forecast" card appear below the rotation chart
   - It shows:
     - Platform: `—` (dash)
     - Confidence: `—`
     - Note: "Waiting for data…"

2. **Let it collect data**:
   - Keep the phone still for ~8 seconds (gives the app enough samples)
   - Watch the live forecast card

3. **After ~8 seconds**, the card should update to show:
   - Platform: LEFT or RIGHT
   - Confidence: e.g., "Confidence: 87%"
   - Note: e.g., "3/2 neighbors"

4. **Move/shake the phone**:
   - The platform prediction might change
   - Confidence updates every 5 seconds

5. **Stop recording**:
   - Tap "Stop Recording"
   - The live forecast card should disappear
   - You should see the labeling offer

### Validation Checklist

- [ ] Live forecast card appears when recording starts
- [ ] Card disappears when recording stops
- [ ] Platform shows `—` initially, updates after ~8 seconds
- [ ] Confidence percentage updates approximately every 5 seconds
- [ ] Different movements produce different predictions
- [ ] No errors in browser console

---

## Training Data Management Testing

**Goal:** Test the label/train workflow.

### Test Steps

1. **Record a trip** and stop recording

2. **A dialog should appear** asking:
   - "Label this trip to add to your training set?"
   - OK = yes, Cancel = skip

3. **Tap OK**, then select a platform:
   - Modal appears with:
     - "Which platform did you arrive at?"
     - "← LEFT" button
     - "RIGHT →" button
     - "Skip" button

4. **Tap LEFT or RIGHT**:
   - You should see an alert: `✓ Saved to training set!`
   - Shows total count: `Total: 11 examples, Left: 6 | Right: 5`
   - Training status updates below the controls

5. **Record and label a few more** trips to build a balanced dataset (aim for 5+ of each)

### Validation Checklist

- [ ] Labeling dialog appears after recording stops (if training data exists)
- [ ] Buttons: LEFT, RIGHT, Skip all work
- [ ] Alert shows success + updated count
- [ ] Training status text updates
- [ ] Predict button remains enabled after labeling
- [ ] Multiple labels are retained across page reloads (persisted to storage)

---

## Real Victoria Line Testing

**Goal:** Collect real training data and validate predictions on actual Stockwell→Brixton trips.

### Data Collection Plan

1. **Collect 20+ real trips** (10+ each direction):
   - Take 5–10 rides to Brixton, label each as left or right
   - Vary: time of day, seat position, phone placement
   - Record the full trip (Stockwell to Brixton ~10 min)

2. **After each trip**:
   - Labeling dialog appears
   - Select the actual platform you arrived at
   - (You can see physical platform signs or check TfL app)

3. **Once you have 50+ examples**, evaluate:
   ```javascript
   (async () => {
     const { TrainingSet } = await import('./training-set.js');
     const { evaluateCrossValidation } = await import('./classifier.js');
     
     const trainSet = new TrainingSet();
     await trainSet.load();
     
     const data = trainSet.examples.map(ex => ({
       features: ex.features,
       label: ex.label
     }));
     
     const metrics = evaluateCrossValidation(data, k = 5, testFraction = 0.2);
     console.log('Metrics:', metrics);
     console.log('Accuracy:', (metrics.accuracy * 100).toFixed(1) + '%');
   })();
   ```

   Expected: 70–95% accuracy

### Troubleshooting

**Issue: "Predict Platform" button stays disabled**
- Make sure you have at least 5 examples of each platform (left AND right)
- Check browser console for errors
- Try: `trainSet.getStats()` in console to see current counts

**Issue: Predictions always show same answer**
- Might not have enough training variety
- Try collecting examples in different conditions
- Check raw training data: `trainSet.examples[0]` in console

**Issue: Live forecast never updates during recording**
- Needs at least 30 samples (~0.5 seconds of data)
- Updates every 5 seconds
- Check console for errors: `[ML] Live prediction failed`

**Issue: Training data disappears after reload**
- Check DevTools → Application → Local Storage
- Should see key: `victoria-line-training-set`
- Check DevTools → Console for storage errors
- Note: training data is stored per-origin — a different URL (or
  incognito window) starts with an empty set

---

## Console Debugging

Open DevTools (F12) and paste these for debugging:

```javascript
// Check app state
console.log('Training set count:', state.trainSet?.count());
console.log('Training stats:', state.trainSet?.getStats());

// Check feature extraction
const { extractFeatures } = await import('./features.js');
const features = extractFeatures(state.data, 60);
console.log('Features:', features);

// Check classifier
console.log('Current prediction:', state.lastLivePrediction);

// Export training set as JSON (for backup)
const json = state.trainSet.toJSON();
console.log(JSON.stringify(json, null, 2));

// Import training set from JSON
state.trainSet.fromJSON(json);
await state.trainSet.save();
```

---

## Next Steps After Testing

1. **If Phase 1 & 2 work** (predict button responds, live forecast updates):
   - Collect 30–50 real training examples
   - Run cross-validation to estimate accuracy
   - If accuracy > 80%, you have a working classifier!

2. **If you hit bugs**:
   - Check browser console (F12) for error messages
   - Note the exact steps to reproduce
   - See Troubleshooting section above

3. **Once validated on real data**:
   - Consider tuning k parameter (default 5)
   - Explore whether certain features matter more (feature importance)
   - Optionally add a "save recording" button for later analysis

---

## Performance Notes

- **Feature extraction**: ~50ms per 10-second window
- **k-NN prediction**: ~10ms (100 training examples, 25 features)
- **Live prediction**: runs every 5 seconds in background, doesn't block UI
- **Storage**: 100 examples ≈ 2–3 MB (stored in IndexedDB)

---

## Architecture Recap

```
app.js
  ├─ loadMLModules() → dynamically import .js files
  │  ├─ features.js (extractFeatures, extractFeaturesWindowed)
  │  ├─ classifier.js (KNNClassifier, evaluateCrossValidation)
  │  └─ training-set.js (TrainingSet, IndexedDB/localStorage)
  │
  ├─ state.trainSet (TrainingSet instance)
  ├─ state.classifier (KNNClassifier, rebuilt per prediction)
  │
  ├─ Phase 1: predictPlatform() → post-hoc k-NN on full recording
  ├─ Phase 2: makeLivePrediction() → streaming k-NN every 5s
  │
  ├─ Label workflow: showLabelingDialog() → labelAndSaveToTrainingSet()
  └─ Init: load ML modules and training set on startup
```

---

Good luck! Once you've tested on real data, you'll have a working platform predictor. 🚇

