# Testing Guide — Victoria Line Motion Lab

How to validate the app end to end. The flow is intentionally tiny:
**record → stop → label → done.** Everything else (saving, prediction,
backup) happens automatically or lives under **Data & tools**.

## 1. Open over HTTPS

DeviceMotion only works on HTTPS. Easiest is GitHub Pages:

```bash
git push origin main
# then visit https://<username>.github.io/<repo>/
```

On iPhone, open in Safari or Chrome and tap **Share → Add to Home Screen**
to install it as a PWA (better sensor permissions and far more durable
local storage than a bookmarked tab).

## 2. Seed test data (no real trips needed)

Open **Data & tools → 🧪 Test Data**. It creates 10 fake examples
(5 left, 5 right) whose world-frame turn direction is the only consistent
signal, expressed in a random phone orientation per example — i.e. the same
structure as real trips. Re-tapping replaces the fake examples; real
labeled trips are left untouched.

After tapping you should see the training status line update to
`Training data: 5L / 5R …` and the **Recordings History** panel appear.

## 3. Record and label a trip

1. **Enable Motion Sensors** (one-time iOS permission prompt).
2. **Start Recording** — charts scroll; a **Live Platform Forecast** card
   appears. With ≥5 of each class it updates every ~5 s; a borderline call
   shows **“Not sure”** rather than a coin-flip.
3. **Stop Recording** — a labeling sheet appears: **← LEFT / RIGHT → / Skip**.
4. Pick the platform you actually arrived at. The trip’s features **and**
   full raw motion data are saved to localStorage, an internal auto-backup
   is written, and an alert shows the new totals plus an **estimated
   accuracy** (once you have ≥5 of each side).

That’s the whole loop. Nothing else needs saving.

### Checklist
- [ ] Labeling sheet appears on stop (LEFT / RIGHT / Skip all work)
- [ ] Save alert shows updated counts and, with enough data, an accuracy %
- [ ] Recordings History lists the new trip (label, date/time, duration)
- [ ] Deleting a recording from History updates counts and persists
- [ ] Data survives a full reload (and app close, if installed as a PWA)

## 4. Backup / restore (Data & tools)

- **⬇️ Backup Data** downloads `victoria-training-YYYY-MM-DD-NLxxRyy.json`
  (features + raw motion). Save it to OneDrive or email it to yourself.
- **⬆️ Restore Data** imports such a file and **merges** it in, deduped by
  `recordingId` — importing the same file twice does not duplicate.

## 5. Real Victoria line data

Collect ~15–20 trips each direction, varying time of day, seat, and phone
placement. Label each by the platform you actually arrived at. Watch the
estimated-accuracy figure in the save alert: below 75% it warns you to
collect more varied examples. The cross-validation number is the source of
truth for whether the classifier actually works — the underlying physical
signal (which Brixton fork the train takes) is small, so don’t assume it
works until the number says so.

## Troubleshooting

- **“Not sure” a lot:** confidence is below 60% — collect more varied data.
- **Forecast never updates:** needs ≥5 of each class and ≥~0.5 s of motion;
  check the console for `[ML] Live prediction failed`.
- **Data missing after reload:** check DevTools → Application → Local Storage
  for `victoria-line-training-set`. Note storage is per-origin, so a
  different URL or a private window starts empty. Installing as a PWA makes
  storage much more durable.
- **No accuracy estimate:** you need ≥5 examples of *each* side first.

## Console helpers

The app runs inside an IIFE, so its state is exposed for debugging under the
`motionLab` global:

```javascript
// Counts and stats
motionLab.state.trainSet?.getStats();

// Inspect one stored example (features + raw motion)
motionLab.state.trainSet.examples[0];

// Export the whole training set as JSON
JSON.stringify(motionLab.state.trainSet.toJSON());
```
