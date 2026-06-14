# Victoria Line Motion Lab

A mobile-first web app for recording your iPhone's motion sensors on the
Victoria line — acceleration (m/s²) and rotation rate (°/s) — and learning to
predict which of Brixton's two platforms a train is heading for. Built for the
tube, but the recorder works anywhere your phone does.

## Features

- Live scrolling charts of acceleration (X/Y/Z) and rotation rate (α/β/γ)
- iOS 13+ motion-permission flow with a clear visual status pill
- Live sample count, sample rate (Hz), and recording duration
- Zero-friction training loop: **record → stop → label → done**, with each
  labelled trip auto-saved to localStorage (features + full raw motion)
- On-device k-NN platform prediction with a live forecast while recording and
  an honest cross-validated accuracy estimate after each label
- Manual JSON backup / restore (save to OneDrive, email, or move to another
  device) as the durable safety net
- Installable PWA with offline support (works with no signal on the tube)
- Safe-area aware layout (notch / Dynamic Island / home indicator),
  landscape support, and battery-friendly sensor + render lifecycle

## Getting started

The DeviceMotion permission API **only works over HTTPS** — opening
`index.html` directly from a file will not work, and neither will plain
HTTP. The easiest free option is GitHub Pages:

1. Push this repository to GitHub.
2. Open **Settings → Pages**.
3. Under *Build and deployment*, choose **Deploy from a branch**, select
   `main` and `/ (root)`, then save.
4. After a minute, your app is live at
   `https://<username>.github.io/<repo-name>/`.
5. Open that URL in Safari on your iPhone. For the full experience, tap
   **Share → Add to Home Screen** to install it as a standalone app.

Any other HTTPS static host (Netlify, Cloudflare Pages, Vercel…) works
just as well.

### Icons to add

The manifest and HTML reference three PNG icons that you need to add to
the repository root (any square logo works):

- `apple-touch-icon.png` — 180×180 (home-screen icon on iOS)
- `icon-192.png` — 192×192
- `icon-512.png` — 512×512

Until they exist, the app still runs fine — installs just get a generic
icon.

## Using the app

1. **Enable Motion Sensors** — triggers the iOS motion-permission prompt
   (required once per site on iOS 13+).
2. **Start Recording** — charts begin scrolling; the status row shows a
   pulsing red dot, live sample count, rate, and duration. With ≥5 labelled
   trips of each platform, a **Live Platform Forecast** card updates every
   ~5 s.
3. **Stop Recording** — a labelling sheet appears: **← LEFT / RIGHT → / Skip**.
4. Pick the platform you actually arrived at. The trip's features **and**
   full raw motion are saved to localStorage automatically, an internal
   auto-backup is written, and an alert shows the new totals plus an
   estimated accuracy (once there are ≥5 of each side).
5. **Clear** — discards an unlabelled session (asks first).

Maintenance actions live under **Data & tools**: **⬇️ Backup Data** /
**⬆️ Restore Data** (JSON import/export, deduped by recording on restore) and
**🧪 Test Data** (seed 10 fake examples to exercise the flow without real
trips). Past trips are listed in the **Recordings History** panel, where each
can be deleted.

The in-memory recording buffer keeps the most recent **20,000 samples**
(~5 minutes at 60 Hz). Older samples are trimmed and the sample count
notes "(oldest trimmed)" when that happens.

## iOS permission troubleshooting

If you tap **Don't Allow** on the motion prompt, iOS **remembers the
denial** and will not ask again on a normal reload. To get re-prompted:

- Fully quit Safari (swipe it away in the app switcher) and reopen the
  page, **or**
- Clear the site's data: *Settings → Safari → Advanced → Website Data*,
  find the site, and delete it.

Also check *Settings → Safari → Motion & Orientation Access* is enabled
(on older iOS versions this global switch must be on).

If the status says "HTTPS required", you're viewing the page over plain
HTTP or from a local file — see *Getting started* above.

## Raw motion sample format

A backup file (**⬇️ Backup Data**) is a training-set JSON whose examples each
carry a `rawMotionData` array — one object per motion sample, in this shape:

```json
[
  {
    "time": 1718102400123,
    "ax": 0.0213,
    "ay": -0.0045,
    "az": 0.1872,
    "rotationAlpha": 1.25,
    "rotationBeta": -0.4,
    "rotationGamma": 0.02,
    "gx": 0.12,
    "gy": -9.78,
    "gz": 0.63
  }
]
```

- `time` — Unix epoch milliseconds
- `ax`, `ay`, `az` — acceleration **excluding gravity**, in m/s²
  (the charts display these divided by 9.81 to show *g*)
- `rotationAlpha/Beta/Gamma` — rotation rate in °/s
- `gx`, `gy`, `gz` — the OS's gravity estimate in device coordinates
  (accelerationIncludingGravity − acceleration), in m/s². Used by the
  ML features to find world-vertical regardless of phone orientation.
  Older recordings without these fields still load; the
  orientation-invariant features just read as zero for them.

Internally the app timestamps samples with the monotonic
`performance.now()` clock (immune to clock changes mid-recording) and
converts back to epoch milliseconds on export, so the file format is
unchanged from earlier versions.

## Platform prediction (ML)

Beyond raw recording, the app can learn to predict which of Brixton's two
platforms a train is heading for, from the vibration signature of the
Stockwell → Brixton junction:

1. Record a trip, then label it **left** or **right** on the sheet that
   appears when you stop. The trip's feature vector and raw motion are
   stored in localStorage.
2. Once you have 5+ labelled examples of each platform, the classifier is
   live: k-nearest neighbours over z-score-normalised,
   class-separation-weighted features. These include the gravity-projected
   **world-frame yaw rate** (the rotation-rate vector projected onto the
   vertical axis the gravity sensor reveals), so the turn-direction signal
   survives any phone orientation.
3. While recording, a **Live Platform Forecast** card re-predicts every
   5 seconds from the last 10 seconds of motion. A borderline call shows
   **"Not sure"** rather than committing to a coin-flip.

The **🧪 Test Data** button seeds 10 fake examples so the flow can be
tested without real trips. See `TESTING.md` and `ML-INTEGRATION-GUIDE.md`.

## Project structure

```
index.html       App shell and markup
style.css        Mobile-first styles (safe areas, touch targets, states)
app.js           Sensor capture, chart rendering, export, ML wiring
features.js      Feature extraction (time/frequency/statistical, ES module)
classifier.js    k-NN classifier with z-score normalisation (ES module)
training-set.js  Labeled training data manager, localStorage (ES module)
manifest.json    PWA install metadata
sw.js            Service worker — offline app-shell cache (optional)
README.md        This file
```

**When you edit any file, bump `CACHE_VERSION` in `sw.js`** (e.g.
`motion-lab-v1` → `motion-lab-v2`) so returning visitors get the new
version instead of the cached one. If you'd rather not deal with
caching at all, simply delete `sw.js` — the app detects its absence
and runs normally, just without offline support.

## Browser support

- iOS / iPadOS 14+ Safari (primary target, including home-screen
  standalone mode)
- Current Chrome, Edge, Firefox on Android and desktop
- Desktop browsers run the UI but have no motion sensors, and the app
  says so instead of failing silently
