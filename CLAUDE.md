# Victoria Line Motion Lab — Project Context

## Current state (2026-06-21)

**What it does:** A PWA for recording iPhone motion data (acceleration + rotation + gravity) on the Victoria Line, classifying which platform the train is heading for (left or right fork at Brixton) using k-NN / logistic regression (auto-selected by CV) with z-score normalization and Fisher-weighted features. Orientation invariance comes from projecting each sample onto the per-sample gravity vector (no integration → no drift) — NOT from trajectory reconstruction (that was tried and removed; see below).

**Data persistence:** All training data (features + full raw motion data) auto-persists to localStorage when you label a trip. Auto-backup to internal localStorage as fallback. Manual backup/restore via JSON download/import.

**UI (streamlined):** Record → Stop → Auto-prompt to label → Done. Data visible in collapsible history panel with delete per-recording. Test Data button for development. Backup/Restore buttons for manual export/import.

**Key implementation details:**
- `features.js`: ~25 features. The orientation-invariant left/right signal is `world_yaw_mean/peak` + `vert_accel_rms`/`horiz_accel_rms`, computed by projecting each sample onto the per-sample gravity vector (no integration → no drift). Plus device-frame time/frequency/statistical features (RMS, peak, skew, jerk, FFT bins, spectral entropy) that Fisher weighting down-weights when phone orientation varies between trips.
- **Fork-localised extraction (`extractForkFeatures`) — the feature window matters more than the feature list.** The fork is a brief (~few-second), gentle (~a few °/s) sustained yaw buried in a recording that can run minutes; averaging `world_yaw_mean` over the whole trip dilutes it ~30× toward zero. Features are extracted from a single 10 s window, located by **ROUTE ANCHORING (v3)**: the journey is a fixed Stockwell→Brixton run that always ends by stopping at the Brixton terminus, and the fork is the last sustained turn on the approach — so the window is the one ENDING where the train comes to rest. The terminus stop is found from the orientation-invariant horizontal-acceleration level (gravity-projected, `< STATIONARY_ACCEL_RMS = 0.2 m/s²` for `≥ MIN_TERMINAL_STOP_SEC = 8 s`), scanning from the end so it locks onto the LAST stop (terminus, not a mid-route hold); `anchoredForkWindow` then walks back to the moment of rest, and requires the pre-stop window to itself contain motion (else it's a stop-after-a-stop, e.g. standing on the platform, and the real approach is earlier). Anchoring by POSITION beats the old amplitude rule because the fork yaw is tiny next to hand jerks and the route's own curve. **Validated on real data:** on the one labelled trip, anchoring moved the window from 1–11 s (boarding fumble, peak 163 °/s hand jerk) to 73–83 s (the actual Brixton approach, peak 32 °/s, train-level). **Fallback (the old v2 selector):** when there is no clean terminal stop (recording cut short, mid-route holds only), pick the 10 s window of largest |mean world-frame yaw|, CLIPPING per-sample yaw to `FORK_YAW_SELECT_CLIP = 30 °/s` so a hand jerk can't outscore the real fork; then whole-recording for sub-window or gravity-less (old) data. **Both** the saved training example **and** the live forecast call `extractForkFeatures`, so their feature distributions match (the synthetic Test Data trips append a stationary tail so they take the same anchored path).
- **Feature versioning + reprocessing.** `features.js` exports `FEATURE_VERSION`; each stored example records the version that produced its `features`. On load (and after Restore), `TrainingSet.reprocessFeatures` re-extracts any stale example from its stored `rawMotionData`, so changing the extractor never leaves a training set mixing two incompatible feature scales. (v3 examples carry full raw data precisely so this is possible.)
- **REMOVED (do not re-add):** a full "inertial navigation / trajectory reconstruction" block (gyro→rotation-matrix integration + double-integrated position, with traj_curvature/range and world-frame jerk features). Its matrix update was mathematically wrong (corrupted R after step 1), and even corrected, doubly-integrated accelerometer position is a drift-dominated random walk — those features were noise. The gravity-projection features above are the correct way to get orientation invariance.
- `classifier.js`: k-NN (k=5) with z-score normalization (stdDev floored at 0.01) + Fisher-style class-separation weighting (only engaged when ≥3 examples per class, else uniform). `predict()` takes an optional `minConfidence`; the live forecast uses 0.6 so a 3/2 split shows "Not sure" instead of a coin-flip.
- Accuracy is estimated honestly via repeated (15×) held-out cross-validation after each label, shown in the save alert once there are ≥5 examples per class.
- `training-set.js`: v3 format stores features + raw motion data + metadata (incl. `featureVersion`) per example; `reprocessFeatures(extractFn, version)` re-extracts stale examples from raw data.
- Service worker cache (v21): offline support via app-shell pattern, now incl. icons. (Bump on every edit.)
- PWA-ready: installable to home screen, better storage persistence than bookmarked URL.

**ML caveats / open questions:**
- The underlying physical signal (which Brixton fork the train takes) is small and brief — a gentle low-speed switch near the terminus. `world_yaw` is the most likely discriminator. Real-world accuracy is unproven; the CV estimate after labeling is the source of truth. `extractForkFeatures` now route-anchors each example onto the approach just before the terminus stop instead of averaging the trip away, which should be the biggest accuracy lever short of more/better data — but with only one labelled trip so far, the left/right SEPARATION (and the decision threshold) cannot be validated until `right` arrivals are recorded. The `STATIONARY_ACCEL_RMS`/`MIN_TERMINAL_STOP_SEC` thresholds are calibrated from that single trip and may need a small retune as real data accumulates.
- Device-frame features only help if the phone is held consistently across trips; Fisher weighting is what's supposed to suppress them otherwise, but on tiny datasets that weighting is itself noisy. Fewer/better features may beat the current ~25-dim set — worth revisiting once real data exists.
- **Sign ambiguity from carry direction** is not yet handled: world-frame yaw about gravity is invariant to which way the phone faces horizontally, so a left fork should read the same sign regardless of facing — but if a future device/axis convention flips the gravity sign, the left/right labels would swap. The classifier learns whatever is consistent within the user's own labelled set, so this only bites if carry/convention changes mid-dataset. Worth a held-out-session check once real data exists.
- **Possible next step:** a matched-filter / shape feature on the (low-passed) yaw-rate transient through the crossover, which is what published IMU fork-detection methods use — likely stronger than the current aggregate mean/peak, but a bigger build. Deferred until real trips confirm the current approach.

---

## Next Steps: Automatic OneDrive Cloud Sync

**Goal:** Automatic backup to OneDrive folder, but ONLY when app detects it's running on the user's phone (device fingerprinting). If opened on someone else's device, data stays local-only (no OneDrive sync).

**What to implement:**

1. **Device fingerprinting (SHA256-based)**
   - Collect device properties: screen dimensions, CPU cores, GPU info, user agent
   - Hash them with SHA256 (one-way, can't reverse or spoof without having the phone)
   - On first "Link OneDrive": store hash in localStorage
   - Before every sync: verify current device fingerprint matches stored hash
   - If mismatch: refuse OneDrive sync, keep data local

2. **OneDrive OAuth integration**
   - Add "Link OneDrive" button to UI
   - OAuth flow to get access token + refresh token
   - Device fingerprint protects the tokens: only this phone can use them
   - Add "Unlink OneDrive" button to revoke access

3. **Automatic sync on label**
   - After every `labelAndSaveToTrainingSet()` call
   - If device fingerprint matches AND token exists: append to OneDrive file
   - If sync fails (network down, token expired): silently fail, data stays local
   - Next time app syncs successfully, catch up any missed examples

4. **OneDrive folder structure**
   - Create `/Motion Lab Training/` folder on user's OneDrive
   - File: `victoria-training-YYYY-MM-DD.json` (rolling daily file)
   - Append each new example (features + raw motion data + timestamp + label)
   - User can download from OneDrive as backup or restore on another device

**Two architectural approaches:**

**Option A: Vercel backend (RECOMMENDED)**
- User clicks "Link OneDrive" → redirects to Vercel Function
- Function handles OAuth token exchange securely (tokens never in browser)
- Function returns safe session ID to app
- App syncs via Function (tokens stay server-side, safer)
- Cost: ~$0-2/month (Vercel free tier)
- Security: High (tokens hidden)

**Option B: PKCE (browser-based, no backend)**
- User clicks "Link OneDrive" → browser OAuth flow with PKCE
- App gets token directly, stores in localStorage
- App syncs directly to OneDrive using token
- Cost: $0
- Security: Medium (tokens in browser, but device fingerprinting prevents misuse)

**Status:** Not started. Awaiting decision on Option A vs B before implementation.

**Files that will change:**
- `app.js`: add device fingerprinting logic, OneDrive sync on label, link/unlink buttons
- `index.html`: add "Link OneDrive" / "Unlink OneDrive" buttons to controls
- `style.css`: styles for new buttons
- New: `onedrive-sync.js` (OAuth + sync logic)
- New: `device-fingerprint.js` (SHA256 hashing, fingerprint verification)
- (Option A only) New: `vercel/` folder with backend function

---

## Key constraints & notes

- **Device fingerprinting is the security boundary.** Without it, anyone with the phone could sync to OneDrive. With it, even if someone borrows the phone, the app won't sync to the owner's OneDrive (only to local storage).
- **Factory reset changes fingerprint** → requires re-linking OneDrive (acceptable tradeoff)
- **iOS updates might shift fingerprint properties** → rare, but would require re-link
- **Fallback is always local:** if sync fails, data is safe in localStorage; no data loss
- **Manual backup/restore stays:** users can still download JSON and email/share it, independent of cloud sync

---

## How to pick up this work

1. Decide: Option A (Vercel backend) or Option B (PKCE, pure browser)?
2. If Option A: set up a Vercel project and create the backend function skeleton
3. If Option B: use the PKCE library (e.g., `pkce` npm package)
4. Implement device fingerprinting first (simplest, no API deps)
5. Wire in OAuth flow
6. Test on two devices: fingerprinting should block sync on the second one
7. Verify manual backup/restore still works as fallback

---

## Handoff notes for next session

- All current tests should pass; no breaking changes planned
- The user installed the app as a PWA (home screen), so localStorage is very persistent already
- User explicitly wants device fingerprinting for security (no auto-sync to others' OneDrive)
- User doesn't want to run their own server, so either Vercel (minimal cost) or pure PKCE (no cost)
- Backup/restore buttons are the safety net; cloud sync is convenience, not the only persistence layer
