# Victoria Line Motion Lab — Project Context

## Current state (2026-07-22) — MAJOR: anchor was landing on Brixton's pre-platform hold, not the true arrival

At n=9 (6L/3R, `Wednesday_vibrations_2.json` uploaded by the user, not in-repo), investigating the user's "how does it handle mid-journey stops" question found the v5 anchor was systematically wrong: on **5 of 9 real trips**, it locked onto a stop ~75-95s in, followed by **30-70+ seconds of further genuine train motion** that never settled before the recording ended (confirmed: NOT walking, user always stops recording at the true halt). User confirmed the physical mechanism: Brixton, a busy terminus, sometimes makes the train wait for the platform to clear before its final approach — the anchor was catching that wait, not the arrival, meaning the fork window had been sampling the wrong segment of the journey on more than half the "dwell"-type trips.

Tried and REJECTED three universal signal-only fixes after real-data validation (each looked good in isolation, broke on real cruising/tail data): reject-if-any-later-motion (breaks legitimate long walk-off tails), a braking-trend gate (real curves/speed-limits produce equally strong false "braking" slopes 20-55% of the time in known-cruising windows), stationary-threshold recalibration (no single value both catches the true end and avoids flagging real cruising as a stop). **Conclusion: no signal-only heuristic can reliably tell "arrived, imperfectly settled" from "still moving" — this is a hard sensor limitation, not a tuning problem.**

**Shipped fix (FEATURE_VERSION 6, commit 39bf8ff):** scope the assumption to where it's actually true rather than fight the physics. [features.js](features.js) `anchoredForkWindow` — tiers 1/2 now reject a candidate if meaningful motion resumes before the recording ends (quiet_remainder, tolerates a brief trailing grab burst) — correct in both live and saved contexts, a pure improvement. New **tier 4 "assume-complete"**: trusts the recording's own grab-trimmed endpoint when no earlier tier qualifies, but ONLY when the caller passes `recordingComplete=true` (new 4th param on `extractForkFeatures`/threaded through `computeApproachProfile`, which always passes true internally since it's only ever called on finished recordings). [app.js](app.js) wiring: the live forecast (`makeLivePrediction`) leaves it false/default (unaffected, still conservative); `labelAndSaveToTrainingSet`, both `reprocessFeatures` call sites, and `generateFakeTrainingData` all pass true. This architectural split — not a better threshold — is what made the fix work: a live, in-progress recording genuinely cannot know it has less to distinguish a wait from arrival, but a recording the user has already tapped Stop on can lean on that fact.

**Validated:** all 5 previously-mis-anchored real trips now land within 1-2s of the true end (was 48-62s off) — checked through the actual shipped code, not just a Python port. Harness grew to 39 assertions (new section I: tier-4 + reject-early-hold regressions; G6 retargeted to the live-mode-relevant path). **Payoff:** `world_yaw_mean` — the original feature the whole orientation-invariant design was built around, which had shown ~zero separation in every prior round of this project — now shows AUC 0.89 between classes. Still not proven at n=9 (permutation p≈0.22, majority-baseline-tying deployed-pipeline LOOCV since Fisher weighting needs 5/class and right=3) — a large, real effect-size jump, honestly not yet statistically significant. Next right-platform trips are high-value: right is still the bottleneck class, and each new trip both grows n AND is now extracted from the physically-correct window.

---

## Previous state (2026-07-21) — 7-trip analysis: Fisher-weighting threshold raised 3→5

Analysed the current dataset (`Newest vibration data.json`, 7 real trips, 4L/3R, all with full raw data and correctly v5-anchored — 4 dwell, 2 short-stop, 1 end-of-braking; anchor type does NOT correlate with label, ruling out a windowing-artifact confound). Headline finding, established by proper permutation test (re-ranking/re-fitting inside every shuffle, not just the real fold): **the deployed classifier (all 27 orientation-invariant features, Fisher-weighted from 3/class) scored LOOCV 1/7 — worse than 93–97% of random label permutations.** That's not "no signal", it's actively anti-correlated: with only 3–4 examples per class, Fisher's within-class spread estimate is itself too noisy to trust, so it amplifies incidental noise directions instead of the real one. Disabling Fisher weighting at the same n (plain uniform z-scored kNN) reverted to unremarkable chance-level accuracy (2–3/7, p≈0.6) — confirming the weighting mechanism, not the data, was the problem.

**Fix:** [classifier.js](classifier.js) `KNNClassifier._ensureNormCache` — Fisher-weighting activation threshold raised from `>= 3` to `>= 5` per class, matching the bar `selectBestModel`'s CV search already requires elsewhere before trusting anything data-driven. Pure classifier-side change, no feature extraction touched, so FEATURE_VERSION is unaffected — no re-extraction or user action needed, takes effect on the next classifier build. Verified via harness (36 assertions; new H1 confirms Fisher weighting still correctly resolves a borderline case once 5/class is reached — the fix doesn't break the mechanism, just gates it appropriately). Full Python permutation-test scripts from this session are NOT saved in-repo (scratch analysis); rerun similarly against any future export if revisiting.

**Diagnostic-only teaser (not validated, do not deploy on this alone):** among individual features, `approach_yaw_lead` and `approach_yaw_trend` (the S-bend swing-order features) direction roughly matches the physics hypothesis again (right trips lean toward positive lead / near-zero trend, left trips negative lead / positive trend), and single-feature threshold LOOCV hits 5/7 on several candidates — but at n=7 this is exactly the kind of cherry-picked-from-27-features result the permutation testing above shows can't be trusted yet. No feature-set change made on this basis.

**Status:** still below the 5/class bar for the classifier's own CV-search and Fisher weighting to activate meaningfully (4L/3R). Next 1–2 right-platform trips are the most valuable data — right is the bottleneck side.

---

## Previous state (2026-07-19) — v5: arrival anchor no longer needs stillness

Post-fix real trips (2026-07-14/16/17, all with full raw data — the IndexedDB fix works) revealed the user's actual recording habit: the recording is stopped within ~2 s of the train halting, sometimes while it is still crawling, so the v4 anchor's required 8 s stationary dwell NEVER exists and every trip fell to the unanchored trailing window (contaminated by the phone-grab burst). **The anchor must not require stillness at the end — user requirement.**

`anchoredForkWindow` is now three tiers, tried in order (returns `{start, cease, type}`):
1. **"dwell"** — the old ≥8 s stationary stretch, scanned from the end (unchanged, most reliable; catches walk-off recordings like the 07-09 trip).
2. **"short-stop"** — a quiet run ≥1 s (`SHORT_STOP_SEC`) within the last 45 s (`TAIL_SEARCH_SEC`): scanning backwards, the LAST sub-threshold run is the arrival, since walking/handling never produces one; a trailing grab burst is skipped naturally.
3. **"end-of-braking"** — no quiet run at all (stopped while crawling): trim up to 6 trailing seconds whose RMS jumps >1.8× (`HANDLING_TRIM_RATIO`) the **minimum** of the preceding 4 s (min, not mean — the braking ramp inflates a mean and hides the grab spike), then anchor at the trimmed end, gated on the last 2 s being at crawl level (< `NEAR_STOP_RMS` 0.45) so a mid-cruise cut is never mistaken for an arrival.

**FEATURE_VERSION → 5** (window semantics changed for any recording without a long dwell) — on next load/restore every raw-carrying example re-extracts and gets a fresh profile; profiles now record `anchorType`. Validated on all 4 real raw trips (dwell / short-stop ×2 / end-of-braking, arrival points sensible, tails excluded) and in the harness: `__v4-test.html` now 35 assertions (G1–G6 cover the three tail habits, feature agreement between dwell and stop-at-halt tails of the same physics, and the mid-cruise no-anchor gate); `__restore-test.html` 7 assertions proves the real 4-trip restore file re-extracts to v5 with ALL FOUR ANCHORED. Known transient: during live recording, a smooth/slow moment can briefly satisfy tier 3 and anchor the rolling forecast early — harmless (re-evaluated every second), never affects stored training data. sw cache **v24**.

Current dataset: `victoria-training-2026-07-19-3L1R-raw-only.json` (parent folder) = the 4 real full-raw trips (3L/1R). The 28 few-second test blips from 07-09 and the 9 raw-less old trips are excluded. Classifier needs 5+ per side — right-platform trips are the bottleneck (1 raw right total).

---

## Previous state (2026-07-09) — dataset analysis + raw-data fix + feature v4

Analysed the first 11 REAL labelled trips (`victoria-training-2026-07-09-5L6R.json`, 5L/6R, kept outside the repo). Three findings drove this session's changes:

1. **The localStorage quota fallback was destroying raw data.** One trip's `rawMotionData` is ~3 MB of JSON vs Safari's ~5 MB quota, so from the second real trip onward every `TrainingSet.save()` hit quota and the fallback stripped raw data from EVERY stored example. Only the newest example survived with raw data (plus example-1, which lives in `Test data.json`). **Fix:** new `raw-store.js` (IndexedDB, keyed by recordingId); `save()` moves raw there first and localStorage keeps only features/metadata/profile. Exports (`exportAsFile`/`triggerDownload`, now **async**) re-inline raw from IDB so backups stay self-contained v3-format files; `load()` migrates any legacy inline raw to IDB; `remove()`/`clear()` delete from IDB; `reprocessFeatures()` fetches raw from IDB on demand. `navigator.storage.persist()` requested at init.
2. **The stored v3 feature vectors carry NO validated left/right signal.** LOOCV of the deployed pipeline: **3/11** (majority baseline 6/11). Best honest protocol found (roughness features + logistic regression): 7/11, permutation test p≈0.17 — chance. Physics explains it: the Brixton approach crossover is an **S-bend onto a parallel road → ~zero net heading change**, so `world_yaw_mean` over any window averages the fork away by construction (confirmed in both raw recordings: lefts read −0.48 mean, rights −0.88 — same sign, no separation). The discriminating information is the S-curve's TIME ORDER and where the switch-clatter roughness sits — destroyed by every v3 aggregate, and unrecoverable for the 9 raw-less trips.
3. **FEATURE_VERSION 4** adds orientation-invariant approach-shape features over the last 30 s before the terminal stop: per-third clipped world-yaw means (`approach_yaw_early/mid/late/net`), OLS `approach_yaw_trend` + `approach_yaw_swing` + `approach_yaw_lead` (which way it swerved first) over the last 20 s, and per-third vertical-accel RMS (`approach_vert_rms_*` — the switch-clatter hypothesis: rights looked rougher, AUC≈0.7, unproven). Every trip now also stores a compact per-second `approachProfile` (yaw/vertRms/horizRms for the last ≤60 s before rest, ~1 KB) in localStorage — the analysis lifeline that would have made this session's dead end impossible. `computeApproachProfile` is exported by features.js; reprocessFeatures takes it as an optional 4th arg and backfills.
4. **Classifier now trains ONLY on `ORIENTATION_INVARIANT_KEYS`** (exported by features.js; classifier.js imports it and prunes in selectBestModel). Device-frame features are still extracted for debugging but structurally cannot influence predictions — verified by a poisoning test. This is a user HARD REQUIREMENT: the model must never learn carry orientation as a label proxy.

**Verification:** headless-Edge harness (`__v4-test.html`, gitignored, kept in repo folder) — 29/29 assertions: orientation invariance across random device orientations (<1e-6), synthetic S-bend trips classified correctly in unseen orientations, IDB round-trip/migration/export-inlining/reprocess/delete, real-dataset load + reprocess (exactly 1 raw-carrying example upgraded). Run it: `python analyze-server` style — serve the PARENT folder (`python -m http.server` won't take POSTs; use the report-server pattern: page POSTs results to `/report`) and open the page in `msedge --headless=new` (note: `--dump-dom --virtual-time-budget` does NOT wait for dynamic imports/IndexedDB — use the POST-back pattern). sw.js cache → **v23**, `raw-store.js` added to APP_SHELL.

**Honest status:** on the existing 11 trips nothing beats chance (new pipeline LOOCV 4/11 on stored vectors — expected, the vectors are information-free). The v4 features are validated on synthetic physics only. **Next step is DATA:** ~10+ new trips recorded with the fixed app (raw + profile retained). Collection protocol: keep the phone still (pocket/lap) from ~1 min before arrival until the train fully stops — example-1's fork window is contaminated by pre-arrival passenger movement (163 °/s hand jerks vs the ~5 °/s fork signal). Then re-run the offline analysis (see `ANALYSIS-2026-07-09.md` in the parent folder) against the raw data + profiles to find the real discriminator and set the window/features accordingly.

---

## Previous state (2026-07-01) — review pass: import hardening + fixes

Review of the whole app; all fixes verified by a headless-Edge test harness (17 assertions) that replayed the exact Restore-button flow against a REAL v3 export from the live app (`Test data.json`, 1 example, pre-`featureVersion`), so **backward compatibility with the current export format is proven**, not assumed. Changes:

- **`TrainingSet.mergeFrom` assigns FRESH local ids to merged examples** (was: kept imported ids). Every device numbers from `example-1`, so restoring a backup into a non-empty set virtually guaranteed id collisions — and `remove(id)` / the history delete button silently deleted the WRONG recording. `recordingId` (the cross-device dedup key) is preserved; exports that lack one get it backfilled from the new id instead of all deduping against each other as `undefined`.
- **`TrainingSet.fromJSON` validates + normalises labels** (lowercases; skips non-left/right with a warning). One bad label in a hand-edited backup used to make every later `addTrainingExample` throw — killing the live forecast and making the post-label save alert report a false "Failed to save". `selectBestModel` also filters malformed examples as defense-in-depth.
- **`stopRecording` retries `ensureTrainSet()`** before offering the label sheet, so a transient startup failure can no longer silently discard a labelled trip.
- **`state.epochBase` re-anchored at every `startRecording`** — `Date.now()` ticks during iOS page suspension, `performance.now()` doesn't, so the load-time offset drifted by total suspended time and put exported `rawMotionData` epoch times hours behind reality on a long-lived PWA. Relative spacing was always correct.
- **History delete reads `e.currentTarget`** (not `e.target`) so future markup inside the button can't break it.
- **`.gitignore` added** — `Test data.json` (a real 3.9 MB personal sensor export sitting untracked in the folder) and `victoria-training-*.json` can no longer be accidentally committed to the public repo.
- README drift fixed (forecast updates every ~1 s from the fork-anchored window, not "every 5 s from the last 10 s"; model selection documented; icons section updated — they exist).
- Service worker cache → **v22**.

Verification harness (not committed): a temp `__import-test.html` module page served by `python -m http.server`, run via `msedge --headless=new --dump-dom --virtual-time-budget` through `cmd` (PowerShell swallows Edge's stdout). Covers: restore into empty set; restore into colliding set (unique ids, preserved recordingId/label/9393-sample raw data); `reprocessFeatures` upgrade to FEATURE_VERSION 3 (45 finite features); re-import dedup; label normalisation; `selectBestModel`+`predictProbability` surviving junk labels; v3 export shape round-trip; ~4 MB localStorage save/load.

## Previous state (2026-06-21)

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
- Service worker cache (v22 as of 2026-07-01): offline support via app-shell pattern, now incl. icons. (Bump on every edit.)
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
