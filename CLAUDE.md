# Victoria Line Motion Lab — Project Context

## Current state (2026-06-14)

**What it does:** A PWA for recording iPhone motion data (acceleration + rotation + gravity) on the Victoria Line, classifying which platform the train is heading for (left or right fork at Brixton) using k-NN with z-score normalization and Fisher-weighted features. Full inertial navigation (gyro integration + world-frame trajectory reconstruction) makes predictions robust to phone orientation and placement.

**Data persistence:** All training data (features + full raw motion data) auto-persists to localStorage when you label a trip. Auto-backup to internal localStorage as fallback. Manual backup/restore via JSON download/import.

**UI (streamlined):** Record → Stop → Auto-prompt to label → Done. Data visible in collapsible history panel with delete per-recording. Test Data button for development. Backup/Restore buttons for manual export/import.

**Key implementation details:**
- `features.js`: ~25 features. The orientation-invariant left/right signal is `world_yaw_mean/peak` + `vert_accel_rms`/`horiz_accel_rms`, computed by projecting each sample onto the per-sample gravity vector (no integration → no drift). Plus device-frame time/frequency/statistical features (RMS, peak, skew, jerk, FFT bins, spectral entropy) that Fisher weighting down-weights when phone orientation varies between trips.
- **REMOVED (do not re-add):** a full "inertial navigation / trajectory reconstruction" block (gyro→rotation-matrix integration + double-integrated position, with traj_curvature/range and world-frame jerk features). Its matrix update was mathematically wrong (corrupted R after step 1), and even corrected, doubly-integrated accelerometer position is a drift-dominated random walk — those features were noise. The gravity-projection features above are the correct way to get orientation invariance.
- `classifier.js`: k-NN (k=5) with z-score normalization (stdDev floored at 0.01) + Fisher-style class-separation weighting (only engaged when ≥3 examples per class, else uniform). `predict()` takes an optional `minConfidence`; the live forecast uses 0.6 so a 3/2 split shows "Not sure" instead of a coin-flip.
- Accuracy is estimated honestly via repeated (15×) held-out cross-validation after each label, shown in the save alert once there are ≥5 examples per class.
- `training-set.js`: v3 format stores features + raw motion data + metadata per example.
- Service worker cache (v12): offline support via app-shell pattern, now incl. icons. (Bump on every edit.)
- PWA-ready: installable to home screen, better storage persistence than bookmarked URL.

**ML caveats / open questions:**
- The underlying physical signal (which Brixton fork the train takes) is small and brief — a gentle low-speed switch near the terminus. `world_yaw` is the most likely discriminator. Real-world accuracy is unproven; the CV estimate after labeling is the source of truth.
- Device-frame features only help if the phone is held consistently across trips; Fisher weighting is what's supposed to suppress them otherwise, but on tiny datasets that weighting is itself noisy. Fewer/better features may beat the current ~25-dim set — worth revisiting once real data exists.

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
