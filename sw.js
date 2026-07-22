/* ============================================================
   Victoria Line Motion Lab — service worker (optional file)
   ------------------------------------------------------------
   Simple offline app shell. Useful on the tube, where this app
   is most likely to be used and connectivity is worst.

   IMPORTANT: bump CACHE_VERSION whenever you edit any file in
   APP_SHELL, otherwise returning visitors keep the old cached
   copy until the background refresh lands.
   ============================================================ */

"use strict";

const CACHE_VERSION = "motion-lab-v26";

// Everything the app needs to boot with no network at all.
const APP_SHELL = [
  "./",
  "./index.html",
  "./style.css",
  "./app.js",
  "./features.js",
  "./classifier.js",
  "./training-set.js",
  "./raw-store.js",
  "./manifest.json",
  "./icon-192.png",
  "./icon-512.png",
  "./apple-touch-icon.png",
];

// Install: pre-cache the shell, then activate immediately rather than
// waiting for every old tab to close.
self.addEventListener("install", (event) => {
  event.waitUntil(
    caches
      .open(CACHE_VERSION)
      .then((cache) => cache.addAll(APP_SHELL))
      .then(() => self.skipWaiting())
  );
});

// Activate: delete caches from previous versions and take control of
// any pages that are already open.
self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches
      .keys()
      .then((keys) =>
        Promise.all(
          keys
            .filter((key) => key !== CACHE_VERSION)
            .map((key) => caches.delete(key))
        )
      )
      .then(() => self.clients.claim())
  );
});

// Fetch: cache-first for speed/offline, with a background network
// refresh so the cache converges on the latest deploy.
self.addEventListener("fetch", (event) => {
  const { request } = event;

  // Only handle same-origin GETs; let everything else pass through.
  if (request.method !== "GET") return;
  if (new URL(request.url).origin !== self.location.origin) return;

  event.respondWith(
    caches.match(request).then((cached) => {
      // Refresh the cache in the background (ignore failures — offline).
      const refresh = fetch(request)
        .then((response) => {
          if (response && response.ok) {
            const copy = response.clone();
            caches.open(CACHE_VERSION).then((cache) => cache.put(request, copy));
          }
          return response;
        })
        .catch(() => undefined);

      if (cached) return cached;

      // Nothing cached: wait for the network, and for page navigations
      // fall back to the cached shell so the app still opens offline.
      return refresh.then((response) => {
        if (response) return response;
        if (request.mode === "navigate") return caches.match("./index.html");
        return Response.error();
      });
    })
  );
});
