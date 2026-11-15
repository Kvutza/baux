/**
 * Same-origin Cross-Origin Isolation Service Worker for static BAUX bundles.
 *
 * Loaded only when `self.crossOriginIsolated === false` on HTTPS / localhost
 * static hosting where custom server COOP/COEP headers cannot be configured.
 * Permitted under strict CSP (`worker-src 'self'`).
 */

"use strict";

self.addEventListener("install", (event) => {
  event.waitUntil(self.skipWaiting());
});

self.addEventListener("activate", (event) => {
  event.waitUntil(self.clients.claim());
});

self.addEventListener("message", (event) => {
  const data = event && event.data;
  if (!data || typeof data !== "object") {
    return;
  }
  if (data.type === "coi-deregister") {
    event.waitUntil(
      self.registration.unregister().then(() => self.clients.matchAll())
    );
  }
});

async function handleIsolatedFetch(request) {
  const response = await fetch(request);
  if (!response || response.status === 0 || response.type === "opaque") {
    return response;
  }

  const headers = new Headers(response.headers);
  headers.set("Cross-Origin-Opener-Policy", "same-origin");
  headers.set("Cross-Origin-Embedder-Policy", "require-corp");
  if (!headers.has("Cross-Origin-Resource-Policy")) {
    headers.set("Cross-Origin-Resource-Policy", "same-origin");
  }

  return new Response(response.body, {
    status: response.status,
    statusText: response.statusText,
    headers,
  });
}

self.addEventListener("fetch", (event) => {
  const req = event.request;
  if (!req) {
    return;
  }

  // Chromium spec requirement: only-if-cached cross-origin requests throw on fetch(req).
  if (req.cache === "only-if-cached" && req.mode !== "same-origin") {
    return;
  }

  let reqUrl;
  try {
    reqUrl = new URL(req.url);
  } catch (_) {
    return;
  }

  if (reqUrl.origin !== self.location.origin) {
    return;
  }

  if (req.method !== "GET" && req.method !== "HEAD") {
    return;
  }

  // Never intercept synthetic local API endpoints if hitting a live dev server
  // without COOP/COEP, though live `baux pages serve` emits native headers.
  if (reqUrl.pathname.startsWith("/api/")) {
    return;
  }

  event.respondWith(handleIsolatedFetch(req));
});
