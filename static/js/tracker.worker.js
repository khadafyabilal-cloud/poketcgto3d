/* tracker.worker.js - runs the OpenCV.js card tracker off the main thread, so camera video and 3D
 * rendering keep their frame rate while OpenCV works.
 *
 *  main -> worker   {type:"init", libs:[opencv.js urls to try], options}
 *                   {type:"frame", id, buffer, w, h, ts, srcW, srcH}   (RGBA pixels at analysis size; buffer is transferred)
 *                   {type:"options", options} {type:"flip", id} {type:"reset"} {type:"dispose"}
 *  worker -> main   {type:"ready", from} | {type:"error", message}
 *                   {type:"result", id, tracks, candidates, mask, w, h, ms, buffer}   (buffer is handed back for reuse)
 */
"use strict";
/* WHAT IS A WEB WORKER? A second JavaScript thread with no access to the page (no DOM, no canvas). The page and the
 * worker talk only through messages: worker.postMessage(data) on one side, onmessage on the other. Big binary buffers
 * can be TRANSFERRED (moved, not copied) - see the second argument of postMessage below - which is why sending a
 * camera frame costs almost nothing.
 * This file is the "server" side of that conversation: it loads OpenCV.js once, then answers each "frame" message
 * with the tracked cards. The message list is in the comment above. */
const v = self.location.search;                                        // ?v=... cache-buster, reused for the scripts
importScripts("card-geometry.js" + v, "card-tracker.js" + v);

let tracker = null;

/** Wait until OpenCV.js is usable. The official builds differ: classic (global `cv` + onRuntimeInitialized),
 *  Promise-style (global `cv` is thenable) or a factory function. All are handled, with a polling fallback.
 *  The ready module is passed around inside an object because the module itself is "thenable": resolving
 *  a Promise with it directly would call its then() again and never finish. */
function whenReady(timeoutMs) {
  return new Promise((resolve, reject) => {
    let settled = false, poll = null, timer = null;
    const finish = (m) => {
      if (settled || !m || !m.Mat) return;
      settled = true; clearInterval(poll); clearTimeout(timer);
      resolve({ cv: m });
    };
    timer = setTimeout(() => { if (!settled) { settled = true; clearInterval(poll); reject(new Error("OpenCV.js did not initialise within " + timeoutMs / 1000 + " s")); } }, timeoutMs);
    let c = self.cv;
    try {
      if (typeof c === "function") c = c();                              // factory build
      if (c && c.Mat) return finish(c);                                  // already initialised
      if (c && typeof c.then === "function") c.then((m) => finish(m && m.Mat ? m : self.cv), () => {});
      if (c && typeof c === "object") c.onRuntimeInitialized = () => finish(c);
    } catch (e) { settled = true; clearTimeout(timer); return reject(e); }
    poll = setInterval(() => finish(self.cv && self.cv.Mat ? self.cv : null), 50);
  });
}

async function loadOpenCv(urls) {
  const errors = [];
  for (const url of urls) {
    try {
      importScripts(url);
      return { cv: (await whenReady(60000)).cv, from: url };
    } catch (e) { errors.push(url + ": " + (e && e.message || e)); }
  }
  throw new Error("Could not load OpenCV.js. Tried " + errors.join(" | "));
}

// The message switch. async because loading OpenCV.js takes seconds; the "init" case awaits it.
self.onmessage = async (e) => {
  const m = e.data;
  try {
    switch (m.type) {
      case "init": {
        const { cv, from } = await loadOpenCv(m.libs);
        tracker = new CardTracker(cv, m.options);
        tracker.setOptions(m.options);
        self.postMessage({ type: "ready", from });
        break;
      }
      case "frame": {
        if (!tracker) { self.postMessage({ type: "result", id: m.id, tracks: [], candidates: 0, buffer: m.buffer }); break; }
        const t0 = performance.now();
        const r = tracker.process(new Uint8ClampedArray(m.buffer), m.w, m.h, m.ts, m.srcW, m.srcH);
        const out = { type: "result", id: m.id, ms: performance.now() - t0, buffer: m.buffer, ...r };
        self.postMessage(out, out.mask ? [m.buffer, out.mask.buffer] : [m.buffer]);
        break;
      }
      case "options": tracker && tracker.setOptions(m.options); break;
      case "flip": tracker && tracker.flip(m.id); break;
      case "reset": tracker && tracker.core.reset(); break;
      case "dispose": tracker && tracker.dispose(); tracker = null; self.close(); break;
    }
  } catch (err) {
    self.postMessage({ type: "error", message: String(err && err.message || err), fatal: m.type === "init" });
    if (m.type === "frame") self.postMessage({ type: "result", id: m.id, tracks: [], candidates: 0, buffer: m.buffer });
  }
};
