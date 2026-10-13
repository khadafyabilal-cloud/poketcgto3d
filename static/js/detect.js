/* detect.js - the camera page: AR "Detect" mode, and (body[data-mode=battle]) the mock Pokémon battle on top of it.
 *
 * READING GUIDE (top to bottom; each banner comment line is one stage)
 *   settings -> helpers -> state -> camera -> tracker worker -> 3-D scene -> results from tracker -> identification
 *   (server) -> per-frame drawing -> UI -> start/stop.
 * Three things run at once, on purpose:
 *   1. the CAMERA VIDEO (a <video> element the browser plays by itself),
 *   2. the TRACKER (a Web Worker running OpenCV.js at ~15-30 fps on a small copy of the frame),
 *   3. the 3-D RENDERER (three.js, one draw per screen refresh via renderer.setAnimationLoop(frame)).
 * They only meet in memory: the worker posts "tracks" (corner positions + a 4x4 pose matrix per card), onResult()
 * stores them, and frame() eases the 3-D anchors toward the newest pose every refresh.
 * Vocabulary: "anchor" = a three.js Group placed on a card; "holder" = a child group that is turned to face the
 * opponent; "entry" = everything we know about one tracked card (anchor, label, model, scan state).
 *
 *   camera (AR.js ArToolkitSource, or plain getUserMedia if AR.js can't load)
 *     └─ every ~30 ms a small copy of the frame goes to tracker.worker.js (OpenCV.js finds the yellow or gray card border)
 *   worker ──► card corners + 3D pose per card (card-geometry.js maths)
 *     └─ three.js draws one anchor per card on a transparent canvas laid exactly over the video
 *   a card that stays put for a few frames is SNAPSHOT by the browser itself (snapshot.js cuts the card out of the frame along
 *     its outer border and straightens it) and only that small upright picture goes to /api/battle/scan; the agent
 *     identifies it and the 3D model URL comes back. A card seen before is answered from memory, with no request at all.
 *
 * AR ON / OFF (detect mode): the top-bar switch turns the live camera + tracker (OpenCV worker) + AR drawing on or off. With AR off
 * nothing runs on the camera: the same three.js renderer shows the model in a 3D viewer in the page, next to the detection results
 * (teach.js) - and photos can be taken / uploaded instead. Correcting a result switches AR off by itself, to save the battery.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";
import { Battle, makeFighter, barState, cardKey, colorOf } from "/static/js/battle-engine.js";   // the battle rules (no 3D, no DOM)

const $ = (id) => document.getElementById(id);
const MODE = document.body.dataset.mode === "battle" ? "battle" : "detect";   // same camera page, two behaviours
const CFG = window.PC3D, VER = "?v=" + CFG.v;
const WORK_W = 480;                       // the tracker looks at the video at this width (speed)
const MAX_BACKING = 1024;                 // longest side of the 3D / overlay canvases (CSS scales them up)
const SCAN_TRIES = 3;

/* ------------------------------------------------------------------ settings */
const DEFAULTS = { sens: 0.5, fov: 63, debug: false };
const S = { ...DEFAULTS };
try { Object.assign(S, JSON.parse(localStorage.getItem("pc3d.battle") || "{}")); } catch (e) { /* private mode: defaults */ }
const saveSettings = () => { try { localStorage.setItem("pc3d.battle", JSON.stringify(S)); } catch (e) { /* ignore */ } };

/* ------------------------------------------------------------------ helpers */
function loadScript(urls, timeoutMs = 20000) {                          // first URL that loads wins
  const list = [].concat(urls);
  return new Promise((resolve, reject) => {
    const next = (i) => {
      if (i >= list.length) return reject(new Error("could not load " + list[0]));
      const s = document.createElement("script"), t = setTimeout(() => { s.remove(); next(i + 1); }, timeoutMs);
      s.src = list[i];
      s.onload = () => { clearTimeout(t); resolve(list[i]); };
      s.onerror = () => { clearTimeout(t); s.remove(); next(i + 1); };
      document.head.appendChild(s);
    };
    next(0);
  });
}
const absUrl = (u) => new URL(u, location.href).href;
const wrapAngle = (a) => Math.atan2(Math.sin(a), Math.cos(a));
function setStatus(text, kind) {
  if ($("status").textContent !== text) $("status").textContent = text;
  $("dot").className = "dot" + (kind ? " " + kind : "");
}
function showError(msg) { $("err").textContent = msg; $("go").disabled = false; $("goNoAR").disabled = false; $("start").hidden = false; setStatus("Not running", "err"); }

/* ------------------------------------------------------------------ state */
let CG, SNAP, sigCache, panel, video, backend = "", worker, workerReady = false, busy = false, running = false;
let vw = 0, vh = 0, bw = 0, bh = 0, workH = 0, interval = 33, lastSent = 0, lastMs = 0, fps = 0, lastResult = 0;
let renderer, scene, camera, K, selectedId = null, scanning = false, lastFrameTime = performance.now();
let AR_ON = true, started = false, switching = false, lastPhoto = null;   // AR switch state (see setAR)
const entries = new Map();                                              // track id -> everything we know about that card
const work = document.createElement("canvas"), wctx = work.getContext("2d", { willReadFrequently: true });
const fx = $("fx"), fctx = fx.getContext("2d");
const maskCanvas = document.createElement("canvas");
const capCanvas = document.createElement("canvas"), capCtx = capCanvas.getContext("2d", { willReadFrequently: true }), outCanvas = document.createElement("canvas");
let idleTicks = 0, lastRender = 0, fxDirty = false, blank = false;     // efficiency bookkeeping (see pump / frame)

/* ------------------------------------------------------------------ camera: AR.js first
 * AR.js's ArToolkitSource opens the camera (rear-facing on phones) and creates the <video> element. We use AR.js only
 * for that job; detection is done by our own OpenCV tracker because AR.js's built-in markers (Hiro/barcode) cannot
 * recognise a Pokémon card. If AR.js can't load, we fall back to the standard getUserMedia browser API. */
async function openCamera() {
  let arError = null;
  try {
    window.THREE = THREE;                                              // AR.js's three.js build expects a global THREE
    await loadScript(CFG.libs.arjs);
  } catch (e) { arError = e; }
  if (window.THREEx && THREEx.ArToolkitSource && !arError) {
    try {
      const src = new THREEx.ArToolkitSource({ sourceType: "webcam", sourceWidth: 1280, sourceHeight: 720 });
      await new Promise((ok, fail) => { src.init(ok, fail); setTimeout(() => fail(new Error("AR.js camera start timed out")), 15000); });
      video = src.domElement; backend = "AR.js";
    } catch (e) { arError = e; video = null; }
  }
  if (!video) {                                                        // AR.js missing or failed: plain camera
    console.warn("AR.js camera unavailable, using getUserMedia:", arError);
    const stream = await navigator.mediaDevices.getUserMedia({ audio: false, video: { facingMode: { ideal: "environment" }, width: { ideal: 1280 }, height: { ideal: 720 } } });
    video = document.createElement("video");
    video.srcObject = stream; video.muted = true; video.setAttribute("playsinline", ""); video.autoplay = true;
    document.body.prepend(video);
    backend = "getUserMedia";
  }
  video.setAttribute("playsinline", ""); video.muted = true;
  await video.play().catch(() => { /* iOS may need a tap: the Start button was one */ });
  for (let i = 0; i < 100 && !video.videoWidth; i++) await new Promise((r) => setTimeout(r, 50));
  if (!video.videoWidth) throw new Error("The camera started but sent no picture.");
}

/* ------------------------------------------------------------------ tracker worker
 * We never wait for the worker: if it is still busy (`busy`) the new frame is simply skipped, and the interval between
 * frames adapts to how long OpenCV takes (interval = 2 x last duration, clamped 33-100 ms). That keeps the phone cool. */
function startWorker() {
  return new Promise((resolve, reject) => {
    worker = new Worker("/static/js/tracker.worker.js" + VER);
    worker.onerror = (e) => reject(new Error("Tracker worker failed: " + (e.message || "could not start")));
    worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === "ready") { workerReady = true; resolve(m.from); }
      else if (m.type === "error") { console.warn("tracker:", m.message); if (m.fatal) reject(new Error(m.message)); }
      else if (m.type === "result") { busy = false; onResult(m); }
    };
    worker.postMessage({ type: "init", libs: CFG.libs.opencv.map(absUrl), options: trackerOptions() });
  });
}
/* AR STABILITY. A card is "lost" when the tracker has not seen it for lostAfterMs and "forgotten" after forgetAfterMs. The
 * library defaults (0.25 s / 1.5 s) suit a quick demo but a hand or a glare easily hides a card for longer, which used to
 * delete the 3D model (and restart the battle countdown). Battle mode is the most patient. On top of that detect.js keeps
 * the model of a vanished card on screen for HOLD_MS and its entry alive for STICKY_MS so it can be re-adopted. */
const STABLE = MODE === "battle"
  ? { lostAfterMs: 1200, forgetAfterMs: 12000, hold: 3000, sticky: 20000 }
  : { lostAfterMs: 700, forgetAfterMs: 5000, hold: 1500, sticky: 6000 };
const trackerOptions = () => ({ sensitivity: S.sens, hfov: S.fov, debug: S.debug, maxCards: 2, lostAfterMs: STABLE.lostAfterMs, forgetAfterMs: STABLE.forgetAfterMs });
/* Is this card's model on screen right now? True while the card is seen AND for a short grace period after it stops being seen. */
const shown = (e, now) => e.hasPose && !!e.model && (e.visible || now - e.lastVis < STABLE.hold);
const toWorker = (o) => worker && worker.postMessage({ type: "options", options: o });

function sendFrame(now) {
  wctx.drawImage(video, 0, 0, WORK_W, workH);
  const img = wctx.getImageData(0, 0, WORK_W, workH);
  busy = true; lastSent = now;
  worker.postMessage({ type: "frame", id: now, buffer: img.data.buffer, w: WORK_W, h: workH, ts: now, srcW: vw, srcH: vh }, [img.data.buffer]);
}
function pump() {
  if (!running) return;
  const now = performance.now();
  const gap = idleTicks > 25 ? Math.max(interval, 140) : interval;         // nothing in view for a while: look ~7x per second instead of up to 30
  if (!document.hidden && workerReady && !busy && now - lastSent >= gap) sendFrame(now);
  if (video.requestVideoFrameCallback) video.requestVideoFrameCallback(pump); else setTimeout(pump, 16);
}

/* ------------------------------------------------------------------ 3D scene
 * Key trick: the three.js camera's projection matrix is computed from the same pinhole numbers K that the pose maths
 * used (card-geometry.js), so a model placed with the pose matrix lands on the card in the video exactly. */
const loader = new GLTFLoader().setDRACOLoader(new DRACOLoader().setDecoderPath(CFG.draco));
const modelCache = new Map();                                           // url -> {promise, refs}
function getModel(url) {
  let m = modelCache.get(url);
  if (!m) {
    m = { promise: loader.loadAsync(url), refs: 0 };
    modelCache.set(url, m);
    m.promise.catch(() => modelCache.delete(url));
    for (const [u, e] of modelCache) {                                  // keep at most 8 loaded; free the oldest unused one
      if (modelCache.size <= 8) break;
      if (e.refs === 0 && u !== url) { disposeGltf(e); modelCache.delete(u); }
    }
  }
  m.refs++;
  return m;
}
function disposeGltf(e) {
  e.promise.then((g) => g.scene.traverse((o) => {
    o.geometry && o.geometry.dispose();
    [].concat(o.material || []).forEach((mat) => { for (const k in mat) mat[k] && mat[k].isTexture && mat[k].dispose(); mat.dispose(); });
  })).catch(() => {});
}

function initScene() {
  renderer = new THREE.WebGLRenderer({ canvas: $("gl"), alpha: true, antialias: true, powerPreference: "high-performance" });
  renderer.setClearColor(0x000000, 0);
  scene = new THREE.Scene();
  camera = new THREE.PerspectiveCamera();
  camera.matrixAutoUpdate = false;                                      // the projection comes from the pinhole model, never updateProjectionMatrix()
  scene.add(new THREE.HemisphereLight(0xffffff, 0x556070, 1.7));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(0.6, 2, 1.2); scene.add(sun);
}
function applyCameraModel() {
  K = CG.intrinsics(vw, vh, S.fov);
  camera.projectionMatrix.fromArray(CG.projectionThree(K, 0.05, 100));
  camera.projectionMatrixInverse.copy(camera.projectionMatrix).invert();
}
function layout() {                                                     // sizes depend on the real video size (changes on rotate)
  if (!video || !video.videoWidth) return;
  vw = video.videoWidth; vh = video.videoHeight;
  const k = Math.min(1, MAX_BACKING / Math.max(vw, vh));
  bw = Math.round(vw * k); bh = Math.round(vh * k);
  renderer.setPixelRatio(1); renderer.setSize(bw, bh, false);
  fx.width = bw; fx.height = bh;
  workH = Math.max(1, Math.round((WORK_W * vh) / vw));
  work.width = WORK_W; work.height = workH;
  applyCameraModel();
}

function shadowTexture() {
  const c = document.createElement("canvas"); c.width = c.height = 64;
  const g = c.getContext("2d"), r = g.createRadialGradient(32, 32, 2, 32, 32, 32);
  r.addColorStop(0, "rgba(0,0,0,.55)"); r.addColorStop(1, "rgba(0,0,0,0)");
  g.fillStyle = r; g.fillRect(0, 0, 64, 64);
  return new THREE.CanvasTexture(c);
}
const SHADOW = { tex: null };

function makeEntry(id) {
  const anchor = new THREE.Group(), holder = new THREE.Group();
  anchor.add(holder); anchor.visible = false; scene.add(anchor);
  if (!SHADOW.tex) SHADOW.tex = shadowTexture();
  const shadow = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: SHADOW.tex, transparent: true, depthWrite: false }));
  shadow.rotation.x = -Math.PI / 2; shadow.position.y = 0.003; shadow.visible = false; anchor.add(shadow);
  const tag = document.createElement("div"); tag.className = "tag"; tag.hidden = true;
  tag.addEventListener("click", () => { selectedId = e.id; openSheet("sPick"); });         // e.id, because an entry can be re-keyed (adoptLost)
  $("labels").appendChild(tag);
  const e = { id, anchor, holder, shadow, tag, state: "new", tries: 0, nextTry: 0, mixer: null, model: null, ref: null, info: null,
              corners: null, visible: false, lastVis: -1e9, gone: 0, stable: false, hasPose: false, heading: 0, flash: 0, bt: 0, fp: 0.9, mh: 1.2, shadowBase: 1,
              tp: new THREE.Vector3(), tq: new THREE.Quaternion(), tmp: new THREE.Vector3() };
  entries.set(id, e);
  return e;
}
function disposeEntry(e) {
  scene.remove(e.anchor); e.tag.remove(); e.mixer && e.mixer.stopAllAction();
  e.shadow.geometry.dispose(); e.shadow.material.dispose();
  if (e.ref) e.ref.refs = Math.max(0, e.ref.refs - 1);
  entries.delete(e.id);
}

/* A track id we have never seen appeared. If a card that already has a model vanished moments ago, this is almost certainly the
 * same card picked up / uncovered (the tracker gave it a new id), so hand it the old entry: model, stats and battle stay. The
 * closest vanished card (by last known position) wins. The next scan only CONFIRMS it (state "verify"): a different card replaces it. */
function adoptLost(t, seen) {
  const mid = (c) => c && [c.reduce((a, p) => a + p[0], 0) / c.length, c.reduce((a, p) => a + p[1], 0) / c.length];
  const to = mid(t.corners);
  let best = null, bestD = Infinity;
  for (const e of entries.values()) {
    if (seen.has(e.id) || !e.gone || !e.model) continue;
    const c = mid(e.corners), d = c && to ? Math.hypot(c[0] - to[0], c[1] - to[1]) : 0;
    if (d < bestD) { best = e; bestD = d; }
  }
  if (!best) return null;
  entries.delete(best.id); best.id = t.id; entries.set(t.id, best);
  if (best.state === "ready") best.state = "verify";
  best.tries = 0; best.nextTry = 0; best.hasPose = false;                           // snap to the new pose instead of gliding across the screen
  return best;
}

/* Clone a loaded glTF, size it to fit on a card and stand it on y = 0. Returns {root, mixer, footprint}. Shared by the
 * player's Pokémon (placeModel) and the battle opponent (spawnOpponent). */
function buildModel(gltf, pk, fit) {
  const root = SkeletonUtils.clone(gltf.scene);
  root.traverse((o) => { if (o.isSkinnedMesh) o.frustumCulled = false; });          // skinned bounds are unreliable while animating
  let mixer = null;
  if (gltf.animations.length) { mixer = new THREE.AnimationMixer(root); mixer.clipAction(gltf.animations[0]).play(); mixer.update(0); }
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root, true), size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
  const hf = pk && pk.height_m ? Math.min(1.25, Math.max(0.7, Math.pow(pk.height_m / 0.9, 0.25))) : 1;   // bigger Pokémon look a bit bigger
  const foot = fit ? fit.foot : 0.68 * hf, tall = fit ? fit.h : 1.05 * hf;                                // `fit` = the smaller battle size (both must stand on one card)
  const s = Math.min(foot / Math.max(size.x, size.z, 1e-3), tall / Math.max(size.y, 1e-3));         // fits the card, stands on it
  root.scale.setScalar(s);
  root.position.set(-c.x * s, -box.min.y * s, -c.z * s);
  return { root, mixer, footprint: Math.max(size.x, size.z) * s, height: size.y * s };
}
function placeModel(e, gltf, pk) {
  e.holder.clear();
  const m = buildModel(gltf, pk);
  e.mixer = m.mixer; e.holder.add(m.root); e.model = m.root; e.fp = m.footprint; e.mh = m.height;
  e.shadowBase = m.footprint * 1.5; e.shadow.scale.setScalar(e.shadowBase); e.shadow.visible = true;
}

/* ------------------------------------------------------------------ results from the tracker
 * The worker's pose is a noisy 4x4 matrix. We decompose it into position + rotation (Matrix4.decompose) and let frame()
 * glide the anchor toward it (lerp/slerp), a second smoothing stage after the tracker's One Euro filter. */
const m4 = new THREE.Matrix4(), unitScale = new THREE.Vector3();
function onResult(m) {
  const now = performance.now();
  if (lastResult) fps = fps * 0.9 + (1000 / Math.max(1, now - lastResult)) * 0.1;
  lastResult = now;
  if (m.ms) { lastMs = lastMs * 0.8 + m.ms * 0.2; interval = Math.min(100, Math.max(inBattleView ? 66 : 33, 2 * lastMs)); }   // in the battle view the pose filter fills the gaps, so OpenCV runs at most ~15x/s   // keep OpenCV under ~half a core
  idleTicks = m.tracks.some((t) => t.visible) ? 0 : idleTicks + 1;
  const seen = new Set(m.tracks.map((t) => t.id));
  for (const t of m.tracks) {
    const e = entries.get(t.id) || adoptLost(t, seen) || makeEntry(t.id);
    e.gone = 0;
    e.corners = t.corners; e.visible = t.visible; e.stable = t.stable;
    if (t.visible) e.lastVis = now;
    if (t.modelView) {
      m4.fromArray(t.modelView).decompose(e.tp, e.tq, unitScale);
      if (!e.hasPose) { e.anchor.position.copy(e.tp); e.anchor.quaternion.copy(e.tq); e.hasPose = true; }
    }
  }
  for (const e of [...entries.values()]) {
    if (seen.has(e.id)) continue;
    e.visible = false; e.stable = false;                                            // the tracker dropped it: keep the model, just stop treating it as seen
    if (!e.gone) e.gone = now;
    if (now - e.gone > STABLE.sticky) { if (selectedId === e.id) selectedId = null; disposeEntry(e); }
  }
  if (m.mask) drawMask(m);
  maybeScan();
}

/* ------------------------------------------------------------------ identification (server)
 * Once a card has been stable for a few frames we send ONE full-quality JPEG + its corners to /api/battle/scan. The
 * server straightens the card and runs the neural recognizer (agent.run_card). Only one request is in flight at a time;
 * a failure retries after 3, 6, 12 seconds (3 tries), then the user can tap the label to pick by hand. */
function maybeScan() {
  if (scanning) return;
  const now = performance.now();
  const todo = [...entries.values()].filter((e) => e.stable && e.visible && e.state !== "ready" && e.state !== "scanning" && e.tries < SCAN_TRIES && now >= e.nextTry);
  if (todo.length) scan(todo[0]);
}

/* A SNAPSHOT of the card: the browser cuts the card out of the camera frame along the border the tracker found and straightens
 * it (snapshot.js). Returns the upright picture (a canvas), its fingerprint, and two small pictures for the results panel. */
function takeSnapshot(e) {
  const k = Math.min(1, 1280 / Math.max(vw, vh));
  capCanvas.width = Math.round(vw * k); capCanvas.height = Math.round(vh * k);
  capCtx.drawImage(video, 0, 0, capCanvas.width, capCanvas.height);
  const src = capCtx.getImageData(0, 0, capCanvas.width, capCanvas.height), corners = e.corners.map((p) => [p[0] * k, p[1] * k]);
  const w = SNAP.warpCard(src.data, src.width, src.height, corners);
  outCanvas.width = w.w; outCanvas.height = w.h;
  outCanvas.getContext("2d").putImageData(new ImageData(w.data, w.w, w.h), 0, 0);
  return { canvas: outCanvas, sig: SNAP.cardSig(w.data, w.w, w.h), frame: outlineThumb(capCanvas, corners, 360), card: scaledUrl(outCanvas, 220) };
}
function scaledUrl(canvas, width, q = 0.72) {
  const c = document.createElement("canvas"); c.width = width; c.height = Math.round((width * canvas.height) / canvas.width);
  c.getContext("2d").drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", q);
}
/** A small copy of a picture with the detected card border drawn on it (what the camera saw, for the results panel). */
function outlineThumb(source, corners, width) {
  const k = width / source.width, c = document.createElement("canvas"), g = c.getContext("2d");
  c.width = width; c.height = Math.round(source.height * k);
  g.drawImage(source, 0, 0, c.width, c.height);
  if (corners) {
    g.lineWidth = 3; g.strokeStyle = "#ffcb05"; g.lineJoin = "round"; g.beginPath();
    corners.forEach((p, i) => (i ? g.lineTo(p[0] * k, p[1] * k) : g.moveTo(p[0] * k, p[1] * k))); g.closePath(); g.stroke();
  }
  return c.toDataURL("image/jpeg", 0.7);
}

async function scan(e) {
  scanning = true; e.state = "scanning"; e.tries++;
  setTag(e, "Reading card…", "");
  try {
    let snap = null;
    if (SNAP && sigCache) { try { snap = takeSnapshot(e); } catch (err) { console.warn("snapshot failed, sending the whole frame instead:", err); } }
    if (snap) {
      e.thumbs = { frame: snap.frame, card: snap.card };
      const hit = sigCache.find(snap.sig);
      if (hit) {                                                       // this card was read before (even under another track id): no request at all
        if (hit.flip) worker.postMessage({ type: "flip", id: e.id });
        await applyResult(e, hit.info);
        return;
      }
    }
    const fd = new FormData();
    if (snap) {                                                        // the small upright card picture (~50 KB) instead of a whole frame
      fd.append("image", await new Promise((r) => snap.canvas.toBlob(r, "image/jpeg", 0.88)), "card.jpg");
      fd.append("snapshot", "1");
    } else {                                                           // fallback: whole frame + corners, the server cuts the card out
      const k = Math.min(1, 1280 / Math.max(vw, vh)), cap = document.createElement("canvas");
      cap.width = Math.round(vw * k); cap.height = Math.round(vh * k);
      cap.getContext("2d").drawImage(video, 0, 0, cap.width, cap.height);
      fd.append("image", await new Promise((r) => cap.toBlob(r, "image/jpeg", 0.85)), "frame.jpg");
      fd.append("corners", JSON.stringify(e.corners.map((p) => [p[0] * k, p[1] * k])));
    }
    fd.append("sensitivity", String(S.sens));
    const r = await fetch("/api/battle/scan", { method: "POST", body: fd }), d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    if (!entries.has(e.id)) return;                                    // the card left while we waited
    if (snap && (d.model_url || d.energy)) sigCache.add(snap.sig, d, !!d.rotated);
    if (d.rotated) worker.postMessage({ type: "flip", id: e.id });     // the card was upside down relative to the tracker's guess
    await applyResult(e, d);
  } catch (err) {
    console.warn("scan failed:", err);
    failScan(e, err.message || "Couldn't reach the server");
  } finally { scanning = false; }
}

/* The server had no Tesseract program, so it asked the browser to read the card text with tesseract.js (client-ocr.js).
 * While it runs, stats.client_ocr is set and the battle countdown waits; when it is done the real HP + attacks replace
 * the placeholders (and, if the opponent hasn't appeared yet, the HP bar and attack buttons are redrawn). */
async function runClientOcr(e, d) {
  const token = d.stats.client_ocr;
  if (G.entry === e) say("Reading the card text in your browser…");
  try {
    const mod = await import("/static/js/client-ocr.js" + VER);
    d.stats = await mod.readCardInBrowser(token, { loadScript, urls: CFG.libs.tesseract });
  } catch (err) {
    console.warn("browser OCR failed:", err);
    delete d.stats.client_ocr; (d.stats.notes = d.stats.notes || []).push("Browser OCR failed (" + (err.message || err) + ") - using default stats.");
  }
  if (G.info === d && (G.state === "countdown" || G.state === "idle")) {                 // redraw only before the fight has started
    G.player = makeFighter(d); G.notes = (d.stats.notes || []).join(" ");
    $("mName").textContent = G.player.name; setBar("m", G.player); renderMoves(); setMovesEnabled(false);
  }
}

async function applyResult(e, d) {
  if (!d.model_url && d.energy) {                                      // Energy card: no 3D model, but we know its type
    e.info = d; e.state = "ready"; e.flash = performance.now(); refreshPanel(e);
    setTag(e, d.energy.name || "Energy card", `Energy · ${d.energy.type || "any type"}${MODE === "battle" ? " · healing coming later" : ""}`);
    return;
  }
  if (!d.model_url) return failScan(e, d.message || "No model for this card");
  if (e.model && e.info && e.info.model_url === d.model_url) {         // same card shown again (or re-checked): nothing to reload
    e.info = d; e.state = "ready"; refreshPanel(e); return;
  }
  const ref = getModel(d.model_url);
  if (e.ref) e.ref.refs = Math.max(0, e.ref.refs - 1);
  e.ref = ref;
  const gltf = await ref.promise;
  if (!entries.has(e.id)) return;
  placeModel(e, gltf, d.pokemon);
  e.info = d; e.state = "ready"; e.flash = performance.now();
  if (MODE === "battle" && d.stats && d.stats.client_ocr) runClientOcr(e, d);
  const p = d.pokemon || {};
  setTag(e, p.name || (d.card && d.card.name) || "Pokémon", [(p.types || []).join(" / "), d.card && d.card.set].filter(Boolean).join("  ·  "));
  refreshPanel(e);
}
/* The results panel (teach.js) follows the card the player is looking at. Never rebuilt for the same answer, so a half-typed correction survives. */
function refreshPanel(e) {
  if (!panel) return;
  const cur = panel.current();
  if (cur && cur.info === e.info) return;
  if (!cur || selected() === e || !entries.has(cur.entryId)) panel.show({ info: e.info, frame: e.thumbs && e.thumbs.frame, card: e.thumbs && e.thumbs.card, entryId: e.id });
}
function failScan(e, msg) {
  if (e.model && e.info) { e.state = "ready"; return; }                // a card we already know: a failed RE-check must not take its model/battle away
  e.state = "failed"; e.nextTry = performance.now() + 3000 * Math.pow(2, e.tries - 1);       // 3 s, 6 s, 12 s
  setTag(e, e.tries >= SCAN_TRIES ? "Couldn't read this card" : "Looking again…", e.tries >= SCAN_TRIES ? msg + " · tap to pick by hand" : msg, true);
}
function setTag(e, title, sub, bad) {
  e.tag.className = "tag" + (bad ? " bad" : "");
  e.tag.replaceChildren(Object.assign(document.createElement("b"), { textContent: title }));
  if (sub) e.tag.appendChild(Object.assign(document.createElement("span"), { textContent: sub }));
}

/* ------------------------------------------------------------------ per-frame drawing
 * Runs every screen refresh: ease anchors, aim each Pokémon at its opponent, render 3-D, then draw the 2-D overlay
 * (corner brackets, optional debug mask) and position the text labels. Video is shown with CSS object-fit:cover, so
 * coverMap() reproduces that scaling to convert video pixels into screen pixels for the labels. */
function coverMap() { const W = innerWidth, H = innerHeight, s = Math.max(W / vw, H / vh); return { s, ox: (W - vw * s) / 2, oy: (H - vh * s) / 2 }; }

function drawBrackets(e, now) {
  const c = e.corners, k = bw / vw, pts = c.map((p) => [p[0] * k, p[1] * k]);
  const pulse = e.flash ? Math.max(0, 1 - (now - e.flash) / 700) : 0;                // one bright pulse when a card is recognised
  fctx.lineWidth = Math.max(2, bw / 220) * (1 + 1.2 * pulse); fctx.lineCap = "round"; fctx.lineJoin = "round";
  fctx.strokeStyle = e.state === "ready" ? "#ffcb05" : e.state === "failed" ? "#ff8a8a" : "rgba(255,255,255,.85)";
  for (let i = 0; i < 4; i++) {
    const a = pts[i], b = pts[(i + 1) % 4], z = pts[(i + 3) % 4], f = 0.2;
    fctx.beginPath();
    fctx.moveTo(a[0] + (z[0] - a[0]) * f, a[1] + (z[1] - a[1]) * f); fctx.lineTo(a[0], a[1]); fctx.lineTo(a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f);
    fctx.stroke();
  }
  if (S.debug) { fctx.fillStyle = "#0f0"; pts.forEach((p, i) => { fctx.beginPath(); fctx.arc(p[0], p[1], 4, 0, 7); fctx.fill(); fctx.fillText(["TL", "TR", "BR", "BL"][i], p[0] + 6, p[1] - 6); }); }
}
function drawMask(m) {
  maskCanvas.width = m.w; maskCanvas.height = m.h;
  const g = maskCanvas.getContext("2d"), img = g.createImageData(m.w, m.h), d = img.data;
  for (let i = 0; i < m.mask.length; i++) if (m.mask[i]) { d[4 * i] = 255; d[4 * i + 2] = 255; d[4 * i + 3] = 120; }
  g.putImageData(img, 0, 0);
  lastMask = maskCanvas;
}
let lastMask = null;

function frame() {
  const now = performance.now();
  if (now - lastFrameTime < 30) return;                               // ~30 fps is plenty for AR and halves the GPU/CPU work on 60-120 Hz screens
  const dt = Math.min(0.1, (now - lastFrameTime) / 1000), kk = 1 - Math.exp(-dt / 0.04);
  lastFrameTime = now;
  if (video.videoWidth && (video.videoWidth !== vw || video.videoHeight !== vh)) layout();     // phone rotated
  const live = [...entries.values()].filter((e) => shown(e, now));
  for (const e of entries.values()) {
    if (e.hasPose) poseFilter.step(e.anchor.position, e.anchor.quaternion, e.tp, e.tq, dt);   // steady when still, quick when moving (PoseFilter, card-geometry.js)
    battleLayout(e, dt);
    e.anchor.visible = shown(e, now);
    if (e.anchor.visible && e.mixer) e.mixer.update(dt);                                          // animate only what is on screen
  }
  // two cards: each Pokémon turns to face the other one; alone it faces you (the card's bottom edge)
  const duel = live.filter((e) => e.model);
  for (const e of live) {
    let target = 0;
    if (duel.length >= 2 && e.model) {
      const other = duel.find((o) => o !== e);
      e.anchor.updateMatrixWorld(true);
      const local = e.anchor.worldToLocal(other.anchor.getWorldPosition(e.tmp));
      target = Math.atan2(local.x, local.z);
    }
    if (MODE === "battle" && inBattleView && e === G.entry) target = Math.PI;                       // turn to face the opponent on the far half of the card
    e.heading += wrapAngle(target - e.heading) * kk;
    e.holder.rotation.y = e.heading;
  }
  if (MODE === "battle") { gameTick(now); battleEffects(now, dt); if (G.opp && G.opp.mixer && G.entry && G.entry.anchor.visible) G.opp.mixer.update(dt); }
  updateFlashes(now);
  const needDraw = live.length > 0 || (MODE === "battle" && !!G.opp) || flashing.size > 0 || pops.length > 0;
  if (needDraw || !blank) { renderer.render(scene, camera); blank = !needDraw; }          // nothing to draw: the empty canvas is drawn once, then left alone

  if (fxDirty || S.debug) { fctx.clearRect(0, 0, bw, bh); fxDirty = false; }
  if (S.debug && lastMask) { fctx.imageSmoothingEnabled = false; fctx.drawImage(lastMask, 0, 0, bw, bh); fctx.imageSmoothingEnabled = true; }
  const map = coverMap();
  for (const e of entries.values()) {
    const show = e.visible && e.corners;
    e.tag.hidden = !show || !e.tag.firstChild || (inBattleView && e === G.entry);        // the battle view replaces the name tag
    if (!show) continue;
    fxDirty = true; drawBrackets(e, now);
    const top = [(e.corners[0][0] + e.corners[1][0]) / 2, Math.min(e.corners[0][1], e.corners[1][1])];
    e.tag.style.left = top[0] * map.s + map.ox + "px"; e.tag.style.top = top[1] * map.s + map.oy + "px";
  }
  const nVis = live.length, ready = live.filter((e) => e.state === "ready").length;
  if (!nVis) setStatus("Looking for a card with a yellow or gray border");
  else if (scanning) setStatus("Reading the card…", "lock");
  else setStatus(nVis === 1 ? (ready ? "1 card · Pokémon on" : "1 card locked") : `${nVis} cards · ${ready} Pokémon on`, "lock");
  if (S.debug) $("stats").textContent = `camera  ${backend}  ${vw}x${vh}\ntracker ${fps.toFixed(0)} fps  ${lastMs.toFixed(1)} ms  (every ${interval.toFixed(0)} ms)\ncards   ${entries.size}  scanning ${scanning}`;
}

/* ------------------------------------------------------------------ UI */
function openSheet(id) { $("res").hidden = true; for (const x of ["sPick", "sSet"]) $(x).hidden = x !== id; if (id === "sPick") setTimeout(() => $("pickQ").focus(), 50); }
const closeSheets = () => { $("sPick").hidden = true; $("sSet").hidden = true; if (AR_ON) $("res").hidden = true; };
/* "Details" (AR on): the results panel as a bottom sheet - what the agent read, ratings, and the Teach form. With AR off the same panel fills the page. */
function openDetails() { for (const x of ["sPick", "sSet"]) $(x).hidden = true; $("res").hidden = false; }
const selected = () => entries.get(selectedId) || [...entries.values()].find((e) => e.visible) || null;

function bindUi() {
  $("bFlip").onclick = () => { const e = selected(); if (e) worker.postMessage({ type: "flip", id: e.id }); };
  $("bPick").onclick = () => openSheet("sPick");
  $("bRate").onclick = openDetails; $("resClose").onclick = () => { $("res").hidden = true; };
  $("pickX").onclick = closeSheets; $("setX").onclick = closeSheets;
  $("bSet").onclick = () => openSheet("sSet");
  $("bDebug").onclick = () => setDebug(!S.debug);
  $("pickQ").addEventListener("keydown", (ev) => { if (ev.key === "Enter") $("pickGo").click(); });
  $("pickGo").onclick = async () => {
    const q = $("pickQ").value.trim(), e = selected();
    if (!q) return;
    if (!e) { $("pickMsg").textContent = "Show a card to the camera first."; return; }
    $("pickMsg").textContent = "Looking it up…";
    try {
      const d = await (await fetch("/api/pokemon?q=" + encodeURIComponent(q))).json();
      if (!d.model_url) { $("pickMsg").textContent = d.message || d.error || "No model found."; return; }
      if (MODE === "battle" && e.info && e.info.stats) { d.stats = e.info.stats; d.card = e.info.card; }   // keep the card's own HP + attacks
      await applyResult(e, d); closeSheets(); $("pickMsg").textContent = "The model appears on the selected card.";
    } catch (err) { $("pickMsg").textContent = "Couldn't load it: " + err.message; }
  };
  const sens = $("sens"), fov = $("fov");
  const sync = () => { sens.value = S.sens; fov.value = S.fov; $("sensV").textContent = (+S.sens).toFixed(2); $("fovV").textContent = S.fov + "°"; };
  sens.oninput = () => { S.sens = +sens.value; sync(); toWorker({ sensitivity: S.sens }); saveSettings(); };
  fov.oninput = () => { S.fov = +fov.value; sync(); toWorker({ hfov: S.fov }); applyCameraModel(); saveSettings(); };
  $("setReset").onclick = () => { Object.assign(S, DEFAULTS, { debug: S.debug }); sync(); toWorker({ sensitivity: S.sens, hfov: S.fov }); applyCameraModel(); saveSettings(); };
  sync(); setDebug(S.debug, true);
}
function setDebug(on, silent) {
  S.debug = on; $("bDebug").setAttribute("aria-pressed", String(on)); $("stats").hidden = !on;
  if (!on) lastMask = null;
  if (!silent) { toWorker({ debug: on }); saveSettings(); }
}

/* ------------------------------------------------------------------ the mock battle (MODE === "battle")
 * Flow:  card read (HP + attacks come with the scan result)  ->  10 s countdown  ->  battle view: random opponent appears on the
 * far half of the card  ->  player picks attacks, opponent answers  ->  somebody faints  ->  "Battle again".
 * The battle is tied to the CARD (cardKey), not to the tracker's track id. When the card leaves the picture the battle
 * pauses; when the same card comes back (new track id), the old HP, opponent and 3D models are re-attached at once -
 * and the server answers the re-scan from its card cache, so nothing is read or loaded twice.
 * A state machine keeps this readable:  idle -> countdown -> loading -> fighting -> over. */
const COUNTDOWN_S = Number(new URLSearchParams(location.search).get("countdown")) || 10;    // seconds a card must be detected before the battle starts (?countdown=3 shortens it for testing)
/* THE BATTLE VIEW. When the countdown ends the page switches to a different view (body.in-battle: darker edges, name tag and
 * HP pills replaced): both Pokémon are shrunk so they fit side by side on the detected card - yours on the near half, the
 * opponent on the far half, facing each other - and everything else of the battle (HP bars, damage numbers) is drawn in 3D
 * as sprites on the card too, so it follows the card exactly like the Pokémon do. Only the attack buttons and the message
 * line stay as normal page controls (you have to be able to tap them). Units: 1 = card width; the card is 1 x 1.4. Sizes were reduced ~30 % at a user's request. */
const BATTLE = { slot: 0.3, foot: 0.3, h: 0.42, hpY: 0.58, hpW: 0.4 };       // deliberately small: the Pokémon and bars together use well under half of the card's area
let inBattleView = false, poseFilter = null;
const pops = [];                                                                       // floating damage numbers {spr, t0, y0}
function setBattleView(on) {
  if (on === inBattleView) return;
  inBattleView = on; document.body.classList.toggle("in-battle", on); fxDirty = true;
}
/** Per card, per frame: glide the player's Pokémon (and its shadow) between "alone on the card" and "my half of the battle". */
function battleLayout(e, dt) {
  const on = MODE === "battle" && inBattleView && e === G.entry ? 1 : 0;
  if (e.bt === on) return;                                                              // settled: nothing to do (the usual case)
  e.bt += (on - e.bt) * (1 - Math.exp(-dt / 0.25));
  if (Math.abs(on - e.bt) < 0.002) e.bt = on;
  const k = Math.min(1, BATTLE.foot / e.fp, BATTLE.h / e.mh), sc = 1 + (k - 1) * e.bt, z = BATTLE.slot * e.bt;
  e.holder.position.z = z; e.holder.scale.setScalar(sc);
  e.shadow.position.z = z; e.shadow.scale.setScalar(e.shadowBase * sc);
}
/** A name + HP bar drawn on a small canvas and shown as a camera-facing sprite. The canvas is only redrawn when the numbers change. */
function hpSprite() {
  const c = document.createElement("canvas"); c.width = 256; c.height = 76;
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  spr.scale.set(BATTLE.hpW, BATTLE.hpW * 76 / 256, 1); spr.renderOrder = 10;
  return { spr, c, key: "" };
}
function drawHpSprite(h, f) {
  const b = barState(f), key = f.name + "|" + b.text + "|" + b.frac.toFixed(3);
  if (h.key === key) return; h.key = key;
  const g = h.c.getContext("2d"), W = 256, H = 76;
  g.clearRect(0, 0, W, H);
  g.fillStyle = "rgba(15,18,32,.86)"; g.beginPath(); g.roundRect ? g.roundRect(2, 2, W - 4, H - 4, 16) : g.rect(2, 2, W - 4, H - 4); g.fill();
  g.fillStyle = "#f1f3fb"; g.font = "700 27px system-ui,sans-serif"; g.textBaseline = "alphabetic"; g.textAlign = "left";
  g.fillText(f.name.length > 14 ? f.name.slice(0, 13) + "…" : f.name, 14, 31);
  g.fillStyle = "rgba(255,255,255,.18)"; g.fillRect(14, 42, W - 28, 12);
  g.fillStyle = b.color; g.fillRect(14, 42, (W - 28) * b.frac, 12);
  g.fillStyle = "#a4abc9"; g.font = "600 17px ui-monospace,Menlo,monospace"; g.textAlign = "right";
  g.fillText(b.text, W - 14, 70);
  h.spr.material.map.needsUpdate = true;
}
function disposeSprite(h) { if (!h) return; h.spr.removeFromParent(); h.spr.material.map && h.spr.material.map.dispose && h.spr.material.map.dispose(); h.spr.material.dispose && h.spr.material.dispose(); }
/** Floating "-30" over the Pokémon that was hit: also a sprite on the card (side = "me" | "opp"). */
function popDamage(side, text, color) {
  if (!G.entry) return;
  const c = document.createElement("canvas"); c.width = 160; c.height = 80;
  const g = c.getContext("2d"); g.font = "800 56px system-ui,sans-serif"; g.textAlign = "center"; g.textBaseline = "middle";
  g.lineWidth = 8; g.strokeStyle = "rgba(0,0,0,.8)"; g.strokeText(text, 80, 42); g.fillStyle = color; g.fillText(text, 80, 42);
  const spr = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), transparent: true, depthTest: false }));
  spr.scale.set(0.26, 0.13, 1); spr.renderOrder = 11;
  const y0 = BATTLE.hpY + 0.09; spr.position.set(0, y0, side === "opp" ? -BATTLE.slot : BATTLE.slot);
  G.entry.anchor.add(spr); pops.push({ spr, t0: performance.now(), y0 });
}
/** Per frame while the battle page is open: the opponent pops in, damage numbers rise and fade. */
function battleEffects(now, dt) {
  if (G.opp && G.opp.t < 1) { G.opp.t = Math.min(1, G.opp.t + dt / 0.5); const t = G.opp.t - 1; G.opp.group.scale.setScalar(Math.max(0.01, 1 + 2.7 * t * t * t + 1.7 * t * t)); }   // ease-out-back
  for (let i = pops.length - 1; i >= 0; i--) {
    const p = pops[i], k = (now - p.t0) / 1100;
    if (k >= 1) { disposeSprite({ spr: p.spr }); pops.splice(i, 1); continue; }
    p.spr.position.y = p.y0 + 0.2 * k; p.spr.material.opacity = k < 0.6 ? 1 : 1 - (k - 0.6) / 0.4;
  }
}
const G = { state: "idle", key: null, entry: null, info: null, player: null, battle: null, opp: null, oppInfo: null, t0: 0, lostAt: 0, busy: false, seen: new Set() };
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const hasCard = (e) => e.info && e.info.stats && e.info.pokemon && (e.state === "ready" || e.state === "verify");

function say(text) {                                                                   // the message box; only touches the page when the text changed
  const box = $("bMsg");
  box.classList.remove("over");                                                         // (endBattle re-adds it so the "Battle again" button isn't clipped)
  if (box.dataset.t !== (text || "")) { box.dataset.t = text || ""; box.hidden = !text; box.replaceChildren(document.createTextNode(text || "")); }
  return box;
}
function setBar(prefix, f) {
  const b = barState(f);
  $(prefix + "Bar").style.width = (b.frac * 100).toFixed(1) + "%"; $(prefix + "Bar").style.background = b.color; $(prefix + "Hp").textContent = b.text;
  const h = prefix === "m" ? G.hpMe : G.opp && G.opp.hp;
  if (h) drawHpSprite(h, f);                                                            // the same numbers, drawn in AR on the card
}
function renderMoves() {
  const box = $("moves"); box.replaceChildren();
  G.player.moves.forEach((m, i) => {
    const b = document.createElement("button"); b.type = "button"; b.className = "mv"; b.style.setProperty("--tc", colorOf(m.type));
    const dmg = m.damage === null ? "No damage" : `${m.damage}${m.modifier || ""} damage`;
    b.setAttribute("aria-label", `${m.name}, ${m.type} type, ${dmg}`);
    b.append(Object.assign(document.createElement("b"), { textContent: m.name }), Object.assign(document.createElement("span"), { textContent: `${dmg} · ${m.type}` }));
    b.onclick = () => playerMove(i);
    box.appendChild(b);
  });
  box.hidden = false;
  document.documentElement.style.setProperty("--movesH", box.offsetHeight + 8 + "px");
}
function setMovesEnabled(on) { for (const b of $("moves").children) b.disabled = !on; }

function startCountdown(e) {
  G.state = "countdown"; G.entry = e; G.info = e.info; G.key = cardKey(e.info); G.t0 = performance.now(); G.busy = false; G.tickAt = 0; G.away = 0;
  G.player = makeFighter(e.info); G.battle = null;
  $("mName").textContent = G.player.name; setBar("m", G.player); $("fMe").hidden = false; $("fOpp").hidden = true;
  renderMoves(); setMovesEnabled(false);
  const notes = (e.info.stats.notes || []).join(" ");
  G.notes = notes;
}
if (MODE === "battle") {                                                               // wording + links that differ between the two pages
  $("startH").textContent = "Battle";
  $("startP").textContent = "Lay one Pokémon card on a table and point your camera at it. The app reads the card's HP and attacks; after ten seconds the view switches to the battle: your Pokémon and a random opponent stand on the card and you take turns attacking with the card's moves.";
  $("swap").href = "/detect"; $("swap").textContent = "Detect";
}
function removeOpponent() {
  if (G.opp) {
    G.opp.group.removeFromParent(); disposeSprite(G.opp.hp);
    G.opp.group.traverse((o) => { o.geometry && o.geometry.dispose && o.geometry.dispose(); });
    if (G.opp.ref) G.opp.ref.refs = Math.max(0, G.opp.ref.refs - 1);
    G.opp = null;
  }
}
async function spawnOpponent() {
  G.state = "loading"; say("Finding an opponent…"); setBattleView(true);               // the countdown is over: switch to the battle view
  try {
    const dex = G.info.pokemon.id;
    const d = await (await fetch("/api/battle/opponent?exclude=" + dex)).json();
    if (!d.model_url) throw new Error(d.error || d.message || "no opponent");
    const ref = getModel(d.model_url), gltf = await ref.promise;
    if (G.state !== "loading") { ref.refs--; return; }                                   // the card left while we waited
    removeOpponent();
    const m = buildModel(gltf, d.pokemon, { foot: BATTLE.foot, h: BATTLE.h }), group = new THREE.Group();
    group.position.set(0, 0, -BATTLE.slot);                                              // the far half of the card, facing the player's Pokémon (+z)
    group.scale.setScalar(0.01);                                                         // pops in (battleEffects)
    const sh = new THREE.Mesh(new THREE.PlaneGeometry(1, 1), new THREE.MeshBasicMaterial({ map: SHADOW.tex, transparent: true, depthWrite: false }));
    sh.rotation.x = -Math.PI / 2; sh.position.y = 0.003; sh.scale.setScalar(m.footprint * 1.5);
    const hpO = hpSprite(); hpO.spr.position.set(0, BATTLE.hpY, 0); group.add(sh, m.root, hpO.spr);
    G.entry.anchor.add(group);
    if (!G.hpMe) { G.hpMe = hpSprite(); G.hpMe.spr.position.set(0, BATTLE.hpY, BATTLE.slot); }
    G.entry.anchor.add(G.hpMe.spr);
    G.opp = { group, root: m.root, mixer: m.mixer, ref, hp: hpO, t: 0 };
    G.oppInfo = d; G.battle = new Battle(G.player, makeFighter(d));
    $("oName").textContent = G.battle.opponent.name; setBar("o", G.battle.opponent); setBar("m", G.player); $("fOpp").hidden = false;
    G.state = "fighting";
    say(`A wild ${G.battle.opponent.name} appeared! Choose an attack.`);
  } catch (err) {
    console.warn("opponent failed:", err);
    say("Couldn't find an opponent (" + (err.message || err) + "). Trying again…");
    G.state = "countdown"; G.tickAt = 0; G.t0 = performance.now() - (COUNTDOWN_S - 3) * 1000; setBattleView(false);
  }
}
function resetGame(keepMsg) {
  removeOpponent(); disposeSprite(G.hpMe); G.hpMe = null; setBattleView(false);
  for (const p of pops) disposeSprite({ spr: p.spr }); pops.length = 0;
  G.state = "idle"; G.entry = null; G.battle = null; G.player = null; G.key = null; G.busy = false;
  $("fOpp").hidden = true; $("fMe").hidden = true; $("moves").hidden = true; if (!keepMsg) say("");
}

/* Called every frame. Decides what the battle should be doing given which cards the camera currently sees. */
function gameTick(now) {
  const cards = [...entries.values()].filter(hasCard);
  if (G.state === "idle") {
    if (cards.length) startCountdown(cards[0]);
    else if (!$("bMsg").textContent) say("Place a Pokémon card in view to start.");
    return;
  }
  // is "my" card still tracked? if not, try to re-attach it to a card with the same identity
  const alive = G.entry && entries.get(G.entry.id) === G.entry;
  if (!alive) {
    if (!G.lostAt) G.lostAt = now;
    setMovesEnabled(false);                                                             // nobody to fight with: pause
    if (G.state === "fighting" && !G.busy) say("Show your card again to continue.");
    const same = cards.find((c) => cardKey(c.info) === G.key);
    const fresh = [...entries.values()].find((c) => c.visible && c.stable && !c.info && c.state !== "scanning" && c.state !== "verify");
    if (same) rebind(same);
    else if (fresh && G.info && now - G.lostAt < 30000) { fresh.state = "verify"; applyResult(fresh, G.info).then(() => { if (fresh.state === "ready") fresh.state = "verify"; }); rebind(fresh); }   // client-side cache: show it at once, the re-scan only confirms
    else if (cards.length && now - G.lostAt > 8000) { resetGame(); startCountdown(cards[0]); }                                   // a different card, and the old one is gone
    return;
  }
  G.lostAt = 0;
  const here = shown(G.entry, now);                                                     // seen now, or only just lost (grace period)
  if (G.state === "countdown") {
    const dtT = G.tickAt ? now - G.tickAt : 0; G.tickAt = now;
    if (G.info.stats && G.info.stats.client_ocr) { say("Reading the card text in your browser…"); G.t0 += dtT; return; }   // wait for tesseract.js
    if (!here) {                                                                        // PAUSE (not restart): the time already counted stays counted
      G.t0 += dtT; G.away = (G.away || 0) + dtT;
      say(`Card hidden - the countdown is paused. Show it again to continue.`);
      if (G.away > 15000) { resetGame(); }                                              // gone for good: start over when a card appears
      return;
    }
    G.away = 0;
    const left = COUNTDOWN_S - (now - G.t0) / 1000;
    if (left <= 0) { spawnOpponent(); return; }
    say(`${G.player.name} · ${G.player.maxHp} HP · battle starts in ${Math.ceil(left)}…` + (G.notes ? `  (${G.notes})` : ""));
  } else if (G.state === "fighting") {
    setMovesEnabled(here && !G.busy && G.battle.turn === "player");
    if (!here && !G.busy) { say("Show your card again to continue."); G.wasAway = true; }
    else if (here && G.wasAway && !G.busy) { G.wasAway = false; say(`Welcome back! ${G.battle.opponent.name} is waiting.`); }
  }
}
function rebind(e) {
  if (G.opp) e.anchor.add(G.opp.group);                                                 // the opponent and the HP stay; only the card's track changed
  if (G.hpMe) e.anchor.add(G.hpMe.spr);
  G.entry = e; G.lostAt = 0;
  if (G.state === "fighting" && !G.busy) say(`Welcome back! ${G.battle.opponent.name} is waiting.`);
}

/* ---- hit effects ---- */
const flashRec = new WeakMap(), flashing = new Set();
function flashModel(root, hex) {
  if (!root) return;
  let rec = flashRec.get(root);
  if (!rec) {                                                                          // give this model its own materials so a flash can't leak onto a twin
    rec = { mats: [] };
    root.traverse((o) => {
      if (!o.isMesh || !o.material) return;
      const list = [].concat(o.material).map((m) => m.clone());
      o.material = Array.isArray(o.material) ? list : list[0];
      list.forEach((m) => rec.mats.push({ m, color: m.color && m.color.clone(), emissive: m.emissive && m.emissive.clone(), ei: m.emissiveIntensity }));
    });
    flashRec.set(root, rec);
  }
  rec.t0 = performance.now(); rec.col = new THREE.Color(hex); flashing.add(rec);
}
function updateFlashes(now) {
  for (const rec of flashing) {
    const t = (now - rec.t0) / 800, k = t >= 1 ? 0 : t < 0.12 ? t / 0.12 : 1 - (t - 0.12) / 0.88;     // quick rise, slower fade
    for (const x of rec.mats) {
      if (x.m.emissive) { x.m.emissive.copy(k ? rec.col : x.emissive); x.m.emissiveIntensity = k ? 2.4 * k : x.ei; }
      else if (x.m.color) x.m.color.copy(x.color).lerp(rec.col, k);                          // unlit materials: tint instead
    }
    if (k === 0) flashing.delete(rec);
  }
}
function show(r) {
  say(r.message);
  if (!r.nonDamaging) {
    const target = r.side === "player" ? G.opp && G.opp.root : G.entry && G.entry.model;     // the one that got hit lights up in the move's colour
    flashModel(target, r.color); popDamage(r.side === "player" ? "opp" : "me", "-" + r.damage, r.color);
  }
  setBar("o", G.battle.opponent); setBar("m", G.battle.player);
}
async function playerMove(i) {
  if (G.state !== "fighting" || G.busy || G.battle.turn !== "player") return;
  G.busy = true; setMovesEnabled(false);
  const r = G.battle.useMove("player", i);
  show(r); await sleep(1500);
  if (!r.fainted) {
    const o = G.battle.useMove("opponent", G.battle.pickOpponentMove());
    show(o); await sleep(1500);
  }
  G.busy = false;
  if (G.battle.winner) endBattle();
}
function endBattle() {
  G.state = "over"; setMovesEnabled(false);
  const box = say(G.battle.winner === "player" ? `You win! ${G.battle.opponent.name} fainted.` : `${G.player.name} fainted. You lose!`);
  const again = document.createElement("button"); again.type = "button"; again.textContent = "Battle again";
  again.onclick = () => { G.battle.reset(); setBar("m", G.player); G.state = "countdown"; G.t0 = performance.now() - COUNTDOWN_S * 1000 + 100; say(""); };
  box.append(document.createElement("br"), again); box.classList.add("over");
}

/* ------------------------------------------------------------------ AR on / off (detect mode)
 * AR ON  = camera + OpenCV tracker (a Web Worker) + the models drawn on the cards.
 * AR OFF = none of that runs: the camera is released, the worker is terminated (its WebAssembly memory is freed) and the AR
 *          scene is emptied. The same three.js renderer now draws ONE model in a 3D viewer in the page, the results panel
 *          (teach.js) fills the rest of the screen and photos can be taken / uploaded. The results the player was looking at
 *          stay on screen; nothing is lost by switching. Switching AR back on restarts the camera and tracker. */
function stopCamera() {
  running = false;
  const st = video && video.srcObject; st && st.getTracks().forEach((t) => t.stop());
}
function stopAR() {
  stopCamera(); busy = false; workerReady = false;
  if (worker) { worker.terminate(); worker = null; }
  if (video) { video.remove(); video = null; }
  renderer && renderer.setAnimationLoop(null);
  for (const e of [...entries.values()]) disposeEntry(e);              // frees the AR models; the answers live on in the panel + sigCache
  selectedId = null; blank = false; idleTicks = 0;
  fctx.clearRect(0, 0, fx.width, fx.height);
}
async function startAR() {
  CG = CG || (await loadScript("/static/js/card-geometry.js" + VER).then(() => window.CardGeometry));
  const workerUp = startWorker();                                       // OpenCV.js (~10 MB the first time, then cached) loads while the camera starts
  workerUp.catch(() => {});
  try {
    setStatus("Starting camera…");
    await openCamera();
    setStatus("Loading OpenCV…");
    await workerUp;
  } catch (e) { stopAR(); throw e; }
  layout();
  $("dock").hidden = false;
  running = true; lastFrameTime = performance.now();
  renderer.setAnimationLoop(frame);
  pump();
}
const cameraProblem = (e) => e && e.name === "NotAllowedError" ? "Camera access was blocked. Allow the camera for this site in your browser settings, then try again."
  : e && e.name === "NotFoundError" ? "No camera was found on this device."
  : "Couldn't start the camera.\n" + (e && e.message || e);

function paintArSwitch() {
  const b = $("arToggle");
  b.setAttribute("aria-pressed", String(AR_ON)); $("arLabel").textContent = AR_ON ? "AR on" : "AR off";
  document.body.classList.toggle("ar-off", !AR_ON);
  $("viewer").hidden = AR_ON; $("srcbar").hidden = AR_ON;
}
/** Turn AR on or off. `why` = a sentence for the status pill ("AR paused while you correct the card…"). */
async function setAR(on, why) {
  if (MODE === "battle" || switching || !started || on === AR_ON) return;
  switching = true;
  try {
    AR_ON = on; paintArSwitch();
    if (!on) {
      stopAR(); $("dock").hidden = true; $("sPick").hidden = $("sSet").hidden = true; $("res").hidden = false;
      await initViewer(); layoutViewer();
      const cur = panel && panel.current();
      await viewerShow(cur && cur.info);
      setStatus(why || "AR off · take or upload a photo", "");
    } else {
      stopViewerLoop(); $("res").hidden = true;
      try { await startAR(); }
      catch (e) { AR_ON = false; paintArSwitch(); $("res").hidden = false; await initViewer(); layoutViewer(); setStatus(cameraProblem(e).split("\n")[0], "err"); }
    }
  } finally { switching = false; }
}

/* ---- the 3D viewer (AR off) ---- */
let vscene, vcam, vctl, vgroup = null, vmix = null, vref = null, vReady = false, vLast = 0, vSpin = true, vToken = 0;
async function initViewer() {
  if (vReady) return;
  const { OrbitControls } = await import("three/addons/controls/OrbitControls.js");   // only downloaded when the viewer is first needed
  vscene = new THREE.Scene(); vcam = new THREE.PerspectiveCamera(40, 1, 0.1, 100); vcam.position.set(0, 1.1, 3.6);
  vscene.add(new THREE.HemisphereLight(0xffffff, 0x445577, 1.6));
  const sun = new THREE.DirectionalLight(0xffffff, 2.2); sun.position.set(3, 4, 5); vscene.add(sun);
  vctl = new OrbitControls(vcam, renderer.domElement);
  Object.assign(vctl, { enableDamping: true, autoRotate: vSpin, autoRotateSpeed: 2.5, enablePan: false, minDistance: 1.5, maxDistance: 8 });
  vctl.target.set(0, 0.95, 0);
  vReady = true;
  $("vSpin").onclick = () => { vSpin = !vSpin; vctl.autoRotate = vSpin; $("vSpin").setAttribute("aria-pressed", String(vSpin)); };
  $("vReset").onclick = () => { vctl.reset(); vcam.position.set(0, 1.1, 3.6); vctl.target.set(0, 0.95, 0); };
  addEventListener("resize", () => { if (!AR_ON && vReady) layoutViewer(); });
}
function layoutViewer() {                                               // the #gl canvas sits over #viewer (CSS); size the drawing buffer to it
  const r = $("viewer").getBoundingClientRect(), w = Math.max(1, Math.round(r.width)), h = Math.max(1, Math.round(r.height));
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.5));        // phones are 3x: drawing 9x the pixels for a small model is wasteful
  renderer.setSize(w, h, false);
  vcam.aspect = w / h; vcam.updateProjectionMatrix();
}
function clearViewerModel() {
  if (vgroup) { vscene.remove(vgroup); vgroup = null; vmix && vmix.stopAllAction(); vmix = null; }
  if (vref) { vref.refs = Math.max(0, vref.refs - 1); vref = null; }
}
async function viewerShow(info) {
  const token = ++vToken;
  $("viewerEmpty").hidden = !!(info && info.model_url);
  if (!info || !info.model_url) {
    clearViewerModel(); stopViewerLoop(); renderer.clear();
    $("viewerEmpty").textContent = info && info.energy ? "Energy cards have no 3D model." : info && info.message ? info.message : "The 3D model appears here.";
    return;
  }
  try {
    const ref = getModel(info.model_url), gltf = await ref.promise;
    if (token !== vToken || AR_ON) { ref.refs = Math.max(0, ref.refs - 1); return; }   // a newer answer arrived, or AR came back, while it loaded
    clearViewerModel();
    const m = buildModel(gltf, info.pokemon);
    vgroup = new THREE.Group(); vgroup.add(m.root); vgroup.scale.setScalar(1.35);
    vscene.add(vgroup); vmix = m.mixer; vref = ref;
    startViewerLoop();
  } catch (err) { $("viewerEmpty").hidden = false; $("viewerEmpty").textContent = "The 3D model could not be loaded: " + (err.message || err); }
}
function viewerFrame() {
  const now = performance.now();
  if (now - vLast < 30 || !vgroup) return;                              // ~30 fps; browsers pause this loop by themselves in a hidden tab
  const dt = Math.min(0.1, (now - vLast) / 1000); vLast = now;
  vmix && vmix.update(dt); vctl.update();
  renderer.render(vscene, vcam);
}
const startViewerLoop = () => { if (vReady) { vLast = performance.now(); renderer.setAnimationLoop(viewerFrame); } };
const stopViewerLoop = () => renderer && renderer.setAnimationLoop(null);

/* ---- a photo instead of the live camera (AR off): resized in the browser, one request ---- */
async function scanPhoto(file) {
  if (!file || switching) return;
  setStatus("Reading the photo…", "lock"); $("bShoot").disabled = $("bUpload").disabled = true;
  try {
    let bmp;
    try { bmp = await createImageBitmap(file, { imageOrientation: "from-image" }); } catch (e) { bmp = await createImageBitmap(file); }
    const k = Math.min(1, 1600 / Math.max(bmp.width, bmp.height)), cv = document.createElement("canvas");
    cv.width = Math.round(bmp.width * k); cv.height = Math.round(bmp.height * k);
    cv.getContext("2d").drawImage(bmp, 0, 0, cv.width, cv.height); bmp.close && bmp.close();
    const blob = await new Promise((r) => cv.toBlob(r, "image/jpeg", 0.88));
    lastPhoto = { blob, cv };
    await sendPhoto();
  } catch (err) {
    setStatus("Couldn't read that photo: " + (err.message || err), "err");
  } finally { $("bShoot").disabled = $("bUpload").disabled = false; $("photo").value = ""; $("shoot").value = ""; }
}
async function sendPhoto() {
  const fd = new FormData();
  fd.append("image", lastPhoto.blob, "photo.jpg"); fd.append("sensitivity", String(S.sens));       // no corners: the server finds the card's outer border itself
  const r = await fetch("/api/battle/scan", { method: "POST", body: fd }), d = await r.json();
  if (!r.ok) throw new Error(d.error || r.statusText);
  const frame = outlineThumb(lastPhoto.cv, d.scan && d.scan.corners, 360);
  panel.show({ info: d, frame, card: d.crop_jpeg, scan: d.scan, entryId: null });
  await viewerShow(d);
  setStatus(d.model_url ? "AR off · " + ((d.pokemon && d.pokemon.name) || "card") + " found" : d.message || "No Pokémon found", d.model_url ? "lock" : "err");
}

/* ---- hooks for the results panel ---- */
function hookPanel(mod) {
  panel = mod.createPanel($("resMain"), {
    beforeEdit: () => { if (MODE === "detect" && AR_ON) setAR(false, "AR paused while you correct the card - it saves battery. Switch AR on again whenever you like."); },
    onNewResult: async (d, old) => {                                    // the Pokémon changed (a correction or a rating): show the new answer everywhere
      sigCache.replace(old, d);
      const cur = panel.current(), e = [...entries.values()].find((x) => x.info === old);
      panel.show({ ...cur, info: d });
      if (e) await applyResult(e, d);
      if (!AR_ON) await viewerShow(d);
    },
    onRetry: (old) => {                                                 // a bare thumbs-down: read the card again (the server skips the rejected guess)
      sigCache.items = sigCache.items.filter((it) => it.info !== old && it.info !== (panel.current() || {}).info);
      const e = [...entries.values()].find((x) => x.info === (panel.current() || {}).info);
      if (e) { e.state = "new"; e.tries = 0; e.nextTry = 0; }
      else if (!AR_ON && lastPhoto) sendPhoto().catch((err) => setStatus(err.message, "err"));
    },
  });
}

/* ------------------------------------------------------------------ start */
document.addEventListener("visibilitychange", () => {                   // save battery: nothing runs while the tab is hidden
  if (!running) return;
  if (document.hidden) renderer.setAnimationLoop(null);
  else { renderer.setAnimationLoop(frame); lastFrameTime = performance.now(); }
});
addEventListener("pagehide", () => { stopCamera(); worker && worker.postMessage({ type: "dispose" }); });

async function start(withAR = true) {
  $("go").disabled = $("goNoAR").disabled = true; $("err").textContent = "";
  if (withAR && (!window.isSecureContext || !navigator.mediaDevices)) {
    return showError("The camera only works on a secure page.\nOn this computer open http://localhost:5000"+location.pathname+". On a phone, start the server with:\n  python app.py --host 0.0.0.0 --https\nand open the https:// address it prints.\n(You can also use the page without AR.)");
  }
  try {
    setStatus("Loading…");
    CG = await loadScript("/static/js/card-geometry.js" + VER).then(() => window.CardGeometry);
    await loadScript("/static/js/snapshot.js" + VER); SNAP = window.CardSnapshot; sigCache = new SNAP.SigCache();
    poseFilter = new CG.PoseFilter(); initScene(); bindUi();
    hookPanel(await import("/static/js/teach.js" + VER));
    AR_ON = withAR; started = true; paintArSwitch();
    $("arToggle").onclick = () => setAR(!AR_ON);
    $("bShoot").onclick = () => $("shoot").click(); $("bUpload").onclick = () => $("photo").click();
    $("photo").onchange = (ev) => scanPhoto(ev.target.files[0]); $("shoot").onchange = (ev) => scanPhoto(ev.target.files[0]);
    if (withAR) await startAR();
    else { await initViewer(); $("res").hidden = false; layoutViewer(); setStatus("AR off · take or upload a photo", ""); }
  } catch (e) {
    started = false; stopAR(); AR_ON = true; paintArSwitch();
    return showError(cameraProblem(e));
  }
  $("start").hidden = true;
}
$("go").onclick = () => start(true);
$("goNoAR").onclick = () => start(false);
if (MODE === "battle") $("goNoAR").hidden = true;                        // the battle needs the camera
{ const go = new URLSearchParams(location.search).get("go");              // the home page's option buttons: /detect?go=ar|viewer, /battle?go=ar
  if (go === "ar") setTimeout(() => start(true), 0);
  else if (go === "viewer" && MODE === "detect") setTimeout(() => start(false), 0); }
