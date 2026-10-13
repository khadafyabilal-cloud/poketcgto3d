/* battle.js - AR battle mode.
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
 *     └─ every ~30 ms a small copy of the frame goes to tracker.worker.js (OpenCV.js finds the yellow card border)
 *   worker ──► card corners + 3D pose per card (card-geometry.js maths)
 *     └─ three.js draws one anchor per card on a transparent canvas laid exactly over the video
 *   a card that stays put for a few frames is sent (full frame + corners) to /api/battle/scan:
 *     OpenCV (Python) straightens it, the agent identifies it, and the 3D model URL comes back.
 */
import * as THREE from "three";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";
import { DRACOLoader } from "three/addons/loaders/DRACOLoader.js";
import * as SkeletonUtils from "three/addons/utils/SkeletonUtils.js";

const $ = (id) => document.getElementById(id);
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
function showError(msg) { $("err").textContent = msg; $("go").disabled = false; $("start").hidden = false; setStatus("Not running", "err"); }

/* ------------------------------------------------------------------ state */
let CG, video, backend = "", worker, workerReady = false, busy = false, running = false;
let vw = 0, vh = 0, bw = 0, bh = 0, workH = 0, interval = 33, lastSent = 0, lastMs = 0, fps = 0, lastResult = 0;
let renderer, scene, camera, K, selectedId = null, scanning = false, lastFrameTime = performance.now();
const entries = new Map();                                              // track id -> everything we know about that card
const work = document.createElement("canvas"), wctx = work.getContext("2d", { willReadFrequently: true });
const fx = $("fx"), fctx = fx.getContext("2d");
const maskCanvas = document.createElement("canvas");

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
const trackerOptions = () => ({ sensitivity: S.sens, hfov: S.fov, debug: S.debug, maxCards: 2 });
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
  if (!document.hidden && workerReady && !busy && now - lastSent >= interval) sendFrame(now);
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
  tag.addEventListener("click", () => { selectedId = id; openSheet("sPick"); });
  $("labels").appendChild(tag);
  const e = { id, anchor, holder, shadow, tag, state: "new", tries: 0, nextTry: 0, mixer: null, model: null, ref: null, info: null,
              corners: null, visible: false, stable: false, hasPose: false, heading: 0, flash: 0,
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

function placeModel(e, gltf, pk) {
  e.holder.clear(); e.mixer = null;
  const root = SkeletonUtils.clone(gltf.scene);
  root.traverse((o) => { if (o.isSkinnedMesh) o.frustumCulled = false; });          // skinned bounds are unreliable while animating
  if (gltf.animations.length) { e.mixer = new THREE.AnimationMixer(root); e.mixer.clipAction(gltf.animations[0]).play(); e.mixer.update(0); }
  root.updateMatrixWorld(true);
  const box = new THREE.Box3().setFromObject(root, true), size = box.getSize(new THREE.Vector3()), c = box.getCenter(new THREE.Vector3());
  const hf = pk && pk.height_m ? Math.min(1.25, Math.max(0.7, Math.pow(pk.height_m / 0.9, 0.25))) : 1;   // bigger Pokémon look a bit bigger
  const s = Math.min((0.95 * hf) / Math.max(size.x, size.z, 1e-3), (1.5 * hf) / Math.max(size.y, 1e-3));   // fits the card, stands on it
  root.scale.setScalar(s);
  root.position.set(-c.x * s, -box.min.y * s, -c.z * s);
  e.holder.add(root); e.model = root;
  e.shadow.scale.setScalar(Math.max(size.x, size.z) * s * 1.5); e.shadow.visible = true;
}

/* ------------------------------------------------------------------ results from the tracker
 * The worker's pose is a noisy 4x4 matrix. We decompose it into position + rotation (Matrix4.decompose) and let frame()
 * glide the anchor toward it (lerp/slerp), a second smoothing stage after the tracker's One Euro filter. */
const m4 = new THREE.Matrix4(), unitScale = new THREE.Vector3();
function onResult(m) {
  const now = performance.now();
  if (lastResult) fps = fps * 0.9 + (1000 / Math.max(1, now - lastResult)) * 0.1;
  lastResult = now;
  if (m.ms) { lastMs = lastMs * 0.8 + m.ms * 0.2; interval = Math.min(100, Math.max(33, 2 * lastMs)); }   // keep OpenCV under ~half a core
  const seen = new Set();
  for (const t of m.tracks) {
    seen.add(t.id);
    const e = entries.get(t.id) || makeEntry(t.id);
    e.corners = t.corners; e.visible = t.visible; e.stable = t.stable;
    if (t.modelView) {
      m4.fromArray(t.modelView).decompose(e.tp, e.tq, unitScale);
      if (!e.hasPose) { e.anchor.position.copy(e.tp); e.anchor.quaternion.copy(e.tq); e.hasPose = true; }
    }
  }
  for (const e of [...entries.values()]) if (!seen.has(e.id)) { if (selectedId === e.id) selectedId = null; disposeEntry(e); }
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

async function scan(e) {
  scanning = true; e.state = "scanning"; e.tries++;
  setTag(e, "Reading card…", "");
  try {
    const k = Math.min(1, 1280 / Math.max(vw, vh)), cap = document.createElement("canvas");
    cap.width = Math.round(vw * k); cap.height = Math.round(vh * k);
    cap.getContext("2d").drawImage(video, 0, 0, cap.width, cap.height);
    const blob = await new Promise((r) => cap.toBlob(r, "image/jpeg", 0.85));
    const fd = new FormData();
    fd.append("image", blob, "frame.jpg");
    fd.append("corners", JSON.stringify(e.corners.map((p) => [p[0] * k, p[1] * k])));
    fd.append("sensitivity", String(S.sens));
    const r = await fetch("/api/battle/scan", { method: "POST", body: fd }), d = await r.json();
    if (!r.ok) throw new Error(d.error || r.statusText);
    if (!entries.has(e.id)) return;                                    // the card left while we waited
    if (d.rotated) worker.postMessage({ type: "flip", id: e.id });     // the card was upside down relative to the tracker's guess
    await applyResult(e, d);
  } catch (err) {
    console.warn("scan failed:", err);
    failScan(e, err.message || "Couldn't reach the server");
  } finally { scanning = false; }
}

async function applyResult(e, d) {
  if (!d.model_url) return failScan(e, d.message || "No model for this card");
  const ref = getModel(d.model_url);
  if (e.ref) e.ref.refs = Math.max(0, e.ref.refs - 1);
  e.ref = ref;
  const gltf = await ref.promise;
  if (!entries.has(e.id)) return;
  placeModel(e, gltf, d.pokemon);
  e.info = d; e.state = "ready"; e.flash = performance.now();
  const p = d.pokemon || {};
  setTag(e, p.name || (d.card && d.card.name) || "Pokémon", [(p.types || []).join(" / "), d.card && d.card.set].filter(Boolean).join("  ·  "));
}
function failScan(e, msg) {
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
  const now = performance.now(), dt = Math.min(0.1, (now - lastFrameTime) / 1000), kk = 1 - Math.exp(-dt / 0.04);
  lastFrameTime = now;
  if (video.videoWidth && (video.videoWidth !== vw || video.videoHeight !== vh)) layout();     // phone rotated
  const live = [...entries.values()].filter((e) => e.visible && e.hasPose);
  for (const e of entries.values()) {
    if (e.hasPose) { e.anchor.position.lerp(e.tp, kk); e.anchor.quaternion.slerp(e.tq, kk); }
    e.anchor.visible = e.visible && e.hasPose && !!e.model;
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
    e.heading += wrapAngle(target - e.heading) * kk;
    e.holder.rotation.y = e.heading;
  }
  renderer.render(scene, camera);

  fctx.clearRect(0, 0, bw, bh);
  if (S.debug && lastMask) { fctx.imageSmoothingEnabled = false; fctx.drawImage(lastMask, 0, 0, bw, bh); fctx.imageSmoothingEnabled = true; }
  const map = coverMap();
  for (const e of entries.values()) {
    const show = e.visible && e.corners;
    e.tag.hidden = !show || !e.tag.firstChild;
    if (!show) continue;
    drawBrackets(e, now);
    const top = [(e.corners[0][0] + e.corners[1][0]) / 2, Math.min(e.corners[0][1], e.corners[1][1])];
    e.tag.style.left = top[0] * map.s + map.ox + "px"; e.tag.style.top = top[1] * map.s + map.oy + "px";
  }
  const nVis = live.length, ready = live.filter((e) => e.state === "ready").length;
  if (!nVis) setStatus("Looking for a card with a yellow border");
  else if (scanning) setStatus("Reading the card…", "lock");
  else setStatus(nVis === 1 ? (ready ? "1 card · Pokémon on" : "1 card locked") : `${nVis} cards · ${ready} Pokémon on`, "lock");
  if (S.debug) $("stats").textContent = `camera  ${backend}  ${vw}x${vh}\ntracker ${fps.toFixed(0)} fps  ${lastMs.toFixed(1)} ms  (every ${interval.toFixed(0)} ms)\ncards   ${entries.size}  scanning ${scanning}`;
}

/* ------------------------------------------------------------------ UI */
function openSheet(id) { for (const s of ["sPick", "sSet"]) $(s).hidden = s !== id; if (id === "sPick") setTimeout(() => $("pickQ").focus(), 50); }
const closeSheets = () => { $("sPick").hidden = true; $("sSet").hidden = true; };
const selected = () => entries.get(selectedId) || [...entries.values()].find((e) => e.visible) || null;

function bindUi() {
  $("bFlip").onclick = () => { const e = selected(); if (e) worker.postMessage({ type: "flip", id: e.id }); };
  $("bPick").onclick = () => openSheet("sPick");
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

/* ------------------------------------------------------------------ start / stop */
function stopCamera() {
  running = false;
  const s = video && video.srcObject; s && s.getTracks().forEach((t) => t.stop());
}
document.addEventListener("visibilitychange", () => {                   // save battery: nothing runs while the tab is hidden
  if (!running) return;
  if (document.hidden) renderer.setAnimationLoop(null);
  else { renderer.setAnimationLoop(frame); lastFrameTime = performance.now(); }
});
addEventListener("pagehide", () => { stopCamera(); worker && worker.postMessage({ type: "dispose" }); });

async function start() {
  $("go").disabled = true; $("err").textContent = "";
  if (!window.isSecureContext || !navigator.mediaDevices) {
    return showError("The camera only works on a secure page.\nOn this computer open http://localhost:5000/battle. On a phone, start the server with:\n  python app.py --host 0.0.0.0 --https\nand open the https:// address it prints.");
  }
  try {
    setStatus("Loading tracker…");
    CG = await loadScript("/static/js/card-geometry.js" + VER).then(() => window.CardGeometry);
    const workerUp = startWorker();                                     // OpenCV.js (~10 MB the first time) loads while the camera starts
    workerUp.catch(() => {});
    setStatus("Starting camera…");
    await openCamera();
    setStatus("Loading OpenCV…");
    await workerUp;
  } catch (e) {
    stopCamera(); worker && worker.terminate();
    const msg = e && e.name === "NotAllowedError" ? "Camera access was blocked. Allow the camera for this site in your browser settings, then try again."
      : e && e.name === "NotFoundError" ? "No camera was found on this device."
      : "Couldn't start battle mode.\n" + (e && e.message || e);
    return showError(msg);
  }
  initScene(); layout(); bindUi();
  $("start").hidden = true; $("dock").hidden = false;
  running = true;
  renderer.setAnimationLoop(frame);
  pump();
}
$("go").onclick = start;
