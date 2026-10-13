// Tests for static/js/card-geometry.js and the pure half of card-tracker.js (TrackerCore).
// Run:  node --test tests/
// The OpenCV step is covered with REAL cv2 output stored in tests/fixtures (python tests/make_fixtures.py).
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import fs from "node:fs";
import zlib from "node:zlib";
import path from "node:path";
import { fileURLToPath } from "node:url";

const require = createRequire(import.meta.url);
const here = path.dirname(fileURLToPath(import.meta.url));
const CG = require("../static/js/card-geometry.js");
const { TrackerCore } = require("../static/js/card-tracker.js");
const fixture = (n) => JSON.parse(fs.readFileSync(path.join(here, "fixtures", n), "utf8"));

/* ------------------------------------------------------------ helpers */
function rng(seed) {                                       // mulberry32: deterministic random numbers
  return () => { seed |= 0; seed = (seed + 0x6d2b79f5) | 0; let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t; return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
const gauss = (r) => Math.sqrt(-2 * Math.log(1 - r())) * Math.cos(2 * Math.PI * r());
const deg = (x) => (x * Math.PI) / 180;
const matmul = (A, B) => { const C = new Array(9).fill(0); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) for (let k = 0; k < 3; k++) C[3 * i + j] += A[3 * i + k] * B[3 * k + j]; return C; };
function rot(yaw, pitch, roll) {                           // same as tests/synth.py
  const [y, p, r] = [yaw, pitch, roll].map(deg);
  const Ry = [Math.cos(y), 0, Math.sin(y), 0, 1, 0, -Math.sin(y), 0, Math.cos(y)];
  const Rx = [1, 0, 0, 0, Math.cos(p), -Math.sin(p), 0, Math.sin(p), Math.cos(p)];
  const Rz = [Math.cos(r), -Math.sin(r), 0, Math.sin(r), Math.cos(r), 0, 0, 0, 1];
  return matmul(Rz, matmul(Rx, Ry));
}
const apply = (R, p) => [0, 1, 2].map((i) => R[3 * i] * p[0] + R[3 * i + 1] * p[1] + R[3 * i + 2] * p[2]);
const H = CG.CARD_H;
const CARD = [[-0.5, -H / 2, 0], [0.5, -H / 2, 0], [0.5, H / 2, 0], [-0.5, H / 2, 0]];       // TL TR BR BL, card-local (OpenCV style)
function project(K, R, t, pts) {
  return pts.map((p) => { const c = apply(R, p).map((v, i) => v + t[i]); return [K.fx * c[0] / c[2] + K.cx, K.fy * c[1] / c[2] + K.cy]; });
}
function randomPose(r, K, opt = {}) {
  for (;;) {
    const R = rot((r() - 0.5) * 90, r() * (opt.maxPitch ?? 60), opt.roll ?? r() * 360);
    const width = 90 + r() * 310, u = 0.2 + 0.6 * r(), v = 0.2 + 0.6 * r();
    const z = K.fx / width, t = [(u * K.w - K.cx) / K.fx * z, (v * K.h - K.cy) / K.fy * z, z];
    const corners = project(K, R, t, CARD);
    if (corners.every((c) => c[0] > 0 && c[0] < K.w && c[1] > 0 && c[1] < K.h)) return { R, t, corners };
  }
}
const angleBetween = (a, b) => Math.acos(Math.max(-1, Math.min(1, a[0] * b[0] + a[1] * b[1] + a[2] * b[2])));
const rotAngle = (A, B) => { const M = matmul(A, [B[0], B[3], B[6], B[1], B[4], B[7], B[2], B[5], B[8]]); return Math.acos(Math.max(-1, Math.min(1, (M[0] + M[4] + M[8] - 1) / 2))); };
const median = (a) => [...a].sort((x, y) => x - y)[Math.floor(a.length / 2)];
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
const normal = (R) => [R[2], R[5], R[8]];                  // card Z axis in the camera frame

/** Line `pts` up with `ref` allowing a cyclic shift; returns {pts, shift, errors}. */
function matchOrder(pts, ref) {
  let best = null;
  for (let k = 0; k < 4; k++) {
    const q = CG.rotateList(pts, k), e = q.map((p, i) => dist(p, ref[i]));
    const s = e.reduce((a, b) => a + b, 0);
    if (!best || s < best.s) best = { q, shift: k, e, s };
  }
  return best;
}
const unz = (b64) => new Uint8Array(zlib.inflateSync(Buffer.from(b64, "base64")));
function rawFrame(f) {
  const masks = f.masks ? Object.fromEntries(Object.entries(f.masks).map(([k, v]) => [k, unz(v)])) : undefined;
  return { w: f.w, h: f.h, srcW: f.srcW, srcH: f.srcH, mask: unz(f.mask), masks,
           cands: f.cands.map((c) => ({ contour: Int32Array.from(c.contour), approx: c.approx, hullArea: c.hullArea, color: c.color })) };
}

/* ------------------------------------------------------------ pure maths */
describe("pose maths", () => {
  const K = CG.intrinsics(960, 540, 63);

  test("homography maps the four points exactly", () => {
    const src = [[0, 0], [1, 0], [1, 1.4], [0, 1.4]], dst = [[100, 80], [400, 120], [380, 470], [90, 400]];
    const h = CG.homography(src, dst);
    src.forEach((p, i) => {
      const w = h[6] * p[0] + h[7] * p[1] + 1;
      assert.ok(dist([(h[0] * p[0] + h[1] * p[1] + h[2]) / w, (h[3] * p[0] + h[4] * p[1] + h[5]) / w], dst[i]) < 1e-6);
    });
  });

  test("noise-free corners give back the exact pose", () => {
    const r = rng(1);
    for (let i = 0; i < 300; i++) {
      const p = randomPose(r, K), est = CG.poseFromCorners(p.corners, K);
      assert.ok(rotAngle(est.R, p.R) < deg(0.05), `rotation off by ${(rotAngle(est.R, p.R) * 180 / Math.PI).toFixed(3)} deg`);
      assert.ok(Math.hypot(...est.t.map((v, k) => v - p.t[k])) / Math.hypot(...p.t) < 1e-4);
      assert.ok(est.residual < 1e-6);
    }
  });

  test("0.6 px corner noise: median error a few degrees / 2 %", () => {
    const r = rng(2), rotErr = [], tErr = [];
    for (let i = 0; i < 400; i++) {
      const p = randomPose(r, K, { maxPitch: 50 });
      const noisy = p.corners.map((c) => [c[0] + 0.6 * gauss(r), c[1] + 0.6 * gauss(r)]);
      const est = CG.poseFromCorners(noisy, K);
      rotErr.push(angleBetween(normal(est.R), normal(p.R)) * 180 / Math.PI);
      tErr.push(Math.hypot(...est.t.map((v, k) => v - p.t[k])) / Math.hypot(...p.t));
    }
    assert.ok(median(rotErr) < 4, `median normal error ${median(rotErr).toFixed(2)} deg`);
    assert.ok(median(tErr) < 0.02, `median position error ${(100 * median(tErr)).toFixed(2)} %`);
  });

  test("a wrong field-of-view guess still anchors the model exactly on the card centre", () => {
    const r = rng(3), Ktrue = CG.intrinsics(960, 540, 78), Kguess = CG.intrinsics(960, 540, 55), proj = CG.projectionThree(Kguess, 0.05, 50);
    for (let i = 0; i < 100; i++) {
      const p = randomPose(r, Ktrue, { maxPitch: 45 }), est = CG.poseFromCorners(p.corners, Kguess);
      const c = CG.projectThree(CG.modelViewThree(est), proj, [0, 0, 0], 960, 540);
      const truthCentre = project(Ktrue, p.R, p.t, [[0, 0, 0]])[0];
      assert.ok(dist(c, truthCentre) < 1e-6, `centre moved by ${dist(c, truthCentre)} px`);
    }
  });

  test("three.js matrices: card corners land exactly on the detected pixels", () => {
    const r = rng(4), proj = CG.projectionThree(K, 0.05, 50);
    // card-local three frame: x right, y out of the card, z toward the bottom edge  => corners at y = 0
    const corners3 = [[-0.5, 0, -H / 2], [0.5, 0, -H / 2], [0.5, 0, H / 2], [-0.5, 0, H / 2]];
    for (let i = 0; i < 200; i++) {
      const p = randomPose(r, K), mv = CG.modelViewThree(CG.poseFromCorners(p.corners, K));
      corners3.forEach((c, k) => assert.ok(dist(CG.projectThree(mv, proj, c, 960, 540), p.corners[k]) < 1e-4));
    }
  });

  test("model 'up' (+Y) points out of the card toward the viewer", () => {
    const r = rng(5);
    for (let i = 0; i < 100; i++) {
      const p = randomPose(r, K), mv = CG.modelViewThree(CG.poseFromCorners(p.corners, K));
      const up = [mv[4], mv[5], mv[6]], pos = [mv[12], mv[13], mv[14]];          // three camera looks down -Z
      assert.ok(up[0] * -pos[0] + up[1] * -pos[1] + up[2] * -pos[2] > 0, "model would be drawn standing INTO the card");
      assert.ok(pos[2] < 0, "card must be in front of the camera");
    }
  });

  test("projection matrix: the principal point is the centre of the screen", () => {
    const P = CG.projectionThree(K, 0.1, 100);
    const c = [0, 0, -5, 1], clip = [0, 1, 2, 3].map((k) => P[k] * c[0] + P[4 + k] * c[1] + P[8 + k] * c[2] + P[12 + k] * c[3]);
    assert.ok(Math.abs(clip[0] / clip[3]) < 1e-12 && Math.abs(clip[1] / clip[3]) < 1e-12);
  });
});

describe("naming the corners", () => {
  const K = CG.intrinsics(960, 540, 63);

  test("the card's width pair is found in every pose (even when it looks square)", () => {
    const r = rng(6);
    for (let i = 0; i < 1000; i++) {
      const p = randomPose(r, K), shuffled = CG.rotateList(p.corners, Math.floor(r() * 4));
      const noisy = shuffled.map((c) => [c[0] + 0.5 * gauss(r), c[1] + 0.5 * gauss(r)]);
      const o = CG.orientQuad(noisy, K), m = matchOrder(o.corners, p.corners);
      assert.ok(m.shift === 0 || m.shift === 2, `width/height swapped (shift ${m.shift})`);          // right up to the 180-degree ambiguity
    }
  });

  test("an upright-ish card is named top-left first (top = higher in the image)", () => {
    const r = rng(7);
    for (let i = 0; i < 300; i++) {
      const p = randomPose(r, K, { maxPitch: 45, roll: (r() - 0.5) * 100 }), o = CG.orientQuad(p.corners, K);
      assert.equal(matchOrder(o.corners, p.corners).shift, 0);
    }
  });

  test("flip turns the answer 180 degrees", () => {
    const r = rng(8), p = randomPose(r, K), a = CG.orientQuad(p.corners, K), b = CG.orientQuad(p.corners, K, { flip: true });
    assert.deepEqual(b.corners, CG.rotateList(a.corners, 2));
  });

  test("with the previous naming a card keeps its orientation, even turned upside down", () => {
    const r = rng(9);
    for (let i = 0; i < 300; i++) {
      const p = randomPose(r, K, { roll: 150 + r() * 60 }), prev = p.corners.map((c) => [c[0] + 3 * gauss(r), c[1] + 3 * gauss(r)]);
      const shuffled = CG.rotateList(p.corners, Math.floor(r() * 4)).reverse();                       // any order in, as OpenCV gives it
      const o = CG.orientQuad(shuffled, K, { prev });
      assert.equal(matchOrder(o.corners, p.corners).shift, 0, "orientation jumped");
    }
  });

  test("a non-card quadrilateral has a large residual", () => {
    const thin = [[100, 100], [160, 100], [160, 420], [100, 420]], square = [[300, 100], [500, 100], [500, 300], [300, 300]];
    assert.ok(CG.orientQuad(thin, K).residual > CG.PARAMS.maxResidual);
    assert.ok(CG.orientQuad(square, K).residual > 0.25);
  });
});

describe("smoothing and matching", () => {
  test("One Euro filter calms a still card but follows a moving one", () => {
    const r = rng(10), f = new CG.OneEuro(1.0, 0.1, 1), raw = [], out = [];
    for (let i = 0; i < 200; i++) { const v = 500 + gauss(r) * 1.2; raw.push(v); out.push(f.filter(v, 1 / 30)); }
    const sd = (a) => Math.sqrt(a.slice(30).reduce((s, v) => s + (v - 500) ** 2, 0) / (a.length - 30));
    assert.ok(sd(out) < 0.55 * sd(raw), `std ${sd(out).toFixed(2)} vs raw ${sd(raw).toFixed(2)}`);
    const g = new CG.OneEuro(1.0, 0.1, 1);
    let last = 0;
    for (let i = 0; i < 90; i++) last = g.filter(100 + i * 12, 1 / 30);                              // 360 px/s ramp
    assert.ok(Math.abs(last - (100 + 89 * 12)) < 12, `lag ${(100 + 89 * 12) - last} px at 360 px/s`);
  });

  test("associate: nearest first, no double use, distance limit", () => {
    const a = CG.associate([[100, 100], [500, 100]], [[490, 110], [110, 95], [900, 900]], 60);
    assert.deepEqual(a.pairs.sort((x, y) => x[0] - y[0]), [[0, 1], [1, 0]]);
    assert.deepEqual(a.newDets, [2]);
    assert.deepEqual(CG.associate([[0, 0]], [[500, 500]], 100).lost, [0]);
  });

  test("sensitivity thresholds are identical to Python's", () => {
    const py = fixture("thresholds.json");
    for (const [s, [lo, hi]] of Object.entries(py)) {
      const t = CG.thresholds(+s);
      assert.deepEqual([t.lo, t.hi], [lo, hi], `sensitivity ${s}`);
    }
  });
});

describe("gray border (silver) support", () => {
  test("gray HSV window equals Python's at every sensitivity", () => {
    for (const [s, [lo, hi]] of Object.entries(fixture("gray_thresholds.json"))) {
      const t = CG.grayThresholds(+s);
      assert.deepEqual([t.lo, t.hi], [lo, hi], `sensitivity ${s}`);
    }
  });
  test("interiorFraction: thin ring is low, solid slab is high", () => {
    const w = 100, h = 100, ring = new Uint8Array(w * h), slab = new Uint8Array(w * h).fill(1);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) if (x < 6 || x >= 94 || y < 6 || y >= 94) ring[y * w + x] = 1;
    const q = [[0, 0], [99, 0], [99, 99], [0, 99]];
    assert.ok(CG.interiorFraction(ring, w, h, q) < 0.05);
    assert.equal(CG.interiorFraction(slab, w, h, q), 1);
  });
  test("dedupe keeps one of two detections of the same card", () => {
    const q = (dx) => [[100 + dx, 100], [200 + dx, 100], [200 + dx, 240], [100 + dx, 240]];
    const out = CG.dedupe([{ quad: q(0), score: 1 }, { quad: q(3), score: 2 }, { quad: q(400), score: 0.5 }]);
    assert.equal(out.length, 2); assert.equal(out[0].score, 2);
  });
});

/* ------------------------------------------------------------ real OpenCV candidates */
describe("one frame of real OpenCV output (cv2 fixtures)", () => {
  const { meta, scenes } = fixture("scenes.json");
  const K = CG.intrinsics(meta.W, meta.H, meta.hfov);

  for (const sc of scenes) {
    test(sc.name, () => {
      const core = new TrackerCore(), tracks = core.update(rawFrame(sc), 0);
      assert.equal(tracks.length, sc.truth.length, `expected ${sc.truth.length} card(s), got ${tracks.length}`);
      for (const tr of sc.truth) {
        const t = tracks.reduce((b, c) => dist(c.corners[0], tr.corners[0]) + 1e5 * 0 < 1e9 && (!b || dist(centre(c.corners), centre(tr.corners)) < dist(centre(b.corners), centre(tr.corners)) ? c : b), null);
        const m = matchOrder(t.corners, tr.corners);
        assert.ok(Math.max(...m.e) < 2.5, `corner errors ${m.e.map((x) => x.toFixed(2))} px`);
        // pose: position within a few %, card plane within a few degrees
        const est = CG.poseFromCorners(t.corners, K), tErr = Math.hypot(...est.t.map((v, k) => v - tr.t[k])) / Math.hypot(...tr.t);
        const nErr = angleBetween(normal(est.R), normal(tr.R)) * 180 / Math.PI;
        assert.ok(tErr < (sc.name === "far" ? 0.08 : 0.04), `position error ${(100 * tErr).toFixed(1)} %`);
        assert.ok(nErr < (sc.name === "far" ? 12 : 8), `card plane off by ${nErr.toFixed(1)} deg`);
      }
    });
  }

  test("agrees with the Python implementation to a hundredth of a pixel", () => {
    for (const sc of scenes.filter((s) => s.python.length)) {
      const tracks = new TrackerCore({ maxCards: 4 }).update(rawFrame(sc), 0);
      assert.equal(tracks.length, sc.python.length, sc.name);
      for (const py of sc.python) {
        const t = tracks.find((c) => dist(centre(c.corners), centre(py)) < 5), m = matchOrder(t.corners, py);
        assert.ok(Math.max(...m.e) < 0.01, `${sc.name}: JS vs Python ${m.e.map((x) => x.toFixed(4))} px`);
      }
    }
  });
});
const centre = (c) => [c.reduce((s, p) => s + p[0], 0) / 4, c.reduce((s, p) => s + p[1], 0) / 4];

/* ------------------------------------------------------------ tracking over time */
describe("tracking a moving card (cv2 fixture sequence)", () => {
  const { meta, frames } = fixture("sequence_move.json");
  const core = new TrackerCore(), K = CG.intrinsics(meta.W, meta.H, meta.hfov);
  const ids = { A: new Set(), B: new Set() }, shifts = { A: new Set(), B: new Set() }, errs = [], visibleGap = [], log = [];

  frames.forEach((f, i) => {
    const tracks = core.update(rawFrame(f), f.ts);
    log.push(tracks.map((t) => t.id));
    for (const tr of f.truth) {
      const t = tracks.filter((c) => c.missedMs === 0).reduce((b, c) => (!b || dist(centre(c.corners), centre(tr.corners)) < dist(centre(b.corners), centre(tr.corners)) ? c : b), null);
      if (!t) continue;
      const m = matchOrder(t.corners, tr.corners);
      ids[tr.card].add(t.id); shifts[tr.card].add(m.shift); errs.push(Math.max(...m.e));
    }
    if (i >= 20 && i < 24) visibleGap.push(tracks.find((t) => ids.A.has(t.id)));
  });

  test("each card keeps one identity - also across a 4-frame occlusion", () => {
    assert.equal(ids.A.size, 1, `card A got ids ${[...ids.A]}`);
    assert.equal(ids.B.size, 1, `card B got ids ${[...ids.B]}`);
    assert.notEqual([...ids.A][0], [...ids.B][0]);
    assert.equal(core.nextId, 3, "no extra tracks were created");
  });

  test("a card hidden for 130 ms is still there (coasting) and then picked up again", () => {
    assert.equal(visibleGap.length, 4);
    for (const t of visibleGap) { assert.ok(t, "track vanished during the gap"); assert.ok(t.visible); assert.ok(t.missedMs > 0); }
  });

  test("corner naming never flips while the cards move", () => {
    assert.equal(shifts.A.size, 1, `card A naming shifts ${[...shifts.A]}`);
    assert.equal(shifts.B.size, 1, `card B naming shifts ${[...shifts.B]}`);
  });

  test("smoothed corners stay close to the truth while moving", () => {
    const mean = errs.reduce((a, b) => a + b, 0) / errs.length, max = Math.max(...errs);
    assert.ok(mean < 3 && max < 6, `worst-corner error: mean ${mean.toFixed(2)} px, max ${max.toFixed(2)} px`);
  });

  test("a card unseen for 1.5 s is forgotten; a new one gets a new id; flip() turns a card 180 degrees", () => {
    const c = new TrackerCore(), f0 = rawFrame(frames[12]);
    const first = c.update(f0, 0), id0 = first[0].id;
    c.flip(id0);
    const after = c.update(f0, 33).find((t) => t.id === id0);
    assert.deepEqual(after.corners.map((p) => p.map((v) => +v.toFixed(0))), CG.rotateList(first[0].corners, 2).map((p) => p.map((v) => +v.toFixed(0))));
    const empty = { ...f0, cands: [] };
    assert.equal(c.update(empty, 400).length > 0, true, "still remembered after 0.4 s");
    assert.equal(c.update(empty, 2200).length, 0, "forgotten after 2.2 s");
    assert.notEqual(c.update(f0, 2300)[0].id, id0);
  });
});

describe("AR stability options (battle mode keeps cards longer)", () => {
  const { frames } = fixture("sequence_static.json"), f0 = rawFrame(frames[12]), empty = { ...f0, cands: [] };
  test("lostAfterMs / forgetAfterMs are respected: a patient tracker keeps the same id and the model visible", () => {
    const c = new TrackerCore({ lostAfterMs: 1200, forgetAfterMs: 12000 });
    const id0 = c.update(f0, 0)[0].id;
    assert.equal(c.update(empty, 1000)[0].visible, true, "still visible after 1.0 s unseen");
    assert.equal(c.update(empty, 1500)[0].visible, false, "hidden (not deleted) after 1.5 s");
    assert.equal(c.update(empty, 9000).length, 1, "still remembered after 9 s");
    const back = c.update(f0, 9100);
    assert.equal(back.length, 1); assert.equal(back[0].id, id0, "same id when the card returns");
  });
  test("the default tracker is impatient (so the page must not rely on it)", () => {
    const c = new TrackerCore(); c.update(f0, 0);
    assert.equal(c.update(empty, 600)[0].visible, false);
  });
  test("a card back after a long absence is the same card even if it reappears far away", () => {
    const c = new TrackerCore({ forgetAfterMs: 12000 }), id0 = c.update(f0, 0)[0].id;
    const moved = { ...f0, cands: f0.cands.map((cd) => ({ ...cd, contour: Int32Array.from(cd.contour, (v, i) => (i % 2 === 0 ? v + Math.round(f0.w * 0.4) : v)) })) };
    const t = c.update(moved, 5000);
    assert.equal(t.length >= 1, true);
    assert.ok(t.some((x) => x.id === id0), "re-acquired as the old track");
  });
  test("associate: a per-track distance limit lets a far track skip while a near one still matches", () => {
    const a = CG.associate([[100, 100], [500, 100]], [[110, 95], [900, 900]], [60, 60]);
    assert.deepEqual(a.pairs, [[0, 0]]);
    const b = CG.associate([[100, 100], [500, 100]], [[110, 95], [900, 900]], [60, 2000]);
    assert.equal(b.pairs.length, 2);
  });
});

describe("outer border only (inner frames are never cards)", () => {
  test("outerClear: 1 with table all round, low when the colour continues outside", () => {
    const w = 200, h = 200, q = [[60, 60], [140, 60], [140, 140], [60, 140]], m = new Uint8Array(w * h);
    assert.equal(CG.outerClear(m, w, h, q), 1);
    m.fill(255);
    assert.ok(CG.outerClear(m, w, h, q) < 0.1);
  });
  test("suppressInner drops a quad wholly inside a bigger one, keeps tiny or separate ones", () => {
    const box = (x0, y0, x1, y1) => ({ quad: [[x0, y0], [x1, y0], [x1, y1], [x0, y1]] });
    const big = box(10, 10, 170, 170), small = box(60, 60, 140, 140), tiny = box(90, 90, 100, 100), beside = box(200, 10, 260, 70);
    const kept = CG.suppressInner([big, small, tiny, beside]);
    assert.deepEqual(kept, [big, tiny, beside]);
    assert.deepEqual(CG.suppressInner([box(0, 0, 10, 10).quad && box(5, 5, 15, 15)]).length, 1);   // a single detection is always kept
  });
  test("the tracker refuses a detection of a very different size as the same card (size gate)", () => {
    const { frames } = fixture("sequence_static.json"), f0 = rawFrame(frames[12]);
    const run = (gate) => { const c = new TrackerCore({ params: { ...CG.PARAMS, sizeGate: gate } }); const a = c.update(f0, 0)[0]; return [a.hits, c.update(f0, 33).find((x) => x.id === a.id).hits]; };
    assert.deepEqual(run([0.6, 1.6]), [1, 2], "same size: accepted");
    assert.deepEqual(run([1.2, 1.6]), [1, 1], "a detection that is 1.0x the size is outside [1.2, 1.6]: refused");
  });
});

describe("a card that does not move (cv2 fixture sequence)", () => {
  const { meta, frames } = fixture("sequence_static.json");

  test("smoothing cuts the frame-to-frame jitter", () => {
    const core = new TrackerCore(), raw = [], smooth = [];
    for (const f of frames) {
      const r = rawFrame(f);
      const cq = CG.candidateQuad(r.cands[0], r.mask, r.w, r.h), sx = r.srcW / r.w;
      const t = core.update(r, f.ts)[0];
      raw.push(CG.orientQuad(cq.quad.map((p) => [(p[0] + 0.5) * sx - 0.5, (p[1] + 0.5) * sx - 0.5]), CG.intrinsics(meta.W, meta.H, 63), { prev: t.corners }).corners);
      smooth.push(t.corners);
    }
    const jitter = (series) => { let s = 0, n = 0; for (let k = 0; k < 4; k++) for (let a = 0; a < 2; a++) { const v = series.slice(8).map((c) => c[k][a]), m = v.reduce((x, y) => x + y, 0) / v.length; s += v.reduce((x, y) => x + (y - m) ** 2, 0) / v.length; n++; } return Math.sqrt(s / n); };
    const jr = jitter(raw), js = jitter(smooth);
    assert.ok(js < 0.7 * jr, `jitter ${js.toFixed(3)} px vs ${jr.toFixed(3)} px unsmoothed`);
  });
});

describe("PoseFilter (render-side motion smoothing)", () => {
  const noisy = (r, s) => ({ x: gauss(r) * s, y: gauss(r) * s, z: gauss(r) * s });
  const qz = (a) => ({ x: 0, y: 0, z: Math.sin(a / 2), w: Math.cos(a / 2) });
  const sdOf = (a) => Math.sqrt(a.reduce((s, v) => s + v * v, 0) / a.length);

  test("a still card: jitter of the pose is cut by more than 70 %", () => {
    const r = rng(21), f = new CG.PoseFilter(), p = { x: 0, y: 0, z: 0 }, q = qz(0), raw = [], out = [];
    for (let i = 0; i < 300; i++) {
      const n = noisy(r, 0.004), a = gauss(r) * 0.01;                        // 0.4 % of a card width, 0.6 degrees
      f.step(p, q, n, qz(a), 1 / 30);
      if (i > 40) { raw.push(n.x); out.push(p.x); }
    }
    assert.ok(sdOf(out) < 0.3 * sdOf(raw), `std ${sdOf(out).toFixed(5)} vs raw ${sdOf(raw).toFixed(5)}`);
  });

  test("a moving card is followed quickly (no floaty lag)", () => {
    const f = new CG.PoseFilter(), p = { x: 0, y: 0, z: 0 }, q = qz(0);
    let lagFrames = 0;
    for (let i = 0; i < 30; i++) f.step(p, q, { x: 0, y: 0, z: 0 }, qz(0), 1 / 30);
    const target = { x: 0.6, y: 0, z: 0 };                                      // the card jumps 0.6 card widths sideways
    while (Math.abs(p.x - target.x) > 0.03 && lagFrames < 100) { f.step(p, q, target, qz(0), 1 / 30); lagFrames++; }
    assert.ok(lagFrames <= 12, `${lagFrames} frames (${(lagFrames / 30).toFixed(2)} s) to get within 5 %`);
  });

  test("rotation takes the short way round and stays a unit quaternion", () => {
    const f = new CG.PoseFilter(), p = { x: 0, y: 0, z: 0 }, q = qz(3.0);
    for (let i = 0; i < 60; i++) f.step(p, q, p, qz(-3.0), 1 / 30);             // +172 deg -> -172 deg is only 16 deg apart
    assert.ok(Math.abs(Math.hypot(q.x, q.y, q.z, q.w) - 1) < 1e-9);
    assert.ok(Math.abs(q.w) < 0.2, "went through w~0 (the 180-degree side), not the long way through 0");
  });

  test("dt = 0 changes nothing", () => {
    const f = new CG.PoseFilter(), p = { x: 1, y: 2, z: 3 }, q = qz(0.5);
    f.step(p, q, { x: 9, y: 9, z: 9 }, qz(2), 0);
    assert.deepEqual([p.x, p.y, p.z], [1, 2, 3]);
  });
});

describe("mask repair helpers (twins of card_detect.py)", () => {
  const q = [[100, 50], [300, 60], [290, 260], [110, 250]];
  const near = (a, b, tol = 1e-3) => a.every((p, i) => Math.hypot(p[0] - b[i][0], p[1] - b[i][1]) < tol);
  test("innerStrip and expandQuad give the same numbers as Python", () => {
    assert.ok(near(CG.innerStrip(q, 0, 7), [[149.9126, 54.2478], [249.9126, 59.2478], [249.7204, 63.093], [149.7204, 58.0930]]));
    assert.ok(near(CG.innerStrip(q, 2, 7, [-0.02, 1.02], [0, 1]), [[293.6, 260.2], [106.4, 249.8], [106.7883, 242.8108], [293.9883, 253.2108]]));
    assert.ok(near(CG.expandQuad(q, 10), [[93.1034, 42.7586], [306.8966, 53.4483], [296.2069, 267.2414], [103.7931, 256.5517]]));
  });
  test("sideClass: yellow, gold, silver, cream-ish silver, and a red card face", () => {
    assert.equal(CG.sideClass(25, 200, 230), "yellow");
    assert.equal(CG.sideClass(21, 230, 200), "yellow");          // gold
    assert.equal(CG.sideClass(8, 45, 225), "gray");              // light silver in warm light
    assert.equal(CG.sideClass(100, 20, 130), "gray");
    assert.equal(CG.sideClass(5, 200, 235), null);               // red/orange card face
    assert.equal(CG.sideClass(0, 0, 40), null);                  // dark table
  });
  test("convexHull drops interior points", () => {
    const h = CG.convexHull([[0, 0], [10, 0], [10, 10], [0, 10], [5, 5], [3, 7], [10, 5]]);
    assert.equal(h.length, 4);
  });
  test("dropInsidePainted removes inner frames only", () => {
    const outer = [[0, 0], [200, 0], [200, 280], [0, 280]], inner = [[20, 20], [180, 20], [180, 260], [20, 260]], other = [[300, 0], [400, 0], [400, 100], [300, 100]];
    const dets = [{ quad: inner }, { quad: other }, { quad: outer }];
    const kept = CG.dropInsidePainted(dets, [outer]);
    assert.deepEqual(kept.map((d) => d.quad), [other, outer]);
  });
});
