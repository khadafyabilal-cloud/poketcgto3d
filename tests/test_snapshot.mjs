// Tests for static/js/snapshot.js: cutting a card out of a frame in the browser, and the card fingerprint memory.
// Run:  node --test tests/test_snapshot.mjs
import { test, describe } from "node:test";
import assert from "node:assert/strict";
import { createRequire } from "node:module";
const require = createRequire(import.meta.url);
const S = require("../static/js/snapshot.js");

const W = 640, H = 480;
/** a frame whose colour at (x, y) is a known function of position, so a warped pixel can be checked exactly */
function frame(f) {
  const d = new Uint8ClampedArray(W * H * 4);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) { const i = (y * W + x) * 4, c = f(x, y); d[i] = c[0]; d[i + 1] = c[1]; d[i + 2] = c[2]; d[i + 3] = 255; }
  return d;
}
const gradient = (x, y) => [x / 3, y / 2, (x + y) / 5];

describe("warpCard", () => {
  test("size follows the card's apparent height, clamped to 440-800 px, with a 6 % margin", () => {
    const q = (h) => [[0, 0], [h * 0.716, 0], [h * 0.716, h], [0, h]];
    assert.equal(S.snapshotSize(q(200)).ch, 440);
    assert.equal(S.snapshotSize(q(600)).ch, 600);
    assert.equal(S.snapshotSize(q(2000)).ch, 800);
    const z = S.snapshotSize(q(600));
    assert.equal(z.w, z.cw + 2 * z.px); assert.ok(Math.abs(z.cw / z.ch - 63 / 88) < 0.002);
  });
  test("an upright card is copied pixel-for-pixel (a gradient comes out as the same gradient)", () => {
    const src = frame(gradient), quad = [[100, 50], [100 + 250, 50], [100 + 250, 50 + 350], [100, 50 + 350]];
    const o = S.warpCard(src, W, H, quad), z = S.snapshotSize(quad);
    const at = (x, y) => Array.from(o.data.slice((y * o.w + x) * 4, (y * o.w + x) * 4 + 3));
    for (const [fx, fy] of [[0.1, 0.1], [0.5, 0.5], [0.9, 0.9], [0.2, 0.8]]) {
      const x = Math.round(z.px + fx * z.cw), y = Math.round(z.py + fy * z.ch), sx = 100 + fx * 250, sy = 50 + fy * 350;
      const want = gradient(sx, sy);
      assert.ok(at(x, y).every((v, k) => Math.abs(v - want[k]) < 4), `pixel ${fx},${fy}: ${at(x, y)} vs ${want.map(Math.round)}`);
    }
  });
  test("a tilted card: the four output corners come from the four quad corners (TL stays top-left)", () => {
    const src = frame(gradient), quad = [[200, 80], [430, 110], [410, 420], [170, 380]];
    const o = S.warpCard(src, W, H, quad), z = S.snapshotSize(quad);
    const at = (x, y) => Array.from(o.data.slice((y * o.w + x) * 4, (y * o.w + x) * 4 + 3));
    [[z.px, z.py, 0], [z.px + z.cw - 1, z.py, 1], [z.px + z.cw - 1, z.py + z.ch - 1, 2], [z.px, z.py + z.ch - 1, 3]].forEach(([x, y, k]) => {
      const want = gradient(quad[k][0], quad[k][1]);
      assert.ok(at(x, y).every((v, c) => Math.abs(v - want[c]) < 14), `corner ${k}: ${at(x, y)} vs ${want.map(Math.round)}`);
    });
  });
  test("a quad that sticks out of the picture does not crash (edge pixels repeat)", () => {
    const o = S.warpCard(frame(gradient), W, H, [[-50, -40], [300, -40], [300, 500], [-50, 500]]);
    assert.equal(o.data.length, o.w * o.h * 4);
  });
});

describe("fingerprints and the memory", () => {
  const card = (seed) => {                                       // a card-like picture: blocks of colour
    let s = seed; const r = () => (s = (s * 16807) % 2147483647) / 2147483647;
    const blocks = Array.from({ length: 40 }, () => [r() * 300, r() * 420, 20 + r() * 90, 20 + r() * 90, [r() * 255, r() * 255, r() * 255]]);
    return (x, y) => { let c = [230, 225, 200]; for (const [bx, by, bw, bh, col] of blocks) if (x >= bx && x < bx + bw && y >= by && y < by + bh) c = col; return c; };
  };
  const sigOf = (f, gain = 1, noise = 0) => {
    const w = 300, h = 420, d = new Uint8ClampedArray(w * h * 4);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const c = f(x, y), i = (y * w + x) * 4; for (let k = 0; k < 3; k++) d[i + k] = c[k] * gain + (Math.sin(x * 12.9 + y * 78.2) * noise); d[i + 3] = 255; }
    return S.cardSig(d, w, h);
  };

  test("the same card in dimmer light (and noisy) is close; a different card is far", () => {
    const a = card(1), same = S.sigDist(sigOf(a), sigOf(a, 0.6, 6)), other = S.sigDist(sigOf(a), sigOf(card(2)));
    assert.ok(same < 0.12, `same card distance ${same}`);
    assert.ok(other > 0.5, `other card distance ${other}`);
  });
  test("SigCache recalls a card, flips for an upside-down one, and refuses strangers", () => {
    const c = new S.SigCache(), a = card(3), sa = sigOf(a), info = { name: "Pikachu" };
    c.add(sa, info, false);
    assert.deepEqual(c.find(sigOf(a, 0.7)), { d: c.find(sigOf(a, 0.7)).d, info, flip: false });
    const up = c.find(S.sigRot180(sa));
    assert.ok(up && up.info === info && up.flip === true, "an upside-down snapshot is the same card, turned");
    assert.equal(c.find(sigOf(card(4))), null);
  });
  test("when the server said the snapshot was upside down, the readable orientation is what is stored", () => {
    const c = new S.SigCache(), a = card(5), sa = sigOf(a), info = { name: "X" };
    c.add(sa, info, true);                                        // `sa` was upside down; the readable card is its turn
    assert.equal(c.find(S.sigRot180(sa)).flip, false);           // a later snapshot that already reads upright needs no flip
    assert.equal(c.find(sa).flip, true);
  });
  test("a newer answer for the same card replaces the older one; replace() and the size limit work", () => {
    const c = new S.SigCache(3), a = card(6), s = sigOf(a), i1 = { v: 1 }, i2 = { v: 2 };
    c.add(s, i1, false); c.add(sigOf(a, 0.9), i2, false);
    assert.equal(c.items.length, 1); assert.equal(c.find(s).info, i2);
    const i3 = { v: 3 }; c.replace(i2, i3); assert.equal(c.find(s).info, i3);
    for (let n = 10; n < 16; n++) c.add(sigOf(card(n)), { n }, false);
    assert.equal(c.items.length, 3);
  });
});
