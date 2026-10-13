/* snapshot.js - "take a snapshot of the card" in the browser, so the server never has to look at a whole camera frame.
 * =====================================================================================================================
 * WHY: the old way uploaded a 1280-px camera frame (100-250 KB) plus the card corners; the server then re-detected the
 * card, warped it upright and only THEN ran the recogniser. Now, as soon as the tracker has the card's OUTER border
 * (its four corners), the browser cuts the card out and straightens it itself (warpCard) and uploads just that small
 * upright picture (~40-70 KB, JPEG). The server skips detection + warping entirely (POST /api/battle/scan, snapshot=1).
 *
 * ALSO HERE: a tiny "fingerprint" of the snapshot (cardSig, 16 x 22 grey values) and a small memory (SigCache). If the
 * same card is shown again - even after the tracker lost it and gave it a new id - the answer is recalled from memory
 * and NOTHING is sent to the server at all.
 *
 * Pure maths on pixel arrays (no DOM), so Node can test it (tests/test_snapshot.mjs). Works in the page and in Node.
 */
(function (root, factory) {
  const CG = typeof module === "object" && module.exports ? require("./card-geometry.js") : root.CardGeometry;
  const api = factory(CG);
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.CardSnapshot = api;
})(typeof self !== "undefined" ? self : this, function (CG) {
  "use strict";

  const CARD_ASPECT = 63 / 88;                                   // a card is 63 x 88 mm
  const PAD = 0.06;                                              // margin of real surroundings around the card (the recogniser likes a little context); same as agent.CARD_PAD
  const SIG_W = 16, SIG_H = 22;                                  // fingerprint size (same aspect as a card)

  /** How big the upright snapshot should be, from the card's apparent size: as tall as the longer of its two "tall"
   *  sides in the frame, clamped to 440-800 px (the same rule as card_detect.rectify, capped lower to save upload size). */
  function snapshotSize(corners, maxH = 800) {
    const side = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);
    const tall = Math.max(side(corners[1], corners[2]), side(corners[3], corners[0]));
    const ch = Math.round(Math.min(maxH, Math.max(440, tall))), cw = Math.round(ch * CARD_ASPECT);
    const px = Math.round(cw * PAD), py = Math.round(ch * PAD);
    return { cw, ch, px, py, w: cw + 2 * px, h: ch + 2 * py };
  }

  /** Warp the quad `corners` (TL, TR, BR, BL; pixels of the RGBA source `src` of size sw x sh) into an upright rectangle.
   *  Perspective warp = for every OUTPUT pixel, find where it comes from in the source (homography maps output -> source)
   *  and blend the 4 nearest source pixels (bilinear). Pixels outside the picture repeat the edge (like OpenCV's BORDER_REPLICATE).
   *  Returns {data: Uint8ClampedArray RGBA, w, h}. */
  function warpCard(src, sw, sh, corners, maxH) {
    const z = snapshotSize(corners, maxH);
    const dst = [[z.px, z.py], [z.px + z.cw, z.py], [z.px + z.cw, z.py + z.ch], [z.px, z.py + z.ch]];
    const H = CG.homography(dst, corners);                       // output rectangle -> the card's place in the frame
    const out = new Uint8ClampedArray(z.w * z.h * 4);
    for (let y = 0; y < z.h; y++) {
      for (let x = 0; x < z.w; x++) {
        const w = H[6] * x + H[7] * y + 1;
        let u = (H[0] * x + H[1] * y + H[2]) / w, v = (H[3] * x + H[4] * y + H[5]) / w;
        u = u < 0 ? 0 : u > sw - 1 ? sw - 1 : u; v = v < 0 ? 0 : v > sh - 1 ? sh - 1 : v;
        const x0 = Math.floor(u), y0 = Math.floor(v), x1 = Math.min(sw - 1, x0 + 1), y1 = Math.min(sh - 1, y0 + 1), fx = u - x0, fy = v - y0;
        const i00 = (y0 * sw + x0) * 4, i10 = (y0 * sw + x1) * 4, i01 = (y1 * sw + x0) * 4, i11 = (y1 * sw + x1) * 4, o = (y * z.w + x) * 4;
        for (let c = 0; c < 3; c++) {
          out[o + c] = (src[i00 + c] * (1 - fx) + src[i10 + c] * fx) * (1 - fy) + (src[i01 + c] * (1 - fx) + src[i11 + c] * fx) * fy;
        }
        out[o + 3] = 255;
      }
    }
    return { data: out, w: z.w, h: z.h };
  }

  /** 16 x 22 grey "fingerprint" of an RGBA picture: box-average down to size, then z-score (subtract the mean, divide by
   *  the spread) so a brighter or dimmer photo of the same card gives (nearly) the same numbers. */
  function cardSig(rgba, w, h) {
    const sig = new Float32Array(SIG_W * SIG_H), cnt = new Float32Array(SIG_W * SIG_H);
    for (let y = 0; y < h; y++) {
      const gy = Math.min(SIG_H - 1, Math.floor((y * SIG_H) / h));
      for (let x = 0; x < w; x++) {
        const gx = Math.min(SIG_W - 1, Math.floor((x * SIG_W) / w)), i = (y * w + x) * 4, k = gy * SIG_W + gx;
        sig[k] += 0.299 * rgba[i] + 0.587 * rgba[i + 1] + 0.114 * rgba[i + 2]; cnt[k]++;
      }
    }
    let m = 0;
    for (let k = 0; k < sig.length; k++) { sig[k] /= cnt[k] || 1; m += sig[k]; }
    m /= sig.length;
    let v = 0;
    for (let k = 0; k < sig.length; k++) v += (sig[k] - m) ** 2;
    const sd = Math.sqrt(v / sig.length) || 1;
    for (let k = 0; k < sig.length; k++) sig[k] = (sig[k] - m) / sd;
    return sig;
  }
  const sigDist = (a, b) => { let s = 0; for (let k = 0; k < a.length; k++) s += Math.abs(a[k] - b[k]); return s / a.length; };
  const sigRot180 = (a) => Float32Array.from(a).reverse();       // turning the picture upside down reverses the grid

  /** A small memory of answers keyed by card fingerprint. Entries store the fingerprint of the card's READABLE orientation
   *  (the way up the server read it), so a card shown upside down is still recognised and the page is told to flip. */
  class SigCache {
    constructor(limit = 24, maxDist = 0.3) { this.limit = limit; this.maxDist = maxDist; this.items = []; }
    /** sig = fingerprint of the snapshot as sent; rotated = the server said the card was upside down in that snapshot */
    add(sig, info, rotated) {
      const readable = rotated ? sigRot180(sig) : sig;
      this.items = this.items.filter((it) => sigDist(it.sig, readable) > 0.05);       // newer answer for the same card replaces the older one
      this.items.push({ sig: readable, info });
      if (this.items.length > this.limit) this.items.shift();
    }
    /** -> {info, flip} (flip = the snapshot is upside down relative to the readable card) or null */
    find(sig) {
      let best = null;
      const turned = sigRot180(sig);
      for (const it of this.items) {
        const d0 = sigDist(sig, it.sig), d1 = sigDist(turned, it.sig), d = Math.min(d0, d1);
        if (d < this.maxDist && (!best || d < best.d)) best = { d, info: it.info, flip: d1 < d0 };
      }
      return best;
    }
    replace(oldInfo, newInfo) { for (const it of this.items) if (it.info === oldInfo) it.info = newInfo; }
    clear() { this.items = []; }
  }

  return { CARD_ASPECT, PAD, SIG_W, SIG_H, snapshotSize, warpCard, cardSig, sigDist, sigRot180, SigCache };
});
