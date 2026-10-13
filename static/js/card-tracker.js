/* card-tracker.js - finds yellow-bordered Pokémon cards in camera frames and follows them.
 *
 * Three layers, so most of it can be tested without OpenCV:
 *   CvDetector   (OpenCV.js)  frame -> yellow mask -> contours -> hull -> 4-corner candidates   [plain data]
 *   TrackerCore  (pure JS)    candidates -> sharp corners -> card identity over time -> smoothing -> 3D pose
 *   CardTracker               the two together: process(rgbaFrame) -> tracks
 *
 * The OpenCV steps are the same as card_detect.py (Python OpenCV):
 *   RGBA -> RGB -> HSV -> inRange(yellow) -> morphology close -> findContours(external) -> convexHull -> approxPolyDP
 * Runs inside a Web Worker (tracker.worker.js) so the camera view / 3D rendering never waits for it.
 *
 * GUIDE FOR LEARNERS
 * ------------------
 *  Frame life-cycle (one camera picture, ~30 times a second):
 *    detect.js        copies the video frame to a small canvas and sends the RGBA pixels to the worker
 *    CvDetector       OpenCV.js: pixels -> yellow mask -> outlines (contours) -> convex hulls -> 4-corner guesses
 *    TrackerCore      plain JavaScript: refine the corners, decide "which card is this?" using the previous frames,
 *                     smooth the corners, compute the 3-D pose (card-geometry.js)
 *    result           a list of "tracks": {id, corners, modelView (4x4 matrix), stable, visible, ...}
 *  Why a Web Worker? JavaScript on the page is single-threaded: if OpenCV takes 20 ms, the video and 3-D drawing would
 *  stutter for 20 ms too. A worker is a second thread; it receives frames by message and replies with results.
 *  Why two classes? CvDetector needs the OpenCV.js library (large, browser-only). TrackerCore is pure JavaScript, so
 *  it can be unit tested in Node with candidates produced by Python OpenCV (tests/make_fixtures.py).
 *  Memory note: OpenCV.js objects (cv.Mat) live in WebAssembly memory and are NOT garbage collected: every
 *  Mat you create must be .delete()d, which is why buffers are allocated once (_alloc) and reused.
 */
(function (root, factory) {
  const CG = typeof module === "object" && module.exports ? require("./card-geometry.js") : root.CardGeometry;
  const api = factory(CG);
  if (typeof module === "object" && module.exports) module.exports = api;
  else Object.assign(root, api);
})(typeof self !== "undefined" ? self : this, function (CG) {
  "use strict";

  /* ================================================================ OpenCV.js half */
  /** Wraps the OpenCV.js calls. Output is plain data so the rest of the pipeline never touches OpenCV. */
  class CvDetector {
    constructor(cv) {
      this.cv = cv;
      this.w = 0; this.h = 0;
      this.mats = {};
      this.sensitivity = 0.5;
      this.kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
    }

    setSensitivity(s) { if (s !== this.sensitivity) { this.sensitivity = s; this._freeBounds(); } }

    _freeBounds() { for (const k of ["lo", "hi", "glo", "ghi"]) { this.mats[k] && this.mats[k].delete(); this.mats[k] = null; } }

    // (re)allocate working buffers only when the size changes. Creating Mats every frame would be slow and leak memory.
    _alloc(w, h) {
      const cv = this.cv, m = this.mats;
      if (this.w === w && this.h === h && m.rgba) return;
      for (const k of Object.keys(m)) m[k] && m[k].delete();
      this.mats = { rgba: new cv.Mat(h, w, cv.CV_8UC4), rgb: new cv.Mat(), hsv: new cv.Mat(), mask: new cv.Mat(), gmask: new cv.Mat(), union: new cv.Mat(), gray: new cv.Mat(), edges: new cv.Mat(), dil: new cv.Mat(), blob: new cv.Mat() };
      this.w = w; this.h = h;
    }

    // The lower/upper HSV bounds as full-size images (OpenCV.js's inRange wants Mats of the same size as the input).
    _bounds() {
      const cv = this.cv, m = this.mats;
      if (!m.lo) {
        const t = CG.thresholds(this.sensitivity);
        m.lo = new cv.Mat(this.h, this.w, cv.CV_8UC3, [t.lo[0], t.lo[1], t.lo[2], 0]);
        m.hi = new cv.Mat(this.h, this.w, cv.CV_8UC3, [t.hi[0], t.hi[1], t.hi[2], 0]);
        const g = CG.grayThresholds(this.sensitivity);                 // the gray (silver) border window
        m.glo = new cv.Mat(this.h, this.w, cv.CV_8UC3, [g.lo[0], g.lo[1], g.lo[2], 0]);
        m.ghi = new cv.Mat(this.h, this.w, cv.CV_8UC3, [g.hi[0], g.hi[1], g.hi[2], 0]);
      }
    }

    /** hull (cv.Mat) -> 4 corners [[x,y]...] or null; loosens approxPolyDP until it has exactly 4 convex corners. */
    _approxQuad(hull, P) {
      const cv = this.cv, peri = cv.arcLength(hull, true);
      for (const eps of P.epsSweep) {
        const approx = new cv.Mat();
        cv.approxPolyDP(hull, approx, eps * peri, true);
        let quad = null;
        if (approx.rows === 4) {
          const d = approx.data32S;
          quad = [[d[0], d[1]], [d[2], d[3]], [d[4], d[5]], [d[6], d[7]]];
        }
        approx.delete();
        if (quad && CG.isConvex(quad)) return quad;
      }
      return null;
    }

    /**
     * rgba: Uint8ClampedArray/Uint8Array of w*h*4. Returns {mask (view, valid until the next call), cands}.
     * cands = [{contour: Int32Array [x0,y0,...], approx: [[x,y]*4], hullArea}]  (same as card_detect.raw_candidates)
     */
    detect(rgba, w, h, P, repair = false) {
      const cv = this.cv;
      P = P || CG.PARAMS;
      this._alloc(w, h);
      this._bounds();
      const m = this.mats;
      m.rgba.data.set(rgba);                                    // copy the frame's pixels into WebAssembly memory
      cv.cvtColor(m.rgba, m.rgb, cv.COLOR_RGBA2RGB);            // drop the alpha channel
      cv.cvtColor(m.rgb, m.hsv, cv.COLOR_RGB2HSV);              // colour space where "yellow" is one hue range
      // One mask + one contour pass per border colour, so a gray card next to a yellow one stays a separate ring.
      cv.inRange(m.hsv, m.lo, m.hi, m.mask);                    // white where yellow, black elsewhere
      cv.inRange(m.hsv, m.glo, m.ghi, m.gmask);                 // white where gray (low colour, mid brightness)
      for (const k of ["mask", "gmask"]) cv.morphologyEx(m[k], m[k], cv.MORPH_CLOSE, this.kernel);   // fill 1-pixel holes in the border ring
      const painted = repair ? this._repair(w, h, P) : [];      // SECOND CHANCE only (see TrackerCore.update): edges heal the colour masks
      cv.bitwise_or(m.mask, m.gmask, m.union);                  // both together, for the debug overlay

      const cands = [], area = w * h;
      for (const [maskMat, color] of [[m.mask, "yellow"], [m.gmask, "gray"]]) {
        const contours = new cv.MatVector(), hier = new cv.Mat();
        cv.findContours(maskMat, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE);
        for (let i = 0; i < contours.size(); i++) {
          const cnt = contours.get(i);
          if (cv.contourArea(cnt) >= P.minArea * area * 0.5) {           // cheap reject of specks
            const hull = new cv.Mat();
            cv.convexHull(cnt, hull, false, true);
            const hullArea = cv.contourArea(hull);
            if (hullArea >= P.minArea * area && hullArea <= P.maxArea * area) {
              const approx = this._approxQuad(hull, P);
              if (approx) cands.push({ contour: Int32Array.from(cnt.data32S), approx, hullArea, color });
            }
            hull.delete();
          }
          cnt.delete();
        }
        contours.delete(); hier.delete();
        this._merged(maskMat, color, w, h, P, cands);          // broken rings: fragments close together are one object
      }
      return { mask: m.union.data, masks: { yellow: m.mask.data, gray: m.gmask.data }, cands, painted };
    }

    /** BROKEN RINGS (twin of card_detect.merged_candidates): thicken the mask so close fragments join, take each joined blob as one
     *  object and draw the hull round the ORIGINAL mask pixels inside it. Pushes candidates onto `out`. */
    _merged(maskMat, color, w, h, P, out) {
      const cv = this.cv, m = this.mats, k = Math.max(P.mergeMinPx, Math.round(P.mergeGap * w)) | 1;
      const kern = cv.getStructuringElement(cv.MORPH_ELLIPSE, new cv.Size(k, k));
      cv.dilate(maskMat, m.dil, kern); kern.delete();
      const contours = new cv.MatVector(), hier = new cv.Mat();
      cv.findContours(m.dil, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      const area = w * h, md = maskMat.data;
      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        if (cv.contourArea(cnt) >= P.minArea * area * 0.5) {
          const r = cv.boundingRect(cnt);
          m.blob.create(h, w, cv.CV_8UC1); m.blob.setTo(new cv.Scalar(0));
          cv.drawContours(m.blob, contours, i, new cv.Scalar(255), -1);
          const bd = m.blob.data, pts = [];
          for (let y = r.y; y < r.y + r.height; y++) for (let x = r.x; x < r.x + r.width; x++) if (md[y * w + x] && bd[y * w + x]) pts.push([x, y]);
          if (pts.length >= 20) {
            const hullPts = CG.convexHull(pts), flat = [];
            for (const p of hullPts) flat.push(p[0], p[1]);
            const hull = cv.matFromArray(hullPts.length, 1, cv.CV_32SC2, flat), hullArea = cv.contourArea(hull);
            if (hullArea >= P.minArea * area && hullArea <= P.maxArea * area) {
              const approx = this._approxQuad(hull, P);
              if (approx) out.push({ contour: Int32Array.from(flat), approx, hullArea, color });
            }
            hull.delete();
          }
        }
        cnt.delete();
      }
      contours.delete(); hier.delete();
    }

    /** MASK REPAIR (twin of card_detect.repair_masks): outlines from Canny edges; if >= ringSides of the 4 strips just inside an
     *  outline are clearly yellow / silver, all four strips are painted into that colour's mask. Returns the painted outlines. */
    _repair(w, h, P) {
      const cv = this.cv, m = this.mats, painted = [], area = w * h;
      cv.cvtColor(m.rgb, m.gray, cv.COLOR_RGB2GRAY);
      cv.GaussianBlur(m.gray, m.gray, new cv.Size(3, 3), 0);
      cv.Canny(m.gray, m.edges, P.cannyLo, P.cannyHi);
      cv.dilate(m.edges, m.edges, this.kernel);
      const contours = new cv.MatVector(), hier = new cv.Mat();
      cv.findContours(m.edges, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
      const hsv = m.hsv.data, masks = { yellow: m.mask.data, gray: m.gmask.data };
      const pixels = (quad, fn) => {                                    // call fn(index) for every pixel centre inside the convex quad
        const x0 = Math.max(0, Math.floor(Math.min(...quad.map((p) => p[0])))), x1 = Math.min(w - 1, Math.ceil(Math.max(...quad.map((p) => p[0]))));
        const y0 = Math.max(0, Math.floor(Math.min(...quad.map((p) => p[1])))), y1 = Math.min(h - 1, Math.ceil(Math.max(...quad.map((p) => p[1]))));
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) if (CG.insideQuad([x, y], quad, 0)) fn(y * w + x);
      };
      const median = (a) => { a.sort((p, q) => p - q); return a[a.length >> 1]; };
      for (let i = 0; i < contours.size(); i++) {
        const cnt = contours.get(i);
        if (cv.contourArea(cnt) >= P.minArea * area * 0.5) {
          const hull = new cv.Mat();
          cv.convexHull(cnt, hull, false, true);
          const hullArea = cv.contourArea(hull);
          const quad = hullArea >= P.minArea * area && hullArea <= P.maxArea * area ? this._approxQuad(hull, P) : null;
          if (quad) {
            const thick = Math.max(2, P.ringThick * Math.min(Math.hypot(quad[1][0] - quad[0][0], quad[1][1] - quad[0][1]), Math.hypot(quad[2][0] - quad[1][0], quad[2][1] - quad[1][1])));
            for (const grow of [0, ...P.ringExpand]) {
              const q = grow === 0 ? quad : CG.expandQuad(quad, grow * thick), classes = [];
              for (let side = 0; side < 4; side++) {
                const H = [], S = [], V = [];
                pixels(CG.innerStrip(q, side, thick), (k) => { H.push(hsv[3 * k]); S.push(hsv[3 * k + 1]); V.push(hsv[3 * k + 2]); });
                classes.push(H.length >= 6 ? CG.sideClass(median(H), median(S), median(V), P) : null);
              }
              const best = classes.filter((c) => c === "yellow").length >= classes.filter((c) => c === "gray").length ? "yellow" : "gray";
              if (classes.filter((c) => c === best).length >= P.ringSides) {
                for (let side = 0; side < 4; side++) pixels(CG.innerStrip(q, side, thick, [-0.02, 1.02], [0, 1]), (k) => { masks[best][k] = 255; });
                painted.push(q);
                break;
              }
            }
          }
          hull.delete();
        }
        cnt.delete();
      }
      contours.delete(); hier.delete();
      return painted;
    }

    dispose() {
      this._freeBounds();
      for (const k of Object.keys(this.mats)) this.mats[k] && this.mats[k].delete();
      this.mats = {};
      this.kernel.delete();
    }
  }

  /* ================================================================ pure-JS half */
  // Tracking settings. "Track" = our memory of one physical card across frames.
  const DEFAULTS = {
    maxCards: 2,
    hfov: 63,                     // assumed horizontal field of view (degrees)
    lostAfterMs: 250,             // a card not seen for this long stops being drawn ...
    forgetAfterMs: 1500,          // ... and is forgotten (identity dropped) after this long
    reacquireAfterMs: 400,        // a card unseen for longer than this may reappear ANYWHERE (picked up and put down) and is still the same card ...
    reacquireFrac: 1.0,           // ... within this fraction of the frame diagonal (instead of maxMoveFrac)
    maxMoveFrac: 0.25,            // a card may move this far (fraction of the frame diagonal) between detections
    stableHits: 4,                // detections before a card counts as stable (worth sending to the recogniser)
    repair: true, repairEveryMs: 400,                         // second detection pass from edge outlines (silver borders in shadow, weak edges, all-yellow cards); see update()
    smooth: { minCutoff: 1.0, beta: 0.1, dCutoff: 1.0 },      // One Euro filter, units: px at 1280-wide video (tuned on tests/fixtures: ~2 px lag at 150 px/s, 73 % less jitter)
    params: CG.PARAMS,
  };

  /** The brain: turns per-frame candidates into steady, identified cards with 3-D poses. Pure JS, no OpenCV. */
  class TrackerCore {
    constructor(opts) {
      this.o = Object.assign({}, DEFAULTS, opts || {});
      this.tracks = [];
      this.nextId = 1;
      this._lastRepair = -1e9;
    }

    setOptions(o) { Object.assign(this.o, o || {}); }
    reset() { this.tracks = []; }

    /** Turn the card with this id 180 degrees (its yellow border looks the same either way). */
    flip(id) {
      const t = this.tracks.find((x) => x.id === id);
      if (t) this._setCorners(t, CG.rotateList(t.corners, 2));
    }

    _setCorners(t, corners) {                      // also restarts the smoothing filters in the new corner order
      t.corners = corners;
      t.filters = null;
    }

    // One One-Euro filter per coordinate (4 corners x 2 = 8). `k` rescales pixels to a 1280-wide reference so the
    // filter behaves the same for any camera resolution; dt = seconds since this card was last updated.
    _smooth(t, rawCorners, dt, k) {
      const S = this.o.smooth;
      if (!t.filters) t.filters = Array.from({ length: 8 }, () => new CG.OneEuro(S.minCutoff, S.beta, S.dCutoff));
      return rawCorners.map((p, i) => [t.filters[2 * i].filter(p[0] * k, dt) / k, t.filters[2 * i + 1].filter(p[1] * k, dt) / k]);
    }

    /**
     * raw = {w, h, srcW, srcH, mask, cands} from the detector (w x h = analysis size, src = video size).
     * ts = timestamp in ms. Returns the list of tracks for this frame.
     */
    update(raw, ts) {
      const o = this.o, P = o.params;
      const sx = raw.srcW / raw.w, sy = raw.srcH / raw.h;
      const K = CG.intrinsics(raw.srcW, raw.srcH, o.hfov);
      const diag = Math.hypot(raw.srcW, raw.srcH);

      // The four phases below are the whole tracker. Read them in order:
      // 1. candidates -> detections with sharp corners, in video pixels
      const toDets = (r) => {
        const out = [];
        for (const c of r.cands) {
          const cq = CG.candidateQuad(c, (r.masks && r.masks[c.color || "yellow"]) || r.mask, r.w, r.h, P);   // each colour is judged on its own mask
          if (!cq) continue;
          const quad = cq.quad.map((p) => [(p[0] + 0.5) * sx - 0.5, (p[1] + 0.5) * sy - 0.5]);
          out.push({ quad, cover: cq.cover, color: cq.color, score: (CG.polyArea(cq.quad) / (r.w * r.h)) * cq.cover,
                     centre: [quad.reduce((s, p) => s + p[0], 0) / 4, quad.reduce((s, p) => s + p[1], 0) / 4] });
        }
        out.sort((a, b) => b.score - a.score);
        return CG.suppressInner(CG.dedupe(out));                            // the same card found in two colours counts once; a frame INSIDE a card is not a card
      };
      let dets = toDets(raw);
      if (raw.repair && o.repair !== false && (!dets.length || ts - this._lastRepair >= o.repairEveryMs)) {   // SECOND PASS: heal the masks from edge outlines (always when nothing was found, else every repairEveryMs: it is cheap but not free)
        this._lastRepair = ts;
        const r2 = raw.repair();
        const outs = (r2.painted || []).map((q) => q.map((p) => [(p[0] + 0.5) * sx - 0.5, (p[1] + 0.5) * sy - 0.5]));
        dets = CG.mergeRepaired(dets, CG.dropInsidePainted(toDets({ ...raw, ...r2 }), outs));   // the OUTER answer wins (same rule as card_detect.merge_passes)
      }

      // 2. which detection is which known card?  (nearest-centre matching; a card may move at most maxMoveFrac of the frame diagonal)
      const live = this.tracks.filter((t) => ts - t.lastSeen <= o.forgetAfterMs);
      const limits = live.map((t) => (ts - t.lastSeen > o.reacquireAfterMs ? o.reacquireFrac : o.maxMoveFrac) * diag);   // a card that vanished for a while may come back far away
      const { pairs, newDets } = CG.associate(live.map((t) => this._centre(t)), dets.map((d) => d.centre), limits);
      const seen = new Set();

      for (const [ti, di] of pairs) {
        const t = live[ti], d = dets[di];
        const ratio = CG.polyArea(d.quad) / Math.max(1e-6, CG.polyArea(t.corners));
        if (ts - t.lastSeen <= o.reacquireAfterMs && (ratio < P.sizeGate[0] || ratio > P.sizeGate[1])) continue;   // a card does not change size between frames: this is something else (an inner frame, a neighbour)
        // `prev: t.corners` makes the naming of TL/TR/BR/BL follow the previous frame, so a card never spins by 180 degrees on its own
        const oriented = CG.orientQuad(d.quad, K, { prev: t.corners, maxResidual: P.maxResidual });
        if (oriented.residual > P.maxResidual) continue;                       // not card-shaped after all
        const dt = Math.min(0.25, Math.max(0.004, (ts - t.lastSeen) / 1000));
        t.raw = oriented.corners;
        t.corners = this._smooth(t, oriented.corners, dt, 1280 / raw.srcW);    // (first call after a flip just re-seeds the filters)
        t.cover = d.cover; t.residual = oriented.residual; t.hits++; t.lastSeen = ts;
        seen.add(t.id);
      }

      // 3. unmatched detections start new cards (best first, up to maxCards)
      for (const di of newDets) {
        const d = dets[di];
        if (this.tracks.filter((t) => ts - t.lastSeen <= o.forgetAfterMs).length >= o.maxCards) {
          const stale = this.tracks.filter((t) => !seen.has(t.id)).sort((a, b) => a.lastSeen - b.lastSeen)[0];
          if (!stale) break;
          this.tracks.splice(this.tracks.indexOf(stale), 1);                    // a slot is taken by a card that vanished
        }
        const oriented = CG.orientQuad(d.quad, K, { maxResidual: P.maxResidual });
        if (oriented.residual > P.maxResidual) continue;
        const t = { id: this.nextId++, corners: oriented.corners, raw: oriented.corners, filters: null, cover: d.cover,
                    residual: oriented.residual, hits: 1, firstSeen: ts, lastSeen: ts, pose: null };
        this._smooth(t, oriented.corners, 1 / 30, 1280 / raw.srcW);
        this.tracks.push(t);
        seen.add(t.id);
      }
      this.tracks = this.tracks.filter((t) => ts - t.lastSeen <= o.forgetAfterMs);

      // 4. pose of every card (refreshed only when it was seen this frame). A card missed for < lostAfterMs keeps its last
      //    pose and stays visible (a finger passing over the border should not make the model blink out).
      return this.tracks.map((t) => {
        if (seen.has(t.id)) {
          const pose = CG.poseFromCorners(t.corners, K);
          if (pose) { t.pose = pose; t.modelView = CG.modelViewThree(pose); }
        }
        const missedMs = ts - t.lastSeen;
        return { id: t.id, corners: t.corners.map((p) => p.slice()), modelView: t.modelView || null, residual: t.residual,
                 cover: t.cover, hits: t.hits, stable: t.hits >= o.stableHits, visible: !!t.modelView && missedMs <= o.lostAfterMs,
                 missedMs, ageMs: ts - t.firstSeen };
      });
    }

    _centre(t) { return [t.corners.reduce((s, p) => s + p[0], 0) / 4, t.corners.reduce((s, p) => s + p[1], 0) / 4]; }
  }

  /* ================================================================ both together */
  /** Convenience wrapper used by the worker: detector + core behind one process() call. */
  class CardTracker {
    constructor(cv, opts) {
      this.core = new TrackerCore(opts);
      this.detector = cv ? new CvDetector(cv) : null;
      this.debug = false;
    }

    setOptions(o) {
      if (!o) return;
      if (o.sensitivity !== undefined && this.detector) this.detector.setSensitivity(o.sensitivity);
      if (o.debug !== undefined) this.debug = !!o.debug;
      this.core.setOptions(o);
    }

    flip(id) { this.core.flip(id); }

    /** One camera frame (RGBA at analysis size) -> {tracks, mask?}. srcW/srcH = real video size. */
    process(rgba, w, h, ts, srcW, srcH) {
      const P = this.core.o.params, det = this.detector.detect(rgba, w, h, P);
      const raw = { w, h, srcW, srcH, mask: det.mask, masks: det.masks, cands: det.cands };
      raw.repair = () => {                                                  // only called when the plain pass found no card
        const d2 = this.detector.detect(rgba, w, h, P, true);               // (the frame is still in `rgba`)
        return { mask: d2.mask, masks: d2.masks, cands: d2.cands, painted: d2.painted };
      };
      const tracks = this.core.update(raw, ts);
      return { tracks, srcW, srcH, candidates: det.cands.length, mask: this.debug ? new Uint8Array(det.mask) : null, w, h };
    }

    dispose() { this.detector && this.detector.dispose(); }
  }

  return { CvDetector, TrackerCore, CardTracker };
});
