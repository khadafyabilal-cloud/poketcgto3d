/* card-geometry.js - the maths behind the AR card tracker. No OpenCV, no DOM: it runs in a Web Worker
 * (importScripts), in Node (require - that is how it is unit-tested) and in a page.
 *
 * Mirrors card_detect.py (the Python OpenCV module). Keep PARAMS / thresholds() identical.
 *
 * Conventions
 *   card-local (OpenCV style): origin = card centre, X right, Y down the card, Z into the card.
 *   1 unit = one card WIDTH, so the card is 1 x CARD_H (= 88/63) units.
 *   camera: pinhole, x right, y down, z forward, in video pixels.
 *   three.js scene: card lies in the XZ plane with +Y pointing OUT of the card (models stand on it) and
 *   +Z toward the card's bottom edge; camera at the origin looking down -Z. modelViewThree() converts.
 *
 * GUIDE FOR LEARNERS - how a flat picture of a card becomes a 3-D placement
 * -------------------------------------------------------------------------
 *  1. OpenCV (card-tracker.js) gives us 4 corner pixels of the yellow border. That is all the "vision" we have.
 *  2. A card is a rigid 63 x 88 mm rectangle. If we know how a rectangle of known size appears as a 4-point shape on
 *     screen, we can compute where it is in space. The tool is the HOMOGRAPHY: the 3x3 matrix H that maps
 *     points of the flat card (x, y, 1) to image pixels. homography() solves for it from the 4 point pairs
 *     (8 equations, 8 unknowns -> solveLinear).
 *  3. POSE: with the camera matrix K (focal length, centre) we compute K^-1 * H. For a rigid plane that equals
 *     [r1 r2 t] up to scale, where r1, r2 are the card's x- and y-axes in camera space and t is the card centre.
 *     The third axis r3 = r1 x r2 (cross product). Because of pixel noise the three axes are not perfectly
 *     perpendicular, so nearestRotation() "cleans" them into a true rotation matrix (polar decomposition).
 *  4. modelViewThree() re-labels the axes to three.js conventions (y up, camera looks down -z) and returns a
 *     4x4 matrix. three.js multiplies every model vertex by it, so the model sits exactly on the card.
 *  5. projectionThree() builds the matching 4x4 PROJECTION matrix from K, so what three.js draws lines up with the
 *     camera video pixel for pixel.
 *  6. Raw corner positions jitter by a pixel or two every frame. OneEuro smooths them: strong smoothing when
 *     the card is still (no visible shake), little when it moves (no lag).
 *  7. associate() decides which detection in this frame is "the same card" as a track from the previous frame.
 *
 * Words: "residual" = how badly a measurement disagrees with the model; here, how un-card-like a quad is.
 * "intrinsics" = the camera's own numbers (focal length, centre). "pose" = position + rotation in space.
 */
(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.CardGeometry = api;
})(typeof self !== "undefined" ? self : this, function () {
  "use strict";

  const CARD_H = 88 / 63;          // card height when the card WIDTH is 1 unit (real cards: 63 x 88 mm)
  const PARAMS = {
    workWidth: 480,                 // frames are analysed at this width
    minArea: 0.006, maxArea: 0.85,  // card size as a fraction of the frame
    epsSweep: [0.02, 0.03, 0.045, 0.06, 0.08],
    minCover: 0.55,                 // fraction of the card's edge that must be yellow
    minFill: 0.9,                   // 4-corner polygon area / hull area
    edgeSkip: 0.12,                 // ignore this much at both ends of a side (rounded corners)
    edgeTol: 2.5,                   // px: narrowest band that counts as "on the side"
    maxResidual: 0.6,               // how un-card-like a quad may look
    hfov: 63,                       // assumed horizontal field of view, degrees
    maxGrayInterior: 0.8,           // GRAY borders only: at most this fraction of the card's inside may itself be gray (metal energy cards are gray all over; laptops / phones are ~0.9+)
    dupDist: 0.35,                  // detections closer than this x card size are the same card seen in two border colours
    outerClearMin: 0.5,             // OUTER BORDER ONLY: at least this fraction of the band just OUTSIDE the card must NOT be border-coloured
    containRatio: 0.2,              // a candidate wholly inside a bigger valid card (and >= this fraction of its area) is an inner frame
    mergeGap: 0.012, mergeMinPx: 5,  // BROKEN RINGS: mask fragments closer than this (fraction of frame width, min px) are joined into one object
    cannyLo: 40, cannyHi: 110,      // MASK REPAIR from edges (second chance when the colour masks find no card): Canny thresholds
    ringThick: 0.035,               // ... width of the colour strip looked at just inside each side (fraction of the card's short side)
    ringSides: 3,                   // ... this many of the 4 sides must show the same border colour
    ringExpand: [1, 2],             // ... when the strips inside an outline are not border-coloured, try the outline grown outward by this many ring widths
    ringYellowH: [14, 40], ringYellowSMin: 90, ringYellowVMin: 110,   // ... a strip's MEDIAN colour counts as yellow / gold when in here
    ringGraySMax: 70, ringGrayVMin: 100,                              // ... or as silver
    replaceRatio: 1.15,             // a repaired detection replaces a plain one only if it is this much bigger and surrounds it
    sizeGate: [0.6, 1.6],           // tracker: a detection must be 0.6-1.6 x the size of the card it continues
  };

  /* ---------------------------------------------------------------- HSV window for the yellow border
   * Same numbers as card_detect.thresholds(): lo/hi are the lower/upper (Hue, Saturation, Value) bounds a pixel
   * must lie between to count as "yellow". Higher sensitivity widens the window. */
  const round = (x) => Math.floor(x + 0.5);
  function thresholds(sensitivity) {
    const s = Math.min(1, Math.max(0, sensitivity === undefined ? 0.5 : +sensitivity));
    return { lo: [round(21 - 4 * s), round(140 - 90 * s), round(150 - 80 * s)], hi: [round(34 + 4 * s), 255, 255] };
  }

  /* GRAY (silver) border window: no colour, mid brightness. Same numbers as card_detect.gray_thresholds(). */
  function grayThresholds(sensitivity) {
    const s = Math.min(1, Math.max(0, sensitivity === undefined ? 0.5 : +sensitivity));
    return { lo: [0, 0, round(110 - 25 * s)], hi: [179, round(28 + 22 * s), round(205 + 20 * s)] };
  }

  /* ---------------------------------------------------------------- small linear algebra */
  const dot = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
  const norm = (a) => Math.hypot(a[0], a[1], a[2]);
  const scale = (a, k) => a.map((v) => v * k);

  // Solve A x = b for x. Gaussian elimination: repeatedly subtract multiples of one row from the rows below it
  // until the matrix is triangular, then solve from the bottom row up ("back substitution"). "Partial pivoting"
  // = pick the row with the largest number in the column first, which keeps rounding errors small.
  function solveLinear(A, b) {
    const n = b.length, M = A.map((row, i) => [...row, b[i]]);
    for (let c = 0; c < n; c++) {
      let p = c;
      for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
      if (Math.abs(M[p][c]) < 1e-12) return null;
      [M[c], M[p]] = [M[p], M[c]];
      for (let r = c + 1; r < n; r++) {
        const f = M[r][c] / M[c][c];
        for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
      }
    }
    const x = new Array(n).fill(0);
    for (let r = n - 1; r >= 0; r--) {
      let s = M[r][n];
      for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
      x[r] = s / M[r][r];
    }
    return x;
  }

  // 3x3 matrices are row-major arrays of 9
  const det3 = (m) => m[0] * (m[4] * m[8] - m[5] * m[7]) - m[1] * (m[3] * m[8] - m[5] * m[6]) + m[2] * (m[3] * m[7] - m[4] * m[6]);
  const transpose3 = (m) => [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
  function inv3(m) {
    const d = det3(m);
    if (Math.abs(d) < 1e-14) return null;
    const c = (i, j, k, l) => m[i] * m[l] - m[j] * m[k];
    return [c(4, 5, 7, 8), -c(1, 2, 7, 8), c(1, 2, 4, 5), -c(3, 5, 6, 8), c(0, 2, 6, 8), -c(0, 2, 3, 5),
            c(3, 4, 6, 7), -c(0, 1, 6, 7), c(0, 1, 3, 4)].map((v) => v / d);
  }

  /** Homography (3x3, row-major, h33 = 1) taking 4 source points to 4 destination points.
   *  Each point pair (x,y)->(u,v) gives two linear equations in the 8 unknown entries of H:
   *      u = (h1 x + h2 y + h3) / (h7 x + h8 y + 1)   and   v = (h4 x + h5 y + h6) / (h7 x + h8 y + 1)
   *  Multiplying out the denominator makes them linear, so 4 pairs = 8 equations = solvable. */
  function homography(src, dst) {
    const A = [], b = [];
    for (let i = 0; i < 4; i++) {
      const [x, y] = src[i], [u, v] = dst[i];
      A.push([x, y, 1, 0, 0, 0, -u * x, -u * y]); b.push(u);
      A.push([0, 0, 0, x, y, 1, -v * x, -v * y]); b.push(v);
    }
    const h = solveLinear(A, b);
    return h && [...h, 1];
  }

  /** The rotation matrix nearest to M (polar decomposition by Newton iteration; no SVD needed).
   *  Idea: averaging R with the inverse of its transpose, over and over, converges to the closest true rotation
   *  (a matrix whose columns are perpendicular unit vectors). */
  function nearestRotation(M) {
    let R = M.slice();
    for (let i = 0; i < 30; i++) {
      const Ri = inv3(transpose3(R));
      if (!Ri) break;
      let change = 0;
      const N = R.map((v, k) => 0.5 * (v + Ri[k]));
      for (let k = 0; k < 9; k++) change = Math.max(change, Math.abs(N[k] - R[k]));
      R = N;
      if (change < 1e-12) break;
    }
    return R;
  }

  /* ---------------------------------------------------------------- camera */
  /** Pinhole camera numbers for a w x h video, assuming a horizontal field of view (default PARAMS.hfov degrees).
   *  focal length in pixels = (width/2) / tan(fov/2). Same as card_detect.intrinsics. */
  function intrinsics(w, h, hfovDeg) {
    const f = (w / 2) / Math.tan(((hfovDeg === undefined ? PARAMS.hfov : hfovDeg) * Math.PI) / 360);
    return { fx: f, fy: f, cx: w / 2, cy: h / 2, w, h };
  }

  /** three.js projection matrix (column-major, for Matrix4.fromArray) matching the pinhole camera K.
   *  Normally three.js builds this from a "fov" angle; we build it from K instead so the 3-D drawing uses the
   *  exact same lens as our pose maths, and we can place the principal point off-centre if needed. near/far are
   *  the clipping distances (things closer than near or farther than far are not drawn). */
  function projectionThree(K, near, far) {
    const { fx, fy, cx, cy, w, h } = K;
    const a = (far + near) / (far - near), b = (2 * far * near) / (far - near);
    return [2 * fx / w, 0, 0, 0,
            0, 2 * fy / h, 0, 0,
            1 - 2 * cx / w, 2 * cy / h - 1, -a, -1,
            0, 0, -b, 0];
  }

  /* ---------------------------------------------------------------- pose of a card from its 4 corners */
  // The card's four corners in card-local coordinates (origin at the centre, 1 unit = card width): TL, TR, BR, BL.
  const RECT = [[-0.5, -CARD_H / 2], [0.5, -CARD_H / 2], [0.5, CARD_H / 2], [-0.5, CARD_H / 2]];

  /**
   * corners = [TL, TR, BR, BL] in pixels (TL->TR is the card WIDTH). Returns {R, t, residual} or null.
   * R (row-major 3x3) and t map card-local (OpenCV style) -> camera. The two axis vectors of the card plane
   * must come out perpendicular and equally long; the sum of those errors is `residual` (~0 for a card,
   * large for the wrong width/height assignment or a non-card shape; unlike an aspect-ratio test it stays
   * valid when the card is tilted away from the camera).
   */
  function poseFromCorners(corners, K) {
    const H = homography(RECT, corners);
    if (!H) return null;
    const col = (j) => [(H[j] - K.cx * H[6 + j]) / K.fx, (H[3 + j] - K.cy * H[6 + j]) / K.fy, H[6 + j]];   // K^-1 * H[:, j]
    const a1 = col(0), a2 = col(1), a3 = col(2);
    const n1 = norm(a1), n2 = norm(a2);
    if (n1 < 1e-9 || n2 < 1e-9) return null;
    const residual = Math.abs(n1 - n2) / ((n1 + n2) / 2) + Math.abs(dot(a1, a2)) / (n1 * n2);
    let s = 2 / (n1 + n2);
    if (a3[2] * s < 0) s = -s;                                    // the card must be in front of the camera
    const r1 = scale(a1, s), r2 = scale(a2, s), t = scale(a3, s), r3 = cross(r1, r2);
    const R = nearestRotation([r1[0], r2[0], r3[0], r1[1], r2[1], r3[1], r1[2], r2[2], r3[2]]);
    return { R, t, residual };
  }

  /** 4x4 column-major matrix for three.js: card-local three frame -> three camera frame.
   *  OpenCV's camera has y pointing DOWN and z forward; three.js has y UP and looks down -z. D = diag(1,-1,-1)
   *  flips between them. The card's own axes are re-labelled so that three.js's +y points OUT of the card
   *  (models stand up from it) and +z points toward the card's bottom edge (a model facing +z looks at the player).
   *  The matrix columns are exactly "where do the card's x, y, z axes and origin end up in camera space?". */
  function modelViewThree(pose) {
    const { R, t } = pose, D = [1, -1, -1];
    const c0 = [R[0], R[3], R[6]], c1 = [R[1], R[4], R[7]], c2 = [R[2], R[5], R[8]];       // columns of R
    const x = c0.map((v, i) => v * D[i]);                                                    // card X  (right)
    const y = c2.map((v, i) => -v * D[i]);                                                   // card OUT of the face (-Z_cv)
    const z = c1.map((v, i) => v * D[i]);                                                    // card toward the bottom edge
    return [x[0], x[1], x[2], 0, y[0], y[1], y[2], 0, z[0], z[1], z[2], 0, t[0] * D[0], t[1] * D[1], t[2] * D[2], 1];
  }

  /** Project a card-local three.js point through modelView + projection -> pixel (for tests / labels). */
  function projectThree(mv, proj, p, w, h) {
    const m = (M, v) => [0, 1, 2, 3].map((r) => M[r] * v[0] + M[4 + r] * v[1] + M[8 + r] * v[2] + M[12 + r] * v[3]);
    const c = m(proj, m(mv, [p[0], p[1], p[2], 1]));
    return [((c[0] / c[3] + 1) / 2) * w, ((1 - c[1] / c[3]) / 2) * h];
  }

  /* ---------------------------------------------------------------- quad ordering */
  /** Sort 4 points clockwise on screen by angle around their centroid (same as card_detect.clockwise). */
  function clockwise(pts) {
    const cx = pts.reduce((s, p) => s + p[0], 0) / pts.length, cy = pts.reduce((s, p) => s + p[1], 0) / pts.length;
    return pts.map((p) => ({ p, a: Math.atan2(p[1] - cy, p[0] - cx) })).sort((u, v) => u.a - v.a).map((o) => o.p);   // y down => clockwise
  }
  const rotateList = (pts, k) => pts.map((_, i) => pts[(i + k) % pts.length]);          // like numpy.roll(pts, -k)
  const sumSq = (a, b) => a.reduce((s, p, i) => s + (p[0] - b[i][0]) ** 2 + (p[1] - b[i][1]) ** 2, 0);

  /**
   * Name the corners [TL, TR, BR, BL]. opts.prev = the same card's corners in the previous frame: the
   * naming that stays closest to it wins, so the card keeps its orientation while it moves. Without
   * `prev`: which sides are the card WIDTH is decided by the pose residual, and the TOP is the side higher
   * up in the image (a plain yellow border looks identical turned 180 degrees); opts.flip turns that 180.
   * Returns {corners, residual}.
   */
  function orientQuad(pts, K, opts) {
    opts = opts || {};
    const cw = clockwise(pts);
    const residualOf = (q) => { const p = poseFromCorners(q, K); return p ? p.residual : Infinity; };
    if (opts.prev) {
      let best = null;
      for (let k = 0; k < 4; k++) {
        const q = rotateList(cw, k), d = sumSq(q, opts.prev);
        if (!best || d < best.d) best = { q, d };
      }
      const residual = residualOf(best.q);
      if (residual <= (opts.maxResidual || PARAMS.maxResidual)) return { corners: best.q, residual };
    }
    const hyp = [0, 1].map((k) => { const q = rotateList(cw, k); return { q, residual: residualOf(q) }; })
      .sort((a, b) => a.residual - b.residual)[0];
    let q = hyp.q;
    if ((q[0][1] + q[1][1]) / 2 > (q[2][1] + q[3][1]) / 2) q = rotateList(q, 2);
    if (opts.flip) q = rotateList(q, 2);
    return { corners: q, residual: hyp.residual };
  }

  /* ---------------------------------------------------------------- finding the sides of a card in a contour */
  // Total least squares line through n points: the line direction is the principal axis of the point cloud.
  // 0.5 * atan2(2 sxy, sxx - syy) is the closed-form angle of that axis from the covariance entries.
  function fitLine(xs, ys, n) {
    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += xs[i]; my += ys[i]; }
    mx /= n; my /= n;
    let sxx = 0, sxy = 0, syy = 0;
    for (let i = 0; i < n; i++) { const dx = xs[i] - mx, dy = ys[i] - my; sxx += dx * dx; sxy += dx * dy; syy += dy * dy; }
    const th = 0.5 * Math.atan2(2 * sxy, sxx - syy);
    return { p: [mx, my], d: [Math.cos(th), Math.sin(th)] };
  }
  // Where two lines (point p + t * direction d) cross: solve p1 + t d1 = p2 + s d2 for t with 2-D cross products.
  function intersect(l1, l2) {
    const den = l1.d[0] * l2.d[1] - l1.d[1] * l2.d[0];
    if (Math.abs(den) < 1e-6) return null;
    const t = ((l2.p[0] - l1.p[0]) * l2.d[1] - (l2.p[1] - l1.p[1]) * l2.d[0]) / den;
    return [l1.p[0] + t * l1.d[0], l1.p[1] + t * l1.d[1]];
  }

  /**
   * Sharpen the 4 corners: least-squares line along each side of the contour, neighbours intersected.
   * contour = flat [x0, y0, x1, y1, ...]; quad = 4 points in order around the card (approxPolyDP).
   * Band around each side is relative to its length and tightened in 3 passes (see card_detect.refine_quad).
   */
  function refineQuad(contour, quad, P) {
    P = P || PARAMS;
    const N = contour.length / 2, xs = new Float64Array(N), ys = new Float64Array(N);
    for (let i = 0; i < N; i++) { xs[i] = contour[2 * i]; ys[i] = contour[2 * i + 1]; }
    const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    const lines = [], lens = [], px = new Float64Array(N), py = new Float64Array(N);
    for (let i = 0; i < 4; i++) {
      const a = quad[i], b = quad[(i + 1) % 4], dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy);
      const ux = dx / len, uy = dy / len;
      let line = { p: a, d: [ux, uy] };
      const widths = [Math.max(P.edgeTol, 0.03 * len), Math.max(P.edgeTol, 0.012 * len), Math.max(0.6 * P.edgeTol, 0.006 * len)];
      for (const width of widths) {
        let n = 0;
        for (let k = 0; k < N; k++) {
          const s = ((xs[k] - a[0]) * ux + (ys[k] - a[1]) * uy) / len;                       // 0..1 along the side
          if (s <= P.edgeSkip || s >= 1 - P.edgeSkip) continue;                              // skip rounded corners
          const off = (xs[k] - line.p[0]) * -line.d[1] + (ys[k] - line.p[1]) * line.d[0];    // distance to current line
          if (Math.abs(off) <= width) { px[n] = xs[k]; py[n] = ys[k]; n++; }
        }
        if (n < 8) return quad.map((p) => p.slice());                                        // not enough evidence
        line = fitLine(px, py, n);
      }
      let nx = -line.d[1], ny = line.d[0];
      if (nx * (line.p[0] - cx) + ny * (line.p[1] - cy) < 0) { nx = -nx; ny = -ny; }         // outward normal
      lines.push({ p: [line.p[0] + 0.5 * nx, line.p[1] + 0.5 * ny], d: line.d });            // contour pixels sit half a pixel inside the edge
      lens.push(len);
    }
    const limit = Math.max(6, 0.05 * (lens[0] + lens[1] + lens[2] + lens[3]) / 4);
    return quad.map((q, i) => {
      const x = intersect(lines[(i + 3) % 4], lines[i]);
      return !x || Math.hypot(x[0] - q[0], x[1] - q[1]) > limit ? q.slice() : x;
    });
  }

  /** Fraction of samples along the four sides (a little inside the edge) that are yellow in the mask. */
  function edgeCover(mask, w, h, quad) {
    const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    let hits = 0, total = 0;
    for (let i = 0; i < 4; i++) {
      const a = quad[i], b = quad[(i + 1) % 4], len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      let ix = cx - (a[0] + b[0]) / 2, iy = cy - (a[1] + b[1]) / 2;
      const il = Math.hypot(ix, iy) || 1;
      ix /= il; iy /= il;
      const depth = Math.max(1.5, 0.025 * len);
      for (let k = 0; k < 20; k++) {
        const t = 0.1 + (0.8 * k) / 19;
        const x = Math.round(a[0] + (b[0] - a[0]) * t + ix * depth), y = Math.round(a[1] + (b[1] - a[1]) * t + iy * depth);
        let hit = false;
        for (let yy = Math.max(y - 1, 0); yy <= Math.min(y + 1, h - 1) && !hit; yy++)
          for (let xx = Math.max(x - 1, 0); xx <= Math.min(x + 1, w - 1); xx++) if (mask[yy * w + xx]) { hit = true; break; }
        hits += hit; total++;
      }
    }
    return hits / total;
  }

  /** Fraction of samples just OUTSIDE the four sides that are NOT border-coloured (1 = clean edge all round). The border is the
   *  outermost ring of a card; a frame INSIDE the card has more card colour around it and scores low. Mirrors card_detect.outer_clear. */
  function outerClear(mask, w, h, quad) {
    const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    let clear = 0, total = 0;
    for (let i = 0; i < 4; i++) {
      const a = quad[i], b = quad[(i + 1) % 4], len = Math.hypot(b[0] - a[0], b[1] - a[1]);
      let ox = (a[0] + b[0]) / 2 - cx, oy = (a[1] + b[1]) / 2 - cy;
      const ol = Math.hypot(ox, oy) || 1;
      ox /= ol; oy /= ol;
      const depth = Math.max(2.5, 0.03 * len);
      for (let k = 0; k < 20; k++) {
        const t = 0.1 + (0.8 * k) / 19;
        const x = Math.round(a[0] + (b[0] - a[0]) * t + ox * depth), y = Math.round(a[1] + (b[1] - a[1]) * t + oy * depth);
        if (x >= 0 && x < w && y >= 0 && y < h) {
          let any = false;
          for (let yy = Math.max(y - 1, 0); yy <= Math.min(y + 1, h - 1) && !any; yy++)
            for (let xx = Math.max(x - 1, 0); xx <= Math.min(x + 1, w - 1); xx++) if (mask[yy * w + xx]) { any = true; break; }
          if (!any) clear++;
        } else clear++;
        total++;
      }
    }
    return clear / total;
  }

  /** Is this quad really an INNER frame of a bigger card? If so, rebuild the card's OUTER border from the mask.
   *  From each side we scan outward along its normal and keep the outermost border-coloured pixel within reach (70% of the side);
   *  those edge points, at 9 places along the side, are fitted with a line. Returns
   *    {status: "none"}            fewer than 3 sides have a ring around them: this quad IS the outermost border
   *    {status: "lost"}            a ring surrounds it but cannot be rebuilt (a whole side hidden): ignore this quad
   *    {status: "ok", quad}        the outer border, 4 corners in the same order as `quad`
   *  Mirrors card_detect.outer_from_inner. */
  function outerFromInner(mask, w, h, quad) {
    const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    const lines = [];
    let sides = 0;
    for (let i = 0; i < 4; i++) {
      const a = quad[i], b = quad[(i + 1) % 4], L = Math.hypot(b[0] - a[0], b[1] - a[1]);
      let nx = (a[0] + b[0]) / 2 - cx, ny = (a[1] + b[1]) / 2 - cy;
      const nl = Math.hypot(nx, ny) || 1;
      nx /= nl; ny /= nl;                                           // unit normal pointing away from the card centre
      const d0 = Math.max(3, 0.04 * L), reach = 0.7 * L;
      let ts = [], os = [];
      for (let k = 0; k < 9; k++) {
        const t = 0.1 + 0.1 * k, px = a[0] + (b[0] - a[0]) * t, py = a[1] + (b[1] - a[1]) * t;
        let last = -1;
        for (let o = d0; o < reach; o += 1) {
          const x = Math.round(px + nx * o), y = Math.round(py + ny * o);
          if (x >= 0 && x < w && y >= 0 && y < h && mask[y * w + x]) last = o;
        }
        if (last >= 0) { ts.push(t); os.push(last); }
      }
      if (ts.length < 4) { lines.push(null); continue; }
      const fit = (T, O) => {                                       // least squares o = c1 * t + c0
        const n = T.length, st = T.reduce((s, v) => s + v, 0), so = O.reduce((s, v) => s + v, 0);
        const stt = T.reduce((s, v) => s + v * v, 0), sto = T.reduce((s, v, j) => s + v * O[j], 0);
        const den = n * stt - st * st, c1 = den ? (n * sto - st * so) / den : 0;
        return [c1, (so - c1 * st) / n];
      };
      const tol = Math.max(2, 0.012 * L);
      let [c1, c0] = fit(ts, os);
      const good = ts.map((t, j) => Math.abs(c1 * t + c0 - os[j]) <= tol);
      const ng = good.filter(Boolean).length;
      if (ng >= 4 && ng < ts.length) { ts = ts.filter((_, j) => good[j]); os = os.filter((_, j) => good[j]); [c1, c0] = fit(ts, os); }
      if (ts.length < 4 || Math.max(...ts.map((t, j) => Math.abs(c1 * t + c0 - os[j]))) > tol) { lines.push(null); continue; }
      sides++;
      const p0 = [a[0] + nx * c0, a[1] + ny * c0], p1 = [b[0] + nx * (c1 + c0), b[1] + ny * (c1 + c0)];
      lines.push({ p: p0, d: [p1[0] - p0[0], p1[1] - p0[1]] });
    }
    if (sides < 3) return { status: "none" };
    if (sides < 4) return { status: "lost" };
    const out = [];
    for (let i = 0; i < 4; i++) {
      const l1 = lines[(i + 3) % 4], l2 = lines[i];
      const den = l1.d[0] * l2.d[1] - l1.d[1] * l2.d[0];
      if (Math.abs(den) < 1e-6) return { status: "lost" };
      const t = ((l2.p[0] - l1.p[0]) * l2.d[1] - (l2.p[1] - l1.p[1]) * l2.d[0]) / den;
      out.push([l1.p[0] + t * l1.d[0], l1.p[1] + t * l1.d[1]]);
    }
    return out.every((p) => Number.isFinite(p[0]) && Number.isFinite(p[1])) ? { status: "ok", quad: out } : { status: "lost" };
  }

  /** Fraction of the card's INSIDE (middle 60% x 60%, 9x9 samples) that is in the mask. A bordered card has a thin ring around
   *  different colours (small value); a solid gray slab is gray throughout (large value). Mirrors card_detect.interior_fraction. */
  function interiorFraction(mask, w, h, quad) {
    let hit = 0, n = 0;
    for (let j = 0; j < 9; j++) for (let i = 0; i < 9; i++) {
      const u = 0.2 + (0.6 * i) / 8, v = 0.2 + (0.6 * j) / 8, [tl, tr, br, bl] = quad;
      const x = Math.round((1 - v) * ((1 - u) * tl[0] + u * tr[0]) + v * ((1 - u) * bl[0] + u * br[0]));
      const y = Math.round((1 - v) * ((1 - u) * tl[1] + u * tr[1]) + v * ((1 - u) * bl[1] + u * br[1]));
      if (x >= 0 && x < w && y >= 0 && y < h) { n++; if (mask[y * w + x]) hit++; }
    }
    return n ? hit / n : 1;
  }

  /** Drop detections that are the same card found twice (e.g. a yellow ring and a gray one). `dets` need {quad, score};
   *  best score wins. Mirrors card_detect.dedupe. */
  function dedupe(dets) {
    const size = (q) => Math.max(Math.max(...q.map((p) => p[0])) - Math.min(...q.map((p) => p[0])), Math.max(...q.map((p) => p[1])) - Math.min(...q.map((p) => p[1])));
    const ctr = (q) => [q.reduce((s, p) => s + p[0], 0) / 4, q.reduce((s, p) => s + p[1], 0) / 4];
    const kept = [];
    for (const d of [...dets].sort((a, b) => b.score - a.score)) {
      const c = ctr(d.quad);
      if (!kept.some((k) => { const kc = ctr(k.quad); return Math.hypot(c[0] - kc[0], c[1] - kc[1]) < PARAMS.dupDist * Math.max(size(d.quad), size(k.quad)); })) kept.push(d);
    }
    return kept;
  }

  /** Drop detections lying completely inside a bigger one (an inner frame of the same card, not a second card). Mirrors
   *  card_detect.suppress_inner. `dets` need {quad}; order is kept. */
  function suppressInner(dets) {
    const inside = (p, q, m = 2) => {                              // inside a convex quad, with m px of slack
      const wind = Math.sign(q.reduce((s, a, i) => s + a[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * a[1], 0)) || 1;   // winding of the quad
      for (let i = 0; i < 4; i++) {
        const a = q[i], b = q[(i + 1) % 4], len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
        if ((wind * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))) / len < -m) return false;   // more than m px outside this side
      }
      return true;
    };
    const areas = dets.map((d) => polyArea(d.quad));
    return dets.filter((d, i) => !dets.some((o, j) => j !== i && areas[j] > areas[i] * 1.03 && areas[i] >= PARAMS.containRatio * areas[j] && d.quad.every((p) => inside(p, o.quad))));
  }

  /* ---------------------------------------------------------------- MASK REPAIR helpers (twins of card_detect.py)
   * A card's outline (edges) is steadier than any colour window, so outlines found from edges get their border strips painted
   * into the colour masks when >= ringSides of the 4 strips just inside the outline are clearly yellow / silver. */
  /** 'yellow' | 'gray' | null for a strip of median HSV (OpenCV scale: H 0-179, S/V 0-255). */
  function sideClass(h, s, v, P = PARAMS) {
    if (h >= P.ringYellowH[0] && h <= P.ringYellowH[1] && s >= P.ringYellowSMin && v >= P.ringYellowVMin) return "yellow";
    if (s <= P.ringGraySMax && v >= P.ringGrayVMin) return "gray";
    return null;
  }
  /** The strip just INSIDE side i (corner i -> i+1) of a convex quad. along = [from, to] fraction of the side's length, depth = [from, to] in units of `thick` px. */
  function innerStrip(quad, i, thick, along = [0.25, 0.75], depth = [0.25, 0.8]) {
    const a = quad[i], b = quad[(i + 1) % 4], cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    const dx = b[0] - a[0], dy = b[1] - a[1];
    let nx = -dy, ny = dx; const l = Math.hypot(nx, ny) || 1; nx /= l; ny /= l;
    if ((cx - (a[0] + b[0]) / 2) * nx + (cy - (a[1] + b[1]) / 2) * ny < 0) { nx = -nx; ny = -ny; }          // n points into the card
    const p0 = [a[0] + along[0] * dx, a[1] + along[0] * dy], p1 = [a[0] + along[1] * dx, a[1] + along[1] * dy];
    return [[p0[0] + depth[0] * thick * nx, p0[1] + depth[0] * thick * ny], [p1[0] + depth[0] * thick * nx, p1[1] + depth[0] * thick * ny],
            [p1[0] + depth[1] * thick * nx, p1[1] + depth[1] * thick * ny], [p0[0] + depth[1] * thick * nx, p0[1] + depth[1] * thick * ny]];
  }
  /** The quad with every side pushed `d` px outwards (scaled about its centre). */
  function expandQuad(quad, d) {
    const cx = quad.reduce((s, p) => s + p[0], 0) / 4, cy = quad.reduce((s, p) => s + p[1], 0) / 4;
    const k = 1 + d / Math.max(Math.hypot(quad[0][0] - cx, quad[0][1] - cy), 1e-6);
    return quad.map((p) => [cx + (p[0] - cx) * k, cy + (p[1] - cy) * k]);
  }
  /** Andrew's monotone-chain convex hull of [[x,y],...] (returns the hull in order). */
  function convexHull(pts) {
    const p = [...pts].sort((a, b) => a[0] - b[0] || a[1] - b[1]);
    if (p.length < 3) return p;
    const cr = (o, a, b) => (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0]);
    const lo = [], up = [];
    for (const q of p) { while (lo.length >= 2 && cr(lo[lo.length - 2], lo[lo.length - 1], q) <= 0) lo.pop(); lo.push(q); }
    for (const q of [...p].reverse()) { while (up.length >= 2 && cr(up[up.length - 2], up[up.length - 1], q) <= 0) up.pop(); up.push(q); }
    return lo.slice(0, -1).concat(up.slice(0, -1));
  }
  /** Is point p inside convex quad q (m px slack)? Same test suppressInner uses. */
  function insideQuad(p, q, m = 2) {
    const wind = Math.sign(q.reduce((s, a, i) => s + a[0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * a[1], 0)) || 1;
    for (let i = 0; i < 4; i++) {
      const a = q[i], b = q[(i + 1) % 4], len = Math.hypot(b[0] - a[0], b[1] - a[1]) || 1;
      if ((wind * ((b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0]))) / len < -m) return false;
    }
    return true;
  }
  /** After a repaired pass: drop detections that have a (bigger) painted outline around them - they are inner frames. `painted` = quads in the same coordinates as dets[].quad. */
  function dropInsidePainted(dets, painted) {
    return dets.filter((d) => !painted.some((o) => polyArea(o) > 1.04 * polyArea(d.quad) && d.quad.every((p) => insideQuad(p, o))));
  }

  /** Combine the plain pass with the edge-repaired pass (twin of card_detect.merge_passes): a repaired detection that surrounds a plain one
   *  replaces it (the plain one was an inner frame); one that is the same card is dropped (the plain one is more accurate); else it is added. */
  function mergeRepaired(plain, repaired) {
    const size = (q) => Math.max(Math.max(...q.map((p) => p[0])) - Math.min(...q.map((p) => p[0])), Math.max(...q.map((p) => p[1])) - Math.min(...q.map((p) => p[1])));
    const ctr = (q) => [q.reduce((s, p) => s + p[0], 0) / 4, q.reduce((s, p) => s + p[1], 0) / 4];
    let out = [...plain];
    for (const r of repaired) {
      const ra = polyArea(r.quad), around = out.filter((p) => ra > PARAMS.replaceRatio * polyArea(p.quad) && p.quad.every((c) => insideQuad(c, r.quad)));
      if (around.length) out = out.filter((p) => !around.includes(p)).concat([r]);
      else {
        const rc = ctr(r.quad);
        if (!out.some((p) => { const pc = ctr(p.quad); return Math.hypot(rc[0] - pc[0], rc[1] - pc[1]) < PARAMS.dupDist * Math.max(size(r.quad), size(p.quad)); })) out.push(r);
      }
    }
    return out.sort((a, b) => b.score - a.score);
  }

  const polyArea = (q) => Math.abs(q.reduce((s, p, i) => s + p[0] * q[(i + 1) % q.length][1] - q[(i + 1) % q.length][0] * p[1], 0)) / 2;
  function isConvex(q) {
    let sign = 0;
    for (let i = 0; i < q.length; i++) {
      const a = q[i], b = q[(i + 1) % q.length], c = q[(i + 2) % q.length];
      const z = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
      if (z !== 0) { if (sign && Math.sign(z) !== sign) return false; sign = Math.sign(z); }
    }
    return true;
  }

  /**
   * Second half of detection, for ONE candidate from the OpenCV step ({contour, approx, hullArea}):
   * shape checks, line-fitted corners, yellow-edge check. Returns {quad, fill, cover} (working-res pixels,
   * not yet named) or null. Mirrors the body of card_detect.detect_cards.
   */
  function candidateQuad(cand, mask, w, h, P) {
    P = P || PARAMS;
    if (!isConvex(cand.approx)) return null;
    const fill = polyArea(cand.approx) / cand.hullArea;
    if (fill < P.minFill) return null;                                    // not 4-sided (e.g. a yellow ball)
    let quad = refineQuad(cand.contour, cand.approx, P);
    const outer = outerFromInner(mask, w, h, quad);                        // a bigger ring of the same colour around it? then this is an inner frame
    if (outer.status === "lost") return null;                              // ... whose outer border can't be rebuilt right now: never track the inside
    if (outer.status === "ok") quad = outer.quad;
    const cover = edgeCover(mask, w, h, quad);
    if (cover < P.minCover) return null;
    if (outerClear(mask, w, h, quad) < P.outerClearMin) return null;       // more border colour outside it: an inner frame, not the OUTER border
    if (cand.color === "gray" && interiorFraction(mask, w, h, quad) > P.maxGrayInterior) return null;   // solid gray object, not a gray border
    return { quad, fill, cover, color: cand.color || "yellow" };
  }

  /* ---------------------------------------------------------------- smoothing + tracking */
  /** One Euro filter (Casiez et al.): heavy smoothing while still, light when moving fast.
   *  It is an exponential low-pass filter  x += alpha * (measurement - x)  whose cutoff frequency rises with the
   *  measured speed:  cutoff = minCutoff + beta * |speed|.  Low cutoff = smooth but laggy; high = responsive but
   *  jittery - so the filter picks "smooth" at rest and "responsive" in motion. alpha(cutoff, dt) converts a cutoff
   *  frequency into the per-frame blend factor. Tuned values: minCutoff 1.0, beta 0.1 (see card-tracker.js). */
  class OneEuro {
    constructor(minCutoff = 1.5, beta = 0.02, dCutoff = 1.0) { Object.assign(this, { minCutoff, beta, dCutoff }); this.x = undefined; this.dx = 0; }
    static alpha(cutoff, dt) { return 1 / (1 + 1 / (2 * Math.PI * cutoff) / dt); }
    filter(v, dt) {
      if (this.x === undefined) { this.x = v; return v; }
      const d = (v - this.x) / dt;
      this.dx += OneEuro.alpha(this.dCutoff, dt) * (d - this.dx);
      this.x += OneEuro.alpha(this.minCutoff + this.beta * Math.abs(this.dx), dt) * (v - this.x);
      return this.x;
    }
  }

  /** PoseFilter - the second smoothing stage, run on every drawn frame (the first is the One Euro filter on the corners).
   *  It glides a pose (position {x,y,z} + rotation {x,y,z,w}) toward the tracker's latest pose. The time constant `tau`
   *  depends on how far away the target is: a tiny error is camera noise, so it is followed slowly (tau ~ tauRest = steady
   *  picture); a big error is real movement, so it is followed fast (tau -> tauMove = no lag). Error = card-widths of
   *  position + half the rotation angle in radians. Works on THREE.Vector3/Quaternion or plain objects, in place. */
  class PoseFilter {
    constructor(o) { Object.assign(this, { tauRest: 0.22, tauMove: 0.035, scale: 0.06 }, o || {}); }
    /** error between pose (p,q) and target (tp,tq) */
    error(p, q, tp, tq) {
      const dot = Math.min(1, Math.abs(q.x * tq.x + q.y * tq.y + q.z * tq.z + q.w * tq.w));
      return Math.hypot(tp.x - p.x, tp.y - p.y, tp.z - p.z) + Math.acos(dot);      // acos(|dot|) = half the rotation angle
    }
    step(p, q, tp, tq, dt) {
      const e = this.error(p, q, tp, tq);
      const tau = this.tauMove + (this.tauRest - this.tauMove) * Math.exp(-e / this.scale);
      const a = 1 - Math.exp(-Math.max(0, dt) / tau);
      p.x += (tp.x - p.x) * a; p.y += (tp.y - p.y) * a; p.z += (tp.z - p.z) * a;
      // shortest-path slerp (normalised lerp is accurate enough for the small steps taken per frame)
      const sgn = q.x * tq.x + q.y * tq.y + q.z * tq.z + q.w * tq.w < 0 ? -1 : 1;
      let x = q.x + (sgn * tq.x - q.x) * a, y = q.y + (sgn * tq.y - q.y) * a, z = q.z + (sgn * tq.z - q.z) * a, w = q.w + (sgn * tq.w - q.w) * a;
      const n = Math.hypot(x, y, z, w) || 1;
      q.x = x / n; q.y = y / n; q.z = z / n; q.w = w / n;
      return e;
    }
  }

  /** Greedy nearest-centre matching of detections to tracks. Returns {pairs: [[track, det]...], newDets, lost}.
   *  List every (track, detection) pair with its distance, sort by distance, then accept pairs closest-first as long
   *  as neither side is already taken and the distance is under maxDist. Left-over detections are NEW cards; left-over
   *  tracks are cards that were not seen this frame. (Simple and fine for <= 2 cards.) */
  function associate(trackCentres, detCentres, maxDist) {          // maxDist: one number, or one limit per track (array)
    const all = [];
    trackCentres.forEach((t, i) => detCentres.forEach((d, j) => all.push([Math.hypot(t[0] - d[0], t[1] - d[1]), i, j])));
    all.sort((a, b) => a[0] - b[0]);
    const usedT = new Set(), usedD = new Set(), pairs = [];
    for (const [dist, i, j] of all) {
      if (usedT.has(i) || usedD.has(j)) continue;
      if (dist > (Array.isArray(maxDist) ? maxDist[i] : maxDist)) continue;       // too far for THIS track (others may still match)
      usedT.add(i); usedD.add(j); pairs.push([i, j]);
    }
    return {
      pairs,
      newDets: detCentres.map((_, j) => j).filter((j) => !usedD.has(j)),
      lost: trackCentres.map((_, i) => i).filter((i) => !usedT.has(i)),
    };
  }

  return { CARD_H, PARAMS, thresholds, grayThresholds, interiorFraction, outerClear, outerFromInner, suppressInner, dedupe, homography, nearestRotation, inv3, intrinsics, projectionThree, poseFromCorners,
           modelViewThree, projectThree, clockwise, rotateList, orientQuad, refineQuad, edgeCover, polyArea, isConvex,
           candidateQuad, OneEuro, PoseFilter, associate, sideClass, innerStrip, expandQuad, convexHull, insideQuad, dropInsidePainted, mergeRepaired };
});
