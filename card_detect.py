"""
card_detect.py - find yellow- or gray-bordered Pokémon cards with OpenCV (Python) and straighten them.
=============================================================================================

THE BIG IDEA (read this first)
------------------------------
A Pokémon card has a bright yellow border. Yellow is rare on a desk, so "find the yellow rectangle"
is a cheap, fast, robust way to find a card in a camera picture - no neural network needed.
Some cards have a GRAY (silver) border instead. Gray is common on a desk, so it is searched with extra care: it gets its
own mask (low colour saturation, mid brightness), must pass the same shape / edge tests, AND the card's inside must not be
gray too (interior_fraction) - which rejects laptops, phones and mouse pads. A gray card on a gray table of nearly the same
shade cannot be separated; use a darker or lighter surface.
Once we know where the four corners are in the picture we can:
   * straighten the card (a "perspective warp") so a recogniser sees it face-on, and
   * work out how the card is tilted in 3-D, so AR can draw a model standing on it (that part
     lives in static/js/card-geometry.js; the maths is explained there).

Where this file is used
  * app.py /api/battle/scan - the browser sends a camera frame plus the card corners its live
    tracker found; this module re-measures those corners on the full-resolution frame and warps
    the card into an upright rectangle, so the recognizer gets a clean, straight card instead
    of a tilted photo (faster and more accurate than recognising the raw frame).
  * command line - draw what was detected / paste a sprite on the card, to try your own photos:
        python card_detect.py photo.jpg -o out.jpg --crop card.jpg --sprite sprites/025_pikachu.jpg

The browser tracker (static/js/card-tracker.js, OpenCV.js) follows the same steps with the same
constants - if you change PARAMS or thresholds() here, change them there too:

    BGR -> HSV -> inRange(yellow) -> close -> findContours(external) -> convexHull -> approxPolyDP
        -> 4 corners -> refine each side with a least-squares line fit -> intersect the lines

Fitting lines to the edges (instead of trusting approxPolyDP's vertices) matters: cards have
rounded corners, so the polygon vertices sit inside the true corners, and one pixel of error at
the low working resolution becomes several pixels on screen.

VOCABULARY
----------
BGR     OpenCV stores colour pictures as Blue-Green-Red (not RGB).
HSV     Hue-Saturation-Value. Hue = "which colour" (yellow is ~25-30 on OpenCV's 0-179 scale), so a
        colour window in HSV survives brighter/darker light much better than one in RGB.
mask    a black/white picture: white where the pixel is "yellow enough".
contour the list of pixels that outline a white blob.
hull    the "rubber band" around a contour (convex hull) - ignores dents where a finger covers the border.
corners we always name them TL, TR, BR, BL = top-left, top-right, bottom-right, bottom-left of the CARD
        (not of the picture), going clockwise.
"""
import argparse
from dataclasses import dataclass

import cv2
import numpy as np

CARD_W, CARD_H = 63.0, 88.0              # official card size in mm
CARD_ASPECT = CARD_W / CARD_H            # ~0.716: a card is a bit more than 1.4x taller than wide

# All tunable numbers in one place. The JavaScript twin (card-geometry.js -> PARAMS) has the same values.
PARAMS = dict(
    work_width=480,                      # frames are shrunk to this width before analysis (speed)
    min_area=0.006,                      # smallest accepted card, as a fraction of the frame
    max_area=0.85,                       # biggest accepted card (anything bigger is probably not a card)
    eps_sweep=(0.02, 0.03, 0.045, 0.06, 0.08),   # approxPolyDP epsilon / hull perimeter, tried in order
    min_cover=0.55,                      # fraction of the card's edge that must be yellow
    min_fill=0.9,                        # 4-corner polygon area / hull area (rejects round yellow things)
    edge_skip=0.12,                      # ignore this much at both ends of a side (rounded corners)
    edge_tol=2.5,                        # px: narrowest band around a side that counts as "on the side"
    max_residual=0.6,                    # how un-card-like a quad may look (see pose_residual)
    hfov=63.0,                           # assumed horizontal field of view of the camera, degrees
    border_colors=("yellow", "gray"),    # border colours looked for (each gets its own mask, so a gray card on a yellow mat can't merge)
    max_gray_interior=0.8,               # GRAY borders only: at most this fraction of the card's inside may itself be gray (rejects solid gray slabs)
    dup_dist=0.35,                       # two detections closer than this (x the larger card's size) are the same card seen in two colours
    outer_clear_min=0.5,                 # OUTER BORDER ONLY: at least this fraction of the band just OUTSIDE the card must NOT be border-coloured
    contain_ratio=0.2,                   # a candidate lying completely inside a bigger valid card (and >= this fraction of its area) is an inner frame, not a card
    merge_gap=0.012,                     # BROKEN RINGS: fragments closer than this fraction of the frame width are joined into one object (see merged_candidates)
    repair=True,                         # MASK REPAIR from edges on/off (see repair_masks)
    replace_ratio=1.15,                  # a repaired detection replaces a plain one only if it is this much bigger and surrounds it (else the plain one, being more accurate, stays)
    merge_min_px=5,                      # ... but never less than this many pixels (kernel size)
    canny_lo=40, canny_hi=110,           # MASK REPAIR: Canny edge thresholds (brightness steps) for finding card outlines
    ring_thick=0.035,                    # ... width of the colour strip examined just inside each side, as a fraction of the card's short side
    ring_expand=(1.0, 2.0),              # ... when the strips inside an outline are NOT border-coloured, try the outline grown outward by this many ring widths (weak outer edge: silver on tan wood)
    ring_sides=3,                        # ... this many of the 4 sides must have the same border colour for the outline to count
    ring_yellow_h=(14, 40), ring_yellow_smin=90, ring_yellow_vmin=110,   # ... a strip counts as yellow/gold when its MEDIAN colour is in here
    ring_gray_smax=70, ring_gray_vmin=100,                                # ... or as silver when its median saturation/brightness is
    size_gate=(0.6, 1.6),                # (browser tracker) a detection may be 0.6-1.6 x the size of the card it continues; anything else is not that card
)


def thresholds(sensitivity=0.5):
    """HSV window for the yellow border (OpenCV hue is 0-179, yellow ~ 25-30).
    sensitivity 0 = strict (fewer false hits), 1 = lenient (dim light, washed-out colours).

    Returns (lower, upper) where each is (Hue, Saturation, Value). A pixel is "yellow" when all three of
    its values lie between lower and upper. Raising sensitivity widens the hue range and lowers the
    minimum saturation/brightness, so pale or shadowed yellow still counts."""
    s = float(np.clip(sensitivity, 0, 1))
    r = lambda x: int(np.floor(x + 0.5))                          # round half up, same as JS Math.round
    return (r(21 - 4 * s), r(140 - 90 * s), r(150 - 80 * s)), (r(34 + 4 * s), 255, 255)


def gray_thresholds(sensitivity=0.5):
    """HSV window for a GRAY (silver) border: almost no colour (low saturation) and neither white nor black (mid brightness).
    Excluding bright whites (> ~215) and dark shadows (< ~100) keeps a card's cream face, black ink and the table's shadows
    out of the mask. sensitivity widens the window like it does for yellow. JS twin: card-geometry.js grayThresholds()."""
    s = float(np.clip(sensitivity, 0, 1))
    r = lambda x: int(np.floor(x + 0.5))
    return (0, 0, r(110 - 25 * s)), (179, r(28 + 22 * s), r(205 + 20 * s))


@dataclass
class Detection:
    """One card found in a picture."""
    corners: np.ndarray      # (4, 2) float32, order TL, TR, BR, BL, in ORIGINAL image pixels
    area: float              # fraction of the frame covered by the card
    cover: float             # 0-1: how much of the card's edge band is yellow
    residual: float          # 0 = perfect perspective view of a 63 x 88 mm rectangle (see pose_residual)
    color: str = "yellow"    # which border colour found it: "yellow" or "gray"

    @property
    def score(self):
        """Used to rank detections: big and clearly-yellow beats small and patchy."""
        return self.area * self.cover

    @property
    def center(self):
        return self.corners.mean(axis=0)


# ------------------------------------------------------------------ geometry helpers
def intrinsics(w, h, hfov=PARAMS["hfov"]):
    """Pinhole camera matrix K for a frame of w x h pixels (principal point at the centre).

    A phone camera behaves like a "pinhole": a 3-D point (X, Y, Z) lands on pixel (f*X/Z + cx, f*Y/Z + cy).
    K stores f (the focal length in pixels) and the centre (cx, cy). We do not know the real focal
    length, so we assume the camera sees `hfov` degrees across the picture's width; basic trigonometry
    then gives f = (w/2) / tan(hfov/2). (The Settings slider in battle mode changes hfov.)"""
    f = (w / 2) / np.tan(np.radians(hfov) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def pose_residual(quad, K):
    """How well `quad` (TL, TR, BR, BL; the TL->TR side is the card WIDTH) matches a perspective view
    of a 63 x 88 mm rectangle seen by camera K. Decomposing the homography of the card plane gives
    two axis vectors that must be perpendicular and equally long; the sum of those two errors is
    the residual (~0 for a card, large for the wrong width/height assignment or non-card shapes).
    Unlike a plain "aspect ratio" test this stays valid when the card is tilted away from the camera.

    A homography is the 3x3 matrix that maps one flat plane onto another. Here: the flat card
    (a unit-width rectangle) onto the quad seen in the picture. Multiplying by inverse(K) removes the
    camera, leaving [r1 r2 t] scaled - r1 and r2 are the card's x- and y-axes in camera space. For a real
    rigid card they are perpendicular (dot product 0) and the same length - that is exactly what we test."""
    rect = np.array([[0, 0], [1, 0], [1, CARD_H / CARD_W], [0, CARD_H / CARD_W]], np.float32)   # the card, width = 1
    A = np.linalg.inv(K) @ cv2.getPerspectiveTransform(rect, np.asarray(quad, np.float32))
    a1, a2 = A[:, 0], A[:, 1]                                    # the card's x- and y-axis directions
    n1, n2 = np.linalg.norm(a1), np.linalg.norm(a2)
    return float(abs(n1 - n2) / ((n1 + n2) / 2) + abs(a1 @ a2) / (n1 * n2))   # unequal length + not perpendicular


def clockwise(pts):
    """Sort 4 points clockwise (as seen on screen) by their angle around their centre."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    c = pts.mean(axis=0)
    return pts[np.argsort(np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0]))]    # y is down => clockwise


def orient_quad(pts, K, flip=False):
    """Name the corners: returns ((TL, TR, BR, BL) float32 (4,2), residual).
    Which pair of sides is the card's width is decided by pose_residual (robust to perspective);
    which end is the TOP is a guess - the side higher up in the image - because a plain yellow
    border looks the same turned 180 degrees. `flip` turns the answer 180 degrees.

    Four clockwise points can be labelled in four ways (which one is "TL"). Rolling by 2 only turns the
    card 180° (same width/height), so we only need to test two starting points (k = 0, 1): the one whose
    residual is lower has the correct width/height assignment."""
    pts = clockwise(pts)
    res, q = min(((pose_residual(np.roll(pts, -k, axis=0), K), np.roll(pts, -k, axis=0)) for k in (0, 1)),
                 key=lambda r: r[0])
    if q[:2, 1].mean() > q[2:, 1].mean():                        # "top" edge lower than "bottom" edge -> swap ends
        q = np.roll(q, -2, axis=0)
    return (np.roll(q, -2, axis=0) if flip else q), res


def match_order(pts, reference):
    """Re-order `pts` (4,2) cyclically so each corner lines up with the same corner of `reference`."""
    pts = np.asarray(pts, np.float32).reshape(4, 2)
    ref = np.asarray(reference, np.float32).reshape(4, 2)
    best = min(range(4), key=lambda k: np.linalg.norm(np.roll(pts, -k, axis=0) - ref))
    return np.roll(pts, -best, axis=0)


def _line(points):
    """Total-least-squares line through points -> (point, unit direction).

    "Total least squares" = minimise the perpendicular distances (not just vertical ones), which is what
    you want for a line that may be steep. The direction is the eigenvector of the point cloud's
    covariance matrix with the LARGEST eigenvalue (eigh sorts ascending, so index 1)."""
    m = points.mean(axis=0)
    w, v = np.linalg.eigh(np.cov((points - m).T))
    return m, v[:, 1]


def _intersect(l1, l2):
    """Where two lines (point + direction) cross; None if they are (nearly) parallel."""
    (p, d), (q, e) = l1, l2
    den = d[0] * e[1] - d[1] * e[0]                              # 2-D cross product of the directions
    if abs(den) < 1e-6:
        return None
    t = ((q[0] - p[0]) * e[1] - (q[1] - p[1]) * e[0]) / den
    return p + t * d


def refine_quad(contour, quad, skip=PARAMS["edge_skip"], tol=PARAMS["edge_tol"]):
    """Sharpen the 4 corners: fit a straight line to the contour points along each side and
    intersect neighbouring lines. Falls back to the approximate corner if a side has too few points.
    `contour` is (N,2) in the same pixel units as `quad`.
    The polygon from approxPolyDP can sit several pixels off the real edge (its vertices lie on the
    rounded corners), and "several" grows with the card's size, so the search band is relative to
    the side length (3 %) and is tightened over three passes around the line it just fitted.

    Walk-through for ONE side (a -> b):
      1. take every contour pixel whose position along the side is between `skip` and 1-`skip`
         (the middle 76% - rounded corners would bend the fit);
      2. keep those within a narrow band of the current line estimate; fit a new line; repeat 3x with a
         narrower band each time, so outliers (a thumb, a shadow) are shed;
      3. push the final line 0.5 px outward (see comment below).
    Then each corner is the intersection of its two neighbouring side lines."""
    quad = np.asarray(quad, np.float64).reshape(4, 2)
    contour = np.asarray(contour, np.float64).reshape(-1, 2)
    centre = quad.mean(axis=0)
    lines, side_len = [], []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        d = b - a
        length = np.linalg.norm(d)
        u = d / length                                           # unit vector along the side
        rel = contour - a
        s = rel @ u / length                                     # position along the side, 0..1
        in_span = (s > skip) & (s < 1 - skip)                    # skip the rounded corners
        point, direction = a, u
        for width in (max(tol, 0.03 * length), max(tol, 0.012 * length), max(0.6 * tol, 0.006 * length)):
            off = (contour - point) @ np.array([-direction[1], direction[0]])    # signed distance to the current line
            pick = in_span & (np.abs(off) <= width)
            if pick.sum() < 8:
                return quad.astype(np.float32)                   # not enough evidence: keep the polygon
            point, direction = _line(contour[pick])
        nrm = np.array([-direction[1], direction[0]])            # normal (perpendicular) of the line
        if nrm @ (point - centre) < 0:                           # push the line 0.5 px outward: contour pixels
            nrm = -nrm                                           # are the OUTERMOST yellow pixels, whose centres
        lines.append((point + 0.5 * nrm, direction))             # lie half a pixel inside the true edge
        side_len.append(length)
    limit = max(6.0, 0.05 * float(np.mean(side_len)))            # a corner may not jump further than this
    out = []
    for i in range(4):
        x = _intersect(lines[i - 1], lines[i])                   # corner i = side i-1 meets side i
        out.append(quad[i] if x is None or np.linalg.norm(x - quad[i]) > limit else x)   # sanity-check the jump
    return np.array(out, np.float32)


def edge_cover(mask, quad):
    """Fraction of samples along the card's four sides (a little inside the edge) that are yellow.

    A real card border is yellow all the way round; a random yellow-ish rectangle (a sticky note, a
    sign) usually is not. We sample 20 points along each side, nudged slightly inward, and ask the
    mask "is anything yellow within one pixel here?"."""
    h, w = mask.shape
    c = quad.mean(axis=0)
    hits = total = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        inward = c - (a + b) / 2                                 # direction from the side's midpoint to the card centre
        inward /= max(np.linalg.norm(inward), 1e-9)
        depth = max(1.5, 0.025 * np.linalg.norm(b - a))          # how far inside the edge to look
        for t in np.linspace(0.1, 0.9, 20):                      # skip the corners
            x, y = np.round(a + (b - a) * t + inward * depth).astype(int)
            patch = mask[max(y - 1, 0):min(y + 2, h), max(x - 1, 0):min(x + 2, w)]   # 3x3 neighbourhood
            hits += bool(patch.size and patch.max())
            total += 1
    return hits / total


# ------------------------------------------------------------------ outer border only
def outer_clear(mask, quad):
    """Fraction of samples just OUTSIDE the card's four sides that are NOT border-coloured (1.0 = clean edge all round).

    A card's border is the outermost thing: beyond it lies the table. A yellow/gray frame INSIDE a card (round the artwork,
    a text box) has more card-colour around it, so it scores low and is thrown away. Mirrors the JS twin (outerClear).
    Samples that fall outside the picture count as clear."""
    h, w = mask.shape
    c = quad.mean(axis=0)
    clear = total = 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        out = (a + b) / 2 - c                                    # direction from the card centre through the side's midpoint
        out /= max(np.linalg.norm(out), 1e-9)
        depth = max(2.5, 0.03 * np.linalg.norm(b - a))           # a few pixels beyond the edge (past the 1-2 px of blur)
        for t in np.linspace(0.1, 0.9, 20):
            x, y = np.round(a + (b - a) * t + out * depth).astype(int)
            if 0 <= x < w and 0 <= y < h:
                patch = mask[max(y - 1, 0):min(y + 2, h), max(x - 1, 0):min(x + 2, w)]
                clear += not patch.max()
            else:
                clear += 1
            total += 1
    return clear / total


def inside_quad(pt, quad, margin=2.0):
    """Is the point inside the convex quad (allowing `margin` px of slack)?"""
    return cv2.pointPolygonTest(np.asarray(quad, np.float32).reshape(-1, 1, 2), (float(pt[0]), float(pt[1])), True) >= -margin


def suppress_inner(dets):
    """Drop detections that lie completely INSIDE a bigger one: that is an inner frame of the same card, not a second card
    (cards cannot overlap on a table). Only a container at least 1/contain_ratio times bigger counts, so two similar
    cards are never merged. `dets` = Detections; returns the survivors in the same order."""
    keep = []
    for d in dets:
        area = cv2.contourArea(d.corners.astype(np.float32))
        inner = False
        for o in dets:
            if o is d:
                continue
            oa = cv2.contourArea(o.corners.astype(np.float32))
            if oa > area * 1.03 and area >= PARAMS["contain_ratio"] * oa and all(inside_quad(p, o.corners) for p in d.corners):
                inner = True
                break
        if not inner:
            keep.append(d)
    return keep


def outer_from_inner(mask, quad):
    """Is this quad really an INNER frame of a bigger card? If so, rebuild the card's OUTER border from the mask.

    A card's outer ring can be broken (a finger holding it, glare, a worn edge) - then the gold frame around the artwork
    can look like "the card". We look OUTWARD from each side of the quad for a bigger ring of the same colour: from a few pixels
    beyond the side we scan along the side's normal and keep the outermost border-coloured pixel within reach (70% of the side).
    Those end points, taken at 9 places along the side, are fitted with a straight line.
        returns ("none", None)  fewer than 3 sides have a ring around them -> this quad IS the outermost border
                ("lost", None)  a ring surrounds it but cannot be rebuilt (e.g. a whole side is hidden) -> ignore this quad
                ("ok", quad2)   the outer border, as 4 corners in the same order as `quad`
    Mirrors outerFromInner() in card-geometry.js."""
    h, w = mask.shape
    c = quad.mean(axis=0)
    lines, sides = [], 0
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        L = float(np.linalg.norm(b - a))
        n = (a + b) / 2 - c
        n = n / max(np.linalg.norm(n), 1e-9)                      # unit normal pointing away from the card centre
        d0, reach = max(3.0, 0.04 * L), 0.7 * L
        offs = np.arange(d0, reach, 1.0)
        ts, os_ = [], []
        for t in np.linspace(0.1, 0.9, 9):
            p = a + (b - a) * t
            xy = np.round(p[None, :] + n[None, :] * offs[:, None]).astype(int)
            ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
            hit = np.zeros(len(offs), bool)
            hit[ok] = mask[xy[ok, 1], xy[ok, 0]] > 0
            idx = np.flatnonzero(hit)                              # the OUTERMOST border-coloured pixel within reach is the ring's outer edge
            if not len(idx):
                continue
            k = idx[-1]
            ts.append(t); os_.append(offs[k])
        if len(ts) < 4:
            lines.append(None)
            continue
        ts, os_ = np.array(ts), np.array(os_)
        tol = max(2.0, 0.012 * L)
        coef = np.polyfit(ts, os_, 1)                              # straight line o = coef[0] * t + coef[1] ...
        good = np.abs(np.polyval(coef, ts) - os_) <= tol
        if good.sum() >= 4 and not good.all():                     # ... refitted once without the outliers (a stray yellow object)
            ts, os_ = ts[good], os_[good]
            coef = np.polyfit(ts, os_, 1)
        if len(ts) < 4 or np.abs(np.polyval(coef, ts) - os_).max() > tol:
            lines.append(None)
            continue
        sides += 1
        p0, p1 = a + n * coef[1], b + n * (coef[0] + coef[1])      # two points on the outer line (offset at t=0 and t=1)
        lines.append((p0, p1 - p0))
    if sides < 3:
        return "none", None
    if sides < 4:
        return "lost", None
    corners = [_intersect(lines[i - 1], lines[i]) for i in range(4)]    # corner i = where sides i-1 and i meet
    if any(c_ is None for c_ in corners):
        return "lost", None
    out = np.array(corners, np.float32)
    return ("ok", out) if np.isfinite(out).all() else ("lost", None)


# ------------------------------------------------------------------ detection
def interior_fraction(mask, quad):
    """Fraction of the card's INSIDE (the middle 60% x 60%) that is also in the mask. A real bordered card has a thin ring
    of border colour around different-coloured artwork -> small value. A solid gray object (laptop lid, phone, mouse pad)
    is gray all the way through -> large value, and is rejected for gray borders."""
    q = np.asarray(quad, np.float32)
    g = np.linspace(0.2, 0.8, 9)
    u, v = np.meshgrid(g, g)                                     # bilinear blend of the four corners (same formula as the JS twin)
    xy = ((1 - v)[..., None] * ((1 - u)[..., None] * q[0] + u[..., None] * q[1])
          + v[..., None] * ((1 - u)[..., None] * q[3] + u[..., None] * q[2])).reshape(-1, 2).round().astype(int)
    h, w = mask.shape
    ok = (xy[:, 0] >= 0) & (xy[:, 0] < w) & (xy[:, 1] >= 0) & (xy[:, 1] < h)
    return float(np.mean(mask[xy[ok, 1], xy[ok, 0]] > 0)) if ok.any() else 1.0


def border_mask(bgr, color="yellow", sensitivity=0.5):
    """Picture -> black/white mask of pixels in the border colour ("yellow" or "gray"), tiny gaps closed (see yellow_mask)."""
    lo, hi = (gray_thresholds if color == "gray" else thresholds)(sensitivity)
    mask = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))


def yellow_mask(bgr, sensitivity=0.5):
    """Picture -> black/white mask of "yellow" pixels, with tiny gaps closed.

    Steps: BGR -> HSV (colour space where hue is its own channel) -> cv2.inRange keeps pixels inside the
    yellow window -> a 3x3 morphological CLOSE (dilate then erode) fills 1-pixel holes so the border
    becomes one connected ring instead of dotted fragments."""
    lo, hi = thresholds(sensitivity)
    mask = cv2.inRange(cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV), np.array(lo, np.uint8), np.array(hi, np.uint8))
    return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))


def approx_quad(hull):
    """Simplify a convex hull to 4 corners, loosening the tolerance until it has exactly 4.

    cv2.approxPolyDP (Douglas-Peucker) replaces a many-pointed outline by a polygon with fewer points that
    stays within `epsilon` of the original. Small epsilon = many points; large = few. We sweep epsilon
    upward until the polygon has exactly 4 corners, or give up (return None) if it never does."""
    peri = cv2.arcLength(hull, True)                             # perimeter of the hull
    for eps in PARAMS["eps_sweep"]:
        approx = cv2.approxPolyDP(hull, eps * peri, True)
        if len(approx) == 4 and cv2.isContourConvex(approx):
            return approx.reshape(4, 2).astype(np.float32)
    return None


# ---- MASK REPAIR from edges ------------------------------------------------------------------------------------------
# Colour windows are fragile: the silver border of a card is bright on one side and in shadow on the other, JPEG noise and
# blur eat thin rings, and a hand covers a corner. The OUTLINE of a card against the table is far steadier (a strong brightness
# step all the way round). So: find outlines from EDGES (Canny), look at the strip of colour just inside each side of every
# outline, and if three or more sides are clearly "yellow/gold" or "silver" the strips are painted into that colour's mask. The
# ordinary pipeline then sees a closed ring and applies ALL its usual tests (shape, cover, outer border only, interior).
def side_class(h, s, v):
    """What border colour does a strip of median HSV look like?  'yellow' (yellow/gold), 'gray' (silver/white-ish) or None."""
    P = PARAMS
    if P["ring_yellow_h"][0] <= h <= P["ring_yellow_h"][1] and s >= P["ring_yellow_smin"] and v >= P["ring_yellow_vmin"]:
        return "yellow"
    if s <= P["ring_gray_smax"] and v >= P["ring_gray_vmin"]:
        return "gray"
    return None


def inner_strip(quad, i, thick, along=(0.25, 0.75), depth=(0.25, 0.8)):
    """The strip just INSIDE side i (corner i -> i+1) of a convex quad: 4 points. `along` = which part of the side's length
    (default 25-75 %: the rounded corners are skipped), `depth` = how far in, in units of `thick` pixels (towards the centre).
    JS twin: CG.innerStrip."""
    a, b = quad[i], quad[(i + 1) % 4]
    c = quad.mean(axis=0)
    d = b - a
    n = np.array([-d[1], d[0]], np.float32)
    n /= (np.linalg.norm(n) or 1.0)
    if np.dot(c - (a + b) / 2, n) < 0:
        n = -n                                                       # make n point into the card
    p0, p1 = a + along[0] * d, a + along[1] * d
    return np.array([p0 + depth[0] * thick * n, p1 + depth[0] * thick * n, p1 + depth[1] * thick * n, p0 + depth[1] * thick * n], np.float32)


def expand_quad(quad, d):
    """The quad with every side pushed `d` pixels outwards (scaled about its centre). JS twin: CG.expandQuad."""
    c = quad.mean(axis=0)
    half = max(np.linalg.norm(quad[0] - c), 1e-6)
    return (c + (quad - c) * (1 + d / half)).astype(np.float32)


def repair_masks(small, masks):
    """Paint the colour strips of edge-found outlines into `masks` (dict colour -> mask, changed in place).
    Returns the list of outlines (4x2 float32 quads, working-resolution pixels) that were painted."""
    P = PARAMS
    painted = []
    hs, ws = small.shape[:2]
    gray = cv2.GaussianBlur(cv2.cvtColor(small, cv2.COLOR_BGR2GRAY), (3, 3), 0)
    edges = cv2.Canny(gray, P["canny_lo"], P["canny_hi"])
    edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    for cnt in contours:
        if cv2.contourArea(cnt) < P["min_area"] * ws * hs * 0.5:
            continue
        hull = cv2.convexHull(cnt)
        area = cv2.contourArea(hull)
        if not P["min_area"] * ws * hs <= area <= P["max_area"] * ws * hs:
            continue
        quad = approx_quad(hull)
        if quad is None:
            continue
        thick = max(2.0, P["ring_thick"] * min(np.linalg.norm(quad[1] - quad[0]), np.linalg.norm(quad[2] - quad[1])))
        for grow in (0.0,) + tuple(P["ring_expand"]):                # 0 = the outline IS the card's outer edge; else it is the INNER edge of the ring, so grow it outward
            q = quad if grow == 0 else expand_quad(quad, grow * thick)
            classes = []
            for i in range(4):
                m = np.zeros((hs, ws), np.uint8)
                cv2.fillConvexPoly(m, np.round(inner_strip(q, i, thick)).astype(np.int32), 255)
                px = hsv[m > 0]
                classes.append(side_class(*np.median(px, axis=0)) if len(px) >= 6 else None)
            best = max(tuple(masks), key=classes.count)
            if classes.count(best) >= P["ring_sides"]:
                for i in range(4):                                   # heal ALL four sides (a hand may cover one), whole length, solid
                    cv2.fillConvexPoly(masks[best], np.round(inner_strip(q, i, thick, (-0.02, 1.02), (0.0, 1.0))).astype(np.int32), 255)
                painted.append(q)
                break
    return painted


def merged_candidates(mask, color, ws, hs):
    """BROKEN RINGS. A real border is rarely one clean ring in the mask: a finger, a glare spot, JPEG noise or a border colour
    that is close to the table's colour leaves it in several FRAGMENTS, and one fragment alone is not card-shaped. So the
    mask is thickened (dilate) just enough to join fragments that are close together, each joined blob is taken as one
    object, and the rubber band (convex hull) is drawn around the ORIGINAL mask pixels inside it. A ring with gaps still
    has the card's outline as its hull. The tests that follow (fill, shape, cover, outer) decide whether it really is a card.
    JS twin: card-tracker.js (_mergedCandidates)."""
    P = PARAMS
    k = max(P["merge_min_px"], int(round(P["merge_gap"] * ws)))
    dil = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k | 1, k | 1)))
    contours, _ = cv2.findContours(dil, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        if cv2.contourArea(cnt) < P["min_area"] * ws * hs * 0.5:
            continue
        blob = np.zeros(mask.shape, np.uint8)
        cv2.drawContours(blob, [cnt], -1, 255, -1)
        pts = cv2.findNonZero(cv2.bitwise_and(mask, blob))
        if pts is None or len(pts) < 20:
            continue
        hull = cv2.convexHull(pts)
        hull_area = cv2.contourArea(hull)
        if not P["min_area"] * ws * hs <= hull_area <= P["max_area"] * ws * hs:
            continue
        quad = approx_quad(hull)
        if quad is not None:
            out.append({"contour": hull.reshape(-1, 2), "approx": quad, "hull_area": float(hull_area), "color": color})
    return out


def raw_candidates(bgr, sensitivity=0.5, work_width=None, colors=None, repair=False):
    """The OpenCV half of detection, as plain data: yellow mask -> external contours -> convex hull ->
    4-corner approximation. The browser tracker's OpenCV.js wrapper produces exactly this structure
    (and everything after it is plain maths in card-geometry.js), so tests can feed one with the other.
    Returns dict(mask, masks, ws, hs, sx, sy, cands=[dict(contour (N,2) int, approx (4,2) float32, hull_area, color)]).
    One mask per border colour (`masks`, e.g. {"yellow": ..., "gray": ...}); `mask` is their union (for debugging); every
    candidate says which colour it came from. Colours are searched separately so a gray card lying on a yellow mat (or the
    reverse) is still seen as its own ring.

    Why shrink first? Analysing a 480-px-wide copy is ~10x faster than a 1600-px photo and the line fit
    in refine_quad recovers sub-pixel accuracy anyway. sx, sy = how much bigger the original is, so the
    caller can scale results back."""
    P = PARAMS
    h0, w0 = bgr.shape[:2]
    ww = work_width or P["work_width"]
    small = cv2.resize(bgr, (ww, max(1, round(h0 * ww / w0))), interpolation=cv2.INTER_AREA) if w0 > ww else bgr
    hs, ws = small.shape[:2]
    masks, cands = {}, []
    colors = colors or P["border_colors"]
    for color in colors:
        masks[color] = border_mask(small, color, sensitivity)
    painted = repair_masks(small, masks) if repair else []       # second chance only (see detect_cards): edges heal the colour masks
    for color in colors:
        mask = masks[color]
        # RETR_EXTERNAL = only outermost outlines (the card border's outer edge, not the inner edge of the ring)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for cnt in contours:
            if cv2.contourArea(cnt) < P["min_area"] * ws * hs * 0.5:  # cheap reject of specks (the outer contour of a
                continue                                              # ring encloses the whole card, so this is its area)
            hull = cv2.convexHull(cnt)                                # the rubber band around the blob
            hull_area = cv2.contourArea(hull)
            if not P["min_area"] * ws * hs <= hull_area <= P["max_area"] * ws * hs:
                continue                                              # too small or too big to be a card
            quad = approx_quad(hull)
            if quad is not None:
                cands.append({"contour": cnt.reshape(-1, 2), "approx": quad, "hull_area": float(hull_area), "color": color})
        cands += merged_candidates(mask, color, ws, hs)
    union = np.zeros((hs, ws), np.uint8)
    for m in masks.values():
        union |= m
    return {"mask": union, "masks": masks, "ws": ws, "hs": hs, "sx": w0 / ws, "sy": h0 / hs, "cands": cands, "painted": painted}


def dedupe(dets):
    """Same card found twice (e.g. its yellow ring AND a gray shadow ring)? Keep the best-scoring one.
    Detections are sorted best-first; one is dropped when its centre is within dup_dist x (larger card's diagonal) of a kept one."""
    kept = []
    for d in sorted(dets, key=lambda d: -d.score):
        near = lambda k: np.linalg.norm(d.center - k.center) < PARAMS["dup_dist"] * max(np.ptp(d.corners, axis=0).max(), np.ptp(k.corners, axis=0).max())
        if not any(near(k) for k in kept):
            kept.append(d)
    return kept


def merge_passes(plain, repaired):
    """Combine the plain-colour pass with the edge-repaired pass (both lists of Detection).
    A repaired detection that lies AROUND a plain one replaces it (the plain one was an inner frame / the card's yellow body
    inside a silver ring); one that is the same card as a plain one is dropped (the plain pass is more accurate); anything
    else is added. JS twin: CG.mergeRepaired."""
    out = list(plain)
    for r in repaired:
        ra = cv2.contourArea(r.corners.astype(np.float32))
        around = [p for p in out if ra > PARAMS["replace_ratio"] * cv2.contourArea(p.corners.astype(np.float32)) and all(inside_quad(c, r.corners) for c in p.corners)]
        if around:
            out = [p for p in out if all(p is not q for q in around)] + [r]
        elif not any(np.linalg.norm(r.center - p.center) < PARAMS["dup_dist"] * max(np.ptp(r.corners, axis=0).max(), np.ptp(p.corners, axis=0).max()) for p in out):
            out.append(r)
    return out


def detect_cards(bgr, max_cards=2, sensitivity=0.5, work_width=None, debug=None, colors=None):
    """Find bordered cards: first with the plain colour masks (fast, most accurate), then once more with the masks healed from
    edge outlines (repair_masks: silver borders in shadow, thin or broken rings, weak edges against the table, an all-yellow
    card whose silver frame the colour window missed). merge_passes() decides which answer wins: the OUTER one.
    The browser tracker does the same (card-tracker.js: raw.repair)."""
    found = _detect_pass(bgr, max_cards, sensitivity, work_width, debug, colors, False)
    if PARAMS["repair"]:
        found = merge_passes(found, _detect_pass(bgr, max_cards, sensitivity, work_width, None, colors, True))
    found.sort(key=lambda d: -d.area * d.cover)
    return found[:max_cards]


def _detect_pass(bgr, max_cards, sensitivity, work_width, debug, colors, repair):
    """One detection pass. Find yellow- or gray-bordered cards. Returns Detections (best first) with corners in ORIGINAL pixels.
    Pass a dict as `debug` to get the working-resolution mask back in debug['mask'].

    Each candidate from raw_candidates() must pass four tests to count as a card:
      1. FILL    - its 4-corner polygon fills >= 90% of its hull (a yellow ball fills only ~64%);
      2. SHAPE   - after corner refinement the quad looks like a 63x88 card seen in perspective
                   (pose_residual <= max_residual);
      3. COVER   - at least 55% of the band just inside its edges is yellow (it is a yellow BORDER);
      4. OUTER   - the band just OUTSIDE its edges is mostly NOT border-coloured (only the outermost ring is the card;
                   frames inside the artwork are ignored), and a candidate lying wholly inside a bigger one is dropped;
      5. then the survivors are sorted by size x cover and the best `max_cards` are returned."""
    P = PARAMS
    raw = raw_candidates(bgr, sensitivity, work_width, colors, repair)
    if debug is not None:
        debug["mask"] = raw["mask"]
    ws, hs = raw["ws"], raw["hs"]
    K = intrinsics(ws, hs, P["hfov"])
    found = []
    for c in raw["cands"]:
        quad = c["approx"]
        if cv2.contourArea(quad) < P["min_fill"] * c["hull_area"]:
            continue                                              # not a 4-sided shape (e.g. a yellow ball)
        quad, residual = orient_quad(refine_quad(c["contour"], quad), K)
        if residual > P["max_residual"]:
            continue                                              # 4 sides, but not a 63 x 88 card
        mask = raw["masks"][c["color"]]                           # judge a candidate against the mask of ITS colour
        status, outer = outer_from_inner(mask, quad)              # a bigger ring of the same colour around it? then THIS is an inner frame
        if status == "lost":
            continue                                              # ... whose outer border can't be rebuilt right now: skip it, never track the inside
        if status == "ok":
            quad, residual = orient_quad(outer, K)
            if residual > P["max_residual"]:
                continue
        cover = edge_cover(mask, quad)
        if cover < P["min_cover"]:
            continue
        if outer_clear(mask, quad) < P["outer_clear_min"]:
            continue                                              # surrounded by more border colour: an inner frame, not the card's OUTER border
        if c["color"] == "gray" and interior_fraction(mask, quad) > P["max_gray_interior"]:
            continue                                              # solid gray object, not a gray BORDER around artwork
        full = np.empty_like(quad)                                # back to original pixel coordinates
        full[:, 0] = (quad[:, 0] + 0.5) * raw["sx"] - 0.5         # +0.5/-0.5: pixel centres vs pixel corners
        full[:, 1] = (quad[:, 1] + 0.5) * raw["sy"] - 0.5
        found.append(Detection(full, float(cv2.contourArea(quad) / (ws * hs)), cover, residual, c["color"]))
    found = suppress_inner(dedupe(found))
    if repair and raw["painted"]:                                 # OUTER BORDER ONLY, also after repair: a card-like outline that was painted AROUND a detection means the detection is an inner frame
        outs = [np.stack([(q[:, 0] + 0.5) * raw["sx"] - 0.5, (q[:, 1] + 0.5) * raw["sy"] - 0.5], axis=1) for q in raw["painted"]]
        found = [d for d in found if not any(cv2.contourArea(o.astype(np.float32)) > 1.04 * cv2.contourArea(d.corners.astype(np.float32))
                                             and all(inside_quad(p, o) for p in d.corners) for o in outs)]
    return found[:max_cards]


def refine_with_hint(bgr, hint, pad=0.3, sensitivity=0.5):
    """Re-measure a card the browser already located. Looks only in a window around `hint`
    ((4,2) corners, pixels of `bgr`) at (nearly) full resolution. Returns (corners, refined?),
    corners keeping the hint's TL/TR/BR/BL order.

    The browser works on a tiny 480-px copy, so its corners are a few pixels off. Here we crop a window
    (the hint's bounding box + 30% margin) out of the FULL-resolution frame and run the same detector
    again - more pixels, more accuracy - then keep the detection that sits at the hint's position."""
    hint = np.asarray(hint, np.float32).reshape(4, 2)
    h, w = bgr.shape[:2]
    x0, y0 = np.floor(hint.min(axis=0) - pad * (hint.max(axis=0) - hint.min(axis=0))).astype(int)
    x1, y1 = np.ceil(hint.max(axis=0) + pad * (hint.max(axis=0) - hint.min(axis=0))).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)    # clamp the window to the picture
    if x1 - x0 < 40 or y1 - y0 < 40:
        return hint, False
    roi = bgr[y0:y1, x0:x1]                                       # "region of interest" = the cropped window
    dets = detect_cards(roi, max_cards=3, sensitivity=sensitivity, work_width=min(640, roi.shape[1]))
    off = np.array([x0, y0], np.float32)                          # add this to move window coords back to frame coords
    dets = [d for d in dets if np.linalg.norm(d.corners.mean(axis=0) + off - hint.mean(axis=0))
            < 0.25 * np.linalg.norm(hint.max(axis=0) - hint.min(axis=0))]    # must be the same card, not a neighbour
    if not dets:
        return hint, False
    best = max(dets, key=lambda d: d.area)
    return match_order(best.corners + off, hint), True


# ------------------------------------------------------------------ straighten / visualise
def rectify(bgr, corners, pad=0.06, out_h=None):
    """Warp the card to an upright rectangle. `corners` = TL, TR, BR, BL. `pad` keeps a margin of
    real surroundings (fraction of the card size) around it - recognisers expect a little context.
    The card is rendered about as large as it appears in the frame, clamped to 440-880 px tall.

    cv2.getPerspectiveTransform takes 4 source points and 4 destination points and returns the 3x3
    matrix M that maps one onto the other; cv2.warpPerspective applies M to every pixel. Here the source
    is the tilted card in the photo and the destination is a flat upright rectangle - it is the same trick
    document-scanner apps use."""
    q = np.asarray(corners, np.float32).reshape(4, 2)
    e = np.linalg.norm(np.roll(q, -1, axis=0) - q, axis=1)       # the four side lengths in pixels
    if out_h is None:
        out_h = int(np.clip(max(e[1], e[3]), 440, 880))          # e[1], e[3] are the two "tall" sides
    cw, ch = round(out_h * CARD_ASPECT), out_h                   # card width/height of the output
    px, py = round(cw * pad), round(ch * pad)                    # margin in pixels
    dst = np.array([[px, py], [px + cw, py], [px + cw, py + ch], [px, py + ch]], np.float32)
    M = cv2.getPerspectiveTransform(q, dst)
    return cv2.warpPerspective(bgr, M, (cw + 2 * px, ch + 2 * py), flags=cv2.INTER_LINEAR,
                               borderMode=cv2.BORDER_REPLICATE)


def overlay_sprite(bgr, corners, sprite_rgba, width=0.78):
    """Visualisation: paste a sprite onto the card plane (warped with the card's perspective), so
    you can see on a photo what the detection found. `sprite_rgba` is a BGRA uint8 image.

    Two chained transforms: (1) sprite pixels -> positions on the card measured in millimetres,
    (2) card millimetres -> picture pixels (the card's own perspective). Multiplying the two 3x3 matrices
    gives one warp, so the sprite is resampled only once."""
    h, w = bgr.shape[:2]
    sw = width * CARD_W                                          # sprite width on the card, in mm
    sh = sw * sprite_rgba.shape[0] / sprite_rgba.shape[1]        # keep the sprite's aspect ratio
    cx, cy = CARD_W / 2, CARD_H * 0.55                           # a bit below the middle, in mm on the card
    src = np.array([[0, 0], [sprite_rgba.shape[1], 0], [sprite_rgba.shape[1], sprite_rgba.shape[0]],
                    [0, sprite_rgba.shape[0]]], np.float32)
    on_card = np.array([[cx - sw / 2, cy - sh / 2], [cx + sw / 2, cy - sh / 2],
                        [cx + sw / 2, cy + sh / 2], [cx - sw / 2, cy + sh / 2]], np.float32)
    card_to_img = cv2.getPerspectiveTransform(np.array([[0, 0], [CARD_W, 0], [CARD_W, CARD_H], [0, CARD_H]],
                                                       np.float32), np.asarray(corners, np.float32))
    M = card_to_img @ cv2.getPerspectiveTransform(src, on_card)
    warped = cv2.warpPerspective(sprite_rgba, M, (w, h), flags=cv2.INTER_NEAREST,
                                 borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    a = warped[..., 3:4].astype(np.float32) / 255                # alpha 0..1
    return (bgr * (1 - a) + warped[..., :3] * a).astype(np.uint8)   # standard alpha blending


def annotate(bgr, detections):
    """Draw what was found (outline, corner labels, edge-cover score) on a copy of the picture."""
    out = bgr.copy()
    t = max(1, round(bgr.shape[1] / 480))                        # line thickness scales with picture size
    for k, d in enumerate(detections):
        q = d.corners.round().astype(np.int32)
        cv2.polylines(out, [q], True, (0, 255, 0), t + 1, cv2.LINE_AA)
        for name, p in zip(("TL", "TR", "BR", "BL"), q):
            cv2.circle(out, tuple(p), 3 * t, (0, 0, 255), -1, cv2.LINE_AA)
            cv2.putText(out, name, (p[0] + 6, p[1] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.35 * t, (255, 255, 255), t, cv2.LINE_AA)
        c = d.center.astype(int)
        cv2.putText(out, f"card {k + 1}  cover {d.cover:.2f}", (c[0] - 40, c[1]), cv2.FONT_HERSHEY_SIMPLEX,
                    0.4 * t, (0, 255, 255), t, cv2.LINE_AA)
    return out


def _load_sprite(path):
    """Load a sprite for --sprite: keep its transparency, or remove a flat background if it has none."""
    from PIL import Image
    import sprites                                              # reuse the project's background remover
    img = Image.open(path)
    has_alpha = img.mode in ("RGBA", "LA") and img.getchannel("A").getextrema()[0] < 255
    img = sprites.crop_to_content(img.convert("RGBA") if has_alpha else sprites.remove_background(img, tol=30))
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGBA2BGRA)


def main():
    """Command-line entry point: `python card_detect.py photo.jpg -o out.jpg --crop card.jpg`."""
    ap = argparse.ArgumentParser(description="Detect yellow- or gray-bordered Pokémon cards in a photo.")
    ap.add_argument("image")
    ap.add_argument("-o", "--out", help="write the annotated photo here")
    ap.add_argument("--crop", help="write the first card, straightened, here")
    ap.add_argument("--sprite", help="paste this sprite image onto every detected card")
    ap.add_argument("--sensitivity", type=float, default=0.5, help="0 strict .. 1 lenient (default 0.5)")
    ap.add_argument("--max-cards", type=int, default=2)
    ap.add_argument("--border", choices=("yellow", "gray", "both"), default="both", help="border colour to look for (default both)")
    a = ap.parse_args()
    img = cv2.imread(a.image)
    if img is None:
        raise SystemExit(f"cannot read {a.image}")
    dets = detect_cards(img, a.max_cards, a.sensitivity, colors=("yellow", "gray") if a.border == "both" else (a.border,))
    print(f"{len(dets)} card(s) found")
    for k, d in enumerate(dets):
        print(f"  card {k + 1}: {d.color} border  area {d.area:.1%}  edge cover {d.cover:.2f}  corners TL,TR,BR,BL =",
              d.corners.round(1).tolist())
    if a.crop and dets:
        cv2.imwrite(a.crop, rectify(img, dets[0].corners))
    if a.out:
        out = img
        if a.sprite:
            sp = _load_sprite(a.sprite)
            for d in dets:
                out = overlay_sprite(out, d.corners, sp)
        cv2.imwrite(a.out, annotate(out, dets))


if __name__ == "__main__":
    main()
