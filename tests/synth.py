"""
synth.py - render synthetic camera frames with yellow-bordered cards at KNOWN 3D poses, so the
detector / pose code can be checked against exact ground truth (no camera or photos needed).

Conventions (same as the browser tracker):
  * card-local frame: origin at the card centre, X to the right, Y DOWN the card, Z into the card;
    1 unit = card width, so the card is 1 x CARD_H units (CARD_H = 88/63).
  * camera: pinhole, x right, y down, z forward (OpenCV). pose = (R, t) maps card -> camera.
"""
import numpy as np
import cv2

CARD_H = 88.0 / 63.0
TEX_W, TEX_H = 315, 440                       # card texture, 5 px per mm


def rot(yaw=0.0, pitch=0.0, roll=0.0):
    """Rotation from degrees: roll about the view axis (Z), pitch about X, yaw about Y."""
    y, p, r = np.radians([yaw, pitch, roll])
    Ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    Rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    Rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return Rz @ Rx @ Ry


def intrinsics(w, h, hfov_deg=63.0):
    f = (w / 2) / np.tan(np.radians(hfov_deg) / 2)
    return np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0, 1]], np.float64)


def card_texture(seed=0, yellow_art=False, border="yellow", inner_frame=False, border_gap=False):
    """BGRA card: yellow (or gray) border, dark frame, art box, text box, rounded corners.
    inner_frame=True adds a second, card-shaped ring of the border colour INSIDE the card (like the gold frame of many
    cards): the detector must ignore it and report only the outer border."""
    rng = np.random.default_rng(seed)
    t = np.zeros((TEX_H, TEX_W, 4), np.uint8)
    yellow = (5, 203, 255) if border == "yellow" else (150, 150, 150)     # BGR of #FFCB05, or a mid silver-gray
    t[..., :3] = yellow
    b = 16                                                       # border ~3.2 mm
    t[b:-b, b:-b, :3] = (40, 45, 60)                             # dark frame
    t[b + 6:-b - 6, b + 6:-b - 6, :3] = (210, 225, 235)          # cream card face
    y0, y1, x0, x1 = b + 40, b + 230, b + 14, TEX_W - b - 14
    art = np.empty((y1 - y0, x1 - x0, 3), np.uint8)              # contiguous copy: cv2 can't draw on a sliced view
    art[:] = (150, 90, 40) if not yellow_art else (40, 210, 250)
    for _ in range(14):                                          # coloured blobs as "artwork"
        c = tuple(int(v) for v in rng.integers(20, 255, 3))
        if yellow_art and rng.random() < 0.5:
            c = (30, 200, 250)
        cv2.circle(art, (int(rng.integers(0, art.shape[1])), int(rng.integers(0, art.shape[0]))),
                   int(rng.integers(10, 50)), c, -1)
    t[y0:y1, x0:x1, :3] = art
    for k in range(7):                                           # text lines
        y = b + 260 + k * 22
        cv2.line(t, (b + 24, y), (TEX_W - b - 24 - int(rng.integers(0, 90)), y), (60, 60, 60), 2)
    if inner_frame:                                              # fractions of the texture (x0, y0, x1, y1); True = centred, "offcentre" = high up
        x0f, y0f, x1f, y1f = {True: (0.185, 0.185, 0.815, 0.815), "offcentre": (0.2, 0.13, 0.8, 0.73)}[inner_frame]
        cv2.rectangle(t, (int(x0f * TEX_W), int(y0f * TEX_H)), (int(x1f * TEX_W), int(y1f * TEX_H)), yellow, 7)
    if border_gap:                                               # a "finger"/glare breaks the outer ring in two places: the inner frame is then no longer enclosed by it
        t[0:b + 3, 100:200, :3] = (210, 225, 235)
        t[200:270, 0:b + 3, :3] = (210, 225, 235)
    t[..., 3] = 255
    r = 17                                                       # rounded corners (3.4 mm)
    for cx, cy in ((r, r), (TEX_W - 1 - r, r), (r, TEX_H - 1 - r), (TEX_W - 1 - r, TEX_H - 1 - r)):
        yy, xx = np.ogrid[:TEX_H, :TEX_W]
        corner = ((xx < r) | (xx > TEX_W - 1 - r)) & ((yy < r) | (yy > TEX_H - 1 - r))
        inside = (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r
        near = (abs(xx - cx) < r + 1) & (abs(yy - cy) < r + 1)
        t[..., 3][corner & near & ~inside] = 0
    return t


def table(w, h, seed=1):
    rng = np.random.default_rng(seed)
    base = np.array([70, 105, 150], np.float32)                  # BGR wood ~ hue 10 (outside the yellow window)
    img = np.tile(base, (h, w, 1))
    grain = cv2.GaussianBlur(rng.normal(0, 14, (h, w)).astype(np.float32), (0, 0), sigmaX=40, sigmaY=1.5)
    img += grain[..., None] * np.array([0.7, 1.0, 1.2], np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def render_scene(w=960, h=540, cards=(), seed=0, hfov=63.0, blur=0.8, noise=3.5, gain=1.0,
                 decoys=False, jpeg=85, gray_decoys=False):
    """cards: list of dicts {yaw,pitch,roll (deg), cx,cy (0-1 centre in the frame), width (px of card width),
    yellow_art (bool), border ("yellow"|"gray"), inner_frame (True|"offcentre"), border_gap (bool: breaks the outer ring), seed}. gray_decoys adds solid gray rectangles (laptop/phone-like) that are NOT cards. Returns (BGR frame, K, truth) where truth[i] has corners/R/t."""
    K = intrinsics(w, h, hfov)
    frame = table(w, h, seed + 1).astype(np.float32)
    rng = np.random.default_rng(seed)
    truth = []
    if decoys:                                                   # yellow things that are NOT cards
        cv2.circle(frame, (int(0.08 * w), int(0.85 * h)), 38, (20, 200, 245), -1)       # ball
        cv2.rectangle(frame, (int(0.9 * w), int(0.05 * h)), (int(0.97 * w), int(0.45 * h)), (20, 200, 245), -1)  # pencil
    if gray_decoys:                                              # solid gray slabs: right colour, no artwork inside
        cv2.rectangle(frame, (int(0.02 * w), int(0.08 * h)), (int(0.2 * w), int(0.42 * h)), (150, 150, 150), -1)
        cv2.rectangle(frame, (int(0.8 * w), int(0.55 * h)), (int(0.97 * w), int(0.95 * h)), (130, 135, 135), -1)
    for c in cards:
        R = rot(c.get("yaw", 0), c.get("pitch", 0), c.get("roll", 0))
        z = K[0, 0] / c["width"]                                 # distance (card-width units) giving that pixel width
        ray = np.linalg.inv(K) @ np.array([c["cx"] * w, c["cy"] * h, 1.0])
        t = ray * z
        S = np.array([[1 / TEX_W, 0, 0.5 / TEX_W - 0.5], [0, CARD_H / TEX_H, CARD_H * (0.5 / TEX_H - 0.5)], [0, 0, 1]])
        H = K @ np.column_stack([R[:, 0], R[:, 1], t]) @ S
        tex = card_texture(c.get("seed", 0), c.get("yellow_art", False), c.get("border", "yellow"), c.get("inner_frame", False), c.get("border_gap", False)).astype(np.float32)
        tex[..., :3] *= tex[..., 3:4] / 255                      # premultiply so edges blend correctly
        warped = cv2.warpPerspective(tex, H, (w, h), flags=cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
        a = warped[..., 3:4] / 255
        frame = frame * (1 - a) + warped[..., :3]
        src = np.array([[-0.5, -0.5], [TEX_W - 0.5, -0.5], [TEX_W - 0.5, TEX_H - 0.5], [-0.5, TEX_H - 0.5]])
        proj = (H @ np.column_stack([src, np.ones(4)]).T).T
        truth.append({"corners": (proj[:, :2] / proj[:, 2:]), "R": R, "t": t})
    frame = np.clip(frame * gain, 0, 255)
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    if noise:
        frame = frame + rng.normal(0, noise, frame.shape)
    frame = np.clip(frame, 0, 255).astype(np.uint8)
    if jpeg:
        frame = cv2.imdecode(cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, jpeg])[1], cv2.IMREAD_COLOR)
    return frame, K, truth
