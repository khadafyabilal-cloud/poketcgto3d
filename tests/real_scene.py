"""real_scene.py - put a REAL card picture (tests/real_cards/*.png) on a table, in perspective, to test card_detect.py.

compose(card_png, background, yaw, pitch, roll, width_frac, cx, cy) -> (frame_bgr, truth_corners TL TR BR BL).
The card gets rounded corners (like a real card) and a soft shadow; the table is a colour with a little noise and a light
gradient so the detector sees something close to a photo. Used by tests/test_real_cards.py and the examples in the docs."""
import math
from pathlib import Path
import cv2, numpy as np

DIR = Path(__file__).parent / "real_cards"
TABLES = {                                    # BGR table colours: dark wood, blue-grey cloth, light wood, green playmat, black mat
    "dark wood": (35, 48, 70), "blue cloth": (115, 100, 85), "light wood": (110, 150, 195), "green mat": (60, 95, 45), "black mat": (28, 26, 26),
    "white desk": (238, 240, 240),
}


def load_card(name):
    im = cv2.imread(str(DIR / name), cv2.IMREAD_UNCHANGED)
    if im.shape[2] == 3:
        im = np.dstack([im, np.full(im.shape[:2], 255, np.uint8)])
    h, w = im.shape[:2]
    r = int(0.045 * w)                        # real cards have rounded corners
    m = np.zeros((h, w), np.uint8)
    cv2.rectangle(m, (r, 0), (w - r, h), 255, -1); cv2.rectangle(m, (0, r), (w, h - r), 255, -1)
    for cx, cy in ((r, r), (w - r, r), (r, h - r), (w - r, h - r)):
        cv2.circle(m, (cx, cy), r, 255, -1)
    im[..., 3] = np.minimum(im[..., 3], m)
    return im


def compose(card, table="dark wood", yaw=0.0, pitch=0.0, roll=0.0, width_frac=0.34, cx=0.5, cy=0.5, size=(960, 540), seed=0, light=1.0):
    rng = np.random.default_rng(seed)
    W, H = size
    bgc = np.array(TABLES[table], np.float32)
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    grad = 1 + 0.12 * ((xx / W - 0.5) * 0.8 + (yy / H - 0.5) * 0.5)
    frame = (bgc[None, None, :] * grad[..., None] + rng.normal(0, 3.0, (H, W, 1))).clip(0, 255)
    ch, cw = card.shape[:2]
    cwid = width_frac * W; chei = cwid * ch / cw
    # card rectangle in 3-D (centre at origin), rotate yaw (about vertical), pitch (about horizontal), roll (in plane)
    pts = np.array([[-.5, -.5, 0], [.5, -.5, 0], [.5, .5, 0], [-.5, .5, 0]], np.float32) * np.array([cwid, chei, 0], np.float32)
    cyaw, syaw, cp, sp, cr, sr = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch), math.cos(roll), math.sin(roll)
    Rz = np.array([[cr, -sr, 0], [sr, cr, 0], [0, 0, 1]]); Ry = np.array([[cyaw, 0, syaw], [0, 1, 0], [-syaw, 0, cyaw]]); Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    P = pts @ (Rz @ Ry @ Rx).T
    f = 1.1 * W; z0 = f * 1.0                  # simple pinhole camera, card about a metre... in "pixels"
    proj = np.stack([f * P[:, 0] / (z0 + P[:, 2]) + cx * W, f * P[:, 1] / (z0 + P[:, 2]) + cy * H], axis=1).astype(np.float32)
    src = np.array([[0, 0], [cw, 0], [cw, ch], [0, ch]], np.float32)
    M = cv2.getPerspectiveTransform(src, proj)
    warped = cv2.warpPerspective(card, M, (W, H), flags=cv2.INTER_AREA if cw > cwid else cv2.INTER_LINEAR, borderValue=(0, 0, 0, 0))
    a = (warped[..., 3:4].astype(np.float32) / 255)
    shadow = cv2.GaussianBlur(a, (0, 0), 6)[..., None] if a.ndim == 2 else cv2.GaussianBlur(a[..., 0], (0, 0), 6)[..., None]
    shift = np.roll(shadow, (6, 6), axis=(0, 1))
    frame = frame * (1 - 0.35 * shift)
    frame = frame * (1 - a) + warped[..., :3].astype(np.float32) * a
    frame = (frame * light).clip(0, 255).astype(np.uint8)
    return frame, proj


def iou(a, b, size=(960, 540)):
    ma, mb = np.zeros(size[::-1], np.uint8), np.zeros(size[::-1], np.uint8)
    cv2.fillConvexPoly(ma, np.round(a).astype(np.int32), 1); cv2.fillConvexPoly(mb, np.round(b).astype(np.int32), 1)
    u = (ma | mb).sum()
    return float((ma & mb).sum() / u) if u else 0.0


# ---- sweeps: many random poses/tables/lighting, made to look like phone photos ----
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import card_detect


def photoize(fr, rng, glare=False):
    f=fr.astype(np.float32)
    f*=rng.uniform(.9,1.1,3)[None,None,:]                       # white-balance cast
    s=rng.uniform(0,1.4)
    if s>0.2: f=cv2.GaussianBlur(f,(0,0),s)
    if glare:
        H,W=f.shape[:2]; g=np.zeros((H,W),np.float32); cv2.ellipse(g,(int(rng.uniform(.35,.65)*W),int(rng.uniform(.3,.7)*H)),(int(W*.06),int(H*.1)),rng.uniform(0,180),0,360,1,-1)
        g=cv2.GaussianBlur(g,(0,0),18)*rng.uniform(.3,.6); f=f*(1-g[...,None])+255*g[...,None]
    f=f.clip(0,255).astype(np.uint8)
    ok,b=cv2.imencode('.jpg',f,[cv2.IMWRITE_JPEG_QUALITY,int(rng.integers(55,90))]); return cv2.imdecode(b,1)
def run(cards,n=30,tables=None,seed=1,verbose=False,portrait=False):
    tabs=tables or [t for t in TABLES if t!='white desk']
    out={}
    for c in cards:
        card=load_card(c); rng=np.random.default_rng(seed); ok=0; fails=[]
        for i in range(n):
            t=tabs[int(rng.integers(len(tabs)))]; yaw,pitch,roll=rng.uniform(-.4,.4),rng.uniform(-.4,.4),rng.uniform(-.5,.5)
            wf=rng.uniform(.3,.55) if portrait else rng.uniform(.15,.29); light=rng.uniform(.65,1.1); glare=rng.random()<.3
            fr,truth=compose(card,t,yaw,pitch,roll,width_frac=wf,cx=rng.uniform(.42,.58),cy=rng.uniform(.45,.55),light=light,seed=i,size=(540,960) if portrait else (960,540))
            fr=photoize(fr,rng,glare)
            d=card_detect.detect_cards(fr); best=max([iou(x.corners,truth,(540,960) if portrait else (960,540)) for x in d],default=0)
            if best>.85 and len(d)==1: ok+=1
            else: fails.append((i,t,round(yaw,2),round(pitch,2),round(roll,2),round(wf,2),round(light,2),glare,round(best,2),len(d)))
        out[c]=(ok,n,fails)
    return out
