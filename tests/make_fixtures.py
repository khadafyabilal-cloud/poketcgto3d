"""
Generate tests/fixtures/*.json for the JavaScript tests.

Synthetic camera frames with known card poses are run through the REAL OpenCV (cv2) front half of the
detector (card_detect.raw_candidates: mask, contours, hull, 4-corner approximation). The JS tests feed
that data to card-geometry.js / card-tracker.js's TrackerCore - the parts that run in the browser - and
compare with the ground truth and with Python's own answer. (OpenCV.js itself can't run in this test.)

    python tests/make_fixtures.py
"""
import base64, json, sys, zlib
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))
import card_detect as cd
from synth import render_scene, intrinsics

OUT = HERE / "fixtures"
W, H = 960, 540


def enc_raw(frame, sensitivity=0.5):
    raw = cd.raw_candidates(frame, sensitivity)
    return {
        "w": raw["ws"], "h": raw["hs"], "srcW": frame.shape[1], "srcH": frame.shape[0],
        "mask": base64.b64encode(zlib.compress(raw["mask"].tobytes(), 9)).decode(),          # union of all border colours
        "masks": {k: base64.b64encode(zlib.compress(v.tobytes(), 9)).decode() for k, v in raw["masks"].items()},   # one per colour
        "cands": [{"contour": c["contour"].astype(int).ravel().tolist(), "color": c["color"],
                   "approx": c["approx"].astype(int).tolist(), "hullArea": c["hull_area"]} for c in raw["cands"]],
    }


def enc_truth(truth):
    return [{"corners": t["corners"].tolist(), "R": t["R"].ravel().tolist(), "t": t["t"].tolist()} for t in truth]


def scenes():
    cases = {
        "face-on": [dict(cx=.5, cy=.5, width=240)],
        "tilt45-yaw20": [dict(cx=.45, cy=.5, width=250, pitch=45, yaw=20)],
        "roll90": [dict(cx=.5, cy=.5, width=240, roll=90, pitch=15)],
        "roll180": [dict(cx=.5, cy=.5, width=240, roll=180, pitch=20)],
        "far": [dict(cx=.5, cy=.5, width=110)],
        "yellow-art": [dict(cx=.5, cy=.5, width=240, pitch=20, yellow_art=True)],
        "two-cards": [dict(cx=.28, cy=.5, width=200, pitch=25, roll=-10), dict(cx=.72, cy=.5, width=200, pitch=25, roll=190)],
        "decoys-only": [],
        "gray-face-on": [dict(cx=.5, cy=.5, width=240, border="gray")],
        "gray-tilt45-yaw20": [dict(cx=.45, cy=.5, width=250, pitch=45, yaw=20, border="gray")],
        "gray-roll90": [dict(cx=.5, cy=.5, width=240, roll=90, pitch=15, border="gray")],
        "inner-frame": [dict(cx=.5, cy=.5, width=240, pitch=20, inner_frame=True)],
        "inner-frame-broken-ring": [dict(cx=.5, cy=.5, width=240, pitch=20, yaw=15, inner_frame=True, border_gap=True)],
        "gray-inner-frame": [dict(cx=.5, cy=.5, width=240, pitch=20, border="gray", inner_frame="offcentre")],
        "inner-frame-offcentre-broken-ring": [dict(cx=.5, cy=.5, width=240, pitch=20, inner_frame="offcentre", border_gap=True)],
        "hidden-outer-border": [dict(cx=.5, cy=.5, width=240, pitch=20, yaw=15, inner_frame="offcentre", border_gap=True)],   # outer border cannot be rebuilt: report NOTHING (the inside must never be tracked)
        "mixed-gray-and-yellow": [dict(cx=.4, cy=.5, width=200, pitch=25, border="gray"), dict(cx=.68, cy=.45, width=200, pitch=25, roll=190)],
    }
    out = []
    for name, cards in cases.items():
        frame, K, truth = render_scene(W, H, cards=cards, seed=3, decoys=True, gray_decoys=True)
        py = cd._detect_pass(frame, 4, 0.5, None, None, None, False)   # the plain pass (the edge-repair second chance needs OpenCV.js and cannot run in Node)
        out.append({"name": name, **enc_raw(frame), "truth": [] if name.startswith("hidden-") else enc_truth(truth),
                    "python": [d.corners.tolist() for d in py]})
    return out


def sequence_move(n=42, fps=30):
    """Card A drifts right while rolling, vanishes for 4 frames (hand over it), comes back; card B joins at frame 10."""
    frames = []
    for i in range(n):
        cards = []
        a_visible = not (20 <= i < 24)
        if a_visible:
            cards.append(dict(cx=.30 + .005 * i, cy=.5, width=230, pitch=25, yaw=10 - i * .4, roll=i * 0.9, seed=1))
        if i >= 10:
            cards.append(dict(cx=.74, cy=.48, width=210, pitch=25, roll=180 + i * 0.2, seed=2))
        frame, K, truth = render_scene(W, H, cards=cards, seed=100 + i)
        keys = (["A"] if a_visible else []) + (["B"] if i >= 10 else [])
        frames.append({"ts": i * 1000 / fps, **enc_raw(frame),
                       "truth": [{"card": k, **t} for k, t in zip(keys, enc_truth(truth))]})
    return frames


def sequence_static(n=30, fps=30):
    frames = []
    for i in range(n):
        frame, K, truth = render_scene(W, H, cards=[dict(cx=.5, cy=.5, width=240, pitch=30, roll=12)], seed=200 + i)
        frames.append({"ts": i * 1000 / fps, **enc_raw(frame), "truth": enc_truth(truth)})
    return frames


def main():
    OUT.mkdir(exist_ok=True)
    K = intrinsics(W, H)
    meta = {"K": {"fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2]}, "hfov": 63, "W": W, "H": H}
    grid = [round(float(s), 2) for s in np.linspace(0, 1, 21)]          # the same doubles the JS test will parse
    (OUT / "thresholds.json").write_text(json.dumps({f"{s:.2f}": list(map(list, cd.thresholds(s))) for s in grid}))
    (OUT / "gray_thresholds.json").write_text(json.dumps({f"{s:.2f}": list(map(list, cd.gray_thresholds(s))) for s in grid}))
    (OUT / "scenes.json").write_text(json.dumps({"meta": meta, "scenes": scenes()}))
    (OUT / "sequence_move.json").write_text(json.dumps({"meta": meta, "frames": sequence_move()}))
    (OUT / "sequence_static.json").write_text(json.dumps({"meta": meta, "frames": sequence_static()}))
    for f in sorted(OUT.glob("*.json")):
        print(f"{f.name:24} {f.stat().st_size // 1024:5d} KB")


if __name__ == "__main__":
    main()
