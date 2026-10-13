"""Tests for card_detect.py (Python OpenCV) on synthetic scenes with known card poses.
Run:  python -m unittest discover -s tests -v"""
import os, sys, tempfile, time, unittest
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import card_detect as cd
from synth import render_scene, card_texture, TEX_W, TEX_H


def corner_error(det_corners, truth_corners):
    """Per-corner pixel error after lining the detection up with the truth (cyclic order)."""
    ref = cd.match_order(truth_corners, det_corners)
    return np.linalg.norm(det_corners - ref, axis=1)


def nearest(dets, truth):
    return min(dets, key=lambda d: np.linalg.norm(d.center - truth["corners"].mean(axis=0)))


class DetectAccuracy(unittest.TestCase):
    CASES = {
        "face-on": [dict(cx=.5, cy=.5, width=240)],
        "tilt 30": [dict(cx=.5, cy=.5, width=240, pitch=30)],
        "tilt 45 + yaw 20 (looks almost square)": [dict(cx=.45, cy=.5, width=250, pitch=45, yaw=20)],
        "tilt 55": [dict(cx=.5, cy=.5, width=260, pitch=55, roll=10)],
        "rolled 30": [dict(cx=.5, cy=.5, width=240, roll=30)],
        "rolled 90": [dict(cx=.5, cy=.5, width=240, roll=90)],
        "rolled 180": [dict(cx=.5, cy=.5, width=240, roll=180)],
        "small / far (110 px wide)": [dict(cx=.5, cy=.5, width=110)],
        "yellow artwork": [dict(cx=.5, cy=.5, width=240, pitch=20, yellow_art=True)],
        "two cards (battle)": [dict(cx=.28, cy=.5, width=200, pitch=25, roll=-10),
                               dict(cx=.72, cy=.5, width=200, pitch=25, roll=190)],
    }

    def test_corners_within_two_pixels(self):
        for name, cards in self.CASES.items():
            with self.subTest(name):
                frame, K, truth = render_scene(cards=cards, seed=3, decoys=True)
                dets = cd.detect_cards(frame)
                self.assertEqual(len(dets), len(cards), f"expected {len(cards)} card(s), got {len(dets)}")
                for tr in truth:
                    err = corner_error(nearest(dets, tr).corners, tr["corners"])
                    self.assertLess(err.max(), 2.5, f"corner errors {err.round(2)} px")

    def test_dim_light(self):
        frame, _, truth = render_scene(cards=[dict(cx=.5, cy=.5, width=240, pitch=20)], gain=0.55, seed=5)
        dets = cd.detect_cards(frame)
        self.assertEqual(len(dets), 1)
        self.assertLess(corner_error(dets[0].corners, truth[0]["corners"]).max(), 3.0)

    def test_no_cards_and_decoys_only(self):
        frame, _, _ = render_scene(cards=[], seed=2, decoys=True)       # yellow ball + yellow pencil
        self.assertEqual(cd.detect_cards(frame), [])

    def test_max_cards_limits_results(self):
        frame, _, _ = render_scene(cards=self.CASES["two cards (battle)"], seed=3)
        self.assertEqual(len(cd.detect_cards(frame, max_cards=1)), 1)

    def test_sensitivity_is_monotonic(self):
        lo0, hi0 = cd.thresholds(0.0)
        lo1, hi1 = cd.thresholds(1.0)
        self.assertGreater(lo0[1], lo1[1]); self.assertGreater(lo0[2], lo1[2])       # lenient accepts paler yellow
        self.assertLess(lo1[0], lo0[0]); self.assertGreater(hi1[0], hi0[0])          # ... and a wider hue window
        self.assertEqual(cd.thresholds(0.5), cd.thresholds(0.5))

    def test_fast_enough(self):
        frame, _, _ = render_scene(w=1280, h=720, cards=[dict(cx=.5, cy=.5, width=300, pitch=25)], seed=1)
        cd.detect_cards(frame)                                             # warm-up
        t = time.perf_counter()
        for _ in range(10):
            cd.detect_cards(frame)
        ms = (time.perf_counter() - t) * 100
        self.assertLess(ms, 40, f"{ms:.1f} ms per 1280x720 frame")


class GrayBorder(unittest.TestCase):
    """Cards with a gray (silver) border are found as well, without turning gray objects into cards."""
    CASES = {
        "face-on": dict(cx=.5, cy=.5, width=240),
        "tilt 45 + yaw 20": dict(cx=.45, cy=.5, width=250, pitch=45, yaw=20),
        "rolled 90": dict(cx=.5, cy=.5, width=240, roll=90, pitch=15),
        "rolled 180": dict(cx=.5, cy=.5, width=240, roll=180),
        "small / far": dict(cx=.5, cy=.5, width=120),
    }

    def test_gray_cards_are_found_within_two_pixels(self):
        for name, c in self.CASES.items():
            with self.subTest(name):
                frame, _, truth = render_scene(960, 540, cards=[dict(c, border="gray")], seed=3, decoys=True, gray_decoys=True)
                dets = cd.detect_cards(frame)
                self.assertEqual(len(dets), 1, f"{len(dets)} detections")
                self.assertEqual(dets[0].color, "gray")
                self.assertLess(corner_error(dets[0].corners, truth[0]["corners"]).max(), 2.5)

    def test_yellow_cards_still_report_yellow(self):
        frame, _, _ = render_scene(960, 540, cards=[dict(cx=.5, cy=.5, width=240)], seed=3, decoys=True, gray_decoys=True)
        self.assertEqual([d.color for d in cd.detect_cards(frame)], ["yellow"])

    def test_solid_gray_objects_are_not_cards(self):
        frame, _, _ = render_scene(960, 540, cards=[], seed=3, decoys=True, gray_decoys=True)
        self.assertEqual(cd.detect_cards(frame), [])
        raw = cd.raw_candidates(frame)
        self.assertTrue(any(c["color"] == "gray" for c in raw["cands"]), "the slabs should be candidates that the interior test then rejects")

    def test_gray_and_yellow_card_side_by_side(self):
        frame, _, truth = render_scene(960, 540, cards=[dict(cx=.4, cy=.5, width=200, border="gray"), dict(cx=.68, cy=.45, width=200)],
                                       seed=3, decoys=True, gray_decoys=True)
        dets = cd.detect_cards(frame)
        self.assertEqual(sorted(d.color for d in dets), ["gray", "yellow"])
        for t in truth:
            self.assertLess(corner_error(nearest(dets, t).corners, t["corners"]).max(), 2.5)

    def test_colour_filter(self):
        frame, _, _ = render_scene(960, 540, cards=[dict(cx=.5, cy=.5, width=240, border="gray")], seed=3)
        self.assertEqual(len(cd.detect_cards(frame, colors=("yellow",))), 0)
        self.assertEqual(len(cd.detect_cards(frame, colors=("gray",))), 1)

    def test_dim_and_bright_light(self):
        for gain in (0.65, 1.25):
            frame, _, _ = render_scene(960, 540, cards=[dict(cx=.5, cy=.5, width=240, border="gray")], seed=3, gain=gain)
            self.assertEqual(len(cd.detect_cards(frame, sensitivity=0.7)), 1, gain)

    def test_one_card_is_not_reported_twice(self):
        frame, _, _ = render_scene(960, 540, cards=[dict(cx=.5, cy=.5, width=240, border="gray")], seed=3)
        self.assertEqual(len(cd.detect_cards(frame, max_cards=4)), 1)

    def test_gray_windows_widen_with_sensitivity(self):
        lo0, hi0 = cd.gray_thresholds(0); lo1, hi1 = cd.gray_thresholds(1)
        self.assertLessEqual(lo1[2], lo0[2]); self.assertGreaterEqual(hi1[1], hi0[1]); self.assertGreaterEqual(hi1[2], hi0[2])

    def test_fast_enough(self):
        frame, _, _ = render_scene(960, 540, cards=[dict(cx=.5, cy=.5, width=240)], seed=3)
        t0 = time.time(); [cd.detect_cards(frame) for _ in range(10)]
        self.assertLess((time.time() - t0) / 10, 0.05)


class Orientation(unittest.TestCase):
    def test_upright_card_is_named_correctly(self):
        frame, _, truth = render_scene(cards=[dict(cx=.5, cy=.5, width=240, pitch=25)], seed=3)
        d = cd.detect_cards(frame)[0]
        self.assertLess(np.linalg.norm(d.corners - truth[0]["corners"], axis=1).max(), 2.5)   # exact TL,TR,BR,BL

    def test_flip_turns_180(self):
        frame, K, _ = render_scene(cards=[dict(cx=.5, cy=.5, width=240)], seed=3)
        d = cd.detect_cards(frame)[0]
        flipped, _ = cd.orient_quad(d.corners, K, flip=True)
        self.assertLess(np.linalg.norm(flipped - np.roll(d.corners, -2, axis=0), axis=1).max(), 1e-3)

    def test_wrong_aspect_has_high_residual(self):
        frame, K, truth = render_scene(cards=[dict(cx=.5, cy=.5, width=240, pitch=35)], seed=3)
        q = truth[0]["corners"].astype(np.float32)
        good, bad = cd.pose_residual(q, K), cd.pose_residual(np.roll(q, -1, axis=0), K)
        self.assertLess(good, 0.15)
        self.assertGreater(bad, 0.4)


class Rectify(unittest.TestCase):
    def test_straightened_card_matches_texture(self):
        frame, _, truth = render_scene(cards=[dict(cx=.5, cy=.5, width=260, pitch=35, yaw=15, roll=20, seed=4)],
                                       seed=4, blur=0.6, noise=2)
        dets = cd.detect_cards(frame)
        self.assertEqual(len(dets), 1)
        # use the detector's corners, but name them like the texture (the detector can't know "up")
        corners = cd.match_order(dets[0].corners, truth[0]["corners"])
        crop = cd.rectify(frame, corners, pad=0.0, out_h=440)
        self.assertEqual(crop.shape[:2], (440, round(440 * cd.CARD_ASPECT)))
        tex = cv2.resize(card_texture(4)[..., :3], (crop.shape[1], crop.shape[0]), interpolation=cv2.INTER_AREA)
        inner = (slice(40, -40), slice(40, -40))                     # ignore the rounded corners / border
        diff = np.abs(crop[inner].astype(float) - tex[inner].astype(float)).mean()
        self.assertLess(diff, 22, f"mean abs difference {diff:.1f}/255")
        corr = np.corrcoef(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)[inner].ravel(),
                           cv2.cvtColor(tex, cv2.COLOR_BGR2GRAY)[inner].ravel())[0, 1]
        self.assertGreater(corr, 0.9, f"correlation {corr:.3f}")

    def test_padding_adds_real_surroundings(self):
        frame, _, truth = render_scene(cards=[dict(cx=.5, cy=.5, width=240)], seed=3)
        d = cd.detect_cards(frame)[0]
        tight, padded = cd.rectify(frame, d.corners, pad=0, out_h=500), cd.rectify(frame, d.corners, pad=0.1, out_h=500)
        self.assertGreater(padded.shape[0], tight.shape[0] * 1.15)
        self.assertGreater(padded.shape[1], tight.shape[1] * 1.15)

    def test_size_follows_apparent_size(self):
        frame, _, _ = render_scene(w=1920, h=1080, cards=[dict(cx=.5, cy=.5, width=800)], seed=3)
        d = cd.detect_cards(frame)[0]
        self.assertLessEqual(cd.rectify(frame, d.corners, pad=0).shape[0], 880)         # clamped


class RefineWithHint(unittest.TestCase):
    def test_hint_from_low_res_tracker_is_sharpened(self):
        frame, _, truth = render_scene(w=1280, h=720, cards=[dict(cx=.5, cy=.5, width=320, pitch=30, roll=8)], seed=6)
        rng = np.random.default_rng(0)
        hint = truth[0]["corners"] + rng.normal(0, 4, (4, 2))                   # a sloppy tracker estimate
        out, ok = cd.refine_with_hint(frame, hint)
        self.assertTrue(ok)
        self.assertLess(corner_error(out, truth[0]["corners"]).max(), 2.5)
        self.assertLess(np.linalg.norm(out - hint, axis=1).min(), 12)            # kept the hint's corner order

    def test_hint_over_empty_table_is_returned_unchanged(self):
        frame, _, _ = render_scene(cards=[], seed=2)
        hint = np.array([[300, 100], [500, 100], [500, 380], [300, 380]], np.float32)
        out, ok = cd.refine_with_hint(frame, hint)
        self.assertFalse(ok)
        np.testing.assert_allclose(out, hint)

    def test_tiny_hint_is_ignored(self):
        frame, _, _ = render_scene(cards=[dict(cx=.5, cy=.5, width=240)], seed=3)
        hint = np.array([[10, 10], [20, 10], [20, 20], [10, 20]], np.float32)
        self.assertFalse(cd.refine_with_hint(frame, hint)[1])


class VisualisationAndCli(unittest.TestCase):
    def test_overlay_changes_only_the_card_area(self):
        frame, _, _ = render_scene(cards=[dict(cx=.5, cy=.5, width=260, pitch=20)], seed=3)
        d = cd.detect_cards(frame)[0]
        sprite = np.zeros((40, 30, 4), np.uint8); sprite[5:35, 4:26] = (0, 0, 255, 255)     # red block
        out = cd.overlay_sprite(frame, d.corners, sprite)
        changed = (np.abs(out.astype(int) - frame.astype(int)).sum(axis=2) > 30)
        self.assertGreater(changed.sum(), 2000)
        poly = np.zeros(frame.shape[:2], np.uint8)
        cv2.fillConvexPoly(poly, d.corners.round().astype(np.int32), 255)
        self.assertEqual(int((changed & (poly == 0)).sum()), 0, "sprite leaked outside the card")

    def test_command_line(self):
        frame, _, _ = render_scene(cards=[dict(cx=.5, cy=.5, width=260)], seed=3)
        with tempfile.TemporaryDirectory() as tmp:
            src, out, crop = (os.path.join(tmp, n) for n in ("in.jpg", "out.jpg", "crop.jpg"))
            cv2.imwrite(src, frame)
            argv = sys.argv
            sys.argv = ["card_detect.py", src, "-o", out, "--crop", crop]
            try:
                cd.main()
            finally:
                sys.argv = argv
            self.assertTrue(os.path.getsize(out) > 1000 and os.path.getsize(crop) > 1000)
            self.assertGreater(cv2.imread(crop).shape[0], 400)



class OuterBorderOnly(unittest.TestCase):
    """The card is its OUTER border. Frames inside the artwork must never be reported or tracked - even when the outer ring is broken."""

    def scene(self, **kw):
        card = dict(cx=.5, cy=.5, width=240, pitch=20)
        card.update(kw)
        return render_scene(cards=[card], seed=3, decoys=True, gray_decoys=True)

    def test_inner_frame_is_ignored_for_both_colours(self):
        for border in ("yellow", "gray"):
            for inner in (True, "offcentre"):
                with self.subTest(border=border, inner=inner):
                    frame, _, truth = self.scene(border=border, inner_frame=inner)
                    dets = cd.detect_cards(frame, max_cards=4)
                    self.assertEqual(len(dets), 1)
                    self.assertLess(corner_error(dets[0].corners, truth[0]["corners"]).max(), 2.5)

    def test_broken_outer_ring_is_rebuilt_from_the_ring_around_the_inner_frame(self):
        for border, inner in (("yellow", True), ("yellow", "offcentre")):
            with self.subTest(border=border, inner=inner):
                frame, _, truth = self.scene(border=border, inner_frame=inner, border_gap=True)
                dets = cd.detect_cards(frame, max_cards=4)
                self.assertEqual(len(dets), 1, "the OUTER card must be found, not the frame inside it")
                self.assertGreater(dets[0].area, 0.12)                       # the whole card (0.15 of the frame), not the inner frame (0.06)
                self.assertLess(corner_error(dets[0].corners, truth[0]["corners"]).max(), 4.0)

    def test_broken_ring_is_never_answered_with_the_inside(self):
        # outer ring has a gap AND there is an off-centre inner frame. Either the card's real outer edge is recovered (edge repair) or nothing
        # is reported - but never the inner frame.
        frame, _, truth = self.scene(yaw=15, inner_frame="offcentre", border_gap=True)
        dets = cd.detect_cards(frame, max_cards=4)
        for d in dets:
            err = corner_error(d.corners, truth[0]["corners"])
            self.assertLess(err.max(), 8.0, f"a detection that is not the card's outer edge: {err.round(1)} px")

    def test_outer_from_inner_says_none_for_a_plain_card(self):
        frame, _, _ = self.scene()
        raw = cd.raw_candidates(frame)
        K = cd.intrinsics(raw["ws"], raw["hs"])
        c = max(raw["cands"], key=lambda c: c["hull_area"])                  # the card (the other candidates are decoys)
        quad, _ = cd.orient_quad(cd.refine_quad(c["contour"], c["approx"]), K)
        self.assertEqual(cd.outer_from_inner(raw["masks"][c["color"]], quad)[0], "none")

    def test_outer_clear_and_suppress_inner(self):
        mask = np.zeros((200, 200), np.uint8)
        quad = np.array([[60, 60], [140, 60], [140, 140], [60, 140]], np.float32)
        self.assertEqual(cd.outer_clear(mask, quad), 1.0)                    # table all round
        mask[:] = 255
        self.assertLess(cd.outer_clear(mask, quad), 0.1)                     # more border colour all round: an inner frame
        big = cd.Detection(np.array([[10, 10], [170, 10], [170, 170], [10, 170]], np.float32), 0.8, 1.0, 0.0)
        small = cd.Detection(quad, 0.2, 1.0, 0.0)
        tiny = cd.Detection(np.array([[90, 90], [100, 90], [100, 100], [90, 100]], np.float32), 0.0025, 1.0, 0.0)
        beside = cd.Detection(np.array([[200, 10], [260, 10], [260, 70], [200, 70]], np.float32), 0.1, 1.0, 0.0)
        kept = cd.suppress_inner([big, small, tiny, beside])
        self.assertEqual([id(k) for k in kept], [id(big), id(tiny), id(beside)])                          # `small` is inside `big`; `tiny` is under 20 % of it (not a frame); `beside` is separate

if __name__ == "__main__":
    unittest.main()
