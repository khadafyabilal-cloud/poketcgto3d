"""Tests for the card text reader (HP / attacks / damage / types), energy cards, the battle opponent and the new pages.
Run:  python -m unittest discover -s tests -v"""
import re
import json, sys, types, unittest
from pathlib import Path

import cv2, numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_server as ts
from test_server import StubRecognizer, agent_mod, app_mod
import card_reader as cr
import textcard

HAVE_OCR = cr.tesseract_available()
need_ocr = unittest.skipUnless(HAVE_OCR, "Tesseract is not installed")


def W(text, left, top, h=40, line=(1, 1, 1), w=None):
    return {"text": text, "left": left, "top": top, "width": w or 12 * len(text), "height": h, "conf": 90, "line": line}


class ParseAttackLines(unittest.TestCase):
    """The structural parser, with hand-made OCR words (no image, no Tesseract)."""

    def test_header_with_damage_and_rule_text(self):
        words = [W("Ember", 300, 50, 50, (1, 1, 1)), W("30", 900, 50, 50, (1, 1, 1)),
                 W("Flip", 50, 140, 28, (1, 1, 2)), W("a", 120, 140, 28, (1, 1, 2)), W("coin.", 150, 140, 28, (1, 1, 2))]
        a = cr.parse_attack_lines(words, 1100)
        self.assertEqual(len(a), 1); self.assertEqual((a[0]["name"], a[0]["damage"]), ("Ember", 30)); self.assertIn("coin", a[0]["text"])

    def test_no_number_is_a_non_damaging_attack(self):
        words = [W("Growl", 300, 50, 52, (1, 1, 1)), W("Lower", 50, 140, 28, (1, 1, 2)), W("their", 150, 140, 28, (1, 1, 2)),
                 W("Surf", 300, 250, 52, (1, 1, 3)), W("50+", 900, 250, 52, (1, 1, 3))]
        a = cr.parse_attack_lines(words, 1100)
        self.assertEqual([x["name"] for x in a], ["Growl", "Surf"])
        self.assertIsNone(a[0]["damage"]); self.assertEqual((a[1]["damage"], a[1]["modifier"]), (50, "+"))

    def test_multiplier_and_lone_symbol_noise(self):
        words = [W("O@", 40, 50, 60), W("Double", 300, 50, 52), W("Kick", 450, 50, 52), W("30×", 900, 50, 52)]
        a = cr.parse_attack_lines(words, 1100)
        self.assertEqual(a[0]["name"], "Double Kick"); self.assertEqual((a[0]["damage"], a[0]["modifier"]), (30, "×"))

    def test_a_number_in_rule_text_is_not_damage(self):
        words = [W("Slam", 300, 50, 52, (1, 1, 1)), W("40", 900, 50, 52, (1, 1, 1)),
                 W("Discard", 50, 140, 28, (1, 1, 2)), W("2", 200, 140, 28, (1, 1, 2)), W("Energy", 250, 140, 28, (1, 1, 2))]
        a = cr.parse_attack_lines(words, 1100)
        self.assertEqual(len(a), 1); self.assertEqual(a[0]["damage"], 40)

    def test_empty(self):
        self.assertEqual(cr.parse_attack_lines([], 1000), [])

    def test_hp_parsing(self):
        self.assertEqual(cr.parse_hp([W("HP", 0, 0), W("120", 50, 0)]), 120)
        self.assertEqual(cr.parse_hp([W("60HP", 0, 0)]), 60)
        self.assertEqual(cr.parse_hp([W("Basic", 0, 0), W("90", 90, 0)]), 90)
        self.assertIsNone(cr.parse_hp([W("Basic", 0, 0)]))
        self.assertIsNone(cr.parse_hp([W("HP", 0, 0), W("9999", 50, 0)]))


class TesseractLookup(unittest.TestCase):
    def setUp(self):
        self.saved = dict(cr._TESS); self.addCleanup(cr._TESS.update, self.saved)
        cr._TESS.update(cmd=None, checked=False)

    @need_ocr
    def test_info_reports_version_and_english(self):
        i = cr.tesseract_info()
        self.assertTrue(i["available"] and i["english"]); self.assertRegex(i["version"], r"^\d")

    def test_env_var_wins_and_missing_means_unavailable(self):
        import os, tempfile
        with tempfile.NamedTemporaryFile(suffix="-tess") as f:
            os.chmod(f.name, 0o755)
            os.environ["TESSERACT_CMD"] = f.name
            try:
                self.assertEqual(cr.tesseract_command(), f.name)
            finally:
                del os.environ["TESSERACT_CMD"]
        cr._TESS.update(cmd=None, checked=False)
        old = cr.shutil.which; cr.shutil.which = lambda n: None
        old_isfile = cr.os.path.isfile; cr.os.path.isfile = lambda p: False
        try:
            self.assertIsNone(cr.tesseract_command()); self.assertFalse(cr.tesseract_info()["available"])
        finally:
            cr.shutil.which = old; cr.os.path.isfile = old_isfile


class Types(unittest.TestCase):
    def test_colours(self):
        for rgb, want in (((226, 70, 40), "fire"), ((50, 130, 220), "water"), ((60, 170, 70), "grass"), ((245, 205, 30), "electric"),
                          ((170, 70, 170), "psychic"), ((225, 225, 225), "normal"), ((30, 30, 35), "dark")):
            self.assertEqual(cr.classify_colour(rgb[::-1]), want, rgb)

    def test_assumed_type(self):
        self.assertEqual(cr.assume_type(["normal", "water"], ["fire"]), "water")          # a coloured energy decides
        self.assertEqual(cr.assume_type(["normal"], ["fire", "flying"]), "fire")           # colorless -> the Pokémon's own type
        self.assertEqual(cr.assume_type([], []), "normal")

    def test_tcg_names(self):
        self.assertEqual([cr.to_type(x) for x in ("Lightning", "Darkness", "Metal", "Colorless", "fire", "Zzz")],
                         ["electric", "dark", "steel", "normal", "fire", None])


@need_ocr
class ReadRealPixels(unittest.TestCase):
    def test_hp_attacks_damage_and_types(self):
        r = cr.read_card(textcard.text_card(), ["fire"])
        self.assertEqual(r["hp"], 120)
        self.assertEqual([(a["name"], a["damage"]) for a in r["attacks"]], [("Ember", 30), ("Growl", None)])
        self.assertEqual(r["attacks"][0]["type"], "fire"); self.assertEqual(r["confidence"], "high")

    def test_costs_modifiers_and_three_attacks(self):
        card = textcard.text_card(hp=60, attacks=(("Thunder Jolt", 30, ["electric", "normal"]), ("Surf", "50+", ["water", "water", "normal"]), ("Rest", None, ["normal"])))
        r = cr.read_card(card, ["electric"])
        self.assertEqual(r["hp"], 60)
        self.assertEqual([(a["name"], a["damage"]) for a in r["attacks"]], [("Thunder Jolt", 30), ("Surf", 50), ("Rest", None)])
        self.assertEqual(r["attacks"][1]["modifier"], "+")
        self.assertEqual([a["type"] for a in r["attacks"]], ["electric", "water", "electric"])      # Rest: colorless -> the Pokémon's type
        self.assertEqual(r["attacks"][1]["cost"], ["water", "water", "normal"])

    def test_blank_card_reads_nothing_and_says_so(self):
        r = cr.read_card(np.full((880, 630, 3), 200, np.uint8), ["fire"])
        self.assertIsNone(r["hp"]); self.assertEqual(r["attacks"], []); self.assertEqual(r["confidence"], "low"); self.assertTrue(r["notes"])


ENERGY = {n: types.SimpleNamespace(set="Base Set", name=n, number=str(100 + i)) for i, n in enumerate(
    ["Fire Energy", "Basic Lightning Energy", "Double Colorless Energy", "Rainbow Energy", "Darkness Energy"])}
CHAR = types.SimpleNamespace(set="Base Set", name="Charmander", number="46")


class AgentStats(ts.Base):
    def setUp(self):
        super().setUp()
        self.card = textcard.text_card()
        StubRecognizer.cards = [(self.card, CHAR)]
        self.reads = 0
        real = cr.read_card

        def counting(*a, **k):
            self.reads += 1
            return real(*a, **k)
        cr.read_card = counting
        self.addCleanup(setattr, cr, "read_card", real)

    @need_ocr
    def test_stats_are_attached_to_the_scan_result(self):
        r = self.agent.run_card(self.card)
        s = r["stats"]
        self.assertEqual(s["hp"], 120); self.assertEqual(s["source"], "ocr")
        self.assertEqual([(a["name"], a["damage"], a["type"]) for a in s["attacks"]], [("Ember", 30, "fire"), ("Growl", None, "fire")])

    @need_ocr
    def test_a_card_is_read_once_even_across_restarts(self):
        self.agent.run_card(self.card); self.agent.run_card(self.card)           # 2nd = scan cache
        self.assertEqual(self.reads, 1)
        self.assertTrue(list((self.tmp / "card_cache").glob("*.json")))
        fresh = agent_mod.PokemonCardAgent(); r = fresh.run_card(self.card)       # new process: scan cache empty, disk cache not
        self.assertEqual(self.reads, 1); self.assertEqual(r["stats"]["hp"], 120)

    @need_ocr
    def test_upside_down_card_is_read_upright(self):
        r = self.agent.run_card(cv2.rotate(self.card, cv2.ROTATE_180))
        self.assertTrue(r["rotated"]); self.assertEqual(r["stats"]["hp"], 120)

    def test_without_tesseract_defaults_are_used_and_explained(self):
        old = cr.tesseract_available; cr.tesseract_available = lambda: False
        self.addCleanup(setattr, cr, "tesseract_available", old)
        r = self.agent.run_card(self.card)
        self.assertEqual(r["stats"]["hp"], 100); self.assertEqual(r["stats"]["source"], "default")
        self.assertTrue(any("Tesseract" in n for n in r["stats"]["notes"])); self.assertEqual(len(r["stats"]["attacks"]), 2)
        self.assertFalse((self.tmp / "card_cache").exists() and list((self.tmp / "card_cache").glob("*.json")))   # defaults are never cached

    def test_manual_pick_gets_default_stats(self):
        self.assertGreaterEqual(len(self.agent.lookup("pikachu")["stats"]["attacks"]), 1)

    def test_complete_stats_fills_gaps(self):
        s = agent_mod.PokemonCardAgent.complete_stats({"hp": 70, "attacks": [], "source": "ocr", "notes": []}, ["water"])
        self.assertEqual(s["hp"], 70); self.assertEqual(s["attacks"][1]["type"], "water"); self.assertEqual(s["source"], "ocr")                   # HP was really read, so it is worth caching


class ClientOcr(ts.Base):
    """Server half of the browser-side OCR (tesseract.js) fallback. The 'browser' is simulated with the real Tesseract program."""
    def setUp(self):
        super().setUp()
        self.card = textcard.text_card()
        StubRecognizer.cards = [(self.card, CHAR)]
        old = cr.tesseract_available; cr.tesseract_available = lambda: False          # pretend the server has no Tesseract
        self.addCleanup(setattr, cr, "tesseract_available", old)
        self.c = app_mod.create_app(self.agent).test_client()

    def browser_words(self, token):
        """What tesseract.js would send back: words read from the PNGs the server published (done here with the real CLI)."""
        out = {}
        for part, psms in (("hp", (7, 6)), ("attack", (6,))):
            png = self.c.get(f"/api/battle/ocr/{token}/{part}.png")
            self.assertEqual(png.status_code, 200); self.assertEqual(png.mimetype, "image/png")
            img = cv2.imdecode(np.frombuffer(png.data, np.uint8), cv2.IMREAD_GRAYSCALE)
            out[part] = [[dict(w, line=list(w["line"])) for w in cr.ocr_words(img, psm=p)] for p in psms]
        return {"token": token, "hp": out["hp"], "attack": out["attack"][0]}

    def test_scan_without_tesseract_asks_the_browser(self):
        s = self.agent.run_card(self.card)["stats"]
        self.assertTrue(s["client_ocr"]); self.assertEqual(s["source"], "default"); self.assertEqual(s["hp"], 100)
        self.assertTrue(any("browser" in n for n in s["notes"]))

    @need_ocr
    def test_browser_words_become_real_stats_and_are_cached(self):
        token = self.agent.run_card(self.card)["stats"]["client_ocr"]
        r = self.c.post("/api/battle/ocr-words", json=self.browser_words(token))
        self.assertEqual(r.status_code, 200)
        s = r.get_json()["stats"]
        self.assertEqual(s["hp"], 120); self.assertEqual([(a["name"], a["damage"]) for a in s["attacks"]], [("Ember", 30), ("Growl", None)])
        self.assertNotIn("client_ocr", s)
        self.assertTrue(list((self.tmp / "card_cache").glob("*.json")))               # next time no OCR at all
        again = self.agent.run_card(self.card)
        self.assertEqual(again["stats"]["hp"], 120)                                    # scan-cache entry was upgraded too
        self.assertEqual(self.c.get(f"/api/battle/ocr/{token}/hp.png").status_code, 404)   # job is finished

    def test_bad_requests(self):
        token = self.agent.run_card(self.card)["stats"]["client_ocr"]
        self.assertEqual(self.c.get("/api/battle/ocr/nope/hp.png").status_code, 404)
        self.assertEqual(self.c.get(f"/api/battle/ocr/{token}/evil.png").status_code, 404)
        self.assertEqual(self.c.post("/api/battle/ocr-words", json={"token": "nope"}).status_code, 404)
        self.assertEqual(self.c.post("/api/battle/ocr-words", data="x").status_code, 404)

    def test_hostile_words_are_sanitised(self):
        token = self.agent.run_card(self.card)["stats"]["client_ocr"]
        junk = [{"text": "A" * 500, "left": -5, "top": 10 ** 9, "width": "x", "height": 3}, "str", None, {"text": "Ok", "left": 1, "top": 2, "width": 3, "height": 4}]
        r = self.c.post("/api/battle/ocr-words", json={"token": token, "hp": [junk, "no", 5], "attack": junk * 300})
        self.assertEqual(r.status_code, 200)
        s = r.get_json()["stats"]
        self.assertEqual(s["hp"], 100); self.assertEqual(s["source"], "default")        # nothing usable -> defaults, never an error

    def test_sanitize_words_limits(self):
        w = cr.sanitize_words([{"text": "Hi", "left": 5, "top": 6, "width": 7, "height": 8, "line": [0, 0, 1]}] * 1000)
        self.assertEqual(len(w), 400); self.assertEqual(w[0]["line"], (0, 0, 1))
        self.assertEqual(cr.sanitize_words("nope"), [])

    def test_health_reports_tesseract(self):
        self.assertIn("tesseract", self.c.get("/api/health").get_json())

    def test_pages_hand_the_script_url_to_the_browser(self):
        self.assertIn("tesseract.js@", self.c.get("/battle").get_data(as_text=True))


class RealSetObjects(ts.Base):
    """The real recogniser's card.set is an object whose str() is a giant dump (this broke card_cache file names on Windows)."""
    class BigSet:
        id, name = "xy12", "Evolutions"
        def __str__(self): return "Set(id='xy12', images=SetImage(symbol='https://images.pokemontcg.io/xy12/symbol.png'), legalities=..., " + "x" * 400 + ")"

    def setUp(self):
        super().setUp()
        self.card = textcard.text_card()
        StubRecognizer.cards = [(self.card, types.SimpleNamespace(set=self.BigSet(), name="Charmander", number="6/108"))]

    def test_set_is_a_short_name(self):
        r = self.agent.run_card(self.card)
        self.assertEqual(r["card"]["set"], "Evolutions"); self.assertEqual(r["card"]["number"], "6/108")

    @need_ocr
    def test_cache_file_name_is_short_and_legal(self):
        self.agent.run_card(self.card)
        names = [p.name for p in (self.tmp / "card_cache").glob("*.json")]
        self.assertEqual(len(names), 1); self.assertLess(len(names[0]), 60)
        self.assertRegex(names[0], r"^[A-Za-z0-9_]+\.json$")

    def test_unwritable_cache_never_breaks_a_scan(self):
        (self.tmp / "card_cache").write_text("a file where the folder should be")       # mkdir/write will fail
        r = self.agent.run_card(self.card)
        self.assertTrue(r["stats"]["attacks"]); self.assertEqual(r["card"]["name"], "Charmander")

    def test_different_sets_same_number_get_different_files(self):
        f = agent_mod.PokemonCardAgent.card_cache_file
        self.assertNotEqual(f({"set": "Base Set", "number": "4"}), f({"set": "Base-Set", "number": "4"}))

    def test_rejected_guess_still_matches_with_clean_labels(self):
        self.agent.recognizer.exec = lambda p: [types.SimpleNamespace(set=self.BigSet(), name="Charmander", number="6/108"),
                                                 types.SimpleNamespace(set=self.BigSet(), name="Charmeleon", number="7/108")]
        a = self.agent.recognize.__self__.recognize.__func__  # noqa: F841
        self.assertEqual(self.agent.recognize("x.jpg", exclude={("Evolutions", "6/108")})["name"], "Charmeleon")


class Energy(ts.Base):
    def test_energy_names(self):
        f = agent_mod.PokemonCardAgent.read_energy
        self.assertEqual(f("Fire Energy"), {"type": "fire", "special": False, "name": "Fire Energy"})
        self.assertEqual(f("Basic Lightning Energy")["type"], "electric"); self.assertFalse(f("Basic Lightning Energy")["special"])
        self.assertEqual(f("Darkness Energy")["type"], "dark"); self.assertEqual(f("Metal Energy")["type"], "steel")
        d = f("Double Colorless Energy"); self.assertEqual(d["type"], "normal"); self.assertTrue(d["special"])
        r = f("Rainbow Energy"); self.assertIsNone(r["type"]); self.assertTrue(r["special"])
        for n in ("Pikachu", "Professor Oak", "", None): self.assertIsNone(f(n))

    def test_energy_card_is_read_cached_and_has_no_model(self):
        card = ts.upright_card(77)
        StubRecognizer.cards = [(card, ENERGY["Fire Energy"])]
        r = self.agent.run_card(card)
        self.assertEqual(r["card_kind"], "energy"); self.assertEqual(r["energy"]["type"], "fire")
        self.assertIsNone(r["model_url"]); self.assertNotIn("stats", r); self.assertNotIn("result_id", r)
        again = self.agent.run_card(card)
        self.assertTrue(again["cached"]); self.assertEqual(again["energy"]["type"], "fire"); self.assertEqual(StubRecognizer.calls, 1)

    def test_trainer_card_is_labelled(self):
        card = ts.upright_card(78)
        StubRecognizer.cards = [(card, ts.TRAINER)]
        self.assertEqual(self.agent.run_card(card)["card_kind"], "trainer")

    def test_energy_without_a_type_word_uses_the_symbol_colour(self):
        card = np.full((880, 630, 3), 60, np.uint8); cv2.circle(card, (315, 400), 90, (40, 70, 226), -1)       # red disc in the middle (BGR)
        StubRecognizer.cards = [(card, types.SimpleNamespace(set="X", name="Basic Energy", number="1"))]
        self.assertEqual(self.agent.run_card(card)["energy"]["type"], "fire")


class Opponent(ts.Base):
    def setUp(self):
        super().setUp()
        g = self.tmp / "models" / "gen1"; g.mkdir(parents=True)
        (g / "025.glb").write_bytes(b"glTF" + bytes(20))
        (self.tmp / "pokemon_cache" / "25.json").write_text(json.dumps({"id": 25, "slug": "pikachu", "name": "Pikachu", "types": ["electric"],
                                                                         "height_m": .4, "weight_kg": 6, "artwork": None}))

    def test_random_opponent_has_a_model_hp_and_typed_moves(self):
        r = self.agent.opponent()
        self.assertEqual(r["model_url"], "/models/gen1/025.glb")
        s = r["stats"]; self.assertIn(s["hp"], range(60, 181, 10))
        self.assertEqual([(m["name"], m["damage"], m["type"]) for m in s["attacks"]], [("Tackle", 20, "normal"), ("Thunder Shock", 30, "electric")])

    def test_exclude_is_respected_when_possible(self):
        g = self.tmp / "models" / "gen1"; (g / "004.glb").write_bytes(b"glTF" + bytes(20))
        for _ in range(8):
            self.assertEqual(self.agent.opponent(exclude=[25])["pokemon"]["id"], 4)

    def test_pool_uses_only_available_models_and_rng_is_injectable(self):
        class R:
            def choice(self, seq): return seq[0]
        self.assertEqual(self.agent.opponent(rng=R())["stats"]["hp"], 60)


class Pages(ts.Base):
    def setUp(self):
        super().setUp()
        self.c = app_mod.create_app(self.agent).test_client()

    def test_home_page_lists_both_modes_and_about(self):
        h = self.c.get("/").get_data(as_text=True)
        self.assertIn("<title>PokéCard 3D</title>", h)
        for href in ("/detect?go=ar", "/detect?go=viewer", "/detect", "/battle?go=ar", "/battle", "/scan"):
            self.assertIn(f'href="{href}"', h)
        about = h.split('id="about"')[1].split("</section>")[0]
        text = re.sub(r"<[^>]+>", " ", about.split("<p>")[1].split("</p>")[0]).strip()
        self.assertEqual(len(re.findall(r"[.!?](?:\s|$)", text)), 2, text)               # the About is exactly two sentences
        self.assertEqual(self.c.get("/scan").status_code, 200)

    def test_detect_and_battle_pages(self):
        d = self.c.get("/detect").get_data(as_text=True); b = self.c.get("/battle").get_data(as_text=True)
        self.assertIn("<title>Detect · PokéCard 3D</title>", d); self.assertIn('data-mode="detect"', d)
        self.assertIn("<title>Battle · PokéCard 3D</title>", b); self.assertIn('data-mode="battle"', b)
        for html in (d, b):
            self.assertNotIn("{{", html); self.assertNotIn("Battle mode", html)
            self.assertIn('id="home"', html)
        self.assertIn('id="moves"', b)
        for needle in ('id="arToggle"', 'id="res"', 'id="resMain"', 'id="photo"', 'id="viewer"'):       # AR switch, results, teach form, photo input, 3D viewer
            self.assertIn(needle, d)

    def test_opponent_route(self):
        g = self.tmp / "models" / "gen1"; g.mkdir(parents=True); (g / "025.glb").write_bytes(b"glTF" + bytes(20))
        d = self.c.get("/api/battle/opponent?exclude=4,6,abc").get_json()
        self.assertEqual(d["pokemon"]["id"], 25); self.assertTrue(d["stats"]["attacks"])


if __name__ == "__main__":
    unittest.main()
