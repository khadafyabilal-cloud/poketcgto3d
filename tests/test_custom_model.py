import json, struct, sys, unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class CustomModel(unittest.TestCase):
    def test_charizard_glb_is_valid_and_small(self):
        data = (ROOT / "models/custom/charizard.glb").read_bytes()
        self.assertEqual(data[:4], b"glTF")
        self.assertEqual(struct.unpack("<I", data[8:12])[0], len(data))
        js = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
        self.assertEqual(js["meshes"][0]["primitives"][0]["attributes"].keys(), {"POSITION", "NORMAL", "COLOR_0"})
        self.assertLess(len(data), 6 * 1024 * 1024)

    def test_agent_maps_species_6_to_it(self):
        src = (ROOT / "agent.py").read_text(encoding="utf-8")
        self.assertIn('CUSTOM_MODELS = {6: "custom/charizard.glb"}', src)
        self.assertLess(src.index("CUSTOM_MODELS.get(pid)"), src.index("self.models.get(pid, card"))

    def test_served_by_flask(self):
        import app as A
        r = A.create_app(agent=object()).test_client().get("/models/custom/charizard.glb")
        self.assertEqual(r.status_code, 200); self.assertEqual(r.data[:4], b"glTF"); r.close()


if __name__ == "__main__":
    unittest.main()


# ---------------------------------------------------------------- Gen 1 models + relief fallback
sys.path.insert(0, str(ROOT / "tests"))
import types
import numpy as np
from PIL import Image, ImageDraw
import test_server as ts                      # installs the stand-ins for trimesh / the recognizer
import agent as agent_mod


def fake_agent(pid, name):
    a = object.__new__(agent_mod.PokemonCardAgent)
    a.match_species = lambda n: pid
    a.fetch_pokemon = lambda p: {"id": p, "slug": name.lower(), "name": name, "types": ["grass"], "height_m": 0.7, "weight_kg": 6.9, "artwork": "http://x/a.png"}
    a.models = types.SimpleNamespace(get=lambda *x: None, count=0)
    a.feedback = types.SimpleNamespace(demoted_stages=lambda dex: set())      # no user ratings in these tests
    return a


class Gen1AndRelief(unittest.TestCase):
    def test_all_151_gen1_models_exist_and_are_valid(self):
        for n in range(1, 152):
            data = (ROOT / "models/gen1" / f"{n:03d}.glb").read_bytes()
            self.assertEqual(data[:4], b"glTF", n)
            js = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
            self.assertTrue(js["meshes"][0]["primitives"], n)
            self.assertLess(len(data), 3 * 1024 * 1024, n)

    def test_gen1_species_use_their_model_only(self):
        res = fake_agent(1, "Bulbasaur")._finish({"name": "Bulbasaur", "set": "x", "number": "1"}, {}, lambda *a: None)
        self.assertEqual(res["model_url"], "/models/gen1/001.glb"); self.assertEqual(res["model_source"]["kind"], "gen1")

    def test_charizard_stays_custom(self):
        res = fake_agent(6, "Charizard")._finish({"name": "Charizard", "set": "x", "number": "4"}, {}, lambda *a: None)
        self.assertEqual(res["model_url"], "/models/custom/charizard.glb")

    def test_later_generations_get_a_relief_from_artwork(self):
        import io, tempfile
        a = fake_agent(252, "Treecko")
        img = Image.new("RGBA", (200, 260), (0, 0, 0, 0)); ImageDraw.Draw(img).ellipse((20, 20, 180, 240), fill=(40, 180, 90, 255))
        a._download_artwork = lambda pk: img
        a._preview = lambda im: "data:,"
        res = a._finish({"name": "Treecko", "set": "x", "number": "1"}, {}, lambda *x: None)
        self.assertEqual(res["model_source"]["kind"], "relief"); self.assertEqual(res["model_url"], "/models/relief/252.glb")
        f = ROOT / "models/relief/252.glb"
        try: self.assertEqual(f.read_bytes()[:4], b"glTF")
        finally: f.unlink(missing_ok=True)


class ReliefShape(unittest.TestCase):
    def test_closed_standing_centred_and_textured(self):
        import relief, tempfile
        img = Image.new("RGBA", (120, 160), (0, 0, 0, 0)); ImageDraw.Draw(img).ellipse((10, 10, 110, 150), fill=(200, 50, 50, 255))
        p = Path(tempfile.mkdtemp()) / "r.glb"; relief.build_relief(img, p)
        data = p.read_bytes(); js = json.loads(data[20:20 + struct.unpack("<I", data[12:16])[0]])
        mn, mx = js["accessors"][0]["min"], js["accessors"][0]["max"]
        self.assertAlmostEqual(mn[1], 0, places=5); self.assertAlmostEqual(mx[1], 1, delta=0.02)
        self.assertAlmostEqual(mn[0], -mx[0], delta=0.02); self.assertLess(mx[2], 0.4); self.assertGreater(mx[2], 0.05)
        self.assertIn("baseColorTexture", js["materials"][0]["pbrMetallicRoughness"])
