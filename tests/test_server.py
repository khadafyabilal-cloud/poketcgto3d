"""
Server-side tests: agent.run_card / lookup / cache, the Flask routes, tls.py, vendor.py, fetch_vendor.py.

The card-recognition model and trimesh are replaced by small stand-ins (they need a GPU-sized install);
everything else - the real agent.py code, OpenCV straightening, Flask - runs for real.
Run:  python -m unittest discover -s tests -v
"""
import io, json, shutil, ssl, sys, tempfile, threading, types, unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))

# ---------------------------------------------------------------- stand-ins for heavy third-party packages
try:
    import trimesh  # noqa: F401
except ImportError:
    tm = types.ModuleType("trimesh")

    class Trimesh:                                            # just enough for agent.build_model
        def __init__(self, vertices, faces, vertex_colors=None, process=False):
            self.vertices, self.faces, self.vertex_normals = vertices, faces, None

        def export(self, path):
            Path(path).write_bytes(b"glTF" + bytes(8) + b"stub-mesh")
    tm.Trimesh = Trimesh
    sys.modules["trimesh"] = tm
for name in ("pokemon_card_recognizer", "pokemon_card_recognizer.api", "pokemon_card_recognizer.api.card_recognizer"):
    sys.modules.setdefault(name, types.ModuleType(name))
sys.modules["pokemon_card_recognizer.api.card_recognizer"].CardRecognizer = object
sys.modules["pokemon_card_recognizer.api.card_recognizer"].OperatingMode = types.SimpleNamespace(SINGLE_IMAGE=1)

import agent as agent_mod                                      # noqa: E402
import app as app_mod                                          # noqa: E402
import card_detect as cd                                       # noqa: E402
import glb_library, tls, vendor                                # noqa: E402
from synth import render_scene                                 # noqa: E402

PIKACHU = types.SimpleNamespace(set="Base Set", name="Pikachu", number="58")
CHARMANDER = types.SimpleNamespace(set="Base Set", name="Charmander", number="46")
TRAINER = types.SimpleNamespace(set="Base Set", name="Professor Oak", number="88")


def face(img):
    """The card face (inside the border) as a small normalised grayscale image."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img[int(.17 * h):int(.83 * h), int(.17 * w):int(.83 * w)], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (48, 64), interpolation=cv2.INTER_AREA).astype(np.float32)
    return (g - g.mean()) / (g.std() + 1e-3)


def same_card(a, b):
    """Normalised correlation of the card faces: ~1 for the same card, ~0 for another one or the same one upside down."""
    return float((face(a) * face(b)).mean()) > 0.8


class StubRecognizer:
    """Reads a card only when it is shown UPRIGHT and looks like one of the registered reference cards."""
    cards = []                                                 # [(reference BGR image, prediction)]
    calls = 0

    def __init__(self, mode=None, set_name=None):
        self.classifier = types.SimpleNamespace(reference=types.SimpleNamespace(lookup_card_prediction=lambda card_prediction: card_prediction))

    def exec(self, path):
        StubRecognizer.calls += 1
        img = cv2.imread(path)
        for ref, pred in StubRecognizer.cards:
            if same_card(img, ref):
                return [pred]
        return None


def upright_card(seed, w=1280, h=720, **pose):
    """A clean, straightened view of synthetic card `seed` (what a perfect tracker + rectify would produce)."""
    frame, _, truth = render_scene(w, h, cards=[dict(cx=.5, cy=.5, width=320, seed=seed, **pose)], seed=1, blur=0.3, noise=1)
    return cd.rectify(frame, truth[0]["corners"], pad=agent_mod.CARD_PAD)


class Base(unittest.TestCase):
    """Builds a real PokemonCardAgent around the stand-ins, with every cache/output folder redirected to a temp dir."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "species.json").write_text(json.dumps({"pikachu": 25, "charmander": 4, "charizard": 6}))
        cache = self.tmp / "pokemon_cache"; cache.mkdir()
        for pid, slug, types_ in ((25, "pikachu", ["electric"]), (4, "charmander", ["fire"]), (6, "charizard", ["fire", "flying"])):
            (cache / f"{pid}.json").write_text(json.dumps({"id": pid, "slug": slug, "name": slug.title(), "types": types_,
                                                           "height_m": 0.4, "weight_kg": 6.0, "artwork": None}))
        packs = self.tmp / "packs"; packs.mkdir()
        (packs / "25.glb").write_bytes(b"glTF" + bytes(32))     # pikachu has a pack model, charmander only a sprite
        (self.tmp / "models").mkdir()
        patches = [(agent_mod, "ROOT", self.tmp), (agent_mod, "POKE_CACHE", cache), (agent_mod, "MODELS", self.tmp / "models"),
                   (agent_mod, "CardRecognizer", StubRecognizer), (glb_library, "CACHE_DIR", self.tmp / "glb"),
                   (agent_mod, "GlbLibrary", lambda: glb_library.GlbLibrary(packs)), (app_mod, "MODELS", self.tmp / "models")]
        for obj, name, value in patches:
            old = getattr(obj, name)
            setattr(obj, name, value)
            self.addCleanup(setattr, obj, name, old)
        StubRecognizer.cards, StubRecognizer.calls = [], 0
        self.agent = agent_mod.PokemonCardAgent()


class AgentBattleScan(Base):
    def setUp(self):
        super().setUp()
        self.pika, self.char = upright_card(11), upright_card(22)
        StubRecognizer.cards = [(self.pika, PIKACHU), (self.char, CHARMANDER)]

    def test_straightened_card_is_identified_and_gets_a_model(self):
        res = self.agent.run_card(self.pika)
        self.assertEqual(res["card"]["name"], "Pikachu")
        self.assertEqual(res["pokemon"]["id"], 25)
        self.assertTrue(res["model_url"].startswith("/models/glb/25"))
        self.assertFalse(res["rotated"]); self.assertFalse(res["cached"])
        self.assertEqual(StubRecognizer.calls, 1)

    def test_same_card_again_skips_the_recogniser(self):
        self.agent.run_card(self.pika)
        again = self.agent.run_card(upright_card(11, pitch=0))      # re-rendered, slightly different pixels
        self.assertTrue(again["cached"]); self.assertEqual(again["card"]["name"], "Pikachu")
        self.assertEqual(StubRecognizer.calls, 1, "recogniser ran again for a card it already knew")
        self.assertEqual(again["trace"][0]["step"], "Cache")

    def test_cache_also_matches_the_card_upside_down(self):
        self.agent.run_card(self.pika)
        res = self.agent.run_card(cv2.rotate(self.pika, cv2.ROTATE_180))
        self.assertTrue(res["cached"]); self.assertTrue(res["rotated"])
        self.assertEqual(StubRecognizer.calls, 1)

    def test_a_different_card_is_not_mistaken_for_a_cached_one(self):
        self.agent.run_card(self.pika)
        res = self.agent.run_card(self.char)
        self.assertFalse(res["cached"]); self.assertEqual(res["card"]["name"], "Charmander")
        self.assertEqual(StubRecognizer.calls, 2)

    def test_upside_down_card_is_read_after_turning_it(self):
        res = self.agent.run_card(cv2.rotate(self.pika, cv2.ROTATE_180))
        self.assertEqual(res["card"]["name"], "Pikachu")
        self.assertTrue(res["rotated"])
        self.assertEqual(StubRecognizer.calls, 2)                   # as given, then turned

    def test_unreadable_card_gives_a_message_and_is_not_cached(self):
        blank = np.full_like(self.pika, 127)
        res = self.agent.run_card(blank)
        self.assertIsNone(res["model_url"]); self.assertIn("Couldn't read", res["message"])
        self.assertEqual(len(self.agent.scan_cache), 0)

    def test_trainer_card_has_no_model_and_is_not_cached(self):
        StubRecognizer.cards = [(self.pika, TRAINER)]
        res = self.agent.run_card(self.pika)
        self.assertIsNone(res["model_url"]); self.assertIn("doesn't look like a Pokémon", res["message"])
        self.assertEqual(len(self.agent.scan_cache), 0)

    def test_pokemon_without_pack_model_falls_back_to_the_sprite_pipeline(self):
        res = self.agent.run_card(self.char)                        # charmander: sprites/004_charmander.jpg -> voxel .glb
        self.assertEqual(res["model_source"]["kind"], "custom")
        self.assertRegex(res["model_url"], r"^/models/4_custom\.glb$")
        self.assertTrue((self.tmp / "models" / "4_custom.glb").exists())

    def test_cache_is_bounded(self):
        old = agent_mod.SCAN_CACHE_MAX
        agent_mod.SCAN_CACHE_MAX = 1
        self.addCleanup(setattr, agent_mod, "SCAN_CACHE_MAX", old)
        self.agent.run_card(self.pika); self.agent.run_card(self.char)
        self.assertEqual(len(self.agent.scan_cache), 1)

    def test_signature_separates_cards_with_the_same_template(self):
        d = lambda a, b: float(np.abs(self.agent._signature(a) - self.agent._signature(b)).mean())
        self.assertLess(d(self.pika, upright_card(11, pitch=0)), agent_mod.SCAN_CACHE_DIST)
        for other in (22, 33, 44, 55):
            self.assertGreater(d(self.pika, upright_card(other)), 1.5 * agent_mod.SCAN_CACHE_DIST)

    def test_pad_constants_agree(self):
        self.assertEqual(app_mod.CARD_PAD, agent_mod.CARD_PAD)


class AgentLookupAndLegacy(Base):
    def test_lookup_by_name_and_number(self):
        for q in ("pikachu", "Pikachu", "25"):
            with self.subTest(q):
                res = self.agent.lookup(q)
                self.assertEqual(res["pokemon"]["id"], 25); self.assertIsNone(res["card"])
                self.assertTrue(res["model_url"].startswith("/models/glb/25"))

    def test_lookup_unknown(self):
        for q in ("zzzzzz", "0"):
            res = self.agent.lookup(q)
            self.assertIsNone(res["model_url"]); self.assertIn("isn't a Pokémon", res["message"])

    def test_original_upload_flow_is_unchanged(self):
        StubRecognizer.cards = [(upright_card(11), PIKACHU)]
        ok, jpg = cv2.imencode(".jpg", upright_card(11))
        res = self.agent.run(jpg.tobytes())
        self.assertEqual(res["card"], {"set": "Base Set", "name": "Pikachu", "number": "58"})
        self.assertEqual(res["pokemon"]["name"], "Pikachu"); self.assertTrue(res["model_url"])
        self.assertNotIn("rotated", res)
        self.assertEqual([t["step"] for t in res["trace"]][:3], ["Recognize", "Recognize", "Identify"])


class Routes(Base):
    def setUp(self):
        super().setUp()
        self.pika = upright_card(11)
        StubRecognizer.cards = [(self.pika, PIKACHU), (upright_card(22), CHARMANDER)]
        self.app = app_mod.create_app(self.agent)
        self.client = self.app.test_client()

    def frame(self, seed=11, **pose):
        pose = {"pitch": 25, "roll": 8, **pose}
        frame, _, _ = render_scene(1280, 720, cards=[dict(cx=.5, cy=.5, width=300, seed=seed, **pose)], seed=5)
        det = cd.detect_cards(frame)[0]
        ok, jpg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return jpg.tobytes(), det.corners

    def post(self, jpg, corners, **extra):
        data = {"image": (io.BytesIO(jpg), "frame.jpg")}
        if corners is not None:
            data["corners"] = corners if isinstance(corners, str) else json.dumps(np.asarray(corners).tolist())
        data.update(extra)
        return self.client.post("/api/battle/scan", data=data, content_type="multipart/form-data")

    # ---- the scan endpoint
    def test_scan_end_to_end(self):
        jpg, corners = self.frame()
        r = self.post(jpg, corners + np.random.default_rng(0).normal(0, 3, (4, 2)))      # a sloppy browser estimate
        self.assertEqual(r.status_code, 200, r.get_data(as_text=True))
        d = r.get_json()
        self.assertEqual(d["card"]["name"], "Pikachu")
        self.assertTrue(d["model_url"].startswith("/models/glb/25"))
        self.assertTrue(d["scan"]["refined"])
        self.assertTrue(d["crop_jpeg"].startswith("data:image/jpeg;base64,"))
        self.assertFalse(d["rotated"])
        self.assertGreater(d["scan"]["crop_size"][1], 400)

    def test_scan_snapshot_mode_skips_server_side_detection(self):
        # the browser already straightened the card: the server must NOT run card_detect and must not send a thumbnail back
        calls = []
        real = cd.refine_with_hint
        cd.refine_with_hint = lambda *a, **k: calls.append(1) or real(*a, **k)
        try:
            ok, jpg = cv2.imencode(".jpg", self.pika)
            r = self.client.post("/api/battle/scan", data={"image": (io.BytesIO(jpg.tobytes()), "card.jpg"), "snapshot": "1"}, content_type="multipart/form-data")
        finally:
            cd.refine_with_hint = real
        d = r.get_json()
        self.assertEqual(r.status_code, 200, d)
        self.assertEqual(d["card"]["name"], "Pikachu"); self.assertEqual(d["scan"]["mode"], "snapshot")
        self.assertNotIn("crop_jpeg", d); self.assertEqual(calls, [])

    def test_scan_without_corners_finds_the_card_itself(self):
        jpg, corners = self.frame()
        d = self.post(jpg, None).get_json()                      # a plain photo (AR off)
        self.assertEqual(d["scan"]["mode"], "auto"); self.assertTrue(d["scan"]["detected"])
        self.assertEqual(d["card"]["name"], "Pikachu")
        got = np.array(d["scan"]["corners"])
        self.assertLess(np.abs(got - corners).max(), 3)          # the outline the page draws on the photo
        ok, plain = cv2.imencode(".jpg", np.full((400, 600, 3), 90, np.uint8))
        e = self.post(plain.tobytes(), None).get_json()          # no card border at all: still answers (whole photo), says so
        self.assertEqual((e["scan"]["mode"], e["scan"]["detected"]), ("auto", False))

    def test_cache_headers(self):
        js = self.client.get("/static/js/detect.js?v=abc")
        self.assertIn("immutable", js.headers["Cache-Control"])                      # versioned scripts are cached for a year
        self.assertNotIn("immutable", self.client.get("/static/js/detect.js").headers.get("Cache-Control", ""))

    def test_pages_are_cached_and_answer_304(self):
        a = self.client.get("/detect")
        etag = a.headers["ETag"]
        b = self.client.get("/detect", headers={"If-None-Match": etag})
        self.assertEqual((a.status_code, b.status_code, len(b.data)), (200, 304, 0))      # an unchanged page costs no body
        self.assertEqual(self.client.get("/detect").data, a.data)
        self.assertNotEqual(self.client.get("/battle").headers["ETag"], etag)             # modes differ

    def test_scan_cache_across_requests(self):
        self.assertEqual(self.post(*self.frame()).get_json()["cached"], False)
        calls = StubRecognizer.calls
        self.assertEqual(self.post(*self.frame(pitch=35, yaw=10)).get_json()["cached"], True)
        self.assertEqual(StubRecognizer.calls, calls)

    def test_scan_upside_down_card_reports_rotated(self):
        d = self.post(*self.frame(roll=180)).get_json()          # the tracker names the printed bottom "top"
        self.assertEqual(d["card"]["name"], "Pikachu"); self.assertTrue(d["rotated"])

    def test_scan_rejects_bad_requests(self):
        jpg, corners = self.frame()
        good = json.dumps(corners.tolist())
        cases = {
            "no image": (lambda: self.client.post("/api/battle/scan", data={"corners": good}, content_type="multipart/form-data"), "No image"),
            "not an image": (lambda: self.post(b"hello", corners), "not an image"),
            "bad json": (lambda: self.post(jpg, "[[1,2"), "corners must be JSON"),
            "three corners": (lambda: self.post(jpg, [[1, 1], [2, 2], [3, 3]]), "corners must be JSON"),
            "nan": (lambda: self.post(jpg, '[[0,0],[1,0],[1,1],[NaN,1]]'), "invalid numbers"),
            "outside": (lambda: self.post(jpg, [[-500, 0], [100, 0], [100, 100], [0, 100]]), "outside the image"),
            "bow-tie": (lambda: self.post(jpg, [[100, 100], [300, 300], [300, 100], [100, 300]]), "convex"),
            "tiny": (lambda: self.post(jpg, [[100, 100], [110, 100], [110, 110], [100, 110]]), "too small"),
        }
        for name, (call, expect) in cases.items():
            with self.subTest(name):
                r = call()
                self.assertEqual(r.status_code, 400, name)
                self.assertIn(expect, r.get_json()["error"])

    def test_scan_on_an_empty_table_still_answers(self):
        frame, _, _ = render_scene(1280, 720, cards=[], seed=2)
        ok, jpg = cv2.imencode(".jpg", frame)
        r = self.post(jpg.tobytes(), [[400, 150], [700, 150], [700, 570], [400, 570]])
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertFalse(d["scan"]["refined"]); self.assertIsNone(d["model_url"]); self.assertTrue(d["message"])

    def test_agent_errors_become_json(self):
        self.agent.run_card = lambda crop: (_ for _ in ()).throw(RuntimeError("boom"))
        r = self.post(*self.frame())
        self.assertEqual(r.status_code, 500); self.assertIn("RuntimeError: boom", r.get_json()["error"])

    # ---- the other endpoints
    def test_pokemon_endpoint(self):
        d = self.client.get("/api/pokemon?q=pikachu").get_json()
        self.assertEqual(d["pokemon"]["id"], 25)
        self.assertEqual(self.client.get("/api/pokemon?q=").status_code, 400)
        self.assertEqual(self.client.get("/api/pokemon?q=" + "x" * 80).status_code, 400)
        self.assertIn("isn't a Pokémon", self.client.get("/api/pokemon?q=nope").get_json()["message"])

    def test_recognize_endpoint_still_works(self):
        ok, jpg = cv2.imencode(".jpg", self.pika)
        r = self.client.post("/api/recognize", data={"image": (io.BytesIO(jpg.tobytes()), "c.jpg")}, content_type="multipart/form-data")
        self.assertEqual(r.get_json()["card"]["name"], "Pikachu")
        self.assertEqual(self.client.post("/api/recognize", data={}, content_type="multipart/form-data").status_code, 400)

    def test_health_and_models(self):
        h = self.client.get("/api/health").get_json()
        self.assertTrue(h["ok"]); self.assertEqual(h["models"], 1); self.assertIn("opencv", h["vendor"])
        (self.tmp / "models" / "x.glb").write_bytes(b"glTF")
        r = self.client.get("/models/x.glb")
        self.assertEqual((r.status_code, r.mimetype), (200, "model/gltf-binary"))
        self.assertEqual(self.client.get("/models/missing.glb").status_code, 404)

    def test_vendor_files_are_cached_for_a_year(self):
        r = self.client.get("/static/vendor/README.txt")
        self.addCleanup(r.close)
        self.assertIn("immutable", r.headers["Cache-Control"])

    # ---- pages
    def test_pages_get_the_import_map_and_config(self):
        for url in ("/detect", "/battle"):
            with self.subTest(url):
                r = self.client.get(url)
                self.assertEqual(r.status_code, 200, url)
                html = r.get_data(as_text=True)
                self.assertNotIn("<!--PC3D_HEAD-->", html)
                self.assertIn('<script type="importmap">', html)
                m = __import__("re").search(r"window\.PC3D=(\{.*?\})</script>", html, __import__("re").S)
                cfg = json.loads(m.group(1).replace("<\\/", "</"))
                self.assertTrue(cfg["libs"]["opencv"][-1].startswith("https://docs.opencv.org/"))
                self.assertTrue(any("AR-js-org/AR.js" in u for u in cfg["libs"]["arjs"]))
                self.assertEqual(r.headers["Cache-Control"], "no-cache")

    def test_battle_page_loads_the_scripts_it_needs(self):
        html = self.client.get("/detect").get_data(as_text=True)
        for needle in ("/static/js/detect.js", "viewport"):
            self.assertIn(needle, html)
        for name in ("detect.js", "card-geometry.js", "card-tracker.js", "tracker.worker.js"):
            self.assertEqual(self.client.get(f"/static/js/{name}").status_code, 200, name)


class Tls(unittest.TestCase):
    def test_certificate_is_valid_reused_and_renewed(self):
        from cryptography import x509
        with tempfile.TemporaryDirectory() as d:
            cert, key = tls.ensure_cert(d)
            ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER); ctx.load_cert_chain(cert, key)          # a real server could use it
            names = tls._cert_names(x509.load_pem_x509_certificate(Path(cert).read_bytes()))
            self.assertTrue({"localhost", "127.0.0.1"} <= names)
            before = Path(cert).read_bytes()
            tls.ensure_cert(d); self.assertEqual(Path(cert).read_bytes(), before, "was regenerated for no reason")
            real = tls.lan_ips
            tls.lan_ips = lambda: ["192.168.77.5"]                                                  # the phone's network changed
            try:
                tls.ensure_cert(d)
            finally:
                tls.lan_ips = real
            self.assertNotEqual(Path(cert).read_bytes(), before)
            self.assertIn("192.168.77.5", tls._cert_names(x509.load_pem_x509_certificate(Path(cert).read_bytes())))

    def test_lan_ips_are_not_loopback(self):
        self.assertTrue(all(not ip.startswith("127.") for ip in tls.lan_ips()))


class Vendor(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        old = vendor.VENDOR_DIR; vendor.VENDOR_DIR = self.tmp; self.addCleanup(setattr, vendor, "VENDOR_DIR", old)

    def put(self, rel):
        p = self.tmp / rel; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b"x")

    def test_cdn_when_nothing_is_self_hosted(self):
        c = vendor.three_config()
        self.assertEqual(c["source"], "cdn"); self.assertTrue(c["importmap"]["imports"]["three"].startswith("https://cdn.jsdelivr.net/"))
        self.assertTrue(all(u.startswith("https://") for u in vendor.script_urls("opencv")))

    def test_local_three_only_when_the_whole_set_is_there(self):
        files = [rel for rel, _ in vendor.MANIFEST["three"]]
        for rel in files[:-1]:
            self.put(rel)
        self.assertEqual(vendor.three_config()["source"], "cdn", "one file missing must not break the page")
        self.put(files[-1])
        c = vendor.three_config()
        self.assertEqual(c["source"], "local")
        self.assertEqual(c["importmap"]["imports"]["three/addons/"], "/static/vendor/three/addons/")
        self.assertEqual(c["draco"], "/static/vendor/three/draco/")

    def test_local_script_is_tried_first_then_the_cdn(self):
        self.put("opencv/opencv.js")
        urls = vendor.script_urls("opencv")
        self.assertEqual(urls[0], "/static/vendor/opencv/opencv.js"); self.assertGreater(len(urls), 1)
        self.assertEqual(vendor.status()["opencv"], "local")


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        body = {"/good.js": b"a" * 5000, "/tiny.js": b"404 page", "/bad.wasm": b"nope" * 10, "/ok.wasm": b"\0asm\1\0\0\0" + bytes(50)}.get(self.path)
        self.send_response(200 if body else 404); self.end_headers(); self.wfile.write(body or b"")

    def log_message(self, *a): pass


class FetchVendor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.srv = HTTPServer(("127.0.0.1", 0), Handler)
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.base = f"http://127.0.0.1:{cls.srv.server_port}"
        sys.path.insert(0, str(ROOT / "scripts"))
        import fetch_vendor
        cls.fv = fetch_vendor

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown(); cls.srv.server_close()

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()); self.addCleanup(shutil.rmtree, self.tmp, True)
        old = vendor.VENDOR_DIR; vendor.VENDOR_DIR = self.tmp; self.addCleanup(setattr, vendor, "VENDOR_DIR", old)
        old = dict(vendor.MIN_BYTES); vendor.MIN_BYTES["a/good.js"] = 1000; self.addCleanup(vendor.MIN_BYTES.update, old)

    def test_falls_back_to_the_next_url_and_writes_atomically(self):
        state, detail = self.fv.fetch("a/good.js", [self.base + "/missing.js", self.base + "/good.js"], False)
        self.assertEqual(state, "got"); self.assertEqual((self.tmp / "a/good.js").stat().st_size, 5000)
        self.assertEqual(list(self.tmp.rglob("*.part")), [])

    def test_error_pages_are_not_saved_as_libraries(self):
        state, detail = self.fv.fetch("a/good.js", [self.base + "/tiny.js"], False)
        self.assertEqual(state, "FAILED"); self.assertFalse((self.tmp / "a/good.js").exists()); self.assertIn("only 8 bytes", detail)

    def test_wasm_must_be_wasm(self):
        self.assertEqual(self.fv.fetch("d/x.wasm", [self.base + "/bad.wasm"], False)[0], "FAILED")
        self.assertEqual(self.fv.fetch("d/x.wasm", [self.base + "/ok.wasm"], False)[0], "got")

    def test_existing_files_are_kept_unless_forced(self):
        self.fv.fetch("a/good.js", [self.base + "/good.js"], False)
        self.assertEqual(self.fv.fetch("a/good.js", [self.base + "/good.js"], False)[0], "have")
        self.assertEqual(self.fv.fetch("a/good.js", [self.base + "/good.js"], True)[0], "got")


if __name__ == "__main__":
    unittest.main()
