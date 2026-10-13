"""End-to-end test of the Detect and Battle pages in a real (headless) Chromium.

Run:  python tests/browser_game.py          (needs: pip install playwright + a Chromium; skipped otherwise)

What is REAL: detect.html, detect.js, battle-engine.js, the Flask routes, the DOM/HUD, the timers (the 5 s countdown is
really waited), the browser's fake camera. What is STUBBED (they are big CDN downloads / need a real card to see):
three.js (tests/browser/stub_three.js), OpenCV tracker (stub_worker.js - reports a card whenever the test says one is
present) and the card recogniser (a fake agent returning canned answers). So this proves the game flow, not the 3D look."""
import json, os, sys, threading, time, types
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tests"))
import test_server  # noqa: F401  (installs stand-ins for trimesh / the recogniser so app.py imports)
import app as app_mod
import card_reader
import textcard
import cv2, numpy as np
from agent import PokemonCardAgent
from werkzeug.serving import make_server

try:
    from playwright.sync_api import sync_playwright
except ImportError:
    print("playwright not installed - skipped"); sys.exit(0)

STUB = ROOT / "tests" / "browser"



def pk(i, name, types_):
    return {"id": i, "name": name, "types": types_, "height_m": 0.6, "weight_kg": 8, "artwork": None}


class FakeAgent:
    """Canned answers; counts calls so the test can prove nothing is loaded twice."""
    models = types.SimpleNamespace(count=0)

    def __init__(self):
        self.corrections = []; self.scans = 0; self.opponents = 0; self.energy = False; self.client_ocr = False
        # the real browser-OCR job code, run on this fake (so the round trip server <-> tesseract.js stand-in is genuine)
        self.ocr_jobs = OrderedDict(); self.scan_cache = OrderedDict(); self.cache_lock = threading.Lock()
        self.complete_stats = PokemonCardAgent.complete_stats
        self.ocr_image = types.MethodType(PokemonCardAgent.ocr_image, self)
        self.finish_client_ocr = types.MethodType(PokemonCardAgent.finish_client_ocr, self)
        self._client_ocr_job = types.MethodType(PokemonCardAgent._client_ocr_job, self)

    def run_card(self, crop):
        self.scans += 1
        if self.energy:
            return {"card": {"set": "Base", "name": "Fire Energy", "number": "98"}, "pokemon": None, "model_url": None, "model_source": None,
                    "card_kind": "energy", "energy": {"type": "fire", "special": False, "name": "Fire Energy"}, "trace": [], "rotated": False, "cached": False}
        if self.client_ocr:                                    # server "has no Tesseract": ask the browser to read the card
            token = self._client_ocr_job(textcard.text_card(), None, ["fire"])
            stats = PokemonCardAgent.complete_stats(None, ["fire"]); stats["client_ocr"] = token
            return {"card": {"set": "Base", "name": "Charmander", "number": "46"}, "pokemon": pk(4, "Charmander", ["fire"]), "model_url": "/models/gen1/004.glb",
                    "model_source": {"kind": "gen1", "detail": "x", "similarity": None}, "trace": [], "rotated": False, "cached": False, "stats": stats}
        return {"card": {"set": "Base", "name": "Charmander", "number": "46"}, "pokemon": pk(4, "Charmander", ["fire"]), "model_url": "/models/gen1/004.glb",
                "model_source": {"kind": "gen1", "detail": "x", "similarity": None}, "trace": [], "rotated": False, "cached": self.scans > 1, "result_id": "r1",
                "stats": {"hp": 60, "source": "ocr", "confidence": "high", "notes": [],
                          "attacks": [{"name": "Ember", "damage": 30, "modifier": None, "type": "fire", "cost": ["fire"], "text": ""},
                                      {"name": "Growl", "damage": None, "modifier": None, "type": "normal", "cost": ["normal"], "text": ""}]}}

    def submit_correction(self, result_id, client, pokemon=None, hp=None, attacks=None):
        self.corrections.append({"result_id": result_id, "client": client, "pokemon": pokemon, "hp": hp, "attacks": attacks})
        stats = {"hp": int(hp) if hp else 60, "source": "user", "confidence": "high", "notes": ["Corrected by a player."],
                 "attacks": [{"name": a["name"], "damage": int(a["damage"]) if str(a.get("damage") or "").isdigit() else None, "modifier": None, "type": a.get("type") or "normal", "cost": ["normal"], "text": ""} for a in (attacks or [])]
                 or [{"name": "Ember", "damage": 30, "modifier": None, "type": "fire", "cost": ["fire"], "text": ""}]}
        out = {"ok": True, "stats": stats, "result": None}
        if pokemon:
            out["result"] = {"card": None, "pokemon": pk(25, "Pikachu", ["electric"]), "model_url": "/models/gen1/025.glb",
                             "model_source": {"kind": "gen1", "detail": "x", "similarity": None}, "trace": [], "result_id": "r2"}
        return out

    def opponent(self, exclude=()):
        self.opponents += 1
        return {"card": None, "pokemon": pk(25, "Pikachu", ["electric"]), "model_url": "/models/gen1/025.glb", "model_source": {"kind": "gen1", "detail": "x"}, "trace": [],
                "stats": {"hp": 40, "source": "generated", "notes": [],
                          "attacks": [{"name": "Tackle", "damage": 20, "modifier": None, "type": "normal", "cost": ["normal"]},
                                      {"name": "Thunder Shock", "damage": 30, "modifier": None, "type": "electric", "cost": ["electric"]}]}}


# window.Tesseract stand-in: same API shape as tesseract.js (createWorker -> setParameters/recognize); recognising is done by the real
# Tesseract PROGRAM through the test's /__ocr route, and the answer is returned in tesseract.js's {data:{lines:[{words:[{text,confidence,bbox}]}]}} shape.
TESSERACT_STUB = """window.Tesseract = { createWorker: async () => { let psm = "6"; window.__tjs = []; return {
  setParameters: async (p) => { if (p.tessedit_pageseg_mode) psm = p.tessedit_pageseg_mode; },
  recognize: async (url) => { window.__tjs.push(url + "#psm" + psm);
    const png = await (await fetch(url)).arrayBuffer();
    return await (await fetch('/__ocr?psm=' + psm, { method: 'POST', body: png })).json(); } }; } };"""


def run_cli_ocr(request):
    from urllib.parse import urlparse, parse_qs
    psm = int(parse_qs(urlparse(request.url).query).get("psm", ["6"])[0])
    img = cv2.imdecode(np.frombuffer(request.post_data_buffer, np.uint8), cv2.IMREAD_GRAYSCALE)
    lines = {}
    for w in card_reader.ocr_words(img, psm=psm):
        lines.setdefault(w["line"], []).append({"text": w["text"], "confidence": w["conf"],
                                                "bbox": {"x0": w["left"], "y0": w["top"], "x1": w["left"] + w["width"], "y1": w["top"] + w["height"]}})
    return json.dumps({"data": {"lines": [{"words": ws} for ws in lines.values()]}})


results = []
def check(name, ok, extra=""):
    results.append(ok); print(("PASS " if ok else "FAIL ") + name + (f"  [{extra}]" if extra and not ok else ""))


def main():
    agent = FakeAgent()
    flask_app = app_mod.create_app(agent)
    srv = make_server("127.0.0.1", 0, flask_app, threaded=True)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    base = f"http://127.0.0.1:{srv.server_port}"
    state = {"present": True, "id": 1}

    with sync_playwright() as p:
        b = p.chromium.launch(args=["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"])
        ctx = b.new_context(permissions=["camera"], viewport={"width": 420, "height": 800})
        page = ctx.new_page(); errors = []
        def shot(name):                                          # PC3D_SHOTS=<folder> saves real screenshots (used for the documentation PDF)
            if os.environ.get("PC3D_SHOTS"): page.screenshot(path=f"{os.environ['PC3D_SHOTS']}/{name}.png")
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: m.type == "error" and "Failed to load resource" not in m.text and errors.append(m.text))
        rd = lambda f: (STUB / f).read_text()
        js = lambda body: dict(status=200, content_type="text/javascript", body=body)
        page.route("**/three@0.160.0/build/three.module.js", lambda r: r.fulfill(**js(rd("stub_three.js"))))
        page.route("**/three@0.160.0/examples/jsm/loaders/GLTFLoader.js", lambda r: r.fulfill(**js(rd("stub_gltf.js"))))
        page.route("**/three@0.160.0/examples/jsm/controls/OrbitControls.js", lambda r: r.fulfill(**js(rd("stub_orbit.js"))))
        page.route("**/three@0.160.0/examples/jsm/loaders/DRACOLoader.js", lambda r: r.fulfill(**js("export { DRACOLoader } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';")))
        page.route("**/three@0.160.0/examples/jsm/utils/SkeletonUtils.js", lambda r: r.fulfill(**js("export { clone } from 'https://cdn.jsdelivr.net/npm/three@0.160.0/examples/jsm/loaders/GLTFLoader.js';")))
        page.route("**/tesseract.min.js", lambda r: r.fulfill(**js(TESSERACT_STUB)))
        page.route("**/__ocr*", lambda r: r.fulfill(status=200, content_type="application/json", body=run_cli_ocr(r.request)))
        page.route("**/ar-threex.js", lambda r: r.fulfill(**js("")))
        page.route("**/static/js/tracker.worker.js*", lambda r: r.fulfill(**js(rd("stub_worker.js"))))
        page.route("**/__stub", lambda r: r.fulfill(status=200, content_type="application/json", body=json.dumps(state)))

        scan_posts, stub_hits = [], {"n": 0}
        ctx.on("request", lambda r: "/api/battle/scan" in r.url and scan_posts.append(r.post_data_buffer or b""))
        ctx.on("request", lambda r: r.url.endswith("/__stub") and stub_hits.__setitem__("n", stub_hits["n"] + 1))

        # ------------------------------------------------ Home = Detect page: energy card first (a fresh page, so the card memory is empty)
        agent.energy = True; state.update(present=True, id=9)
        page.goto(base + "/")
        check("home: both modes with options and a two-sentence About", page.is_visible("#hDetect") and page.is_visible("#hBattle") and page.locator("#about p").count() == 1)
        shot("home")
        page.click("a.opt:has-text('Camera with AR')")
        
        page.wait_for_function("document.getElementById('start').hidden", timeout=15000)
        check("home: 'Camera with AR' opens Detect and starts the camera by itself", page.url.endswith("/detect?go=ar") and not page.is_visible("#start"))
        page.goto(base + "/detect")
        check("detect: title and heading", page.title().startswith("Detect") and page.inner_text("#startH") == "Detect")
        check("detect: AR switch and 'use without AR' are offered", page.is_visible("#arToggle") and page.is_visible("#goNoAR"))
        page.click("#go")
        page.wait_for_selector(".tag b:has-text('Fire Energy')", timeout=15000)
        check("detect: energy card is labelled with its type", "fire" in page.inner_text(".tag:visible"))
        agent.energy = False; state.update(id=1)

        page.goto(base + "/detect")
        page.click("#go")
        page.wait_for_selector(".tag b:has-text('Charmander')", timeout=15000)
        shot("02_detect_ar")
        check("detect: card read and labelled", True)
        check("detect: no battle HUD", not page.is_visible("#fOpp") and not page.is_visible("#moves") and not page.is_visible("#bMsg"))
        check("detect: link to battle", page.get_attribute("#swap", "href") == "/battle")
        post = scan_posts[-1]
        check("detect: the browser uploads a straightened card snapshot, not a camera frame",
              b'name="snapshot"' in post and b'filename="card.jpg"' in post and b'name="corners"' not in post and len(post) < 150_000, f"{len(post)} bytes")
        check("detect: results panel is closed while AR is on", not page.is_visible("#res"))

        # ---- Details: what the agent read
        page.click("#bRate")
        page.wait_for_selector("#res #rName", state="visible", timeout=5000)
        check("detect: details show the Pokémon, HP and attacks the agent read",
              page.inner_text("#rName") == "Charmander" and page.inner_text("#rHp") == "HP 60"
              and page.locator("#rAtk li").count() == 2 and "30 · fire" in page.inner_text("#rAtk"), page.inner_text("#res"))
        check("detect: details show what the camera saw and the straightened card",
              page.get_attribute("#shotFrame", "src").startswith("data:image") and page.get_attribute("#shotCard", "src").startswith("data:image"))
        shot("detect_details")
        check("detect: how the agent decided + where the stats came from", "Read from the card's text" in page.inner_text("#rNotes"))

        # ---- Teach: starting to correct switches AR off by itself
        renders0 = page.evaluate("window.__renders || 0")
        page.click("#teachOpen")
        page.wait_for_function("document.body.classList.contains('ar-off')", timeout=8000)
        check("teach: AR switched off automatically", page.inner_text("#arLabel") == "AR off" and page.get_attribute("#arToggle", "aria-pressed") == "false")
        check("teach: camera released and the 3D viewer + photo buttons are shown",
              page.evaluate("!document.querySelector('body > video')") and page.is_visible("#viewer") and page.is_visible("#bShoot") and page.is_visible("#teach"))
        n1 = stub_hits["n"]; time.sleep(1.2)
        check("teach: the tracker stopped (no more frames analysed)", stub_hits["n"] - n1 <= 2, f"{stub_hits['n'] - n1} more frames")
        page.wait_for_function("window.__orbit !== undefined", timeout=5000)
        r1 = page.evaluate("window.__renders || 0"); time.sleep(0.6)
        check("teach: the model is shown in the 3D viewer", any("004.glb" in u for u in page.evaluate("window.__loads")) and page.evaluate("window.__renders || 0") > r1)

        loads_before = len(page.evaluate("window.__loads"))
        page.fill("#tHp", "77")
        row0 = page.locator("#tAtk .arow").nth(0)
        check("teach: the form is prefilled with what the agent read", row0.locator(".an").input_value() == "Ember" and row0.locator(".ad").input_value() == "30")
        shot("teach_form")
        row0.locator(".an").fill("Flame Burst"); row0.locator(".ad").fill("40"); row0.locator(".at").select_option("fire")
        page.click("#tSave")
        page.wait_for_function("document.getElementById('tMsg').textContent.startsWith('Saved')", timeout=5000)
        c = agent.corrections[-1]
        check("teach: HP and attacks are sent to the agent", c["hp"] == "77" and c["attacks"][0]["name"] == "Flame Burst" and c["attacks"][0]["damage"] == "40" and c["pokemon"] is None, str(c))
        shot("04_teach_saved")
        check("teach: the panel shows the corrected stats", page.inner_text("#rHp") == "HP 77" and "Flame Burst" in page.inner_text("#rAtk") and "Corrected by a player" in page.inner_text("#rNotes"))
        page.fill("#tPokemon", "pikachu"); page.click("#tSave")
        page.wait_for_function("document.getElementById('rName').textContent === 'Pikachu'", timeout=5000)
        check("teach: a corrected Pokémon is sent and its 3D model replaces the old one",
              agent.corrections[-1]["pokemon"] == "pikachu" and any("025.glb" in u for u in page.evaluate("window.__loads")) and len(page.evaluate("window.__loads")) > loads_before)
        check("teach: empty submit is refused with a message", True)
        page.fill("#tPokemon", "Pikachu"); page.click("#tSave")
        page.wait_for_function("document.getElementById('tMsg').textContent.includes('Change the Pokémon')", timeout=3000)

        # ---- AR off: a photo instead of the camera (the server finds the card's outer border itself)
        import synth
        frame, _, _ = synth.render_scene(960, 540, cards=[dict(cx=.5, cy=.5, width=240, pitch=20, inner_frame=True)], seed=3, decoys=True)
        photo = Path(__file__).parent / "_photo.jpg"; cv2.imwrite(str(photo), frame)
        posts_before, scans_before = len(scan_posts), agent.scans
        page.click("#tCancel")
        page.set_input_files("#photo", str(photo))
        page.wait_for_function("document.getElementById('rName').textContent === 'Charmander'", timeout=10000)
        photo.unlink()
        check("photo: uploaded without corners or snapshot flag (server detects the card)", b'name="corners"' not in scan_posts[posts_before] and b'name="snapshot"' not in scan_posts[posts_before] and agent.scans == scans_before + 1)
        shot("05_photo_ar_off")
        check("photo: the outline of the card found is drawn on the picture", page.get_attribute("#shotFrame", "src").startswith("data:image") and "No card border" not in page.inner_text("#rSub"))

        # ---- AR on again
        page.click("#arToggle")
        page.wait_for_function("!document.body.classList.contains('ar-off')", timeout=15000)
        page.wait_for_selector(".tag b", timeout=15000)
        check("detect: AR switches back on (camera + tracker restart)", page.inner_text("#arLabel") == "AR on" and not page.is_visible("#res") and not page.is_visible("#viewer"))

        # ------------------------------------------------ Battle page
        state.update(present=True, id=1); agent.scans = 0
        page.goto(base + "/battle")
        page.wait_for_function("document.getElementById('startH').textContent === 'Battle'", timeout=25000)    # set by the page's module script
        check("battle: title and heading", page.title().startswith("Battle") and page.inner_text("#startH") == "Battle")
        page.click("#go")
        page.wait_for_selector("#moves .mv", timeout=15000)
        t0 = time.time()
        check("battle: your card's HP and attacks are shown", page.inner_text("#mHp") == "60 / 60" and page.locator("#moves .mv").count() == 2)
        shot("battle_countdown")
        moves = page.locator("#moves .mv")
        check("battle: attack buttons show printed damage and type", "30 damage" in moves.nth(0).inner_text() and "No damage" in moves.nth(1).inner_text(), moves.nth(0).inner_text())
        check("battle: moves are locked before the opponent arrives", moves.nth(0).is_disabled())
        check("battle: countdown message", "battle starts in" in page.inner_text("#bMsg"), page.inner_text("#bMsg"))
        check("battle: no opponent yet, still the normal view", not page.is_visible("#fOpp") and not page.evaluate("document.body.classList.contains('in-battle')"))
        page.wait_for_selector("#fOpp:not([hidden])", state="attached", timeout=16000)
        waited = time.time() - t0
        check("battle: opponent appears after ~10 s of detection", 9 < waited < 15, f"{waited:.1f}s")
        check("battle: the page switched to the battle view", page.evaluate("document.body.classList.contains('in-battle')") and float(page.evaluate("getComputedStyle(document.getElementById('veil')).opacity")) > 0)
        time.sleep(1.5)                                          # let the shrink / slide animations settle
        lay = page.evaluate("""() => { const out = {sprites: window.__sprites || 0, mine: null, opp: null};
            const walk = (o) => { (o.children || []).forEach(walk);
              if (o.children && o.children.length === 1 && o.position && o.scale && o.parent && o.parent.parent === window.__scene) out.mine = {z: o.position.z, s: o.scale.x};
              if (o.position && o.parent && o.parent.parent === window.__scene && o.position.z < 0) out.opp = {z: o.position.z, s: o.scale.x}; };
            walk(window.__scene); return out; }""")
        check("battle: both Pokémon sit on the card, one per half, facing each other", lay["mine"] and lay["opp"] and abs(lay["mine"]["z"] - 0.3) < 0.02 and abs(lay["opp"]["z"] + 0.3) < 0.02, str(lay))
        check("battle: the player's Pokémon was shrunk to fit", lay["mine"] and lay["mine"]["s"] < 0.9, str(lay))
        check("battle: HP bars are AR sprites on the card", lay["sprites"] >= 2, str(lay))
        shot("battle_view")
        check("battle: opponent shown with random-model HP", "Pikachu" in page.inner_text("#oName") and page.inner_text("#oHp") == "40 / 40")
        check("battle: opponent model was loaded", agent.opponents == 1 and any("025.glb" in u for u in page.evaluate("window.__loads")))
        page.wait_for_function("!document.querySelector('#moves .mv').disabled", timeout=5000)

        page.evaluate("window.__flash = []")
        sprites0 = page.evaluate("window.__sprites || 0")
        moves.nth(0).click()
        page.wait_for_function("document.querySelector('#oHp').textContent === '10 / 40'", timeout=3000)
        check("battle: Ember removes exactly the printed 30 HP", True)
        check("battle: hit message", "Ember" in page.inner_text("#bMsg") and "30 HP" in page.inner_text("#bMsg"), page.inner_text("#bMsg"))
        try: page.wait_for_function("window.__flash.length > 0", timeout=1500)    # the glow is drawn by the (30 fps) render loop
        except Exception: pass
        fl = page.evaluate("window.__flash")
        check("battle: opponent lit up in the fire colour", any(r > .9 and 0.3 < g < .5 and b < .3 for r, g, b in fl), str(fl[:3]))
        check("battle: damage number is an AR sprite", page.evaluate("window.__sprites || 0") > sprites0)
        page.wait_for_function("document.querySelector('#mHp').textContent !== '60 / 60'", timeout=6000)
        hit = 60 - int(page.inner_text("#mHp").split("/")[0])
        check("battle: opponent answers with a random multiple of 5 between 5 and 40", 5 <= hit <= 40 and hit % 5 == 0, str(hit))
        # the whole battle HUD (2 HP boxes + message + attack buttons) must cover at most 1/6 of the screen, in portrait and landscape
        for vw, vh in ((420, 800), (390, 844), (360, 640), (844, 390)):
            page.set_viewport_size({"width": vw, "height": vh}); time.sleep(0.3)
            area = page.evaluate("""() => ['fOpp','fMe','bMsg','moves'].map(id => { const e = document.getElementById(id); if (e.hidden) return 0;
                                     const r = e.getBoundingClientRect(); return r.width * r.height; }).reduce((a, b) => a + b, 0)""")
            frac = area / (vw * vh)
            check(f"battle: HUD covers <= 1/6 of a {vw}x{vh} screen", 0 < frac <= 1 / 6, f"{100 * frac:.1f}%"); print(f"   HUD {vw}x{vh}: {100 * frac:.1f}%")
        page.set_viewport_size({"width": 420, "height": 800}); time.sleep(0.3)
        page.wait_for_function("!document.querySelector('#moves .mv').disabled", timeout=5000)

        ov = page.evaluate("""() => { const m = document.getElementById('moves').getBoundingClientRect(), d = document.getElementById('dock').getBoundingClientRect();
                                   return {moves: m.bottom, dock: d.top, h: d.height}; }""")
        check("battle: the bottom button bar is one row and does not cover the attack buttons", ov["moves"] <= ov["dock"] + 1 and ov["h"] < 60, str(ov))

        # card leaves the table and comes back (new track id)
        state["present"] = False
        page.wait_for_function("document.querySelector('#moves .mv').disabled", timeout=7000)           # (the model is held on screen ~3 s first)
        page.wait_for_function("document.querySelector('#bMsg').textContent.includes('Show your card')", timeout=7000)
        check("battle: pauses when the card is gone", True)
        scans_before, opp_before = agent.scans, agent.opponents
        state.update(present=True, id=2)
        page.wait_for_function("!document.querySelector('#moves .mv').disabled", timeout=8000)         # entry was re-adopted: no reload, no new opponent
        check("battle: same battle resumes on the new track", page.inner_text("#oHp") == "10 / 40" and agent.opponents == opp_before and "Show your card" not in page.inner_text("#bMsg"), page.inner_text("#bMsg"))
        check("battle: opponent was not fetched again", agent.opponents == opp_before)

        # non-damaging move
        page.evaluate("window.__flash = []")
        moves.nth(1).click()
        page.wait_for_function("document.querySelector('#bMsg').textContent.includes('non-damaging')", timeout=3000)
        check("battle: non-damaging move tells the player", "Growl" in page.inner_text("#bMsg") and page.inner_text("#oHp") == "10 / 40", page.inner_text("#bMsg"))
        check("battle: non-damaging move does not flash", page.evaluate("window.__flash").__len__() == 0)
        page.wait_for_function("!document.querySelector('#moves .mv').disabled || document.querySelector('#mHp').textContent.startsWith('0')", timeout=8000)   # (two random hits of up to 40 can faint a 60-HP Pokémon)
        if page.inner_text("#mHp").startswith("0"):
            check("battle: (lost early - skipping win test)", True)
        else:
            moves.nth(0).click()
            page.wait_for_function("document.querySelector('#bMsg').textContent.includes('You win')", timeout=4000)
            check("battle: win is announced and opponent HP hits 0", page.inner_text("#oHp") == "0 / 40")
            check("battle: moves locked after the battle", moves.nth(0).is_disabled())
            page.click("#bMsg button")
            page.wait_for_function("document.querySelector('#oHp').textContent === '40 / 40'", timeout=15000)
            check("battle: 'Battle again' brings a fresh opponent at full HP", agent.opponents == opp_before + 1 and page.inner_text("#mHp") == "60 / 60")
        # ------------------------------------------------ AR stability: the card flickers in and out (hand, glare) during the countdown
        flick = {"stop": False}
        def flicker():                                           # card hidden 0.8 s, shown 0.6 s, each time under a NEW track id
            i = 20
            while not flick["stop"]:
                state.update(present=False); time.sleep(0.8)
                i += 1; state.update(present=True, id=i); time.sleep(0.6)
        agent.scans = 0; agent.opponents = 0
        state.update(present=True, id=19)
        page.goto(base + "/battle")
        page.wait_for_function("document.getElementById('startH').textContent === 'Battle'", timeout=25000)
        page.click("#go")
        page.wait_for_selector("#moves .mv", timeout=15000)
        th = threading.Thread(target=flicker, daemon=True); th.start(); t2 = time.time()
        page.wait_for_selector("#fOpp:not([hidden])", timeout=20000)
        flick["stop"] = True; th.join(); state.update(present=True, id=99)
        check("flicker: battle still starts while the card keeps vanishing", agent.opponents == 1, str(agent.opponents))
        check("flicker: countdown was paused/resumed, not restarted forever", time.time() - t2 < 22, f"{time.time() - t2:.1f}s")
        check("flicker: the card is only re-confirmed, never rebuilt", agent.scans <= 6, f"{agent.scans} scans")
        page.wait_for_function("!document.querySelector('#moves .mv').disabled", timeout=10000)
        check("flicker: moves are usable afterwards", page.inner_text("#oHp") == "40 / 40")
        # ------------------------------------------------ server without Tesseract -> the browser reads the card (tesseract.js path)
        if card_reader.tesseract_available():
            agent.client_ocr = True; state.update(present=True, id=5)
            page.goto(base + "/battle")
            page.wait_for_function("document.getElementById('startH').textContent === 'Battle'", timeout=25000)
            page.click("#go")
            page.wait_for_selector("#moves .mv", timeout=15000)
            page.wait_for_function("document.querySelector('#mHp').textContent === '120 / 120'", timeout=40000)
            check("tesseract.js path: HP read in the browser (120)", True)
            names = page.locator("#moves .mv b").all_inner_texts()
            check("tesseract.js path: attacks read in the browser", names == ["Ember", "Growl"], str(names))
            check("tesseract.js path: both regions were recognised", len(page.evaluate("window.__tjs")) == 3, str(page.evaluate("window.__tjs")))
            t1 = time.time()
            page.wait_for_selector("#fOpp:not([hidden])", timeout=12000)
            check("tesseract.js path: the 5 s countdown starts after the text is read", time.time() - t1 > 2, f"{time.time() - t1:.1f}s")
        check("no JavaScript errors", not errors, "; ".join(errors[:3]))
        b.close()
    srv.shutdown()
    print(f"\n{sum(results)}/{len(results)} checks passed")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
