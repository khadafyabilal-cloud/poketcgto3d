"""
app.py - the web server.

  GET  /                  HOME = Detect: camera + card tracking + the 3D model on the card (AR on), or - with AR switched off - a 3D
                          viewer + detection results + the "Teach the agent" form (photo upload / take photo)
  GET  /                  home page: pick Detect or Battle, with options (static/home.html)
  GET  /detect, /battle   the camera pages (static/detect.html, ?go=ar|viewer starts at once);  GET /scan = single photo page
  GET  /battle            the mock Pokémon battle: same camera page + opponent, health bars and attack buttons
  GET  /api/battle/ocr/<token>/<hp|attack>.png + POST /api/battle/ocr-words   browser-side OCR (tesseract.js) round trip
  GET  /api/battle/opponent a random opponent (3D model + made-up HP and moves)
  POST /api/recognize     multipart 'image'                         -> agent.run           (whole photo; kept for scripts)
  POST /api/battle/scan   multipart 'image' + ONE of three modes    -> agent.run_card      (detect + battle pages)
                          snapshot=1            the browser already cut the card out and straightened it (cheapest: a
                                                small upright picture, no server-side detection) - what AR mode sends
                          corners=[[x,y] x4]    JSON, TL TR BR BL in pixels of the image, found by the browser tracker;
                                                OpenCV (card_detect.py) re-measures and straightens the card
                          (neither)             a plain photo: the server looks for the card border itself (AR off)
  POST /api/feedback/correct  JSON {result_id, client, pokemon?, hp?, attacks?[{name,damage,type}]} -> agent.submit_correction
                          the Detect page's "Teach the agent" form: remembered, used instead of OCR next time
  GET  /api/pokemon?q=    pick a Pokémon by name or Pokédex number -> agent.lookup
  GET  /api/health        what is loaded / self-hosted
  GET  /models/<file>     .glb 3D models (slimmed pack models live in models/glb/)

HOW A REQUEST FLOWS (the mental model)
  browser ──HTTP──► Flask route ──► (card_detect.py and/or agent.py) ──► JSON ──► browser draws the 3D model
  Flask is a tiny Python web framework: `@app.get("/path")` registers a function that runs when the browser asks for /path.
  Pages (/, /detect and /battle) are plain HTML files in static/; the JavaScript in static/js does the camera + 3D work in the browser.
  Only the heavy "which card is this?" question comes back to Python.

Run:  python app.py                                    this computer only (http://127.0.0.1:5000)
      python app.py --host 0.0.0.0 --https             phones on your Wi-Fi (camera needs HTTPS - see tls.py)
"""
import argparse, base64, json, os, sys
from pathlib import Path

import cv2
import numpy as np
from flask import Flask, Response, jsonify, redirect, request, send_from_directory

import card_detect
import vendor

BASE = Path(__file__).parent
STATIC = BASE / "static"
MODELS = BASE / "models"; MODELS.mkdir(exist_ok=True)       # same folder agent.py writes its .glb files to
CARD_PAD = 0.06                                             # keep equal to agent.CARD_PAD (checked by the tests)
MAX_IMAGE_PX = 4096


def build_id():
    """Cache-buster. Changes whenever a script in static/js changes -> used as ?v= so browsers never run stale code."""
    files = list((STATIC / "js").glob("*.js"))
    return format(int(max((f.stat().st_mtime for f in files), default=0)), "x")


def head_block():
    """<script> tags injected into the pages: the import map (self-hosted three.js if present, else CDN) and window.PC3D.

    window.PC3D is a small settings object the page JavaScript reads: {v: cache-buster, draco: decoder path,
    libs: {opencv: [urls], arjs: [urls]}}. json.dumps(...).replace("</", "<\\/") stops a URL containing "</script>"
    from ending the tag early."""
    three = vendor.three_config()
    cfg = {"v": build_id(), "draco": three["draco"],
           "libs": {"opencv": vendor.script_urls("opencv"), "arjs": vendor.script_urls("arjs"), "tesseract": vendor.TESSERACTJS_URLS}}
    safe = lambda o: json.dumps(o).replace("</", "<\\/")
    return f'<script type="importmap">{safe(three["importmap"])}</script>\n<script>window.PC3D={safe(cfg)}</script>'


def parse_corners(raw, w, h):
    """Validate the browser's card corners -> (4,2) float32. Raises ValueError with a message fit for the user.

    NEVER trust data from a browser: it could be malformed or malicious. Checks: it parses as 4 pairs of finite
    numbers, the points lie (almost) inside the picture, they form a convex four-sided shape, and it is big enough."""
    try:
        pts = np.array(json.loads(raw), dtype=np.float32).reshape(4, 2)
    except Exception:
        raise ValueError("corners must be JSON [[x,y],[x,y],[x,y],[x,y]] (top-left, top-right, bottom-right, bottom-left)")
    if not np.isfinite(pts).all():
        raise ValueError("corners contain invalid numbers")
    m = 0.05
    if (pts[:, 0] < -m * w).any() or (pts[:, 0] > (1 + m) * w).any() or (pts[:, 1] < -m * h).any() or (pts[:, 1] > (1 + m) * h).any():
        raise ValueError("corners are outside the image")
    if not cv2.isContourConvex(pts.reshape(-1, 1, 2)):
        raise ValueError("corners must form a convex four-sided shape, in order around the card")
    if cv2.contourArea(pts) < 0.004 * w * h:
        raise ValueError("the card is too small in the picture - move closer")
    return pts


def create_app(agent=None):
    """Build the Flask app. `agent` may be given (tests); otherwise it is created on first use.

    The "app factory" pattern: the tests call create_app(agent=FakeAgent()) to get an isolated server."""
    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
    state = {"agent": agent}

    def get_agent():
        # created lazily because importing the recognizer + loading its database takes many seconds
        if state["agent"] is None:
            from agent import PokemonCardAgent                 # heavy import: only when really needed
            state["agent"] = PokemonCardAgent()
        return state["agent"]
    app.get_agent = get_agent

    try:                                                       # optional: gzip/brotli for scripts, wasm and JSON
        from flask_compress import Compress
        app.config.setdefault("COMPRESS_MIMETYPES", ["text/html", "text/css", "text/xml", "application/json",
                                                      "application/javascript", "text/javascript", "application/wasm"])
        Compress(app)
    except ImportError:
        pass

    @app.after_request
    def cache_headers(r):
        # runs after every response. Files under /static/vendor never change for a given URL -> cache for a year.
        # Our own scripts are loaded as /static/js/x.js?v=<build id> (the id changes with every edit) -> the same year-long
        # cache is safe, and a phone then never re-downloads or re-checks a script it already has.
        if request.path.startswith("/static/vendor/"):
            r.headers["Cache-Control"] = "public, max-age=31536000, immutable"     # versioned library files
        elif request.path.startswith("/static/js/") and request.args.get("v"):
            r.headers["Cache-Control"] = "public, max-age=31536000, immutable"
        return r

    page_cache = {}                                            # (name, mode, title) -> (file mtime, html, etag): pages are built once, not per request

    def page(name, mode=None, title=None):
        # An HTML page with the <!--PC3D_HEAD--> marker replaced by head_block(). Look in static/ first, then next to app.py, so a misplaced file still works.
        # Efficiency: the finished page is kept in RAM until the file changes, and sent with an ETag so a phone that already has
        # it gets a tiny "304 Not Modified" instead of the whole page again.
        for d in (STATIC, BASE):
            f = d / name
            if f.exists():
                key, mtime = (name, mode, title), f.stat().st_mtime
                hit = page_cache.get(key)
                if not hit or hit[0] != mtime:
                    html = f.read_text(encoding="utf-8").replace("<!--PC3D_HEAD-->", head_block())
                    if mode:                                   # detect.html serves two pages: Detect and Battle
                        html = html.replace("{{MODE}}", mode).replace("{{TITLE}}", title or mode.title())
                    import hashlib
                    hit = page_cache[key] = (mtime, html, '"' + hashlib.md5(html.encode("utf-8")).hexdigest()[:16] + '"')
                if request.headers.get("If-None-Match") == hit[2]:
                    return Response(status=304, headers={"ETag": hit[2], "Cache-Control": "no-cache"})
                return Response(hit[1], mimetype="text/html", headers={"Cache-Control": "no-cache", "ETag": hit[2]})
        return f"{name} not found next to app.py or in static/", 404

    @app.get("/")
    def index():
        return page("home.html")                               # the landing page: choose Detect or Battle (and its options)

    @app.get("/detect")
    def detect():
        return page("detect.html", "detect", "Detect")         # ?go=ar | ?go=viewer starts straight away (links on the home page)

    @app.get("/scan")
    def scan_page():
        return page("index.html")                              # the original single-photo page

    @app.get("/battle")
    def battle():
        return page("detect.html", "battle", "Battle")

    @app.get("/models/<path:f>")
    def model(f):
        # max_age lets the browser keep models for a day (they're immutable per file name)
        # (the browser also revalidates with the ETag Flask adds, so an unchanged model costs a 304, not a download)
        return send_from_directory(MODELS, f, mimetype="model/gltf-binary", max_age=7 * 86400)

    @app.get("/api/health")
    def health():
        a = state["agent"]
        import card_reader
        return jsonify(ok=True, agent_loaded=a is not None, models=a.models.count if a else None, vendor=vendor.status(), tesseract=card_reader.tesseract_info())

    @app.post("/api/recognize")
    def recognize():
        f = request.files.get("image")
        if not f:
            return jsonify(error="No image uploaded"), 400
        try:
            return jsonify(get_agent().run(f.read()))
        except Exception as e:                                 # show the error in the UI instead of a blank page
            app.logger.exception("agent failed")
            return jsonify(error=f"{type(e).__name__}: {e}"), 500

    @app.post("/api/battle/scan")
    def battle_scan():
        # The scan pipeline. The picture arrives in one of three modes (see the module docstring):
        #   snapshot=1  -> it IS the straightened card (the browser did the work): go straight to the recogniser.
        #   corners     -> a camera frame + the tracker's corners: re-measure them at full resolution, then straighten.
        #   (neither)   -> a plain photo (AR off): find the card border ourselves; if there is none, use the whole photo.
        # Then agent.run_card (learned memory -> cache -> recogniser, 180-degree retry) and return JSON.
        f = request.files.get("image")
        if not f:
            return jsonify(error="No image uploaded"), 400
        img = cv2.imdecode(np.frombuffer(f.read(), np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            return jsonify(error="That file is not an image"), 400
        h, w = img.shape[:2]
        if max(h, w) > MAX_IMAGE_PX:
            return jsonify(error=f"Image too large (max {MAX_IMAGE_PX} px)"), 400
        try:
            sensitivity = float(np.clip(float(request.form.get("sensitivity", 0.5)), 0, 1))
        except ValueError:
            return jsonify(error="sensitivity must be a number"), 400
        try:
            scan = {"mode": "snapshot"}
            if request.form.get("snapshot") == "1":
                crop = img if max(h, w) <= 1100 else cv2.resize(img, None, fx=1100 / max(h, w), fy=1100 / max(h, w), interpolation=cv2.INTER_AREA)
            elif request.form.get("corners"):
                try:
                    hint = parse_corners(request.form["corners"], w, h)
                except ValueError as e:
                    return jsonify(error=str(e)), 400
                corners, refined = card_detect.refine_with_hint(img, hint, sensitivity=sensitivity)   # OpenCV, full resolution
                crop = card_detect.rectify(img, corners, pad=CARD_PAD)                                 # upright card
                scan = {"mode": "corners", "refined": bool(refined)}
            else:
                dets = card_detect.detect_cards(img, max_cards=1, sensitivity=sensitivity)
                if dets:
                    crop = card_detect.rectify(img, dets[0].corners, pad=CARD_PAD)
                    scan = {"mode": "auto", "detected": True, "border": dets[0].color, "corners": np.round(dets[0].corners, 1).tolist()}
                else:                                                                                  # no border found: let the recogniser try the whole photo
                    crop = img if max(h, w) <= 1100 else cv2.resize(img, None, fx=1100 / max(h, w), fy=1100 / max(h, w), interpolation=cv2.INTER_AREA)
                    scan = {"mode": "auto", "detected": False}
            res = get_agent().run_card(crop)
            if scan["mode"] != "snapshot":                     # the browser has its own copy of a snapshot; otherwise send a small preview back
                thumb = cv2.resize(crop, (160, round(160 * crop.shape[0] / crop.shape[1])), interpolation=cv2.INTER_AREA)
                res["crop_jpeg"] = "data:image/jpeg;base64," + base64.b64encode(cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 80])[1]).decode()
            scan["crop_size"] = [int(crop.shape[1]), int(crop.shape[0])]
            res["scan"] = scan
            return jsonify(res)
        except Exception as e:
            app.logger.exception("battle scan failed")
            return jsonify(error=f"{type(e).__name__}: {e}"), 500

    @app.get("/api/battle/ocr/<token>/<part>.png")
    def ocr_region(token, part):
        # the picture the browser's tesseract.js should read (only exists when the server has no Tesseract program)
        png = get_agent().ocr_image(token, part)
        if png is None:
            return jsonify(error="unknown OCR job"), 404
        return Response(png, mimetype="image/png", headers={"Cache-Control": "no-store"})

    @app.post("/api/battle/ocr-words")
    def ocr_words():
        # JSON {token, hp: [[word...], ...], attack: [word...]} - words as {text,left,top,width,height,conf,line}; validated server-side
        d = request.get_json(silent=True) or {}
        out = get_agent().finish_client_ocr(str(d.get("token", ""))[:40], d.get("hp"), d.get("attack"))
        return jsonify(out), (200 if out.get("ok") else 404)

    @app.get("/api/battle/opponent")
    def opponent():
        # ?exclude=25,6 -> do not pick these Pokédex numbers. Returns lookup()-shaped JSON plus stats {hp, attacks}.
        try:
            ex = [int(x) for x in request.args.get("exclude", "").split(",") if x.strip().isdigit()][:20]
            return jsonify(get_agent().opponent(ex))
        except Exception as e:
            app.logger.exception("opponent failed")
            return jsonify(error=f"{type(e).__name__}: {e}"), 500

    @app.get("/api/pokemon")
    def pokemon():
        q = request.args.get("q", "").strip()
        if not q or len(q) > 40:
            return jsonify(error="Give a Pokémon name or Pokédex number (?q=pikachu)"), 400
        try:
            return jsonify(get_agent().lookup(q))
        except Exception as e:
            app.logger.exception("lookup failed")
            return jsonify(error=f"{type(e).__name__}: {e}"), 500

    @app.post("/api/feedback")
    def feedback():
        # JSON body: {result_id, client, target: "detection"|"model", vote: 1|-1, correct?: "pikachu"|"25", comment?}
        # The answer carries {ok, stats, result}; `result` (when not null) is a replacement answer to show right away.
        d = request.get_json(silent=True) or {}
        try:
            vote = int(d.get("vote", 0))
        except (TypeError, ValueError):
            vote = 0
        try:
            out = get_agent().submit_feedback(str(d.get("result_id", ""))[:40], d.get("client"), d.get("target"), vote,
                                              d.get("correct"), str(d.get("comment") or ""))
            return jsonify(out), (200 if out.get("ok") else 400)
        except Exception as e:
            app.logger.exception("feedback failed")
            return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500

    @app.post("/api/feedback/correct")
    def feedback_correct():
        # JSON body: {result_id, client, pokemon?, hp?, attacks?: [{name, damage, type}]}  ->  {ok, stats, result}
        d = request.get_json(silent=True) or {}
        try:
            out = get_agent().submit_correction(str(d.get("result_id", ""))[:40], d.get("client"), d.get("pokemon"), d.get("hp"), d.get("attacks"))
            return jsonify(out), (200 if out.get("ok") else 400)
        except Exception as e:
            app.logger.exception("correction failed")
            return jsonify(ok=False, error=f"{type(e).__name__}: {e}"), 500

    @app.get("/api/feedback/stats")
    def feedback_stats():
        dex = request.args.get("dex", type=int)                # optional: model scores for one Pokémon
        return jsonify(get_agent().feedback.stats(dex))

    return app


app = create_app()                                             # `flask --app app run` / gunicorn work too


def main(argv=None):
    ap = argparse.ArgumentParser(description="PokéCard 3D web server")
    ap.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"),
                    help="127.0.0.1 = this computer only (default); 0.0.0.0 = reachable from phones on your network")
    ap.add_argument("--port", type=int, default=int(os.environ.get("PORT", 5000)))
    ap.add_argument("--https", action="store_true", help="serve HTTPS with a self-signed certificate (phones need HTTPS for the camera)")
    ap.add_argument("--cert"), ap.add_argument("--key")
    args = ap.parse_args(argv)

    print("Loading card recognizer (first run can take a while)…")
    app.get_agent()                                            # load now, not on the first scan
    print("browser libraries:", ", ".join(f"{k}: {v}" for k, v in vendor.status().items()),
          "(python scripts/fetch_vendor.py self-hosts them)")

    ssl_context = None
    if args.cert and args.key:
        ssl_context = (args.cert, args.key)
    elif args.https:
        import tls
        ssl_context = tls.ensure_cert()
    scheme = "https" if ssl_context else "http"
    if args.host in ("0.0.0.0", "::"):
        import tls
        print("Open on your phone (same Wi-Fi):")
        for ip in tls.lan_ips():
            print(f"   {scheme}://{ip}:{args.port}/detect   (and /battle for the game)")
        if not ssl_context:
            print("   ! Phones only allow the camera on HTTPS - restart with --https (or use a tunnel, see README).")
        else:
            print("   (the browser warns about the self-made certificate: choose 'advanced' / 'continue')")
    print(f"On this computer: {scheme}://127.0.0.1:{args.port}/   detect: {scheme}://127.0.0.1:{args.port}/detect   battle: {scheme}://127.0.0.1:{args.port}/battle")
    app.run(host=args.host, port=args.port, threaded=True, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
