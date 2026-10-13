"""
agent.py - PokemonCardAgent
===========================

WHAT THIS FILE DOES
-------------------
It turns "a picture of a Pokémon card" into "a 3D model URL the browser can display".
Everything else in the project (the web server, the AR page, the card detector) either feeds
pictures to this class or shows what it returns.

Think of the class as an *agent*: a small program that owns several "tools" (methods) and decides
which to call, in what order, logging every decision so the web page can show a trace.

THE PIPELINE (read it top to bottom - the methods below appear in this same order)
---------------------------------------------------------------------------------
    card photo
        │  TOOL 1  recognize()        pokemon-card-recognizer (a neural network) reads the card ->
        ▼                             {set, name, number}   e.g. {"Base Set", "Charizard", "4"}
    card name
        │  TOOL 2  match_species()    "Dark Charizard", "Mr. Mime", "Zekrom-GX" ... -> Pokédex number
        ▼
    Pokédex number (pid)
        │  TOOL 3  fetch_pokemon()    PokeAPI: types, height, weight, artwork URL (cached on disk)
        ▼
    WHICH 3D MODEL?  (first rule that matches wins - see _finish())
        a. CUSTOM_MODELS   hand-picked model for one species (your Charizard STL)
        b. models/gen1/    the textured Gen-1 models (#1-151)             <- only these are used for Gen 1
        c. model pack      model_packs/regular.zip, if you own it         (glb_library.py)
        d. relief          smooth textured 3D figure inflated from the official artwork (relief.py)
        e. voxels          last resort: sprite -> blocky cubes (build_model() below)

BATTLE MODE adds two things on top (run_card): the card arrives already cut out and straightened by
card_detect.py, and finished answers are cached by a tiny image fingerprint so the slow neural network
only runs once per card, not once per camera frame.

WHAT TO LEARN FROM THIS FILE
----------------------------
* how to chain independent "tools" into one pipeline and keep a human-readable trace (log())
* how to cache at two levels: Pokémon facts (disk, pokemon_cache/) and card identification (RAM, scan_cache)
* how to protect a non-thread-safe model with a lock (Flask runs several requests at once)
* how a 2-D sprite can be turned into a 3-D mesh with nothing but numpy (build_model)
"""
import base64, copy, io, json, os, re, tempfile, threading, time
from collections import OrderedDict           # a dict that remembers insertion order -> perfect for an LRU cache
from pathlib import Path
import cv2, numpy as np, requests, trimesh    # cv2 = OpenCV, trimesh = writes .glb files
from PIL import Image                         # Pillow: loading / resizing pictures
from scipy.ndimage import distance_transform_edt   # "how far is each pixel from the nearest empty pixel?"
from pokemon_card_recognizer.api.card_recognizer import CardRecognizer, OperatingMode
import sprites                                  # our own sprite-finding module (sprites.py)
from sprites import norm                        # norm("Mr. Mime") -> "mrmime": lower-case letters/digits only
from glb_library import GlbLibrary          # pack of real Pokémon .glb models (regular.zip)
from feedback import FeedbackStore, PendingResults   # user ratings + what the app learns from them (feedback.py)
import uuid, random
import card_reader                          # reads HP / attacks / damage / energy circles off the card (card_reader.py)

# ----------------------------------------------------------------------------
# Windows fix: the recognizer's OCR step writes a temp PNG and re-opens it *by name*
# while Python still holds it open, which Windows forbids ("Permission denied").
# This wrapper creates the file with delete=False and removes it on close() instead.
# (Linux/macOS allow the re-open, so the fix is only installed when os.name == "nt".)
# ----------------------------------------------------------------------------
if os.name == "nt":
    _orig_ntf = tempfile.NamedTemporaryFile      # keep the real function so we can call it

    class _WinTempFile:
        """Looks like a NamedTemporaryFile, but the file is only deleted when close() is called."""
        def __init__(self, *args, **kwargs):
            self._auto_delete = kwargs.pop("delete", True)           # remember what the caller asked for...
            self._f = _orig_ntf(*args, delete=False, **kwargs)       # ...but never let Python delete it early
            self.name = self._f.name
        def __getattr__(self, item): return getattr(self._f, item)  # forward .write(), .flush(), ... to the real file
        def __enter__(self): return self
        def __exit__(self, *exc): self.close()
        def close(self):
            try: self._f.close()
            finally:
                if self._auto_delete:
                    try: os.remove(self.name)
                    except OSError: pass

    tempfile.NamedTemporaryFile = _WinTempFile   # monkey-patch: the recognizer now gets our version

# ----------------------------------------------------------------------------
# Configuration constants (all the "knobs" of the agent live here)
# ----------------------------------------------------------------------------
API = "https://pokeapi.co/api/v2"               # free Pokémon data API
ROOT = Path(__file__).parent                    # the folder this file is in
MODELS = ROOT / "models"; MODELS.mkdir(exist_ok=True)   # generated .glb files are cached here
POKE_CACHE = ROOT / "pokemon_cache"; POKE_CACHE.mkdir(exist_ok=True)   # PokeAPI answers, so repeat scans need no network
MAX_SCAN_PX = 1600   # photos are shrunk to this before OCR: big phone photos only make recognition slower

# Hand-made models that ALWAYS win for a species (Pokédex number -> file under models/). Charizard = the retro STL
# converted by scripts/stl_to_glb.py. For these Pokémon neither the model pack nor the sprite->voxel path is used.
CUSTOM_MODELS = {6: "custom/charizard.glb"}
GEN1_DIR = "gen1"                      # models/gen1/NNN.glb: the hand-made Gen 1 models (scripts/fbx_to_glb.py). They are the ONLY models used for #1-151.
CARD_PAD = 0.06          # margin of real surroundings kept around a straightened card (fraction of its size)
SCAN_CACHE_MAX = 128     # straightened cards remembered (battle mode shows the same few cards again and again)
SCAN_CACHE_DIST = 0.28   # thumbnail difference below this = same card (same card: <= 0.22, different cards: >= 0.50 in tests)

MIN_MATCH = 0.15     # a sprite whose colours match the official artwork less than this is rejected
SPRITE_MAX = 96      # sprites bigger than this (e.g. a user's 512px upload) are shrunk to this
ARTWORK_GRID = 72    # official artwork is shrunk to this many voxels on its longest side

# The 6 faces of a unit cube: (axis, direction, 4 corners in counter-clockwise order
# seen from outside). Used to emit only the faces that are visible (not touching another voxel).
# axis 0/1/2 = x/y/z, direction +1/-1 = which side of the voxel the face is on.
FACES = [
    (0, 1,  [(1,0,0),(1,1,0),(1,1,1),(1,0,1)]), (0, -1, [(0,0,0),(0,0,1),(0,1,1),(0,1,0)]),
    (1, 1,  [(0,1,0),(0,1,1),(1,1,1),(1,1,0)]), (1, -1, [(0,0,0),(1,0,0),(1,0,1),(0,0,1)]),
    (2, 1,  [(0,0,1),(1,0,1),(1,1,1),(0,1,1)]), (2, -1, [(0,0,0),(0,1,0),(1,1,0),(1,0,0)]),
]


class PokemonCardAgent:
    """One instance is created lazily by app.py (the first time a scan is needed) and shared by all requests."""

    def __init__(self, set_name="master"):
        # The neural-network recognizer keeps internal state while it works, so two requests must never
        # use it at the same time. Every call to the recognizer is wrapped in `with self.lock:`.
        self.lock = threading.Lock()
        # Loads the card database + OCR models (slow, so we do it once at start-up).
        self.recognizer = CardRecognizer(mode=OperatingMode.SINGLE_IMAGE, set_name=set_name)
        self.species = self._load_species()     # {"charizard": 6, "mrmime": 122, ...}
        self.models = GlbLibrary()              # indexes model_packs/*.zip (names only, instant)
        # Battle-mode memory: id -> (card fingerprint, finished result). OrderedDict = oldest first,
        # so popping from the front evicts the least-recently-used card (an "LRU cache").
        self.scan_cache = OrderedDict()
        self._cache_seq = 0                     # counter that hands out ids for cache entries
        self.cache_lock = threading.Lock()      # cache is touched by several request threads
        self.ocr_jobs = OrderedDict()           # token -> region images waiting for the BROWSER's tesseract.js (see client_ocr)
        self.http = requests.Session()          # re-uses the HTTPS connection to PokeAPI
        # User ratings (thumbs up/down). `feedback` = SQLite file + learned-card memory; `pending` = the last answers
        # given, kept in RAM so a vote can refer to one by id. See feedback.py for the whole story.
        self.feedback = FeedbackStore(ROOT / "feedback", match_dist=SCAN_CACHE_DIST)
        self.pending = PendingResults()
        ti = card_reader.tesseract_info()
        print("Tesseract OCR:", f"{ti['version']} ({ti['cmd']})" + ("" if ti["english"] else " - English data missing") if ti["available"]
              else "NOT FOUND - Battle will use default HP/attacks (README 1.1b)")
        print(f"3D model pack: {self.models.count} models for {len(self.models.index)} Pokémon")

    # ======================= TOOL 1: recognise the card =======================
    def recognize(self, path, exclude=()):
        """Run pokemon-card-recognizer on an image file -> {set, name, number} or None.

        `exclude` = {(set, number), ...} cards users have thumbs-downed for this very picture; those guesses are skipped
        and the recogniser's next-best guess is used instead (learning mechanism 1 in feedback.py).

        The library embeds the image into a vector and finds the nearest card in its database, so it
        returns the exact printing (set + number), not just a name."""
        with self.lock:                                           # one recognition at a time
            preds = self.recognizer.exec(path)                    # list of ranked guesses
            if preds is None or len(preds) == 0:
                return None                                       # nothing that looks like a card
            # turn the best guess (preds[0]) into a readable record from the reference database
            for p in preds:                                       # ranked best-first
                c = self.recognizer.classifier.reference.lookup_card_prediction(card_prediction=p)
                rec = {"set": self.set_label(c.set), "name": str(c.name), "number": str(c.number)}
                if (rec["set"], rec["number"]) not in exclude:    # skip guesses people have rejected for this card
                    return rec
        return None

    @staticmethod
    def set_label(s):
        """A short, printable name for the card's set. The recogniser's database stores the set as an OBJECT whose str() is a
        huge dump (id, image URLs, legalities, dates...). Using that as text broke file names and the UI on Windows, so take
        the set's name (else its id), and cap the length."""
        label = getattr(s, "name", None) or getattr(s, "id", None) or str(s)
        return re.sub(r"\s+", " ", str(label)).strip()[:60]

    # ======================= TOOL 2: card name -> species =====================
    def _load_species(self):
        """Download (once) and cache a {normalised name: dex number} table from PokeAPI.

        PokeAPI lists species like {"name": "charizard", "url": ".../pokemon-species/6/"}; the Pokédex
        number is the last path segment of the url. The table is saved to species.json so we only ever
        download it once."""
        f = ROOT / "species.json"
        if f.exists():
            return json.loads(f.read_text())
        r = requests.get(f"{API}/pokemon-species?limit=2000", timeout=20); r.raise_for_status()
        d = {norm(x["name"]): int(x["url"].rstrip("/").split("/")[-1]) for x in r.json()["results"]}
        f.write_text(json.dumps(d)); return d

    def match_species(self, card_name):
        """'Dark Charizard', 'Charizard ex', 'Mr. Mime' ... -> dex number (or None for Trainer/Energy).

        Card names carry extra words (owner, suffix: "Brock's Onix", "Pikachu VMAX"), so we slide a window
        over the words and test every 3-, 2- and 1-word phrase against the species table."""
        words = [w for w in re.split(r"[\s&]+", card_name) if w]
        for i in range(len(words)):              # leftmost match wins; try 3-,2-,1-word names
            for n in (3, 2, 1):
                key = norm("".join(words[i:i + n]))      # "Mr." + "Mime" -> "mrmime"
                if key in self.species:
                    return self.species[key]
        flat = norm(card_name)                   # fallback: name hidden inside a word ("Zekrom-GX")
        hits = [k for k in self.species if len(k) >= 4 and k in flat]
        return self.species[max(hits, key=len)] if hits else None   # longest hit = most specific name

    # ======================= TOOL 3: Pokémon facts =============================
    def fetch_pokemon(self, pid):
        """Types / size / artwork URL. Cached on disk so each Pokémon hits the network once ever.

        The record returned here is also what the web page shows (name, types) and what detect.js uses to
        size the model (height_m)."""
        cache = POKE_CACHE / f"{pid}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        d = self.http.get(f"{API}/pokemon/{pid}", timeout=20).json()
        s = d["sprites"]
        # prefer the high-resolution "official artwork", fall back to the small front sprite
        art = (s.get("other", {}).get("official-artwork", {}) or {}).get("front_default") or s.get("front_default")
        pk = {"id": d["id"], "slug": d["name"], "name": d["name"].replace("-", " ").title(),
              "types": [t["type"]["name"] for t in d["types"]],
              "height_m": d["height"] / 10, "weight_kg": d["weight"] / 10, "artwork": art}   # PokeAPI uses decimetres / hectograms
        cache.write_text(json.dumps(pk))
        return pk

    # ======================= TOOL 4: pick the sprite ==========================
    # (Only used by the last-resort voxel path. The relief builder uses the artwork directly.)
    def _download_artwork(self, pk):
        """Official artwork as RGBA, trimmed. Used as a fallback AND as the comparison reference."""
        if not pk["artwork"]:
            return None
        img = Image.open(io.BytesIO(self.http.get(pk["artwork"], timeout=30).content)).convert("RGBA")
        return sprites.crop_to_content(img)      # cut away the transparent margin

    def choose_sprite(self, pk, artwork, log):
        """
        Try each sprite source in priority order. Before accepting a candidate, COMPARE it
        with the official artwork (colour-palette similarity). A file that looks nothing like
        the Pokémon (wrong name, wrong picture) is rejected and the next source is tried.
        Returns (image, kind, detail, similarity).
        """
        candidates = [
            # 1) a sprite the user dropped into sprites/ (see sprites.find_custom_sprite)
            ("custom", lambda: sprites.find_custom_sprite(pk["id"], pk["slug"])),
            # 2) the FireRed/LeafGreen sprite chart cut up by sprites.sheet_sprite
            ("sheet",  lambda: (lambda s: (s, "FireRed/LeafGreen chart") if s else None)(sprites.sheet_sprite(pk["id"]))),
        ]
        for kind, finder in candidates:
            found = finder()
            if not found:
                log("Sprite", f"No {kind} sprite for #{pk['id']}")
                continue
            img, detail = found
            sim = sprites.similarity(img, artwork) if artwork else 1.0     # the comparison step (0..1)
            log("Compare", f"{kind} sprite ({detail}) vs official artwork: similarity {sim:.2f}")
            if sim >= MIN_MATCH:
                return img, kind, detail, sim
            log("Compare", f"Below {MIN_MATCH} - rejected, trying next source")
        if artwork is None:
            return None, None, None, 0.0
        return artwork, "artwork", "official artwork (assumed - no usable sprite)", 1.0

    # ======================= TOOL 5: sprite -> 3D model =======================
    def build_model(self, img, kind, pid):
        """
        Turn a flat RGBA sprite into a 3D voxel model and save it as a .glb.
        One sprite pixel = one cube. Depth ('puffiness') comes from how far each pixel is
        from the sprite's outline, so the middle is thick and the edges thin.

        The algorithm in five steps (follow the numbered comments below):
          1. choose a resolution            2. make a 3-D boolean grid ("filled")
          3. keep only visible cube faces   4. write triangles + colours  5. export with trimesh
        """
        out = MODELS / f"{pid}_{kind}.glb"
        if out.exists():
            return out, True                      # already built earlier -> reuse

        # --- 1. resolution ---------------------------------------------------------------
        # Pixel-art sprites keep their native resolution (1 pixel = 1 voxel) unless huge;
        # photographic artwork is shrunk smoothly so it doesn't create millions of cubes.
        longest = max(img.size)
        if kind == "artwork" and longest != ARTWORK_GRID:
            s = ARTWORK_GRID / longest
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.LANCZOS)   # smooth shrink
        elif kind != "artwork" and longest > SPRITE_MAX:
            s = SPRITE_MAX / longest
            img = img.resize((max(1, round(img.width * s)), max(1, round(img.height * s))), Image.NEAREST)  # keep pixels crisp

        a = np.asarray(img)                                # shape (rows, cols, 4) RGBA
        rgb = a[..., :3][::-1].transpose(1, 0, 2)          # arrays indexed [x, y(up), channel]  (flip rows: image y points down)
        mask = (a[..., 3] > 128)[::-1].T                   # True where the sprite has a pixel (alpha above half)
        W, H = mask.shape

        # --- 2. thickness ----------------------------------------------------------------
        # distance (in pixels) from each sprite pixel to the nearest transparent pixel;
        # np.pad adds a transparent border so pixels touching the image edge count as "outline"
        d = distance_transform_edt(np.pad(mask, 1))[1:-1, 1:-1]
        half = max(3, round(0.2 * max(W, H)))              # max half-thickness in voxels
        # sqrt() gives a rounded, pillow-like profile; every sprite pixel gets >= 1 layer
        h = np.maximum(1, np.round(half * np.sqrt(np.clip(d / max(d.max(), 1), 0, 1)))).astype(int)

        # filled[x, y, z] is True for voxels that exist (z runs front-to-back through the model)
        k = np.arange(2 * half)[None, None, :] + 0.5 - half    # z position of each layer, centred on 0
        filled = mask[..., None] & (np.abs(k) < h[..., None])  # a voxel exists if its |z| is inside that pixel's thickness
        color = np.broadcast_to(rgb[:, :, None, :], (W, H, 2 * half, 3))   # same colour through depth

        # --- 3+4. Build the mesh: only faces whose neighbour voxel is empty are visible. ----
        # (Internal faces between two touching cubes can never be seen, so skipping them keeps the file small.)
        P, V, F, C, n = np.pad(filled, 1), [], [], [], 0
        for ax, sg, quad in FACES:
            # shift the grid one step along this axis; a voxel with an empty neighbour there shows a face
            neighbour = np.roll(P, -sg, axis=ax)[1:-1, 1:-1, 1:-1]
            idx = np.argwhere(filled & ~neighbour)                         # visible faces
            m = len(idx)
            if not m:
                continue
            V.append((idx[:, None, :] + np.array(quad)[None]).reshape(-1, 3))   # 4 corners / face
            C.append(np.repeat(color[idx[:, 0], idx[:, 1], idx[:, 2]], 4, axis=0))
            st = n + 4 * np.arange(m)[:, None]
            F.append(np.concatenate([st + [0, 1, 2], st + [0, 2, 3]])); n += 4 * m   # 2 triangles / face
        V = np.concatenate(V) - np.array([W / 2, H / 2, half])             # centre on the origin
        # glTF stores vertex colours in LINEAR space; sprites are sRGB, so convert.
        lin = ((np.concatenate(C) / 255.0) ** 2.2 * 255).astype(np.uint8)
        rgba = np.hstack([lin, np.full((len(lin), 1), 255, np.uint8)])
        # --- 5. export ---------------------------------------------------------------------
        mesh = trimesh.Trimesh(V, np.concatenate(F), vertex_colors=rgba, process=False)
        _ = mesh.vertex_normals                                            # export normals -> crisp shading
        mesh.export(str(out))
        return out, False

    @staticmethod
    def _preview(img):
        """Sprite as a data-URI PNG (enlarged, crisp) so the web page can show what was used."""
        big = img.resize((img.width * 4, img.height * 4), Image.NEAREST) if max(img.size) < 100 else img
        buf = io.BytesIO(); big.save(buf, "PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    # ======================= the agent loop ====================================
    def _new_run(self):
        """A fresh result dict + a log() that records a timed step into its trace.

        `res` is the JSON the browser receives. `log(step, detail)` appends to res["trace"], which the scan
        page prints as a step-by-step explanation of what the agent did and how long each step took."""
        trace, t0 = [], time.time()
        def log(step, detail): trace.append({"step": step, "detail": detail, "t": round(time.time() - t0, 2)})
        res = {"card": None, "pokemon": None, "model_url": None, "model_source": None,
               "sprite_png": None, "trace": trace, "message": None}
        return res, log

    def run(self, image_bytes):
        """Full pipeline for one uploaded/captured image (the scan page). Returns a JSON-able dict."""
        res, log = self._new_run()

        # --- 1. recognise the card (the library wants a file path) ---
        fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        try:
            im = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            im.thumbnail((MAX_SCAN_PX, MAX_SCAN_PX))          # no-op for small images
            im.save(path, quality=95)
            log("Recognize", "Running pokemon-card-recognizer…")
            card = self.recognize(path)
        finally:
            os.remove(path)                                   # always clean up the temp file
        if not card:
            res["message"] = "No card detected. Try a flatter, brighter, closer shot."
            log("Recognize", "No card found"); return res
        res = self._finish(card, res, log)                    # everything after "which card is it?"
        return self._register(res, "scan", jpeg=self._jpeg(im, 640))

    def _recognize_array(self, bgr, exclude=()):
        """recognize() for an in-memory BGR image (OpenCV's channel order is Blue-Green-Red)."""
        fd, path = tempfile.mkstemp(suffix=".jpg"); os.close(fd)
        try:
            cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 93])
            return self.recognize(path, exclude)
        finally:
            os.remove(path)

    # ---- battle mode: a card that was already found and straightened (card_detect.rectify) ----
    @staticmethod
    def _signature(crop, pad=CARD_PAD):
        """Tiny colour fingerprint of the card face (24x32, brightness/contrast-normalised per channel).
        Used to recognise a card we have already identified, so the slow recogniser only runs once per
        card. Unlike a plain 8x8 hash it separates cards that share the same template (same layout, different art).

        Steps: crop away the padding and the yellow border -> shrink to 24x32 -> subtract the mean and
        divide by the standard deviation of each colour channel. The last step makes the fingerprint
        independent of how bright the room is, so the same card under a different lamp still matches."""
        h, w = crop.shape[:2]
        m = (pad + 0.09) / (1 + 2 * pad)                       # skip the padding and the yellow border
        t = cv2.resize(crop[int(h * m):int(h * (1 - m)), int(w * m):int(w * (1 - m))], (24, 32),
                       interpolation=cv2.INTER_AREA).astype(np.float32)
        t -= t.mean(axis=(0, 1))
        return t / (t.std(axis=(0, 1)) + 1e-3)                 # +1e-3 avoids dividing by zero on flat images

    def _cache_lookup(self, crop):
        """-> (result copy, rotated) if this card (either way up) was identified before, else None.

        We compare against BOTH the crop and the crop turned 180°, because on a table the card can be
        seen upside down. `rotated` tells the caller which way it matched."""
        s0, s1 = self._signature(crop), self._signature(cv2.rotate(crop, cv2.ROTATE_180))
        with self.cache_lock:
            best = None
            for cid, (sig, _) in self.scan_cache.items():
                d0, d1 = float(np.abs(s0 - sig).mean()), float(np.abs(s1 - sig).mean())   # mean absolute difference
                d, rot = (d0, False) if d0 <= d1 else (d1, True)
                if d < SCAN_CACHE_DIST and (best is None or d < best[0]):
                    best = (d, cid, rot)
            if best is None:
                return None
            self.scan_cache.move_to_end(best[1])                 # most recently used stays longest
            return copy.deepcopy(self.scan_cache[best[1]][1]), best[2]   # deepcopy: callers may edit their copy

    def _cache_store(self, crop, rotated, res):
        """Remember a finished result under the fingerprint of the card's readable (upright) orientation."""
        sig = self._signature(cv2.rotate(crop, cv2.ROTATE_180) if rotated else crop)    # the readable way up
        with self.cache_lock:
            self._cache_seq += 1
            self.scan_cache[self._cache_seq] = (sig, copy.deepcopy(res))
            while len(self.scan_cache) > SCAN_CACHE_MAX:
                self.scan_cache.popitem(last=False)              # evict the oldest entry

    def run_card(self, crop):
        """Pipeline for a card already cut out and straightened (BGR numpy image, upright as seen by the
        camera - it may still be upside down). Tries the recogniser as given, then turned 180 degrees.
        Adds res["rotated"] = True when the card was upside down relative to the crop. Results are cached.

        Order of attempts (fastest first):
          0. LEARNED MEMORY - a card people have rated before (feedback.py). Their thumbs beat the recogniser.
          1. SCAN CACHE     - a card we identified earlier in this server run.
          2. RECOGNISER     - the neural network, skipping guesses people rejected for this card.
        Every successful answer gets a fresh result_id so the browser can rate it (_register)."""
        res, log = self._new_run()
        turned = cv2.rotate(crop, cv2.ROTATE_180)
        entry, l_rot, excluded = self.feedback.match(self._signature(crop), self._signature(turned))
        if entry:                                                # 0. learned from a thumbs-up or a correction
            card = {"name": entry["card_name"] or "", "set": entry["card_set"] or "", "number": entry["card_number"] or ""}
            log("Learned", f"Matches a card people rated before ({entry['kind']}) - recogniser skipped")
            res = self._finish(card, res, log, pid=entry["dex"] if entry["dex"] else None, show_card=bool(card["name"]))
            res["rotated"], res["cached"], res["learned"] = l_rot, False, True
            self._attach_stats(res, crop, l_rot)
            return self._register(res, "battle", crop=crop, rotated=l_rot)
        hit = self._cache_lookup(crop) if not excluded else None  # 1. have we seen this card before?
        if hit:
            cached, rotated = hit
            cached["rotated"], cached["cached"] = rotated, True
            cached["trace"] = [{"step": "Cache", "detail": "Same card as an earlier scan - recogniser skipped", "t": 0.0}]
            return self._register(cached, "battle", crop=crop, rotated=rotated)
        log("Recognize", "Running pokemon-card-recognizer on the straightened card…")
        if excluded:
            log("Recognize", f"Skipping {len(excluded)} guess(es) people marked wrong for this card")
        rotated, card = False, self._recognize_array(crop, excluded)       # 2. try as seen
        if not card:
            log("Recognize", "Nothing readable - trying the card turned 180°")
            rotated, card = True, self._recognize_array(turned, excluded)  # 3. then upside down
        if not card:
            res["message"] = "Couldn't read the card. Hold it flat, in good light, and keep it in view."
            res["rotated"] = False
            log("Recognize", "No card found"); return res
        res = self._finish(card, res, log)                       # 4. same ending as the scan page
        res["rotated"], res["cached"] = rotated, False
        if res.get("energy") and res["energy"]["type"] is None and not res["energy"]["special"]:
            res["energy"]["type"] = card_reader.energy_type_from_crop(cv2.rotate(crop, cv2.ROTATE_180) if rotated else crop)   # name had no type word: use the symbol colour
        self._attach_stats(res, crop, rotated)                   # HP + attacks (card_reader.py), cached on disk per card
        if (res.get("model_url") or res.get("energy")) and not excluded:     # remember good results only
            self._cache_store(crop, rotated, res)
        return self._register(res, "battle", crop=crop, rotated=rotated)

    # ======================= user ratings (see feedback.py) ======================
    @staticmethod
    def _jpeg(img, side):
        """A PIL image or BGR array -> JPEG bytes no larger than `side` px (training pictures stay small)."""
        if isinstance(img, np.ndarray):
            h, w = img.shape[:2]; k = min(1.0, side / max(h, w))
            if k < 1: img = cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 88]); return buf.tobytes() if ok else None
        img = img.copy(); img.thumbnail((side, side)); b = io.BytesIO(); img.save(b, "JPEG", quality=88); return b.getvalue()

    def _register(self, res, source, crop=None, rotated=False, jpeg=None):
        """Give a finished answer an id the browser can send back with a vote, and remember what the vote is about.
        Kept in RAM only (PendingResults); it reaches the database when somebody actually votes."""
        if not res.get("model_url") or not res.get("pokemon"):
            return res                                           # nothing to rate
        rid = uuid.uuid4().hex[:16]
        card = res.get("card") or {}
        ctx = {"source": source, "dex": res["pokemon"]["id"], "card_set": card.get("set") or None,
               "card_number": card.get("number") or None, "card_name": card.get("name") or None,
               "model_kind": (res.get("model_source") or {}).get("kind"), "model_url": res["model_url"],
               "signature": None, "crop_jpeg": jpeg}
        if crop is not None:                                     # battle mode: fingerprint of the READABLE orientation
            ctx["signature"] = self._signature(cv2.rotate(crop, cv2.ROTATE_180) if rotated else crop)
            ctx["crop_jpeg"] = self._jpeg(cv2.rotate(crop, cv2.ROTATE_180) if rotated else crop, 440)
        self.pending.put(rid, ctx)
        res["result_id"] = rid
        return res

    def submit_feedback(self, result_id, client, target, vote, correct=None, comment=""):
        """Record a thumbs-up (+1) / thumbs-down (-1) and apply what can be learned from it immediately.

        target  'detection' = "was this the right Pokémon/card?"   'model' = "is this 3D model good?"
        correct (detection thumbs-down only) a name or Pokédex number of what the card really is.
        Returns {"ok": True, "stats": {...}, "result": <new answer>|None}. `result` is present when the vote changes what
        should be on screen (a correction, or a model that has just been demoted); the browser then swaps to it."""
        if target not in ("detection", "model") or vote not in (1, -1):
            return {"ok": False, "error": "target must be 'detection' or 'model' and vote 1 or -1"}
        client = str(client or "")[:64] or "anonymous"
        ctx = self.pending.get(result_id) or self.feedback.get_result(result_id)
        if not ctx:
            return {"ok": False, "error": "That result has expired - scan the card again."}
        fix = None
        if correct not in (None, "") and target == "detection" and vote < 0:
            q = str(correct).strip()
            fix = int(q) if q.isdigit() else self.match_species(q)
            if not fix or fix < 1:
                return {"ok": False, "error": f"“{q}” isn't a Pokémon I know."}
        before = self.feedback.demoted_stages(ctx["dex"])        # model kinds already retired, to notice a NEW retirement
        self.feedback.save_result(result_id, ctx)
        self.feedback.add_vote(result_id, client, target, vote, fix, comment)
        sig = ctx.get("signature")
        if target == "detection" and sig is not None:
            card = {"set": ctx["card_set"], "number": ctx["card_number"], "name": ctx["card_name"]}
            if vote > 0:
                self.feedback.learn("confirmed", sig, ctx["dex"], card)
            elif fix:
                self.feedback.learn("corrected", sig, fix, {"set": None, "number": None, "name": None})
            elif ctx["card_set"]:
                self.feedback.learn("rejected", sig, None, card)
        with self.cache_lock:
            self.scan_cache.clear()                              # cached answers may now be out of date
        new = None
        if fix and fix != ctx["dex"]:
            new = self.lookup(fix)
        elif target == "model" and self.feedback.demoted_stages(ctx["dex"]) != before:
            new = self.lookup(ctx["dex"])                        # a model was just demoted: show the replacement
        if new is not None and new.get("result_id"):            # lookup() registered it as "manual": inherit the card's picture,
            c2 = self.pending.get(new["result_id"])             # so a later vote on the new answer can still learn
            if c2 is not None:
                c2.update(source=ctx["source"], signature=sig, crop_jpeg=ctx.get("crop_jpeg"))
        return {"ok": True, "stats": self.feedback.stats(ctx["dex"]), "result": new}

    # ======================= energy cards, card text, opponents =====================
    @staticmethod
    def read_energy(name):
        """'Fire Energy' / 'Basic Lightning Energy' / 'Double Colorless Energy' -> {"type", "special", "name"} or None.
        Energy cards are recognised by the word "Energy" in the card name (the recogniser already read it); the TYPE comes
        from the type word in the name. `type` uses PokeAPI names (lightning -> electric, ...) and is None for cards that
        work for any type (Rainbow Energy). `special` = anything other than a plain "Basic <Type> Energy"/"<Type> Energy".
        Later this is what a type-based healing mechanic can read."""
        words = re.findall(r"[A-Za-z]+", name or "")
        if not any(w.lower() == "energy" for w in words):
            return None
        etype = next((t for t in (card_reader.to_type(w) for w in words) if t), None)
        plain = {"basic", "energy"} | {w.lower() for w in words if card_reader.to_type(w)}
        special = any(w.lower() not in plain for w in words) or "double" in (name or "").lower()
        return {"type": etype, "special": bool(special), "name": name}

    def card_stats(self, crop, card, pokemon_types):
        """HP + attacks for a Pokémon card, read from the straightened picture (card_reader.py) - read ONCE per card:
        the answer is cached on disk in card_cache/<set>_<number>.json, so the same card shown again (this session or
        after a restart) is never read twice. Falls back to default stats, with a note, when text can't be read."""
        key = None
        if card and card.get("set") and card.get("number"):
            key = self.card_cache_file(card)
            if key.exists():
                try:
                    return json.loads(key.read_text())
                except Exception:
                    pass                                          # damaged file: read again
        stats = None
        if crop is not None:
            try:
                stats = card_reader.read_card(crop, pokemon_types, CARD_PAD)
            except Exception as e:                                # OCR must never break the scan
                print("card_reader failed:", type(e).__name__, e)
            if stats is None and not card_reader.tesseract_available():
                stats = {"hp": None, "attacks": [], "confidence": "low", "source": "default",
                         "notes": ["Card text can't be read because Tesseract isn't installed (see README) - using default stats."]}
        stats = self.complete_stats(stats, pokemon_types)
        if key and stats["source"] == "ocr":
            self._save_card_cache(key, stats)
        elif crop is not None and stats["source"] == "default" and not card_reader.tesseract_available():
            try:                                                  # no Tesseract program here: let the browser read it with tesseract.js
                stats["client_ocr"] = self._client_ocr_job(crop, key, pokemon_types)
                stats["notes"].append("Reading the card text in your browser (tesseract.js)…")
            except Exception as e:
                print("client OCR job failed:", type(e).__name__, e)
        return stats

    @staticmethod
    def card_cache_file(card):
        """card_cache/<set>_<number>_<hash>.json - always a short, legal file name (Windows forbids many characters and
        long paths). The hash keeps two different sets with the same cleaned-up name apart."""
        import hashlib
        slug = lambda t, n: re.sub(r"[^A-Za-z0-9]+", "_", str(t)).strip("_")[:n]
        h = hashlib.sha1(f"{card['set']}|{card['number']}".encode("utf-8")).hexdigest()[:8]
        return ROOT / "card_cache" / f"{slug(card['set'], 30)}_{slug(card['number'], 12)}_{h}.json"

    @staticmethod
    def _save_card_cache(key, stats):
        """Write the cache file; a disk problem must never break a scan, so failures are only logged."""
        try:
            key.parent.mkdir(exist_ok=True)
            key.write_text(json.dumps(stats), encoding="utf-8")
        except OSError as e:
            print("card cache not saved:", e)

    # ---- browser-side OCR (tesseract.js): used when the Tesseract PROGRAM is not installed on the server ----
    def _client_ocr_job(self, crop, key, pokemon_types):
        """Prepare the two region pictures for the browser and remember how to finish the job when its words come back.
        The browser downloads /api/battle/ocr/<token>/hp.png and attack.png, runs tesseract.js on them (WebAssembly, on the
        phone/desktop CPU), and POSTs the recognised words to /api/battle/ocr-words. Parsing stays here, in Python."""
        r = card_reader.prepare_regions(crop, CARD_PAD)
        png = lambda g: cv2.imencode(".png", g)[1].tobytes()
        token = uuid.uuid4().hex[:16]
        self.ocr_jobs[token] = {"hp": png(r["hp"]), "attack": png(r["attack"]), "width": int(r["attack"].shape[1]), "circles": r["circles"],
                                "types": list(pokemon_types or []), "key": key}
        while len(self.ocr_jobs) > 64:
            self.ocr_jobs.popitem(last=False)
        return token

    def ocr_image(self, token, part):
        """PNG bytes of one region for the browser, or None."""
        job = self.ocr_jobs.get(token)
        return job.get(part) if job and part in ("hp", "attack") else None

    def finish_client_ocr(self, token, hp_lists, attack_words):
        """Words read by the browser -> stats (validated, parsed, cached on disk like a server-side read).
        Returns {"ok": True, "stats": {...}} or {"ok": False, "error": ...}."""
        job = self.ocr_jobs.get(token)
        if not job:
            return {"ok": False, "error": "That OCR job has expired - show the card again."}
        hp_lists = [card_reader.sanitize_words(w) for w in (hp_lists if isinstance(hp_lists, list) else [])[:3]]
        stats = card_reader.stats_from_words(hp_lists, card_reader.sanitize_words(attack_words, max_x=job["width"] + 50),
                                             job["width"], job["circles"], job["types"])
        stats = self.complete_stats(stats, job["types"])
        stats["notes"].append("Read in the browser with tesseract.js.")
        key = job["key"]
        if key and stats["source"] == "ocr":
            self._save_card_cache(key, stats)
        with self.cache_lock:                                     # cards already remembered with the placeholder stats get the real ones
            for _, cached in self.scan_cache.values():
                if (cached.get("stats") or {}).get("client_ocr") == token:
                    cached["stats"] = copy.deepcopy(stats)
        self.ocr_jobs.pop(token, None)
        return {"ok": True, "stats": stats}

    @staticmethod
    def complete_stats(stats, pokemon_types):
        """Fill gaps with defaults so the battle always has an HP and at least one move. Anything we guessed is listed in
        `notes` so the UI can tell the player."""
        stats = dict(stats or {"hp": None, "attacks": [], "confidence": "low", "source": "default", "notes": []})
        stats["notes"] = list(stats.get("notes") or [])
        was_read = stats.get("source") == "ocr" and bool(stats.get("hp") or stats.get("attacks"))   # did OCR give us anything real?
        main = (pokemon_types or ["normal"])[0]
        if not stats.get("hp"):
            stats["hp"] = 100
            if not any("HP" in n for n in stats["notes"]):
                stats["notes"].append("HP not read - using 100.")
        if not stats.get("attacks"):
            stats["attacks"] = [{"name": "Tackle", "damage": 20, "modifier": None, "type": "normal", "cost": ["normal"], "text": ""},
                                {"name": main.title() + " Strike", "damage": 30, "modifier": None, "type": main, "cost": [main], "text": ""}]
            if not any("attacks" in n for n in stats["notes"]):
                stats["notes"].append("No attacks read - using default moves.")
        stats["source"] = "ocr" if was_read else "default"        # only real readings are cached on disk
        return stats

    def _attach_stats(self, res, crop, rotated):
        """Add res["stats"] for a Pokémon card read from `crop` (it is turned upright first if it was upside down)."""
        if not res.get("pokemon") or res.get("stats"):
            return res
        upright = cv2.rotate(crop, cv2.ROTATE_180) if rotated else crop
        types = res["pokemon"].get("types") or []
        fix = self.feedback.match_stats(self._signature(upright), res.get("card"))     # a player already told us what this card says?
        if fix and fix.get("hp") and fix.get("attacks"):
            res["stats"] = self.stats_from_fix(fix, None, types)                        # complete correction: no OCR needed at all
            return res
        res["stats"] = self.card_stats(upright, res.get("card"), types)
        if fix:                                                                         # partial correction: lay it over what was read
            res["stats"] = self.stats_from_fix(fix, res["stats"], types)
        return res

    # ======================= player corrections of HP + attacks (the Detect page's "Teach the agent") =====================
    @staticmethod
    def clean_attacks(raw, pokemon_types=()):
        """Validate + normalise attacks typed by a player -> [{name, damage|None, modifier, type, cost, text}] (at most 4).
        Never trust the browser: every field is length-limited and type-checked. Rows without a name are skipped.
        damage: empty / 0 / 'none' = a non-damaging move; digits with an optional + or x ('30+') = damage and modifier.
        type: one of the 18 PokeAPI types; blank = assumed (the Pokémon's own type)."""
        out = []
        for a in (raw if isinstance(raw, list) else [])[:12]:        # look at a few rows more than we keep: blank rows are skipped
            if len(out) >= 4:
                break
            if not isinstance(a, dict):
                continue
            name = re.sub(r"\s+", " ", re.sub(r"[\x00-\x1f\x7f<>]", "", str(a.get("name") or ""))).strip()[:30]
            if not name:
                continue
            m = re.match(r"^\s*(\d{1,3})\s*([+x×-]?)\s*$", str(a.get("damage") if a.get("damage") is not None else ""))
            damage = int(m.group(1)) if m and int(m.group(1)) > 0 else None
            modifier = {"+": "+", "x": "×", "×": "×", "-": "-"}.get(m.group(2)) if m and damage else None
            t = card_reader.to_type(str(a.get("type") or "")) or card_reader.assume_type([], list(pokemon_types or []))
            out.append({"name": name, "damage": damage, "modifier": modifier, "type": t, "cost": [t], "text": ""})
        return out

    @staticmethod
    def clean_hp(raw):
        """'60' / 60 -> 60 within 10..500 (cards go up to 340), else None."""
        try:
            v = int(float(str(raw).strip()))
        except (TypeError, ValueError):
            return None
        return v if 10 <= v <= 500 else None

    def stats_from_fix(self, fix, base, pokemon_types):
        """Merge a player's correction over `base` stats (what OCR read, or None) -> a full stats dict with source "user"."""
        stats = copy.deepcopy(base) if base else self.complete_stats(None, pokemon_types)
        if fix.get("hp"):
            stats["hp"] = int(fix["hp"])
        if fix.get("attacks"):
            stats["attacks"] = copy.deepcopy(fix["attacks"])
        stats["source"], stats["confidence"] = "user", "high"
        stats["notes"] = ["Corrected by a player."]
        stats.pop("client_ocr", None)                           # nothing left to read in the browser
        return stats

    def submit_correction(self, result_id, client, pokemon=None, hp=None, attacks=None):
        """The Detect page's "Teach the agent" form: a player says what the card REALLY shows.
          pokemon  name or Pokédex number if the detected Pokémon was wrong (like a thumbs-down with a correction)
          hp       the number printed top right        attacks  [{name, damage, type}] as printed under the picture
        Learned immediately: the card is remembered (feedback.save_stats_fix + card_cache), so next time its HP and attacks
        come from the player, not from OCR. Returns {ok, stats, result|None}; `result` is a replacement answer when the Pokémon changed."""
        client = str(client or "")[:64] or "anonymous"
        ctx = self.pending.get(result_id) or self.feedback.get_result(result_id)
        if not ctx:
            return {"ok": False, "error": "That result has expired - scan the card again."}
        new = None
        dex = ctx["dex"]
        if pokemon not in (None, ""):
            q = str(pokemon).strip()
            fixed = int(q) if q.isdigit() else self.match_species(q)
            if not fixed or fixed < 1:
                return {"ok": False, "error": f"“{q}” isn't a Pokémon I know."}
            if fixed != ctx["dex"]:
                out = self.submit_feedback(result_id, client, "detection", -1, correct=str(fixed), comment="corrected in the Teach form")
                if not out.get("ok"):
                    return out
                new, dex = out.get("result"), fixed
        types = ((new or {}).get("pokemon") or {}).get("types") or []
        if not types:
            try:
                types = (self.lookup(dex).get("pokemon") or {}).get("types") or []
            except Exception:
                types = []
        hp_val = self.clean_hp(hp) if hp not in (None, "") else None
        if hp not in (None, "") and hp_val is None:
            return {"ok": False, "error": "HP must be a number between 10 and 500."}
        atk = self.clean_attacks(attacks, types) if attacks is not None else None
        if attacks and not atk:
            return {"ok": False, "error": "Each attack needs a name."}
        if hp_val is None and not atk and new is None:
            return {"ok": False, "error": "Nothing to save - change the Pokémon, the HP or an attack."}
        stats = None
        if hp_val is not None or atk:
            self.feedback.save_result(result_id, ctx)
            self.feedback.save_stats_fix(result_id, client, ctx, dex, hp_val, atk or None)
            stats = self.stats_from_fix({"hp": hp_val, "attacks": atk}, None, types)
            if hp_val is None or not atk:                           # partial: keep what the card/OCR gave for the other half
                prev = (self.card_cache_file({"set": ctx["card_set"], "number": ctx["card_number"]}) if ctx.get("card_set") and ctx.get("card_number") else None)
                if prev is not None and prev.exists():
                    try:
                        old = json.loads(prev.read_text())
                        stats = self.stats_from_fix({"hp": hp_val, "attacks": atk}, old, types)
                    except Exception:
                        pass
            if ctx.get("card_set") and ctx.get("card_number"):
                self._save_card_cache(self.card_cache_file({"set": ctx["card_set"], "number": ctx["card_number"]}), stats)
            with self.cache_lock:
                self.scan_cache.clear()                              # cached answers are out of date now
        if new is not None and stats is not None:
            new["stats"] = stats
        return {"ok": True, "stats": stats, "result": new}

    MOVE_NAMES = {"normal": "Tackle", "fire": "Ember", "water": "Water Gun", "electric": "Thunder Shock", "grass": "Vine Whip",
                  "ice": "Ice Shard", "fighting": "Karate Chop", "poison": "Poison Sting", "ground": "Mud Slap", "flying": "Gust",
                  "psychic": "Confusion", "bug": "Bug Bite", "rock": "Rock Throw", "ghost": "Lick", "dragon": "Dragon Rage",
                  "dark": "Bite", "steel": "Metal Claw", "fairy": "Fairy Wind"}

    def opponent(self, exclude=(), rng=random):
        """A random Gen 1 Pokémon for the player to fight (battle mode starts this 5 s after a card is detected).
        Returns the same shape as lookup() plus res["stats"]: HP 60-180 (steps of 10) and two moves made from its types."""
        gen1 = sorted(int(p.stem) for p in (MODELS / GEN1_DIR).glob("[0-9][0-9][0-9].glb")) or list(range(1, 152))
        pool = [p for p in gen1 if p not in set(exclude)] or gen1
        res = None
        for _ in range(5):                                        # PokeAPI hiccup on one Pokémon -> try another
            res = self.lookup(rng.choice(pool))
            if res.get("model_url"):
                break
        if not res or not res.get("model_url"):
            return res
        types = res["pokemon"].get("types") or ["normal"]
        moves = [{"name": "Tackle", "damage": 20, "modifier": None, "type": "normal", "cost": ["normal"], "text": ""}]
        for t, dmg in zip(types[:2], (30, 40)):
            moves.append({"name": self.MOVE_NAMES.get(t, t.title() + " Strike"), "damage": dmg, "modifier": None, "type": t, "cost": [t], "text": ""})
        res["stats"] = {"hp": rng.choice(range(60, 181, 10)), "attacks": moves, "confidence": "high", "source": "generated",
                        "notes": ["Opponent: random Pokémon with made-up stats."]}
        return res

    # ---- manual pick: "pikachu" or "25" ----
    def lookup(self, query):
        """Build the result for a Pokémon chosen by name or Pokédex number (no card needed).
        Used by the "Pick Pokémon" sheet in battle mode when the automatic read is wrong."""
        res, log = self._new_run()
        q = str(query).strip()
        pid = int(q) if q.isdigit() else self.match_species(q)
        if pid is None or pid < 1:
            res["message"] = f"“{q}” isn't a Pokémon I know."
            log("Lookup", "No species matched"); return res
        log("Lookup", f"Picked by hand: {q}")
        res = self._finish({"name": q if not q.isdigit() else "", "set": "", "number": ""}, res, log, pid=pid, show_card=False)
        if res.get("pokemon"):
            res["stats"] = self.complete_stats(None, res["pokemon"].get("types") or [])      # no card to read: default HP and moves
        return self._register(res, "manual")

    # ---- everything after "which card is it": species -> facts -> 3D model ----
    def _finish(self, card, res, log, pid=None, show_card=True):
        """The decision tree that picks the 3D model. Each `return res` below is a successful exit; the
        checks are ordered from "most specific / best quality" to "last resort"."""
        if show_card:
            res["card"] = card
            log("Recognize", f"{card['name']} · {card['set']} #{card['number']}")

        # --- 2. which Pokémon is it? (Trainer/Energy cards stop here) ---
        if pid is None:
            pid = self.match_species(card["name"])
        if pid is None:
            energy = self.read_energy(card["name"])
            if energy:                                           # Energy cards: no 3D model, but we know their type
                res["card_kind"], res["energy"] = "energy", energy
                res["message"] = f"{card['name']}: an Energy card ({energy['type'] or 'any type'}) - no 3D model."
                log("Identify", f"Energy card · type {energy['type'] or 'any'}{' · special' if energy['special'] else ''}"); return res
            res["card_kind"] = "trainer" if card["name"] else "unknown"
            res["message"] = f"“{card['name']}” doesn't look like a Pokémon card (trainer/energy?), so no model was made."
            log("Identify", "No Pokémon species matched"); return res
        try:
            pk = self.fetch_pokemon(pid)
        except Exception as e:                   # offline? a pack model can still be shown
            log("Identify", f"PokeAPI unavailable ({type(e).__name__}); continuing without details")
            pk = {"id": pid, "slug": norm(card["name"]), "name": card["name"] or f"#{pid}", "types": [],
                  "height_m": 0, "weight_kg": 0, "artwork": None}
        res["pokemon"] = pk
        log("Identify", f"Matched #{pk['id']} {pk['name']} ({', '.join(pk['types'])})")

        # Model kinds people have thumbs-downed too often for this Pokémon are skipped (feedback.demoted_stages).
        skip = self.feedback.demoted_stages(pid)
        if skip:
            log("Ratings", f"Skipping model kind(s) rated poorly by users: {', '.join(sorted(skip))}")

        # --- 2a. custom model for this species: it replaces everything below ---
        custom = CUSTOM_MODELS.get(pid)
        if custom and "custom" not in skip and (Path(__file__).parent / "models" / custom).exists():
            res["model_url"] = f"/models/{custom}"
            res["model_source"] = {"kind": "custom_model", "similarity": None, "detail": f"custom model {custom}"}
            log("Model", f"Using the custom {pk['name']} model")
            return res

        # --- 2a'. Gen 1: the textured models shipped in models/gen1/ - nothing else is used for these Pokémon ---
        g1 = MODELS / GEN1_DIR / f"{pid:03d}.glb"            # {pid:03d} -> 1 becomes "001"
        if pid <= 151 and g1.exists() and "gen1" not in skip:
            res["model_url"] = f"/models/{GEN1_DIR}/{g1.name}"
            res["model_source"] = {"kind": "gen1", "similarity": None, "detail": f"Gen 1 model #{pid:03d}"}
            log("Model", f"Using the Gen 1 model for {pk['name']}")
            return res

        # --- 2b. a ready-made 3D model from the optional model pack (regular.zip) ---
        hit = self.models.get(pid, card["name"]) if "glb" not in skip else None
        if hit:
            path, info = hit
            res["model_url"] = f"/models/glb/{path.name}"
            res["model_source"] = {"kind": "glb", "similarity": None,
                                   "detail": f"{info['member']} from {info['pack']}"}
            size = (f"{info['raw_kb']} KB → {info['slim_kb']} KB after removing unused animations"
                    if "raw_kb" in info else f"{info['slim_kb']} KB, cached")
            log("Model", f"Using pack model ({size})")
            return res
        log("Model", "No pack model for this Pokémon" if self.models.count
            else "No model pack found (put regular.zip in model_packs/)" )
        log("Model", "Building a model from the official artwork instead")

        # --- 2c. everything else: a smooth textured relief built from the official artwork (much closer to the Gen 1
        #         models than blocky voxels). If the artwork can't be fetched we fall through to the voxel path below. ---
        try:
            art = self._download_artwork(pk) if "relief" not in skip else None
            if art is not None:
                rel = MODELS / "relief" / f"{pk['id']}.glb"      # cached on disk: built once per Pokémon, ever
                if not rel.exists():
                    import relief                                # imported here so a broken scipy can't stop the server starting
                    relief.build_relief(art, rel)
                res["model_url"] = f"/models/relief/{rel.name}"
                res["model_source"] = {"kind": "relief", "similarity": None, "detail": "inflated official artwork"}
                res["sprite_png"] = self._preview(art)
                log("Model", f"Built a textured 3D relief from the official artwork ({rel.stat().st_size // 1024} KB)")
                return res
        except Exception as e:                                   # never let a model failure end the request: fall back
            log("Model", f"Relief model failed ({type(e).__name__}: {e}); falling back to voxels")

        # --- 3. last resort: pick + verify a sprite (falling back to the artwork itself) ---
        artwork = self._download_artwork(pk)
        img, kind, detail, sim = self.choose_sprite(pk, artwork, log)
        if img is None:
            res["message"] = "No sprite or artwork available for this Pokémon."; return res
        if kind == "artwork":
            log("Sprite", "Using official artwork as the assumed sprite")
        res["model_source"] = {"kind": kind, "detail": detail, "similarity": round(sim, 2)}
        res["sprite_png"] = self._preview(img)

        # --- 4. build (or reuse) the 3D voxel model ---
        out, cached = self.build_model(img, kind, pk["id"])
        res["model_url"] = f"/models/{out.name}"
        log("Model", f"{'Loaded cached' if cached else 'Generated'} 3D voxel model from {kind} ({out.stat().st_size // 1024} KB GLB)")
        return res
