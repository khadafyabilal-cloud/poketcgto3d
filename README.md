# PokéCard 3D

Scan or upload a Pokémon card → the agent identifies it → a **3D model** appears. In **Detect** (`/detect`) you point a phone or desktop camera at cards lying on a table; the app finds each card by its **yellow border** and stands the Pokémon's 3D model on top of it, in augmented reality.

This README has three parts:

1. [Pre-setup](#1-pre-setup) – everything to install or prepare before the first run.
2. [How the code works](#2-how-the-code-works) – a guided tour of every file and function, in the order data flows through them.
3. [How each sprite and model is generated, and how to build a database for them](#3-how-each-sprite-and-model-is-generated--building-a-database) – exact steps, formats and a ready-to-use schema.

For which computer (phone or server) runs what, and how much data crosses the internet, see **`hardware.txt`**.

Every source file also starts with a "how it works" explanation and has step-by-step comments, so you can learn by reading the code itself. A good reading order is: `agent.py` → `card_detect.py` → `static/js/card-geometry.js` → `static/js/card-tracker.js` → `static/js/detect.js` → `static/js/battle-engine.js` → `card_reader.py` → `app.py` → `relief.py` → `scripts/fbx_to_glb.py`.

---

## 1. Pre-setup

### 1.1 Required
| Need | Why | How |
|---|---|---|
| **Python ≥ 3.11.4** | the server and the model builders | python.org |
| **Python packages** | see `requirements.txt` | `python -m venv .venv`, activate it, then `pip install -r requirements.txt` |
| **Internet on first use** | PokeAPI (Pokémon facts + official artwork) and the browser libraries from CDNs | none; results are cached on disk afterwards |
| **A first start with patience** | the card recognizer loads its neural network and card database (slow once, then it stays loaded) | `python app.py` |

`requirements.txt` includes: `pokemon-card-recognizer` (the neural network that reads cards; GPL-3.0) with `bidict==0.22.1` (newer versions cannot load its pickle), `flask`, `requests`, `numpy`, `scipy`, `pillow`, `trimesh`, `opencv-python-headless` (card detection), `cryptography` (HTTPS certificate for phones) and optional `flask-compress`.

### 1.1b Tesseract OCR (needed to read HP and attacks off cards)
`card_reader.py` calls the free **Tesseract** program (it is *not* a pip package; the app just runs the `tesseract` command).
Install it once: **Windows** `winget install UB-Mannheim.TesseractOCR` (then add its folder to PATH) · **macOS** `brew install tesseract` · **Ubuntu/Debian** `sudo apt install tesseract-ocr`. Check with `tesseract --version`, or `python card_reader.py --check [straightened-card.png]` (also prints what it reads from a picture). If it is installed but not on PATH, set `TESSERACT_CMD` to the full path of `tesseract`/`tesseract.exe` (the Windows installer's default folder is tried automatically). `/api/health` and the start-up log show the detected version. Source and docs: <https://github.com/tesseract-ocr/tesseract>.
Without it everything still runs, but the Battle page uses default stats (100 HP, "Tackle" 20 + a type move) and says so on screen.

### 1.2 Optional but recommended
| Item | What it gives you | How |
|---|---|---|
| `model_packs/regular.zip` | real 3D models for Pokémon **beyond Gen 1** (leave the zip closed) | copy the file into `model_packs/` |
| Self-hosted browser libraries | works without internet on the phone, faster loads (OpenCV.js alone is ~10 MB) | `python scripts/fetch_vendor.py` (files go to `static/vendor/`, cached for a year) |
| Your own sprites | override the sprite used by the **voxel fallback** | put `25.jpg`, `pikachu.png`… in `sprites/` (see §3.3) |
| A GPU | faster recognition | an NVIDIA GPU is used automatically |

### 1.3 The 3D models that ship with the project
These are already in the folder; you only rebuild them if you change the source files.
| Path | What | Made by |
|---|---|---|
| `models/gen1/001.glb … 151.glb` | the textured Gen 1 models (the only models used for #1–151) | `python scripts/fbx_to_glb.py <Pokemon-Gen1 folder> models/gen1` |
| `models/custom/charizard.glb` | your retro Charizard STL, slimmed and coloured | `python scripts/stl_to_glb.py in.stl models/custom/charizard.glb --faces 90000` |
| `models/relief/`, `models/glb/`, `models/*.glb` | caches created at run time (safe to delete) | the app |

### 1.4 Running it
```bash
python app.py                           # this computer only:  http://127.0.0.1:5000   (Detect: /detect · Battle: /battle)
python app.py --host 0.0.0.0 --https    # phones on the same Wi-Fi: open the https://<ip>:5000/detect (or /battle) address it prints
```
* **Why HTTPS for phones?** Browsers only allow camera access on `localhost` or HTTPS. `--https` creates a self-signed certificate (`tls.py`); the phone shows a warning once – choose *Advanced → Continue*. A tunnel (cloudflared / ngrok) is an alternative.
* **Firewall:** allow Python on port 5000; the phone and laptop must be on the same network (no "guest/client isolation").
* **Tests:** `python -m unittest discover -s tests`, `node --test tests/test_geometry.mjs`, `node --test tests/test_battle_engine.mjs`, and (needs `pip install playwright` + Chromium) `python tests/browser_game.py`.

### 1.5 Folder layout
```
app.py                    Flask web server and API routes
agent.py                  the agent: recognise → identify → choose a 3D model
feedback.py               user ratings: SQLite store, learned cards, model scores, training export
card_detect.py            OpenCV (Python): find the yellow border, straighten the card
card_reader.py            reads HP, attack names, damage numbers and energy circles off a straightened card (Tesseract OCR + OpenCV)
relief.py                 build a smooth textured 3D "relief" from a flat picture
glb_library.py            reads the optional model pack, slims and caches its models
sprites.py                finds / cuts / compares 2D sprites (custom folder, sprite chart)
vendor.py, tls.py         browser-library URLs (CDN or local) and the HTTPS certificate
export_sheet_sprites.py   optional: export the sprite chart as JPEGs into sprites/
scripts/                  fbx_to_glb.py, stl_to_glb.py, fetch_vendor.py, preview_glb.py, export_feedback.py
static/index.html         scan page (upload/camera → spinning 3D viewer)
static/detect.html        camera page, served twice: /detect (AR viewer) and /battle (the game)
static/js/                card-geometry.js, card-tracker.js, tracker.worker.js, detect.js, battle-engine.js, rating.js
models/                   gen1/, custom/, relief/, glb/ (+ voxel .glb caches)
model_packs/              put regular.zip here (not included)
sprites/                  YOUR sprite images
sprite_sheets/            the FireRed/LeafGreen sprite chart
pokemon_cache/            cached PokeAPI answers
card_cache/               created on first read: one JSON per card with its HP + attacks (never read twice)
feedback/                 created on first vote: feedback.db (SQLite) + crops/ (rated card pictures)
tests/                    Python + Node tests (synthetic cards with known poses); tests/browser_game.py = headless-Chromium test of the game
```

---

## 2. How the code works

### 2.1 The user journeys
```
SCAN PAGE   photo ─► POST /api/recognize ─────────────────────────► agent.run(image)  ─┐
DETECT camera ─► (browser) OpenCV.js tracker finds the card's OUTER border
            ─► (browser) snapshot.js cuts + straightens the card ─► POST /api/battle/scan (snapshot=1, ~40-70 KB)
                                                                              ─► agent.run_card(crop) ─┤
            AR off: photo / upload ─► POST /api/battle/scan (server finds the border itself) ─────────┤
            Teach form ─► POST /api/feedback/correct ─► agent.submit_correction (stats_fix + feedback) 
                                                                                                       ▼
                                                                  agent._finish: card → species → facts → which 3D model
                                                                                                       │
browser: three.js loads  model_url  and draws it (viewer in index.html, or on the card in detect.js) ◄───┘
```

### 2.2 `agent.py` – the agent (read this first)
A class `PokemonCardAgent` with small "tools" (methods) chained by `_finish()`. Each step is logged into a trace the web page shows.

| Function | What it does |
|---|---|
| `recognize(path)` | runs the neural network on an image file → `{set, name, number}` (locked: one scan at a time) |
| `_load_species()` / `match_species(name)` | PokeAPI species table (downloaded once to `species.json`). Slides a 1–3 word window over the card name so "Brock's Onix", "Mr. Mime", "Pikachu VMAX" all resolve to a Pokédex number |
| `fetch_pokemon(pid)` | types, height, weight, artwork URL; cached in `pokemon_cache/<id>.json` |
| `run(bytes)` | scan-page entry: shrink photo → recognize → `_finish` |
| `run_card(crop)` | battle entry for an already-straightened card: cache lookup → recognize → retry upside-down → `_finish` → cache store; adds `rotated` and `cached` |
| `_signature(crop)` / `_cache_lookup` / `_cache_store` | 24×32 normalised-colour fingerprint of the card face + an LRU memory (128 cards) so the slow network runs once per card |
| `lookup(query)` | "Pick Pokémon" by name or number, no card needed |
| `_finish(...)` | **the model decision tree** (below) |
| `choose_sprite`, `build_model` | last-resort voxel path (sprite → cubes) |

**Which model does a card get?** (first rule that matches wins, in `_finish`)
1. `CUSTOM_MODELS` – a hand-picked model for one species (Charizard → your STL).
2. `models/gen1/NNN.glb` – the textured Gen 1 model for Pokédex #1–151. **Nothing else is used for these.**
3. The optional model pack (`regular.zip`) via `glb_library.py`.
4. **Relief** – `relief.py` inflates the official artwork into a smooth textured 3D figure.
5. Voxel cubes from a sprite (only if the artwork cannot be downloaded).

### 2.3 `card_detect.py` – finding the card (Python OpenCV)

**Yellow and gray borders.** The detector looks for two border colours, each with its own mask (`border_mask`): yellow (hue window) and gray/silver (`gray_thresholds`: low saturation, mid brightness; bright whites and dark shadows excluded). A gray candidate must also pass an *interior test* – the middle of the card may not itself be gray (`interior_fraction`, limit `max_gray_interior`) – which rejects solid gray objects such as laptops, phones and mouse pads. A card found in both colours is reported once (`dedupe`), and `Detection.color` says which border matched. The browser tracker (`card-geometry.js` / `card-tracker.js`) does the same with identical constants; `python card_detect.py photo.jpg --border gray|yellow|both` tries it on a photo. Limits: a gray card on a table of almost the same gray cannot be separated from it (use a lighter or darker surface), and gray borders are tested on synthetic cards only.
Idea: the yellow border is rare on a desk, so "find the yellow rectangle" is a cheap, robust detector.
`yellow_mask` (BGR→HSV→`inRange`→close) → `raw_candidates` (contours → convex hull → `approxPolyDP` to 4 corners) → `detect_cards` (tests: fill ≥ 0.9, card-shaped via `pose_residual`, ≥ 55 % of the edge yellow) → `refine_quad` (least-squares line per side, intersect lines = sub-pixel corners) → `orient_quad` (name TL/TR/BR/BL). `refine_with_hint` re-measures a browser-found card at full resolution; `rectify` warps it upright (perspective transform); `overlay_sprite`/`annotate` are visual aids. CLI: `python card_detect.py photo.jpg -o out.jpg --crop card.jpg`.

**Second pass: repairing weak borders (added after real-card tests).** Some real cards defeat the plain colour masks: silver borders in shadow or on tan wood, all-yellow cards whose whole face is yellow, gold/metal energy cards. So `detect_cards` runs a second pass (`repair_masks`): Canny edges find the outline, the thin strip just *inside* each side is classified as yellow or silver by its median colour, and the strips are painted into that colour's mask so the normal tests apply. `merge_passes` then keeps the OUTER answer (a repaired detection replaces a plain one only if it is 1.15x bigger and surrounds it) and drops repaired duplicates. `max_gray_interior` is 0.8 so metal energy cards (grey body) pass while solid slabs (~0.9+) are still rejected. The browser runs the same pass (`CG.mergeRepaired`, every ~400 ms or whenever nothing is found). Tests: `tests/test_real_cards.py` puts the real pictures in `tests/real_cards/` (Team Rocket's Ampharos, Ampharos 049/198, Pinsir, Ninetales, 8 energy types, gold Psychic energy) on simulated tables with perspective, shadow, glare, JPEG and white-balance changes. **Limits:** success rates (roughly 65-90 % per frame on those sweeps, metal energy lowest) come from simulated scenes, not a phone; a white desk and silver-on-tan-wood remain hard; repaired quads are a little less exact than plain ones; the OpenCV.js repair path cannot be run in the build environment. AR sprite sizes: `BATTLE` in `detect.js` (slot 0.3, HP bar width 0.4 card widths).

### 2.4 The browser side (`static/js/`)
| File | Role |
|---|---|
| `card-geometry.js` | pure maths: homography from 4 corners → card pose (rotation + position) → three.js matrices; One Euro smoothing; track association. Same numbers as `card_detect.py` (verified by tests) |
| `card-tracker.js` | `CvDetector` (OpenCV.js: the same pipeline as Python), `TrackerCore` (identity over time, smoothing, pose), `CardTracker` (both) |
| `tracker.worker.js` | runs the tracker in a Web Worker so video and 3D never stutter |
| `detect.js` | camera (AR.js, fallback `getUserMedia`), worker I/O, three.js scene, server calls, UI |

Detect flow: video → small canvas → worker (OpenCV.js) → corners + pose per card → three.js draws the model with a camera whose projection matches the pose maths → when a card has been stable for ~4 detections the full frame + corners go to `/api/battle/scan` → model URL returns → model fades in on the card. Two cards: each Pokémon turns to face the other. Tap a label to pick a Pokémon by hand.

### 2.5 Other files
* `app.py` – routes `/`, `/detect`, `/battle`, `/models/<f>`, `/api/health`, `/api/recognize`, `/api/battle/scan`, `/api/battle/opponent`, `/api/pokemon`, `/api/feedback` (POST a vote), `/api/feedback/stats`; validates corners; injects the import map and `window.PC3D` into the pages.
* `vendor.py` / `scripts/fetch_vendor.py` – local-or-CDN URLs for three.js, OpenCV.js, AR.js.
* `glb_library.py` – indexes the model pack (names only), slims models (keeps one idle animation), caches them. Starts with a glTF/GLB primer.
* `sprites.py` – custom/sheet sprite lookup, background removal, palette similarity.

### 2.6 Ratings: teaching the app with 👍 / 👎 (`feedback.py`, `static/js/rating.js`)

Every answer that has a 3D model carries a `result_id`. Under the result (scan page) or in the **Rate** sheet (Detect/Battle, for the selected card) there are two questions:

| Question | 👍 does | 👎 does |
|---|---|---|
| **Is this the right Pokémon?** (detection) | Stores the card's picture fingerprint as *confirmed*; the next time that card is shown the recogniser is skipped. | Opens an optional box: type what it really is (`pikachu` or `25`). With a correction the model switches at once and the card is remembered as *corrected*. Without one, that guess is remembered as *rejected* and the card is scanned again, skipping it. |
| **Is the 3D model good?** (model) | +1 for this Pokémon's kind of model (`custom`, `gen1`, `glb`, `relief`, `voxel`). | −1. When a kind reaches a score of −3 (ups minus downs, each person counted once) it is skipped for that Pokémon and the next kind in the chain is used; the new model is shown immediately. Up-votes lift the score again. Set `PC3D_DEMOTE_SCORE=-1` to demote on a single thumbs-down. |

**What this is and is not.** The card-recognition neural network is *not* re-trained while the app runs (that needs a GPU and thousands of examples). The ratings change behaviour through (1) the learned-card memory, (2) model demotion and (3) a labelled dataset: `PYTHONPATH=. python scripts/export_feedback.py` writes `feedback_export/labels.jsonl` plus `images/*.jpg` (straightened cards) which you can use to fine-tune a classifier offline and swap in later.

**What is stored** (only after somebody votes; unrated scans leave nothing behind), in `feedback/feedback.db`:
* `result` – id, time, source (`battle`/`scan`/`manual`), Pokédex number shown, card set/number/name, model kind and URL, a 24×32 colour fingerprint, and the file name of the straightened card photo in `feedback/crops/`. The whole camera frame is never stored.
* `vote` – result, anonymous client id, `detection`/`model`, ±1, optional correction and comment. One row per person per question; changing your mind replaces it.
* `learned` – the memory the recogniser consults: `confirmed`, `corrected` or `rejected` + fingerprint.

The *client id* is a random string kept in the browser's localStorage so one person counts once. It is not tied to a name or account. There is no login, so a determined person could vote from several browsers; use `PC3D_DEMOTE_SCORE` and the export's vote counts accordingly, and put the app behind your own authentication if you open it to the public.

---

### 2.7 Detect, Battle, card reading and energy cards

* **Home page** (`/`, `static/home.html`) opens first: a Detect panel (Camera with AR, Without AR, Open Detect), a Battle panel (Start battle, Open Battle) and a two-sentence About. The buttons link to `/detect?go=ar`, `/detect?go=viewer`, `/battle?go=ar`; `detect.js` reads `?go=` and starts at once. The original single-photo page is at `/scan`.
* **Detect** (`/detect`) Point the camera at a card with a yellow or gray border and its 3D model stands on it. The **AR switch** in the top bar turns the camera + tracker off/on; with AR off the model is shown in a 3D viewer (drag to rotate) and you can load a **photo** instead. **Details** opens the results panel: Pokémon, HP, attacks (read by OCR), the camera view, the straightened card snapshot, and how the agent decided.
* **Teach (train the agent from the UI).** In the panel, *Correct this* opens a form: Pokémon, HP and up to 4 attacks (name, damage, type). Touching the form **switches AR off automatically** (saves battery/CPU). *Save* POSTs only the changed fields to `/api/feedback/correct` -> `agent.submit_correction`: a corrected HP/attack list is stored in the `stats_fix` table (keyed by set+number, or by the card's fingerprint) and used instead of OCR the next time that card is seen, and the `card_cache` entry is updated (source "user"); a corrected Pokémon goes through the same path as a 👎 (the wrong match is down-weighted, the right one learned). Exports (`/api/feedback/export`) include `corrected_hp` / `corrected_attacks`. Limits: validation keeps ≤4 attacks and HP 10-500.
* **Motion smoothing (two stages).** (1) In the tracker (`card-tracker.js`) every corner coordinate goes through a One Euro filter. (2) On every drawn frame the Pokémon's pose goes through `PoseFilter` (`card-geometry.js`): the further the pose is from the newest measurement, the faster it follows - small errors are camera noise and are followed slowly (time constant 0.22 s), real movement is followed in ~0.035 s. Tests: `tests/test_geometry.mjs` (jitter cut > 70 % on a still card, caught up in ≤ 12 frames after a 0.6-card jump, rotation takes the short way). Constants are first guesses tuned on synthetic data; if a model still shakes raise `tauRest`, if it lags lower `tauMove`.
* **Battle on a phone.** In the battle view OpenCV runs at most ~15x/s (the pose filter fills the gaps), the HP bars are tiny canvas textures updated only on change, damage sprites are removed after 1.1 s, the opponent model is released when the battle resets and nothing is rendered while the tab is hidden.
* **Only the outer border counts.** The detector measures the card's outermost coloured border. If the outer ring is broken (finger, glare) an inner printed frame must not be mistaken for the card: candidates must have clear space outside them (`outer_clear`), a quad wholly inside a bigger one is dropped (`suppress_inner`), and `outer_from_inner` scans outward from an inner frame and rebuilds the outer quad from line fits (needs all 4 sides, else the candidate is rejected). Same rules in `card_detect.py` and `card-geometry.js`. Thresholds are first guesses tuned on synthetic cards.
* **Snapshot first, to save resources.** Once the corners are known the browser warps the card itself (`snapshot.js`) and uploads only that small upright picture; the server skips detection. A 16x22 fingerprint cache (`SigCache`) answers a re-shown card with no request at all. Other savings: ~30 fps render cap, no rendering when nothing changed, idle tracker rate when no card is visible, pixel ratio ≤1.5 in the viewer, OrbitControls loaded only when AR is first turned off, cache headers (versioned scripts immutable, models 7 days + ETag), no thumbnail in snapshot mode. **Whole-app extras:** HTML pages are built once and served with an ETag (an unchanged page is a 304 with no body); the Scan page shrinks photos to 1600 px / JPEG 0.82 before upload (the server shrinks to 1600 anyway), its camera auto-scan skips frames that look like the last one and pauses in a hidden tab, and it frees preview object URLs.
* **Battle** (`/battle`) is the same camera page plus a mock Pokémon battle (`static/js/battle-engine.js` holds the rules, `detect.js` the screen):
  1. Show a Pokémon card. The server identifies it (as before) **and reads its text** (`card_reader.py`): the **HP** in the top-right corner, each **attack name** under the picture, the **number** printed to the right of the attack (that is the damage), and the coloured **energy circles** to its left.
  2. Your card's attacks (name, damage, type) appear as buttons. The card must stay **detected for 10 seconds** (`COUNTDOWN_S` in `detect.js`; `?countdown=3` on the URL shortens it for testing; the clock pauses while the card is hidden). Then the page **switches to the battle view** (`body.in-battle`: darker screen edges, name tag and HP pills replaced) and a **random Gen 1 Pokémon** (3D model, 60-180 HP, made-up moves from its types) appears.
  3. **Everything is drawn in AR on the card.** Both Pokémon are shrunk to fit on the detected card (`BATTLE` constants: yours on the near half, the opponent on the far half, facing each other); each one's name + HP bar is a camera-facing sprite above it (canvas texture, redrawn only when the HP changes) and damage numbers are sprites that rise and fade. Only the attack buttons and the one-line message remain normal page controls, because they have to be tappable.
  4. Tap an attack: the opponent loses the printed damage (rounded to a multiple of 5) and **lights up in the attack's type colour**; then the opponent answers with a hit of a **random 5, 10, 15 … 40** (all damage in the game is a multiple of 5; constants `OPPONENT_DAMAGE`/`DAMAGE_STEP` in `battle-engine.js`). An attack with **no number** shows "non-damaging move" and does nothing. First to 0 HP loses; "Battle again" starts a new one.
  The battle HUD (two HP boxes, one message line, four attack buttons) is deliberately compact: together it covers at most 1/6 of the screen (about 11-16 % measured; `tests/browser_game.py` checks four screen sizes). CSS is in `static/detect.html` (`.fighter`, `#bMsg`, `#moves`, `.mv`).
* **Move type** is not printed on cards, so it is *assumed*: the first coloured energy in the attack's cost, else the Pokémon's own primary type, else normal.
* **How the text is found.** Tesseract cannot recognise a font by name, so the "Gill Sans" attack names are found by *structure*: lines printed bigger than the small rule text, ending in a number in the right-hand column. This is a heuristic tuned on a synthetic card layout; real cards (old base-set, full-art, abilities) may be misread, so the page lists what it had to guess, and wrong numbers are used as read.
* **Caching, so a card is never loaded twice.** (1) server `scan_cache`: same card (either way up) → recogniser skipped; (2) `card_cache/<set>_<number>.json`: HP + attacks stored on disk, survives restarts; (3) the browser keeps loaded 3D models and, during a battle, re-attaches the running battle (HP, opponent) to the same card the moment it re-appears – the server re-scan only confirms it.
* **Energy cards** (Fire Energy, Basic Lightning Energy, Double Colorless Energy, …) are recognised by the word *Energy* in the card name. The scan result contains `card_kind: "energy"` and `energy: {type, special, name}` (types use PokeAPI names: lightning→electric, darkness→dark, metal→steel, colorless→normal; Rainbow-style cards have `type: null`). They get a label but no 3D model. Nothing consumes them yet: this is the hook for a future type-based healing mechanic. Trainer cards get `card_kind: "trainer"`.
* **Two OCR engines, same parser.** (1) Normally the *server* runs the Tesseract program (`card_reader.read_card`). (2) If the server has no Tesseract, the *browser* does it with **tesseract.js** (<https://github.com/naptha/tesseract.js>, Tesseract compiled to WebAssembly; `static/js/client-ocr.js`): the scan result carries `stats.client_ocr = <token>`; the page downloads the two prepared pictures (`/api/battle/ocr/<token>/hp.png`, `attack.png`), recognises them on the phone's CPU in a Web Worker, and POSTs the words to `/api/battle/ocr-words`; the server validates them, parses them with the same `parse_attack_lines`, caches the card on disk and returns the stats. The Battle countdown waits for this. The first use downloads about 10 MB (tesseract.js worker, WASM core and English data, from jsDelivr / the tessdata CDN) so the device needs internet that once; it is loaded from the CDN, not self-hosted. Browser-side OCR is slower (seconds on a phone) than the server program.
* **AR stability (the model must not vanish).** Three layers in `detect.js` (constants in `STABLE`): (1) the tracker is told to be patient (`lostAfterMs`/`forgetAfterMs`: Battle 1.2 s / 12 s, Detect 0.7 s / 5 s; library defaults were 0.25 s / 1.5 s) and a card unseen for >0.4 s may reappear anywhere in the frame and still be the same track (`reacquireAfterMs/Frac` in `card-tracker.js`); (2) the model is *held on screen* for 3 s (Battle) after the card stops being seen, and a vanished card's entry (model, stats, battle) is kept 20 s - if the tracker gives a returning card a new id, `adoptLost()` hands the old entry to the nearest new track and the next scan only re-confirms it (a failed re-check never removes a known card); (3) the 10 s countdown *pauses* while the card is hidden instead of restarting, and the game only resets after 15 s of absence. Tune `STABLE` if the model lingers too long after you remove the card.
* **Opponent API:** `GET /api/battle/opponent?exclude=4,6` → same JSON as `/api/pokemon` plus `stats {hp, attacks[]}`.
* **Stats JSON** (`res["stats"]`): `{hp, attacks:[{name, damage|null, modifier "+"|"×"|null, type, cost[], text}], confidence, source: "ocr"|"default"|"generated", notes[]}`. For the database in section 3, add a `card_stats` table (card_set, card_number, hp) and `attack` table (card_set, card_number, position, name, damage, modifier, type, cost) from `card_cache/*.json`.

---

## 3. How each sprite and model is generated – building a database

The app currently keeps its data in **files and memory** (images in `sprites/`, models in `models/`, Pokémon facts in `pokemon_cache/*.json`, recently identified cards in RAM). This section documents exactly how each artefact is produced, then proposes a database that stores them.

### 3.1 The five ways a 3D model is produced
| `model_source.kind` | When | Steps | Output |
|---|---|---|---|
| `custom_model` | species listed in `agent.CUSTOM_MODELS` | `scripts/stl_to_glb.py`: read binary STL → weld vertices → **vertex-clustering decimation** (grid cells merged, bisection on cell size to hit `--faces`) → rotate Z-up → Y-up, face +Z → centre, stand on y=0, 1 unit tall → smooth normals → colour by position (belly cream, wings teal, rest orange) → write GLB | `models/custom/*.glb` |
| `gen1` | Pokédex #1–151 | `scripts/fbx_to_glb.py`: parse binary FBX (or OBJ) → bake the node hierarchy's transforms into the vertices → triangulate polygons → read per-polygon materials and find each material's texture by file name → **turn the model to face +Z using the eye geometry** → weld vertices → embed textures (≤512 px, JPEG or PNG-with-alpha) → write GLB | `models/gen1/NNN.glb` |
| `glb` | Pokémon in the optional pack | `glb_library.get`: find `NNN[-variant].glb` in the zip → keep one idle animation (byte-for-byte copy of meshes/textures) → cache | `models/glb/*.glb` |
| `relief` | everything else with downloadable artwork | `relief.build_relief`: trim artwork → bleed colour into transparent pixels → alpha → coarse mask → **distance transform** → circular "pillow" thickness → front surface (+z) and back surface (−z) meet at the outline → normals from the slope → UV = the artwork itself (back mirrored) → GLB, alpha-cutout material | `models/relief/<dex>.glb` |
| `custom` / `sheet` / `artwork` | last resort: the artwork cannot be fetched | `agent.build_model`: sprite → 3D grid of cubes with thickness = √(distance to outline) → keep only visible cube faces → vertex colours (sRGB→linear) → trimesh export | `models/<dex>_<kind>.glb` |

All GLBs share a convention: **Y-up, front facing +Z, standing on y = 0, roughly 1 unit tall** (the relief and Gen 1 models are normalised that way; battle mode then scales to fit the card).

### 3.2 Where a *2D sprite* comes from
`agent.choose_sprite` tries, in order (all end as a trimmed RGBA image):
1. **custom** – `sprites/` file named `25.jpg`, `025_pikachu.png`, `pikachu_front.webp`… (`sprites.find_custom_sprite`). Real transparency is kept; otherwise the flat background (colour of the top-left pixel) is flood-filled from the border to transparent (`remove_background`, tolerance 30 for JPEG noise).
2. **sheet** – `sprites.sheet_sprite(dex)`: pure arithmetic on the FireRed/LeafGreen chart. Each Pokémon owns a 2×2 block of 64×64 cells; position = `X0 + BLOCK_W*col`, `Y0 + BLOCK_H*row` with `row, col = divmod(dex-1, 15)` (#1–150; Mew, Teddiursa, Deoxys are special-cased); crop the front-normal cell, remove the background.
3. **artwork** – the PokeAPI official artwork, trimmed (`crop_to_content`).
A candidate must pass `sprites.similarity(sprite, artwork) ≥ MIN_MATCH` (0.15): a 64-bin RGB histogram intersection, so a wrong picture is rejected.

### 3.3 What the card detector produces (per card, per frame)
Browser tracker output (`TrackerCore.update`), one record per card:
`id`, `corners` (4×[x,y] in video pixels, order TL,TR,BR,BL), `modelView` (16 numbers, column-major 4×4 for three.js), `residual` (0 = perfect 63×88 rectangle), `cover` (0–1 yellow edge fraction), `hits`, `stable`, `visible`, `missedMs`, `ageMs`.
Server (`/api/battle/scan`) returns: `card {set,name,number}`, `pokemon {id,slug,name,types,height_m,weight_kg,artwork}`, `model_url`, `model_source {kind,detail,similarity}`, `rotated` (card was upside down), `cached`, `crop_jpeg` (160-px straightened card as a data URI), `scan {refined, crop_size}`, `trace[]` (timed steps) and `message` on failure.

### 3.4 Proposed database (SQLite shown; works the same in PostgreSQL)
Store **small binary data as BLOBs** (sprites, signatures) and **big files on disk** (models), keeping only the path and a hash in the database.

```sql
-- One row per Pokémon species (source: PokeAPI via agent.fetch_pokemon)
CREATE TABLE pokemon (
  dex          INTEGER PRIMARY KEY,           -- Pokédex number
  slug         TEXT NOT NULL,                 -- "mr-mime"
  name         TEXT NOT NULL,
  types        TEXT NOT NULL,                 -- JSON array: ["grass","poison"]
  height_m     REAL, weight_kg REAL,
  artwork_url  TEXT
);

-- 2D sprites / artwork (the three sources in section 3.2). One species can have several.
CREATE TABLE sprite (
  id           INTEGER PRIMARY KEY,
  dex          INTEGER NOT NULL REFERENCES pokemon(dex),
  kind         TEXT NOT NULL CHECK (kind IN ('custom','sheet','artwork')),
  detail       TEXT,                          -- file name / "FireRed/LeafGreen chart" / artwork URL
  width        INTEGER, height INTEGER,
  png          BLOB NOT NULL,                 -- trimmed RGBA, PNG-encoded (typically 1-50 KB)
  sha256       TEXT NOT NULL,                 -- de-duplicate identical pictures
  similarity   REAL,                          -- score vs official artwork (NULL for artwork itself)
  created_at   TEXT DEFAULT CURRENT_TIMESTAMP,
  UNIQUE (dex, kind, sha256)
);

-- 3D models (section 3.1). The file stays on disk; the DB knows where and what it is.
CREATE TABLE model (
  id           INTEGER PRIMARY KEY,
  dex          INTEGER NOT NULL REFERENCES pokemon(dex),
  kind         TEXT NOT NULL CHECK (kind IN ('custom_model','gen1','glb','relief','voxel')),
  path         TEXT NOT NULL,                 -- "gen1/025.glb" (relative to models/)
  bytes        INTEGER, triangles INTEGER,
  sha256       TEXT NOT NULL,
  sprite_id    INTEGER REFERENCES sprite(id), -- the picture it was built from (relief/voxel)
  detail       TEXT,                          -- "Pikachu.fbx", "regular.zip:25.glb", "inflated official artwork"
  is_active    INTEGER DEFAULT 1,             -- the one _finish() should serve for this dex
  created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX model_dex ON model(dex, is_active);

-- Cards as identified by the recogniser (the printing, not just the species)
CREATE TABLE card (
  id           INTEGER PRIMARY KEY,
  card_set     TEXT NOT NULL, number TEXT NOT NULL, name TEXT NOT NULL,
  dex          INTEGER REFERENCES pokemon(dex),   -- NULL for Trainer / Energy cards
  UNIQUE (card_set, number)
);

-- Replaces the in-RAM scan_cache of agent.py: "have I seen this card before?"
CREATE TABLE card_signature (
  id           INTEGER PRIMARY KEY,
  card_id      INTEGER NOT NULL REFERENCES card(id),
  signature    BLOB NOT NULL,                 -- agent._signature(): 24x32x3 float32 = 9216 bytes (float16 = 4608)
  created_at   TEXT DEFAULT CURRENT_TIMESTAMP
);

-- Every detection the server handled (analytics, tuning, regression tests)
CREATE TABLE detection (
  id           INTEGER PRIMARY KEY,
  ts           TEXT DEFAULT CURRENT_TIMESTAMP,
  session      TEXT,                          -- anonymous id from the browser, optional
  frame_w      INTEGER, frame_h INTEGER,
  corners      TEXT NOT NULL,                 -- JSON [[x,y]*4] TL,TR,BR,BL as refined by refine_with_hint
  refined      INTEGER,                       -- scan.refined
  sensitivity  REAL, hfov REAL,
  cover        REAL, residual REAL,           -- from the tracker
  card_id      INTEGER REFERENCES card(id),   -- NULL if unreadable
  model_id     INTEGER REFERENCES model(id),
  rotated      INTEGER, cached INTEGER,
  duration_ms  INTEGER
);
```

### 3.5 Filling it from the existing code (sketch)
```python
import hashlib, io, json, sqlite3
from PIL import Image

db = sqlite3.connect("pokecard.db"); db.executescript(open("schema.sql").read())

def png_bytes(img: Image.Image) -> bytes:
    b = io.BytesIO(); img.save(b, "PNG", optimize=True); return b.getvalue()

# sprites: reuse the exact functions the agent uses
import sprites
for dex in range(1, 152):
    img = sprites.sheet_sprite(dex)                  # or sprites.find_custom_sprite(dex, slug)[0]
    if img is not None:
        data = png_bytes(img)
        db.execute("INSERT OR IGNORE INTO sprite(dex,kind,detail,width,height,png,sha256) VALUES (?,?,?,?,?,?,?)",
                   (dex, "sheet", "FireRed/LeafGreen chart", img.width, img.height, data, hashlib.sha256(data).hexdigest()))

# models: scan the folders, one row per .glb
from pathlib import Path
for p in Path("models/gen1").glob("*.glb"):
    dex = int(p.stem)
    db.execute("INSERT INTO model(dex,kind,path,bytes,sha256) VALUES (?,?,?,?,?)",
               (dex, "gen1", f"gen1/{p.name}", p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))
db.commit()
```
To make the agent use the database, replace three places in `agent.py`: (1) `_finish` step 2a/2a′ → `SELECT path FROM model WHERE dex=? AND is_active=1 ORDER BY kind`; (2) `_cache_lookup` / `_cache_store` → read/write `card_signature` (compare with the same mean-absolute-difference, threshold `SCAN_CACHE_DIST = 0.28`); (3) `app.battle_scan` → insert a `detection` row from the response.

Notes for a database design: a card **signature** compares by *distance*, not equality, so keep them in a small table scanned in Python (hundreds of rows) or use a vector index (e.g. pgvector) at scale; keep model files on disk or object storage and serve them with the existing `/models/<path>` route; store the `sha256` so rebuilt models can be detected as changed.

---

## 4. Tuning, troubleshooting, status

**Ratings tuning.** `PC3D_DEMOTE_SCORE` (default −3) in the environment; `MODEL_DEMOTE_SCORE` in `feedback.py`; learned-card matching reuses `SCAN_CACHE_DIST`. Delete `feedback/` to forget everything learned.

**Tuning.** `MIN_MATCH`, `ARTWORK_GRID`, `SPRITE_MAX`, `SCAN_CACHE_DIST`, `CARD_PAD` in `agent.py`; `PARAMS`/`thresholds()` in `card_detect.py` **and** `card-geometry.js` (must match); `depth`/`grid` in `relief.build_relief`; *Settings* on the camera pages (border sensitivity, field of view); `max_gray_interior` and `dup_dist` for gray borders.

**Troubleshooting.**
- `ModuleNotFoundError: cryptography` → `pip install cryptography` (needed for `--https`). `cv2` missing → `pip install opencv-python-headless`.
- Border not found → raise *Border sensitivity* in Settings; *Debug* shows the mask. Cards that touch merge into one; leave a gap.
- Model looks flat/stretched → adjust *Field of view* (assumed 63°). Upside down → *Flip card*. Wrong Pokémon → *Pick Pokémon*.
- Page stuck on "Loading OpenCV…" → the phone cannot reach the CDN; run `python scripts/fetch_vendor.py` on the laptop.
- `BidictBase._from_other() …` → `pip install bidict==0.22.1`. `PermissionError … Temp\tmp*.png` on Windows → patched at the top of `agent.py`.
- "No model pack found" in the log is normal without `regular.zip`; relief models are used instead.
- Slow scans on CPU are normal.

**Verification status.** Tested here: border detection and pose maths (synthetic frames with known poses; JS and Python agree), tracking, server API, model converters (rendered previews), relief geometry, and the page error path. **Not** tested in the build environment (no CDN access, no camera or phone): the OpenCV.js wrapper, AR.js, three.js rendering, the real recognizer, and relief models built from real artwork. Expect some tuning on first real use.

**Model limits.** Gen 1 models are exported in their rest pose (no animation; some T-poses). About 15 Pokémon (Lickitung, Koffing, Weezing, Rhyhorn, Chansey, Tangela, Horsea, Seadra, Seaking, Jynx, Electabuzz, Magmar, Tauros, Kabuto, Kabutops) only ship flat low-detail files in the source folder. Facing was detected from the eyes; ~40 models without an eye part were checked by eye.

The recognizer library is GPL-3.0; Pokémon sprites, artwork and models belong to their respective owners – for personal/educational use.
#   P o k e m o n T C G a r  
 #   P o k e m o n T C G a r  
 