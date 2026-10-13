"""
card_reader.py - reads the GAME TEXT on a straightened Pokémon card: HP, attack names, damage numbers, attack types.
==================================================================================================================

WHY THIS FILE EXISTS
--------------------
The recogniser (agent.py) tells us WHICH card it is (set + number + name). The mock battle also needs what is PRINTED
on the card:

    top-right corner ........ HP number          -> the Pokémon's health points in the battle
    under the picture ....... attack names       -> the moves the player can choose (the printed font is "Gill Sans")
    right of each attack .... damage number      -> exactly how much the move hurts; no number = non-damaging move
    left of each attack ..... coloured energy    -> what the move costs, which also tells us the move's TYPE
                              circles

HOW THE READING WORKS (and what to expect)
------------------------------------------
1. Cut the card into regions by fractions of its size (cards share one layout): `HP_BOX`, `ATTACK_BOX`.
2. Upscale + increase contrast, then run the open-source OCR program Tesseract on the region. Tesseract cannot tell
   a font by name, so "Gill Sans" is not detected as such; the attack lines are recognised by STRUCTURE instead:
   a line whose name is printed bigger than the small rule text, and/or ends with a number on the right.
3. Energy circles are found with OpenCV (HoughCircles) so they are never mistaken for letters, and their colour is
   classified into a type (fire, water, ...).
4. `parse_attack_lines` (pure Python, no image) turns OCR words into attacks. It is separate so it can be tested with
   fake words, and so you can read it without needing Tesseract installed.
5. The result is cached on disk per card (agent.py), so a card is only ever read once.

It is a HEURISTIC reader: if a number is misread the battle uses the misread value, and cards with unusual layouts
(old 1999 cards, full-art, Pokémon-V with abilities) can be read imperfectly. `read_card` therefore returns
`confidence` and `notes`, and the battle UI says plainly when it had to guess. If Tesseract is not installed the
function returns None and the agent falls back to default stats (README, pre-setup).
"""
import re, shutil, subprocess, tempfile, os
from collections import defaultdict
from statistics import median

import cv2
import numpy as np

# ------------------------------------------------------------------ where things are on the card
# (x0, y0, x1, y1) as fractions of the WHOLE card (yellow border included, 0,0 = top-left).
HP_BOX = (0.50, 0.025, 0.96, 0.125)       # "HP 120" sits in the top-right corner
ATTACK_BOX = (0.04, 0.50, 0.96, 0.86)     # from just under the picture to just above weakness/resistance
DAMAGE_X = 0.74                           # a number right of this fraction of the attack box is a damage number
SYMBOL_X = 0.36                           # energy circles live left of this fraction of the attack box

# Types use PokeAPI's names everywhere in this app (the card game says Lightning/Darkness/Metal/Colorless).
TCG_TO_TYPE = {"lightning": "electric", "darkness": "dark", "metal": "steel", "colorless": "normal"}
TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying", "psychic",
         "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"]


_TESS = {"cmd": None, "checked": False, "version": None}


def tesseract_command():
    """Path of the Tesseract program, or None. Looked up once, in this order:
       1. the TESSERACT_CMD environment variable (full path; use it when Tesseract is installed somewhere unusual),
       2. `tesseract` on PATH,
       3. the default install folders (the Windows installer does NOT add itself to PATH, a common first-run problem)."""
    if not _TESS["checked"]:
        _TESS["checked"] = True
        cands = [os.environ.get("TESSERACT_CMD"), shutil.which("tesseract"),
                 r"C:\Program Files\Tesseract-OCR\tesseract.exe", r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                 "/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"]
        for c in cands:
            if c and os.path.isfile(c) and os.access(c, os.X_OK):
                _TESS["cmd"] = c
                break
    return _TESS["cmd"]


def tesseract_available():
    return tesseract_command() is not None


def tesseract_info():
    """{'available', 'cmd', 'version', 'english'} - shown by /api/health and `python card_reader.py --check`.
    'english' = the English language data (eng.traineddata) is installed, which reading needs."""
    cmd = tesseract_command()
    if not cmd:
        return {"available": False, "cmd": None, "version": None, "english": False}
    try:
        v = subprocess.run([cmd, "--version"], capture_output=True, text=True, timeout=10)
        ver = (v.stdout or v.stderr).splitlines()[0].replace("tesseract", "").strip()
        langs = subprocess.run([cmd, "--list-langs"], capture_output=True, text=True, timeout=10).stdout.split()
        return {"available": True, "cmd": cmd, "version": ver, "english": "eng" in langs}
    except Exception as e:
        return {"available": False, "cmd": cmd, "version": f"error: {e}", "english": False}


def to_type(word):
    """'Lightning' / 'fire' / 'Darkness' -> PokeAPI type name, or None."""
    w = (word or "").strip().lower()
    w = TCG_TO_TYPE.get(w, w)
    return w if w in TYPES else None


# ------------------------------------------------------------------ colour -> type
def classify_colour(bgr):
    """Colour of an energy circle -> type name. `bgr` = mean (b, g, r). Uses HSV: hue says the family, saturation and
    brightness separate grey/white (colorless -> normal), black (dark), and steel."""
    px = np.uint8([[[int(c) for c in bgr]]])
    h, s, v = [int(x) for x in cv2.cvtColor(px, cv2.COLOR_BGR2HSV)[0, 0]]       # OpenCV hue is 0-179
    if v < 70:
        return "dark"
    if s < 45:
        return "steel" if v < 170 else "normal"
    if h < 8 or h >= 165:
        return "fire" if s > 150 else "fairy"
    if h < 18:
        return "fighting" if v < 170 else "fire"
    if h < 36:
        return "electric"
    if h < 85:
        return "grass"
    if h < 135:
        return "water"
    if h < 158:
        return "psychic"
    return "fairy"


def energy_type_from_crop(crop, pad=0.06):
    """Fallback for ENERGY cards whose name has no type word: the big symbol sits in the middle of the card; classify
    the colour of a ring around its centre (the glyph itself is white). Returns a type name."""
    h, w = crop.shape[:2]
    c = crop[int(h * 0.30):int(h * 0.55), int(w * 0.30):int(w * 0.70)]
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    keep = (hsv[..., 1] > 60) & (hsv[..., 2] > 70)
    px = c[keep] if keep.sum() > 50 else c.reshape(-1, 3)
    return classify_colour(px.mean(axis=0))


# ------------------------------------------------------------------ OCR plumbing
def _region(crop, box, pad):
    """Cut `box` (fractions of the card) out of the straightened crop, which has `pad` extra margin on each side."""
    h, w = crop.shape[:2]
    f = lambda v: (v + pad) / (1 + 2 * pad)
    x0, y0, x1, y1 = (f(v) for v in box)
    return crop[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)]


def _prepare(img, width=1100):
    """Grey, upscale to a fixed width (Tesseract likes ~30 px tall letters), local contrast boost."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    k = width / max(1, g.shape[1])
    g = cv2.resize(g, None, fx=k, fy=k, interpolation=cv2.INTER_CUBIC if k > 1 else cv2.INTER_AREA)
    return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(g), k


def ocr_words(gray, psm=6, whitelist=None, timeout=15):
    """Run Tesseract on a grey image -> list of words {text, left, top, width, height, conf, line} (pixels of `gray`).
    `line` is a (block, paragraph, line) key; words with the same key are on one text line."""
    fd, path = tempfile.mkstemp(suffix=".png"); os.close(fd)
    try:
        cv2.imwrite(path, gray)
        # --oem 1 = the LSTM neural engine (Tesseract 4+/5, the accurate one); --dpi 300 tells it our upscaled image is
        # print-sized (otherwise it guesses and prints warnings); preserve_interword_spaces keeps the gap before damage numbers.
        cmd = [tesseract_command(), path, "stdout", "--oem", "1", "--psm", str(psm), "--dpi", "300", "-l", "eng",
               "-c", "preserve_interword_spaces=1"]
        if whitelist:
            cmd += ["-c", f"tessedit_char_whitelist={whitelist}"]
        run = subprocess.run(cmd + ["tsv"], capture_output=True, text=True, timeout=timeout)
        if run.returncode != 0:
            raise RuntimeError("tesseract failed: " + (run.stderr or "").strip()[:200])
        out = run.stdout
    finally:
        os.remove(path)
    words = []
    for row in out.splitlines()[1:]:
        c = row.split("\t")
        if len(c) < 12 or c[0] != "5" or not c[11].strip():
            continue
        try:
            conf = float(c[10])
        except ValueError:
            continue
        if conf < 0:
            continue
        words.append({"text": c[11].strip(), "left": int(c[6]), "top": int(c[7]), "width": int(c[8]), "height": int(c[9]),
                      "conf": conf, "line": (int(c[2]), int(c[3]), int(c[4]))})
    return words


# ------------------------------------------------------------------ HP
def parse_hp(words):
    """OCR words from the top-right corner -> HP number or None.
    Prefers a number touching the letters 'HP'; otherwise the biggest plausible multiple of 10 (10-340)."""
    texts = [w["text"] for w in words]
    joined = " ".join(texts)
    m = re.search(r"HP\s*[:\-]?\s*(\d{2,3})", joined, re.I) or re.search(r"(\d{2,3})\s*HP", joined, re.I)
    if m and 10 <= int(m.group(1)) <= 340:
        return int(m.group(1))
    nums = [int(n) for n in re.findall(r"\d{2,3}", joined) if 10 <= int(n) <= 340 and int(n) % 10 == 0]
    return max(nums) if nums else None


def read_hp(crop, pad=0.06):
    region = _region(crop, HP_BOX, pad)
    gray, _ = _prepare(region, 700)
    for psm in (7, 6):                                   # one line first, then a block
        hp = parse_hp(ocr_words(gray, psm=psm))
        if hp:
            return hp
    return None


# ------------------------------------------------------------------ energy circles (attack cost + type)
def find_energy_circles(region, max_x=SYMBOL_X):
    """Coloured/grey circles in the left part of the attack region -> [{x, y, r, type}] in region pixels.
    HoughCircles looks for round outlines; radius is a fraction of the region width (energy symbols are small)."""
    h, w = region.shape[:2]
    left = region[:, :int(w * max_x)]
    g = cv2.medianBlur(cv2.cvtColor(left, cv2.COLOR_BGR2GRAY), 5)
    rmin, rmax = int(w * 0.018), int(w * 0.040)
    found = cv2.HoughCircles(g, cv2.HOUGH_GRADIENT, dp=1.2, minDist=int(rmin * 1.6), param1=90, param2=24,
                             minRadius=rmin, maxRadius=rmax)
    out = []
    if found is None:
        return out
    for x, y, r in np.round(found[0]).astype(int):
        mask = np.zeros(left.shape[:2], np.uint8)
        cv2.circle(mask, (x, y), int(r * 0.85), 255, -1)
        cv2.circle(mask, (x, y), int(r * 0.45), 0, -1)               # ring only: the white glyph in the middle is ignored
        if mask.sum() == 0:
            continue
        mean = cv2.mean(left, mask=mask)[:3]
        out.append({"x": int(x), "y": int(y), "r": int(r), "type": classify_colour(mean)})
    return out


# ------------------------------------------------------------------ attack lines
NUM = re.compile(r"^[\(\[]?(\d{1,3})\s*([+×xX\-–*]?)[\)\]]?$")


def _clean_name(tokens):
    """Join name tokens, dropping OCR debris (lone symbols, stray punctuation) around the edges."""
    toks = [re.sub(r"[^A-Za-z0-9'’.\-é ]", "", t).strip() for t in tokens]
    toks = [t for t in toks if t]
    while toks and len(re.sub(r"[^A-Za-z]", "", toks[0])) < 2 and not toks[0].isdigit():
        toks.pop(0)                                                    # leading lone letters = misread energy symbols
    while toks and len(re.sub(r"[^A-Za-z]", "", toks[-1])) < 1:
        toks.pop()
    return " ".join(toks).strip(" -.")


def parse_attack_lines(words, region_width, circles=()):
    """The structural heart of the reader. `words` come from ocr_words() on the attack region.

    1. Group words into text lines and measure each line's letter height.
    2. A line is an ATTACK HEADER when its letters are noticeably bigger than the median rule-text line
       (attack names are printed large) OR it ends in a damage number in the right-hand column.
       Rule text under an attack is small, sentence-like and has no number column.
    3. Damage = the number in the right-hand column ("30", "20+", "50×"). No number -> damage None (non-damaging).
    4. Energy circles (already found by OpenCV) that share the line's height give the cost; names start right of them.
    Returns [{name, damage, modifier, cost, text}] in reading order."""
    lines = defaultdict(list)
    for w in words:
        lines[w["line"]].append(w)
    rows = []
    for key, ws in lines.items():
        ws.sort(key=lambda w: w["left"])
        rows.append({"words": ws, "top": min(w["top"] for w in ws), "bottom": max(w["top"] + w["height"] for w in ws),
                     "h": median(w["height"] for w in ws)})
    rows.sort(key=lambda r: r["top"])
    # merge fragments that sit at the same height (Tesseract sometimes splits one printed line into two)
    merged = []
    for r in rows:
        if merged and abs((r["top"] + r["bottom"]) / 2 - (merged[-1]["top"] + merged[-1]["bottom"]) / 2) < 0.5 * max(r["h"], merged[-1]["h"]):
            merged[-1]["words"] = sorted(merged[-1]["words"] + r["words"], key=lambda w: w["left"])
            merged[-1]["bottom"] = max(merged[-1]["bottom"], r["bottom"]); merged[-1]["h"] = median(w["height"] for w in merged[-1]["words"])
        else:
            merged.append(r)
    rows = merged
    if not rows:
        return []
    base_h = float(np.percentile([r["h"] for r in rows], 25))        # size of the small rule text (headers are bigger)
    attacks, cur = [], None
    for r in rows:
        ws = r["words"]
        dmg = None
        last = ws[-1]
        m = NUM.match(last["text"])
        if m and last["left"] >= DAMAGE_X * region_width and len(ws) >= 2:
            dmg = (int(m.group(1)), m.group(2) or None)
            name_words = ws[:-1]
        else:
            name_words = ws
        # a bare number alone on a line at the right edge belongs to the previous header (OCR split the row)
        if len(ws) == 1 and dmg is None and NUM.match(ws[0]["text"]) and ws[0]["left"] >= DAMAGE_X * region_width and cur is not None and cur["damage"] is None:
            mm = NUM.match(ws[0]["text"]); cur["damage"] = int(mm.group(1)); cur["modifier"] = mm.group(2) or None
            continue
        # symbols (OpenCV circles) sharing this line push the name start to their right edge
        cy = (r["top"] + r["bottom"]) / 2
        first = next((w["left"] for w in name_words if len(re.sub(r"[^A-Za-z]", "", w["text"])) >= 3), None)   # where the name really starts
        mine = [c for c in circles if abs(c["y"] - cy) < 0.9 * r["h"] + 6 and (first is None or c["x"] + c["r"] <= first + 4)]
        dedup = []
        for c in sorted(mine, key=lambda c: c["x"]):                        # Hough often reports one symbol twice
            if not dedup or c["x"] - dedup[-1]["x"] > 1.3 * c["r"]:
                dedup.append(c)
        mine = dedup
        edge = max((c["x"] + c["r"] for c in mine), default=0)
        name_words = [w for w in name_words if w["left"] + w["width"] / 2 > edge]
        name = _clean_name([w["text"] for w in name_words])
        big = r["h"] >= 1.18 * base_h
        header = bool(name) and (dmg is not None or (big and len(name.split()) <= 4 and name[0].isupper() and not name.endswith(".")))
        if header:
            cur = {"name": name, "damage": dmg[0] if dmg else None, "modifier": dmg[1] if dmg else None,
                   "cost": [c["type"] for c in sorted(mine, key=lambda c: c["x"])], "text": ""}
            attacks.append(cur)
        elif cur is not None and name:
            cur["text"] = (cur["text"] + " " + name).strip()[:200]       # rule text under the attack (kept for display)
    return attacks


def assume_type(cost, pokemon_types):
    """The move's type is not printed, so it is ASSUMED: the first coloured (non-colorless) energy in its cost; else the
    Pokémon's own primary type; else normal."""
    for t in cost:
        if t != "normal":
            return t
    if pokemon_types:
        return pokemon_types[0]
    return "normal"


# ------------------------------------------------------------------ everything together
def prepare_regions(crop, pad=0.06):
    """Cut + enhance the two areas to be read: {"hp": grey image, "attack": grey image, "circles": [...]}.
    Used by BOTH readers: the server's Tesseract program (read_card) and the browser's tesseract.js (client_ocr)."""
    atk = _region(crop, ATTACK_BOX, pad)
    atk_gray, _ = _prepare(atk)
    hp_gray, _ = _prepare(_region(crop, HP_BOX, pad), 700)
    circles = find_energy_circles(cv2.resize(atk, (atk_gray.shape[1], atk_gray.shape[0]), interpolation=cv2.INTER_CUBIC))
    return {"hp": hp_gray, "attack": atk_gray, "circles": circles}


def stats_from_words(hp_word_lists, attack_words, attack_width, circles, pokemon_types=()):
    """OCR words (from EITHER reader) -> the stats dict. `hp_word_lists` = one word list per attempt (first that parses wins)."""
    notes = []
    hp = next((h for h in (parse_hp(w) for w in hp_word_lists) if h), None)
    attacks = parse_attack_lines(attack_words, attack_width, circles)[:4]
    for a in attacks:
        a["type"] = assume_type(a["cost"], list(pokemon_types))
    if hp is None:
        notes.append("Couldn't read the HP number - using 100.")
    if not attacks:
        notes.append("Couldn't read any attacks on this card.")
    conf = "high" if hp and attacks else ("partial" if hp or attacks else "low")
    return {"hp": hp, "attacks": attacks, "confidence": conf, "notes": notes, "source": "ocr"}


def sanitize_words(raw, max_x=4000, max_y=4000, limit=400):
    """Words that came from a BROWSER (tesseract.js) are untrusted input: keep only well-formed, bounded entries.
    Accepts [{text,left,top,width,height,conf,line}] and returns the same shape with clean types."""
    out = []
    if not isinstance(raw, list):
        return out
    for w in raw[:limit]:
        try:
            line = w.get("line", [0, 0, 0])
            out.append({"text": str(w["text"])[:40].strip(), "left": int(max(0, min(max_x, w["left"]))), "top": int(max(0, min(max_y, w["top"]))),
                        "width": int(max(1, min(max_x, w["width"]))), "height": int(max(1, min(max_y, w["height"]))),
                        "conf": float(w.get("conf", 50)), "line": tuple(int(x) for x in line[:3])})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return [w for w in out if w["text"]]


def read_card(crop, pokemon_types=(), pad=0.06):
    """Read HP + attacks from a straightened, upright card image (BGR) with the Tesseract PROGRAM on this computer.
    Returns a dict, or None if Tesseract is missing (the agent then offers the browser-side reader, client_ocr):

        {"hp": 120 | None,
         "attacks": [{"name": "Ember", "damage": 30, "modifier": None, "type": "fire", "cost": ["fire"], "text": "..."}],
         "confidence": "high" | "partial" | "low",    # high = HP and >=1 attack found
         "notes": [str, ...],                         # what had to be guessed, for the UI
         "source": "ocr"}
    """
    if not tesseract_available():
        return None
    r = prepare_regions(crop, pad)
    hp_lists = []
    for psm in (7, 6):                                   # one line first, then a block
        hp_lists.append(ocr_words(r["hp"], psm=psm))
        if parse_hp(hp_lists[-1]):
            break
    return stats_from_words(hp_lists, ocr_words(r["attack"], psm=6), r["attack"].shape[1], r["circles"], pokemon_types)


if __name__ == "__main__":                       # python card_reader.py --check [card.png]
    import json, sys
    info = tesseract_info()
    print("Tesseract:", json.dumps(info))
    if not info["available"]:
        print("Not found. Install it (README 1.1b) or set TESSERACT_CMD to the full path of tesseract(.exe).")
        sys.exit(1)
    if not info["english"]:
        print("English data missing: install eng.traineddata (package tesseract-ocr-eng, or tessdata_fast on GitHub) or set TESSDATA_PREFIX.")
        sys.exit(1)
    if len(sys.argv) > 2:                         # try it on a real photo of a straightened card
        img = cv2.imread(sys.argv[2])
        print(json.dumps(read_card(img, []) if img is not None else "cannot open " + sys.argv[2], indent=1))
