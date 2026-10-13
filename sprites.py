"""
sprites.py - everything about FINDING a 2D sprite for a Pokémon.
================================================================

This module is deliberately light (Pillow + numpy + scipy only) so it can be
tested without the card-recognizer or 3D libraries.

A "sprite" here is a small RGBA picture (RGB colours + an Alpha/transparency channel) of ONE Pokémon with a
transparent background. The 3D builders (agent.build_model, relief.py) turn such a picture into a model, and
agent.choose_sprite() uses the helpers below.

A sprite can come from three places (the agent tries them in this order):
  1. CUSTOM  - a JPEG/PNG you dropped into the  sprites/  folder.
  2. SHEET   - sliced out of the FireRed/LeafGreen chart in  sprite_sheets/.
  3. (fallback, handled in agent.py) the official artwork from PokeAPI.

HOW EACH SPRITE IS PRODUCED (useful if you want to store sprites in a database - see README "Building a database")
  custom : file on disk -> (keep its alpha OR flood-fill the flat background to transparent) -> trim transparent margin
  sheet  : fixed grid arithmetic (dex number -> row/column) -> crop 64x64 cell -> background removal -> trim
  artwork: download PNG from PokeAPI -> trim transparent margin
  All three end as a trimmed RGBA PIL image; that image (PNG bytes) is the natural thing to store.
"""
import re, unicodedata
from functools import lru_cache
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.ndimage import label

ROOT = Path(__file__).parent
SPRITE_DIR = ROOT / "sprites"                        # user-supplied sprites (jpg/png/webp)
SHEET_PATH = ROOT / "sprite_sheets" / "firered_leafgreen.png"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

# ----------------------------------------------------------------------------
# Sprite-sheet geometry (measured from the FireRed/LeafGreen chart by MishuK9).
# Every Pokémon owns a 2x2 block of 64x64 cells separated by 1px black lines:
#     [ front normal ][ front shiny ]
#     [ back normal  ][ back shiny  ]
# and a header strip (name / number) sits above each block.
# ----------------------------------------------------------------------------
X0, Y0 = 11, 46        # top-left pixel of the very first sprite cell
CELL = 64              # sprite cell is 64 x 64 pixels
BLOCK_W = 130          # horizontal distance between Pokémon blocks (2 cells + 2 lines)
BLOCK_H = 165          # vertical distance between block rows (header + 2 cells + lines)
PER_ROW = 15           # Pokémon per row of the chart

# Pokédex #1-150 are laid out row by row. The last row holds a few extras,
# given here as dex number -> (block_row, block_col).
SPECIAL = {151: (10, 0),   # Mew
           216: (10, 1),   # Teddiursa
           386: (10, 2)}   # Deoxys (the sheet shows the Attack form first)


def norm(s: str) -> str:
    """Lower-case and keep only a-z/0-9 so 'Mr. Mime', 'mr-mime', 'MR_MIME' all match.

    The gender symbols are spelled out first (Nidoran♀ -> nidoranf) because they would otherwise be dropped and
    the two Nidorans would collide. unicodedata.normalize("NFKD") splits accented letters (é -> e + accent) so
    the accent can then be thrown away by the regular expression."""
    s = unicodedata.normalize("NFKD", s.replace("♀", "f").replace("♂", "m"))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def remove_background(img: Image.Image, tol: int = 0) -> Image.Image:
    """
    Make the flat background colour transparent.
    The colour is read from the top-left pixel, then flood-filled inwards from the
    border only. Pixels of the same colour *inside* the sprite (e.g. white eyes)
    are therefore kept. `tol` allows slightly different colours, which is needed
    for JPEGs because compression adds noise to flat areas.
    """
    rgb = np.asarray(img.convert("RGB")).astype(int)
    close = np.abs(rgb - rgb[0, 0]).max(axis=2) <= tol       # "looks like background": every channel within tol of the corner colour
    lab, _ = label(close)                                     # connected regions: label every blob of background-coloured pixels 1, 2, 3...
    border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))   # which blob numbers touch the picture's edge?
    bg = np.isin(lab, border[border != 0])                    # regions touching the edge (0 = "not background-coloured")
    out = np.dstack([rgb.astype(np.uint8), np.where(bg, 0, 255).astype(np.uint8)])
    return crop_to_content(Image.fromarray(out, "RGBA"))


def crop_to_content(img: Image.Image) -> Image.Image:
    """Trim fully transparent borders so the model is centred on the sprite."""
    # threshold the alpha channel (>40 = "visible"), then getbbox() gives the smallest rectangle containing every visible pixel
    box = img.getchannel("A").point(lambda a: 255 if a > 40 else 0).getbbox()
    return img.crop(box) if box else img


# ------------------------------ 1. custom folder ------------------------------
def find_custom_sprite(dex_id: int, slug: str):
    """
    Look in sprites/ for a file belonging to this Pokémon. Accepted names:
        25.jpg   025.jpg   pikachu.jpg   025_pikachu.jpg   pikachu_front.png ...
    Returns (RGBA image, filename) or None.
    """
    want = norm(slug)
    for f in sorted(SPRITE_DIR.glob("*")):
        if f.suffix.lower() not in IMG_EXTS:
            continue
        if f.stem.isdigit():                                  # "025" style: match by number
            hit = int(f.stem) == dex_id
        else:
            name = re.sub(r"^\d+[\s_\-.]*", "", f.stem)       # drop a leading "025_"
            name = re.sub(r"[\s_\-.]*(front|sprite|normal)$", "", name, flags=re.I)
            hit = norm(name) == want
        if hit:
            img = Image.open(f)
            has_alpha = img.mode in ("RGBA", "LA") and img.getchannel("A").getextrema()[0] < 255
            img = img.convert("RGBA") if has_alpha else remove_background(img, tol=30)
            return crop_to_content(img), f.name
    return None


# --------------------------------- 2. the sheet --------------------------------
@lru_cache(maxsize=1)
def _sheet():
    return Image.open(SHEET_PATH).convert("RGB") if SHEET_PATH.exists() else None


def sheet_sprite(dex_id: int):
    """Slice the *front, normal-colour* sprite for a dex number out of the chart (or None)."""
    sheet = _sheet()
    if sheet is None:
        return None
    if dex_id in SPECIAL:
        row, col = SPECIAL[dex_id]
    elif 1 <= dex_id <= 150:
        row, col = divmod(dex_id - 1, PER_ROW)
    else:
        return None                                           # not on this chart
    # pixel position of this Pokémon's top-left "front, normal colour" cell, from the chart geometry above
    x, y = X0 + BLOCK_W * col, Y0 + BLOCK_H * row
    return remove_background(sheet.crop((x, y, x + CELL, y + CELL)), tol=0)


# ------------------------- comparing a sprite with artwork ---------------------
def _palette(img: Image.Image) -> np.ndarray:
    """Coarse colour histogram (4 levels per channel = 64 bins) of the opaque pixels.

    `>> 6` shifts each 0-255 channel right by 6 bits, leaving 0-3 (4 levels). Combining three 0-3 numbers as
    r*16 + g*4 + b gives one bin number 0-63. The histogram is then divided by its sum so it adds up to 1,
    which makes sprites of different sizes comparable."""
    a = np.asarray(img.convert("RGBA"))
    px = a[..., :3][a[..., 3] > 128] >> 6
    h = np.bincount(px[:, 0] * 16 + px[:, 1] * 4 + px[:, 2], minlength=64).astype(float)
    return h / max(h.sum(), 1)


def similarity(a: Image.Image, b: Image.Image) -> float:
    """Histogram intersection, 0 (nothing in common) .. 1 (identical palette).

    For every colour bin take the SMALLER of the two amounts and add them up. Two pictures of the same
    Pokémon (different size/style) share most colours -> high score; a picture of a different Pokémon -> low.
    agent.choose_sprite rejects a sprite scoring below MIN_MATCH."""
    return float(np.minimum(_palette(a), _palette(b)).sum())
