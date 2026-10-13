"""Draws a synthetic Pokémon card WITH printed game text (HP, attacks, energy circles) in the real layout, so the OCR
reader can be tested without photos. The fonts are DejaVu (Gill Sans is not freely available)."""
import cv2, numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
RGB = {"fire": (226, 70, 40), "water": (50, 130, 220), "grass": (60, 170, 70), "electric": (245, 205, 30),
       "psychic": (170, 70, 170), "normal": (225, 225, 225), "fighting": (160, 90, 40), "dark": (30, 30, 35)}


def text_card(hp=120, attacks=(("Ember", 30, ["fire"]), ("Growl", None, ["normal"])), w=630, h=880, pad=0.06, name="Charmander"):
    """-> BGR image shaped like card_detect.rectify output (card plus `pad` margin of table)."""
    im = Image.new("RGB", (w, h), (250, 214, 40))                          # yellow border
    d = ImageDraw.Draw(im)
    d.rectangle([int(.045 * w), int(.03 * h), int(.955 * w), int(.97 * h)], fill=(240, 232, 205))
    d.text((int(.08 * w), int(.045 * h)), name, font=ImageFont.truetype(BOLD, 34), fill=(20, 20, 20))
    f_hp = ImageFont.truetype(BOLD, 22); d.text((int(.60 * w), int(.062 * h)), "HP", font=f_hp, fill=(200, 30, 30))
    d.text((int(.68 * w), int(.045 * h)), str(hp), font=ImageFont.truetype(BOLD, 40), fill=(200, 30, 30))
    d.rectangle([int(.08 * w), int(.13 * h), int(.92 * w), int(.46 * h)], fill=(120, 170, 220))       # the picture
    y = int(.52 * h)
    fn, fr = ImageFont.truetype(BOLD, 36), ImageFont.truetype(FONT, 20)
    for nm, dmg, cost in attacks:
        x = int(.08 * w)
        for t in cost:
            r = 17; cx, cy = x + r, y + 24
            d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=RGB[t], outline=(30, 30, 30), width=2)
            if t == "normal": d.text((cx - 8, cy - 12), "★", font=ImageFont.truetype(BOLD, 20), fill=(60, 60, 60))
            x += 2 * r + 6
        d.text((int(.30 * w) if len(cost) < 3 else x + 10, y + 3), nm, font=fn, fill=(15, 15, 15))
        if dmg is not None:
            txt = str(dmg) if not isinstance(dmg, str) else dmg
            d.text((int(.80 * w), y + 1), txt, font=ImageFont.truetype(BOLD, 40), fill=(15, 15, 15))
        d.text((int(.08 * w), y + 56), "Flip a coin. If heads, this attack does more.", font=fr, fill=(40, 40, 40))
        y += 112
    img = cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR)
    p = int(pad * w)
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(70, 70, 70))
