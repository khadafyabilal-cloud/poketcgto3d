"""
Optional helper: cut every front sprite out of the chart and save it as a JPEG in
sprites/ (named 025_pikachu.jpg ...). Run:  python export_sheet_sprites.py
Use the output as a starting point, then replace/add your own files.
"""
from PIL import Image
from sprites import SPRITE_DIR, sheet_sprite

NAMES = """bulbasaur ivysaur venusaur charmander charmeleon charizard squirtle wartortle blastoise
caterpie metapod butterfree weedle kakuna beedrill pidgey pidgeotto pidgeot rattata raticate spearow
fearow ekans arbok pikachu raichu sandshrew sandslash nidoran-f nidorina nidoqueen nidoran-m nidorino
nidoking clefairy clefable vulpix ninetales jigglypuff wigglytuff zubat golbat oddish gloom vileplume
paras parasect venonat venomoth diglett dugtrio meowth persian psyduck golduck mankey primeape
growlithe arcanine poliwag poliwhirl poliwrath abra kadabra alakazam machop machoke machamp bellsprout
weepinbell victreebel tentacool tentacruel geodude graveler golem ponyta rapidash slowpoke slowbro
magnemite magneton farfetchd doduo dodrio seel dewgong grimer muk shellder cloyster gastly haunter
gengar onix drowzee hypno krabby kingler voltorb electrode exeggcute exeggutor cubone marowak
hitmonlee hitmonchan lickitung koffing weezing rhyhorn rhydon chansey tangela kangaskhan horsea
seadra goldeen seaking staryu starmie mr-mime scyther jynx electabuzz magmar pinsir tauros magikarp
gyarados lapras ditto eevee vaporeon jolteon flareon porygon omanyte omastar kabuto kabutops
aerodactyl snorlax articuno zapdos moltres dratini dragonair dragonite mewtwo mew""".split()
NAMES = {i + 1: n for i, n in enumerate(NAMES)}
NAMES[216] = "teddiursa"

SPRITE_DIR.mkdir(exist_ok=True)
for dex, name in NAMES.items():
    spr = sheet_sprite(dex)
    if spr is None:
        continue
    flat = Image.new("RGB", spr.size, (200, 200, 200))        # JPEG has no transparency:
    flat.paste(spr, mask=spr.getchannel("A"))                 # flat grey background
    flat.save(SPRITE_DIR / f"{dex:03d}_{name}.jpg", quality=98, subsampling=0)
print("wrote", len(list(SPRITE_DIR.glob('*.jpg'))), "sprites to", SPRITE_DIR)
