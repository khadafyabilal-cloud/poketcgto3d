"""
glb_library.py - use a pack of ready-made Pokémon .glb models (e.g. regular.zip).

What it does
  1. INDEX   - lists the pack once at start-up (names only, nothing is extracted).
               Files are named by Pokédex number: 25.glb, 916-F.glb, 6-Mega.glb ...
  2. SLIM    - the pack's models carry up to ~84 animation clips each (249.glb = 14.7 MB,
               99 % of it animation). We keep the mesh + textures untouched and keep only ONE
               looping idle clip, which typically shrinks a model 10-100x. This is a lossless
               re-pack of the binary: meshes (Draco) and textures (WebP) are copied byte-for-byte.
  3. CACHE   - slimmed models are written once to models/glb/ and reused (browser-cached too).

Run  `python glb_library.py --warm`  to pre-build every model with all CPU cores (optional).

glTF / GLB PRIMER (you need this to read slim_glb below)
  A .glb is one binary file with a small header and two "chunks":
     JSON chunk : a description of the scene - which meshes, materials, animations exist and where their numbers live
     BIN  chunk : one big blob of raw bytes holding all those numbers (vertex positions, indices, animation keyframes, images)
  The JSON points into the blob through three layers of indirection:
     buffer      the whole BIN blob
     bufferView  a byte range of the buffer         ("bytes 1000-5000")
     accessor    how to read a bufferView as numbers ("these 300 bytes are 25 VEC3 floats")
  A mesh primitive refers to accessors by NUMBER (attributes: POSITION -> accessor 3). Therefore deleting animations
  means: drop the unused accessors, copy the bytes that remain, and renumber everything that pointed at the survivors.
  That renumbering is what the three numbered steps inside slim_glb do.

How a card name becomes a file here: the pack names files by Pokédex number (25.glb, 6-Mega.glb). The regular
expression _NAME splits "6-Mega" into dex=6, variant="mega"; _pick_variant then chooses between variants using words on the card.
"""
import json, os, re, struct, sys, zipfile, threading
from pathlib import Path

ROOT = Path(__file__).parent
PACK_DIR = Path(os.environ.get("POKEMON_MODEL_PACKS", ROOT / "model_packs"))   # put regular.zip here
CACHE_DIR = ROOT / "models" / "glb"
CACHE_VERSION = "v1"                      # bump to invalidate cached slim files if the logic changes

# glTF binary constants
_JSON, _BIN = 0x4E4F534A, 0x004E4942      # chunk types: "JSON", "BIN\0"
_COMP = {5120: 1, 5121: 1, 5122: 2, 5123: 2, 5125: 4, 5126: 4}                 # bytes per component
_NCOMP = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT2": 4, "MAT3": 9, "MAT4": 16}

# Keywords in a form/variant name -> words we look for in a card name ("Alolan Vulpix")
FORM_HINTS = {"alola": ("alolan",), "galar": ("galarian",), "hisui": ("hisuian",),
              "paldea": ("paldean",), "mega": ("mega",), "gmax": ("gigantamax", "vmax")}


# ------------------------------- GLB read / write -------------------------------
def _parse(b: bytes):
    """Split a .glb into (json dict, binary bytes).

    Layout: 12-byte header (magic "glTF", version, total length), then chunks, each = 4-byte length,
    4-byte type, data. "<I" means "little-endian unsigned 32-bit integer"."""
    if struct.unpack_from("<I", b)[0] != 0x46546C67:                  # 0x46546C67 = the ASCII letters "glTF"
        raise ValueError("not a GLB")
    off, js, binc = 12, None, b""
    while off < len(b):
        n, t = struct.unpack_from("<II", b, off)
        chunk = b[off + 8: off + 8 + n]
        if t == _JSON: js = json.loads(chunk)
        elif t == _BIN: binc = chunk
        off += 8 + n
    return js, binc


def _pack(js: dict, binc: bytes) -> bytes:
    """Write (json, binary) back into a .glb (chunks padded to 4 bytes as the spec requires)."""
    j = json.dumps(js, separators=(",", ":")).encode()
    j += b" " * (-len(j) % 4)                 # (-len % 4) = how many bytes are missing to reach a multiple of 4
    binc += b"\0" * (-len(binc) % 4)
    total = 12 + 8 + len(j) + 8 + len(binc)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(j), _JSON) + j
            + struct.pack("<II", len(binc), _BIN) + binc)


def _pick_clip(anims):
    """Prefer a looping 'wait' clip, then any 'idle', then 'wait', then the first clip.
    (Pack models name their clips like "wait_loop"/"idle"; a calm looping clip looks best in AR.)"""
    names = [(a.get("name") or "").lower() for a in anims]
    for test in (lambda n: "wait" in n and "loop" in n, lambda n: "idle" in n,
                 lambda n: "wait" in n, lambda n: True):
        for i, n in enumerate(names):
            if test(n):
                return i
    return None


def slim_glb(data: bytes) -> bytes:
    """
    Return the model with only one animation clip. Safe by design: on ANY surprise
    (unexpected structure, error) the original bytes are returned unchanged.
    """
    try:
        js, binc = _parse(data)
        anims = js.get("animations") or []
        if len(anims) <= 1:
            return data                                   # nothing worth removing
        keep = _pick_clip(anims)
        js["animations"] = [anims[keep]]

        # 1) which accessors are still needed?  (mesh data, skin matrices, kept clip)
        #    Everything else (the other ~80 animation clips) is simply never copied.
        used = set()
        for m in js.get("meshes", []):
            for p in m["primitives"]:
                used.update(p.get("attributes", {}).values())
                if "indices" in p: used.add(p["indices"])
                for t in p.get("targets", []): used.update(t.values())
        for s in js.get("skins", []):
            if "inverseBindMatrices" in s: used.add(s["inverseBindMatrices"])
        for sm in js["animations"][0]["samplers"]:
            used.update((sm["input"], sm["output"]))
        amap = {old: new for new, old in enumerate(sorted(used))}     # old accessor number -> new (renumbered) number

        # 2) rebuild the binary buffer: verbatim copies of image + Draco views, plus ONE new
        #    view holding the (tiny) animation / skin accessor data.
        out, new_bv, bvmap = bytearray(), [], {}
        old_bv = js["bufferViews"]

        def copy_view(i):                                  # byte-for-byte copy of an old view
            if i not in bvmap:
                v = old_bv[i]
                out.extend(b"\0" * (-len(out) % 4))
                new_bv.append({"buffer": 0, "byteOffset": len(out), "byteLength": v["byteLength"]})
                out.extend(binc[v.get("byteOffset", 0): v.get("byteOffset", 0) + v["byteLength"]])
                bvmap[i] = len(new_bv) - 1
            return bvmap[i]

        for img in js.get("images", []):
            if "bufferView" in img: img["bufferView"] = copy_view(img["bufferView"])
        for m in js.get("meshes", []):
            for p in m["primitives"]:
                d = p.get("extensions", {}).get("KHR_draco_mesh_compression")
                if d: d["bufferView"] = copy_view(d["bufferView"])

        blob, new_acc = bytearray(), []
        for old in sorted(used):
            a = dict(js["accessors"][old])
            if "sparse" in a: raise ValueError("sparse accessor")
            if "bufferView" in a:
                v = old_bv[a["bufferView"]]
                if v.get("byteStride"): raise ValueError("strided accessor")
                n = a["count"] * _COMP[a["componentType"]] * _NCOMP[a["type"]]
                s = v.get("byteOffset", 0) + a.get("byteOffset", 0)
                blob.extend(b"\0" * (-len(blob) % 4))
                a["byteOffset"] = len(blob)
                blob.extend(binc[s: s + n])
                a["bufferView"] = -1                       # patched below
            new_acc.append(a)
        if blob:
            out.extend(b"\0" * (-len(out) % 4))
            new_bv.append({"buffer": 0, "byteOffset": len(out), "byteLength": len(blob)})
            out.extend(blob)
            for a in new_acc:
                if a.get("bufferView") == -1: a["bufferView"] = len(new_bv) - 1

        # 3) re-point every index at the renumbered accessors / views
        js["accessors"], js["bufferViews"], js["buffers"] = new_acc, new_bv, [{"byteLength": len(out)}]
        for m in js.get("meshes", []):
            for p in m["primitives"]:
                p["attributes"] = {k: amap[v] for k, v in p.get("attributes", {}).items()}
                if "indices" in p: p["indices"] = amap[p["indices"]]
                for t in p.get("targets", []):
                    for k in t: t[k] = amap[t[k]]
        for s in js.get("skins", []):
            if "inverseBindMatrices" in s: s["inverseBindMatrices"] = amap[s["inverseBindMatrices"]]
        for sm in js["animations"][0]["samplers"]:
            sm["input"], sm["output"] = amap[sm["input"]], amap[sm["output"]]
        slim = _pack(js, bytes(out))
        return slim if len(slim) < len(data) else data
    except Exception:
        return data


# ----------------------------------- library -----------------------------------
class GlbLibrary:
    """Index of every model in PACK_DIR (zip files and/or loose .glb files).

    self.index looks like {25: {"": (pack_path, "25.glb")}, 6: {"": ..., "mega": ..., "gmax": ...}}:
    Pokédex number -> variant name -> where the file lives. Only NAMES are read at start-up, so
    indexing a 1 GB zip takes milliseconds; a model's bytes are read and slimmed on first use."""
    _NAME = re.compile(r"(?:^|[/\\])(\d+)(?:[-_]([A-Za-z0-9]+))?\.glb$", re.I)

    def __init__(self, pack_dir: Path = PACK_DIR):
        self.index = {}                  # dex -> {variant: (pack path, member name)}
        self._zips, self._lock = {}, threading.Lock()
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        if pack_dir.exists():
            for p in sorted(pack_dir.rglob("*")):
                if p.suffix.lower() == ".zip":
                    with zipfile.ZipFile(p) as z:
                        for name in z.namelist(): self._add(p, name)
                elif p.suffix.lower() == ".glb":
                    self._add(p, None)
        self.count = sum(len(v) for v in self.index.values())

    def _add(self, pack, member):
        m = self._NAME.search(member or pack.name)
        if m:
            self.index.setdefault(int(m[1]), {})[(m[2] or "").lower()] = (pack, member)

    def has(self, dex): return dex in self.index

    def _pick_variant(self, dex, card_name=""):
        """Choose a variant: a form named on the card, else the plain model, else male, else first."""
        variants, low = self.index[dex], card_name.lower()
        for v in variants:
            if v and any(h in low for h in FORM_HINTS.get(v, ())): return v
        for v in ("", "m"):
            if v in variants: return v
        return sorted(variants)[0]

    def _read(self, pack, member):
        if member is None: return pack.read_bytes()
        with self._lock:                                  # one shared ZipFile handle, reads serialised
            z = self._zips.get(pack) or self._zips.setdefault(pack, zipfile.ZipFile(pack))
            return z.read(member)

    def get(self, dex, card_name=""):
        """-> (cached slim .glb path, info dict) or None. Builds the slim file on first use."""
        if dex not in self.index: return None
        var = self._pick_variant(dex, card_name)
        pack, member = self.index[dex][var]
        out = CACHE_DIR / f"{dex}{'-' + var if var else ''}.{CACHE_VERSION}.glb"
        info = {"pack": pack.name, "member": member or pack.name, "variant": var, "cached": out.exists()}
        if not out.exists():
            raw = self._read(pack, member)
            slim = slim_glb(raw)
            tmp = out.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp")
            tmp.write_bytes(slim); os.replace(tmp, out)          # atomic: no half-written files
            info.update(raw_kb=len(raw) // 1024, slim_kb=len(slim) // 1024)
        else:
            info["slim_kb"] = out.stat().st_size // 1024
        return out, info


def _warm_one(args):                                       # top-level so it can run in a process pool
    dex, lib = args
    return dex, lib.get(dex)[1].get("slim_kb")


if __name__ == "__main__" and "--warm" in sys.argv:
    from concurrent.futures import ProcessPoolExecutor
    lib = GlbLibrary()
    print(f"Indexed {lib.count} models for {len(lib.index)} Pokémon; building slim copies…")
    with ProcessPoolExecutor() as ex:
        total = sum(kb or 0 for _, kb in ex.map(_warm_one, [(d, lib) for d in lib.index], chunksize=8))
    print(f"Done. Total cache size ≈ {total // 1024} MB in {CACHE_DIR}")
