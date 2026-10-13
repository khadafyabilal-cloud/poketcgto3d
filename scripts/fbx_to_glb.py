"""Convert Pokémon FBX / OBJ models (with their textures) to compact, textured GLB files - pure Python (numpy + Pillow).

    python scripts/fbx_to_glb.py <Pokemon-Gen1 folder> models/gen1            # all of them
    python scripts/fbx_to_glb.py <Pokemon-Gen1 folder> models/gen1 --only 1 25 6

Reads binary FBX (7.x) directly: geometry, normals, UVs, per-polygon materials, the node hierarchy (so the mesh is
placed the way the file's author posed it) and each material's diffuse texture, found by file name anywhere inside the
Pokémon's folder (PNG/TGA/JPG). Skeleton and animation data are ignored: the model is exported in its rest pose.
Output: Y-up, 1 unit tall, centred on x/z, standing on y=0, textures shrunk to --tex px and embedded.

HOW THE PIECES FIT (read in this order)
  1. read_fbx()         parses the FBX container into a tree of Node(name, props, kids). A binary FBX is a list of
                        nested records: each has a name, a list of typed properties (numbers, strings, big arrays,
                        often zlib-compressed) and child records. Think of it as a very compact JSON.
  2. scene_meshes()     understands that tree: "Objects" holds Models (the transform hierarchy), Geometry (vertices,
                        polygons, normals, UVs), Materials and Textures; "Connections" says which object is attached to
                        which (a Geometry belongs to a Model, a Texture to a Material, a Model to its parent Model).
                        It bakes every Model's world transform into the vertices and returns plain numpy arrays.
  3. read_obj()         the same result from a Wavefront .obj + .mtl (a few Pokémon only ship that).
  4. facing_rotation()  the files disagree on which way a Pokémon faces; the eyes are on the front, so turn the model
                        until the eyes point +Z (the direction battle mode expects).
  5. build_glb()        writes the textured binary glTF (see glb_library.py for a glTF primer).
  6. load_best()/main() pick the best file per folder and run the conversion for every Pokémon.

GLOSSARY
  vertex     a point in 3-D.      face/triangle  three vertices joined into a surface.     normal  the direction a surface faces (lighting).
  UV         2-D coordinates telling each vertex which spot of the texture picture it shows (U = left-right, V = up-down).
  polygon    FBX stores faces with any number of corners; we split them into triangles ("fan triangulation").
  winding    the order in which a triangle's corners are listed decides which side is "front"; a mirrored transform flips it.
"""
import argparse, io, json, os, re, struct, sys, zlib
from pathlib import Path
import numpy as np
from PIL import Image

# ----------------------------------------------------------------------------- binary FBX reader
class Node:
    __slots__ = ("name", "props", "kids")
    def __init__(s, name, props, kids): s.name, s.props, s.kids = name, props, kids
    def find(s, name): return next((k for k in s.kids if k.name == name), None)
    def all(s, name): return [k for k in s.kids if k.name == name]
    def val(s, name, default=None):
        k = s.find(name); return k.props[0] if k and k.props else default


def _prop(b, o):
    """Decode ONE property starting at byte offset o -> (value, next offset). The first byte is a type letter:
    Y/I/L int16/32/64, F/D float32/64, C bool, S string, R raw bytes, and lower-case f/d/l/i/b = arrays (optionally zlib-compressed)."""
    t = chr(b[o]); o += 1
    if t == "Y": return struct.unpack_from("<h", b, o)[0], o + 2
    if t == "C": return bool(b[o]), o + 1
    if t == "I": return struct.unpack_from("<i", b, o)[0], o + 4
    if t == "F": return struct.unpack_from("<f", b, o)[0], o + 4
    if t == "D": return struct.unpack_from("<d", b, o)[0], o + 8
    if t == "L": return struct.unpack_from("<q", b, o)[0], o + 8
    if t in "fdlib":
        n, enc, clen = struct.unpack_from("<III", b, o); o += 12
        dt = {"f": "<f4", "d": "<f8", "l": "<i8", "i": "<i4", "b": "u1"}[t]
        raw = b[o:o + clen]; o += clen
        if enc == 1: raw = zlib.decompress(raw)
        return np.frombuffer(raw, dt, count=n), o
    if t in "SR":
        n = struct.unpack_from("<I", b, o)[0]; o += 4
        raw = bytes(b[o:o + n]); o += n
        return (raw.decode("utf-8", "replace") if t == "S" else raw), o
    raise ValueError("bad FBX property type " + t)


def read_fbx(path):
    """Read a binary FBX file into a tree of Node objects (see the module docstring)."""
    b = Path(path).read_bytes()
    if not b.startswith(b"Kaydara FBX Binary"): raise ValueError("not a binary FBX (ASCII FBX unsupported)")
    ver = struct.unpack_from("<I", b, 23)[0]; wide = ver >= 7500
    hdr = 24 if wide else 12; nul = 25 if wide else 13

    def node(o):
        if wide: end, npr, plen = struct.unpack_from("<QQQ", b, o)
        else: end, npr, plen = struct.unpack_from("<III", b, o)
        if end == 0: return None, o + nul
        nl = b[o + hdr]; name = bytes(b[o + hdr + 1:o + hdr + 1 + nl]).decode("latin1")
        p = o + hdr + 1 + nl; props = []
        for _ in range(npr):
            v, p = _prop(b, p); props.append(v)
        kids = []
        while p < end:
            k, p = node(p)
            if k is None: break
            kids.append(k)
        return Node(name, props, kids), end
    top, o = [], 27
    while o < len(b) - nul:
        n, o = node(o)
        if n is None: break
        top.append(n)
    return Node("root", [], top)

# ----------------------------------------------------------------------------- scene helpers
def euler(rx, ry, rz):
    """Rotation matrix from three angles in degrees (FBX default order: rotate about X, then Y, then Z)."""                                   # FBX default order XYZ -> R = Rz Ry Rx
    cx, sx, cy, sy, cz, sz = *(f(np.radians(a)) for a in (rx,) for f in (np.cos, np.sin)), *(f(np.radians(a)) for a in (ry,) for f in (np.cos, np.sin)), *(f(np.radians(a)) for a in (rz,) for f in (np.cos, np.sin))
    X = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]); Y = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]); Z = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    return Z @ Y @ X


def props70(n):
    out = {}
    p = n.find("Properties70")
    for k in (p.all("P") if p else []):
        if len(k.props) >= 5: out[k.props[0]] = k.props[4:] if len(k.props[4:]) > 1 else k.props[4]
    return out


def local_matrix(model):
    """The 4x4 matrix that places a Model relative to its parent: translate * rotate * scale (+ pivot/offset terms FBX defines).
    Chaining these matrices from the root down gives the model's position in the world."""
    P = props70(model)
    g = lambda k, d: np.array(P.get(k, d), float).reshape(-1)[:3] if k in P else np.array(d, float)
    def T(v): m = np.eye(4); m[:3, 3] = v; return m
    def R(v): m = np.eye(4); m[:3, :3] = euler(*v); return m
    def S(v): return np.diag([*v, 1.0])
    t, ro, rp = g("Lcl Translation", [0, 0, 0]), g("RotationOffset", [0, 0, 0]), g("RotationPivot", [0, 0, 0])
    pre, rot, post = g("PreRotation", [0, 0, 0]), g("Lcl Rotation", [0, 0, 0]), g("PostRotation", [0, 0, 0])
    so, sp, sc = g("ScalingOffset", [0, 0, 0]), g("ScalingPivot", [0, 0, 0]), g("Lcl Scaling", [1, 1, 1])
    return T(t) @ T(ro) @ T(rp) @ R(pre) @ R(rot) @ np.linalg.inv(R(post)) @ np.linalg.inv(T(rp)) @ T(so) @ T(sp) @ S(sc) @ np.linalg.inv(T(sp))


def find_texture(folder, name):
    """FBX stores the texture's path on the ripper's computer (C:\\Users\\...). Keep only the file NAME and search the Pokémon's own folder for it."""
    stem = re.split(r"[\\/]", name)[-1]; base = os.path.splitext(stem)[0].lower()
    best = None
    for p in folder.rglob("*"):
        if p.suffix.lower() in (".png", ".tga", ".jpg", ".jpeg", ".bmp") and p.stem.lower() == base:
            if best is None or p.suffix.lower() == os.path.splitext(stem)[1].lower() or "tex" in p.parent.name.lower(): best = p
    return best


def scene_meshes(fbx, folder):
    """-> list of dict(pos, nrm, uv, faces, material_index) in world space, and list of materials {tex: Path|None, color}"""
    obj = fbx.find("Objects"); conns = fbx.find("Connections")
    by_id = {}
    for n in obj.kids:
        if n.props and isinstance(n.props[0], int): by_id[n.props[0]] = n
    parent, children = {}, {}
    prop_conn = {}
    for c in conns.kids:
        if c.name != "C": continue
        kind, a, b = c.props[0], c.props[1], c.props[2]
        children.setdefault(b, []).append(a)
        if kind == "OO": parent.setdefault(a, b)
        if kind == "OP" and len(c.props) > 3: prop_conn[(a, b)] = c.props[3]
    world = {}
    def W(i):
        if i in world: return world[i]
        n = by_id.get(i)
        if n is None or n.name != "Model": world[i] = np.eye(4); return world[i]
        p = parent.get(i); world[i] = (W(p) if p in by_id else np.eye(4)) @ local_matrix(n); return world[i]
    mats = {}
    def material(mid):
        if mid in mats: return mats[mid]
        n = by_id[mid]; tex = None; col = (0.8, 0.8, 0.8)
        P = props70(n)
        for k in ("DiffuseColor", "Diffuse"):
            if k in P and not isinstance(P[k], (str, bytes)): col = tuple(float(x) for x in np.array(P[k]).reshape(-1)[:3]); break
        cands, seen, queue = [], set(), list(children.get(mid, []))
        while queue:                                           # Texture -> (LayeredTexture ->) Material
            ch = queue.pop(0)
            if ch in seen: continue
            seen.add(ch); t = by_id.get(ch)
            if t is None: continue
            if t.name == "LayeredTexture": queue += children.get(ch, [])
            elif t.name == "Texture": cands.append((0 if prop_conn.get((ch, mid), "DiffuseColor") in ("DiffuseColor", "Maya|baseColor", "base_color", "Maya|TEX_color_map") else 1, ch))
        for _, ch in sorted(cands):
            t = by_id[ch]
            for key in ("RelativeFilename", "FileName"):
                s_ = t.val(key)
                if s_:
                    tex = find_texture(folder, s_)
                    if tex: break
            if tex: break
        mats[mid] = {"tex": tex, "color": col, "name": str(n.props[1]).split("\x00")[0] if len(n.props) > 1 else str(mid)}
        return mats[mid]
    out = []
    for mid_, n in by_id.items():
        if n.name != "Model": continue
        geos = [c for c in children.get(mid_, []) if c in by_id and by_id[c].name == "Geometry"]
        if not geos: continue
        g = by_id[geos[0]]
        verts = np.asarray(g.val("Vertices"), float).reshape(-1, 3)
        pvi = np.asarray(g.val("PolygonVertexIndex"), np.int64)
        if len(pvi) == 0: continue
        ends = np.flatnonzero(pvi < 0); starts = np.r_[0, ends[:-1] + 1]
        idx = np.where(pvi < 0, ~pvi, pvi)
        npoly = len(ends); counts = ends - starts + 1
        poly_of = np.repeat(np.arange(npoly), counts)            # polygon id of each polygon-vertex
        nvert = len(pvi)
        def layer(kind, data_key, idx_key):
            L = next((x for x in g.all(kind) if x.find(data_key) is not None), None)   # exporters may write several; the first is often an empty stub
            if L is None: return None
            mapping = L.val("MappingInformationType", ""); ref = L.val("ReferenceInformationType", "")
            d = L.find(data_key); d = None if d is None else np.asarray(d.props[0])
            ix = L.find(idx_key); ix = None if ix is None else np.asarray(ix.props[0], np.int64)
            return mapping, ref, d, ix
        def expand(L, comp):
            mapping, ref, d, ix = L
            d = d.reshape(-1, comp)
            if mapping == "ByPolygonVertex": rows = ix if ref != "Direct" and ix is not None else np.arange(nvert)
            elif mapping == "ByVertice" or mapping == "ByVertex": rows = (ix[idx] if ref != "Direct" and ix is not None else idx)
            elif mapping == "ByPolygon": rows = (ix if ref != "Direct" and ix is not None else np.arange(npoly))[poly_of]
            else: rows = np.zeros(nvert, np.int64)
            return d[rows]
        Ln = layer("LayerElementNormal", "Normals", "NormalsIndex"); Lu = layer("LayerElementUV", "UV", "UVIndex")
        Lm = max(g.all("LayerElementMaterial"), key=lambda x: len(np.unique(x.find("Materials").props[0])) if x.find("Materials") else 0, default=None)
        M = world_m = W(mid_)
        pos = verts[idx]; nrm = expand(Ln, 3) if Ln and Ln[2] is not None else None; uv = expand(Lu, 2) if Lu and Lu[2] is not None else None
        if Lm is not None and Lm.val("MappingInformationType", "") == "ByPolygon":
            mi = np.asarray(Lm.find("Materials").props[0], np.int64)
            pm = mi[:npoly] if len(mi) >= npoly else np.zeros(npoly, np.int64)
        else: pm = np.zeros(npoly, np.int64)
        # triangulate (fan)
        tri = []
        for k in range(1, int(counts.max()) - 1):
            ok = counts >= k + 2
            s = starts[ok]; tri.append(np.stack([s, s + k, s + k + 1], 1)); 
        tri = np.concatenate(tri) if tri else np.zeros((0, 3), np.int64)
        tri_poly = poly_of[tri[:, 0]]
        # world transform
        A = M[:3, :3]; pos = pos @ A.T + M[:3, 3]
        if nrm is not None:
            nrm = nrm @ np.linalg.inv(A).T; nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-12)
        if np.linalg.det(A) < 0: tri = tri[:, ::-1]
        mlist = [c for c in children.get(mid_, []) if c in by_id and by_id[c].name == "Material"]
        out.append(dict(pos=pos, nrm=nrm, uv=uv, tri=tri, tri_mat=pm[tri_poly], mats=[material(m) for m in mlist], name=n.props[1]))
    return out

# ----------------------------------------------------------------------------- OBJ (a few folders ship only this)
def read_obj(path):
    path = Path(path); V, VT, VN, faces = [], [], [], []; cur = None; mtl = {}; mats = []; order = {}
    mf = None
    for ln in path.read_text(errors="ignore").splitlines():
        t = ln.split()
        if not t: continue
        if t[0] == "v": V.append([float(x) for x in t[1:4]])
        elif t[0] == "vt": VT.append([float(x) for x in t[1:3]])
        elif t[0] == "vn": VN.append([float(x) for x in t[1:4]])
        elif t[0] == "mtllib": mf = path.parent / " ".join(t[1:])
        elif t[0] == "usemtl": cur = " ".join(t[1:]); order.setdefault(cur, len(order))
        elif t[0] == "f":
            ids = []
            for tok in t[1:]:
                a = (tok.split("/") + ["", ""])[:3]
                ids.append((int(a[0]) - 1 if int(a[0]) > 0 else len(V) + int(a[0]), int(a[1]) - 1 if a[1] else -1, int(a[2]) - 1 if a[2] else -1))
            for k in range(1, len(ids) - 1): faces.append((ids[0], ids[k], ids[k + 1], order.get(cur, 0) if cur else 0))
    if mf and mf.exists():
        c = None
        for ln in mf.read_text(errors="ignore").splitlines():
            t = ln.split(None, 1)
            if len(t) < 2: continue
            if t[0] == "newmtl": c = t[1].strip(); mtl[c] = {"tex": None, "color": (0.8, 0.8, 0.8), "name": c}
            elif t[0].lower() == "map_kd" and c: mtl[c]["tex"] = find_texture(path.parent.parent if path.parent.name.lower() == "source" else path.parent, t[1])
    names = sorted(order, key=order.get) or [None]
    mats = [mtl.get(n, {"tex": None, "color": (0.8, 0.8, 0.8), "name": str(n)}) for n in names]
    V, VT, VN = np.array(V), np.array(VT) if VT else None, np.array(VN) if VN else None
    pos, nrm, uv, tm = [], [], [], []
    for a, b, c, m in faces:
        for v in (a, b, c):
            pos.append(V[v[0]]); uv.append(VT[v[1]] if VT is not None and v[1] >= 0 else (0, 0)); nrm.append(VN[v[2]] if VN is not None and v[2] >= 0 else (0, 0, 0)); 
        tm.append(m)
    nrm = np.array(nrm); n = len(faces)
    return [dict(pos=np.array(pos), nrm=nrm if np.abs(nrm).sum() else None, uv=np.array(uv), tri=np.arange(3 * n).reshape(-1, 3), tri_mat=np.array(tm), mats=mats, name=path.stem)]

# ----------------------------------------------------------------------------- GLB writer
def smooth_normals(pos, tri):
    """Recompute normals when the file has none: add up the face normals around each (welded) vertex and normalise."""
    key = np.round(pos, 5); _, inv = np.unique(key, axis=0, return_inverse=True); inv = inv.ravel()
    fn = np.cross(pos[tri[:, 1]] - pos[tri[:, 0]], pos[tri[:, 2]] - pos[tri[:, 0]])
    acc = np.zeros((inv.max() + 1, 3))
    for k in range(3): np.add.at(acc, inv[tri[:, k]], fn)
    n = acc[inv]; return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)


def encode_tex(path, size):
    """Load a texture (PNG/TGA/...), shrink it to `size`, and return (bytes, mime type, has_transparency). Opaque textures become JPEG (small),
    transparent ones PNG (alpha is needed for eye/flame cut-outs)."""
    im = Image.open(path); im.load()
    if im.mode in ("P", "LA", "PA"): im = im.convert("RGBA")
    has_a = im.mode == "RGBA" and np.asarray(im)[..., 3].min() < 250
    im = im.convert("RGBA" if has_a else "RGB")
    if max(im.size) > size:
        k = size / max(im.size); im = im.resize((max(1, round(im.width * k)), max(1, round(im.height * k))), Image.LANCZOS)
    buf = io.BytesIO()
    if has_a: im.save(buf, "PNG", optimize=True)
    else: im.save(buf, "JPEG", quality=88)
    return buf.getvalue(), ("image/png" if has_a else "image/jpeg"), has_a


FACING_FIX = {96: 180.0}          # Pokédex no. -> degrees to turn, for files with no eye part to detect the front from (checked by eye)


def facing_rotation(meshes):
    """The files don't agree which way a Pokémon faces. Eyes sit on the front, so turn the model until the eyes point +Z.
    Falls back to 'already facing +Z' when no eye part is found or the eyes are centred."""
    pat = re.compile(r"eye|iris|pupil", re.I); pts = []
    for m in meshes:                                                   # per triangle: a body mesh can carry the eye material too
        is_eye = [bool(pat.search(str(x.get("name", "")) + (x["tex"].stem if x.get("tex") else ""))) for x in m["mats"]]
        mesh_eye = bool(pat.search(m["name"]))
        for mi in np.unique(m["tri_mat"]):
            if mesh_eye or (mi < len(is_eye) and is_eye[mi]): pts.append(m["pos"][m["tri"][m["tri_mat"] == mi].ravel()])
    eyes = pts
    if not eyes: return np.eye(3), None
    allp = np.concatenate([m["pos"] for m in meshes]); centre = (allp.min(0) + allp.max(0)) / 2
    e = np.concatenate(eyes).mean(0) - centre
    d = np.array([e[0], e[2]]); height = allp[:, 1].max() - allp[:, 1].min()
    # A single eye (REye only) sits off to one side, so x says nothing about facing: use x only when it clearly dominates.
    if abs(d[0]) > 2.5 * abs(d[1]): ang = np.pi / 2 * np.sign(d[0])
    elif abs(d[1]) >= 0.08 * max(height, 1e-9): ang = 0.0 if d[1] > 0 else np.pi
    else: return np.eye(3), None
    t = -ang; c, s_ = np.cos(t), np.sin(t)
    return np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]), float(np.degrees(ang))


def convert(meshes, out_path, tex_px=512, max_tris=60000, dex=None):
    rot, ang = facing_rotation(meshes)
    if dex in FACING_FIX:
        t = np.radians(-FACING_FIX[dex]); c, s_ = np.cos(t), np.sin(t)
        rot, ang = np.array([[c, 0, s_], [0, 1, 0], [-s_, 0, c]]) @ rot, FACING_FIX[dex]
    info = build_glb(meshes, None, None, out_path, tex_px, max_tris, rot)
    info["turned"] = ang
    return info


def build_glb(meshes, lo, hi, out_path, tex_px, max_tris, rot=None):
    rot = np.eye(3) if rot is None else rot
    allp = np.concatenate([m["pos"] @ rot.T for m in meshes]); lo, hi = allp.min(0), allp.max(0)
    height = max(hi[1] - lo[1], 1e-9); centre = np.array([(lo[0] + hi[0]) / 2, lo[1], (lo[2] + hi[2]) / 2])
    chunks, views, accs, images, textures, materials, prims = [], [], [], [], [], [], []
    def add_view(data, target=None):
        pad = (-sum(len(c) for c in chunks)) % 4
        if pad: chunks.append(b"\0" * pad)
        off = sum(len(c) for c in chunks); chunks.append(data)
        views.append({"buffer": 0, "byteOffset": off, "byteLength": len(data), **({"target": target} if target else {})}); return len(views) - 1
    tex_cache, mat_cache = {}, {}
    def material_for(m):
        key = (str(m["tex"]), m["color"])
        if key in mat_cache: return mat_cache[key]
        pbr = {"metallicFactor": 0.0, "roughnessFactor": 0.8}; extra = {}
        if m["tex"] is not None:
            if str(m["tex"]) not in tex_cache:
                data, mime, has_a = encode_tex(m["tex"], tex_px)
                images.append({"bufferView": add_view(data), "mimeType": mime}); textures.append({"source": len(images) - 1, "sampler": 0})
                tex_cache[str(m["tex"])] = (len(textures) - 1, has_a)
            ti, has_a = tex_cache[str(m["tex"])]
            pbr["baseColorTexture"] = {"index": ti}
            if has_a: extra["alphaMode"] = "MASK"; extra["alphaCutoff"] = 0.5
        else: pbr["baseColorFactor"] = [*m["color"], 1.0]
        materials.append({"pbrMetallicRoughness": pbr, "doubleSided": True, **extra}); mat_cache[key] = len(materials) - 1
        return mat_cache[key]
    total = sum(len(m["tri"]) for m in meshes); keep = min(1.0, max_tris / max(total, 1))
    for m in meshes:
        pos = ((m["pos"] @ rot.T) - centre) / height
        nrm = None if m["nrm"] is None else m["nrm"] @ rot.T
        uv = m["uv"]; tri = m["tri"]; tm = m["tri_mat"]
        if nrm is None: nrm = smooth_normals(pos, tri)
        for mi in np.unique(tm):
            sel = tri[tm == mi]
            if len(sel) == 0: continue
            used, inv = np.unique(sel.ravel(), return_inverse=True)
            P, N = pos[used], nrm[used]; UV = uv[used] if uv is not None else np.zeros((len(used), 2))
            UV = np.stack([UV[:, 0], 1 - UV[:, 1]], 1)
            # weld identical vertices
            rows = np.hstack([np.round(P, 5), np.round(N, 3), np.round(UV, 5)])
            _, first, back = np.unique(rows, axis=0, return_index=True, return_inverse=True)
            back = back.ravel(); P, N, UV = P[first], N[first], UV[first]; idx = back[inv].reshape(-1, 3)
            mat = m["mats"][mi] if mi < len(m["mats"]) else {"tex": None, "color": (0.8, 0.8, 0.8)}
            a = {}
            for nm, arr, dt in (("POSITION", P, "<f4"), ("NORMAL", N, "<f4"), ("TEXCOORD_0", UV, "<f4")):
                v = add_view(arr.astype(dt).tobytes(), 34962)
                accs.append({"bufferView": v, "componentType": 5126, "count": len(arr), "type": "VEC3" if arr.shape[1] == 3 else "VEC2",
                             **({"min": arr.min(0).tolist(), "max": arr.max(0).tolist()} if nm == "POSITION" else {})}); a[nm] = len(accs) - 1
            v = add_view(idx.astype("<u4").ravel().tobytes(), 34963)
            accs.append({"bufferView": v, "componentType": 5125, "count": int(idx.size), "type": "SCALAR"})
            prims.append({"attributes": a, "indices": len(accs) - 1, "material": material_for(mat)})
    body = b"".join(chunks); body += b"\0" * (-len(body) % 4)
    js = {"asset": {"version": "2.0", "generator": "fbx_to_glb.py"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
          "meshes": [{"primitives": prims}], "materials": materials, "buffers": [{"byteLength": len(body)}], "bufferViews": views, "accessors": accs,
          "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 10497, "wrapT": 10497}]}
    if images: js["images"] = images; js["textures"] = textures
    j = json.dumps(js, separators=(",", ":")).encode(); j += b" " * (-len(j) % 4)
    data = struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(body)) + struct.pack("<I4s", len(j), b"JSON") + j + struct.pack("<I4s", len(body), b"BIN\0") + body
    Path(out_path).parent.mkdir(parents=True, exist_ok=True); Path(out_path).write_bytes(data)
    return dict(tris=int(sum(len(m["tri"]) for m in meshes)), kb=len(data) // 1024, prims=len(prims), textured=sum(1 for m in materials if "baseColorTexture" in m["pbrMetallicRoughness"]), size=(hi - lo).tolist())

# ----------------------------------------------------------------------------- picking the right file per Pokémon
def candidates(folder):
    f = Path(folder); fb = lambda p: [x for x in p.glob("*") if x.suffix.lower() == ".fbx"]
    top = [x for x in fb(f) if not re.search(r"anim|split", x.stem, re.I)]
    ph = fb(f / "placeholder") if (f / "placeholder").exists() else []
    src = [x for x in fb(f / "source")] if (f / "source").exists() else []
    objs = [x for x in (f / "source").glob("*.obj")] if (f / "source").exists() else []
    return top + objs + ph + src


def load_best(folder):
    """Try every model file in the folder; keep the one with the most triangles (the real model beats a 800-triangle placeholder)."""
    best, last = None, None
    for c in candidates(folder):
        try:
            ms = scene_meshes(read_fbx(c), Path(folder)) if c.suffix.lower() == ".fbx" else read_obj(c)
            ms = [m for m in ms if len(m["tri"])]
            if not ms: continue
            score = sum(len(m["tri"]) for m in ms) * (1.0 if any(x["tex"] for m in ms for x in m["mats"]) else 0.2)
            if best is None or score > best[0] * 1.15: best = (score, c, ms)        # earlier candidates win near-ties
        except Exception as e: last = f"{c.name}: {type(e).__name__}: {e}"
    if best is None: raise RuntimeError(last or "no model file found")
    return best[1], best[2]


def dex_of(name): return int(re.match(r"\d+", name).group())


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("src"); ap.add_argument("out")
    ap.add_argument("--only", type=int, nargs="*"); ap.add_argument("--tex", type=int, default=512)
    a = ap.parse_args(); report = {}
    for d in sorted(Path(a.src).iterdir()):
        if not d.is_dir() or not re.match(r"\d+", d.name): continue
        n = dex_of(d.name)
        if a.only and n not in a.only: continue
        try:
            c, ms = load_best(d)
            info = convert(ms, Path(a.out) / f"{n:03d}.glb", a.tex, dex=n)
            info["source"] = str(c.relative_to(d)); report[n] = info
            print(f"{n:3d} {d.name:22s} {info['tris']:6d} tris  {info['kb']:5d} KB  tex {info['textured']}/{info['prims']}  turned {info['turned']}  <- {info['source']}")
        except Exception as e:
            report[n] = {"error": f"{type(e).__name__}: {e}"}; print(f"{n:3d} {d.name:22s} FAILED {e}")
    json.dump(report, open(Path(a.out) / "report.json", "w"), indent=1)

if __name__ == "__main__":
    main()
