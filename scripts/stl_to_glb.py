"""Convert an STL into a light, phone-friendly GLB (needs only numpy).

    python scripts/stl_to_glb.py retro_charizard.stl models/custom/charizard.glb --faces 90000

STL is Z-up; GLB/three.js is Y-up, so the model is rotated (x,y,z)->(x,z,-y): a figure facing -Y ends up facing +Z.
It is centred on x/z, stood on y=0 and scaled to 1 unit tall. Detail is reduced by vertex clustering (merging
nearby vertices on a grid), then smooth normals are computed. The STL has no colours, so a palette is painted
from the shape: belly/underside cream, wing membranes teal, the rest orange.
"""
import argparse, json, struct
import numpy as np


def read_stl(path):
    b = open(path, "rb").read()
    n = struct.unpack("<I", b[80:84])[0]
    if 84 + 50 * n == len(b):                                    # binary STL
        dt = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
        return np.frombuffer(b, dt, count=n, offset=84)["v"].reshape(-1, 3).astype(np.float64)
    vs = [list(map(float, l.split()[1:4])) for l in b.decode("ascii", "ignore").splitlines() if l.strip().startswith("vertex")]
    return np.array(vs, dtype=np.float64)


def weld(tri_verts, decimals=5):
    u, inv = np.unique(np.round(tri_verts, decimals), axis=0, return_inverse=True)
    return u, inv.reshape(-1, 3)


def cluster(v, f, target_faces):
    lo, hi = v.min(0), v.max(0); size = (hi - lo).max()
    best = None
    a, b = size / 400, size / 10
    for _ in range(18):                                           # bisect the cell size to land near the target
        cell = (a + b) / 2
        key = np.floor((v - lo) / cell).astype(np.int64)
        _, ci = np.unique(key, axis=0, return_inverse=True)
        ci = ci.ravel(); g = ci[f]
        ok = (g[:, 0] != g[:, 1]) & (g[:, 1] != g[:, 2]) & (g[:, 0] != g[:, 2])
        nf = int(ok.sum()); best = (cell, ci, g[ok])
        if nf > target_faces: a = cell
        else: b = cell
    cell, ci, g = best
    cnt = np.bincount(ci).astype(np.float64)
    nv = np.stack([np.bincount(ci, weights=v[:, k]) / cnt for k in range(3)], 1)
    used = np.unique(g); remap = -np.ones(len(nv), np.int64); remap[used] = np.arange(len(used))
    return nv[used], remap[g]


def normals(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    vn = np.zeros_like(v)
    for k in range(3): np.add.at(vn, f[:, k], fn)
    return vn / np.maximum(np.linalg.norm(vn, axis=1, keepdims=True), 1e-12)


def palette(v, n):
    """v is Y-up, facing +Z. Colours by position/orientation (heuristic - the STL carries no colour)."""
    h = v[:, 1] / max(v[:, 1].max(), 1e-9)
    orange = np.array([0.93, 0.45, 0.13]); cream = np.array([0.98, 0.84, 0.52]); teal = np.array([0.26, 0.62, 0.66])
    col = np.tile(orange, (len(v), 1))
    belly = (n[:, 2] > 0.35) & (np.abs(v[:, 0]) < 0.22) & (h > 0.1) & (h < 0.62) & (v[:, 2] > np.percentile(v[:, 2], 55))
    col[belly] = cream
    wing = (np.abs(v[:, 0]) > 0.3) & (h > 0.62) & (np.abs(n[:, 2]) < 0.9) & (v[:, 2] < np.percentile(v[:, 2], 60))
    col[wing] = teal
    return col


def glb(v, f, n, col):
    f = f.astype(np.uint32); idx = f.tobytes()
    parts = [v.astype("<f4").tobytes(), n.astype("<f4").tobytes(), np.hstack([col, np.ones((len(col), 1))]).astype("<f4").tobytes(), idx]
    offs = np.cumsum([0] + [len(p) for p in parts]); body = b"".join(parts)
    js = {"asset": {"version": "2.0", "generator": "stl_to_glb.py"}, "scene": 0, "scenes": [{"nodes": [0]}],
          "nodes": [{"mesh": 0, "name": "charizard"}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "COLOR_0": 2}, "indices": 3, "material": 0}]}],
          "materials": [{"pbrMetallicRoughness": {"baseColorFactor": [1, 1, 1, 1], "metallicFactor": 0, "roughnessFactor": 0.55}}],
          "buffers": [{"byteLength": len(body)}],
          "bufferViews": [{"buffer": 0, "byteOffset": int(offs[i]), "byteLength": len(parts[i]), "target": 34963 if i == 3 else 34962} for i in range(4)],
          "accessors": [
              {"bufferView": 0, "componentType": 5126, "count": len(v), "type": "VEC3", "min": v.min(0).tolist(), "max": v.max(0).tolist()},
              {"bufferView": 1, "componentType": 5126, "count": len(v), "type": "VEC3"},
              {"bufferView": 2, "componentType": 5126, "count": len(v), "type": "VEC4"},
              {"bufferView": 3, "componentType": 5125, "count": int(f.size), "type": "SCALAR"}]}
    j = json.dumps(js, separators=(",", ":")).encode(); j += b" " * (-len(j) % 4); body += b"\0" * (-len(body) % 4)
    return struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(body)) + struct.pack("<I4s", len(j), b"JSON") + j + struct.pack("<I4s", len(body), b"BIN\0") + body


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("stl"); ap.add_argument("out"); ap.add_argument("--faces", type=int, default=90000)
    ap.add_argument("--no-color", action="store_true")
    a = ap.parse_args()
    tv = read_stl(a.stl); v, f = weld(tv)
    print(f"STL: {len(f)} triangles, {len(v)} vertices")
    v, f = cluster(v, f, a.faces)
    v = v @ np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]], float).T          # Z-up -> Y-up, facing +Z
    v -= [(v[:, 0].min() + v[:, 0].max()) / 2, v[:, 1].min(), (v[:, 2].min() + v[:, 2].max()) / 2]
    v /= v[:, 1].max()                                                      # 1 unit tall
    n = normals(v, f)
    col = np.ones_like(v) * [0.93, 0.45, 0.13] if a.no_color else palette(v, n)
    data = glb(v, f, n, col)
    open(a.out, "wb").write(data)
    print(f"-> {a.out}: {len(f)} triangles, {len(v)} vertices, {len(data)//1024} KB")


if __name__ == "__main__":
    main()
