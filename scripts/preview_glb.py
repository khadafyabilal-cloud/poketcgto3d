"""Tiny software renderer: python scripts/preview_glb.py model.glb out.png  -> front / side / back / 3-quarter views (no GPU needed)."""
import io, json, struct, sys
import numpy as np
from PIL import Image


def load(path):
    b = open(path, "rb").read(); jl = struct.unpack("<I", b[12:16])[0]; js = json.loads(b[20:20 + jl]); bin_ = b[28 + jl:]
    def acc(i):
        a = js["accessors"][i]; v = js["bufferViews"][a["bufferView"]]
        n = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4}[a["type"]]; dt = {5126: "<f4", 5125: "<u4", 5123: "<u2"}[a["componentType"]]
        return np.frombuffer(bin_, dt, count=a["count"] * n, offset=v["byteOffset"]).reshape(-1, n)
    imgs = {}
    def texture(mi):
        pb = js["materials"][mi]["pbrMetallicRoughness"]
        if "baseColorTexture" not in pb: return None, pb.get("baseColorFactor", [.8, .8, .8, 1])
        t = pb["baseColorTexture"]["index"]
        if t not in imgs:
            im = js["images"][js["textures"][t]["source"]]; v = js["bufferViews"][im["bufferView"]]
            imgs[t] = np.asarray(Image.open(io.BytesIO(bin_[v["byteOffset"]:v["byteOffset"] + v["byteLength"]])).convert("RGB"))
        return imgs[t], None
    out = []
    for p in js["meshes"][0]["primitives"]:
        tex, col = texture(p["material"])
        out.append((acc(p["attributes"]["POSITION"]), acc(p["attributes"]["NORMAL"]), acc(p["attributes"]["TEXCOORD_0"]), acc(p["indices"]).reshape(-1, 3), tex, col))
    return out


def render(prims, yaw, W=360, H=420):
    img = np.full((H, W, 3), 235, np.uint8); zb = np.full((H, W), 1e9)
    c, s = np.cos(yaw), np.sin(yaw); R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    for P, N, UV, F, tex, col in prims:
        Q = (P - [0, .5, 0]) @ R.T; NN = N @ R.T
        X = Q[:, 0] * H * .8 + W / 2; Y = H / 2 - Q[:, 1] * H * .8
        for f in F:
            x, y, z = X[f], Y[f], Q[f, 2]
            x0, x1, y0, y1 = int(max(x.min(), 0)), int(min(x.max() + 1, W - 1)), int(max(y.min(), 0)), int(min(y.max() + 1, H - 1))
            if x1 <= x0 or y1 <= y0: continue
            d = (y[1] - y[2]) * (x[0] - x[2]) + (x[2] - x[1]) * (y[0] - y[2])
            if abs(d) < 1e-9: continue
            gx, gy = np.meshgrid(np.arange(x0, x1 + 1) + .5, np.arange(y0, y1 + 1) + .5)
            l0 = ((y[1] - y[2]) * (gx - x[2]) + (x[2] - x[1]) * (gy - y[2])) / d; l1 = ((y[2] - y[0]) * (gx - x[2]) + (x[0] - x[2]) * (gy - y[2])) / d; l2 = 1 - l0 - l1
            m = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
            if not m.any(): continue
            zz = l0 * z[0] + l1 * z[1] + l2 * z[2]; ys, xs = np.nonzero(m); ys2, xs2 = ys + y0, xs + x0
            ok = zz[ys, xs] < zb[ys2, xs2]
            ys, xs, ys2, xs2 = ys[ok], xs[ok], ys2[ok], xs2[ok]
            if not len(ys): continue
            zb[ys2, xs2] = zz[ys, xs]
            w = np.stack([l0[ys, xs], l1[ys, xs], l2[ys, xs]], 1)
            nz = np.clip(-(w @ NN[f])[:, 2] * -1, -1, 1); sh = (0.55 + 0.45 * np.clip(nz, 0, 1))[:, None]
            if tex is not None:
                uv = w @ UV[f]; th, tw = tex.shape[:2]
                col_ = tex[(np.mod(uv[:, 1], 1) * (th - 1)).astype(int), (np.mod(uv[:, 0], 1) * (tw - 1)).astype(int)]
            else: col_ = np.array(col[:3]) * 255
            img[ys2, xs2] = np.clip(col_ * sh, 0, 255)
    return img


if __name__ == "__main__":
    prims = load(sys.argv[1])
    Image.fromarray(np.hstack([render(prims, a) for a in (0, np.pi / 2, np.pi, np.pi * 0.25)])).save(sys.argv[2])
