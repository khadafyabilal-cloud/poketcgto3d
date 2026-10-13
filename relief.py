"""
relief.py - turn a flat picture (official artwork) into a smooth, textured, "inflated" 3D figure.

The old sprite -> voxel path makes blocky cubes. This makes a rounded, pillow-profile solid instead:
  * the silhouette comes from the picture's alpha channel,
  * thickness follows the distance to the outline (thick in the middle, zero at the edge, circular profile),
  * the artwork itself is the texture, on both the front and (mirrored) the back, so colours, eyes and
    shading stay as sharp as the source image - there are no cubes.
Used for Pokémon that have no hand-made model. Pure numpy/scipy/Pillow; writes a GLB (Y-up, 1 unit tall,
standing on y=0, front facing +Z).

HOW IT WORKS - a walk-through (the numbered comments in build_relief follow this list)
  1. Trim the picture and "bleed" colour into the transparent pixels, so texture filtering at the silhouette
     edge never blends in black.
  2. Shrink the silhouette (alpha channel) to a coarse GRID x GRID mask: one mask cell = one square of the mesh.
  3. HEIGHT FIELD: for every point inside the figure compute its distance to the outline (distance_transform_edt)
     and turn it into a thickness with a circular profile  z = sqrt(1-(1-d)^2)  (0 at the edge, max in the middle).
     That is the shape of a pillow / inflated balloon.
  4. MESH: the grid nodes become vertices. The same grid is used twice: z = +height for the FRONT surface and
     z = -height for the BACK. Where the height is 0 (the outline) both surfaces touch, so the solid is closed.
  5. NORMALS (which way each point faces, for lighting) come from the slope of the height field.
  6. UV coordinates map each vertex back onto the picture so the artwork is the texture; the back uses the mirrored picture.
  7. Write everything as a binary glTF (.glb) by hand - see glb_library.py for a primer on that format.
"""
import io, json, struct
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt, gaussian_filter, binary_dilation


def build_relief(img: Image.Image, out_path, grid: int = 112, tex_px: int = 640, depth: float = 0.2):
    """img: RGBA artwork. grid: mesh resolution along the longest side. tex_px: texture size cap.
    depth: half-thickness as a fraction of the figure's smaller dimension. Returns the output Path."""
    img = img.convert("RGBA")
    # --- 1. trim + colour bleed ---
    bbox = img.getchannel("A").point(lambda v: 255 if v > 24 else 0).getbbox()
    if bbox is None:
        raise ValueError("picture is fully transparent")
    img = img.crop(bbox)
    if max(img.size) > tex_px:
        k = tex_px / max(img.size); img = img.resize((max(1, round(img.width * k)), max(1, round(img.height * k))), Image.LANCZOS)
    w, h = img.size
    # an alpha-bled texture: fill transparent pixels with nearby colour so edge filtering never shows a dark halo
    rgba = np.asarray(img).copy()
    solid = rgba[..., 3] > 24
    if (~solid).any():
        _, (iy, ix) = distance_transform_edt(~solid, return_indices=True)
        rgba[..., :3] = rgba[iy, ix, :3]
    tex = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO(); tex.save(buf, "PNG", optimize=True); png = buf.getvalue()

    # --- 2. coarse mask: gw x gh cells ---   (cells are grid x gh)
    gw = grid if w >= h else max(8, round(grid * w / h)); gh = grid if h > w else max(8, round(grid * h / w))
    a = np.asarray(img.getchannel("A").resize((gw, gh), Image.BILINEAR)) > 100
    P = np.pad(a, 1)
    node_in = P[:-1, :-1][:gh + 1, :gw + 1] | P[:-1, 1:][:gh + 1, :gw + 1] | P[1:, :-1][:gh + 1, :gw + 1] | P[1:, 1:][:gh + 1, :gw + 1]
    # nodes on the outline (touching any empty cell) get zero thickness so the front and back surfaces meet there with no crack
    inner = ~node_in | False
    Q = np.pad(a, 1)
    inner = Q[:-1, :-1][:gh + 1, :gw + 1] & Q[:-1, 1:][:gh + 1, :gw + 1] & Q[1:, :-1][:gh + 1, :gw + 1] & Q[1:, 1:][:gh + 1, :gw + 1]
    d = distance_transform_edt(np.pad(inner, 1))[1:-1, 1:-1]
    d = gaussian_filter(d, 1.0) * inner
    dn = np.clip(d / max(d.max(), 1e-9), 0, 1)
    prof = np.sqrt(np.clip(1 - (1 - dn) ** 2, 0, 1))                         # circular "pillow" profile, 0 at the outline
    prof = gaussian_filter(prof, 0.8) * inner

    # --- 3. height field -> thickness --- (earlier lines computed the distance map `d` and its pillow profile `prof`)
    unit = 1.0 / gh                                                           # world size of one cell (figure is 1 tall)
    half = depth * min(gw, gh) * unit * 1.0 + 0.02
    z = prof * half
    rr, cc = np.mgrid[0:gh + 1, 0:gw + 1]
    X = (cc - gw / 2) * unit; Y = (gh - rr) * unit
    # --- 5. normals from the slope of z (np.gradient = finite differences) ---
    gy, gx = np.gradient(z, unit)                                             # dz/dY is -dz/drow; handle sign below
    nx, ny = -gx, gy                                                          # front normal (-dz/dx, -dz/dY, 1)
    nrm_f = np.stack([nx, ny, np.ones_like(z)], -1); nrm_f /= np.linalg.norm(nrm_f, axis=-1, keepdims=True)
    nrm_b = np.stack([nx, ny, -np.ones_like(z)], -1); nrm_b /= np.linalg.norm(nrm_b, axis=-1, keepdims=True)
    nrm_b[..., :2] *= -1                                                      # back surface mirrors the slope
    # --- 6. texture coordinates: column/width, row/height ---
    U = cc / gw; Vv = rr / gh

    # --- 4. vertices + triangles (front surface, then back surface with reversed winding) ---
    verts, norms, uvs, idx = [], [], [], []
    base = (gh + 1) * (gw + 1)
    flat = lambda A: A.reshape(-1, A.shape[-1]) if A.ndim == 3 else A.reshape(-1)
    verts.append(np.stack([X, Y, z], -1).reshape(-1, 3)); verts.append(np.stack([X, Y, -z], -1).reshape(-1, 3))
    norms.append(flat(nrm_f)); norms.append(flat(nrm_b))
    uvs.append(np.stack([U, Vv], -1).reshape(-1, 2)); uvs.append(np.stack([1 - U, Vv], -1).reshape(-1, 2))   # back is the mirrored picture
    r, c = np.nonzero(a)
    n00 = r * (gw + 1) + c; n01 = n00 + 1; n10 = n00 + (gw + 1); n11 = n10 + 1
    idx.append(np.concatenate([np.stack([n00, n10, n01], 1), np.stack([n01, n10, n11], 1)]))            # front (CCW seen from +Z)
    idx.append(base + np.concatenate([np.stack([n00, n01, n10], 1), np.stack([n01, n11, n10], 1)]))     # back (reversed)
    V = np.concatenate(verts); N = np.concatenate(norms); T = np.concatenate(uvs); F = np.concatenate(idx).astype(np.uint32)
    used = np.unique(F); remap = np.full(len(V), -1, np.int64); remap[used] = np.arange(len(used))
    V, N, T, F = V[used], N[used], T[used], remap[F].astype(np.uint32)
    V[:, 1] -= V[:, 1].min(); V[:, 0] -= (V[:, 0].min() + V[:, 0].max()) / 2           # stand on y=0, centred

    # --- 7. pack into a .glb: positions, normals, uvs, indices and the PNG texture become bufferViews ---
    chunks = []
    def view(data, target=None):
        pad = (-sum(len(x) for x in chunks)) % 4
        if pad: chunks.append(b"\0" * pad)
        off = sum(len(x) for x in chunks); chunks.append(data)
        return {"buffer": 0, "byteOffset": off, "byteLength": len(data), **({"target": target} if target else {})}
    views = [view(V.astype("<f4").tobytes(), 34962), view(N.astype("<f4").tobytes(), 34962), view(T.astype("<f4").tobytes(), 34962),
             view(F.tobytes(), 34963), view(png)]
    body = b"".join(chunks); body += b"\0" * (-len(body) % 4)
    js = {"asset": {"version": "2.0", "generator": "relief.py"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0, "name": "relief"}],
          "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "NORMAL": 1, "TEXCOORD_0": 2}, "indices": 3, "material": 0}]}],
          "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}, "metallicFactor": 0, "roughnessFactor": 0.7},
                         "alphaMode": "MASK", "alphaCutoff": 0.5, "doubleSided": True}],
          "textures": [{"source": 0, "sampler": 0}], "images": [{"bufferView": 4, "mimeType": "image/png"}],
          "samplers": [{"magFilter": 9729, "minFilter": 9987, "wrapS": 33071, "wrapT": 33071}],
          "buffers": [{"byteLength": len(body)}], "bufferViews": views,
          "accessors": [{"bufferView": 0, "componentType": 5126, "count": len(V), "type": "VEC3", "min": V.min(0).tolist(), "max": V.max(0).tolist()},
                        {"bufferView": 1, "componentType": 5126, "count": len(N), "type": "VEC3"},
                        {"bufferView": 2, "componentType": 5126, "count": len(T), "type": "VEC2"},
                        {"bufferView": 3, "componentType": 5125, "count": int(F.size), "type": "SCALAR"}]}
    j = json.dumps(js, separators=(",", ":")).encode(); j += b" " * (-len(j) % 4)
    data = struct.pack("<4sII", b"glTF", 2, 28 + len(j) + len(body)) + struct.pack("<I4s", len(j), b"JSON") + j + struct.pack("<I4s", len(body), b"BIN\0") + body
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True); out_path.write_bytes(data)
    return out_path
