// A tiny stand-in for three.js, only for tests/browser_game.py (the real library is a CDN download).
// It implements just enough scene-graph behaviour for detect.js, and records every "emissive flash" in window.__flash.
export class Vector3 { constructor(x = 0, y = 0, z = 0) { this.x = x; this.y = y; this.z = z; }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; } copy(v) { return this.set(v.x, v.y, v.z); } clone() { return new Vector3(this.x, this.y, this.z); }
  lerp(v, t) { return this.set(this.x + (v.x - this.x) * t, this.y + (v.y - this.y) * t, this.z + (v.z - this.z) * t); }
  setScalar(s) { return this.set(s, s, s); } project() { this.x = this.x * 0.1; this.y = this.y * 0.1; return this; } applyMatrix4() { return this; } }
export class Quaternion { constructor() { this.x = 0; this.y = 0; this.z = 0; this.w = 1; } copy() { return this; } slerp() { return this; } }
export class Matrix4 { constructor() { this.e = []; } fromArray(a) { this.e = a; return this; } decompose(p, q, s) { p.set(0, 0, -3); s.set(1, 1, 1); return this; } copy() { return this; } invert() { return this; } }
export class Color { constructor(c) { this.r = this.g = this.b = 0; if (c !== undefined) this.set(c); }
  set(c) { if (typeof c === "string") { const n = parseInt(c.slice(1), 16); this.r = (n >> 16) / 255; this.g = ((n >> 8) & 255) / 255; this.b = (n & 255) / 255; } return this; }
  copy(c) { this.r = c.r; this.g = c.g; this.b = c.b; if (this.__em && (c.r || c.g || c.b)) (window.__flash = window.__flash || []).push([+c.r.toFixed(2), +c.g.toFixed(2), +c.b.toFixed(2)]); return this; }
  clone() { const c = new Color(); c.r = this.r; c.g = this.g; c.b = this.b; return c; } lerp(c, t) { this.r += (c.r - this.r) * t; return this; } }
class Mat { constructor() { this.color = new Color(); this.emissive = new Color(); this.emissive.__em = true; this.emissiveIntensity = 0; }
  clone() { return new Mat(); } dispose() {} }
export class MeshBasicMaterial extends Mat { constructor() { super(); } }
export class PlaneGeometry { dispose() {} }
export class CanvasTexture { constructor(c) { this.image = c; } dispose() {} }
export class SpriteMaterial extends Mat { constructor(o) { super(); Object.assign(this, o || {}); } dispose() {} }
export class Object3D {
  constructor() { this.children = []; this.parent = null; this.visible = true; this.position = new Vector3(); this.quaternion = new Quaternion();
    this.rotation = { x: 0, y: 0, z: 0 }; this.scale = new Vector3(1, 1, 1); this.matrixAutoUpdate = true; }
  add(...o) { for (const c of o) { c.removeFromParent(); c.parent = this; this.children.push(c); } return this; }
  remove(c) { const i = this.children.indexOf(c); if (i >= 0) { this.children.splice(i, 1); c.parent = null; } return this; }
  clear() { this.children.forEach((c) => (c.parent = null)); this.children = []; return this; }
  removeFromParent() { this.parent && this.parent.remove(this); return this; }
  traverse(f) { f(this); this.children.forEach((c) => c.traverse(f)); }
  localToWorld(v) { return v; } worldToLocal(v) { return v; } getWorldPosition(v) { return v.set(0, 0, 0); } updateMatrixWorld() {}
}
export class Group extends Object3D {}
export class Sprite extends Object3D { constructor(m) { super(); this.material = m; this.isSprite = true; window.__sprites = (window.__sprites || 0) + 1; } }
export class Scene extends Object3D { constructor() { super(); window.__scene = this; } }
export class Mesh extends Object3D { constructor(g, m) { super(); this.isMesh = true; this.geometry = g || new PlaneGeometry(); this.material = m || new Mat(); } }
export class PerspectiveCamera extends Object3D { constructor() { super(); this.projectionMatrix = new Matrix4(); this.projectionMatrixInverse = new Matrix4(); } updateProjectionMatrix() {} }
export class HemisphereLight extends Object3D {}
export class DirectionalLight extends Object3D {}
export class AnimationMixer { constructor() {} clipAction() { return { play() {} }; } update() {} stopAllAction() {} }
export class Box3 { setFromObject() { return this; } getSize(v) { return v.set(1, 1, 1); } getCenter(v) { return v.set(0, 0.5, 0); } get min() { return new Vector3(); } }
export class WebGLRenderer { constructor() { this.raf = null; } setClearColor() {} setPixelRatio() {} setSize() {} clear() {} render() { window.__renders = (window.__renders || 0) + 1; }
  setAnimationLoop(fn) { cancelAnimationFrame(this.raf); const loop = (t) => { if (fn) { fn(t); this.raf = requestAnimationFrame(loop); } }; if (fn) this.raf = requestAnimationFrame(loop); } }
