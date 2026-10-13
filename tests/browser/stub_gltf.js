import { Group, Mesh } from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
export class GLTFLoader { setDRACOLoader() { return this; } async loadAsync(url) { window.__loads = (window.__loads || []); window.__loads.push(url);
  const g = new Group(); g.add(new Mesh()); return { scene: g, animations: [] }; } }
export class DRACOLoader { setDecoderPath() { return this; } }
export const clone = (o) => { const g = new Group(); o.traverse((c) => { if (c.isMesh) g.add(new Mesh(c.geometry, c.material.clone())); }); return g; };
