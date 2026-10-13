// Stand-in for three/addons/controls/OrbitControls.js (tests/browser_game.py): the viewer only needs these members.
import { Vector3 } from "https://cdn.jsdelivr.net/npm/three@0.160.0/build/three.module.js";
export class OrbitControls { constructor(cam, el) { this.target = new Vector3(); this.autoRotate = false; window.__orbit = this; } update() {} reset() {} saveState() {} addEventListener() {} }
