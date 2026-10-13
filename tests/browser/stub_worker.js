// Stands in for tracker.worker.js (OpenCV.js): reports one card whenever /__stub says it is present.
let n = 0;
onmessage = async (e) => {
  const m = e.data;
  if (m.type === "init") return postMessage({ type: "ready", from: "stub" });
  if (m.type !== "frame") return;
  const s = await (await fetch("/__stub")).json();
  const tracks = s.present ? [{ id: s.id, visible: true, stable: ++n > 3, corners: [[.35, .2], [.65, .2], [.65, .8], [.35, .8]].map(([x, y]) => [x * m.srcW, y * m.srcH]),
                               modelView: [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, -3, 1] }] : [];
  if (!s.present) n = 0;
  postMessage({ type: "result", tracks, ms: 5, mask: null });
};
