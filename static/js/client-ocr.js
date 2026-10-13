/* client-ocr.js - reads the card's HP and attacks IN THE BROWSER with tesseract.js (https://github.com/naptha/tesseract.js).
 *
 * WHEN THIS RUNS
 *   Normally the server reads the card with the Tesseract program (card_reader.py). If that program is not installed there,
 *   the scan result carries  stats.client_ocr = <token>  and the Battle page calls readCardInBrowser(token) instead:
 *
 *     1. download the two small pictures the server prepared:  /api/battle/ocr/<token>/hp.png  and  attack.png
 *        (the server already straightened the card, cropped the HP corner and the attack area and boosted the contrast)
 *     2. run tesseract.js on them. It is the same Tesseract engine compiled to WebAssembly, so this uses the PHONE's CPU
 *        inside a Web Worker (it does not block the page). First use downloads ~10 MB (worker, WASM core, English data)
 *        from the CDN; the browser caches it afterwards.
 *     3. send the recognised WORDS (text + position + size) to /api/battle/ocr-words. The server turns words into
 *        HP / attacks / damage with the same parser as the server-side reader (card_reader.parse_attack_lines), caches the
 *        result on disk, and returns the finished stats.
 *   Keeping the parsing on the server means both readers behave identically and the rules live in one place.
 */
let workerPromise = null;

/** Convert tesseract.js output to the word shape card_reader.py expects: {text,left,top,width,height,conf,line:[b,p,l]}. */
export function wordsFromResult(data) {
  const out = [];
  let lines = data.lines;
  if (!lines && data.blocks) lines = data.blocks.flatMap((b) => b.paragraphs.flatMap((p) => p.lines));      // newer tesseract.js nests them
  (lines || []).forEach((ln, li) => {
    (ln.words || []).forEach((w) => {
      if (!w.text || !w.text.trim()) return;
      const b = w.bbox;
      out.push({ text: w.text.trim(), left: b.x0, top: b.y0, width: Math.max(1, b.x1 - b.x0), height: Math.max(1, b.y1 - b.y0), conf: w.confidence, line: [0, 0, li] });
    });
  });
  return out;
}

async function getWorker(loadScript, urls) {
  if (!workerPromise) {
    workerPromise = (async () => {
      if (!window.Tesseract) await loadScript(urls);
      const w = await window.Tesseract.createWorker("eng");                    // downloads worker + core + eng data on first use
      await w.setParameters({ preserve_interword_spaces: "1" });
      return w;
    })();
    workerPromise.catch(() => { workerPromise = null; });                       // allow a retry after a failure
  }
  return workerPromise;
}

/** Whole round trip. Returns the server's finished stats object. Throws with a readable message on failure. */
export async function readCardInBrowser(token, { loadScript, urls, timeoutMs = 60000 }) {
  const work = (async () => {
    const worker = await getWorker(loadScript, urls);
    const recognise = async (part, psm) => {
      await worker.setParameters({ tessedit_pageseg_mode: String(psm) });
      const { data } = await worker.recognize(`/api/battle/ocr/${token}/${part}.png`, {}, { blocks: true });
      return wordsFromResult(data);
    };
    const hp = [await recognise("hp", 7)];
    hp.push(await recognise("hp", 6));                                           // second attempt as a block; the server uses the first that parses
    const attack = await recognise("attack", 6);
    const r = await fetch("/api/battle/ocr-words", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ token, hp, attack }) });
    const d = await r.json();
    if (!d.ok) throw new Error(d.error || "the server could not use the OCR result");
    return d.stats;
  })();
  const timeout = new Promise((_, rej) => setTimeout(() => rej(new Error("reading the card in the browser took too long")), timeoutMs));
  return Promise.race([work, timeout]);
}
