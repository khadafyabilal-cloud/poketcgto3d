// rating.js - the thumbs up / thumbs down panel shared by the scan page and the camera pages.
//
// WHAT IT DOES
//   mountRating(el, result, onNewResult) fills `el` with two questions about the answer the server just gave:
//     1. "Right Pokémon?"   (detection)  thumbs-down opens a box to type what the card really is.
//     2. "Good 3D model?"   (model)
//   Each tap POSTs {result_id, client, target, vote, correct} to /api/feedback. The server answers with
//   {ok, stats, result}; (onRetry, if given, runs after a bare thumbs-down so the page can scan the card again); when `result` is not null (a correction, or a model that was just retired because of
//   too many thumbs-downs) we call onNewResult(result) so the page can swap to it at once.
//
// PRIVACY: `client` is a random id kept in this browser's localStorage so one person counts once. It is not
// linked to a name, account or device. Nothing is sent until a thumb is tapped.
//
// This file only builds buttons and sends small JSON messages; the "learning" happens on the server (feedback.py).

const STYLE_ID = "rating-style";
const CSS = `
.rate{margin-top:14px;display:grid;gap:10px}
.rate .q{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.rate .q span.t{flex:1 1 130px;color:var(--fg,#eee);font-size:14px}
.rate button.th{min-width:46px;min-height:44px;border-radius:12px;border:1px solid var(--line,#394066);background:transparent;color:var(--fg,#eee);font-size:20px;cursor:pointer;touch-action:manipulation}
.rate button.th[aria-pressed=true]{border-color:var(--yellow,#ffcb05);background:#ffcb0526}
.rate button.th:focus-visible,.rate input:focus-visible,.rate button.go:focus-visible{outline:2px solid var(--yellow,#ffcb05);outline-offset:2px}
.rate .fix{display:flex;gap:8px;flex-wrap:wrap}
.rate .fix input{flex:1 1 150px;min-height:44px;padding:0 12px;border-radius:12px;border:1px solid var(--line,#394066);background:#0b0d1a;color:var(--fg,#eee);font-size:16px}
.rate button.go{min-height:44px;padding:0 16px;border-radius:12px;border:0;background:var(--yellow,#ffcb05);color:#1a1a1a;font-weight:700;cursor:pointer}
.rate .msg{font-size:13px;color:var(--mut,#9aa3c7);min-height:1.2em}
`;

/** Anonymous id for "one person, one vote". Falls back to a per-page id when storage is blocked. */
let memoryId = null;
export function clientId() {
  try {
    let id = localStorage.getItem("pc3d_client");
    if (!id) { id = (crypto.randomUUID ? crypto.randomUUID() : String(Math.random()).slice(2) + Date.now()); localStorage.setItem("pc3d_client", id); }
    return id;
  } catch { return memoryId || (memoryId = "m" + Math.random().toString(36).slice(2)); }
}

/** Send one vote. Returns the parsed JSON ({ok, stats, result} or {ok:false, error}). */
export async function sendVote(payload) {
  const r = await fetch("/api/feedback", { method: "POST", headers: { "Content-Type": "application/json" },
                                           body: JSON.stringify({ client: clientId(), ...payload }) });
  return r.json().catch(() => ({ ok: false, error: "Bad answer from the server" }));
}

function el(tag, attrs = {}, text) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
}

export function mountRating(host, result, onNewResult, onRetry) {
  if (!document.getElementById(STYLE_ID)) { const s = el("style", { id: STYLE_ID }); s.textContent = CSS; document.head.appendChild(s); }
  host.replaceChildren();
  if (!result || !result.result_id) return;                       // nothing to rate (no model was shown)
  const rid = result.result_id, wrap = el("div", { class: "rate" });
  const msg = el("div", { class: "msg", role: "status", "aria-live": "polite" });
  const pressed = {};                                              // target -> vote the user chose

  function question(target, label, upLabel, downLabel) {
    const row = el("div", { class: "q" });
    row.appendChild(el("span", { class: "t" }, label));
    const up = el("button", { type: "button", class: "th", "aria-label": upLabel, "aria-pressed": "false" }, "👍");
    const down = el("button", { type: "button", class: "th", "aria-label": downLabel, "aria-pressed": "false" }, "👎");
    row.append(up, down);
    const mark = v => { up.setAttribute("aria-pressed", String(v > 0)); down.setAttribute("aria-pressed", String(v < 0)); };
    up.onclick = () => vote(target, 1, null, mark);
    down.onclick = () => { if (target === "detection") { fix.hidden = false; fixInput.focus(); mark(-1); pressed.detection = -1; } else vote(target, -1, null, mark); };
    return row;
  }

  async function vote(target, v, correct, mark) {
    msg.textContent = "Saving…";
    const out = await sendVote({ result_id: rid, target, vote: v, correct });
    if (!out.ok) { msg.textContent = out.error || "Could not save your vote."; return; }
    pressed[target] = v; mark(v);
    if (target === "detection" && v > 0) fix.hidden = true;
    if (out.result && out.result.model_url && onNewResult) {
      msg.textContent = target === "model" ? "Thanks - switched to a different model." : "Thanks - corrected.";
      onNewResult(out.result);
    } else if (target === "detection" && v < 0 && !correct && onRetry) {
      msg.textContent = "Thanks - looking at the card again…";
      onRetry();                                                   // camera pages: scan this card again, skipping the rejected guess
    } else {
      msg.textContent = v > 0 ? "Thanks - saved." : (target === "detection" ? "Thanks - this card won't be read that way again." : "Thanks - saved. Enough thumbs-down and a different model is used.");
    }
  }

  const q1 = question("detection", "Is this the right Pokémon?", "Yes, right Pokémon", "No, wrong Pokémon");
  const q2 = question("model", "Is the 3D model good?", "Good 3D model", "Bad 3D model");
  const fix = el("form", { class: "fix", hidden: "" });
  const fixInput = el("input", { type: "text", placeholder: "What is it? (optional) e.g. pikachu or 25", "aria-label": "Correct Pokémon name or number", autocomplete: "off", maxlength: "40" });
  const go = el("button", { type: "submit", class: "go" }, "Send thumbs-down");
  fix.append(fixInput, go);
  fix.onsubmit = ev => { ev.preventDefault(); const v = fixInput.value.trim();
    const [up, down] = q1.querySelectorAll("button"); vote("detection", -1, v || null, x => { up.setAttribute("aria-pressed", "false"); down.setAttribute("aria-pressed", "true"); }); };
  wrap.append(q1, fix, q2, msg);
  host.appendChild(wrap);
}
