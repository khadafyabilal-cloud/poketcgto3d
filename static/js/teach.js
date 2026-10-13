/* teach.js - the Detect page's RESULTS panel and its "Teach the agent" form.
 * =====================================================================================================================
 * WHAT THE PLAYER SEES
 *   - what the camera saw (the frame with the card's outer border drawn on it) and the straightened card the agent read;
 *   - the Pokémon, set/number, types, the HP and every attack the agent read (name, damage, type), where each fact came
 *     from (read from the card text / learned from players / corrected by a player / default), and how the agent decided;
 *   - 👍 / 👎 on the detection and on the 3D model (rating.js);
 *   - "Teach the agent": a form to correct the Pokémon, the HP and the attacks. POST /api/feedback/correct.
 *
 * HOW IT TALKS TO THE REST OF THE PAGE (hooks, so this file knows nothing about cameras or three.js)
 *   hooks.beforeEdit()          called the moment the player starts correcting something. detect.js switches AR OFF
 *                               there, to free the camera + tracker while they type.
 *   hooks.onNewResult(result)   the Pokémon changed (a correction or a rating): show the new answer + its 3D model.
 *   hooks.onRetry()             a bare thumbs-down: scan the card again.
 *   hooks.onSaved(info)         HP / attacks were saved: info.stats now holds them.
 *
 * Everything the player types goes to the SERVER, which validates it again (agent.clean_attacks / clean_hp): the form's
 * limits here (maxlength, min/max) are only to help the person, never a safety measure.
 */
import { clientId, mountRating } from "/static/js/rating.js";

const TYPES = ["normal", "fire", "water", "electric", "grass", "ice", "fighting", "poison", "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy"];
const TYPE_COLORS = { normal: "#c9c5b2", fire: "#ff6a2b", water: "#3d9bff", electric: "#ffe030", grass: "#4fd05a", ice: "#7fe6f2", fighting: "#d6482f", poison: "#b45bd8",
  ground: "#d8b055", flying: "#9db7ff", psychic: "#ff5fa8", bug: "#a8cf2a", rock: "#b8a15a", ghost: "#7a62c9", dragon: "#6f50ff", dark: "#7a665a", steel: "#a9b8c9", fairy: "#ff9fd8" };
const MAX_ATTACKS = 4;
const SOURCE_LABEL = {
  ocr: "Read from the card's text", user: "Corrected by a player", default: "Default stats (the text couldn't be read)",
};

const CSS = `
#res h2{margin:0;font-size:20px}#res h3{margin:18px 0 6px;font-size:15px}
#res .empty{color:var(--mut,#a4abc9);padding:18px 4px}
#res .sub{color:var(--mut,#a4abc9);margin-top:2px}
#res .chips{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
#res .chip{padding:2px 10px;border-radius:99px;background:#232948;font-size:12.5px;text-transform:capitalize}
#res .chip.good{background:#ffcb0533;color:#ffe27a}
#res .shots{display:flex;gap:10px;margin-top:12px;align-items:flex-start}
#res figure{margin:0;flex:1 1 0;min-width:0}#res figure img{width:100%;max-height:230px;object-fit:contain;border-radius:10px;background:#000;display:block}
#res figcaption{font-size:12px;color:var(--mut,#a4abc9);margin-top:4px}
#res .hpline{font-weight:700;font-size:16px}
#res ul.atk{list-style:none;margin:6px 0 0;padding:0;display:grid;gap:6px}
#res ul.atk li{position:relative;padding:7px 10px 7px 16px;border-radius:10px;background:#181c30;border:1px solid var(--line,#2a3050);display:flex;justify-content:space-between;gap:10px;align-items:baseline}
#res ul.atk li::before{content:"";position:absolute;left:0;top:0;bottom:0;width:6px;border-radius:10px 0 0 10px;background:var(--tc,#c9c5b2)}
#res ul.atk small{color:var(--mut,#a4abc9)}
#res .hint{color:var(--mut,#a4abc9);font-size:13px;margin:6px 0 0}
#res details{margin-top:12px;color:var(--mut,#a4abc9);font-size:13px}#res details ol{margin:6px 0 0;padding-left:18px}#res details b{color:var(--fg,#f1f3fb)}
#res button,#res select,#res input{font:inherit;color:inherit}
#res .primary{min-height:46px;padding:0 18px;border-radius:12px;border:0;background:var(--yellow,#ffcb05);color:#1a1a1a;font-weight:700;cursor:pointer;margin-top:14px}
#res .ghost{min-height:46px;padding:0 16px;border-radius:12px;background:transparent;border:1px solid var(--line,#2a3050);cursor:pointer}
#teach{margin-top:14px;padding:14px;border-radius:14px;border:1px solid var(--yellow,#ffcb05);background:#10142a}
#teach label{display:block;margin:10px 0 4px;color:var(--mut,#a4abc9);font-size:13px}
#teach input,#teach select{width:100%;min-height:44px;padding:0 12px;border-radius:10px;border:1px solid var(--line,#2a3050);background:#181c30;font-size:16px}
#teach fieldset{border:0;margin:12px 0 0;padding:0}#teach legend{color:var(--mut,#a4abc9);font-size:13px;padding:0}
#teach .arow{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(0,1fr) minmax(0,1.1fr) 44px;gap:6px;margin-top:6px}
#teach .arow .ax{min-height:44px;border-radius:10px;border:1px solid var(--line,#2a3050);background:transparent;cursor:pointer}
@media(max-width:520px){#teach .arow{grid-template-columns:1fr 1fr 44px}#teach .arow .an{grid-column:1/-1}}
#teach .btns{display:flex;gap:8px;margin-top:14px;flex-wrap:wrap}#teach .btns .primary{margin-top:0;flex:1}
#tMsg{min-height:1.3em;margin-top:8px;font-size:14px}#tMsg.bad{color:var(--bad,#ff8a8a)}#tMsg.ok{color:#9be59b}
#res :focus-visible{outline:2px solid var(--yellow,#ffcb05);outline-offset:2px}
`;

const el = (tag, attrs = {}, text) => {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
  if (text != null) e.textContent = text;
  return e;
};
const dmgText = (m) => (m.damage ? `${m.damage}${m.modifier || ""}` : "no damage");
const lc = (s) => String(s || "").trim().toLowerCase();

/** Which stats does the player see/edit? Placeholder "default" attacks are NOT prefilled: they were never on the card. */
export const isRealStats = (s) => !!s && (s.source === "ocr" || s.source === "user");

export function createPanel(root, hooks = {}) {
  if (!document.getElementById("teach-style")) { const s = el("style", { id: "teach-style" }); s.textContent = CSS; document.head.appendChild(s); }
  root.innerHTML = `
    <div class="empty" id="resEmpty">No card yet. Aim the camera at a card, or take / upload a photo of one.</div>
    <div id="resBody" hidden>
      <h2 id="rName"></h2><div class="sub" id="rSub"></div><div class="chips" id="rChips"></div>
      <div class="shots"><figure><img id="shotFrame" alt="The camera picture with the detected card border"><figcaption>What the camera saw (outer border found)</figcaption></figure>
        <figure><img id="shotCard" alt="The straightened card"><figcaption>The card the agent read</figcaption></figure></div>
      <h3>HP and attacks the agent read</h3>
      <div class="hpline" id="rHp"></div><ul class="atk" id="rAtk"></ul><p class="hint" id="rNotes"></p>
      <details id="how"><summary>How the agent decided</summary><ol id="rTrace"></ol></details>
      <div id="rateBox"></div>
      <button type="button" class="primary" id="teachOpen">Teach the agent</button>
      <form id="teach" hidden novalidate>
        <h3 style="margin-top:0">Teach the agent</h3>
        <p class="hint">Fix anything it got wrong. What you save is remembered for this card: next time its HP and attacks come from you, not from reading the text.</p>
        <label for="tPokemon">Pokémon (name or Pokédex number)</label><input id="tPokemon" type="text" autocomplete="off" maxlength="40">
        <label for="tHp">HP (the number top right)</label><input id="tHp" type="number" inputmode="numeric" min="10" max="500">
        <fieldset id="tAtk"><legend>Attacks, as printed under the picture (leave damage blank if the attack has no number)</legend></fieldset>
        <div class="btns"><button type="button" class="ghost" id="tAdd">Add attack</button></div>
        <div class="btns"><button type="submit" class="primary" id="tSave">Save and teach</button><button type="button" class="ghost" id="tCancel">Cancel</button></div>
        <div id="tMsg" role="status" aria-live="polite"></div>
      </form>
    </div>`;
  const $ = (id) => root.querySelector("#" + id);
  let rec = null, orig = null;

  function atkRow(m = {}) {
    const row = el("div", { class: "arow" });
    const name = el("input", { class: "an", type: "text", placeholder: "Attack name", maxlength: "30", "aria-label": "Attack name", autocomplete: "off" });
    const dmg = el("input", { class: "ad", type: "text", inputmode: "numeric", placeholder: "Damage", maxlength: "5", "aria-label": "Damage (blank = none)", autocomplete: "off" });
    const type = el("select", { class: "at", "aria-label": "Attack type" });
    type.append(el("option", { value: "" }, "type: auto"), ...TYPES.map((t) => el("option", { value: t }, t)));
    const rm = el("button", { type: "button", class: "ax", "aria-label": "Remove this attack" }, "×");
    name.value = m.name || ""; dmg.value = m.damage ? `${m.damage}${m.modifier || ""}` : ""; type.value = m.type || "";
    rm.onclick = () => { row.remove(); $("tAdd").disabled = false; };
    row.append(name, dmg, type, rm);
    return row;
  }
  const rows = () => [...$("tAtk").querySelectorAll(".arow")];
  const readRows = () => rows().map((r) => ({ name: r.querySelector(".an").value.trim(), damage: r.querySelector(".ad").value.trim(), type: r.querySelector(".at").value })).filter((a) => a.name);

  function fillForm() {
    const info = rec.info, p = info.pokemon || {}, st = info.stats || {};
    $("tPokemon").value = p.name || (info.card && info.card.name) || "";
    $("tHp").value = isRealStats(st) && st.hp ? st.hp : "";
    $("tAtk").querySelectorAll(".arow").forEach((r) => r.remove());
    const list = isRealStats(st) ? st.attacks || [] : [];
    (list.length ? list : [{}, {}]).slice(0, MAX_ATTACKS).forEach((m) => $("tAtk").appendChild(atkRow(m)));
    $("tAdd").disabled = rows().length >= MAX_ATTACKS;
    orig = { pokemon: lc($("tPokemon").value), hp: $("tHp").value, attacks: JSON.stringify(readRows().map((a) => [lc(a.name), a.damage.replace(/\s/g, ""), a.type])), real: isRealStats(st) };
    setMsg("");
  }
  function setMsg(text, kind) { const m = $("tMsg"); m.textContent = text; m.className = kind || ""; }

  function openTeach() {
    if (!rec) return;
    hooks.beforeEdit && hooks.beforeEdit();                  // AR off: the camera and tracker stop while the player types
    fillForm(); $("teach").hidden = false; $("teachOpen").hidden = true;
    $("tPokemon").focus({ preventScroll: false });
  }
  $("teachOpen").onclick = openTeach;
  $("teach").addEventListener("focusin", () => hooks.beforeEdit && hooks.beforeEdit());      // any field touched = the player is correcting
  $("tCancel").onclick = () => { $("teach").hidden = true; $("teachOpen").hidden = false; };
  $("tAdd").onclick = () => { if (rows().length < MAX_ATTACKS) $("tAtk").appendChild(atkRow()); $("tAdd").disabled = rows().length >= MAX_ATTACKS; };

  $("teach").onsubmit = async (ev) => {
    ev.preventDefault();
    if (!rec || !rec.info.result_id) return setMsg("This answer has expired - scan the card again, then teach it.", "bad");
    const body = { result_id: rec.info.result_id, client: clientId() };
    const poke = $("tPokemon").value.trim();
    if (poke && lc(poke) !== orig.pokemon) body.pokemon = poke;
    const hp = $("tHp").value.trim();
    if (hp && (hp !== String(orig.hp) || !orig.real)) body.hp = hp;
    const atk = readRows();
    const key = JSON.stringify(atk.map((a) => [lc(a.name), a.damage.replace(/\s/g, ""), a.type]));
    if (atk.length && (key !== orig.attacks || !orig.real)) body.attacks = atk;
    if (!body.pokemon && !body.hp && !body.attacks) return setMsg("Change the Pokémon, the HP or an attack first.", "bad");
    $("tSave").disabled = true; setMsg("Saving…");
    let out;
    try {
      const r = await fetch("/api/feedback/correct", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      out = await r.json();
    } catch (err) { out = { ok: false, error: "Couldn't reach the server: " + err.message }; }
    $("tSave").disabled = false;
    if (!out.ok) return setMsg(out.error || "Could not save.", "bad");
    const oldInfo = rec.info;
    if (out.result && out.result.model_url) {                // the Pokémon changed: a replacement answer (+ its model) came back
      out.result.stats = out.stats || out.result.stats || oldInfo.stats;
      out.result.card = oldInfo.card;
      hooks.onNewResult && (await hooks.onNewResult(out.result, oldInfo));
    } else if (out.stats) {
      oldInfo.stats = out.stats;
      hooks.onSaved && hooks.onSaved(oldInfo);
      show({ ...rec, info: oldInfo });
    }
    $("teach").hidden = false; $("teachOpen").hidden = true; fillForm();      // the form now shows what was saved
    setMsg("Saved. The agent will use this for this card from now on.", "ok");
  };

  function show(next) {
    rec = next;
    const info = rec && rec.info;
    $("resEmpty").hidden = !!info; $("resBody").hidden = !info;
    if (!info) return;
    const p = info.pokemon || {}, c = info.card || {}, st = info.stats || {};
    $("rName").textContent = p.name ? p.name[0].toUpperCase() + p.name.slice(1) : c.name || (info.energy && info.energy.name) || "Unknown card";
    $("rSub").textContent = [c.name && p.name && lc(c.name) !== lc(p.name) ? c.name : "", c.set, c.number ? "#" + c.number : ""].filter(Boolean).join(" · ") || (rec.scan && rec.scan.detected === false ? "No card border found - the whole photo was read" : "");
    const chips = $("rChips"); chips.replaceChildren();
    (p.types || []).forEach((t) => chips.append(el("span", { class: "chip", style: `border-left:4px solid ${TYPE_COLORS[t] || "#888"}` }, t)));
    if (info.energy) chips.append(el("span", { class: "chip" }, `Energy · ${info.energy.type || "any type"}`));
    if (info.learned) chips.append(el("span", { class: "chip good" }, "Learned from players"));
    else if (info.cached) chips.append(el("span", { class: "chip" }, "From the card memory"));
    if (info.model_source) chips.append(el("span", { class: "chip" }, `3D: ${info.model_source.kind}`));
    const set = (id, src) => { const i = $(id); if (src) { i.src = src; i.parentElement.hidden = false; } else i.parentElement.hidden = true; };
    set("shotFrame", rec.frame); set("shotCard", rec.card || info.crop_jpeg);
    const hasStats = !!(info.stats && info.pokemon);
    $("rHp").textContent = hasStats ? `HP ${st.hp}` : info.energy ? "Energy cards have no HP or attacks." : "";
    const ul = $("rAtk"); ul.replaceChildren();
    for (const m of hasStats ? st.attacks || [] : []) {
      const li = el("li"); li.style.setProperty("--tc", TYPE_COLORS[m.type] || "#c9c5b2");
      li.append(el("b", {}, m.name), el("small", {}, `${dmgText(m)} · ${m.type}`));
      ul.append(li);
    }
    const notes = [SOURCE_LABEL[st.source], ...(st.notes || [])].filter(Boolean);
    $("rNotes").textContent = hasStats ? notes.join(" · ") : "";
    const tr = $("rTrace"); tr.replaceChildren();
    for (const t of info.trace || []) { const li = el("li"); li.append(el("b", {}, t.step), ` (${t.t}s) - ${t.detail}`); tr.append(li); }
    $("how").hidden = !(info.trace && info.trace.length);
    $("teachOpen").disabled = !info.pokemon;
    $("teachOpen").hidden = !$("teach").hidden;
    const box = $("rateBox"); box.replaceChildren();
    if (info.result_id) mountRating(box, info, (d) => hooks.onNewResult && hooks.onNewResult(d, info), () => hooks.onRetry && hooks.onRetry());
  }
  return { show, el: root, current: () => rec, openTeach, hide() { $("teach").hidden = true; } };
}
