/* battle-engine.js - the rules of the mock Pokémon battle. No camera, no 3D, no page: just numbers, so it can be read
 * (and tested with Node) on its own.  detect.js drives it and draws the results.
 *
 * THE RULES (deliberately simple - this is a mock battle)
 *   - Each side has HP. The player's HP is the number printed in the top-right corner of the card (read by the server:
 *     card_reader.py). The opponent's HP is random.
 *   - A move has a name, a TYPE and a damage number. The damage is the number printed next to the attack on the card,
 *     rounded to a multiple of 5 (see damageOf). A move with no number is NON-DAMAGING: it does nothing, and the player is told so.
 *   - Turns alternate: player picks a move, then the opponent answers with a random move. The OPPONENT's damage is
 *     random too: 5, 10, 15 ... 40 (OPPONENT_DAMAGE). Every point of damage in the game is a multiple of 5.
 *   - The side that reaches 0 HP faints and the battle ends.
 *   - No type chart, accuracy or crits: the number on the card is the damage. (Weakness/resistance printed on cards
 *     could be added later; the same goes for healing with Energy cards - the server already reports their type.)
 */

/* Colour each type "lights up" in when it hits (also used for the move buttons). Keys are PokeAPI type names. */
export const TYPE_COLORS = {
  normal: "#c9c5b2", fire: "#ff6a2b", water: "#3d9bff", electric: "#ffe030", grass: "#4fd05a", ice: "#7fe6f2",
  fighting: "#d6482f", poison: "#b45bd8", ground: "#d8b055", flying: "#9db7ff", psychic: "#ff5fa8", bug: "#a8cf2a",
  rock: "#b8a15a", ghost: "#7a62c9", dragon: "#6f50ff", dark: "#7a665a", steel: "#a9b8c9", fairy: "#ff9fd8",
};
export const DAMAGE_STEP = 5;
export const OPPONENT_DAMAGE = { min: 5, max: 40 };         // the opponent hits for a random multiple of 5 in this range
export const colorOf = (type) => TYPE_COLORS[type] || TYPE_COLORS.normal;

/** A printed damage value -> a multiple of 5 (at least 5), or null if the attack does no damage.
 *  Accepts what the server sends: 30, "30", null, undefined, 0. Odd printed values are rounded to the nearest 5 (e.g. 12 -> 10, 13 -> 15). */
export function damageOf(move) {
  const n = move && move.damage;
  if (n === null || n === undefined || n === "") return null;
  const v = Math.floor(Number(n));
  return Number.isFinite(v) && v > 0 ? Math.max(DAMAGE_STEP, Math.round(v / DAMAGE_STEP) * DAMAGE_STEP) : null;
}

/** Build a fighter from the server's answer: {pokemon:{name,types}, stats:{hp, attacks:[{name,damage,type,modifier}]}}. */
export function makeFighter(info) {
  const stats = (info && info.stats) || {};
  const hp = Math.max(1, Math.floor(Number(stats.hp)) || 100);
  const moves = (stats.attacks || []).slice(0, 4).map((m) => ({
    name: String(m.name || "Attack"), damage: damageOf(m), modifier: m.modifier || null, type: m.type || "normal",
  }));
  if (!moves.length) moves.push({ name: "Tackle", damage: 20, modifier: null, type: "normal" });
  const p = (info && info.pokemon) || {};
  return { name: p.name || (info && info.card && info.card.name) || "Pokémon", types: p.types || [], hp, maxHp: hp, moves };
}

export class Battle {
  /** @param player,opponent fighters from makeFighter   @param rng function returning [0,1) (injectable for tests) */
  constructor(player, opponent, rng = Math.random) {
    this.player = player; this.opponent = opponent; this.rng = rng;
    this.turn = "player"; this.winner = null; this.log = [];
  }

  /** Put both sides back to full HP, player first. */
  reset() {
    this.player.hp = this.player.maxHp; this.opponent.hp = this.opponent.maxHp;
    this.turn = "player"; this.winner = null; this.log = [];
  }

  /** `side` ("player"|"opponent") uses its move number `index`. Returns what happened:
   *  {ok, attacker, defender, move, damage, nonDamaging, color, message, fainted, winner}   (ok=false if not allowed) */
  useMove(side, index) {
    if (this.winner || side !== this.turn) return { ok: false, message: this.winner ? "The battle is over." : "Not your turn." };
    const me = side === "player" ? this.player : this.opponent, foe = side === "player" ? this.opponent : this.player;
    const move = me.moves[index];
    if (!move) return { ok: false, message: "That move doesn't exist." };
    const nonDamaging = side === "player" && move.damage === null;                   // the opponent always hits (random 5-40)
    const raw = side === "opponent" ? this.opponentDamage() : move.damage === null ? 0 : move.damage;
    const dealt = Math.min(raw, foe.hp);                                              // can't take more than the HP that is left
    foe.hp -= dealt;
    const fainted = foe.hp <= 0;
    if (fainted) this.winner = side;
    this.turn = fainted ? "over" : side === "player" ? "opponent" : "player";
    const message = nonDamaging ? `${me.name} used ${move.name}. It is a non-damaging move - no damage dealt.`
      : `${me.name} used ${move.name}! ${foe.name} lost ${dealt} HP.` + (fainted ? ` ${foe.name} fainted!` : "");
    const out = { ok: true, side, attacker: me, defender: foe, move, damage: dealt, nonDamaging, color: colorOf(move.type), message, fainted, winner: this.winner };
    this.log.push(out);
    return out;
  }

  /** The opponent's hit: a random multiple of 5 from 5 to 40 (8 equally likely values). */
  opponentDamage() {
    const steps = (OPPONENT_DAMAGE.max - OPPONENT_DAMAGE.min) / DAMAGE_STEP + 1;
    return OPPONENT_DAMAGE.min + DAMAGE_STEP * Math.min(steps - 1, Math.floor(this.rng() * steps));
  }

  /** Which move the opponent uses: a random one that does damage (any move if it has none). */
  pickOpponentMove() {
    const idx = this.opponent.moves.map((m, i) => (m.damage !== null ? i : -1)).filter((i) => i >= 0);
    const pool = idx.length ? idx : this.opponent.moves.map((_, i) => i);
    return pool[Math.min(pool.length - 1, Math.floor(this.rng() * pool.length))];
  }
}

/** Health bar fraction 0..1 and a traffic-light colour (green > 50 %, amber > 20 %, red below). */
export function barState(f) {
  const frac = Math.max(0, Math.min(1, f.hp / f.maxHp));
  return { frac, color: frac > 0.5 ? "#4fd05a" : frac > 0.2 ? "#ffcb05" : "#ff5a4f", text: `${Math.max(0, f.hp)} / ${f.maxHp}` };
}

/** Same card? (used to re-attach a running battle when a card is shown again). Set+number when known, else the Pokédex id. */
export function cardKey(info) {
  const c = info && info.card, p = info && info.pokemon;
  if (c && c.set && c.number) return `${c.set}#${c.number}`;
  return p ? `dex${p.id}` : null;
}
