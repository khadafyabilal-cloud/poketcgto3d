// Tests for static/js/battle-engine.js (run: node --test tests/test_battle_engine.mjs)
import test from "node:test";
import assert from "node:assert/strict";
import { Battle, makeFighter, damageOf, barState, cardKey, colorOf, TYPE_COLORS } from "../static/js/battle-engine.js";

const info = (hp, attacks, name = "Charmander") => ({ pokemon: { name, types: ["fire"], id: 4 }, card: { set: "Base", number: "46", name }, stats: { hp, attacks } });
const mk = () => new Battle(makeFighter(info(60, [{ name: "Ember", damage: 30, type: "fire" }, { name: "Growl", damage: null, type: "normal" }])),
                             makeFighter(info(50, [{ name: "Tackle", damage: 20, type: "normal" }, { name: "Splash", damage: null }], "Magikarp")), () => 0);

test("HP comes from the card and moves keep their printed damage", () => {
  const f = makeFighter(info(120, [{ name: "Ember", damage: "30", type: "fire", modifier: "+" }]));
  assert.equal(f.hp, 120); assert.equal(f.moves[0].damage, 30); assert.equal(f.moves[0].modifier, "+");
});
test("missing or zero damage means non-damaging", () => {
  for (const d of [null, undefined, "", 0, "x", -5]) assert.equal(damageOf({ damage: d }), null, String(d));
  assert.equal(damageOf({ damage: 40 }), 40);
});
test("defaults when the card could not be read", () => {
  const f = makeFighter({ pokemon: { name: "Mew" } });
  assert.equal(f.hp, 100); assert.equal(f.moves.length, 1);
});
test("a damaging move removes exactly the printed number and uses the move's type colour", () => {
  const b = mk(), r = b.useMove("player", 0);
  assert.ok(r.ok); assert.equal(r.damage, 30); assert.equal(b.opponent.hp, 20); assert.equal(r.color, TYPE_COLORS.fire);
  assert.match(r.message, /Ember.*30 HP/);
});
test("a move without a number says it is non-damaging and changes no HP", () => {
  const b = mk(), r = b.useMove("player", 1);
  assert.ok(r.nonDamaging); assert.equal(r.damage, 0); assert.equal(b.opponent.hp, 50); assert.match(r.message, /non-damaging/);
  assert.equal(b.turn, "opponent");                       // it still uses up the turn
});
test("turns alternate and out-of-turn moves are refused", () => {
  const b = mk();
  assert.equal(b.useMove("opponent", 0).ok, false);
  b.useMove("player", 1); assert.equal(b.useMove("player", 0).ok, false);
  assert.ok(b.useMove("opponent", b.pickOpponentMove()).ok); assert.equal(b.turn, "player");
});
test("opponent never picks a non-damaging move while it has a damaging one", () => {
  const b = mk(); b.rng = () => 0.99;
  assert.equal(b.opponent.moves[b.pickOpponentMove()].damage, 20);
});
test("fainting ends the battle, HP never drops below 0, and reset restores everything", () => {
  const b = mk();
  b.useMove("player", 0); b.useMove("opponent", 0); const r = b.useMove("player", 0);
  assert.equal(b.opponent.hp, 0); assert.ok(r.fainted); assert.equal(b.winner, "player"); assert.equal(b.turn, "over");
  assert.equal(r.damage, 20);                              // only the 20 HP that were left
  assert.equal(b.useMove("opponent", 0).ok, false);
  b.reset(); assert.equal(b.opponent.hp, 50); assert.equal(b.player.hp, 60); assert.equal(b.winner, null);
});
test("player can lose too", () => {
  const b = new Battle(makeFighter(info(20, [{ name: "Scratch", damage: 10 }])), makeFighter(info(100, [{ name: "Slam", damage: 50 }], "Onix")), () => 0.99);   // rng .99 -> the opponent's biggest hit, 40
  b.useMove("player", 0); const r = b.useMove("opponent", 0);
  assert.ok(r.fainted); assert.equal(b.winner, "opponent");
});
test("health bar fractions and colours", () => {
  assert.equal(barState({ hp: 60, maxHp: 60 }).color, "#4fd05a");
  assert.equal(barState({ hp: 20, maxHp: 60 }).color, "#ffcb05");
  assert.equal(barState({ hp: 5, maxHp: 60 }).color, "#ff5a4f");
  assert.equal(barState({ hp: -3, maxHp: 60 }).frac, 0);
});
test("every Pokémon type has a colour and unknown types fall back", () => {
  for (const t of ["normal","fire","water","electric","grass","ice","fighting","poison","ground","flying","psychic","bug","rock","ghost","dragon","dark","steel","fairy"]) assert.ok(TYPE_COLORS[t], t);
  assert.equal(colorOf("nope"), TYPE_COLORS.normal);
});
test("cardKey is stable for the same printing", () => {
  assert.equal(cardKey(info(1, [])), "Base#46");
  assert.equal(cardKey({ pokemon: { id: 25 } }), "dex25");
});

test("opponent damage is random, 5-40, always a multiple of 5, and each of the 8 values can happen", () => {
  const seen = new Set();
  for (let i = 0; i < 1000; i++) {
    const b = mk(); b.rng = Math.random; b.player.hp = b.player.maxHp = 999;
    b.useMove("player", 0); const r = b.useMove("opponent", b.pickOpponentMove());
    assert.ok(r.damage >= 5 && r.damage <= 40 && r.damage % 5 === 0, String(r.damage)); seen.add(r.damage);
  }
  assert.deepEqual([...seen].sort((a, b) => a - b), [5, 10, 15, 20, 25, 30, 35, 40]);
  const edge = (x) => { const b = mk(); b.rng = () => x; return b.opponentDamage(); };
  assert.equal(edge(0), 5); assert.equal(edge(0.9999999), 40);
});
test("opponent damage ignores its card numbers (a 20-damage move still hits 5-40) and is capped by the HP that is left", () => {
  const b = mk(); b.rng = () => 0.99; b.player.hp = 15;
  b.useMove("player", 0); const r = b.useMove("opponent", 0);
  assert.equal(r.damage, 15); assert.ok(r.fainted);
});
test("player damage is a multiple of 5 too: printed numbers are rounded", () => {
  for (const [raw, want] of [[30, 30], [12, 10], [13, 15], [2, 5], [47, 45], ["25", 25]]) assert.equal(damageOf({ damage: raw }), want, String(raw));
  const b = new Battle(makeFighter(info(60, [{ name: "Odd", damage: 33 }])), makeFighter(info(50, [{ name: "T", damage: 5 }], "X")), () => 0);
  assert.equal(b.useMove("player", 0).damage % 5, 0);
});
