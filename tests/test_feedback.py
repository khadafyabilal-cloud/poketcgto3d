"""Tests for user ratings: feedback.py, agent.submit_feedback, learned cards, model demotion, the /api/feedback routes.
Run:  python -m unittest discover -s tests -v"""
import json, sys, unittest
from pathlib import Path

import cv2
sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_server as ts                                       # reuses the stand-ins, Base and upright_card()
from test_server import StubRecognizer, PIKACHU, CHARMANDER, upright_card, agent_mod, app_mod
import feedback


class Fb(ts.Base):
    def setUp(self):
        super().setUp()
        self.pika, self.char = upright_card(11), upright_card(22)
        StubRecognizer.cards = [(self.pika, PIKACHU), (self.char, CHARMANDER)]

    def vote(self, res, target, vote, client="c1", **kw):
        return self.agent.submit_feedback(res["result_id"], client, target, vote, **kw)


class Results(Fb):
    def test_answers_carry_a_result_id(self):
        r = self.agent.run_card(self.pika)
        self.assertTrue(r["result_id"])
        self.assertNotEqual(r["result_id"], self.agent.run_card(self.char)["result_id"])
        self.assertTrue(self.agent.lookup("pikachu")["result_id"])

    def test_cached_answer_gets_its_own_id(self):
        a = self.agent.run_card(self.pika); b = self.agent.run_card(self.pika)
        self.assertTrue(b["cached"]); self.assertNotEqual(a["result_id"], b["result_id"])

    def test_no_id_without_a_model(self):
        self.assertNotIn("result_id", self.agent.lookup("zzzz"))

    def test_nothing_is_stored_until_someone_votes(self):
        self.agent.run_card(self.pika)
        self.assertEqual(self.agent.feedback.stats()["rated_results"], 0)
        self.assertEqual(list((self.tmp / "feedback" / "crops").iterdir()), [])


class Votes(Fb):
    def test_bad_input_and_unknown_id(self):
        r = self.agent.run_card(self.pika)
        self.assertFalse(self.vote(r, "other", 1)["ok"])
        self.assertFalse(self.vote(r, "model", 5)["ok"])
        self.assertFalse(self.agent.submit_feedback("nope", "c", "model", 1)["ok"])

    def test_vote_is_saved_and_can_be_changed(self):
        r = self.agent.run_card(self.pika)
        self.assertTrue(self.vote(r, "detection", 1)["ok"]); self.vote(r, "detection", -1)
        st = self.agent.feedback.stats()
        self.assertEqual(st["detection"], {"up": 0, "down": 1})        # one person, one vote: the later one replaced it
        self.assertEqual(st["rated_results"], 1)
        self.assertEqual(len(list((self.tmp / "feedback" / "crops").iterdir())), 1)

    def test_vote_survives_pending_eviction(self):
        r = self.agent.run_card(self.pika); self.vote(r, "model", 1)
        self.agent.pending.items.clear()
        self.assertTrue(self.vote(r, "model", -1, client="c2")["ok"])


class Learning(Fb):
    def test_thumbs_up_skips_the_recogniser_next_time(self):
        r = self.agent.run_card(self.pika); self.vote(r, "detection", 1)
        calls = StubRecognizer.calls
        again = self.agent.run_card(upright_card(11, pitch=0))
        self.assertTrue(again["learned"]); self.assertEqual(again["pokemon"]["id"], 25)
        self.assertEqual(StubRecognizer.calls, calls)
        self.assertEqual(again["trace"][0]["step"], "Learned")

    def test_learned_card_also_matches_upside_down(self):
        r = self.agent.run_card(self.pika); self.vote(r, "detection", 1)
        res = self.agent.run_card(cv2.rotate(self.pika, cv2.ROTATE_180))
        self.assertTrue(res["learned"]); self.assertTrue(res["rotated"])

    def test_correction_replaces_the_pokemon_and_is_remembered(self):
        r = self.agent.run_card(self.pika)
        out = self.vote(r, "detection", -1, correct="charmander")
        self.assertTrue(out["ok"]); self.assertEqual(out["result"]["pokemon"]["id"], 4)
        self.assertTrue(out["result"]["result_id"])
        again = self.agent.run_card(self.pika)
        self.assertEqual(again["pokemon"]["id"], 4); self.assertTrue(again["learned"])

    def test_correction_by_number_and_bad_correction(self):
        r = self.agent.run_card(self.pika)
        self.assertFalse(self.vote(r, "detection", -1, correct="zzzz")["ok"])
        self.assertEqual(self.vote(r, "detection", -1, correct="4")["result"]["pokemon"]["id"], 4)

    def test_voting_on_the_corrected_answer_keeps_the_card_picture(self):
        r = self.agent.run_card(self.pika)
        new = self.vote(r, "detection", -1, correct="charmander")["result"]
        self.assertTrue(self.vote(new, "detection", 1, client="c2")["ok"])
        self.assertEqual(self.agent.run_card(self.pika)["pokemon"]["id"], 4)

    def test_rejection_without_correction_skips_that_guess(self):
        # recogniser's ranked guesses for the pikachu picture: Pikachu first, Charmander second
        def exec_(path):
            return [PIKACHU, CHARMANDER]
        self.agent.recognizer.exec = exec_
        r = self.agent.run_card(self.pika)
        self.assertEqual(r["card"]["name"], "Pikachu")
        self.vote(r, "detection", -1)
        again = self.agent.run_card(self.pika)
        self.assertEqual(again["card"]["name"], "Charmander"); self.assertFalse(again.get("learned"))

    def test_other_cards_are_unaffected(self):
        r = self.agent.run_card(self.pika); self.vote(r, "detection", 1)
        c = self.agent.run_card(self.char)
        self.assertNotIn("learned", c); self.assertEqual(c["card"]["name"], "Charmander")

    def test_learned_memory_persists_across_restarts(self):
        r = self.agent.run_card(self.pika); self.vote(r, "detection", 1)
        fresh = agent_mod.PokemonCardAgent()
        self.assertTrue(fresh.run_card(self.pika)["learned"])


class ModelRatings(Fb):
    def three_downs(self, res):
        out = None
        for c in ("a", "b", "c"):
            out = self.vote(res, "model", -1, client=c)
        return out

    def test_three_thumbs_down_retire_the_model_kind(self):
        r = self.agent.run_card(self.pika)
        self.assertEqual(r["model_source"]["kind"], "glb")
        out = self.three_downs(r)
        self.assertIsNotNone(out["result"]); self.assertNotEqual(out["result"]["model_source"]["kind"], "glb")
        self.assertIn("glb", out["stats"]["demoted"])
        self.assertNotEqual(self.agent.lookup("pikachu")["model_source"]["kind"], "glb")

    def test_two_thumbs_down_are_not_enough_and_ups_offset_them(self):
        r = self.agent.run_card(self.pika)
        self.vote(r, "model", -1, client="a"); out = self.vote(r, "model", -1, client="b")
        self.assertIsNone(out["result"])
        self.vote(r, "model", 1, client="d"); self.assertIsNone(self.vote(r, "model", -1, client="e")["result"])

    def test_one_person_cannot_retire_a_model_alone(self):
        r1 = self.agent.run_card(self.pika)
        for _ in range(5):
            r = self.agent.lookup("pikachu"); self.vote(r, "model", -1, client="same")
        self.assertEqual(self.agent.feedback.demoted_stages(25), set())

    def test_up_votes_can_bring_it_back(self):
        r = self.agent.run_card(self.pika); self.three_downs(r)
        for c in ("u1", "u2", "u3", "u4"):
            self.agent.feedback.add_vote(r["result_id"], c, "model", 1)
        self.assertNotIn("glb", self.agent.feedback.demoted_stages(25))

    def test_threshold_is_configurable(self):
        old = feedback.MODEL_DEMOTE_SCORE; feedback.MODEL_DEMOTE_SCORE = -1
        self.addCleanup(setattr, feedback, "MODEL_DEMOTE_SCORE", old)
        r = self.agent.run_card(self.pika)
        self.assertIsNotNone(self.vote(r, "model", -1)["result"])


class Export(Fb):
    def test_export_writes_labels_and_images(self):
        r = self.agent.run_card(self.pika); self.vote(r, "detection", -1, correct="charmander"); self.vote(r, "model", 1)
        r2 = self.agent.run_card(self.char); self.vote(r2, "detection", 1)
        n = self.agent.feedback.export(self.tmp / "out")
        self.assertEqual(n, 2)
        rows = [json.loads(l) for l in (self.tmp / "out" / "labels.jsonl").read_text().splitlines()]
        by = {x["predicted_dex"]: x for x in rows}
        self.assertEqual(by[25]["label_dex"], 4); self.assertEqual(by[25]["corrected_dex"], 4)
        self.assertEqual(by[4]["label_dex"], 4); self.assertEqual(by[25]["model_votes"], {"up": 1, "down": 0})
        for x in rows:
            self.assertTrue((self.tmp / "out" / x["image"]).exists())


class Routes(Fb):
    def setUp(self):
        super().setUp()
        self.c = app_mod.create_app(self.agent).test_client()

    def test_feedback_route(self):
        r = self.agent.run_card(self.pika)
        ok = self.c.post("/api/feedback", json={"result_id": r["result_id"], "client": "x", "target": "model", "vote": 1})
        self.assertEqual(ok.status_code, 200); self.assertTrue(ok.get_json()["ok"])
        bad = self.c.post("/api/feedback", json={"result_id": r["result_id"], "target": "model", "vote": 9})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(self.c.post("/api/feedback", data="junk").status_code, 400)

    def test_stats_route(self):
        r = self.agent.run_card(self.pika); self.vote(r, "model", -1)
        d = self.c.get("/api/feedback/stats?dex=25").get_json()
        self.assertEqual(d["models"]["glb"], {"up": 0, "down": 1})

    def test_static_files_served(self):
        for p in ("/static/js/rating.js", "/battle", "/"):
            self.assertEqual(self.c.get(p).status_code, 200, p)



class Corrections(Fb):
    """The Detect page's "Teach the agent" form: a player types the real HP / attacks / Pokémon."""

    def correct(self, res, **kw):
        return self.agent.submit_correction(res["result_id"], kw.pop("client", "c1"), **kw)

    ATK = [{"name": "Thunder Jolt", "damage": "30", "type": "electric"}, {"name": "Growl", "damage": "", "type": ""}, {"name": "Slam", "damage": "40+", "type": "normal"}]

    def test_clean_attacks_and_hp(self):
        a = self.agent.clean_attacks(self.ATK + [{"name": ""}, "junk", {"name": "<b>X</b>\x00y", "damage": "999"}, {"name": "5th"}], ["electric"])
        self.assertEqual(len(a), 4)                                           # at most four, rows without a name skipped
        self.assertEqual((a[0]["name"], a[0]["damage"], a[0]["type"]), ("Thunder Jolt", 30, "electric"))
        self.assertEqual((a[1]["damage"], a[1]["type"]), (None, "electric"))   # no number = non-damaging; type assumed from the Pokémon
        self.assertEqual((a[2]["damage"], a[2]["modifier"]), (40, "+"))
        self.assertEqual(self.agent.clean_hp("60"), 60)
        for bad in ("", "abc", "5", "9999", None):
            self.assertIsNone(self.agent.clean_hp(bad), bad)

    def test_correction_is_used_instead_of_ocr_next_time(self):
        r = self.agent.run_card(self.pika)
        out = self.correct(r, hp="70", attacks=self.ATK)
        self.assertTrue(out["ok"], out)
        self.assertEqual((out["stats"]["hp"], out["stats"]["source"]), (70, "user"))
        again = self.agent.run_card(self.pika)                                # the same card shown again (a new scan)
        self.assertEqual(again["stats"]["hp"], 70)
        self.assertEqual([x["name"] for x in again["stats"]["attacks"]], ["Thunder Jolt", "Growl", "Slam"])
        self.assertEqual(again["stats"]["notes"], ["Corrected by a player."])

    def test_other_cards_are_not_affected(self):
        self.correct(self.agent.run_card(self.pika), hp="70", attacks=self.ATK)
        self.assertNotEqual(self.agent.run_card(self.char)["stats"]["source"], "user")

    def test_partial_correction_keeps_the_rest(self):
        r = self.agent.run_card(self.pika)
        base = r["stats"]
        out = self.correct(r, hp="90")                                          # only the HP
        self.assertEqual(out["stats"]["hp"], 90)
        again = self.agent.run_card(self.pika)["stats"]
        self.assertEqual(again["hp"], 90); self.assertTrue(again["attacks"])

    def test_pokemon_correction_returns_the_new_answer_with_the_stats(self):
        r = self.agent.run_card(self.pika)
        out = self.correct(r, pokemon="charmander", hp="60", attacks=self.ATK[:1])
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["result"]["pokemon"]["name"].lower(), "charmander")
        self.assertEqual(out["result"]["stats"]["hp"], 60)
        self.assertEqual(self.agent.feedback.stats()["detection"]["down"], 1)   # counted as a thumbs-down with a correction
        again = self.agent.run_card(self.pika)
        self.assertEqual(again["pokemon"]["name"].lower(), "charmander")        # the learned Pokémon correction

    def test_bad_input_is_refused_with_a_message(self):
        r = self.agent.run_card(self.pika)
        for kw, text in (({"hp": "abc"}, "HP must be"), ({"pokemon": "zzzzzz"}, "know"), ({"attacks": [{"name": ""}]}, "needs a name"), ({}, "Nothing to save")):
            out = self.correct(r, **kw)
            self.assertFalse(out["ok"], kw); self.assertIn(text, out["error"], kw)
        self.assertFalse(self.agent.submit_correction("nope", "c", hp="60")["ok"])

    def test_corrections_reach_the_training_export_and_stats(self):
        r = self.agent.run_card(self.pika)
        self.correct(r, hp="70", attacks=self.ATK)
        self.assertEqual(self.agent.feedback.stats()["stats_fixes"], 1)
        n = self.agent.feedback.export(self.tmp / "exp")
        row = json.loads((self.tmp / "exp" / "labels.jsonl").read_text().splitlines()[0])
        self.assertEqual((n, row["corrected_hp"], row["corrected_attacks"][0]["name"]), (1, 70, "Thunder Jolt"))
        self.assertTrue(row["image"])                                           # the card picture is exported with the label

    def test_route(self):
        c = app_mod.create_app(self.agent).test_client()
        r = self.agent.run_card(self.pika)
        ok = c.post("/api/feedback/correct", json={"result_id": r["result_id"], "client": "c1", "hp": 80, "attacks": self.ATK})
        self.assertEqual((ok.status_code, ok.get_json()["stats"]["hp"]), (200, 80))
        bad = c.post("/api/feedback/correct", json={"result_id": r["result_id"], "hp": "x"})
        self.assertEqual(bad.status_code, 400)
        self.assertEqual(c.post("/api/feedback/correct", data="not json").status_code, 400)

if __name__ == "__main__":
    unittest.main()
