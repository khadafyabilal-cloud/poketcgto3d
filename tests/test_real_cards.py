"""Real-card regression test: real card pictures (grey/silver borders, all-yellow cards, gold + metal energy ...) on simulated tables.
Fixed random seeds, so the numbers are repeatable. Thresholds are set a little under what the detector scores today."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import real_scene as rs

N = 12
CASES = {                                      # card -> minimum successes out of N (landscape and portrait each)
    "ampharos_team_rockets.png": 7, "ampharos_chaos_rising.png": 7, "pinsir.png": 7, "ninetales.png": 7, "psychic_energy.png": 7,
    "energy_lightning.png": 7, "energy_metal.png": 5, "energy_fire.png": 7, "energy_water.png": 7, "energy_grass.png": 7,
    "energy_fighting.png": 7, "energy_darkness.png": 7, "energy_psychic.png": 7,
}


class RealCards(unittest.TestCase):
    def test_landscape_and_portrait(self):
        for portrait in (False, True):
            res = rs.run(list(CASES), n=N, portrait=portrait)
            for card, (ok, n, _f) in res.items():
                with self.subTest(card=card, portrait=portrait):
                    self.assertGreaterEqual(ok, CASES[card], f"{card}: only {ok}/{n}")


if __name__ == "__main__":
    unittest.main()
