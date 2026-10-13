"""
feedback.py - user ratings (thumbs up / down) and what the app learns from them.
================================================================================

WHAT "TRAINING THROUGH THE WEB UI" MEANS HERE (please read - it is not magic)
-----------------------------------------------------------------------------
The card recogniser is a pre-trained neural network (pokemon-card-recognizer). Its weights are NOT changed while
the app runs: re-training a deep network from a handful of thumbs is slow, needs a GPU and would easily make it
worse. Instead, user ratings feed four *learning mechanisms* that change what the app does on the very next scan,
plus a *dataset* you can use to really fine-tune a network offline:

  1. LEARNED CARDS (detection votes)
       thumbs-up   -> remember "this card picture = this card/Pokémon" (a fingerprint of the card face, see
                      agent._signature). Next time the same card is shown the recogniser is skipped.
       thumbs-down + the right Pokémon typed in -> remember the CORRECTION; next time that card face gets the
                      corrected Pokémon.
       thumbs-down without a correction -> remember "NOT this card"; next time the recogniser's next-best guess is used.
  2. MODEL RATINGS (3D-model votes)
       Votes are counted per (Pokémon, kind of model: custom / gen1 / pack / relief / voxel). When a kind's score
       (ups minus downs, one vote per person) falls to MODEL_DEMOTE_SCORE (-3) that kind is skipped for that
       Pokémon and the next kind in the chain is used (agent._finish). Up-votes raise the score again.
  3. STAT CORRECTIONS (Detect page "Teach")
       A player types the HP and attacks the card really shows. They are stored (stats_fix) by card set+number, or by
       card-face fingerprint, and used INSTEAD of OCR the next time that card is seen.
  4. TRAINING DATA EXPORT
       Every rated card keeps its straightened picture + label in feedback/crops. `python scripts/export_feedback.py`
       writes labels.jsonl + images, ready to fine-tune a classifier offline.

Where it runs: all of this is SERVER-side Python + SQLite (no extra hardware). The browser only sends a few hundred
bytes per vote.

WHY SQLITE: it is a database in a single file (feedback/feedback.db), part of Python's standard library, safe for
the small write volume of thumbs, and easy to inspect with any SQLite viewer.

DATA PRIVACY: a card picture is stored only when somebody votes on it (nothing is kept for unrated scans), and
only the straightened ~440-px card, not the whole camera frame.
"""
import json, os, sqlite3, threading, time
from collections import OrderedDict
from pathlib import Path

import numpy as np

# Votes needed before a model kind is skipped: score = ups - downs per (Pokémon, kind) <= this value.
# Override with the environment variable PC3D_DEMOTE_SCORE (e.g. -1 = a single thumbs-down is enough).
MODEL_DEMOTE_SCORE = int(os.environ.get("PC3D_DEMOTE_SCORE", "-3"))

# agent.py names the model by "model_source.kind"; the chain in agent._finish works in "stages".
# Anything not listed here (custom sprite / sheet / artwork) is the voxel fallback.
STAGE_OF_KIND = {"custom_model": "custom", "gen1": "gen1", "glb": "glb", "relief": "relief"}


def stage_of(kind):
    """model_source.kind -> stage name used for ratings and demotion."""
    return STAGE_OF_KIND.get(kind, "voxel")


SCHEMA = """
CREATE TABLE IF NOT EXISTS result (          -- one row per rated result (written on the FIRST vote only)
  id          TEXT PRIMARY KEY,              -- random id the server handed to the browser with the answer
  ts          REAL NOT NULL,
  source      TEXT NOT NULL,                 -- 'battle' | 'scan' | 'manual'
  dex         INTEGER,                       -- the Pokémon the app showed
  card_set    TEXT, card_number TEXT, card_name TEXT,
  model_kind  TEXT, stage TEXT, model_url TEXT,
  signature   BLOB,                          -- 24x32x3 float16 card fingerprint (NULL for whole-photo scans)
  crop_file   TEXT                           -- feedback/crops/<id>.jpg
);
CREATE TABLE IF NOT EXISTS vote (            -- one row per person per result per question
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  result_id   TEXT NOT NULL REFERENCES result(id),
  client      TEXT NOT NULL,                 -- anonymous id the browser keeps in localStorage
  target      TEXT NOT NULL CHECK (target IN ('detection','model')),
  vote        INTEGER NOT NULL CHECK (vote IN (1,-1)),
  correct_dex INTEGER,                       -- for a detection thumbs-down: what it really was
  comment     TEXT,
  ts          REAL NOT NULL,
  UNIQUE (result_id, client, target)         -- changing your mind replaces your earlier vote
);
CREATE TABLE IF NOT EXISTS learned (         -- the memory the recogniser consults (mechanism 1)
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  kind        TEXT NOT NULL CHECK (kind IN ('confirmed','corrected','rejected')),
  signature   BLOB NOT NULL,
  dex         INTEGER,                       -- confirmed / corrected: the Pokémon
  card_set    TEXT, card_number TEXT, card_name TEXT,
  ts          REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS stats_fix (       -- what a player says the card's HP / attacks really are (Detect page "Teach")
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  result_id   TEXT, client TEXT,
  dex         INTEGER,
  card_set    TEXT, card_number TEXT,           -- the key when the card is a known printing
  signature   BLOB,                             -- otherwise the card face fingerprint is the key
  hp          INTEGER,                          -- NULL = the player did not change the HP
  attacks     TEXT,                             -- JSON list of {name, damage, type}; NULL = not changed
  ts          REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS vote_result ON vote(result_id);
"""


def _sig_to_blob(sig):
    return np.asarray(sig, np.float16).tobytes()


def _blob_to_sig(blob):
    return np.frombuffer(blob, np.float16).astype(np.float32).reshape(32, 24, 3)    # cv2.resize((24,32)) -> shape (32, 24, 3)


class FeedbackStore:
    """Thread-safe wrapper around the SQLite file. Flask serves requests on several threads, hence the lock."""

    def __init__(self, directory, match_dist=0.28):
        self.dir = Path(directory)
        self.crops = self.dir / "crops"
        self.crops.mkdir(parents=True, exist_ok=True)
        self.match_dist = match_dist                 # same meaning as agent.SCAN_CACHE_DIST (mean absolute difference)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(str(self.dir / "feedback.db"), check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._learned = []                           # in-memory copy for fast matching (hundreds of rows at most)
        self._reload()

    # ------------------------------------------------------------ learned memory (mechanism 1)
    def _reload(self):
        with self.lock:
            self._learned = [dict(r, sig=_blob_to_sig(r["signature"])) for r in
                             self.db.execute("SELECT id, kind, signature, dex, card_set, card_number, card_name FROM learned")]

    def _near(self, sig):
        """All learned rows whose fingerprint is within match_dist of `sig`, as [(distance, row)]."""
        out = []
        for row in self._learned:
            d = float(np.abs(sig - row["sig"]).mean())
            if d < self.match_dist:
                out.append((d, row))
        return out

    def match(self, sig_seen, sig_turned):
        """Look a card up in the learned memory. `sig_seen` = fingerprint as the camera saw it, `sig_turned` = the same
        picture turned 180 degrees (a card may lie upside down).
        Returns (entry or None, rotated, excluded) where
          entry    = the closest 'confirmed'/'corrected' row (a dict) or None,
          rotated  = True if the match was found on the turned picture,
          excluded = set of (set, number) pairs the recogniser must NOT answer for this card (thumbs-down 'rejected')."""
        with self.lock:
            best, excluded = None, set()
            for sig, rotated in ((sig_seen, False), (sig_turned, True)):
                for d, row in self._near(sig):
                    if row["kind"] == "rejected":
                        excluded.add((row["card_set"], row["card_number"]))
                    elif best is None or (d, -row["id"]) < (best[0], -best[1]["id"]):     # nearest wins; newest on a tie
                        best = (d, row, rotated)
            return (best[1], best[2], excluded) if best else (None, False, excluded)

    def learn(self, kind, sig, dex=None, card=None):
        """Add a memory row. A newer opinion about the same card replaces older, contradicting rows."""
        card = card or {}
        with self.lock:
            for d, row in self._near(sig):                       # forget rows this vote contradicts / duplicates
                same_card = (row["card_set"], row["card_number"]) == (card.get("set"), card.get("number"))
                if (kind == "confirmed" and row["kind"] == "rejected" and same_card) \
                        or (kind == "rejected" and row["kind"] == "confirmed" and same_card) \
                        or (kind in ("corrected", "confirmed") and row["kind"] in ("corrected", "confirmed") and d < 0.05):
                    self.db.execute("DELETE FROM learned WHERE id=?", (row["id"],))
            self.db.execute("INSERT INTO learned(kind, signature, dex, card_set, card_number, card_name, ts) VALUES (?,?,?,?,?,?,?)",
                            (kind, _sig_to_blob(sig), dex, card.get("set"), card.get("number"), card.get("name"), time.time()))
            self.db.commit()
            self._reload()

    # ------------------------------------------------------------ stat corrections (mechanism 3)
    def save_stats_fix(self, rid, client, ctx, dex, hp, attacks):
        """Remember a player's HP and/or attacks (None = unchanged) for the card in `ctx`."""
        sig = ctx.get("signature")
        with self.lock:
            self.db.execute("INSERT INTO stats_fix(result_id, client, dex, card_set, card_number, signature, hp, attacks, ts) VALUES (?,?,?,?,?,?,?,?,?)",
                            (rid, client, dex, ctx.get("card_set"), ctx.get("card_number"), _sig_to_blob(sig) if sig is not None else None,
                             hp, json.dumps(attacks, ensure_ascii=False) if attacks else None, time.time()))
            self.db.commit()

    def match_stats(self, sig, card=None, max_dist=0.16):
        """The player-supplied {hp, attacks} for this card or None. Same printing (set+number) wins, otherwise a very
        similar card-face fingerprint. Newest answers win, per field (an HP-only fix and a later attacks-only fix combine)."""
        card = card or {}
        with self.lock:
            rows = self.db.execute("SELECT * FROM stats_fix ORDER BY id DESC").fetchall()
        hp, attacks = None, None
        for r in rows:
            same = bool(card.get("set") and card.get("number") and (r["card_set"], r["card_number"]) == (card.get("set"), card.get("number")))
            if not same and sig is not None and r["signature"]:
                same = float(np.abs(sig - _blob_to_sig(r["signature"])).mean()) < max_dist
            if not same:
                continue
            if hp is None and r["hp"]:
                hp = r["hp"]
            if attacks is None and r["attacks"]:
                attacks = json.loads(r["attacks"])
        return {"hp": hp, "attacks": attacks} if hp or attacks else None

    # ------------------------------------------------------------ results + votes
    def save_result(self, rid, ctx):
        """Persist a result (and its picture) the first time somebody votes on it."""
        with self.lock:
            if self.db.execute("SELECT 1 FROM result WHERE id=?", (rid,)).fetchone():
                return
            crop_file = None
            if ctx.get("crop_jpeg"):
                crop_file = f"{rid}.jpg"
                (self.crops / crop_file).write_bytes(ctx["crop_jpeg"])
            sig = ctx.get("signature")
            self.db.execute("INSERT INTO result(id, ts, source, dex, card_set, card_number, card_name, model_kind, stage, model_url, signature, crop_file)"
                            " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (rid, time.time(), ctx.get("source", "scan"), ctx.get("dex"), ctx.get("card_set"), ctx.get("card_number"),
                             ctx.get("card_name"), ctx.get("model_kind"), stage_of(ctx.get("model_kind")), ctx.get("model_url"),
                             _sig_to_blob(sig) if sig is not None else None, crop_file))
            self.db.commit()

    def get_result(self, rid):
        """A previously saved result as a ctx dict (so a vote still works after a server restart), or None."""
        with self.lock:
            r = self.db.execute("SELECT * FROM result WHERE id=?", (rid,)).fetchone()
        if not r:
            return None
        crop = (self.crops / r["crop_file"]).read_bytes() if r["crop_file"] and (self.crops / r["crop_file"]).exists() else None
        return {"source": r["source"], "dex": r["dex"], "card_set": r["card_set"], "card_number": r["card_number"], "card_name": r["card_name"],
                "model_kind": r["model_kind"], "model_url": r["model_url"], "crop_jpeg": crop,
                "signature": _blob_to_sig(r["signature"]) if r["signature"] else None}

    def add_vote(self, rid, client, target, vote, correct_dex=None, comment=""):
        """Insert or replace this person's vote on this question about this result."""
        with self.lock:
            self.db.execute(
                "INSERT INTO vote(result_id, client, target, vote, correct_dex, comment, ts) VALUES (?,?,?,?,?,?,?) "
                "ON CONFLICT(result_id, client, target) DO UPDATE SET vote=excluded.vote, correct_dex=excluded.correct_dex, "
                "comment=excluded.comment, ts=excluded.ts",
                (rid, client, target, vote, correct_dex, (comment or "")[:300], time.time()))
            self.db.commit()

    # ------------------------------------------------------------ model ratings (mechanism 2)
    def model_votes(self, dex, stage):
        """(ups, downs) for this Pokémon and model kind. One vote per person: a person's LATEST vote on this
        (Pokémon, kind) counts, so clicking thumbs-down on ten results does not count ten times."""
        with self.lock:
            rows = self.db.execute(
                "SELECT v.client, v.vote FROM vote v JOIN result r ON r.id = v.result_id "
                "WHERE v.target='model' AND r.dex=? AND r.stage=? ORDER BY v.ts", (dex, stage)).fetchall()
        latest = {}
        for r in rows:
            latest[r["client"]] = r["vote"]
        ups = sum(1 for v in latest.values() if v > 0)
        return ups, len(latest) - ups

    def demoted_stages(self, dex):
        """Stages the agent should skip for this Pokémon (score <= MODEL_DEMOTE_SCORE)."""
        out = set()
        for stage in ("custom", "gen1", "glb", "relief"):         # the voxel fallback is never skipped
            ups, downs = self.model_votes(dex, stage)
            if ups - downs <= MODEL_DEMOTE_SCORE:
                out.add(stage)
        return out

    def stats(self, dex=None):
        """Numbers for the UI / dashboards: per model kind (for one Pokémon, or all), plus detection totals."""
        with self.lock:
            det = self.db.execute("SELECT vote, COUNT(*) n FROM vote WHERE target='detection' GROUP BY vote").fetchall()
            learned = self.db.execute("SELECT kind, COUNT(*) n FROM learned GROUP BY kind").fetchall()
            results = self.db.execute("SELECT COUNT(*) n FROM result").fetchone()["n"]
            fixes = self.db.execute("SELECT COUNT(*) n FROM stats_fix").fetchone()["n"]
        out = {"detection": {"up": 0, "down": 0}, "learned": {r["kind"]: r["n"] for r in learned}, "rated_results": results, "stats_fixes": fixes}
        for r in det:
            out["detection"]["up" if r["vote"] > 0 else "down"] = r["n"]
        if dex is not None:
            out["models"] = {s: dict(zip(("up", "down"), self.model_votes(dex, s))) for s in ("custom", "gen1", "glb", "relief", "voxel")}
            out["demoted"] = sorted(self.demoted_stages(dex))
        return out

    # ------------------------------------------------------------ training data export (mechanism 3)
    def export(self, out_dir):
        """Write a training set: <out_dir>/images/<id>.jpg and <out_dir>/labels.jsonl, one line per rated result:
        {image, predicted_dex, predicted_card, detection_votes {up, down}, corrected_dex (most recent correction or null),
         label_dex (best guess at the truth: correction > predicted if net up-voted > null), model_kind, model_votes {up, down}}.
        Returns the number of lines written."""
        out_dir = Path(out_dir); (out_dir / "images").mkdir(parents=True, exist_ok=True)
        n = 0
        with self.lock, open(out_dir / "labels.jsonl", "w", encoding="utf-8") as f:
            for r in self.db.execute("SELECT * FROM result ORDER BY ts").fetchall():
                votes = self.db.execute("SELECT * FROM vote WHERE result_id=? ORDER BY ts", (r["id"],)).fetchall()
                det = [v for v in votes if v["target"] == "detection"]
                mod = [v for v in votes if v["target"] == "model"]
                corrected = next((v["correct_dex"] for v in reversed(det) if v["correct_dex"]), None)
                up, down = sum(v["vote"] > 0 for v in det), sum(v["vote"] < 0 for v in det)
                label = corrected or (r["dex"] if up > down else None)
                fx = self.db.execute("SELECT hp, attacks FROM stats_fix WHERE result_id=? ORDER BY id", (r["id"],)).fetchall()
                chp = next((x["hp"] for x in reversed(fx) if x["hp"]), None)
                catk = next((json.loads(x["attacks"]) for x in reversed(fx) if x["attacks"]), None)
                image = None
                if r["crop_file"] and (self.crops / r["crop_file"]).exists():
                    image = f"images/{r['crop_file']}"
                    (out_dir / image).write_bytes((self.crops / r["crop_file"]).read_bytes())
                f.write(json.dumps({"id": r["id"], "image": image, "source": r["source"], "predicted_dex": r["dex"],
                                    "predicted_card": {"set": r["card_set"], "number": r["card_number"], "name": r["card_name"]},
                                    "detection_votes": {"up": up, "down": down}, "corrected_dex": corrected, "label_dex": label,
                                    "corrected_hp": chp, "corrected_attacks": catk,
                                    "model_kind": r["model_kind"],
                                    "model_votes": {"up": sum(v["vote"] > 0 for v in mod), "down": sum(v["vote"] < 0 for v in mod)}},
                                   ensure_ascii=False) + "\n")
                n += 1
        return n


class PendingResults:
    """The last N answers the server gave, kept in RAM so a vote can refer to one by id.
    Nothing here is written to disk until somebody votes (see FeedbackStore.save_result)."""

    def __init__(self, limit=256):
        self.limit, self.items, self.lock = limit, OrderedDict(), threading.Lock()

    def put(self, rid, ctx):
        with self.lock:
            self.items[rid] = ctx
            while len(self.items) > self.limit:
                self.items.popitem(last=False)

    def get(self, rid):
        with self.lock:
            return self.items.get(rid)
