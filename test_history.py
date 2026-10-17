"""Tests for the hand-history store (history.py) + the seam contract (seam.py) — P1.

history.py is isolated by passing an EXPLICIT db path (the same contract as Game's other
stores — test_paths.py). The one seam.get_history test isolates via GUTCHECK_DATA_DIR
because the seam reads the global store (the resolution production relies on). The
end-to-end test drives a real Game through settled hands and asserts the rows land — the
L4 "the _accumulate_session hook actually fires" check, not just a unit-level mock.
"""

import os
import random
import sqlite3
import tempfile
import unittest
from pathlib import Path

from gutcheck import game as G
from gutcheck import history
from gutcheck import paths
from gutcheck import seam as seam_mod


def _rec(n, **over):
    r = {"hand_no": n, "seed": 1000 + n, "start_stacks": [200] * 6,
         "assign": [0, 1, 2, 3, 4, 5], "log": ["d", "c", "f"], "pot": 10 * n,
         "hero_net": n - 2, "board": ["Ah", "Kd", "2c"], "street": "flop"}
    r.update(over)
    return r


class TestHistoryStore(unittest.TestCase):
    def setUp(self):
        self.db = Path(tempfile.mkdtemp()) / "history.db"

    def test_record_returns_true_then_false_on_dup(self):
        self.assertTrue(history.record_hand("sA", _rec(1), path=self.db))
        self.assertFalse(history.record_hand("sA", _rec(1), path=self.db))   # idempotent (OR IGNORE)

    def test_hand_no_unique_per_session_not_global(self):
        self.assertTrue(history.record_hand("sA", _rec(1), path=self.db))
        self.assertTrue(history.record_hand("sB", _rec(1), path=self.db))    # same hand_no, diff session
        self.assertEqual(history.get_history(0, path=self.db)["count"], 2)

    def test_get_hand_point_lookup(self):
        history.record_hand("sA", _rec(3), path=self.db)
        got = history.get_hand("sA", 3, path=self.db)
        self.assertIsNotNone(got)
        self.assertEqual(got["hand_no"], 3)
        self.assertEqual(got["seed"], 1003)
        self.assertEqual(got["assign"], [0, 1, 2, 3, 4, 5])   # JSON field re-parsed
        self.assertEqual(got["log"], ["d", "c", "f"])

    def test_get_hand_scoped_to_session(self):
        history.record_hand("sA", _rec(1), path=self.db)
        history.record_hand("sB", _rec(1, seed=2001), path=self.db)   # same hand_no, diff session+seed
        self.assertEqual(history.get_hand("sA", 1, path=self.db)["seed"], 1001)
        self.assertEqual(history.get_hand("sB", 1, path=self.db)["seed"], 2001)
        self.assertIsNone(history.get_hand("sA", 99, path=self.db))   # missing → None

    def test_get_hand_rejects_bool_hand_no(self):
        history.record_hand("sA", _rec(1), path=self.db)
        self.assertIsNone(history.get_hand("sA", True, path=self.db))   # bool ⊆ int — must not match hand 1

    def test_half_open_watermark(self):
        for i in (1, 2, 3):
            history.record_hand("sA", _rec(i), path=self.db)
        full = history.get_history(0, path=self.db)
        self.assertEqual(full["count"], 3)
        self.assertEqual(history.get_history(full["watermark"], path=self.db)["count"], 0)  # since=last → none
        self.assertEqual(history.get_history(1, path=self.db)["count"], 2)                   # mid cursor → newer

    def test_limit_then_continue_skips_nothing(self):
        for i in range(1, 6):
            history.record_hand("sA", _rec(i), path=self.db)
        first = history.get_history(0, limit=2, path=self.db)
        self.assertEqual(first["count"], 2)
        rest = history.get_history(first["watermark"], path=self.db)
        self.assertEqual(rest["count"], 3)     # the 3 beyond the limit, no row dropped at the boundary

    def test_latest_returns_most_recent_window_ascending(self):
        # latest=True → the LAST `limit` hands (the coach's "what did I just play"), still oldest-first
        for i in range(1, 11):                                 # 10 hands → ids 1..10
            history.record_hand("sA", _rec(i), path=self.db)
        out = history.get_history(0, limit=3, path=self.db, latest=True)
        self.assertEqual(out["count"], 3)
        self.assertEqual([h["id"] for h in out["hands"]], [8, 9, 10])   # the 3 NEWEST, ascending
        self.assertEqual(out["watermark"], 10)                # watermark = newest id (not the oldest in window)
        # contrast: latest=False (the forward cursor) returns the OLDEST `limit`
        self.assertEqual([h["id"] for h in history.get_history(0, limit=3, path=self.db)["hands"]], [1, 2, 3])

    def test_latest_window_smaller_than_limit_returns_all(self):
        for i in range(1, 3):                                 # only 2 hands
            history.record_hand("sA", _rec(i), path=self.db)
        out = history.get_history(0, limit=20, path=self.db, latest=True)
        self.assertEqual([h["id"] for h in out["hands"]], [1, 2])   # all of them, ascending, no crash
        self.assertEqual(out["watermark"], 2)

    def test_latest_composes_with_session_scope(self):
        for sid in ("sA", "sB"):
            for n in range(1, 5):                             # 4 hands each, interleaved ids
                history.record_hand(sid, _rec(n), path=self.db)
        out = history.get_history(0, limit=2, path=self.db, session_id="sA", latest=True)
        self.assertEqual(out["count"], 2)
        self.assertTrue(all(h["session_id"] == "sA" for h in out["hands"]))   # the 2 newest of sA only
        self.assertEqual([h["id"] for h in out["hands"]], sorted(h["id"] for h in out["hands"]))

    def test_json_round_trip(self):
        history.record_hand("sA", _rec(1, board=["Ah", "Kd"], assign=[5, 4, 3, 2, 1, 0]), path=self.db)
        h = history.get_history(0, path=self.db)["hands"][0]
        self.assertEqual(h["board"], ["Ah", "Kd"])
        self.assertEqual(h["assign"], [5, 4, 3, 2, 1, 0])
        self.assertEqual(h["log"], ["d", "c", "f"])

    def test_ordering_is_insertion_id_not_hand_no(self):
        for i in (3, 1, 2):                       # ids assigned in insert order → 1,2,3
            history.record_hand("sA", _rec(i), path=self.db)
        hands = history.get_history(0, path=self.db)["hands"]
        self.assertEqual([h["hand_no"] for h in hands], [3, 1, 2])   # by id (insert), not hand_no value

    def test_empty_returns_since_as_watermark(self):
        self.assertEqual(history.get_history(7, path=self.db), {"hands": [], "watermark": 7, "count": 0})

    def test_session_filter_scopes_to_one_session(self):
        # P3(d): interleave two sessions in global-id space; the filter returns only the named one.
        for sid, n in (("sA", 1), ("sB", 1), ("sA", 2), ("sB", 2), ("sA", 3)):   # ids 1..5
            history.record_hand(sid, _rec(n), path=self.db)
        a = history.get_history(0, path=self.db, session_id="sA")
        b = history.get_history(0, path=self.db, session_id="sB")
        self.assertEqual([h["session_id"] for h in a["hands"]], ["sA", "sA", "sA"])
        self.assertEqual([h["session_id"] for h in b["hands"]], ["sB", "sB"])
        self.assertEqual(a["watermark"], 5)     # global id of the last sA row (interleaved), not 3
        self.assertEqual(b["watermark"], 4)

    def test_session_filter_watermark_resume_skips_and_overlaps_nothing(self):
        # the half-open watermark composes with the scope: resume from it, see only newer rows of that session
        for sid, n in (("sA", 1), ("sB", 1), ("sA", 2), ("sB", 2), ("sA", 3)):   # sA at ids 1,3,5
            history.record_hand(sid, _rec(n), path=self.db)
        first = history.get_history(0, path=self.db, session_id="sA")
        self.assertEqual([h["id"] for h in first["hands"]], [1, 3, 5])
        # resume from a mid cursor (the global id 3): only sA's id-5 row, no re-scan of 1/3, no sB leakage
        rest = history.get_history(3, path=self.db, session_id="sA")
        self.assertEqual([h["id"] for h in rest["hands"]], [5])
        self.assertEqual(history.get_history(first["watermark"], path=self.db, session_id="sA")["count"], 0)

    def test_session_filter_absent_or_empty_means_all(self):
        history.record_hand("sA", _rec(1), path=self.db)
        history.record_hand("sB", _rec(1), path=self.db)
        allc = history.get_history(0, path=self.db)["count"]
        self.assertEqual(history.get_history(0, path=self.db, session_id=None)["count"], allc)   # default
        self.assertEqual(history.get_history(0, path=self.db, session_id="")["count"], allc)     # empty str

    def test_session_filter_no_such_session_is_empty(self):
        history.record_hand("sA", _rec(1), path=self.db)
        out = history.get_history(0, path=self.db, session_id="ghost")
        self.assertEqual(out["count"], 0)
        self.assertEqual(out["watermark"], 0)   # unchanged since (nothing matched)

    def test_bad_since_coerced_to_zero(self):
        history.record_hand("sA", _rec(1), path=self.db)
        self.assertEqual(history.get_history(-5, path=self.db)["count"], 1)    # negative → 0
        self.assertEqual(history.get_history("x", path=self.db)["count"], 1)   # non-int → 0
        self.assertEqual(history.get_history(None, path=self.db)["count"], 1)  # None → 0

    def test_missing_session_id_raises(self):
        with self.assertRaises(ValueError):
            history.record_hand("", _rec(1), path=self.db)

    def test_corrupt_blob_keeps_the_hand_raw(self):
        history.record_hand("sA", _rec(1), path=self.db)
        con = sqlite3.connect(str(self.db))                # corrupt one JSON column directly
        con.execute("UPDATE hands SET board='{not json' WHERE hand_no=1")
        con.commit()
        con.close()
        hands = history.get_history(0, path=self.db)["hands"]
        self.assertEqual(len(hands), 1)                    # row NOT dropped — a corrupt blob is still a hand
        self.assertEqual(hands[0]["board"], "{not json")   # left raw rather than discarded


class _StubGame:
    """A minimal stand-in exposing only the attrs/methods the seam routes to."""
    def __init__(self):
        self.calls = []
        self.history_db = None     # seam captures game.history_db; None → history resolves the env path

    def view(self):
        return {"state": "ok"}

    def calibration_summary(self):
        return {"exact": {"n": 3}}

    def record_review_read(self, **kw):
        self.calls.append(("review", kw))
        return {"ok": True, "kw": kw}

    def record_live_read(self, **kw):
        self.calls.append(("live", kw))
        return {"ok": True, "kw": kw}


class TestSeam(unittest.TestCase):
    def setUp(self):
        self.stub = _StubGame()
        self.s = seam_mod.Seam(self.stub)

    def test_get_table_state_delegates(self):
        self.assertEqual(self.s.get_table_state(), {"state": "ok"})

    def test_get_calibration_delegates(self):
        self.assertEqual(self.s.get_calibration(), {"exact": {"n": 3}})

    def test_commit_read_review_routes(self):
        out = self.s.commit_read("review", session_id="x", hand_no=1, n=1, gut=50, lean="call")
        self.assertEqual(self.stub.calls[-1][0], "review")
        self.assertTrue(out["ok"])

    def test_commit_read_live_routes(self):
        out = self.s.commit_read("live", gut=50, lean="call", truth_pct=60.0)
        self.assertEqual(self.stub.calls[-1][0], "live")
        self.assertTrue(out["ok"])

    def test_commit_read_unknown_kind_routes_nothing(self):
        out = self.s.commit_read("bogus", x=1)
        self.assertFalse(out["ok"])
        self.assertEqual(self.stub.calls, [])              # nothing reached the engine

    def test_get_history_reads_resolved_store(self):
        # the seam reads the env-resolved store (the production path) — isolate via the env var
        prev = os.environ.get("GUTCHECK_DATA_DIR")
        os.environ["GUTCHECK_DATA_DIR"] = tempfile.mkdtemp()
        try:
            history._INITIALIZED.discard(str(paths.history_db()))   # fresh WAL on the redirected path
            history.record_hand("sA", _rec(1), path=paths.history_db())
            self.assertEqual(self.s.get_history(0)["count"], 1)
        finally:
            if prev is None:
                os.environ.pop("GUTCHECK_DATA_DIR", None)
            else:
                os.environ["GUTCHECK_DATA_DIR"] = prev


class TestEndToEnd(unittest.TestCase):
    """A real Game settling hands writes rows to its INJECTED history store (the hook fires)."""

    def test_settled_hands_land_in_history(self):
        random.seed(7)
        d = Path(tempfile.mkdtemp())
        hist = d / "history.db"
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", hist)
        g.sit_down(hero="Phill")
        for _ in range(20):                               # call/check down to settle hands
            g.new_hand()
            guard = 0
            while True:
                v = g.view()
                if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                    break
                o = v.get("options") or {}
                if o.get("is_check"):
                    g.act("check")
                elif o.get("check_or_call"):
                    g.act("call")
                else:
                    break
                guard += 1
                if guard > 30:
                    break
        out = history.get_history(0, path=hist)
        self.assertGreater(out["count"], 0,
                           "no hands persisted — the _accumulate_session history hook did not fire")
        sid = g.s["session"]["id"]
        for h in out["hands"]:
            self.assertEqual(h["session_id"], sid)        # every row belongs to the live session
            self.assertIn("seed", h)
            self.assertTrue(h["log"])                      # replayable: non-empty action log
        # the watermark advances and a second poll from it sees no new hands
        self.assertEqual(history.get_history(out["watermark"], path=hist)["count"], 0)

    def test_seam_commit_read_binds_to_real_game(self):
        # L1 MED-1: the stub-based TestSeam can't catch signature drift between the seam's
        # commit_read(**kw) and the REAL engine methods (a renamed param would break production yet
        # keep the stub tests green). Exercise the real keyword binding through a real injected Game.
        d = Path(tempfile.mkdtemp())
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", d / "history.db")
        s = seam_mod.Seam(g)
        # review: kwargs must bind to record_review_read(session_id, hand_no, n, gut, lean). A missing
        # hand returns a structured miss — proving the call reached the method's LOGIC (no TypeError).
        out_r = s.commit_read("review", session_id="nope", hand_no=1, n=1, gut=50, lean="call")
        self.assertIn("ok", out_r)
        self.assertFalse(out_r["ok"])      # "hand not found", but bound correctly (not an arg error)
        # live: kwargs must bind to record_live_read(gut, lean, truth_pct, ...). This path succeeds.
        out_l = s.commit_read("live", gut=55, lean="call", truth_pct=60.0)
        self.assertTrue(out_l["ok"])

    def test_seam_get_history_uses_injected_game_path(self):
        # codex L3 HIGH: Seam.get_history must read the SAME store the Game writes to (game.history_db),
        # NOT re-resolve the env-default — else an isolated Game reads/creates real state and returns
        # stale/empty history. Fresh injected path, NO GUTCHECK_DATA_DIR override: the seam must follow it.
        d = Path(tempfile.mkdtemp())
        hist = d / "history.db"
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", hist)
        s = seam_mod.Seam(g)
        history.record_hand("SEAM-INJECT", _rec(1), path=hist)   # write ONLY to the injected store
        out = s.get_history(0)
        self.assertEqual(out["count"], 1)                        # the seam read the injected store...
        self.assertEqual(out["hands"][0]["session_id"], "SEAM-INJECT")   # ...not the env default

    def test_seam_get_history_session_filter_passes_through(self):
        # P3(d): the seam forwards session_id to the store so the MCP/HTTP "replay session X" works end-to-end
        d = Path(tempfile.mkdtemp())
        hist = d / "history.db"
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", hist)
        s = seam_mod.Seam(g)
        history.record_hand("S1", _rec(1), path=hist)
        history.record_hand("S2", _rec(1), path=hist)
        self.assertEqual(s.get_history(0)["count"], 2)                       # no scope → all
        self.assertEqual([h["session_id"] for h in s.get_history(0, session_id="S1")["hands"]], ["S1"])
        self.assertEqual(s.get_history(0, session_id="S2")["count"], 1)

    def test_history_db_is_isolated_not_the_real_store(self):
        # guards the test-isolation contract: a Game with an injected history path must NOT
        # write to the default ~/.gutcheck/history.db (the 2026-06-18 real-state-corruption class)
        d = Path(tempfile.mkdtemp())
        hist = d / "history.db"
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", hist)
        self.assertEqual(g.history_db, hist)               # injected path wins over paths.history_db()


if __name__ == "__main__":
    unittest.main(verbosity=2)
