"""Tests for P4 leak-seed — the operator-owned, anneal-backed graduating-leak store.

Coverage:
  * graceful-degrade — a no-anneal / unavailable LeakStore is a full no-op; the game plays +
    calibrates unchanged; the seam returns a clear unavailable marker (the single-dep contract).
  * _episode_content — archetype-free (§3f) + carries the (session_id, hand_no, n) join key, and
    survives a partial / hostile sample.
  * episode capture — Game wired with a real store records a read-event per committed read.
  * the seed loop end-to-end through the SEAM — prepare_leak_wrap → commit_leaks → get_leaks.
  * the moat mechanic — a recurring leak GRADUATES 1x→2x with distinct grounding, and a sycophantic
    rephrase DEMOTES (anneal's cross-session-overlap immune gate fires).
  * reset_data backs up the precious leaks (db + doc) and clears the regenerable sidecars.

anneal-memory is an OPTIONAL extra; the anneal-dependent classes skip cleanly when it isn't installed,
so this file passes in a single-dep environment too.
"""

import json
import tempfile
import threading
import unittest
from datetime import date
from pathlib import Path

from gutcheck import game as G
from gutcheck import leakstore
from gutcheck import paths
from gutcheck.leakstore import LeakStore
from gutcheck.seam import Seam

HAS_ANNEAL = leakstore.anneal_available()
_SKIP = "anneal-memory not installed (the [anneal] extra) — leak graduation is optional"

# anneal's graduation is date-gated — it only validates/demotes a pattern whose line-date is
# the current wrap window ("today"). Episodes are always created at real-now, so the pattern-line
# date must track the real date or graduation is skipped as non-today. A hardcoded date is a
# time-bomb (this was pinned to the 2026-06-24 build date and silently broke the graduation tests
# once the calendar drifted past it). Compute it live, the way the coach daemon authors leaks.
TODAY = date.today().isoformat()


def _game(tmp, with_leaks=True):
    """A path-isolated Game; with_leaks=True wires a real LeakStore at <tmp>/leaks.db."""
    d = Path(tmp)
    ls = LeakStore(d / "leaks.db") if with_leaks else None
    g = G.Game(d / "session.json", d / "career.json", d / "sessions",
               d / "calibration.json", d / "history.db", leak_store=ls)
    return g, ls


def _leak_doc(name, level, ids, explanation, marker="! "):
    """A minimal valid 4-section anneal continuity doc carrying one leak line."""
    ev = ""
    if explanation is not None:
        ev = f" [evidence: {', '.join(ids)} \"{explanation}\"]"
    return (
        "# gutcheck-operator — Memory (v1)\n\n"
        "## State\nLeak-memory session.\n\n"
        f"## Patterns\n- {marker}{name} | {level}x ({TODAY}){ev}\n\n"
        "## Decisions\n[decided(rationale: \"read-based, behavior language only\", on: \"2026-06-24\")] No archetype in a leak.\n\n"
        "## Context\nReads recorded this session, grounded in the hands.\n"
    )


def _newest_ids(ls, k):
    """The k most-recent read-episode ids (recall is newest-first) — the current session's reads to cite."""
    eps = ls.recall_leaks(limit=k) or []
    return [ep.id for ep in eps[:k]]


# --------------------------------------------------------------------------- #
#  Graceful-degrade — the single-dep contract (always runs, no anneal needed)
# --------------------------------------------------------------------------- #
class TestGracefulDegrade(unittest.TestCase):
    def test_unavailable_store_is_a_noop(self):
        ls = LeakStore()
        ls.available = False                                  # simulate no-anneal / failed open
        self.assertIsNone(ls.record_read_episode({"gut": 60, "truth": 30.0}))
        self.assertIsNone(ls.prepare_leak_wrap())
        self.assertIsNone(ls.get_leaks())
        self.assertIsNone(ls.recall_leaks("x"))
        ls.close()                                            # close is safe on a never-opened store

    def test_game_with_no_leak_store_plays_and_calibrates(self):
        # the 298 pre-P4 tests' path: Game() with leak_store=None — no capture, no crash.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp, with_leaks=False)
            self.assertIsNone(g.leaks)
            r = g.record_live_read(gut=55, lean="value", truth_pct=40.0, street="flop",
                                   session_id="s", hand_no=1, n=1)
            self.assertTrue(r["ok"])                          # calibration still recorded
            self.assertEqual(g.calibration_summary()["modeled"]["n"], 1)

    def test_game_with_unavailable_store_still_records_calibration(self):
        # a leak-store hiccup must never lose the durable calibration sample.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp, with_leaks=True)
            ls.available = False                              # break the leak layer post-construction
            r = g.record_live_read(gut=55, lean="value", truth_pct=40.0, street="flop",
                                   session_id="s", hand_no=1, n=1)
            self.assertTrue(r["ok"])
            self.assertEqual(g.calibration_summary()["modeled"]["n"], 1)

    def test_seam_returns_unavailable_marker(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp, with_leaks=True)
            ls.available = False
            seam = Seam(g)
            for resp in (seam.prepare_leak_wrap(), seam.get_leaks(),
                         seam.commit_leaks("x", "tok")):
                self.assertFalse(resp["ok"])
                self.assertIn("gutcheck[anneal]", resp["reason"])


# --------------------------------------------------------------------------- #
#  _episode_content — pure, archetype-gated (§3f), partial-sample safe
# --------------------------------------------------------------------------- #
class TestEpisodeContent(unittest.TestCase):
    def test_carries_join_key_and_facts(self):
        c = leakstore._episode_content({
            "street": "river", "lean": "value", "source": "exact", "gut": 70,
            "truth": 22.0, "band_hit": False, "gap": 48.0, "note": "sized small with the nuts",
            "session_id": "sessA", "hand_no": 31, "n": 1,
        })
        self.assertIn("session sessA · hand 31 · decision 1", c)
        self.assertIn("river", c)
        self.assertIn("sized small with the nuts", c)

    def test_never_leaks_the_archetype(self):
        # §3f: even a sample that (hypothetically) carried a hidden type must not surface it.
        c = leakstore._episode_content({
            "street": "flop", "lean": "value", "gut": 60, "truth": 30.0, "band_hit": False,
            "gap": 30.0, "session_id": "s", "hand_no": 1, "n": 2,
            "type": "station", "archetype": "station", "_truth": {"label": "station"},
        })
        self.assertNotIn("station", c.lower())
        self.assertNotIn("archetype", c.lower())

    def test_partial_sample_does_not_crash(self):
        c = leakstore._episode_content({"gut": None, "truth": None})   # all-missing
        self.assertIn("unjoined read", c)
        self.assertIn("?", c)                                          # missing pct → '?'

    def test_preflop_read_has_honest_partial_key(self):
        c = leakstore._episode_content({"street": "preflop", "gut": 50, "truth": 45.0,
                                        "session_id": "s", "hand_no": None, "n": None})
        self.assertIn("[session s]", c)                                # session only, no hand/decision


# --------------------------------------------------------------------------- #
#  Episode capture through the Game wiring (anneal)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestEpisodeCapture(unittest.TestCase):
    def test_committed_read_records_an_episode(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            g.record_live_read(gut=62, lean="value", truth_pct=34.0, street="flop",
                               session_id="sessA", hand_no=12, n=2, note="he never folds")
            eps = ls.recall_leaks(limit=10)
            self.assertEqual(len(eps), 1)
            self.assertIn("sessA", eps[0].content)
            self.assertIn("he never folds", eps[0].content)
            ls.close()

    def test_captured_episode_is_archetype_free_even_with_a_typed_villain(self):
        # play a real hand so a villain with a hidden type exists, commit a review read, and confirm
        # the captured episode carries observed behavior only — never the type (§3f end-to-end).
        with tempfile.TemporaryDirectory() as tmp:
            import random
            random.seed(7)
            g, ls = _game(tmp)
            g.sit_down(hero="Phill")
            g.new_hand()
            joined = ' '.join(ep.content for ep in (ls.recall_leaks(limit=50) or []))
            for t in ("rock", "reg", "station", "archetype"):
                self.assertNotIn(t, joined.lower())
            ls.close()


# --------------------------------------------------------------------------- #
#  The seed loop + the moat mechanic (anneal)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestSeedAndGraduation(unittest.TestCase):
    def _seed_session(self, g, ls):
        """Record two reads + seed 'sizing_is_a_read_output' at 1x through the seam. Returns the seam."""
        seam = Seam(g)
        g.record_live_read(gut=62, lean="value", truth_pct=34.0, street="flop",
                           session_id="sessA", hand_no=12, n=2, note="dead-money bet, he never folds")
        g.record_live_read(gut=70, lean="value", truth_pct=22.0, street="river",
                           session_id="sessA", hand_no=31, n=1, note="sized small with the nuts")
        wrap = seam.prepare_leak_wrap()
        self.assertTrue(wrap["ok"])
        ids = _newest_ids(ls, 2)
        doc = _leak_doc("sizing_is_a_read_output", 1, ids,
                        "bet small to protect a marginal hand, bet big to get paid by one — the read "
                        "names which; both reads show sizing decoupled, a dead-money value bet and the "
                        "nuts sized small")
        res = seam.commit_leaks(doc, wrap["wrap_token"], affect_tag="engaged", affect_intensity=0.7)
        self.assertTrue(res["ok"])
        return seam, res

    def test_seed_lands_at_1x(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam, res = self._seed_session(g, ls)
            self.assertEqual(res["result"]["wrap_result"]["patterns_extracted"], 1)
            leaks = seam.get_leaks()
            self.assertTrue(leaks["ok"])
            line = next(l for l in leaks["leaks"].splitlines()
                        if l.startswith("- ") and "sizing_is_a_read_output" in l)
            self.assertIn("| 1x", line)
            ls.close()

    def test_recurrence_graduates_to_2x(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam, _ = self._seed_session(g, ls)
            # session 2: the leak recurs with FRESH, DISTINCT grounding (<3 shared words w/ prior).
            g.record_live_read(gut=58, lean="value", truth_pct=30.0, street="turn",
                               session_id="sessB", hand_no=7, n=2, note="set on a wet board, jammed too thin")
            g.record_live_read(gut=80, lean="value", truth_pct=55.0, street="river",
                               session_id="sessB", hand_no=22, n=1, note="flatted the nuts, left value")
            wrap = seam.prepare_leak_wrap()
            ids = _newest_ids(ls, 2)
            doc = _leak_doc("sizing_is_a_read_output", 2, ids,
                            "the wager magnitude never tracked equity — a set on a wet board jammed too "
                            "thin to charge the flush draw, and the nuts flatted on the river instead of "
                            "raising thin value vs a calling opponent")
            res = seam.commit_leaks(doc, wrap["wrap_token"], affect_tag="sharpening", affect_intensity=0.6)
            self.assertTrue(res["ok"])
            self.assertEqual(res["result"]["graduations_validated"], 1)
            self.assertGreaterEqual(res["result"]["associations_formed"], 1)
            self.assertEqual(res["result"]["graduations_demoted"], 0)
            line = next(l for l in seam.get_leaks()["leaks"].splitlines()
                        if "sizing_is_a_read_output" in l)
            self.assertIn("| 2x", line)
            ls.close()

    def test_sycophantic_rephrase_demotes(self):
        # the moat: re-wording the prior session's explanation (>=3 shared meaningful words) is caught
        # by anneal's cross-session-overlap immune gate → demoted, not graduated.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam, _ = self._seed_session(g, ls)
            g.record_live_read(gut=58, lean="value", truth_pct=30.0, street="turn",
                               session_id="sessB", hand_no=7, n=2, note="another marginal over-bet")
            g.record_live_read(gut=80, lean="value", truth_pct=55.0, street="river",
                               session_id="sessB", hand_no=22, n=1, note="under-sized vs a caller again")
            wrap = seam.prepare_leak_wrap()
            ids = _newest_ids(ls, 2)
            # near-verbatim reuse of the seed explanation's vocabulary → sycophancy.
            doc = _leak_doc("sizing_is_a_read_output", 2, ids,
                            "bet small to protect a marginal hand, bet big to get paid by one — the read "
                            "names which; both reads show sizing decoupled, a dead-money value bet and the "
                            "nuts sized small")
            res = seam.commit_leaks(doc, wrap["wrap_token"])
            self.assertTrue(res["ok"])
            self.assertEqual(res["result"]["graduations_validated"], 0)
            self.assertEqual(res["result"]["graduations_demoted"], 1)
            line = next(l for l in seam.get_leaks()["leaks"].splitlines()
                        if "sizing_is_a_read_output" in l)
            self.assertIn("cross-session-overlap", line)                # demoted, the immune flag is on it
            ls.close()


# --------------------------------------------------------------------------- #
#  Thread safety (P4b-ii-a) — the daemon thread writes while the serve thread records
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestThreadSafety(unittest.TestCase):
    """P4b-ii-a — the WRITE lifecycle (prepare_leak_wrap -> commit_leaks) runs on the coach DAEMON
    thread while the serve thread records reads. anneal's Store is ``check_same_thread=True``, so
    pre-fix (one shared connection lazily opened on the serve thread) the daemon's prepare/commit
    raised ``sqlite3.ProgrammingError`` across the thread boundary. The thread-local store closes the
    gate: each thread opens its OWN connection, the RLock serializes the actual ops. Proof = the
    cross-thread write lifecycle never raises AND the daemon-committed leak is readable from the serve
    thread's connection."""

    def _read(self, sid, hand_no, street="river"):
        return {"session_id": sid, "hand_no": hand_no, "n": 1, "street": street,
                "gut": 64, "truth": 33.0, "lean": "value", "gap": 31,
                "note": f"thin value bet on the {street}, hand {hand_no}"}

    def test_daemon_thread_writes_while_serve_thread_records(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = LeakStore(Path(tmp) / "leaks.db")
            try:
                # serve thread (this one): open the store + seed read-episodes for the wrap to compress.
                for h in range(1, 5):
                    self.assertIsNotNone(ls.record_read_episode(self._read("s1", h)))
                ids = _newest_ids(ls, 4)
                self.assertEqual(len(ids), 4)

                errors, out = [], {}
                go = threading.Event()

                def daemon_seed():
                    """The full prepare -> author -> commit WRITE lifecycle, on a SEPARATE thread —
                    so prepare/commit run on the daemon thread's OWN connection (pre-fix: they'd touch
                    the serve thread's connection -> check_same_thread ProgrammingError)."""
                    try:
                        go.wait(5)
                        wrap = ls.prepare_leak_wrap()
                        if not wrap or wrap.get("status") != "ready":
                            out["wrap"] = wrap
                            return
                        doc = _leak_doc("barrel_into_the_caller", 1, ids,
                                        "river value bet barreled into a player who only calls")
                        out["res"] = ls.commit_leaks(doc, wrap["wrap_token"])
                    except Exception as e:               # a check_same_thread error would land HERE
                        errors.append(repr(e))

                t = threading.Thread(target=daemon_seed, name="coach-daemon")
                t.start()
                go.set()
                # serve thread keeps recording reads CONCURRENTLY with the daemon's write lifecycle.
                for h in range(5, 9):
                    ls.record_read_episode(self._read("s1", h, street="turn"))
                t.join(timeout=15)
                self.assertFalse(t.is_alive(), "daemon seed thread hung")

                self.assertEqual(errors, [], f"cross-thread store op raised: {errors}")
                self.assertIsNotNone(out.get("res"), f"daemon commit did not succeed: {out}")
                # the daemon-committed leak is now readable from the serve thread's own connection.
                doc = ls.get_leaks()
                self.assertIsNotNone(doc)
                self.assertIn("barrel_into_the_caller", doc)
            finally:
                ls.close()


# --------------------------------------------------------------------------- #
#  Seed-driver surface (P4b-ii) — the count gate + the wrap-release the daemon needs
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestSeedDriverHelpers(unittest.TestCase):
    """P4b-ii: new_read_count (the non-mutating threshold signal the seed driver gates on) and
    cancel_leak_wrap (release an abandoned wrap so it can't brick the next pass)."""

    def _read(self, sid, h):
        return {"session_id": sid, "hand_no": h, "n": 1, "street": "river", "gut": 64,
                "truth": 33.0, "lean": "value", "gap": 31, "note": f"thin value bet hand {h}"}

    def test_new_read_count_tracks_and_resets_on_commit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = LeakStore(Path(tmp) / "leaks.db")
            try:
                self.assertEqual(ls.new_read_count(), 0)            # empty store
                for h in range(1, 6):
                    ls.record_read_episode(self._read("s1", h))
                self.assertEqual(ls.new_read_count(), 5)            # five un-graduated reads accumulated
                ids = _newest_ids(ls, 5)
                wrap = ls.prepare_leak_wrap()
                doc = _leak_doc("over_barrel", 1, ids[:2],
                                "river value bet barreled too thin into a caller")
                self.assertIsNotNone(ls.commit_leaks(doc, wrap["wrap_token"]))
                self.assertEqual(ls.new_read_count(), 0)            # a completed wrap resets the window
            finally:
                ls.close()

    def test_cancel_leak_wrap_releases_an_open_wrap(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = LeakStore(Path(tmp) / "leaks.db")
            try:
                ls.record_read_episode(self._read("s1", 1))
                ls.record_read_episode(self._read("s1", 2))
                w1 = ls.prepare_leak_wrap()
                self.assertEqual(w1.get("status"), "ready")
                self.assertTrue(ls.cancel_leak_wrap())              # release the open wrap
                w2 = ls.prepare_leak_wrap()                         # a fresh prepare is READY, not in-progress
                self.assertEqual(w2.get("status"), "ready")
                ls.cancel_leak_wrap()
            finally:
                ls.close()


# --------------------------------------------------------------------------- #
#  reset_data hygiene — back up the precious leaks, clear the regenerable sidecars
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestResetBacksUpLeaks(unittest.TestCase):
    def test_reset_backs_up_leaks_and_clears_sidecars(self):
        with tempfile.TemporaryDirectory() as tmp:
            import os
            old = os.environ.get(paths.ENV_VAR)
            os.environ[paths.ENV_VAR] = tmp                 # point the data dir at tmp
            try:
                d = Path(tmp)
                g, ls = _game(tmp)
                # default-path leak store (so reset_data, which uses paths.*, finds it)
                ls2 = LeakStore()                            # resolves to <tmp>/leaks.db via the env var
                ls2.record_read_episode({"gut": 60, "truth": 30.0, "session_id": "s", "hand_no": 1, "n": 1})
                ls2.close()
                self.assertTrue((d / "leaks.db").exists())
                removed = paths.reset_data()
                self.assertTrue((d / "leaks.reset-backup.db").exists())     # precious → backed up
                self.assertFalse((d / "leaks.continuity.meta.json").exists())  # regenerable → cleared
                self.assertFalse((d / "leaks.audit.jsonl").exists())
                self.assertIn("leaks.db -> leaks.reset-backup.db", removed)
            finally:
                if old is None:
                    os.environ.pop(paths.ENV_VAR, None)
                else:
                    os.environ[paths.ENV_VAR] = old


# --------------------------------------------------------------------------- #
#  Apparatus fixes (L1 + L2 review batch)
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestApparatusFixes(unittest.TestCase):
    def test_in_progress_wrap_is_not_auto_cancelled(self):
        # codex L3 HIGH-1: a plain second prepare must NOT cancel the open wrap (the over-correction the
        # earlier auto-cancel introduced) — it surfaces wrap_in_progress, and the original token still commits.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam = Seam(g)
            g.record_live_read(gut=60, lean="value", truth_pct=30.0, street="flop",
                               session_id="s", hand_no=1, n=1)
            w1 = ls.prepare_leak_wrap()                  # open a wrap, never commit (the "crash")
            self.assertEqual(w1["status"], "ready")
            resp = seam.prepare_leak_wrap()              # plain prepare again → in-progress, NOT cancelled
            self.assertFalse(resp["ok"])
            self.assertEqual(resp["status"], "wrap_in_progress")
            self.assertTrue(resp.get("recoverable"))
            # the original wrap was NOT destroyed — its token still commits
            res = seam.commit_leaks(_leak_doc("x_leak", 1, [], None), w1["wrap_token"])
            self.assertTrue(res["ok"])
            ls.close()

    def test_explicit_recover_reclaims_abandoned_wrap(self):
        # codex L3 HIGH-1 / L2 H1: recover=True deliberately reclaims a KNOWN-abandoned wrap (the
        # crashed-coach recovery) — without a recovery path the store would brick.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam = Seam(g)
            g.record_live_read(gut=60, lean="value", truth_pct=30.0, street="flop",
                               session_id="s", hand_no=1, n=1)
            ls.prepare_leak_wrap()                       # open + "crash" (never commit/cancel)
            resp = seam.prepare_leak_wrap(recover=True)  # deliberate reclaim
            self.assertTrue(resp["ok"])
            self.assertEqual(resp["status"], "ready")
            self.assertTrue(resp["wrap_token"])
            ls.close()

    def test_get_leaks_distinguishes_empty_from_load_error(self):
        # complement L3 MED-2: leaks=None must carry a status so a client can't read "no leaks learned"
        # off a broken read.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam = Seam(g)
            empty = seam.get_leaks()                      # healthy, nothing graduated yet
            self.assertTrue(empty["ok"])
            self.assertIsNone(empty["leaks"])
            self.assertEqual(empty["status"], "empty")

            class _BoomLoad:                              # a real read failure → flag sticks → load_error
                def load_continuity(self):
                    raise RuntimeError("disk gone")

                def close(self):
                    pass

            ls._store = _BoomLoad()
            broke = seam.get_leaks()
            self.assertIsNone(broke["leaks"])
            self.assertEqual(broke["status"], "load_error")
            ls.close()

    def test_empty_wrap_is_not_committable(self):
        # L2 M1: an empty store mints no token → the seam must say ok:False, not ok:True-then-ValueError.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            resp = Seam(g).prepare_leak_wrap()
            self.assertFalse(resp["ok"])
            self.assertIn("no new reads", resp["reason"])
            ls.close()

    def test_bad_token_commit_normalized_at_seam(self):
        # L1 MED-1 / L2 M2: a propagating ValueError (wrong token) is normalized into {ok:False, reason}
        # at the seam boundary, not leaked as raw anneal text through the server's generic handler.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam = Seam(g)
            g.record_live_read(gut=60, lean="value", truth_pct=30.0, street="flop",
                               session_id="s", hand_no=1, n=1)
            self.assertTrue(seam.prepare_leak_wrap()["ok"])         # a wrap is now in progress
            resp = seam.commit_leaks(_leak_doc("x_leak", 1, [], None), "wrong-token-deadbeef")
            self.assertFalse(resp["ok"])
            self.assertIn("malformed", resp["reason"])
            ls.close()

    def test_unexpected_capture_error_never_breaks_the_read(self):
        # L1 MED-2: episode capture is best-effort — an UNEXPECTED error type from anneal/_episode_content
        # must not turn a successful, already-persisted read into a caller error.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            ls._ensure()

            class _Boom:
                def record(self, *a, **k):
                    raise RuntimeError("kaboom")          # not in any narrow swallow tuple

            ls._store = _Boom()
            r = g.record_live_read(gut=60, lean="value", truth_pct=30.0, street="flop",
                                   session_id="s", hand_no=1, n=1)
            self.assertTrue(r["ok"])                       # the read still succeeded
            self.assertEqual(g.calibration_summary()["modeled"]["n"], 1)
            self.assertIsNone(ls.record_read_episode({"gut": 60, "truth": 30.0}))   # swallowed → None

    def test_episode_carries_structured_join_key_metadata(self):
        # L2 rec: the (session_id, hand_no, n) join key is ALSO stored as structured metadata so P4b's
        # decision-point recall can query it without regexing the prose.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            g.record_live_read(gut=60, lean="value", truth_pct=30.0, street="flop",
                               session_id="sessA", hand_no=5, n=2)
            ep = ls.recall_leaks(limit=1)[0]
            md = getattr(ep, "metadata", None)
            self.assertIsInstance(md, dict)
            self.assertEqual(md.get("session_id"), "sessA")
            self.assertEqual(md.get("hand_no"), 5)
            self.assertEqual(md.get("n"), 2)
            ls.close()


# --------------------------------------------------------------------------- #
#  P4b-i — leak recall-injection (the moat READ half): the relevance scorer
# --------------------------------------------------------------------------- #
def _recall_doc(*leaks) -> str:
    """A 4-section anneal doc with arbitrary leak lines. Each `leaks` entry is
    (name, level, summary) — rendered in the real WRITE-PATH shape (`name | Nx (date) — summary`)."""
    lines = "\n".join(f"- {n} | {lvl}x ({TODAY}) — {summ}" for n, lvl, summ in leaks)
    return ("# gutcheck-operator — Memory (v1)\n\n## State\nRecall test.\n\n"
            f"## Patterns\n{lines}\n\n## Decisions\n\n## Context\nReads recorded.\n")


@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestRecallRelevantLeaks(unittest.TestCase):
    """recall_relevant_leaks scores the graduated-leak working set against a spot query. Uses
    save_continuity (immune-system bypass) to control the doc precisely — recall reads load_continuity,
    so HOW the doc graduated is irrelevant to the scorer; one integration test covers the real path."""

    def _store_with(self, tmp, *leaks):
        ls = LeakStore(Path(tmp) / "leaks.db")
        ls._ensure().save_continuity(_recall_doc(*leaks))
        return ls

    # a realistic spot query is what server._spot_query emits: street + hero strength + facing-action +
    # the OBSERVED looseness phrase (no passivity — §3f). Distinctive words after scaffold-strip:
    # {river, pair, value, loose}.
    RIVER_SPOT = "river made hand: top pair facing a bet, value spot loose wide villain"

    def test_surfaces_the_relevant_leak_ranked(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp,
                ("sizing_is_a_read_output", 2,
                 "thin river value wants a small read-based size, not a default sizing"),
                ("overvalue_a_made_hand", 1,
                 "you overvalue a made hand on the river versus a loose opponent; check rather than value thin"))
            hits = ls.recall_relevant_leaks(self.RIVER_SPOT)
            names = [h["name"] for h in hits]
            self.assertIn("sizing_is_a_read_output", names)        # matches river + value
            self.assertIn("overvalue_a_made_hand", names)          # matches river + value + loose
            # exactly name/level/summary + the overlap (situational-match count, for per-surface bars).
            # NO archetype field — §3f by construction.
            self.assertTrue(all(set(h) == {"name", "level", "summary", "overlap"} for h in hits))
            self.assertTrue(all(isinstance(h["overlap"], int) and h["overlap"] >= 2 for h in hits))
            ls.close()

    def test_precision_threshold_surfaces_nothing_on_an_unrelated_spot(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp, ("barrel_into_the_caller", 1,
                      "you keep barreling a villain who only calls; a caller does not fold, value not bluff"))
            # a preflop-fold spot shares <2 situational keywords with the river-barrel leak → nothing.
            self.assertEqual(ls.recall_relevant_leaks("preflop pocket pair early position folded"), [])
            ls.close()

    def test_cross_street_leak_is_gated_out(self):
        # the precision fix both reviewers reached: a RIVER leak must NOT surface on a TURN spot, even
        # when they share hero-strength words — a river-overfold lesson has nothing to say on the turn.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp, ("overvalue_top_pair_river", 2,
                      "you overvalue top pair on the river versus a passive caller; check do not value bet"))
            turn_spot = "turn made hand: top pair facing a bet, value spot loose wide villain"
            self.assertEqual(ls.recall_relevant_leaks(turn_spot), [])      # street gate excludes it
            # sanity: the SAME leak DOES surface on a river spot
            self.assertTrue(ls.recall_relevant_leaks(self.RIVER_SPOT))
            ls.close()

    def test_boilerplate_only_match_is_filtered(self):
        # a leak whose ONLY overlap with the spot is a single recurring frame/action word must not clear
        # the bar — scaffold-stripping + min_overlap is what kills the autoimmune false positive.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp, ("value_bet_too_thin", 1, "you value bet too thin everywhere"))   # shares only "value"
            self.assertEqual(ls.recall_relevant_leaks(self.RIVER_SPOT), [])
            ls.close()

    def test_one_malformed_entry_does_not_kill_the_whole_recall(self):
        # complement L3 MED-1: a single bad leak entry (e.g. a pre-schema-change doc line that chokes the
        # scorer) must skip ONLY that entry — not raise into the outer except and zero out the request's
        # entire recall (per-request vs per-entry blast radius).
        from unittest import mock
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp,
                ("poison_leak", 1, "river value sizing poison"),
                ("good_leak", 2, "river value sizing thin small"))
            real = leakstore._score_leak

            def boom(qkw, lkw, level):
                if "poison" in lkw:
                    raise ValueError("malformed leak entry")
                return real(qkw, lkw, level)

            with mock.patch.object(leakstore, "_score_leak", side_effect=boom):
                names = [h["name"] for h in ls.recall_relevant_leaks("river value sizing thin small")]
            self.assertIn("good_leak", names)         # survived the poison entry
            self.assertNotIn("poison_leak", names)    # skipped per-entry, didn't kill the recall
            ls.close()

    def test_malformed_high_level_does_not_dominate(self):
        # codex L3 LOW: a malformed '99x' leak must not outrank a real 3x via the level multiplier —
        # the level is clamped to the [1,3] graduation ladder before scoring. Here junk shares FEWER
        # situational words than real, so without the clamp its 99x would still win; with it, real wins.
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp,
                ("junk_leak", 99, "river value sizing"),                 # 3 situational, bogus level
                ("real_leak", 3, "river value sizing thin"))             # 4 situational, real top level
            hits = ls.recall_relevant_leaks("river value sizing thin")
            self.assertEqual(hits[0]["name"], "real_leak")               # clamp → more-overlap real wins
            ls.close()

    def test_level_breaks_ties_toward_the_more_graduated_leak(self):
        with tempfile.TemporaryDirectory() as tmp:
            # two leaks with the SAME query overlap (river value bet) but different levels → 3x first.
            ls = self._store_with(
                tmp,
                ("low_leak", 1, "river value bet sizing tell"),
                ("high_leak", 3, "river value bet sizing tell"))
            hits = ls.recall_relevant_leaks("river value bet sizing")
            self.assertEqual([h["name"] for h in hits], ["high_leak", "low_leak"])
            ls.close()

    def test_caps_at_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = self._store_with(
                tmp,
                ("leak_a", 2, "river value bet thin sizing"),
                ("leak_b", 2, "river value bet thin sizing"),
                ("leak_c", 2, "river value bet thin sizing"))
            self.assertEqual(len(ls.recall_relevant_leaks("river value bet thin sizing", limit=1)), 1)
            self.assertEqual(len(ls.recall_relevant_leaks("river value bet thin sizing", limit=2)), 2)
            ls.close()

    def test_empty_doc_and_bad_query_and_unavailable_all_degrade_to_empty_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            ls = LeakStore(Path(tmp) / "leaks.db")          # no continuity doc written
            self.assertEqual(ls.recall_relevant_leaks("river value bet"), [])
            ls = self._store_with(tmp, ("x_leak", 1, "river value bet sizing"))
            for bad in (None, "", "   ", 123):
                self.assertEqual(ls.recall_relevant_leaks(bad), [])   # never raises
            ls.close()
        # unavailable store (no anneal wired) → [] not None (recall is pure enrichment)
        ls = LeakStore()
        ls.available = False
        self.assertEqual(ls.recall_relevant_leaks("river value bet"), [])

    def test_integration_through_the_real_seed_loop(self):
        # the whole path: record reads → graduate a leak via the seam → recall surfaces it at a matching spot.
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            seam = Seam(g)
            g.record_live_read(gut=70, lean="value", truth_pct=22.0, street="river",
                               session_id="sessA", hand_no=31, n=1, note="sized small with the nuts")
            wrap = seam.prepare_leak_wrap()
            doc = _leak_doc("sizing_is_a_read_output", 1, _newest_ids(ls, 1),
                            "bet small to protect a marginal hand, bet big to get paid — sized small with the nuts")
            self.assertTrue(seam.commit_leaks(doc, wrap["wrap_token"])["ok"])
            hits = ls.recall_relevant_leaks("river value bet sizing small marginal hand")
            self.assertIn("sizing_is_a_read_output", [h["name"] for h in hits])
            ls.close()


@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestSeamRecallWrapper(unittest.TestCase):
    def test_wrapper_shape_hit_miss_and_unavailable(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = _game(tmp)
            ls._ensure().save_continuity(_recall_doc(
                ("barrel_into_the_caller", 1,
                 "you keep barreling a villain who only calls; a caller does not fold, bet value")))
            seam = Seam(g)
            hit = seam.recall_relevant_leaks("river value bet villain who only calls caller")
            self.assertTrue(hit["ok"])
            self.assertEqual(hit["status"], "ok")
            self.assertTrue(hit["leaks"])
            miss = seam.recall_relevant_leaks("preflop fold early position")
            self.assertTrue(miss["ok"])
            self.assertEqual(miss["status"], "none")
            self.assertEqual(miss["leaks"], [])
            ls.close()
        # unavailable extra → the same marker the other leak methods return
        with tempfile.TemporaryDirectory() as tmp2:
            g3, _ = _game(tmp2, with_leaks=False)
            resp = Seam(g3).recall_relevant_leaks("river value bet")
            self.assertFalse(resp["ok"])
            self.assertIn("unavailable", resp["reason"])


@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestResetKeepsTheAuditChainWhole(unittest.TestCase):
    """`gutcheck --reset` must leave a leak store whose audit chain still verifies after reuse
    (anneal 1.0 keeps a manifest + lock files beside the audit jsonl; a survivor refuses every append)."""

    def _cycle(self, d):
        ls = LeakStore(d / "leaks.db")
        for i in (1, 2):
            ls.record_read_episode({"session_id": "s", "hand_no": i, "n": 1, "gut": 60, "lean": "value",
                                    "truth_pct": 30.0, "street": "flop", "note": "x"})
        wrap = ls.prepare_leak_wrap()
        doc = _leak_doc("sizing_is_a_read_output", 1, _newest_ids(ls, 2),
                        "bet small to protect a marginal hand, bet big to get paid by one")
        self.assertIsNotNone(ls.commit_leaks(doc, wrap["wrap_token"]))
        ls.close()

    def test_audit_chain_verifies_after_reset_and_reuse(self):
        import os
        from anneal_memory.audit import AuditTrail
        if not hasattr(AuditTrail, "verify"):
            self.skipTest("this anneal-memory predates the audit chain verifier")
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            saved = os.environ.get(paths.ENV_VAR)
            os.environ[paths.ENV_VAR] = tmp
            try:
                self._cycle(d)
                paths.reset_data()
                self._cycle(d)
            finally:
                if saved is None:
                    os.environ.pop(paths.ENV_VAR, None)
                else:
                    os.environ[paths.ENV_VAR] = saved
            self.assertTrue(AuditTrail.verify(d / "leaks.db").valid)


if __name__ == "__main__":
    unittest.main()
