"""gutcheck live commit-beat suite (stdlib unittest, zero deps).

Covers the live gut-commit beat's plumbing — the analog of the review drill at the LIVE decision:
  - /api/read/commit: the server computes the modeled equity in CODE from authoritative game state
    (anti-fab — no browser truth, no coach worker), logs the calibration, hands the number back for
    the reveal; a double-commit of the same spot dedups on the frozen join key,
  - /api/read/spot + server._decision_spot (the drill render payload: has_model / no equity leak),
  - equity.modeled_from_spec (the single code compute the server commit + the worker prose share),
  - the channel's opaque `meta` column (set_meta / poll_meta + the old-db migration) — still the
    worker's reuse-once path when prose IS asked for ('talk it through').

The browser-side beat (renderLiveDrill/commitLogRead/talkItThrough) is JS, gated visually by Phill;
here we pin the contract it rides on. A real range MC runs (small sample count for speed).

    ./venv/bin/python test_live_commit.py
"""

import json
import os
import sqlite3
import tempfile
import threading
import unittest
import urllib.error
import urllib.request
from http.server import HTTPServer
from pathlib import Path

from gutcheck import coach
from gutcheck import coach_watch
from gutcheck import coach_daemon
from gutcheck import equity
from gutcheck import game as G
from gutcheck import paths
from gutcheck import seam as seam_mod
from gutcheck import server


def _modeled_ctx(samples=400):
    """A live decision context WITH a modeled_spec (what game.coach_decision_context bakes when there
    is a live opponent). Real cards + a real pokerkit range so equity.hero_vs actually computes."""
    return {
        "kind": "decision",
        "hand_no": 7,
        "hero": {"hole": ["As", "Kd"], "position": "BTN", "strength": "made hand: top pair",
                 "stack": 960},
        "board": ["Ah", "7c", "2d"],
        "street": "flop",
        "pot": 80,
        "to_call": 25,
        "pot_odds": {"call": 25, "pot": 80, "breakeven_pct": 23.8},
        "legal": {"can_fold": True, "can_check": False, "can_raise": True,
                  "min_raise_to": 50, "max_raise_to": 960},
        "action_this_hand": ["Hank bets 25"],
        "modeled_spec": {
            "hero_hole": ["As", "Kd"], "board": ["Ah", "7c", "2d"],
            "range": "22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A7o+, KTo+, QTo+, JTo",
            "samples": samples,
            "vs": {"name": "Hank", "position": "BB", "vpip": 48, "hands": 22, "raised": True,
                   "range_pct": 40},
            "live_count": 1, "heads_up": True, "pot_way": 2,
        },
        "villains": [
            {"name": "Hank", "position": "BB", "stack": 720, "in_hand": True,
             "observed": {"hands": 22, "vpip_pct": 48, "pf_raises": 2, "showdowns": 6, "net": -90},
             "_truth": {"type": "station"}},
        ],
    }


def _plain_ctx():
    """A decision context with NO modeled_spec (no live opponent to model — the rare has_model=False)."""
    c = _modeled_ctx()
    c.pop("modeled_spec")
    return c


def _drive_to_live_decision(g, postflop=False):
    """Advance a sat-down Game to a hero decision that carries a modeled_spec (a live opponent to model
    equity against), leaving the hand PAUSED there (coach_decision_context is read-only — it doesn't act).
    Returns the live ctx, or None if a bounded search finds none. Mirrors test_calibration's pattern;
    villain generation is SystemRandom (unseedable), so we drive until a has-model spot appears rather
    than reproducing a specific one — any such spot exercises the same /api/read/commit path.

    `postflop=True` keeps check/calling THROUGH preflop until a POSTFLOP spot (n is not None): the very
    first has-model decision is always preflop (n=None), where the join key — and so the join-key dedup —
    can't be exercised (L1 MED-1: the dedup ships untested if the helper stops preflop)."""
    for _ in range(200):
        g.new_hand()
        guard = 0
        while True:
            v = g.view()
            if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                break
            ctx = g.coach_decision_context()
            if ctx is not None and ctx.get("modeled_spec") and (not postflop or ctx.get("n") is not None):
                return ctx
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
    return None


class TestReadCommitRoute(unittest.TestCase):
    """The /api/read/commit ROUTE: the live read is logged against the modeled equity the SERVER computes
    in CODE from authoritative game state (anti-fab — no browser-supplied truth, no coach worker). Drives a
    REAL in-process server to a live hero decision, then POSTs reads. Guards the seam end-to-end: the note
    forwards, the equity comes back for the reveal, and a double-commit of the same spot dedups."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._prev_dir = os.environ.get("GUTCHECK_DATA_DIR")
        os.environ["GUTCHECK_DATA_DIR"] = self.tmp   # the server's Game + coach channel share this dir
        paths.ensure_data_dir()
        d = Path(self.tmp)
        self.g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                        d / "calibration.json", d / "history.db")
        self.g.sit_down(hero="Phill")
        server.GAME = self.g
        server.SEAM = seam_mod.Seam(self.g)
        self.httpd = HTTPServer(("127.0.0.1", 0), server.Handler)
        self.port = self.httpd.server_address[1]
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def tearDown(self):
        self.httpd.shutdown()
        self.httpd.server_close()
        if self._prev_dir is None:
            os.environ.pop("GUTCHECK_DATA_DIR", None)
        else:
            os.environ["GUTCHECK_DATA_DIR"] = self._prev_dir

    def _post(self, path, body):
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}{path}", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        return json.loads(urllib.request.urlopen(req).read())

    def _get(self, path):
        return json.loads(urllib.request.urlopen(f"http://127.0.0.1:{self.port}{path}").read())

    def _open_read(self):
        """The browser's first step — GET /api/read/spot for the live drill payload + the spot-identity
        token the commit must round-trip. Returns (spot, token)."""
        r = self._get("/api/read/spot")
        return r.get("spot"), r.get("token")

    def test_commit_computes_equity_logs_and_forwards_note(self):
        ctx = _drive_to_live_decision(self.g, postflop=True)
        self.assertIsNotNone(ctx, "need a live POSTFLOP hero decision with a modeled opponent")
        self.assertIsNotNone(ctx["n"], "postflop drive must yield a real join-key n")
        _, token = self._open_read()
        res = self._post("/api/read/commit",
                         {"gut": 55, "lean": "call", "note": "  she snap-called, looks strong  ", "token": token})
        self.assertTrue(res.get("ok"), res)
        # the server computed the truth in CODE and handed it back for the reveal
        self.assertIsNotNone(res.get("equity"))
        self.assertIsNotNone(res["equity"].get("hero_pct"))
        recent = self.g.calibration_summary()["modeled"]["recent"]
        self.assertEqual(recent[-1]["note"], "she snap-called, looks strong")   # trimmed, through the ROUTE
        # the REAL join key was frozen from the live context (P3a reconciliation depends on it)
        self.assertEqual(recent[-1]["session_id"], ctx["session_id"])
        self.assertEqual(recent[-1]["hand_no"], ctx["hand_no"])
        self.assertEqual(recent[-1]["n"], ctx["n"])

    def test_commit_without_note_is_null(self):
        self.assertIsNotNone(_drive_to_live_decision(self.g))
        _, token = self._open_read()
        res = self._post("/api/read/commit", {"gut": 50, "lean": "check", "token": token})   # no note key
        self.assertTrue(res.get("ok"), res)
        self.assertIsNone(self.g.calibration_summary()["modeled"]["recent"][-1]["note"])

    def test_double_commit_of_same_spot_dedups(self):
        # logging the SAME still-frozen decision twice (a re-click / retry) must add ONE sample, not two —
        # the join-key dedup. Driven to POSTFLOP so n is real (a preflop n=None has no join key, by design).
        ctx = _drive_to_live_decision(self.g, postflop=True)
        self.assertIsNotNone(ctx)
        self.assertIsNotNone(ctx["n"])
        _, token = self._open_read()
        before = len(self.g.calibration_summary()["modeled"]["recent"])
        r1 = self._post("/api/read/commit", {"gut": 40, "lean": "call", "token": token})
        r2 = self._post("/api/read/commit", {"gut": 70, "lean": "fold", "token": token})   # same spot, 2nd attempt
        self.assertTrue(r1.get("ok"), r1)
        self.assertFalse(r2.get("ok"))                 # the join-key dedup fires (now actually exercised)
        self.assertEqual(r2.get("reason"), "already logged")
        self.assertEqual(len(self.g.calibration_summary()["modeled"]["recent"]), before + 1)

    def test_stale_token_is_rejected_as_spot_moved(self):
        # HIGH-1: a read committed against a DIFFERENT decision than the drill rendered (the hand advanced
        # on another tab/device) must be REJECTED, not logged against the wrong spot. A wrong token stands
        # in for "the spot moved" — the server recomputes the current token and refuses the mismatch.
        self.assertIsNotNone(_drive_to_live_decision(self.g, postflop=True))
        before = len(self.g.calibration_summary()["modeled"]["recent"])
        res = self._post("/api/read/commit", {"gut": 50, "lean": "call", "token": "deadbeefdeadbeef"})
        self.assertFalse(res.get("ok"))
        self.assertEqual(res.get("reason"), "spot moved")
        self.assertEqual(len(self.g.calibration_summary()["modeled"]["recent"]), before)   # nothing logged
        # a MISSING token is refused too (a commit that never fetched a spot)
        res2 = self._post("/api/read/commit", {"gut": 50, "lean": "call"})
        self.assertFalse(res2.get("ok"))
        self.assertEqual(res2.get("reason"), "spot moved")

    def test_commit_with_no_live_decision_is_spot_moved(self):
        # a brand-new server with no live hand → coach_decision_context() is None → the read can't be
        # calibrated (the anti-fab guard: there is no authoritative spot to recompute the truth from).
        d = Path(tempfile.mkdtemp())
        fresh = G.Game(d / "session.json", d / "career.json", d / "sessions",
                       d / "calibration.json", d / "history.db")   # NOT sat down → no decision
        server.GAME = fresh
        server.SEAM = seam_mod.Seam(fresh)
        res = self._post("/api/read/commit", {"gut": 50, "lean": "check", "token": "whatever"})
        self.assertFalse(res.get("ok"))
        self.assertEqual(res.get("reason"), "spot moved")

    def test_read_spot_returns_spot_and_token_without_equity(self):
        self.assertIsNotNone(_drive_to_live_decision(self.g))
        spot, token = self._open_read()
        self.assertTrue(spot["has_model"])
        self.assertTrue(isinstance(token, str) and token)   # the spot-identity token is handed out
        # the live drill's lean buttons read spot.legal.can_raise — without it the raise lean silently
        # drops when facing a bet (codex L3 MED: _decision_spot never carried `legal`).
        self.assertIn("legal", spot)
        self.assertIn("can_raise", spot["legal"])
        self.assertNotIn("equity", spot)               # the number NEVER leaks before commit
        self.assertNotIn("modeled_equity", spot)

    def test_spot_token_deterministic_and_advance_sensitive(self):
        # the WHOLE token guard rests on determinism: two builds of the SAME un-advanced decision must
        # produce the SAME token (else every legitimate commit falsely rejects as 'spot moved'). codex L3
        # asked for this directly — the stale-token test proves bad tokens reject, not that a real one survives.
        ctx = _drive_to_live_decision(self.g, postflop=True)
        self.assertIsNotNone(ctx)
        tok1 = server._spot_token(self.g.coach_decision_context())
        tok2 = server._spot_token(self.g.coach_decision_context())
        self.assertEqual(tok1, tok2)                    # deterministic across calls on the frozen decision
        # and advancing the hand MUST change the identity, so a stale read is caught
        o = self.g.view().get("options") or {}
        self.g.act("check" if o.get("is_check") else "call" if o.get("check_or_call") else "fold")
        ctx2 = self.g.coach_decision_context()
        if ctx2 is not None:                            # advanced to a later hero decision → different token
            self.assertNotEqual(server._spot_token(ctx2), tok1)
        # (if the hand ended, coach_decision_context is None → the commit guard rejects on None anyway)

    def test_talk_through_with_stale_token_rejected(self):
        # codex L3 MED: 'talk it through' echoes the committed equity into the coach context so the prose
        # cites the same number. If the spot moved (another device advanced the hand), injecting the stale
        # equity would narrate the NEW spot with the OLD number — clean calibration, wrong coaching. A
        # mismatched token must reject, not splice.
        self.assertIsNotNone(_drive_to_live_decision(self.g, postflop=True))
        body = {"kind": "decision", "equity": {"hero_pct": 50, "band": "a coin flip"},
                "token": "deadbeefdeadbeef"}
        req = urllib.request.Request(f"http://127.0.0.1:{self.port}/api/coach/ask",
                                     data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"}, method="POST")
        try:
            urllib.request.urlopen(req)
            self.fail("expected a 409 spot-moved rejection")
        except urllib.error.HTTPError as e:
            self.assertEqual(e.code, 409)
            self.assertEqual(json.loads(e.read()).get("error"), "spot moved")


class TestModeledFromSpec(unittest.TestCase):
    """equity.modeled_from_spec — the SINGLE code path for the live number (server commit + worker prose
    both call it). A real range MC runs (small sample count for speed)."""

    def _spec(self, **over):
        s = {"hero_hole": ["As", "Kd"], "board": ["Ah", "7c", "2d"], "range": "22+, A2s+, KQo",
             "samples": 300, "vs": {"name": "Hank", "range_pct": 40}, "live_count": 1,
             "heads_up": True, "pot_way": 2}
        s.update(over)
        return s

    def test_blob_shape_and_passthrough(self):
        me = equity.modeled_from_spec(self._spec())
        self.assertIsInstance(me["hero_pct"], (int, float))
        self.assertEqual(me["band"], equity.band(me["hero_pct"]))   # band derived from the same %
        self.assertEqual(me["vs"], {"name": "Hank", "range_pct": 40})
        self.assertEqual((me["live_count"], me["heads_up"], me["pot_way"]), (1, True, 2))

    def test_none_and_malformed_specs_yield_none_not_crash(self):
        self.assertIsNone(equity.modeled_from_spec(None))
        self.assertIsNone(equity.modeled_from_spec({}))
        self.assertIsNone(equity.modeled_from_spec({"hero_hole": ["As"], "range": "22+"}))  # bad hole count


class TestChannelMeta(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "coach_test.db"

    def test_set_and_poll_meta_roundtrip(self):
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        self.assertEqual(coach.poll_meta(rid, self.tmp), {"status": "pending", "meta": None})
        ok = coach.set_meta(rid, {"modeled_equity": {"hero_pct": 63, "band": "a favorite"}}, self.tmp)
        self.assertTrue(ok)
        res = coach.poll_meta(rid, self.tmp)
        self.assertEqual(res["status"], "ready")
        self.assertEqual(res["meta"]["modeled_equity"]["hero_pct"], 63)

    def test_poll_meta_missing_id(self):
        self.assertEqual(coach.poll_meta(999999, self.tmp), {"status": "missing", "meta": None})

    def test_set_meta_unknown_id_false(self):
        self.assertFalse(coach.set_meta(999999, {"modeled_equity": {}}, self.tmp))

    def test_set_meta_is_write_once(self):
        # the FIRST computed number is frozen; a concurrent/restarted worker's wiggled value can't
        # overwrite it (codex/kimi L3 — keeps the drill reveal and the prose byte-identical).
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        self.assertTrue(coach.set_meta(rid, {"modeled_equity": {"hero_pct": 60}}, self.tmp))
        self.assertFalse(coach.set_meta(rid, {"modeled_equity": {"hero_pct": 64}}, self.tmp))
        self.assertEqual(coach.poll_meta(rid, self.tmp)["meta"]["modeled_equity"]["hero_pct"], 60)

    def test_get_row_includes_meta_column(self):
        # poll_meta depends on get() exposing `meta` as a dict key (complement L3 HIGH-1 — make explicit)
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        self.assertIn("meta", coach.get(rid, self.tmp))

    def test_poll_meta_answered_without_meta_is_terminal(self):
        # worker answered (prose) but produced no number → 'no_equity' (terminal), NOT 'pending' — so the
        # app stops waiting AND still surfaces the prose, instead of timing out + suppressing it (codex L3).
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        coach.respond(rid, "coached — no clean number on this spot", self.tmp)
        self.assertEqual(coach.poll_meta(rid, self.tmp), {"status": "no_equity", "meta": None})

    def test_poll_meta_unanswered_without_meta_is_pending(self):
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        self.assertEqual(coach.poll_meta(rid, self.tmp), {"status": "pending", "meta": None})

    def test_poll_meta_corrupt_blob_is_terminal(self):
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        con = sqlite3.connect(str(self.tmp))            # write a corrupt blob, bypassing set_meta's json.dumps
        try:
            con.execute("UPDATE requests SET meta=? WHERE id=?", ("{not json", rid)); con.commit()
        finally:
            con.close()
        self.assertEqual(coach.poll_meta(rid, self.tmp), {"status": "no_equity", "meta": None})

    def test_meta_does_not_disturb_prose_poll(self):
        rid = coach.ask("decision", {"x": 1}, self.tmp)
        coach.set_meta(rid, {"modeled_equity": {"hero_pct": 50}}, self.tmp)
        # the prose channel is independent of the meta blob — still pending until a worker responds
        self.assertEqual(coach.poll(rid, self.tmp)["status"], "pending")
        coach.respond(rid, "bet for value", self.tmp)
        self.assertEqual(coach.poll(rid, self.tmp)["response"], "bet for value")
        # and the meta survives the respond
        self.assertEqual(coach.poll_meta(rid, self.tmp)["meta"]["modeled_equity"]["hero_pct"], 50)

    def test_old_db_without_meta_column_is_migrated(self):
        # an older coach.db predates the meta column — create that exact shape, then prove the
        # channel migrates it on first connect (set/poll_meta work, the existing row is intact).
        old = Path(tempfile.mkdtemp()) / "old.db"
        raw = sqlite3.connect(str(old))
        raw.execute(
            """CREATE TABLE requests (
                   id INTEGER PRIMARY KEY AUTOINCREMENT, created REAL NOT NULL, kind TEXT NOT NULL,
                   context TEXT NOT NULL, status TEXT NOT NULL DEFAULT 'pending',
                   response TEXT, answered REAL)""")
        raw.execute("INSERT INTO requests (created, kind, context, status) VALUES (1.0,'decision','{}','pending')")
        raw.commit(); raw.close()
        probe = sqlite3.connect(str(old))
        try:
            cols = {r[1] for r in probe.execute("PRAGMA table_info(requests)")}
        finally:
            probe.close()
        self.assertNotIn("meta", cols)                      # precondition: genuinely old schema
        # first channel touch migrates; the pre-existing row (id 1) is still pollable + can take meta
        self.assertEqual(coach.poll_meta(1, old), {"status": "pending", "meta": None})
        self.assertTrue(coach.set_meta(1, {"modeled_equity": {"hero_pct": 71}}, old))
        self.assertEqual(coach.poll_meta(1, old)["meta"]["modeled_equity"]["hero_pct"], 71)


class TestWorkerPersistsEquity(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp()) / "coach_test.db"

    def test_format_spot_persists_modeled_equity(self):
        rid = coach.ask("decision", _modeled_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        prose = coach_watch.format_spot(req, self.tmp)        # worker lays out the spot → computes + persists
        res = coach.poll_meta(rid, self.tmp)
        self.assertEqual(res["status"], "ready")
        me = res["meta"]["modeled_equity"]
        self.assertIsInstance(me["hero_pct"], (int, float))
        self.assertEqual(me["band"], coach_watch.equity.band(me["hero_pct"]))
        self.assertEqual(me["vs"]["name"], "Hank")
        # THE invariant: the persisted number is the SAME one the prose cites (one computation, shared)
        self.assertIn(f"~{me['hero_pct']}%", prose)

    def test_stub_answer_persists_modeled_equity(self):
        rid = coach.ask("decision", _modeled_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        stub = coach_watch.stub_answer(req, self.tmp)
        me = coach.poll_meta(rid, self.tmp)["meta"]["modeled_equity"]
        self.assertIn(f"~{me['hero_pct']}%", stub)            # stub prose cites the same persisted number

    def test_daemon_answer_one_persists_on_caller_db(self):
        # the daemon path threads db end-to-end: the meta lands on self.tmp, NOT the default coach.DB
        rid = coach.ask("decision", _modeled_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        coach_daemon.answer_one(req, runner=lambda _p: "raise for value", db=self.tmp)
        self.assertEqual(coach.poll(rid, self.tmp)["response"], "raise for value")
        self.assertEqual(coach.poll_meta(rid, self.tmp)["status"], "ready")

    def test_worker_reuses_stored_equity_not_recompute(self):
        # canonicalization: a number already persisted for this request is REUSED, never recomputed — so a
        # restarted/second worker cites the SAME number the browser already revealed (codex/kimi L3 HIGH).
        rid = coach.ask("decision", _modeled_ctx(), self.tmp)
        coach.set_meta(rid, {"modeled_equity": {"hero_pct": 99, "band": "crushing",
                                                "vs": {"name": "Hank", "range_pct": 40}}}, self.tmp)
        req = coach.get(rid, self.tmp)
        coach_watch._ensure_modeled_equity(req["context"], rid, self.tmp)
        self.assertEqual(req["context"]["modeled_equity"]["hero_pct"], 99)   # reused stored, did NOT recompute

    def test_no_persist_without_req_id(self):
        # a bare call (no id — e.g. a test) computes into ctx but persists nothing
        ctx = _modeled_ctx()
        coach_watch._ensure_modeled_equity(ctx)
        self.assertIsNotNone(ctx.get("modeled_equity"))      # ctx mutated
        # nothing was written anywhere (no id) — and the default db is untouched by this call

    def test_no_spec_no_meta(self):
        rid = coach.ask("decision", _plain_ctx(), self.tmp)
        req = coach.get(rid, self.tmp)
        coach_watch.format_spot(req, self.tmp)
        self.assertEqual(coach.poll_meta(rid, self.tmp)["status"], "pending")   # nothing to model → no meta


class TestDecisionSpot(unittest.TestCase):
    def test_has_model_true_shape(self):
        spot = server._decision_spot(_modeled_ctx())
        self.assertTrue(spot["has_model"])
        self.assertEqual(spot["board"], ["Ah", "7c", "2d"])
        self.assertEqual(spot["hero_hole"], ["As", "Kd"])
        self.assertEqual(spot["pot"], 80)
        self.assertEqual(spot["to_call"], 25)
        self.assertEqual(spot["cards_to_come"], 2)           # flop (3 on board) → 2 to come
        self.assertEqual(spot["live_count"], 1)
        self.assertEqual(spot["vs"], "Hank")
        # the spot carries NO equity — the number must not reach the browser pre-commit
        self.assertNotIn("equity", spot)
        self.assertNotIn("modeled_equity", spot)

    def test_has_model_false_shape(self):
        spot = server._decision_spot(_plain_ctx())
        self.assertFalse(spot["has_model"])
        self.assertEqual(spot["hero_hole"], ["As", "Kd"])    # still renders the spot
        self.assertNotIn("live_count", spot)
        self.assertNotIn("vs", spot)

    def test_cards_to_come_by_street(self):
        for board, expect in ([], 5), (["Ah", "7c", "2d"], 2), (["Ah", "7c", "2d", "9s"], 1), \
                             (["Ah", "7c", "2d", "9s", "3h"], 0):
            ctx = {"board": board}
            self.assertEqual(server._decision_spot(ctx)["cards_to_come"], expect)


if __name__ == "__main__":
    unittest.main(verbosity=2)
