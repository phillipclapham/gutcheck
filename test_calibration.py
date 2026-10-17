"""Calibration-trend unit suite (stdlib unittest, zero deps).

The feature: persist every committed gut-vs-truth read into ~/.gutcheck/calibration.json and surface
the read-accuracy ARC over time, split by SOURCE — "exact" (review drill, vs enumerated equity = the
clean headline signal) vs "modeled" (live beat, vs the range-model's equity = flagged secondary).

The load-bearing invariant under test: the trend's TRUTH is canonical, never client-supplied —
record_review_read recomputes the exact equity from (session_id, hand_no, n) server-side, so a
hand-edited report or a hostile client cannot inject a fake number into the felt-sense oracle.
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from gutcheck import equity
from gutcheck import game as G


def _fresh_game(tmp):
    d = Path(tmp)
    return G.Game(d / "session.json", d / "career.json", d / "sessions",
                  d / "calibration.json", d / "history.db")


def _play_and_getup(seed, hands=45):
    """Play a deterministic session (hero checks/calls through), get up, and return
    (game, session_id, reviewable_pool) — the fixture for review/reconciliation tests."""
    random.seed(seed)
    tmp = tempfile.mkdtemp()
    g = _fresh_game(tmp)
    g.sit_down(hero="Phill")
    for _ in range(hands):
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
    g.get_up()
    report = json.loads(sorted((Path(tmp) / "sessions").glob("*.json"))[-1].read_text())
    pool = (report.get("biggest_pots") or []) + (report.get("notable_hands") or [])
    return g, report["id"], pool


def _play_more(g, hands=45):
    """Play ANOTHER full session on an existing game's store (continues the RNG, so the same
    hand_no holds DIFFERENT cards than the prior session); get up; return its session_id. The
    fixture for the cross-session reconciliation tests (hand_no collides across sessions)."""
    g.sit_down(hero="Phill")
    sid = g.s["session"]["id"]
    for _ in range(hands):
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
    g.get_up()
    return sid


class TestCalibrationMath(unittest.TestCase):
    """gap sign + abs_gap + band_hit are computed from gut/truth via the canonical equity.band()."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def test_same_band_is_a_hit(self):
        # 60 and 70 are both "a favorite" (58-75) → band hit; gut below truth → negative gap.
        s = self.g._record_calibration("exact", 60, 70.0, "call", "flop", "sid", 3, 1, 2)
        self.assertTrue(s["band_hit"])
        self.assertEqual(s["gut_band"], "a favorite")
        self.assertEqual(s["truth_band"], "a favorite")
        self.assertEqual(s["gap"], -10.0)
        self.assertEqual(s["abs_gap"], 10.0)

    def test_different_band_is_a_miss(self):
        # 30 = "behind" (18-42), 70 = "a favorite" → band miss.
        s = self.g._record_calibration("exact", 30, 70.0, "fold", "turn", "sid", 4, 1, 1)
        self.assertFalse(s["band_hit"])
        self.assertEqual(s["gut_band"], "behind")
        self.assertEqual(s["truth_band"], "a favorite")
        self.assertEqual(s["gap"], -40.0)

    def test_bands_match_equity_module(self):
        # the trend's bands MUST be the same equity.band() the drill reveal uses (no drift).
        s = self.g._record_calibration("exact", 12, 88.0, None, "river", "sid", 1, 1, 1)
        self.assertEqual(s["gut_band"], equity.band(12))
        self.assertEqual(s["truth_band"], equity.band(88.0))


class TestSourceSplitAndPersistence(unittest.TestCase):
    """exact and modeled are tracked as independent lifetime ledgers; the file round-trips."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def test_sources_are_independent(self):
        self.g._record_calibration("exact", 50, 52.0, None, "flop", "a", 1, 1, 1)
        self.g._record_calibration("exact", 50, 90.0, None, "flop", "a", 2, 1, 1)
        self.g._record_calibration("modeled", 50, 51.0, None, "turn", "b", None, None, 2)
        summ = self.g.calibration_summary()
        self.assertEqual(summ["exact"]["n"], 2)
        self.assertEqual(summ["modeled"]["n"], 1)
        # the modeled ledger must not contaminate the exact (headline) ledger.
        self.assertNotEqual(summ["exact"]["avg_abs_gap"], summ["modeled"]["avg_abs_gap"])

    def test_reload_preserves_trend(self):
        self.g._record_calibration("exact", 40, 45.0, "call", "flop", "a", 1, 1, 1)
        self.g._record_calibration("modeled", 60, 58.0, "bet", "turn", "b", None, None, 1)
        g2 = _fresh_game(self.tmp)            # a fresh process over the same data dir
        summ = g2.calibration_summary()
        self.assertEqual(summ["exact"]["n"], 1)
        self.assertEqual(summ["modeled"]["n"], 1)
        self.assertEqual(len(g2.calib["samples"]), 2)

    def test_corrupt_file_starts_fresh_not_crash(self):
        (Path(self.tmp) / "calibration.json").write_text("{ this is not json")
        g2 = _fresh_game(self.tmp)
        self.assertEqual(g2.calibration_summary()["exact"]["n"], 0)


class TestReadNote(unittest.TestCase):
    """P3c — the optional PROSE read-note: the read in the player's own words (often sharper than the
    digit). Stored, sanitized, exposed to the coach in recent[], and round-trips through reload."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def test_clean_note_sanitizes(self):
        self.assertEqual(G._clean_note("  weak kicker, drawing thin  "), "weak kicker, drawing thin")
        self.assertIsNone(G._clean_note("   "))           # blank → None
        self.assertIsNone(G._clean_note(""))              # empty → None
        self.assertIsNone(G._clean_note(None))            # non-string → None
        self.assertIsNone(G._clean_note(42))              # non-string → None (no truth-claim shape)
        self.assertEqual(len(G._clean_note("x" * 500)), G._NOTE_MAX)   # capped at _NOTE_MAX
        self.assertIsNone(G._clean_note("​​"))  # zero-width-only → None (not a "blank" note; L1)
        self.assertEqual(G._clean_note("a\x00\x07b"), "ab")           # control chars dropped, text kept
        self.assertEqual(G._clean_note("first\nsecond"), "first\nsecond")   # real newlines survive
        self.assertEqual(G._clean_note("weak kicker"), "weak kicker")  # NBSP → space, words kept (L3)
        self.assertIsNone(G._clean_note("́́"))             # combining-marks-only → blank → None (L3)
        self.assertEqual(G._clean_note("é"), "é")        # combining mark on real text → kept

    def test_note_stored_and_exposed_to_coach(self):
        self.g._record_calibration("exact", 45, 33.0, "check", "flop", "s", 8, 1, 2,
                                   note="  weak kicker, low-prob straight  ")
        recent = self.g.calibration_summary()["exact"]["recent"]
        self.assertEqual(recent[-1]["note"], "weak kicker, low-prob straight")   # trimmed, in coach view

    def test_note_is_optional(self):
        self.g._record_calibration("exact", 50, 52.0, None, "flop", "s", 1, 1, 1)   # no note kwarg
        self.assertIsNone(self.g.calibration_summary()["exact"]["recent"][-1]["note"])

    def test_live_read_persists_note(self):
        # the live beat (modeled source): the in-the-moment prose read rides into modeled recent[].note,
        # where the interactive coach (and P3a reconciliation) can read it.
        out = self.g.record_live_read(55, "call", 60.0, street="flop", vs=2, req_id="r1",
                                      note="  she snap-called, looks strong  ")
        self.assertTrue(out["ok"])
        self.assertEqual(self.g.calibration_summary()["modeled"]["recent"][-1]["note"],
                         "she snap-called, looks strong")   # trimmed, on the modeled (live) arc

    def test_live_read_note_optional(self):
        out = self.g.record_live_read(55, "call", 60.0, street="flop", vs=2, req_id="r2")   # no note
        self.assertTrue(out["ok"])
        self.assertIsNone(self.g.calibration_summary()["modeled"]["recent"][-1]["note"])

    def test_review_read_passes_note_through(self):
        # the actual product path: record_review_read accepts an optional note and forwards it.
        # (it recomputes truth from a real hand; here we assert the signature accepts note without error
        #  for a not-found hand — the note plumbing is exercised; full path covered by the API L4 check)
        res = self.g.record_review_read("nope", 1, 1, 50, "call", note="my read")
        self.assertFalse(res["ok"])   # hand not found, but the note kwarg was accepted (no TypeError)

    def test_note_round_trips_and_reload_recaps(self):
        self.g._record_calibration("exact", 40, 45.0, "call", "flop", "s", 1, 1, 1, note="my read")
        g2 = _fresh_game(self.tmp)
        self.assertEqual(g2.calibration_summary()["exact"]["recent"][-1]["note"], "my read")
        # a hand-edited file with an over-long note is RE-capped on load (defensive, like the join key)
        p = Path(self.tmp) / "calibration.json"
        data = json.loads(p.read_text())
        data["samples"][-1]["note"] = "z" * 9999
        p.write_text(json.dumps(data))
        g3 = _fresh_game(self.tmp)
        self.assertEqual(len(g3.calib["samples"][-1]["note"]), G._NOTE_MAX)


class TestCapAndLifetimeSurvival(unittest.TestCase):
    """Samples are capped, but the LIFETIME counters survive truncation so long-run stats stay true."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def test_lifetime_survives_sample_truncation(self):
        orig = G.CALIB_CAP
        G.CALIB_CAP = 5
        try:
            for i in range(12):
                # alternate hit/miss deterministically: even i → same band (hit), odd → miss
                if i % 2 == 0:
                    self.g._record_calibration("exact", 60, 70.0, None, "flop", "a", i, 1, 1)  # hit
                else:
                    self.g._record_calibration("exact", 30, 70.0, None, "flop", "a", i, 1, 1)  # miss
        finally:
            G.CALIB_CAP = orig
        self.assertLessEqual(len(self.g.calib["samples"]), 5)        # log truncated
        summ = self.g.calibration_summary()
        self.assertEqual(summ["exact"]["n"], 12)                     # lifetime count intact
        self.assertEqual(summ["exact"]["band_hit_rate"], 50)         # 6 hits / 12
        self.assertLessEqual(len(summ["exact"]["recent"]), G.CALIB_RECENT)


class TestDirection(unittest.TestCase):
    """The arrow rides the abs-gap trend (newer half vs older half), with a min-sample guard."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def _record_gap(self, gap_pts, k):
        # truth fixed at 50 ("a coin flip"); gut = 50 - gap → abs_gap == gap_pts.
        for _ in range(k):
            self.g._record_calibration("exact", 50 - gap_pts, 50.0, None, "flop", "a", 1, 1, 1)

    def test_too_few_samples_is_none(self):
        self._record_gap(10, 4)
        self.assertIsNone(self.g.calibration_summary()["exact"]["direction"])

    def test_sharpening(self):
        self._record_gap(30, 6)   # older: wide gaps
        self._record_gap(2, 6)    # newer: tight gaps
        self.assertEqual(self.g.calibration_summary()["exact"]["direction"], "sharpening")

    def test_drifting(self):
        self._record_gap(2, 6)
        self._record_gap(30, 6)
        self.assertEqual(self.g.calibration_summary()["exact"]["direction"], "drifting")

    def test_holding(self):
        self._record_gap(8, 12)
        self.assertEqual(self.g.calibration_summary()["exact"]["direction"], "holding")


class TestInputValidationAndLiveRead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def test_live_read_rejects_bad_gut(self):
        self.assertFalse(self.g.record_live_read(150, "call", 60.0)["ok"])   # gut > 100
        self.assertFalse(self.g.record_live_read("x", "call", 60.0)["ok"])   # gut not int
        self.assertFalse(self.g.record_live_read(50, "call", None)["ok"])    # no truth
        self.assertEqual(self.g.calibration_summary()["modeled"]["n"], 0)    # nothing logged

    def test_live_read_stamps_session_id(self):
        self.g.sit_down(hero="Phill")
        sid = self.g.s["session"]["id"]
        out = self.g.record_live_read(55, "call", 60.0, street="flop", vs=2)
        self.assertTrue(out["ok"])
        self.assertEqual(self.g.calib["samples"][-1]["session_id"], sid)
        self.assertEqual(self.g.calib["samples"][-1]["source"], "modeled")

    def test_bad_lean_is_dropped_not_stored(self):
        self.g.record_live_read(55, "shove", 60.0)   # "shove" is not a valid lean token
        self.assertIsNone(self.g.calib["samples"][-1]["lean"])

    def test_raise_is_a_valid_lean(self):
        # facing a bet, RAISE is a real action — it must PERSIST, not silently drop to None.
        # (the bug: _LEANS lacked "raise", so a raise lean was validated away even though the UI
        #  should offer it when facing — fold/call/RAISE.)
        out = self.g.record_live_read(60, "raise", 55.0)
        self.assertTrue(out["ok"])
        self.assertEqual(self.g.calib["samples"][-1]["lean"], "raise")


class TestReviewReadCanonicalTruth(unittest.TestCase):
    """The anti-fabrication core: record_review_read logs the CANONICAL exact equity recomputed
    server-side from (session_id, hand_no, n) — the client never supplies a truth value."""

    @classmethod
    def setUpClass(cls):
        random.seed(7)                                   # same deterministic report as test_equity
        cls.tmp = tempfile.mkdtemp()
        g = _fresh_game(cls.tmp)
        g.sit_down(hero="Phill")
        for _ in range(45):
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
        g.get_up()
        cls.g = g
        report = json.loads(sorted((Path(cls.tmp) / "sessions").glob("*.json"))[-1].read_text())
        cls.sid = report["id"]
        cls.pool = (report.get("biggest_pots") or []) + (report.get("notable_hands") or [])

    def _first_dp(self):
        for p in self.pool:
            out = self.g.coach_review_context(self.sid, p["hand_no"])
            if out and out[0].get("decision_points"):
                for dp in out[0]["decision_points"]:
                    if dp.get("equity") is not None:
                        return p["hand_no"], dp
        return None, None

    def test_logged_truth_equals_canonical_equity(self):
        hand_no, dp = self._first_dp()
        self.assertIsNotNone(dp, "need a reviewable hand with a postflop equity")
        out = self.g.record_review_read(self.sid, hand_no, dp["n"], gut=44, lean="call")
        self.assertTrue(out["ok"])
        # the trend's truth is the decision point's canonical equity — NOT anything a client passed.
        self.assertEqual(out["truth"], round(dp["equity"], 1))   # rounded for display
        s = self.g.calib["samples"][-1]
        self.assertEqual(s["source"], "exact")
        self.assertEqual(s["truth"], float(dp["equity"]))        # stored at full precision
        # the full join key is stamped on the sample so the coach never has to INFER which
        # hand/decision a read came from (anti-fabrication — P0).
        self.assertEqual(s["session_id"], self.sid)
        self.assertEqual(s["hand_no"], hand_no)
        self.assertEqual(s["n"], dp["n"])

    def test_review_read_persists_note_end_to_end(self):
        # the REAL product happy path: record_review_read → equity recompute → _record_calibration →
        # the prose note lands on the stored sample (and in the coach-facing recent[]). L1 coverage gap.
        hand_no, dp = self._first_dp()
        self.assertIsNotNone(dp, "need a reviewable hand with a postflop equity")
        out = self.g.record_review_read(self.sid, hand_no, dp["n"], gut=44, lean="call",
                                        note="  weak kicker, drawing thin  ")
        self.assertTrue(out["ok"])
        self.assertEqual(self.g.calib["samples"][-1]["note"], "weak kicker, drawing thin")  # trimmed, stored
        self.assertEqual(self.g.calibration_summary()["exact"]["recent"][-1]["note"],
                         "weak kicker, drawing thin")                                        # reaches the coach

    def test_decision_point_exposes_can_raise(self):
        # the review drill gates the RAISE lean on REAL legality, not the to_call>0 proxy (an
        # all-in/capped spot faces a bet but can't raise) — so each dp must carry can_raise (codex L3).
        _, dp = self._first_dp()
        self.assertIsNotNone(dp, "need a reviewable hand")
        self.assertIn("can_raise", dp)
        self.assertIsInstance(dp["can_raise"], bool)

    def test_unknown_hand_logs_nothing(self):
        before = self.g.calibration_summary()["exact"]["n"]
        out = self.g.record_review_read(self.sid, 99999, 1, gut=50, lean="call")
        self.assertFalse(out["ok"])
        self.assertEqual(self.g.calibration_summary()["exact"]["n"], before)

    def test_bad_gut_rejected(self):
        hand_no, dp = self._first_dp()
        before = self.g.calibration_summary()["exact"]["n"]
        self.assertFalse(self.g.record_review_read(self.sid, hand_no, dp["n"], 101, "call")["ok"])
        self.assertFalse(self.g.record_review_read(self.sid, hand_no, dp["n"], "x", "call")["ok"])
        self.assertEqual(self.g.calibration_summary()["exact"]["n"], before)

    def test_nonexistent_n_fails_closed_logs_nothing(self):
        # the join axis must fail closed: a valid hand but an n that matches NO decision point
        # records nothing, so a bad index can never log a read against the WRONG decision (L1 LOW).
        hand_no, dp = self._first_dp()
        before = self.g.calibration_summary()["exact"]["n"]
        out = self.g.record_review_read(self.sid, hand_no, 9999, gut=50, lean="call")
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "no such decision / no equity")
        self.assertEqual(self.g.calibration_summary()["exact"]["n"], before)

    def test_stored_join_key_resolves_to_same_decision(self):
        # kimi HIGH (converted to a guard): an explicit join key must not enable silent
        # MISattribution. Prove the STORED (session_id, hand_no, n) re-resolves — via the same
        # deterministic seed+log replay — to the decision whose equity was recorded, so a coach
        # joining on the key can't be pointed at the wrong spot.
        hand_no, dp = self._first_dp()
        out = self.g.record_review_read(self.sid, hand_no, dp["n"], gut=44, lean="call")
        self.assertTrue(out["ok"])
        s = self.g.calib["samples"][-1]
        ctx, _ = self.g.coach_review_context(s["session_id"], s["hand_no"])
        rdp = next(d for d in ctx["decision_points"] if d.get("n") == s["n"])
        self.assertEqual(rdp["equity"], s["truth"])   # the join key → the very decision recorded


class TestJoinKeyExposure(unittest.TestCase):
    """P0 — the read↔hand↔truth JOIN must be STORED and EXPOSED in calibration_summary().recent,
    so the coach reads which hand/decision a read came from instead of inferring it (inference-with-
    money = fabrication; flow's anti-fab discipline at the poker substrate)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)

    def test_recent_exposes_full_join_key_for_exact(self):
        self.g._record_calibration("exact", 60, 70.0, "call", "flop", "sid-7", 12, 3, 2)
        r = self.g.calibration_summary()["exact"]["recent"][-1]
        self.assertEqual(r["session_id"], "sid-7")
        self.assertEqual(r["hand_no"], 12)
        self.assertEqual(r["n"], 3)

    def test_recent_join_key_is_null_for_live_reads(self):
        # a live read has no reconstructed (hand_no, n) — they must surface as null, HONESTLY,
        # never a fabricated guess; session_id is still stamped from the active session.
        self.g.sit_down(hero="Phill")
        sid = self.g.s["session"]["id"]
        self.assertTrue(self.g.record_live_read(55, "call", 60.0, street="flop", vs=2)["ok"])
        r = self.g.calibration_summary()["modeled"]["recent"][-1]
        self.assertEqual(r["session_id"], sid)
        self.assertIsNone(r["hand_no"])
        self.assertIsNone(r["n"])

    def test_join_key_survives_reload(self):
        self.g._record_calibration("exact", 60, 70.0, "call", "flop", "sid-9", 4, 5, 1)
        g2 = _fresh_game(self.tmp)                       # _clean_sample whitelist must keep n
        r = g2.calibration_summary()["exact"]["recent"][-1]
        self.assertEqual((r["session_id"], r["hand_no"], r["n"]), ("sid-9", 4, 5))

    def test_old_schema_sample_without_n_loads_as_null(self):
        # backward-compat: a pre-P0 sample has no "n" key → loads + exposes n=None, never crashes.
        old = {"ts": "2026-06-19T10:00:00", "source": "exact", "gut": 50, "truth": 52.0,
               "gap": -2.0, "abs_gap": 2.0, "gut_band": "a coin flip", "truth_band": "behind",
               "band_hit": False, "lean": None, "street": "flop", "session_id": "old",
               "hand_no": 1, "vs": 1}                    # no "n"
        (Path(self.tmp) / "calibration.json").write_text(json.dumps(
            {"samples": [old], "lifetime": {"exact": {"n": 1, "band_hits": 0, "abs_gap_sum": 2}}}))
        g = _fresh_game(self.tmp)
        self.assertIsNone(g.calib["samples"][0]["n"])
        r = g.calibration_summary()["exact"]["recent"][-1]
        self.assertEqual(r["hand_no"], 1)
        self.assertIsNone(r["n"])

    def test_corrupt_join_key_normalized_to_null_on_load(self):
        # L3 codex: a hand-edited file must NOT publish a corrupt join key as authoritative —
        # a non-str session_id / non-int hand_no / non-int n each reads back as None ("unknown"),
        # so the coach can never join on a fabricated key.
        bad = {"ts": "2026-06-19T10:00:00", "source": "exact", "gut": 50, "truth": 52.0,
               "gap": -2.0, "abs_gap": 2.0, "gut_band": "a coin flip", "truth_band": "a coin flip",
               "band_hit": True, "lean": None, "street": "flop",
               "session_id": 123, "hand_no": "4", "n": "3", "vs": 1}   # all wrong-typed
        (Path(self.tmp) / "calibration.json").write_text(json.dumps(
            {"samples": [bad], "lifetime": {"exact": {"n": 1, "band_hits": 1, "abs_gap_sum": 2}}}))
        g = _fresh_game(self.tmp)
        s = g.calib["samples"][0]
        self.assertIsNone(s["session_id"])
        self.assertIsNone(s["hand_no"])
        self.assertIsNone(s["n"])


class TestLiveReadStoresJoinKey(unittest.TestCase):
    """P3a — record_live_read now stamps the live (hand_no, n) so the read can reconcile to its
    EXACT decision once the hand completes. Bad/absent values normalize to None (unjoinable),
    never a fabricated join."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.g = _fresh_game(self.tmp)
        self.g.sit_down(hero="Phill")

    def test_stores_live_join_key(self):
        sid = self.g.s["session"]["id"]
        self.assertTrue(self.g.record_live_read(55, "call", 60.0, street="flop", vs=2,
                                                hand_no=7, n=3, session_id=sid)["ok"])
        s = self.g.calib["samples"][-1]
        self.assertEqual((s["session_id"], s["hand_no"], s["n"]), (sid, 7, 3))

    def test_bad_join_key_normalized_to_none(self):
        sid = self.g.s["session"]["id"]
        self.assertTrue(self.g.record_live_read(55, "call", 60.0, hand_no="x", n=True,
                                                session_id=sid)["ok"])
        s = self.g.calib["samples"][-1]
        self.assertEqual(s["session_id"], sid)
        self.assertIsNone(s["hand_no"])   # bad types → None even within a frozen context
        self.assertIsNone(s["n"])

    def test_absent_join_key_is_none(self):
        self.assertTrue(self.g.record_live_read(55, "call", 60.0)["ok"])   # nothing frozen passed
        s = self.g.calib["samples"][-1]
        self.assertIsNone(s["hand_no"])
        self.assertIsNone(s["n"])

    def test_partial_frozen_context_drops_join_key(self):
        # codex re-review HIGH: the join key is ALL-OR-NOTHING. A partial row (hand_no/n present but
        # NO frozen session_id — e.g. a stale pre-P3a coach.db row) must NOT mix the current session id
        # with a frozen hand ordinal; it drops the join key entirely → unjoinable, never cross-joins.
        out = self.g.record_live_read(55, "call", 60.0, hand_no=9, n=2)   # hand_no/n but no session_id
        self.assertTrue(out["ok"])
        s = self.g.calib["samples"][-1]
        self.assertEqual(s["session_id"], self.g.s["session"]["id"])      # current session
        self.assertIsNone(s["hand_no"])   # join key dropped (no frozen session_id)
        self.assertIsNone(s["n"])


class TestLiveDecisionOrdinal(unittest.TestCase):
    """The P3a JOIN AXIS: a live read's n (counted in coach_decision_context) must equal the n
    coach_review_context assigns the SAME decision — else the modeled−exact join points at the
    wrong spot. Preflop live decisions carry n=None (review numbers only postflop spots)."""

    def _play_to_postflop_decision(self, seed):
        random.seed(seed)
        tmp = tempfile.mkdtemp()
        g = _fresh_game(tmp)
        g.sit_down(hero="Phill")
        sid = g.s["session"]["id"]
        captured = saw_preflop = None
        for _ in range(120):
            g.new_hand()
            hand_no = g.s["hand_no"]
            cand = None
            guard = 0
            while True:
                v = g.view()
                if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                    break
                ctx = g.coach_decision_context()
                if ctx is not None:
                    if ctx["n"] is None and saw_preflop is None:
                        saw_preflop = dict(ctx)
                    if ctx["n"] is not None and cand is None:
                        cand = (hand_no, ctx["n"], ctx["street"])
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
            # accept only if the hand settled into the canonical store (so review can reconstruct
            # it mid-session via the history.db fallback — proving the real reconciliation path)
            if cand and g.coach_review_context(sid, hand_no) is not None:
                captured = cand
                break
        return g, sid, captured, saw_preflop

    def test_live_n_matches_review_n(self):
        g, sid, captured, _ = self._play_to_postflop_decision(11)
        self.assertIsNotNone(captured, "need a recorded hand with a postflop hero decision")
        hand_no, n, street = captured
        built = g.coach_review_context(sid, hand_no)   # MID-SESSION → history.db fallback
        dp = next((d for d in built[0]["decision_points"] if d["n"] == n), None)
        self.assertIsNotNone(dp, f"review must number the same decision n={n} the live read captured")
        self.assertEqual(dp["street"], street)         # same n → same decision → same street

    def test_preflop_live_decision_has_null_n(self):
        _, _, _, saw_preflop = self._play_to_postflop_decision(11)
        self.assertIsNotNone(saw_preflop, "need a preflop hero decision")
        self.assertIsNone(saw_preflop["n"])

    def test_live_n_matches_review_n_for_EVERY_decision(self):
        # the FULL invariant (not just the first spot): every hero postflop decision's LIVE n must
        # equal review's n for the same decision — across MULTIPLE decisions in one hand. (L1's
        # 3000-seed empirical run had 0 mismatches; this locks one rich multi-decision hand in the suite.)
        random.seed(3)
        tmp = tempfile.mkdtemp()
        g = _fresh_game(tmp)
        g.sit_down(hero="Phill")
        sid = g.s["session"]["id"]
        target = None        # (hand_no, [(n, street), ...]) for a settled hand with >= 2 postflop decisions
        for _ in range(200):
            g.new_hand()
            hand_no = g.s["hand_no"]
            live = []
            guard = 0
            while True:
                v = g.view()
                if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                    break
                ctx = g.coach_decision_context()
                if ctx is not None and ctx["n"] is not None:
                    live.append((ctx["n"], ctx["street"]))
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
            if len(live) >= 2 and g.coach_review_context(sid, hand_no) is not None:
                target = (hand_no, live)
                break
        self.assertIsNotNone(target, "need a settled hand with >= 2 hero postflop decisions")
        hand_no, live = target
        built = g.coach_review_context(sid, hand_no)
        review = {d["n"]: d["street"] for d in built[0]["decision_points"]}
        for nn, street in live:                  # each captured live decision == review's same-n decision
            self.assertIn(nn, review, f"review has no decision n={nn}")
            self.assertEqual(review[nn], street, f"street drift at n={nn}: live {street} vs review {review[nn]}")


class TestModeledExactReconciliation(unittest.TestCase):
    """P3a — a live (modeled) read JOINS via (hand_no, n) to its EXACT decision, and the delta is
    surfaced. The exact equity is the REVIEW ORACLE's, recomputed server-side — never fabricated.
    Reads without a join key, for an unrecorded hand, or for another session are SKIPPED."""

    @classmethod
    def setUpClass(cls):
        cls.g, cls.sid, cls.pool = _play_and_getup(7)

    def _first_dp(self):
        for p in self.pool:
            built = self.g.coach_review_context(self.sid, p["hand_no"])
            if built and built[0].get("decision_points"):
                for dp in built[0]["decision_points"]:
                    if dp.get("equity") is not None:
                        return p["hand_no"], dp
        return None, None

    def test_reconciles_modeled_vs_exact(self):
        hand_no, dp = self._first_dp()
        self.assertIsNotNone(dp, "need a reviewable hand with a postflop exact equity")
        modeled = 71.0
        self.g._record_calibration("modeled", 64, modeled, "call", dp.get("street"),
                                   self.sid, hand_no, dp["n"], 2)
        rows = self.g.reconcile_live_reads(self.sid)
        # filter by the unique modeled value — the shared class fixture accumulates reads across tests
        match = [r for r in rows if r["hand_no"] == hand_no and r["n"] == dp["n"]
                 and r["modeled"] == round(modeled, 1)]
        self.assertEqual(len(match), 1)
        r = match[0]
        self.assertEqual(r["exact"], round(dp["equity"], 1))   # the review oracle's exact equity
        self.assertEqual(r["modeled"], round(modeled, 1))
        self.assertEqual(r["delta"], round(dp["equity"] - modeled, 1))
        self.assertEqual(r["exact_band"], equity.band(dp["equity"]))
        self.assertEqual(r["modeled_band"], equity.band(modeled))

    def test_skips_read_without_join_key(self):
        # a live read with no n (e.g. a preflop live read) is omitted — never inferred
        self.g._record_calibration("modeled", 50, 55.0, None, "flop", self.sid, None, None, 1)
        self.assertTrue(all(r["n"] is not None for r in self.g.reconcile_live_reads(self.sid)))

    def test_skips_unrecorded_hand(self):
        before = len(self.g.reconcile_live_reads(self.sid))
        self.g._record_calibration("modeled", 50, 55.0, None, "flop", self.sid, 999999, 1, 1)
        self.assertEqual(len(self.g.reconcile_live_reads(self.sid)), before)  # no fabricated delta

    def test_excludes_other_session(self):
        hand_no, dp = self._first_dp()
        self.g._record_calibration("modeled", 60, 70.0, None, dp.get("street"),
                                   self.sid, hand_no, dp["n"], 2)
        self.assertEqual(self.g.reconcile_live_reads("a-different-session"), [])


class TestCrossSessionStaleCommit(unittest.TestCase):
    """HIGH (L1/L3, reproduced): a delayed fire-and-forget live commit must NOT cross a session
    boundary. The join key is frozen ATOMICALLY in the decision context, so a read MADE in session A
    but committed while seated in session B is attributed to A (its true session) — never to B, where
    it would join to a DIFFERENT physical hand at the same (hand_no, n) and FABRICATE a delta."""

    def test_stale_commit_attributed_to_frozen_session_not_current(self):
        g, sid_a, pool = _play_and_getup(7)
        hand_no, dp = None, None
        for p in pool:
            built = g.coach_review_context(sid_a, p["hand_no"])
            if built and built[0].get("decision_points"):
                dp = next((d for d in built[0]["decision_points"] if d.get("equity") is not None), None)
                if dp:
                    hand_no = p["hand_no"]
                    break
        self.assertIsNotNone(dp, "need a reviewable session-A decision")
        # sit down a NEW session on the same store, then a STALE commit lands carrying A's frozen key
        g.sit_down(hero="Phill")
        sid_b = g.s["session"]["id"]
        self.assertNotEqual(sid_a, sid_b)
        out = g.record_live_read(60, "call", 55.0, street=dp.get("street"),
                                 hand_no=hand_no, n=dp["n"], session_id=sid_a)   # frozen A id
        self.assertTrue(out["ok"])
        self.assertEqual(g.calib["samples"][-1]["session_id"], sid_a)   # attributed to A, not current B
        # reconcile(B) must NOT see it → no fabricated delta against B's different hand at that (hand_no, n)
        self.assertEqual(g.reconcile_live_reads(sid_b), [])
        # reconcile(A) joins it correctly to A's real decision
        match = [r for r in g.reconcile_live_reads(sid_a)
                 if r["hand_no"] == hand_no and r["n"] == dp["n"]]
        self.assertEqual(len(match), 1)
        self.assertEqual(match[0]["exact"], round(dp["equity"], 1))


class TestAllSessionsReconciliation(unittest.TestCase):
    """The cross-session range-collapse trend (🎯 Reads, disentangle-3) reconciles EVERY joinable
    live read across ALL sessions via reconcile_live_reads(None). hand_no COLLIDES across sessions, so
    the replay cache MUST key on (session_id, hand_no): a hand_no-only cache would cross-join one
    session's read to another's exact equity and FABRICATE a delta. These guard the all-sessions path
    stays per-session-correct, and that the per-session scope is unchanged by the addition."""

    @staticmethod
    def _dps(g, sid, hand_no):
        built = g.coach_review_context(sid, hand_no)
        if built and built[0].get("decision_points"):
            return {d["n"]: d for d in built[0]["decision_points"] if d.get("equity") is not None}
        return {}

    def _colliding_decision(self, g, sid_a, sid_b):
        """Find a (hand_no, n) present in BOTH sessions with DISTINCT exact equity — the collision
        that a hand_no-only cache would mis-join."""
        for hand_no in range(1, 46):
            a, b = self._dps(g, sid_a, hand_no), self._dps(g, sid_b, hand_no)
            for n in a:
                if n in b and round(a[n]["equity"], 1) != round(b[n]["equity"], 1):
                    return hand_no, n, a[n], b[n]
        return None

    def test_all_sessions_keys_cache_on_session_not_hand(self):
        g, sid_a, _ = _play_and_getup(7)
        sid_b = _play_more(g)                     # 2nd session, SAME store → hand_no collides, cards differ
        self.assertNotEqual(sid_a, sid_b)
        coll = self._colliding_decision(g, sid_a, sid_b)
        self.assertIsNotNone(coll, "need a (hand_no,n) with distinct exact equity in both sessions")
        hand_no, n, dp_a, dp_b = coll
        # a live read in EACH session at the SAME (hand_no, n), with unique modeled values to find them
        g._record_calibration("modeled", 50, 33.0, "call", dp_a.get("street"), sid_a, hand_no, n, 1)
        g._record_calibration("modeled", 50, 34.0, "call", dp_b.get("street"), sid_b, hand_no, n, 1)
        rows = g.reconcile_live_reads(None)       # ALL sessions
        ra = [r for r in rows if r["session_id"] == sid_a and r["hand_no"] == hand_no and r["modeled"] == 33.0]
        rb = [r for r in rows if r["session_id"] == sid_b and r["hand_no"] == hand_no and r["modeled"] == 34.0]
        self.assertEqual(len(ra), 1)
        self.assertEqual(len(rb), 1)
        # each joined to ITS OWN session's exact — NOT cross-contaminated by the hand_no collision
        self.assertEqual(ra[0]["exact"], round(dp_a["equity"], 1))
        self.assertEqual(rb[0]["exact"], round(dp_b["equity"], 1))
        self.assertNotEqual(ra[0]["exact"], rb[0]["exact"])   # proves the cache key carries the session

    def test_all_sessions_spans_sessions_and_carries_session_id(self):
        g, sid_a, _ = _play_and_getup(11)
        sid_b = _play_more(g)
        da = self._dps(g, sid_a, next(h for h in range(1, 46) if self._dps(g, sid_a, h)))
        db = self._dps(g, sid_b, next(h for h in range(1, 46) if self._dps(g, sid_b, h)))
        ha = next(h for h in range(1, 46) if self._dps(g, sid_a, h))
        hb = next(h for h in range(1, 46) if self._dps(g, sid_b, h))
        g._record_calibration("modeled", 40, 41.0, "call", None, sid_a, ha, next(iter(da)), 1)
        g._record_calibration("modeled", 40, 42.0, "call", None, sid_b, hb, next(iter(db)), 1)
        sids = {r["session_id"] for r in g.reconcile_live_reads(None)}
        self.assertIn(sid_a, sids)
        self.assertIn(sid_b, sids)                # the trend spans BOTH sessions
        # per-session scope is unchanged: a session-id string sees only its own reads
        self.assertEqual({r["session_id"] for r in g.reconcile_live_reads(sid_a)}, {sid_a})

    def test_none_is_empty_with_no_joinable_reads(self):
        g, _sid, _ = _play_and_getup(3)           # played, but no live reads recorded
        self.assertEqual(g.reconcile_live_reads(None), [])

    def test_non_string_session_id_is_rejected(self):
        g, sid, _ = _play_and_getup(5)
        self.assertEqual(g.reconcile_live_reads(123), [])   # not None, not str → []


class TestHardening(unittest.TestCase):
    """Regression guards for the apparatus fix cluster (L1/L2 findings)."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def _write_calib(self, obj):
        p = Path(self.tmp) / "calibration.json"
        p.write_text(json.dumps(obj) if not isinstance(obj, str) else obj)

    def test_wrong_shaped_root_does_not_block_construction(self):
        # MED-3: a valid-JSON-but-wrong-shape file must not crash Game.__init__ (would block the app)
        for bad in ("[]", '"x"', "5", "null"):
            self._write_calib(bad)
            g = _fresh_game(self.tmp)                         # must not raise
            self.assertEqual(g.calibration_summary()["exact"]["n"], 0)

    def test_malformed_sample_dropped_on_load_not_crashing_summary(self):
        # MED-2: a partial/old-schema sample must be dropped on load, never crash the summary GET
        good = {"ts": "2026-06-19T10:00:00", "source": "exact", "gut": 50, "truth": 52.0,
                "gap": -2.0, "abs_gap": 2.0, "gut_band": "a coin flip", "truth_band": "behind",
                "band_hit": False, "lean": None, "street": "flop", "session_id": "x",
                "hand_no": 1, "vs": 1}
        bad = {"source": "exact", "gut": 50}                  # missing truth/abs_gap/band_hit/ts
        self._write_calib({"samples": [good, bad],
                           "lifetime": {"exact": {"n": 2, "band_hits": 0, "abs_gap_sum": 4}}})
        g = _fresh_game(self.tmp)
        self.assertEqual(len(g.calib["samples"]), 1)          # bad row dropped
        summ = g.calibration_summary()                        # must not raise
        self.assertEqual(summ["exact"]["recent_n"], 1)

    def test_band_hits_clamped_to_n_on_load(self):
        # LOW-3: a hand-edited file with band_hits > n must not render a >100% rate
        self._write_calib({"samples": [],
                           "lifetime": {"exact": {"n": 2, "band_hits": 5, "abs_gap_sum": 0}}})
        g = _fresh_game(self.tmp)
        self.assertLessEqual(g.calibration_summary()["exact"]["band_hit_rate"], 100)

    def test_nan_inf_truth_rejected_and_file_stays_valid_json(self):
        # MED-5: a corrupt worker number must never append nor poison the durable JSON
        g = _fresh_game(self.tmp)
        for bad in (float("nan"), float("inf"), -5.0, 150.0):
            self.assertFalse(g.record_live_read(50, "call", bad)["ok"])
        self.assertEqual(len(g.calib["samples"]), 0)
        # a valid read after the rejects writes the file — it must be strictly-parseable JSON
        g.record_live_read(50, "call", 60.0)
        raw = (Path(self.tmp) / "calibration.json").read_text()
        self.assertNotIn("NaN", raw)
        self.assertNotIn("Infinity", raw)
        json.loads(raw)                                       # strict parse must succeed

    def test_bool_gut_rejected(self):
        # LOW-1: bool ⊆ int — true/false must not slip through as gut=1/0
        g = _fresh_game(self.tmp)
        self.assertFalse(g.record_live_read(True, "call", 60.0)["ok"])
        self.assertEqual(len(g.calib["samples"]), 0)

    def test_live_double_log_deduped_by_req_id(self):
        # MED-1: the same live coach id names one spot → logs once (a retry/double-click is a no-op)
        g = _fresh_game(self.tmp)
        self.assertTrue(g.record_live_read(55, "call", 60.0, req_id=7)["ok"])
        out = g.record_live_read(55, "call", 60.0, req_id=7)
        self.assertFalse(out["ok"])
        self.assertEqual(out["reason"], "already logged")
        self.assertEqual(g.calibration_summary()["modeled"]["n"], 1)

    def test_avg_abs_gap_keeps_sub_point_precision(self):
        # MED-4: the integer accumulator floored sub-point error → "0.0 off" on a systematic ~0.4 miss
        g = _fresh_game(self.tmp)
        g._record_calibration("exact", 50, 49.6, None, "flop", "x", 1, 1, 1)   # 0.4 off
        g._record_calibration("exact", 50, 50.4, None, "flop", "x", 2, 1, 1)   # 0.4 off
        self.assertEqual(g.calibration_summary()["exact"]["avg_abs_gap"], 0.4)

    def test_summary_exposes_recent_window_denominator(self):
        # the arrow + recent% must describe the SAME window the sparkline shows (not the all-time n)
        g = _fresh_game(self.tmp)
        for _ in range(12):
            g._record_calibration("exact", 60, 70.0, None, "flop", "x", 1, 1, 1)
        summ = g.calibration_summary()["exact"]
        self.assertEqual(summ["recent_n"], 12)
        self.assertEqual(summ["recent_band_hit_rate"], 100)   # all hits (60 & 70 both "a favorite")
        self.assertEqual(summ["n"], 12)

    def test_inconsistent_derived_fields_recomputed_on_load(self):
        # L3 codex+kimi MED: a hand-edited row with a lying abs_gap/band_hit must be recomputed on
        # load from gut+truth, never trusted into the summary.
        lying = {"ts": "2026-06-19T10:00:00", "source": "exact", "gut": 50, "truth": 80.0,
                 "gap": 0.0, "abs_gap": 0.1, "gut_band": "crushing", "truth_band": "crushing",
                 "band_hit": True, "lean": None, "street": "flop", "session_id": "x",
                 "hand_no": 1, "vs": 1}
        self._write_calib({"samples": [lying], "lifetime": {"exact": {"n": 0, "band_hits": 0,
                                                                      "abs_gap_sum": 0}}})
        g = _fresh_game(self.tmp)
        s = g.calib["samples"][0]
        self.assertEqual(s["abs_gap"], 30.0)            # recomputed |50-80|, not the lying 0.1
        self.assertFalse(s["band_hit"])                 # recomputed: coin flip vs clear favorite

    def test_non_finite_lifetime_counter_rejected_on_load(self):
        # L3 codex: Inf/NaN in a lifetime counter must not survive (would re-emit as invalid JSON)
        self._write_calib('{"samples": [], "lifetime": {"exact": '
                          '{"n": 1, "band_hits": 0, "abs_gap_sum": Infinity}}}')
        g = _fresh_game(self.tmp)                        # must not raise
        summ = g.calibration_summary()["exact"]
        self.assertEqual(summ["avg_abs_gap"], 0.0)       # Infinity rejected → counter stays 0

    def test_atomic_write_refuses_non_finite(self):
        # L3 codex backstop: no writer can ever persist NaN/Infinity — it fails LOUD instead
        g = _fresh_game(self.tmp)
        g.calib["lifetime"]["exact"]["abs_gap_sum"] = float("inf")
        with self.assertRaises(ValueError):
            g._save_calib()

    def test_truth_band_uses_full_precision_matching_the_drill(self):
        # L3 kimi: banding must use the UNROUNDED truth (matching the drill reveal) — 41.97 is
        # "behind" (<42), but rounding to 42.0 would flip it to "a coin flip" and fake a band-miss.
        g = _fresh_game(self.tmp)
        s = g._record_calibration("exact", 30, 41.97, None, "flop", "x", 1, 1, 1)
        self.assertTrue(s["band_hit"])                   # 30 behind, 41.97 behind → hit
        g2 = _fresh_game(self.tmp)                        # reload: recompute from stored 41.97
        self.assertTrue(g2.calib["samples"][0]["band_hit"])   # load == write


if __name__ == "__main__":
    unittest.main()
