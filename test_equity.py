"""Slice-3 equity-oracle + review-drill unit suite (stdlib unittest, zero deps).

Covers the equity oracle (equity.py) — the EXACT enumeration path (review/drill) and the Monte-Carlo
path (future decision/range coaching) plus every uncomputable guard — and the review decision-point
capture in game.coach_review_context (the prediction-error drill data). The oracle's exact path is
fully deterministic, so its assertions are hard equalities; the Monte-Carlo path is bounded-asserted.

    ./venv/bin/python test_equity.py
"""

import json
import random
import tempfile
import unittest
from pathlib import Path

from gutcheck import equity
from gutcheck import game as G


class TestExactEquity(unittest.TestCase):
    """The review/drill path — enumerated vs KNOWN cards, so deterministic and truly exact."""

    def test_river_is_exact_and_deterministic(self):
        # AK with a pair of kings vs J-high on a complete board — hero wins every time.
        r = equity.hero_vs_known(["Ah", "Kd"], [["Jh", "Th"]], ["Qh", "7h", "2c", "Kc", "3d"])
        self.assertEqual(r["hero_pct"], 100.0)
        self.assertEqual(r["villains"], [0.0])
        self.assertTrue(r["exact"])
        self.assertEqual(r["cards_to_come"], 0)
        # determinism: identical inputs → byte-identical result (no Monte-Carlo wiggle)
        self.assertEqual(r, equity.hero_vs_known(["Ah", "Kd"], [["Jh", "Th"]],
                                                 ["Qh", "7h", "2c", "Kc", "3d"]))

    def test_set_crushes_overpair_on_flop(self):
        r = equity.hero_vs_known(["7c", "7d"], [["Ah", "As"]], ["7s", "2c", "9h"])
        self.assertTrue(r["exact"])
        self.assertEqual(r["cards_to_come"], 2)
        self.assertGreater(r["hero_pct"], 85.0)            # a set is a huge favorite over an overpair
        self.assertAlmostEqual(r["hero_pct"] + r["villains"][0], 100.0, places=1)  # equities sum to 100

    def test_multiway_known(self):
        r = equity.hero_vs_known(["As", "Qd"], [["Ah", "Kh"], ["7c", "7s"]], ["Qs", "7d", "2c"])
        self.assertTrue(r["exact"])
        self.assertEqual(len(r["villains"]), 2)
        self.assertAlmostEqual(r["hero_pct"] + sum(r["villains"]), 100.0, places=1)

    def test_preflop_known_falls_back_to_sampling(self):
        # 5 cards to come → enumeration would explode → Monte-Carlo path → not flagged exact
        r = equity.hero_vs_known(["Ah", "As"], [["Kh", "Ks"]], [])
        self.assertFalse(r["exact"])
        self.assertGreater(r["hero_pct"], 75.0)            # AA still crushes KK preflop

    def test_exact_rejects_colliding_cards(self):
        # the exact oracle must FAIL CLOSED on an impossible deck, never fabricate an "exact" number
        self.assertIsNone(equity.hero_vs_known(["Ah", "Kd"], [["Ah", "Qs"]], ["7c", "8d", "9h"]))  # shares Ah
        self.assertIsNone(equity.hero_vs_known(["Ah", "Kd"], [["7c", "Qs"]], ["7c", "8d", "9h"]))   # villain holds board 7c
        self.assertIsNone(equity.hero_vs_known(["Ah", "Ah"], [["Kh", "Ks"]], ["7c", "8d", "9h"]))   # dup within hero hand

    def test_known_collision_guard_covers_preflop_fallback(self):
        # the >2-cards-to-come fallback to Monte-Carlo must ALSO fail closed on a duplicate card
        # (codex L3: the guard lived only in the exact path, so preflop slipped a bad spot through)
        self.assertIsNone(equity.hero_vs_known(["Ah", "Ah"], [["Kh", "Ks"]], []))            # preflop dup hero card
        self.assertIsNone(equity.hero_vs_known(["Ah", "Kd"], [["Ah", "Qs"]], ["7c"]))        # 4 to come, shared Ah

    def test_river_chop_splits_50_50(self):
        # a royal on the board: both players "play the board", so equity chops exactly 50/50
        r = equity.hero_vs_known(["2c", "3d"], [["4h", "5s"]], ["Ah", "Kh", "Qh", "Jh", "Th"])
        self.assertEqual(r["hero_pct"], 50.0)
        self.assertEqual(r["villains"], [50.0])
        self.assertTrue(r["exact"])


class TestMonteCarloEquity(unittest.TestCase):
    """The range/decision path — sampled, so bounded assertions."""

    def test_aa_vs_kk_preflop(self):
        r = equity.hero_vs(["Ah", "As"], ["KhKs"], [])
        self.assertGreater(r["hero_pct"], 78.0)
        self.assertLess(r["hero_pct"], 87.0)
        self.assertFalse(r["exact"])

    def test_vs_range_string(self):
        r = equity.hero_vs(["As", "Qd"], ["22+, A2s+, KTo+"], ["Qs", "7d", "2c"])
        self.assertIsNotNone(r)
        self.assertGreater(r["hero_pct"], 50.0)            # top pair top kicker ahead of that range

    def test_river_via_montecarlo_is_flagged_exact(self):
        # hero_vs on a complete board: nothing to sample → exact regardless of path
        r = equity.hero_vs(["Ah", "Kd"], ["JhTh"], ["Qh", "7h", "2c", "Kc", "3d"])
        self.assertTrue(r["exact"])
        self.assertEqual(r["hero_pct"], 100.0)


class TestBands(unittest.TestCase):
    """The felt-sense translation: equity % → a label the gut reads. Locks the thresholds (the JS
    mirror eqBand in static/index.html must match these)."""

    def test_band_mapping(self):
        cases = [(0, "drawing dead"), (1.9, "drawing dead"), (2, "drawing thin"), (17, "drawing thin"),
                 (18, "behind"), (41, "behind"), (42, "a coin flip"), (57, "a coin flip"),
                 (58, "a favorite"), (74, "a favorite"), (75, "a clear favorite"), (89, "a clear favorite"),
                 (90, "crushing"), (100, "crushing")]
        for pct, label in cases:
            self.assertEqual(equity.band(pct), label, f"{pct}% → {equity.band(pct)!r} != {label!r}")
        self.assertIsNone(equity.band(None))

    def test_js_eqband_thresholds_match_python(self):
        # the JS mirror eqBand in static/index.html MUST match equity._EQUITY_BANDS, or the live slider
        # readout silently disagrees with the reveal/coach. Parse the JS table + compare (L1 NOTE-3).
        import re
        from gutcheck import server
        # resolve static/ THROUGH the package (server.STATIC = <pkg>/static), not relative to this
        # test file — static/ now lives inside the gutcheck/ package, not at the repo root.
        html = (server.STATIC / "index.html").read_text()
        m = re.search(r"function eqBand\(pct\).*?of \[(.*?)\]\)", html, re.S)
        self.assertIsNotNone(m, "eqBand threshold table not found in index.html")
        js = [(int(hi), label) for hi, label in re.findall(r'\[(\d+),"([^"]+)"\]', m.group(1))]
        self.assertEqual(js, equity._EQUITY_BANDS, "JS eqBand thresholds drifted from Python _EQUITY_BANDS")


class TestGuards(unittest.TestCase):
    def test_fewer_than_two_entrants(self):
        self.assertIsNone(equity.equities(["AhKd"], []))

    def test_villain_holds_a_board_card(self):
        # a "range" that conflicts with the board is fully dead → refused, not crashed
        self.assertIsNone(equity.hero_vs(["Ah", "Kd"], ["Qs7d"], ["Qs", "7d", "2c"]))

    def test_known_empty_villains(self):
        self.assertIsNone(equity.hero_vs_known(["Ah", "Kd"], [], ["Qs", "7d", "2c"]))

    def test_malformed_hole(self):
        self.assertIsNone(equity.hero_vs(["Ah"], ["KhKs"], []))            # 1-card hero
        self.assertIsNone(equity.hero_vs_known(["Ah", "Kd"], [["Jh"]], ["Qh", "7h", "2c"]))

    def test_equities_never_raises_on_malformed_strings(self):
        # the contract is "returns None, never raises" — a bad rank/range must not throw (codex L3:
        # parse failures happened before the try). Returns None instead of a ValueError.
        self.assertIsNone(equity.equities(["ZZ", "KhKs"], []))             # invalid rank
        self.assertIsNone(equity.equities(["KhKs", "not-a-range"], ["Qh", "7h", "2c"]))


class TestReviewDecisionPoints(unittest.TestCase):
    """The drill data: game.coach_review_context must capture each hero POSTFLOP decision with an
    exact equity, a frame to pause at, and the action actually taken."""

    @classmethod
    def setUpClass(cls):
        random.seed(7)                                   # deterministic hands for a stable report
        cls.d = Path(tempfile.mkdtemp())
        g = G.Game(cls.d / "session.json", cls.d / "career.json", cls.d / "sessions",
                   history_db=cls.d / "history.db")
        g.sit_down(hero="Phill")
        for _ in range(45):                              # call down to manufacture postflop decisions
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
        report = json.loads(sorted((cls.d / "sessions").glob("*.json"))[-1].read_text())
        cls.sid = report["id"]
        cls.pool = (report.get("biggest_pots") or []) + (report.get("notable_hands") or [])

    def _first_with_dps(self):
        for p in self.pool:
            out = self.g.coach_review_context(self.sid, p["hand_no"])
            if out and out[0].get("decision_points"):
                return out
        return None

    def test_some_hand_yields_decision_points(self):
        self.assertIsNotNone(self._first_with_dps(), "expected at least one reviewable hand with a "
                                                     "hero postflop decision across 45 call-down hands")

    def test_decision_point_shape_and_invariants(self):
        out = self._first_with_dps()
        ctx, frames = out
        dps = ctx["decision_points"]
        for i, dp in enumerate(dps):
            self.assertEqual(dp["n"], i + 1)                          # 1-based, in order
            self.assertIn(dp["street"], ("flop", "turn", "river"))    # POSTFLOP only
            self.assertGreaterEqual(dp["frame_index"], 0)
            self.assertLess(dp["frame_index"], len(frames))           # a real frame to pause before
            self.assertGreaterEqual(dp["vs"], 1)                      # equity is vs >=1 live villain
            if dp["equity"] is not None:
                self.assertGreaterEqual(dp["equity"], 0.0)
                self.assertLessEqual(dp["equity"], 100.0)
                self.assertTrue(dp["exact"])                          # known cards postflop = exact
            self.assertIn(dp["cards_to_come"], (0, 1, 2, None))
            self.assertTrue(dp["action"])                             # what the hero actually did


class TestLiveModeledEquity(unittest.TestCase):
    """game.coach_decision_context bakes a modeled-equity SPEC (the MC is deferred to the worker, off the
    single-threaded ask path); the worker (coach_watch._ensure_modeled_equity) turns it into the equity."""

    def _first_spec(self):
        import random
        import tempfile
        from pathlib import Path
        random.seed(5)
        d = Path(tempfile.mkdtemp())
        g = G.Game(d / "s.json", d / "c.json", d / "sessions",
                   history_db=d / "history.db")
        g.sit_down(hero="Phill")
        for _ in range(20):
            g.new_hand()
            guard = 0
            while True:
                v = g.view()
                if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                    break
                if v.get("board"):                       # postflop hero decision
                    ctx = g.coach_decision_context()
                    if ctx and ctx.get("modeled_spec"):
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

    def test_spec_shape_and_no_equity_on_ask_path(self):
        ctx = self._first_spec()
        self.assertIsNotNone(ctx, "expected a postflop hero decision with a modeled spec")
        # the ask path bakes ONLY the spec — no Monte-Carlo here (it would stall the single-threaded server)
        self.assertIsNone(ctx.get("modeled_equity"))
        spec = ctx["modeled_spec"]
        for k in ("hero_hole", "board", "range", "vs", "live_count", "heads_up", "pot_way"):
            self.assertIn(k, spec)
        self.assertEqual(spec["pot_way"], spec["live_count"] + 1)        # +1 for the hero (M1 fix)
        self.assertEqual(spec["heads_up"], spec["live_count"] == 1)
        self.assertIn("hands", spec["vs"])                              # thin-sample signal forwarded (L2 MED)

    def test_select_target_last_aggressor_by_action_order(self):
        from gutcheck.game import _select_target
        # BTN (seat 5) opens, SB (seat 0) 3-bets → the target is the SB (LAST aggressor by ACTION),
        # not the higher-seat BTN. cand is in seat order; the old aggressors[-1] bug picked BTN (codex L3).
        cand = [{"seat": 0, "name": "SB", "range_pct": 18}, {"seat": 5, "name": "BTN", "range_pct": 25}]
        hist = [{"seat": 5, "token": "r6"}, {"seat": 0, "token": "r18"}]
        self.assertEqual(_select_target(cand, hist, [0, 5])["name"], "SB")
        # no live aggressor → the tightest range (clearest threat)
        cand2 = [{"seat": 0, "name": "A", "range_pct": 40}, {"seat": 2, "name": "B", "range_pct": 22}]
        self.assertEqual(_select_target(cand2, [{"seat": 0, "token": "c"}], [0, 2])["name"], "B")
        # a raiser who has since folded (not in live) must NOT be picked
        cand3 = [{"seat": 0, "name": "live", "range_pct": 30}]
        self.assertEqual(_select_target(cand3, [{"seat": 9, "token": "r10"}], [0])["name"], "live")

    def test_worker_computes_equity_from_spec(self):
        from gutcheck import coach_watch
        ctx = self._first_spec()
        self.assertIsNotNone(ctx)
        coach_watch._ensure_modeled_equity(ctx)                         # the worker-side MC
        me = ctx["modeled_equity"]
        self.assertGreaterEqual(me["hero_pct"], 0.0)
        self.assertLessEqual(me["hero_pct"], 100.0)
        self.assertEqual(me["pot_way"], ctx["modeled_spec"]["pot_way"])
        # the rendered line must carry the CORRECT pot-way count (M1) and not contradict itself (M2)
        line = coach_watch._format_modeled_equity(me)
        if not me["heads_up"]:
            self.assertIn(f"{me['pot_way']}-way", line)
            self.assertNotIn("heads-up", line)                          # M2: no "heads-up" in a multiway line


if __name__ == "__main__":
    unittest.main(verbosity=2)
