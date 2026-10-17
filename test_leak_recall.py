"""P4b-i wiring tests — leak recall-injection across the modules (the moat READ half).

The store-level scorer + seam wrapper live in test_leakstore.py. This file covers the WIRING:
  * server._spot_query — the spot → relevance-query builder, and its §3f safety (PLAYER-VISIBLE
    signals only; the hidden _truth archetype NEVER enters the query), tolerant of both the decision
    and the review ctx shapes; folded villains excluded.
  * server._vpip_behavior — observed-VPIP → behavior-language phrase (no 'type').
  * server._attach_relevant_leaks — recalls into ctx['relevant_leaks'], best-effort.
  * coach_watch._format_leaks / _stub_leaks_line — rendering the recalled leaks for the LLM coach
    and the facts-only floor, and the empty-case (no section when nothing recalled).
"""

import tempfile
import unittest
from pathlib import Path

from gutcheck import coach_watch
from gutcheck import leakstore
from gutcheck import server
from gutcheck.leakstore import LeakStore
from gutcheck.seam import Seam
import test_leakstore as TL   # reuse _game / _recall_doc fixtures

HAS_ANNEAL = leakstore.anneal_available()
_SKIP = "anneal-memory not installed (the [anneal] extra) — leak recall is optional"


# --------------------------------------------------------------------------- #
#  server._spot_query + _vpip_behavior — §3f-safe query construction
# --------------------------------------------------------------------------- #
class TestSpotQuery(unittest.TestCase):
    def test_vpip_behavior_is_looseness_only_never_passivity(self):
        # §3f / MED-1: VPIP → LOOSENESS only. Passive/calling/folding are NOT derivable from VPIP and
        # would re-encode the hidden archetype (only the station is high-VPIP in the current roster).
        self.assertEqual(server._vpip_behavior(None), "")
        self.assertEqual(server._vpip_behavior(8), "very tight villain")
        self.assertEqual(server._vpip_behavior(62), "loose wide villain")
        self.assertEqual(server._vpip_behavior(30), "balanced villain")
        # never names an archetype 'type', and never asserts a passive/aggressive read VPIP can't support
        for vp in (8, 30, 62):
            phrase = server._vpip_behavior(vp)
            for forbidden in ("station", "nit", "TAG", "rock", "reg", "passive", "calling", "folding",
                              "aggressive"):
                self.assertNotIn(forbidden, phrase)

    def test_decision_ctx_query_uses_only_observed_signals(self):
        ctx = {
            "street": "river",
            "hero": {"strength": "made hand: top pair", "position": "BTN"},
            "to_call": 40, "pot": 120,
            "board": ["Ah", "Kd", "7c", "2s", "9h"],
            "villains": [
                # a HIDDEN archetype on the live villain — must NOT leak into the query (§3f)
                {"name": "V1", "in_hand": True, "observed": {"vpip_pct": 62},
                 "_truth": {"type": "station", "label": "The Station"}},
                {"name": "V2", "in_hand": False, "observed": {"vpip_pct": 9}},  # folded → excluded
            ],
        }
        q = server._spot_query(ctx)
        self.assertIn("river", q)
        self.assertIn("made hand: top pair", q)
        self.assertIn("facing a bet", q)
        self.assertIn("loose wide villain", q)   # from the LIVE villain's observed VPIP (looseness only)
        # §3f: no hidden-archetype words, no passivity assertion, and the folded villain (tight) is absent
        for forbidden in ("station", "The Station", "rock", "reg", "passive", "calling", "very tight"):
            self.assertNotIn(forbidden, q)

    def test_checked_line_frames_differently(self):
        ctx = {"street": "turn", "to_call": 0, "board": ["Ah", "Kd", "7c", "2s"], "villains": []}
        self.assertIn("checked to", server._spot_query(ctx))

    def test_does_not_fabricate_street_from_final_board(self):
        # codex L3 MED: recall is DECISION-ONLY. _spot_query must NOT derive a street from final_board
        # (always 5 cards → always 'river'), which would mis-route review recall (river leaks on a
        # turn-reviewed mistake). A review-shaped ctx yields no street word.
        review_ish = {
            "hero": {"position": "CO", "net": -80},
            "final_board": ["Ah", "Kd", "7c", "2s", "9h"],   # 5 cards — must NOT become 'river'
            "result": {"hero_net": -80},
            "villains": [{"name": "V1", "observed": {"vpip_pct": 60}}],
        }
        q = server._spot_query(review_ish)
        for street in ("river", "turn", "flop", "preflop"):
            self.assertNotIn(street, q)        # no street fabricated from final_board
        self.assertNotIn("lost", q)            # LOW-2: outcome ≠ read-quality, not in the query
        self.assertNotIn("won", q)

    def test_empty_ctx_does_not_crash(self):
        self.assertIsInstance(server._spot_query({}), str)


class TestAttachGuardMED2(unittest.TestCase):
    """MED-2: _attach_relevant_leaks is the enrichment boundary — it must NEVER 500 the coach, even on
    a malformed ctx (the query builder runs before the seam's own error swallowing). Doesn't need anneal:
    the try/except catches the _spot_query raise AND a None/unavailable SEAM alike."""

    def test_malformed_ctx_attaches_empty_list_never_raises(self):
        prior = server.SEAM
        server.SEAM = None     # even with no seam wired, the boundary must degrade, not raise
        try:
            for bad in ({"villains": "notalist"}, {"villains": ["notadict"]},
                        {"villains": [{"observed": "notadict"}]},
                        {"villains": [{"observed": {"vpip_pct": "high"}, "in_hand": True}]}):
                try:
                    out = server._attach_relevant_leaks(dict(bad))
                except Exception as e:
                    self.fail(f"_attach_relevant_leaks raised on {bad!r}: {e!r}")
                self.assertEqual(out["relevant_leaks"], [])
        finally:
            server.SEAM = prior


# --------------------------------------------------------------------------- #
#  server._attach_relevant_leaks — recall into ctx, via the live SEAM
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestAttachRelevantLeaks(unittest.TestCase):
    def test_attaches_recalled_leaks_to_ctx(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = TL._game(tmp)
            ls._ensure().save_continuity(TL._recall_doc(
                ("sizing_is_a_read_output", 2,
                 "you bet sizing as a default, not a read-output; thin river value wants a small bet")))
            prior = server.SEAM
            server.SEAM = Seam(g)               # point the module global at our isolated seam
            try:
                ctx = {"street": "river", "hero": {"strength": "made hand: top pair"},
                       "to_call": 20, "board": ["Ah", "Kd", "7c", "2s", "9h"],
                       "villains": [{"name": "V1", "in_hand": True, "observed": {"vpip_pct": 62}}]}
                out = server._attach_relevant_leaks(ctx)
                self.assertIs(out, ctx)                                # returns ctx for chaining
                self.assertIn("relevant_leaks", ctx)
                self.assertIn("sizing_is_a_read_output", [l["name"] for l in ctx["relevant_leaks"]])
            finally:
                server.SEAM = prior
                ls.close()

    def test_attaches_empty_list_when_nothing_relevant(self):
        with tempfile.TemporaryDirectory() as tmp:
            g, ls = TL._game(tmp)
            ls._ensure().save_continuity(TL._recall_doc(
                ("barrel_into_the_caller", 1, "barreling a villain who only calls on the river")))
            prior = server.SEAM
            server.SEAM = Seam(g)
            try:
                ctx = {"street": "preflop", "hero": {"strength": "preflop AKs"},
                       "to_call": 0, "board": [], "villains": []}
                server._attach_relevant_leaks(ctx)
                self.assertEqual(ctx["relevant_leaks"], [])
            finally:
                server.SEAM = prior
                ls.close()


# --------------------------------------------------------------------------- #
#  coach_watch rendering — the LLM-coach section + the facts-only floor line
# --------------------------------------------------------------------------- #
class TestLeakRendering(unittest.TestCase):
    # overlap=3 → clears the tighter stub bar (a strong situational match)
    LEAKS = [{"name": "sizing_is_a_read_output", "level": 2, "overlap": 3,
              "summary": "thin river value wants a small bet, not a default size"}]

    def test_format_leaks_renders_doctrine_framed_section(self):
        out = coach_watch._format_leaks({"relevant_leaks": self.LEAKS})
        self.assertIn("Read-leaks YOU'VE shown", out)
        self.assertIn("sizing is a read output", out)          # de-snaked for the coach
        self.assertIn("(2x)", out)
        # HIGH-2: the prompt must demand the read→action IMPLICATION, leading with the FORBIDDEN action
        self.assertIn("read→action implication", out)
        self.assertIn("FORBIDS", out)

    def test_format_leaks_empty_when_none(self):
        self.assertEqual(coach_watch._format_leaks({"relevant_leaks": []}), "")
        self.assertEqual(coach_watch._format_leaks({}), "")

    def test_stub_leaks_line_is_fact_shaped(self):
        line = coach_watch._stub_leaks_line({"relevant_leaks": self.LEAKS})
        self.assertIn("Your read-leaks for spots like this", line)
        self.assertIn("sizing is a read output", line)
        self.assertEqual(coach_watch._stub_leaks_line({"relevant_leaks": []}), "")

    def test_stub_applies_a_tighter_bar_than_the_coach(self):
        # Phill's visual-gate call: the un-judged facts floor shows only STRONG matches; a borderline
        # 2-overlap leak reaches the LLM coach (which can weigh fit) but NOT the bare stub.
        weak = {"name": "weak_match", "level": 3, "overlap": 2, "summary": "river value default"}
        strong = {"name": "strong_match", "level": 1, "overlap": 3, "summary": "thin river value loose small"}
        ctx = {"relevant_leaks": [strong, weak]}
        coach_out = coach_watch._format_leaks(ctx)
        self.assertIn("strong match", coach_out)
        self.assertIn("weak match", coach_out)                 # coach sees BOTH (it can judge fit)
        stub_out = coach_watch._stub_leaks_line(ctx)
        self.assertIn("strong match", stub_out)
        self.assertNotIn("weak match", stub_out)               # stub shows ONLY the strong match
        # a leak with no overlap field (e.g. an external caller) fails the stub bar closed
        self.assertEqual(coach_watch._stub_leaks_line({"relevant_leaks": [{"name": "x", "summary": "y"}]}), "")

    def test_format_spot_includes_the_section_when_present(self):
        req = {"id": 1, "kind": "decision", "context": {
            "hand_no": 7, "street": "river",
            "hero": {"hole": ["As", "Ks"], "position": "BTN", "strength": "made hand: top pair", "stack": 200},
            "board": ["Ah", "Kd", "7c", "2s", "9h"], "pot": 120, "to_call": 0,
            "legal": {"can_fold": False, "can_check": True, "can_raise": True,
                      "min_raise_to": 10, "max_raise_to": 200},
            "action_this_hand": [], "villains": [],
            "relevant_leaks": self.LEAKS,
        }}
        spot = coach_watch.format_spot(req)
        self.assertIn("Read-leaks YOU'VE shown", spot)
        self.assertIn("sizing is a read output", spot)
        # and absent when the ctx carries none (the common case)
        req["context"]["relevant_leaks"] = []
        self.assertNotIn("Read-leaks YOU'VE shown", coach_watch.format_spot(req))


# --------------------------------------------------------------------------- #
#  L4 — integration-semantics: the WHOLE chain does what we claim
# --------------------------------------------------------------------------- #
@unittest.skipUnless(HAS_ANNEAL, _SKIP)
class TestEndToEndRecallChain(unittest.TestCase):
    """The claim: 'the relevant leak reaches the coach's context AT the live decision point.' Prove it
    through the real wired path — server-side recall → ctx attach → coach.ask (JSON round-trip via
    coach.db) → the exact build_prompt the daemon feeds claude — not just the unit-tested seam."""

    def _graduate_sizing_leak(self, g, ls):
        seam = Seam(g)
        g.record_live_read(gut=70, lean="value", truth_pct=22.0, street="river",
                           session_id="sA", hand_no=31, n=1, note="sized small with the nuts")
        wrap = seam.prepare_leak_wrap()
        # explanation shares river+value+loose with the river spot → overlap 3, clears the tighter stub
        # bar too (and grounds on the episode via river/value/small for graduation).
        doc = TL._leak_doc("sizing_is_a_read_output", 1, TL._newest_ids(ls, 1),
                           "thin river value betting versus a loose opponent wants a small read-based size")
        self.assertTrue(seam.commit_leaks(doc, wrap["wrap_token"])["ok"])
        return seam

    def _decision_ctx(self, *, street="river", strength="made hand: top pair", to_call=40, vpip=62):
        # mirrors coach_decision_context's shape, with a HIDDEN station archetype on the live villain
        board = {"river": ["Ah", "Kd", "7c", "2s", "9h"], "preflop": []}[street]
        return {"kind": "decision", "street": street, "hand_no": 31,
                "hero": {"hole": ["As", "Ks"], "position": "BTN", "strength": strength, "stack": 200},
                "board": board, "pot": 120, "to_call": to_call,
                "legal": {"can_fold": True, "can_check": to_call == 0, "can_raise": True,
                          "min_raise_to": 80, "max_raise_to": 200},
                "action_this_hand": [],
                "villains": [{"name": "V1", "position": "CO", "in_hand": True,
                              "observed": {"vpip_pct": vpip, "hands": 40},
                              "_truth": {"type": "station", "label": "The Station"}}]}

    def test_relevant_leak_reaches_the_coach_prompt_and_3f_holds(self):
        with tempfile.TemporaryDirectory() as tmp:
            from gutcheck import coach
            from gutcheck import coach_daemon
            coach_db = Path(tmp) / "coach.db"
            g, ls = TL._game(tmp)
            seam = self._graduate_sizing_leak(g, ls)
            prior = server.SEAM
            server.SEAM = seam
            try:
                ctx = self._decision_ctx()
                # the live serve-thread step:
                server._attach_relevant_leaks(ctx)
                self.assertIn("sizing_is_a_read_output", [l["name"] for l in ctx["relevant_leaks"]])
                # JSON round-trip through coach.db (what crosses to the worker thread):
                rid = coach.ask("decision", ctx, path=coach_db)
                req = coach.get(rid, path=coach_db)
                self.assertIn("sizing_is_a_read_output",
                              [l["name"] for l in req["context"].get("relevant_leaks", [])])
                # the EXACT prompt the daemon feeds claude:
                prompt = coach_daemon.build_prompt(req, db=coach_db)
                self.assertIn("Read-leaks YOU'VE shown", prompt)
                self.assertIn("sizing is a read output", prompt)
                self.assertIn("FORBIDS", prompt)
                # §3f (the precise invariant): the hidden archetype of THIS villain is NOT in the SPOT
                # body the coach reads. (The doctrine discusses 'station' as pedagogy — that's teaching,
                # not this hand's answer key; the spot rendering is where a leak would actually spoil.)
                spot = coach_watch.format_spot(req, db=coach_db)
                self.assertNotIn("station", spot.lower())
                self.assertNotIn("The Station", spot)
                # facts-floor stub (coaching OFF) also surfaces the operator's own leak as fact
                self.assertIn("Your read-leaks for spots like this",
                              coach_watch.stub_answer(req, db=coach_db, facts_mode=True))
            finally:
                server.SEAM = prior
                ls.close()

    def test_unrelated_spot_injects_no_leak_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            from gutcheck import coach
            from gutcheck import coach_daemon
            coach_db = Path(tmp) / "coach.db"
            g, ls = TL._game(tmp)
            seam = self._graduate_sizing_leak(g, ls)
            prior = server.SEAM
            server.SEAM = seam
            try:
                ctx = self._decision_ctx(street="preflop", strength="preflop AKs", to_call=0)
                server._attach_relevant_leaks(ctx)
                self.assertEqual(ctx["relevant_leaks"], [])          # river leak doesn't match a preflop spot
                req = coach.get(coach.ask("decision", ctx, path=coach_db), path=coach_db)
                self.assertNotIn("Read-leaks YOU'VE shown", coach_daemon.build_prompt(req, db=coach_db))
            finally:
                server.SEAM = prior
                ls.close()


if __name__ == "__main__":
    unittest.main()
