"""Tests for the sizing-tell (P3(c)): per-villain OBSERVED betting → shown-strength, showdown-only.

Driven through a REAL Game (the hero folds to every bet so the villains battle on to showdown),
then the tell is asserted on RNG-INDEPENDENT invariants — output shape, internal sum-consistency,
the postflop-only and showdown-only filters as invariants, session scoping, and live-session name
resolution. A fixed seed (7) reliably produces villain postflop aggression + showdowns; no assertion
depends on the specific cards dealt, only on properties the method must hold for ANY data.
"""

import random
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from gutcheck import game as G

# the postflop_strength vocabulary — a leaked PREFLOP raise would carry preflop_bucket labels
# ('premium'/'playable'/'trash') instead, so this set is also the postflop-only guard.
_STRENGTHS = {"air", "weak", "medium", "strong", "monster"}


def _play(seed, hands=30):
    """Play a real session where the hero folds to every bet — leaving the villains to play their
    hands out to showdown, which is exactly the data the showdown-observed tell feeds on."""
    random.seed(seed)
    d = Path(tempfile.mkdtemp())
    g = G.Game(d / "s.json", d / "c.json", d / "sess", d / "cal.json", d / "h.db")
    # sit_down draws the master seed from SystemRandom, so random.seed alone never made the deals
    # repeat (this test flaked on CI); pin that draw so the same session plays every run.
    with unittest.mock.patch.object(G.random, "SystemRandom", lambda: random.Random(seed)):
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
            elif o.get("fold"):
                g.act("fold")          # fold to bets → the villains play it out to showdown
            else:
                break
            guard += 1
            if guard > 40:
                break
    return g


class TestSizingTells(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = _play(7)
        cls.sid = cls.g.s["session"]["id"]
        cls.out = cls.g.sizing_tells(cls.sid)

    def test_surfaces_only_showdown_villains(self):
        self.assertEqual(self.out["session_id"], self.sid)
        self.assertTrue(self.out["villains"])            # seed 7 produces showdowns
        self.assertIn("note", self.out)
        for v in self.out["villains"]:
            self.assertGreaterEqual(v["showdowns"], 1)   # a villain appears ONLY if she showed down

    def test_internal_sums_consistent(self):
        for v in self.out["villains"]:
            ap = v["aggressive_postflop"]
            self.assertEqual(sum(ap["strength_counts"].values()), ap["n"])   # strength counts tile n
            self.assertEqual(sum(b["n"] for b in v["by_size"]), ap["n"])      # size buckets tile n
            self.assertLessEqual(ap["strong_or_better"], ap["n"])
            sc = ap["strength_counts"]
            self.assertEqual(ap["strong_or_better"], sc.get("strong", 0) + sc.get("monster", 0))

    def test_postflop_only_strength_vocab(self):
        # every recorded strength is a POSTFLOP made-hand label; a leaked preflop raise would carry
        # preflop_bucket vocab ('premium'/'trash') and fail here — this IS the postflop-only guard.
        for v in self.out["villains"]:
            seen = set(v["aggressive_postflop"]["strength_counts"])
            for b in v["by_size"]:
                seen |= set(b["strength_counts"])
            self.assertTrue(seen <= _STRENGTHS, f"non-postflop strength leaked: {seen - _STRENGTHS}")

    def test_size_buckets_present_and_ordered(self):
        for v in self.out["villains"]:
            self.assertEqual([b["size"] for b in v["by_size"]], ["small", "medium", "big", "overbet"])

    def test_aggression_signal_has_content(self):
        # the tell isn't vacuous — at least one villain bet/raised postflop on a hand she showed down
        self.assertTrue(any(v["aggressive_postflop"]["n"] > 0 for v in self.out["villains"]))

    def test_live_session_resolves_real_names(self):
        # the live session has no getup report yet → names must resolve via the live TABLE (the
        # production live-coaching path), not the P{idx} fallback. Assert the actual contract: every
        # surfaced name appears in the live player table (not merely "isn't a P# fallback").
        names = {v["name"] for v in self.out["villains"]}
        live_names = {p["name"] for p in self.g.s["players"][1:] if p.get("name")}   # villains, hero excluded
        self.assertTrue(names <= live_names, f"tell names {names} not all in the live table {live_names}")

    def test_folded_villain_bet_is_not_counted(self):
        # complement L3 / §3f no-leakage: a villain who BET postflop then FOLDED before showdown must
        # contribute NOTHING — her holes were never revealed, so counting her bet would be a spoiler.
        # Frozen REAL bot-generated hand (seed 2057366049): player 2 bet/raised postflop then folded;
        # players 1 & 3 reached showdown. Her postflop raise must be absent from the tell.
        from gutcheck import history
        d = Path(tempfile.mkdtemp())
        g = G.Game(d / "s.json", d / "c.json", d / "sess", d / "cal.json", d / "h.db")
        rec = {"hand_no": 1, "seed": 2057366049, "start_stacks": [200] * 6, "assign": [1, 2, 3, 4, 5, 0],
               "log": ["c", "f", "f", "f", "c", "r6", "c", "c", "c", "c", "c", "c", "r11", "r30", "c", "f", "c", "c"],
               "pot": 0, "hero_net": 0, "board": [], "street": "river"}
        history.record_hand("BTF", rec, path=g.history_db)
        pidxs = {v["player_index"] for v in g.sizing_tells("BTF")["villains"]}
        self.assertNotIn(2, pidxs, "a villain who bet then FOLDED pre-showdown leaked into the tell")
        self.assertEqual(pidxs, {1, 3})        # only the two showdown villains appear

    def test_session_scoping_isolates(self):
        out = self.g.sizing_tells("some-other-session-with-no-hands")
        self.assertEqual(out["villains"], [])                 # no leakage from the live session
        self.assertEqual(out["session_id"], "some-other-session-with-no-hands")

    def test_none_defaults_to_live_session(self):
        self.assertEqual(self.g.sizing_tells(None)["session_id"], self.sid)

    def test_incomplete_hand_does_not_fabricate_showdowns(self):
        # codex L3: a syntactically-valid but INCOMPLETE log (the hand never reaches a terminal state)
        # must NOT be read as a showdown — that would fabricate showdowns AND reveal strength from a
        # hand nobody saw down (a spoiler). One preflop call leaves the hand live; the tell must skip it.
        from gutcheck import history
        d = Path(tempfile.mkdtemp())
        g = G.Game(d / "s.json", d / "c.json", d / "sess", d / "cal.json", d / "h.db")
        rec = {"hand_no": 1, "seed": 12345, "start_stacks": [200] * 6, "assign": [0, 1, 2, 3, 4, 5],
               "log": ["c"], "pot": 3, "hero_net": 0, "board": [], "street": "preflop"}
        history.record_hand("INCOMPLETE", rec, path=g.history_db)
        self.assertEqual(g.sizing_tells("INCOMPLETE")["villains"], [])   # no fabricated showdowns

    def test_no_session_at_all_is_clean_empty(self):
        # a fresh, never-seated Game has no live session id → clean empty, no crash
        d = Path(tempfile.mkdtemp())
        g = G.Game(d / "s.json", d / "c.json", d / "sess", d / "cal.json", d / "h.db")
        out = g.sizing_tells(None)
        self.assertIsNone(out["session_id"])
        self.assertEqual(out["villains"], [])


class TestSizeBucket(unittest.TestCase):
    """Direct boundary tests for the bet-size bucketing — the seed-driven fixture only ever
    populates medium/big, so the bucket edges (half-open [lo, hi)) are locked down here."""

    def test_bucket_edges_are_half_open(self):
        cases = [
            (0.0, "small"), (0.25, "small"), (0.49, "small"),
            (0.5, "medium"), (0.66, "medium"), (0.84, "medium"),
            (0.85, "big"), (1.0, "big"), (1.24, "big"),
            (1.25, "overbet"), (2.0, "overbet"), (99.0, "overbet"),
        ]
        for frac, expected in cases:
            self.assertEqual(G._size_bucket(frac), expected, f"frac {frac} → {expected}")

    def test_buckets_tile_the_range(self):
        # every bucket label is reachable and the labels match _SIZE_BUCKETS order
        labels = [b[0] for b in G._SIZE_BUCKETS]
        self.assertEqual(labels, ["small", "medium", "big", "overbet"])
        hit = {G._size_bucket(f) for f in (0.0, 0.6, 1.0, 5.0)}
        self.assertEqual(hit, set(labels))


if __name__ == "__main__":
    unittest.main(verbosity=2)
