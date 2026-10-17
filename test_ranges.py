"""Villain range-model unit suite (stdlib unittest, zero deps).

Covers ranges.py: the Chen-ordered top-X% range builder, the VPIP→width mapping with the position skew
and raise cap, and the guarantee that every emitted range string parses in pokerkit (so equity.hero_vs
can consume it). The model is a read-based ESTIMATE, so assertions are on structure/monotonicity/bounds,
not exact hand lists.

    ./venv/bin/python test_ranges.py
"""

import unittest

from gutcheck import ranges
from pokerkit import parse_range


def _combos(range_str: str) -> int:
    return len(list(parse_range(range_str)))


class TestOrdering(unittest.TestCase):
    def test_aa_strongest_trash_at_bottom(self):
        order = [n for n, _, _ in ranges._ORDERED]
        self.assertEqual(order[0], "AA")
        self.assertTrue(order[-1].endswith("o"))    # weakest is low offsuit junk (exact hand = a Chen tiebreak)
        for premium in ("AA", "KK", "AKs", "QQ"):   # no top-tier hand leaked into the bottom
            self.assertNotIn(premium, order[-12:])

    def test_169_hands_generated(self):
        self.assertEqual(len(ranges._ORDERED), 169)   # 13 pairs + 78 suited + 78 offsuit

    def test_top5_is_premium_only(self):
        top = ranges.top_pct_range(5)
        for hand in ("AA", "KK", "QQ", "AKs"):
            self.assertIn(hand, top.split(", "))
        self.assertNotIn("72o", top.split(", "))


class TestTopPctRange(unittest.TestCase):
    def test_width_tracks_pct(self):
        # the emitted combo count should land near the requested % of 1326 (combo-weighted accumulation)
        for pct in (10, 25, 50, 70):
            combos = _combos(ranges.top_pct_range(pct))
            target = pct / 100 * 1326
            self.assertLess(abs(combos - target), 70, f"{pct}% → {combos} combos, target {target:.0f}")

    def test_monotonic(self):
        widths = [_combos(ranges.top_pct_range(p)) for p in (5, 15, 30, 50, 75)]
        self.assertEqual(widths, sorted(widths))      # wider pct → at least as many combos

    def test_clamps(self):
        self.assertGreater(_combos(ranges.top_pct_range(0)), 0)     # never empty (min-width floor)
        self.assertLess(_combos(ranges.top_pct_range(100)), 1326)   # never the full deck (92% cap)

    def test_set_mining_floor_includes_pairs(self):
        # pairs must be present by their set-mining entry width, even though Chen ranks them low (L2 HIGH:
        # a range without low sets distorts equity on low boards). All pairs in by ~18%; 88 by ~12%.
        all_pairs = {"22", "33", "44", "55", "66", "77", "88", "99", "TT", "JJ", "QQ", "KK", "AA"}
        wide = set(ranges.top_pct_range(20).split(", "))
        self.assertTrue(all_pairs <= wide, f"missing pairs at 20%: {all_pairs - wide}")
        tight = set(ranges.top_pct_range(12).split(", "))
        self.assertIn("88", tight)                                  # set-mines present in a tight range
        self.assertNotIn("22", tight)                              # but not the smallest yet (entry ~18%)


class TestVillainRange(unittest.TestCase):
    def test_all_outputs_parse_in_pokerkit(self):
        for vp in (None, 8, 22, 45, 70):
            for pos in ("UTG", "CO", "BTN", "SB", "BB"):
                for raised in (False, True):
                    r = ranges.villain_range(vp, pos, raised)
                    self.assertGreater(_combos(r), 0)               # parses + non-empty

    def test_raise_tightens(self):
        loose = _combos(ranges.villain_range(55, "BTN", raised=False))
        raised = _combos(ranges.villain_range(55, "BTN", raised=True))
        self.assertLess(raised, loose)                              # a raise caps to a value range

    def test_position_skew(self):
        early = _combos(ranges.villain_range(30, "UTG"))
        late = _combos(ranges.villain_range(30, "BTN"))
        self.assertLess(early, late)                               # same player wider on the button

    def test_no_sample_uses_default(self):
        self.assertGreater(_combos(ranges.villain_range(None, "BB")), 0)   # None VPIP → neutral default

    def test_effective_width_bounds(self):
        self.assertGreaterEqual(ranges.effective_width(0, "UTG"), ranges._MIN_WIDTH_PCT)
        self.assertLessEqual(ranges.effective_width(100, "BTN"), ranges._MAX_WIDTH_PCT)
        self.assertLessEqual(ranges.effective_width(80, "BTN", raised=True), ranges._RAISE_CAP_PCT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
