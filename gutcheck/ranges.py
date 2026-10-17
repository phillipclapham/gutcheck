"""Villain range model — maps an OBSERVED read (VPIP %, position, aggression) to a pokerkit range
string, so the LIVE coach can compute REAL modeled equity (equity.hero_vs) instead of eyeballing it.

Honest by construction:
- The range is derived from the OBSERVED VPIP (what any player at the table can see), NEVER the hidden
  archetype — so there is no truth-leak in the live panel. The resulting equity is therefore "equity
  IF your read is right"; the coach always states the assumed range explicitly so it stays legible and
  Phill can judge the read himself. This is the decision-time counterpart to the review drill's EXACT
  equity — fuzzier by nature (cards are hidden), which is exactly why the range is surfaced, not hidden.

First cut (sourdough — commit the shape, tune via play): a Chen-formula ordering of all 169 starting
hands → "the top X% of combos" where X is the observed VPIP (tightened when the villain has raised this
hand, nudged by position). Postflop uses that same preflop-continue range evaluated on the live board —
it does NOT yet model street-by-street narrowing (a documented simplification that runs slightly
pessimistic for the hero on later streets, since a real villain's range tightens as money goes in;
street-aware narrowing is Slice-4 range-refinement territory).
"""

from __future__ import annotations

import math

_RANKS = "23456789TJQKA"          # index 0..12 ; index+2 == the card's numeric rank (A=14)
_TOTAL_COMBOS = 1326              # C(52,2)

# A raise represents the TOP of a range, so a raiser's range is capped here regardless of their VPIP
# (a 60%-VPIP station that suddenly raises is not raising 60% of hands). Tunable via play.
_RAISE_CAP_PCT = 18.0
_DEFAULT_VPIP = 33.0             # no sample yet → a neutral default (unknowns trend tighter than 40);
                                # the coach flags low confidence separately via the hands-seen count
_MIN_WIDTH_PCT = 4.0            # never emit an empty range
_MAX_WIDTH_PCT = 92.0

# Set-mining floor (L2): a pocket pair's range-construction value FAR exceeds its Chen / raw-equity
# rank — a small pair's worth is the ~12% of the time it flops a set, which the strength ordering
# can't see. Real ranges include all pairs by ~18% width, so a Chen-only ordering produced raiser
# ranges that structurally COULDN'T hold a low set (a ~5-12pt equity distortion on low boards — the
# exact confident-but-wrong number this apparatus exists to kill). So top_pct_range ensures each pair
# is present once the width reaches its set-mining entry depth. TT+ already rank in via Chen.
_PAIR_ENTRY_PCT = {"99": 8, "88": 9, "77": 11, "66": 13, "55": 15, "44": 16, "33": 17, "22": 18}


def _high_points(num: int) -> float:
    """Chen-formula high-card points for a card's numeric rank (A=14)."""
    return {14: 10.0, 13: 8.0, 12: 7.0, 11: 6.0}.get(num, num / 2.0)


def _round_half_up(x: float) -> int:
    return math.floor(x + 0.5)


def _chen(i: int, j: int, suited: bool) -> int:
    """Chen-formula strength score for a starting hand given rank indices i, j (0..12)."""
    num_hi = max(i, j) + 2
    if i == j:                                   # pair
        return _round_half_up(max(_high_points(num_hi) * 2, 5.0))
    score = _high_points(num_hi)
    if suited:
        score += 2
    diff = abs(i - j)
    score -= {1: 0, 2: 1, 3: 2, 4: 4}.get(diff, 5)   # gap penalty (cards between = diff-1)
    if diff <= 2 and num_hi <= 11:               # straight bonus: 0/1 gap, both below Q
        score += 1
    return _round_half_up(max(score, 0.0))


def _all_hands() -> list[tuple[str, int, int]]:
    """Every one of the 169 starting hands as (notation, combo_count, chen_score)."""
    out = []
    for i in range(13):
        for j in range(13):
            if i == j:
                out.append((_RANKS[i] * 2, 6, _chen(i, j, False)))           # pair: 6 combos
            elif i > j:                                                       # high rank first
                out.append((_RANKS[i] + _RANKS[j] + "s", 4, _chen(i, j, True)))    # suited: 4
                out.append((_RANKS[i] + _RANKS[j] + "o", 12, _chen(i, j, False)))  # offsuit: 12
    return out


# Ordered strongest → weakest: Chen score desc, then high rank desc, then low rank desc. Computed once.
def _rank_key(h: tuple[str, int, int]) -> tuple:
    notation, _, score = h
    hi = _RANKS.index(notation[0])
    lo = _RANKS.index(notation[1])
    return (-score, -hi, -lo)


_ORDERED = sorted(_all_hands(), key=_rank_key)


def top_pct_range(pct: float) -> str:
    """The top `pct`% of starting hands by strength, as a pokerkit range string ('AA, KK, AKs, ...').
    Accumulates COMBO-weighted (a pair is 6 combos, suited 4, offsuit 12) down the strength order until
    it covers pct% of all 1326 combos — so the width matches a VPIP %, which is itself combo-weighted."""
    pct = max(_MIN_WIDTH_PCT, min(_MAX_WIDTH_PCT, pct))
    target = pct / 100.0 * _TOTAL_COMBOS
    picked, combos = [], 0
    for notation, n, _ in _ORDERED:
        if combos >= target:
            break
        picked.append(notation)
        combos += n
    # set-mining floor: add any pocket pair whose entry depth this width has reached but the Chen
    # ordering left out (a pair's set value isn't visible to a strength score — see _PAIR_ENTRY_PCT).
    have = set(picked)
    for pair, entry in _PAIR_ENTRY_PCT.items():
        if pct >= entry and pair not in have:
            picked.append(pair)
    return ", ".join(picked)


# position → a width multiplier: a player's range in a SPECIFIC seat skews tighter than their overall
# VPIP in early position, wider on the button (their VPIP is the cross-position average).
_POS_MULT = {"UTG": 0.80, "HJ": 0.88, "MP": 0.88, "CO": 1.08, "BTN": 1.20, "SB": 0.95, "BB": 1.0}


def effective_width(vpip_pct: float | None, position: str | None = None,
                    raised: bool = False) -> float:
    """The modeled range WIDTH (% of combos) for a villain, after the position skew and raise cap —
    surfaced alongside the equity so the coach can state the assumed range ('vs their ~55% range')."""
    width = _DEFAULT_VPIP if vpip_pct is None else float(vpip_pct)
    width *= _POS_MULT.get((position or "").upper(), 1.0)
    if raised:
        width = min(width, _RAISE_CAP_PCT)       # a raise = the strong top of the range
    return max(_MIN_WIDTH_PCT, min(_MAX_WIDTH_PCT, width))


def villain_range(vpip_pct: float | None, position: str | None = None,
                  raised: bool = False) -> str:
    """Model a villain's hand range from the OBSERVED read. vpip_pct None / no-sample → a neutral
    default (the coach should flag low confidence separately). raised=True caps the width to a value
    range (the top of their range). Returns a pokerkit range string for equity.hero_vs."""
    return top_pct_range(effective_width(vpip_pct, position, raised))


if __name__ == "__main__":
    # Smoke: widths should track VPIP, raises should tighten, and every string must parse in pokerkit.
    from pokerkit import parse_range
    for vp, pos, raised in [(10, "UTG", False), (10, "UTG", True), (28, "CO", False),
                            (55, "BTN", False), (55, "BTN", True), (None, "BB", False)]:
        r = villain_range(vp, pos, raised)
        combos = len(list(parse_range(r)))
        n = len(r.split(", "))
        print(f"VPIP={vp} {pos} raised={raised}: {n} hands / {combos} combos ({round(100*combos/1326)}%) "
              f"→ {r[:60]}{'…' if len(r) > 60 else ''}")
    print("\ntop 5% :", top_pct_range(5))
    print("169 hands generated:", len(_ORDERED), "| widest emitted (cap 92%):",
          len(top_pct_range(100).split(", ")), "hands")
