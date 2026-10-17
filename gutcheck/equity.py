"""Equity oracle — REAL hand-vs-range equity over pokerkit's calculate_equities.

The keystone of the felt-sense prediction-error loop (SCOPE decision 7): this is the
GROUND-TRUTH number Phill's gut is checked against. Computed in CODE, NEVER by the
coaching LLM — the entire point of Slice 3 is that the equity is real, not eyeballed,
and an LLM-generated number would be back to guessing (and fabricable). game.py bakes
the result into the frozen coach context; the coach reads it as a given and explains it.

Card format: the engine carries cards as `repr(card)` strings ('Ah','Kd',Td',...),
which is byte-identical to pokerkit's `Card.parse` input — so the bridge is a plain join.
Ranges are pokerkit range strings ('22+, A2s+, KQo') OR a single concrete combo ('AhKd')
— `parse_range` accepts both (a concrete combo = a 1-combo range, verified).

Determinism note: calculate_equities is Monte-Carlo when cards are still to come, so the
result wiggles ±~0.5% run-to-run at the default sample count. We round to whole percents
on the way out — Phill's gut is not calibrated to half a point, and implying that precision
would be a lie. On a complete board (river) there's nothing to sample: every "sample" is
the same showdown, so the number is exact regardless of sample_count.
"""

from __future__ import annotations

import itertools
import warnings

from pokerkit import Card, Deck, StandardHighHand, calculate_equities, parse_range

warnings.filterwarnings("ignore", category=UserWarning, module="pokerkit")

# The full 52-card deck as repr strings, for exact runout enumeration (the review path).
_DECK = [repr(c) for c in Deck.STANDARD]

# Monte-Carlo sample count. 8000 is the knee of the accuracy/latency curve (measured): run-to-run
# spread ~1% (vs 1.6% at 4000), heads-up ~1.5s, 3-way ~2.2s. The ~1% wiggle is invisible at
# gut-estimation resolution but tight enough that re-asking the same spot reads as "ground truth,"
# not a number that drifts. Multiway preflop runs slower (~3.7s 4-way) but those all-live spots are
# rare and this runs in the coach worker, off the app's hot path. Overridable per call.
DEFAULT_SAMPLES = 8000

# hold'em: 2 hole cards each, a 5-card board.
_HOLE_COUNT = 2
_BOARD_COUNT = 5


# Felt-sense translation of an equity % into the language a felt-sense player actually reads. The
# percentage is the precise ground truth, but the BAND is how the gut LEARNS to map onto it over reps
# (Phill plays by feel and doesn't read raw equity yet): the drill/coach lead with the band, the %
# calibrates it, and "I felt a favorite → it was 72%" teaches the gut what 72% feels like in context.
# JS mirror in static/index.html (`eqBand`) MUST keep these thresholds identical.
_EQUITY_BANDS = [(2, "drawing dead"), (18, "drawing thin"), (42, "behind"),
                 (58, "a coin flip"), (75, "a favorite"), (90, "a clear favorite")]


def band(pct: float | None) -> str | None:
    """An equity % → a felt-sense label ('drawing dead' … 'crushing'). None passes through."""
    if pct is None:
        return None
    for hi, label in _EQUITY_BANDS:
        if pct < hi:
            return label
    return "crushing"


def hole_str(hole: list[str]) -> str:
    """['Ah','Kd'] -> 'AhKd' (the engine's repr-card list -> a Card.parse / parse_range token)."""
    return "".join(hole)


def board_str(board: list[str] | None) -> str:
    """['Qs','7d','2c'] -> 'Qs7d2c'; [] / None -> '' (preflop, an empty board)."""
    return "".join(board or [])


def _dead_set(board: str, *concrete: str) -> set[str]:
    """The cards that cannot appear in any opponent range: the board + every known concrete
    holding (the hero's cards, plus any villain whose actual hand we already know). Used to
    detect a range that is fully blocked, which calculate_equities cannot evaluate."""
    dead: set[str] = set()
    for chunk in (board, *concrete):
        for c in Card.parse(chunk):
            dead.add(repr(c))
    return dead


def _range_has_live_combo(range_str: str, dead: set[str]) -> bool:
    """Does this range contain at least one combo that conflicts with NO dead card? A range that
    is entirely blocked (e.g. a villain whose only modeled holding shares a card with the board or
    the hero) leaves calculate_equities with nothing to sample -> we must refuse, not feed it."""
    for combo in parse_range(range_str):
        if not any(repr(c) in dead for c in combo):
            return True
    return False


def equities(ranges: list[str], board: list[str] | None = None,
             *, samples: int = DEFAULT_SAMPLES) -> list[float] | None:
    """Raw equity per entrant, in the order given, as floats in [0, 100] rounded to 1 decimal.

    ranges[i] is a pokerkit range string OR a concrete combo ('AhKd'). board is the engine's
    repr-card list (or [] preflop). Returns None — never raises — if the spot is uncomputable
    (fewer than 2 entrants, an empty/all-dead range, or any pokerkit failure); the caller treats
    None as "no equity available" and the coach falls back to pot-odds reasoning."""
    if not ranges or len(ranges) < 2:
        return None
    bstr = board_str(board)
    # everything from here can choke on a malformed card/range string (Card.parse / parse_range raise),
    # so it ALL lives under the fail-closed try — equities() promises None, never an exception (codex L3).
    # The only universally-dead cards are the board (we don't know the other ranges' combos); a concrete
    # range is still blocked-checked against the board so "villain holds a board card" is refused cleanly.
    try:
        dead = _dead_set(bstr)
        for r in ranges:
            if not _range_has_live_combo(r, dead):
                return None
        raw = calculate_equities(
            [parse_range(r) for r in ranges],
            Card.parse(bstr),
            _HOLE_COUNT, _BOARD_COUNT, Deck.STANDARD, (StandardHighHand,),
            sample_count=samples,
        )
    except Exception:
        return None
    return [round(100.0 * x, 1) for x in raw]


def hero_vs(hero_hole: list[str], villain_ranges: list[str], board: list[str] | None = None,
            *, samples: int = DEFAULT_SAMPLES) -> dict | None:
    """Hero-centric equity: the hero's % plus each villain's %, computed against the supplied
    villain ranges on the given board. The single oracle behind BOTH coach paths:

      - decision-time: villain_ranges = modeled ranges (from the OBSERVED read) -> "equity if
        your read is right";
      - review-time: villain_ranges = each villain's ACTUAL revealed combo -> exact ground truth.

    Returns {hero_pct, villains:[pct,...], samples, cards_to_come, exact} or None if uncomputable.
    `exact` is True when the board is complete (a river showdown — the number is not an estimate)."""
    if not hero_hole or len(hero_hole) != _HOLE_COUNT or not villain_ranges:
        return None
    board = board or []
    result = equities([hole_str(hero_hole)] + list(villain_ranges), board, samples=samples)
    if result is None:
        return None
    cards_to_come = _BOARD_COUNT - len(board)
    # 'exact' ONLY when there's nothing left to sample: a complete board AND every opponent is a single
    # concrete combo. A multi-combo RANGE on the river still Monte-Carlo-samples WHICH combo, so it's an
    # estimate — never label that exact (the hero hole is always 1 combo, so only villains need checking).
    exact = cards_to_come == 0 and all(len(list(parse_range(r))) == 1 for r in villain_ranges)
    return {
        "hero_pct": result[0],
        "villains": result[1:],
        "samples": samples,
        "cards_to_come": cards_to_come,
        "exact": exact,
    }


def modeled_from_spec(spec: dict | None) -> dict | None:
    """The LIVE modeled-equity blob from a frozen decision spec (the `modeled_spec` game.py bakes in
    coach_decision_context): the hero's % vs the target opponent's OBSERVED-read range, plus the spec
    passthrough the reveal + the coach prose narrate (vs / live_count / heads_up / pot_way).

    The SINGLE code path for the live number, so the drill reveal and the prose can never cite two
    different MC runs. Two callers share it: the server computes it synchronously at the live commit
    (anti-fabrication — the truth is recomputed from authoritative game state, never trusted from the
    browser), and the coach worker reuses it for the narration. Fail-safe: an empty or uncomputable
    spec yields None (no equity line), never a crash — a bad spot must not break the read."""
    if not spec:
        return None
    try:
        # `or 2500` (not a plain default) coerces a present-but-None/0 samples to the real default —
        # `hero_vs(samples=None)` would mis-run the MC; a malformed spec must degrade to the default,
        # never crash or sample zero.
        eq = hero_vs(spec["hero_hole"], [spec["range"]], spec.get("board"),
                     samples=spec.get("samples") or 2500)
    except Exception:
        return None
    if not eq:
        return None
    return {
        "hero_pct": eq["hero_pct"], "band": band(eq["hero_pct"]), "vs": spec.get("vs"),
        "live_count": spec.get("live_count"), "heads_up": spec.get("heads_up"),
        "pot_way": spec.get("pot_way"),
    }


def _exact_equities(holes: list[list[str]], board: list[str]) -> list[float] | None:
    """EXACT equity per player — enumerate EVERY remaining board runout, no sampling. holes[i] is a
    2-card list (hero first); board is 3-5 cards. The caller must keep cards-to-come <= 2 (river=0,
    turn=1, flop=2 → at most ~1326 runouts); preflop's ~1.7M runouts would explode, so hero_vs_known
    routes it to Monte Carlo instead. This is the review path: cards are revealed, so equity is a true
    fraction of outcomes, not an estimate — which is exactly the 'exact truth' the drill promises."""
    # Fail CLOSED on an impossible deck: any duplicate across holes+board means the spot can't exist,
    # and enumerating it would return a confident-but-fabricated "exact" number — exactly the lie this
    # oracle must never tell (the Monte-Carlo path already refuses dead ranges; the exact path must too).
    flat = [c for h in holes for c in h] + list(board)
    if len(set(flat)) != len(flat):
        return None
    used: set[str] = set(flat)
    rem = [c for c in _DECK if c not in used]
    need = _BOARD_COUNT - len(board)
    if need < 0 or need > len(rem):
        return None
    bstr = "".join(board)
    eq = [0.0] * len(holes)
    runs = 0
    for combo in itertools.combinations(rem, need):
        full = bstr + "".join(combo)
        hands = [StandardHighHand.from_game("".join(h), full) for h in holes]
        best = max(hands)
        winners = [i for i, x in enumerate(hands) if x == best]
        share = 1.0 / len(winners)
        for i in winners:
            eq[i] += share
        runs += 1
    if runs == 0:
        return None
    return [round(100.0 * x / runs, 1) for x in eq]


def hero_vs_known(hero_hole: list[str], villain_holes: list[list[str]],
                  board: list[str] | None = None) -> dict | None:
    """Hero equity vs OPPONENTS WHOSE ACTUAL CARDS ARE KNOWN — the review/drill path. On a postflop
    board (<=2 cards to come) this enumerates every runout for a truly EXACT number; preflop it falls
    back to Monte Carlo (enumeration would explode). Same return shape as hero_vs; `exact` is True only
    when the number was enumerated. None if uncomputable."""
    if not hero_hole or len(hero_hole) != _HOLE_COUNT or not villain_holes:
        return None
    board = board or []
    holes = [list(hero_hole)] + [list(v) for v in villain_holes]
    if any(len(h) != _HOLE_COUNT for h in holes):
        return None
    # Fail closed on an impossible deck on BOTH paths — the preflop fallback (below) must not slip a
    # duplicate-card spot past the guard that only _exact_equities applies (codex L3).
    flat = [c for h in holes for c in h] + list(board)
    if len(set(flat)) != len(flat):
        return None
    cards_to_come = _BOARD_COUNT - len(board)
    if cards_to_come > 2:                       # preflop / too many runouts — sample instead
        return hero_vs(hero_hole, [hole_str(v) for v in villain_holes], board)
    try:
        res = _exact_equities(holes, list(board))
    except Exception:
        return None
    if res is None:
        return None
    return {"hero_pct": res[0], "villains": res[1:], "samples": None,
            "cards_to_come": cards_to_come, "exact": True}


if __name__ == "__main__":
    # Smoke: known spots pin the oracle to reality (the whole point — real, not eyeballed).
    aa = hero_vs(["Ah", "As"], ["KhKs"], [])
    print("AA vs KK preflop:", aa["hero_pct"], "% (expect ~82)")

    # set vs overpair on a dry board — hero crushing, mostly locked with 2 to come
    setv = hero_vs(["7c", "7d"], ["AhAs"], ["7s", "2c", "9h"])
    print("77 (set) vs AA on 7s2c9h:", setv["hero_pct"], "% | exact:", setv["exact"])

    # exact river showdown — top pair vs a made flush, deterministic (no sampling)
    riv = hero_vs(["Ah", "Kd"], ["JhTh"], ["Qh", "7h", "2c", "Kc", "3d"])
    print("AK (pair of K) vs J-high flush, river:", riv["hero_pct"], "% | exact:", riv["exact"])

    # vs a wide station range on the flop, hero with top pair
    rng = hero_vs(["As", "Qd"], ["22+, A2s+, K5s+, Q8s+, J8s+, T8s+, A7o+, KTo+, QTo+, JTo"],
                  ["Qs", "7d", "2c"])
    print("AQ (top pair) vs wide range on Qs7d2c:", rng["hero_pct"], "%")

    # the review path — EXACT equity vs a known hand (enumerated, no sampling)
    k_riv = hero_vs_known(["Ah", "Kd"], [["Jh", "Th"]], ["Qh", "7h", "2c", "Kc", "3d"])
    print("KNOWN AK(pairK) vs JT, river:", k_riv["hero_pct"], "% | exact:", k_riv["exact"])
    k_flop = hero_vs_known(["7c", "7d"], [["Ah", "As"]], ["7s", "2c", "9h"])
    print("KNOWN 77(set) vs AA, flop:", k_flop["hero_pct"], "% | exact:", k_flop["exact"])
    k_3way = hero_vs_known(["As", "Qd"], [["Ah", "Kh"], ["7c", "7s"]], ["Qs", "7d", "2c"])
    print("KNOWN AQ vs {AK, 77set} 3-way flop:", k_3way["hero_pct"], "% | exact:", k_3way["exact"])

    # uncomputable guards
    print("None on <2 entrants:", equities(["AhKd"], []) is None)
    print("None on all-dead range:", hero_vs(["Ah", "Kd"], ["AhKd"], ["Qs", "7d", "2c"]) is None)
