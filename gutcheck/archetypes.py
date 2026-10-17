"""Villain archetypes: legible, tunable, exploitable opponents.

Three v0 player types, each a distinct exploit lesson:
  - rock     "The Nit"   : ultra-tight, value-only, never bluffs, folds to heat.
                            Exploit: steal relentlessly; fold when it wakes up.
  - reg      "The TAG"   : tight-aggressive, c-bets, value-bets, occasional bluff.
                            Exploit: the toughest; play straightforward, pick spots.
  - station  "The Calling Station": loose-passive, calls too much, rarely folds/raises.
                            Exploit: value-bet thin and relentlessly; never bluff it.

Decisions key off a hand-strength read + position + a seeded RNG. Made-hand
strength only in v0 (draws are narrated by the coach layer, not played by bots).
"""

from __future__ import annotations

import pokerkit as pk

RANK_ORDER = "23456789TJQKA"
LABEL_TIER = {
    "High card": 0, "One pair": 1, "Two pair": 2, "Three of a kind": 3,
    "Straight": 4, "Flush": 5, "Full house": 6, "Four of a kind": 7,
    "Straight flush": 8,
}


def _rv(card: str) -> int:
    return RANK_ORDER.index(card[0])


def preflop_bucket(hole: list[str]) -> str:
    """premium / strong / playable / trash."""
    r1, r2 = hole[0][0], hole[1][0]
    s1, s2 = hole[0][1], hole[1][1]
    v1, v2 = _rv(hole[0]), _rv(hole[1])
    hi, lo = max(v1, v2), min(v1, v2)
    pair = r1 == r2
    suited = s1 == s2
    gap = hi - lo
    T, A_, K, Q, J = (RANK_ORDER.index(x) for x in "TAKQJ")

    if pair and hi >= T:                       # TT+
        return "premium"
    if hi == A_ and lo >= K:                    # AK
        return "premium"
    if hi == A_ and lo >= Q:                    # AQ
        return "strong"
    if pair:                                    # 22-99
        return "strong" if hi >= RANK_ORDER.index("7") else "playable"
    if hi == A_:                                # any other ace
        return "playable" if (suited or lo >= T) else "trash"
    if hi >= K and lo >= T:                     # KT+/KJ/KQ
        return "strong" if suited else "playable"
    if suited and gap <= 2 and lo >= RANK_ORDER.index("4"):  # suited connectors
        return "playable"
    if hi >= J and lo >= RANK_ORDER.index("9"):  # J9+/JT
        return "playable"
    return "trash"


def postflop_strength(hole: list[str], board: list[str]) -> str:
    """air / weak / medium / strong / monster (made-hand only)."""
    h = pk.StandardHighHand.from_game("".join(hole), "".join(board))
    tier = LABEL_TIER[h.entry.label.value]
    if tier == 0:
        return "air"
    if tier == 1:  # one pair — refine
        board_ranks = sorted((_rv(c) for c in board), reverse=True)
        hole_ranks = [_rv(c) for c in hole]
        top_board = board_ranks[0]
        if hole[0][0] == hole[1][0] and hole_ranks[0] > top_board:
            return "strong"             # overpair
        if top_board in hole_ranks:
            return "medium"             # top pair
        return "weak"                   # middle/bottom/board pair
    if tier == 2:
        return "strong"                 # two pair
    return "monster"                    # set+


STRENGTH_RANK = {"air": 0, "weak": 1, "medium": 2, "strong": 3, "monster": 4,
                 "trash": 0, "playable": 1, "premium": 4}


def _clamp_raise(ctx: dict, target: int) -> str:
    lo, hi = ctx["legal"]["min_raise_to"], ctx["legal"]["max_raise_to"]
    if lo is None:
        return "c"
    target = max(lo, min(hi, int(target)))
    return f"r{target}"


def decide(archetype: str, ctx: dict) -> str:
    """Return an action token: 'f' | 'c' | 'rN'.

    ctx keys: hole, board, legal, pot, bb, street_index, rng, facing_bet,
              current_level (the bet amount to call up to this street).
    """
    legal = ctx["legal"]
    rng = ctx["rng"]
    facing = legal["call_amount"] > 0
    can_raise = legal["raise"]
    pot = ctx["pot"]
    bb = ctx["bb"]

    if ctx["street_index"] == 0:
        strength = STRENGTH_RANK[preflop_bucket(ctx["hole"])]
        open_to = 3 * bb
    else:
        strength = STRENGTH_RANK[postflop_strength(ctx["hole"], ctx["board"])]
        open_to = ctx["current_level"] + max(bb, int(0.66 * pot))

    # ---- archetype policies ----
    if archetype == "rock":
        # plays only strong+; raises premium/monster for value; never bluffs.
        if not facing:
            if can_raise and strength >= 4:
                return _clamp_raise(ctx, open_to)
            if strength >= 3:
                return "c"               # slow with strong but not nut
            return "c" if legal["call_amount"] == 0 else "f"
        # facing a bet
        if strength >= 4 and can_raise and rng.random() < 0.6:
            return _clamp_raise(ctx, open_to)
        if strength >= 3:
            return "c"
        return "f"

    if archetype == "reg":
        # TAG: opens strong, c-bets, value-raises, bluffs ~12%.
        bluff = rng.random() < 0.12
        if not facing:
            if can_raise and (strength >= 3 or (strength >= 2 and rng.random() < 0.5) or bluff):
                return _clamp_raise(ctx, open_to)
            if strength >= 1:
                return "c"
            return "c" if legal["call_amount"] == 0 else "f"
        if strength >= 3 and can_raise and rng.random() < 0.55:
            return _clamp_raise(ctx, open_to)
        if strength >= 2:
            return "c"
        if strength >= 1 and ctx["street_index"] == 0:   # call one bet w/ playable preflop
            return "c"
        if bluff and can_raise and rng.random() < 0.4:
            return _clamp_raise(ctx, open_to)
        return "f"

    if archetype == "station":
        # loose-passive: calls almost anything, rarely raises, almost never folds.
        if not facing:
            if can_raise and strength >= 4 and rng.random() < 0.4:
                return _clamp_raise(ctx, open_to)
            return "c"
        if strength >= 4 and can_raise and rng.random() < 0.3:
            return _clamp_raise(ctx, open_to)
        # calls with anything but the rawest air, and even some air
        if strength >= 1 or rng.random() < 0.45:
            return "c"
        return "f"

    raise ValueError(f"unknown archetype: {archetype!r}")


ARCHETYPES = {
    "rock": "The Nit",
    "reg": "The TAG",
    "station": "The Calling Station",
}
