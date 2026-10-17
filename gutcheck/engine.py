"""Deterministic No-Limit Hold'em cash engine over pokerkit.

Integrity guarantee: cards come from a real seeded shuffle + pokerkit's real
hand evaluation, never from a narrator's imagination.

Determinism: seeding the global `random` immediately before `create_state`
fixes the ENTIRE deck order (the deck is shuffled once at creation; board cards
and burns just pop from that pre-shuffled deque). So a hand's full state ==
(seed, betting_action_log). We replay by re-seeding, recreating, and re-applying
the logged betting actions. Villain decisions at the live frontier are logged
verbatim, so their own RNG never needs to be reproducible.

All dealing/burning is automated; we drive only fold / call / raise.
"""

from __future__ import annotations

import random
import warnings
from dataclasses import dataclass, field

import pokerkit as pk

warnings.filterwarnings("ignore", category=UserWarning, module="pokerkit")

A = pk.Automation
_AUTOMATIONS = (
    A.ANTE_POSTING,
    A.BET_COLLECTION,
    A.BLIND_OR_STRADDLE_POSTING,
    A.CARD_BURNING,
    A.HOLE_DEALING,
    A.BOARD_DEALING,
    A.RUNOUT_COUNT_SELECTION,  # auto-run the board on all-ins (else CASH_GAME hands wedge)
    A.HOLE_CARDS_SHOWING_OR_MUCKING,
    A.HAND_KILLING,
    A.CHIPS_PUSHING,
    A.CHIPS_PULLING,
)

# 6-max position names by pokerkit seat index (seat0=SB, seat1=BB).
POSITIONS_6 = {0: "SB", 1: "BB", 2: "UTG", 3: "HJ", 4: "CO", 5: "BTN"}
STREETS = {0: "preflop", 1: "flop", 2: "turn", 3: "river"}


def position_name(seat: int, n: int) -> str:
    if n == 6:
        return POSITIONS_6[seat]
    # generic fallback
    if seat == 0:
        return "SB"
    if seat == 1:
        return "BB"
    if seat == n - 1:
        return "BTN"
    return f"P{seat}"


@dataclass
class Hand:
    seed: int
    small_blind: int
    big_blind: int
    starting_stacks: list[int]
    action_log: list[str] = field(default_factory=list)
    state: pk.State = field(init=False)
    history: list = field(default_factory=list, init=False)  # [{seat,token,street,facing}]

    def __post_init__(self) -> None:
        random.seed(self.seed)  # fixes the entire deck order for this hand
        self.state = pk.NoLimitTexasHoldem.create_state(
            _AUTOMATIONS,
            False,
            0,
            (self.small_blind, self.big_blind),
            self.big_blind,
            self.starting_stacks,
            len(self.starting_stacks),
            mode=pk.Mode.CASH_GAME,
        )
        # blinds posted per seat at creation (BLIND_OR_STRADDLE_POSTING automation) —
        # needed to attribute total contribution per seat for the uncalled-bet
        # correction in total_pot().
        self._blinds = list(self.state.bets)
        for token in self.action_log:
            self._capture_apply(token)

    def _capture_apply(self, token: str) -> None:
        """Apply an action while recording who acted, on what street, facing what,
        and how many chips it put in (`added` — for exact pot reconstruction; pot()
        reads 0 once pokerkit auto-pushes at hand end, so we accumulate from here).

        `added` is derived from the action INTENT read PRE-apply (facing for a call,
        target-minus-already-in for a raise, both capped at stack) — NOT from the
        post-apply stack delta. The hand-ending action triggers pokerkit's automated
        CHIPS_PUSHING, which credits the winner's stack inside `_apply`; a post-delta
        read would then see the actor's stack RISE and record a bogus negative `added`,
        corrupting total_pot() exactly when the closing actor is a winner. Reading the
        committed amount before the push is push-immune."""
        seat = self.state.actor_index
        street = self.state.street_index or 0
        facing = self.state.checking_or_calling_amount or 0
        # No open-folds. pokerkit PERMITS folding when nothing is owed, but such a
        # "non-standard fold" corrupts side-pot settlement (push_chips asserts when
        # everyone folds down to one live player with a multi-index side pot) — and
        # it's strictly dominated by checking. Refuse it structurally so a bad
        # deterministic log can never be created or replayed.
        if token == "f" and facing == 0:
            raise ValueError("open-fold illegal (nothing to call → check instead)")
        stack_before = self.state.stacks[seat]
        bet_before = self.state.bets[seat]
        if token == "f":
            added = 0
        elif token == "c":
            added = min(facing, stack_before)
        elif token.startswith("r"):
            added = min(int(token[1:]) - bet_before, stack_before)
        else:
            added = 0
        self._apply(token)
        self.history.append({"seat": seat, "token": token, "street": street,
                             "facing": facing, "added": added})

    def total_pot(self) -> int:
        """The pot actually CONTESTED — total chips committed minus the uncalled
        bet that gets returned. Valid even after chips are pushed (built from the
        push-immune per-action `added`, not post-apply stack deltas).

        The last aggressor's excess over the next-highest contributor is returned,
        never won — so a shove over a shorter stack, or any bet that's folded to,
        must NOT inflate the pot. Subtracting (top - 2nd-highest contribution) is
        EXACT for any single hand, including multi-way side pots: every seat at or
        below the second-highest contribution is fully matched, so the only unmatched
        chips are the single top contributor's excess over the second-highest.
        """
        contrib = list(self._blinds)               # chips per seat (blinds posted)
        for e in self.history:
            contrib[e["seat"]] += e.get("added", 0)
        total = sum(contrib)
        if len(contrib) >= 2:
            top, second = sorted(contrib)[-2:][::-1]
            total -= max(0, top - second)          # return the uncalled excess
        return total

    def _apply(self, token: str) -> None:
        if token == "f":
            self.state.fold()
        elif token == "c":
            self.state.check_or_call()
        elif token.startswith("r"):
            self.state.complete_bet_or_raise_to(int(token[1:]))
        else:
            raise ValueError(f"bad action token: {token!r}")

    def record(self, token: str) -> None:
        """Apply a NEW frontier action and persist it to the log."""
        self._capture_apply(token)
        self.action_log.append(token)

    # ---- reads ----
    @property
    def over(self) -> bool:
        return not self.state.status

    @property
    def actor(self) -> int | None:
        return self.state.actor_index

    @property
    def n(self) -> int:
        return len(self.starting_stacks)

    @property
    def street_index(self) -> int:
        return self.state.street_index or 0

    def street_name(self) -> str:
        return STREETS.get(self.street_index, "?")

    def board(self) -> list[str]:
        return [repr(c) for cl in self.state.board_cards for c in cl]

    def hole(self, seat: int) -> list[str]:
        return [repr(c) for c in self.state.hole_cards[seat]]

    @property
    def stacks(self) -> list[int]:
        return list(self.state.stacks)

    @property
    def bets(self) -> list[int]:
        return list(self.state.bets)

    def pot(self) -> int:
        return sum(self.state.bets) + sum(p.amount for p in self.state.pots)

    def call_amount(self) -> int:
        return self.state.checking_or_calling_amount or 0

    def min_raise_to(self) -> int | None:
        try:
            return self.state.min_completion_betting_or_raising_to_amount
        except Exception:
            return None

    def max_raise_to(self) -> int | None:
        try:
            return self.state.max_completion_betting_or_raising_to_amount
        except Exception:
            return None

    def legal(self) -> dict:
        return {
            # fold only when facing a bet — no open-folds (see _capture_apply)
            "fold": self.state.can_fold() and self.call_amount() > 0,
            "check_or_call": self.state.can_check_or_call(),
            "raise": self.state.can_complete_bet_or_raise_to(),
            "call_amount": self.call_amount(),
            "min_raise_to": self.min_raise_to(),
            "max_raise_to": self.max_raise_to(),
        }

    def live_seats(self) -> list[int]:
        return [i for i, s in enumerate(self.state.statuses) if s]

    def net(self) -> list[int]:
        """Net chips per seat (this hand). Valid once the hand is over."""
        return [self.state.stacks[i] - self.starting_stacks[i] for i in range(self.n)]


if __name__ == "__main__":
    # Full-hand smoke: limp to showdown, verify board + chip conservation + replay.
    h = Hand(seed=99, small_blind=1, big_blind=2, starting_stacks=[200] * 6)
    while not h.over and h.actor is not None:
        h.record("c")
    print("board:", h.board(), "| over:", h.over, "| sum stacks:", sum(h.stacks))
    print("net:", h.net())
    # replay from the same seed+log must reproduce identical board
    h2 = Hand(seed=99, small_blind=1, big_blind=2, starting_stacks=[200] * 6,
              action_log=list(h.action_log))
    print("replay board matches:", h2.board() == h.board(), "| net matches:", h2.net() == h.net())
