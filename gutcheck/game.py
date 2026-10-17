"""gutcheck core game logic — session state machine returning structured views.

Canonical logic for the web surface (server.py). Wraps engine.py + archetypes.py.

Two persistence layers (Slice 0b):
  - career.json  — DURABLE ledger: bankroll, lifetime stats, bankroll-over-time,
                   session summaries. Survives table resets / corrupt session loads.
  - session.json — the ACTIVE table (a "sit-down"): seating, stacks, current hand,
                   and a session accumulator (per-hand replay records, hero
                   aggression) that freezes into a report on get-up.
Full session reports land in sessions/<id>.json (the artifact the coach reads).

Lifecycle: sit down (buy in from bankroll) → play hands → bust? reload-from-bankroll
or get up → get up (cash out → report + career update). Villains are drawn fresh each
sit-down from a name pool with RANDOM archetype assignment, so a name carries zero
type info — reads must come from observed play (the anonymous-table training the
reading-trainer pivot is for).

(play.py is the earlier CLI prototype; this module supersedes it for the app.)
"""

from __future__ import annotations

import collections
import json
import math
import os
import random
import re
import secrets
import unicodedata
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

from . import equity
from . import history
from . import paths
from . import ranges
from .engine import Hand, position_name, STREETS
from .archetypes import decide, ARCHETYPES, preflop_bucket, postflop_strength

# Persisted state lives in a per-user data dir (paths.py / GUTCHECK_DATA_DIR), never the
# repo. The Game defaults below resolve it LAZILY (in __init__, not at import) so a test or
# smoke that sets GUTCHECK_DATA_DIR redirects every write — the structural fix for the
# 2026-06-18 default-path corruption (a smoke debited the real bankroll via bare Game()).

SB, BB = 1, 2
BUYIN = 200                 # 100bb buy-in
DEFAULT_BANKROLL = 4000     # 20 buy-ins
N_VILLAINS = 5              # 6-max: hero + 5

# Calibration trend (the felt-sense read-accuracy log). Two SOURCES, kept un-entangled:
# "exact"   = the review drill, gut vs EXACT enumerated equity (clean ground truth → the headline arc).
# "modeled" = the live commit-beat, gut vs MODELED equity (carries the range-model's own error → flagged).
CALIB_CAP = 2000            # capped recent sample log; lifetime counters survive truncation
CALIB_RECENT = 40          # the recent window: the sparkline, the direction arrow, the recent_* stats
_LEANS = {"fold", "call", "check", "bet", "raise"}   # facing a bet → fold/call/RAISE; checked to → check/bet
_NOTE_MAX = 280            # "quick read notes" — tweet-length prose, not an essay; bound to keep the
                           # calibration store + the coach prompt small (player-supplied → sanitize)

# --- sizing-tell (P3(c)) constants ---
# Coarse bet-size buckets as a fraction of the pot before the action. Coarse on purpose: the v0
# villains size ~constantly (≈⅔ pot), so finer buckets would manufacture false precision; the
# half-open [lo, hi) ranges tile [0, ∞) so every bet/raise lands in exactly one bucket.
_SIZE_BUCKETS = (("small", 0.0, 0.5), ("medium", 0.5, 0.85), ("big", 0.85, 1.25), ("overbet", 1.25, float("inf")))
# made-hand strengths (archetypes.postflop_strength) that count as a value-grade "strong+" hand —
# the aggression-tell summary ("she bet and showed strong") keys on these. 'medium' (one pair) is NOT.
_STRONG_PLUS = ("strong", "monster")
_SIZING_NOTE = ("Showdown-observed only: counts come from hands that reached showdown (her holes were "
                "revealed), so the read stays earnable — folded-out hands are excluded. Postflop "
                "bet/raises only; strength is measured at the STREET she bet, not the final showdown.")


def _size_bucket(frac: float) -> str:
    """Map a bet-size fraction (of the pot before the action) to its coarse _SIZE_BUCKETS label. The
    half-open [lo, hi) ranges tile [0, ∞), so every non-negative frac lands in exactly one bucket
    (0.0 → 'small', the upper edge of each bucket belongs to the NEXT one). Upstream guarantees frac
    ≥ 0 (raise_to/pot, both ≥ 0); the final-bucket fallback is purely defensive."""
    for label, lo, hi in _SIZE_BUCKETS:
        if lo <= frac < hi:
            return label
    return _SIZE_BUCKETS[-1][0]   # frac == +inf (impossible here) — the overbet tail


def _clean_note(x) -> str | None:
    """Sanitize an optional player-supplied read note: a normalized, trimmed, length-capped string.
    This is the player's OWN read in prose (the signal the digit is a lossy projection of), NOT a
    truth claim — so it's accepted from the browser; bounded so a fat-fingered/hostile client can't
    bloat the store or the coach prompt. Non-string → None. The cleaning, in order:
      - Unicode space separators (NBSP/NNBSP/ideographic space, category Zs) → an ASCII space, so
        pasted text keeps its word boundaries instead of fusing ("weak<NBSP>kicker" → "weak kicker"; L3).
      - control / zero-width / format / line-&-paragraph separators → DROPPED (str.strip alone leaves
        these, so a "​"-only note would read as blank yet still persist into the coach's context; L1).
      - ordinary horizontal/line whitespace (tab/newline) is kept so a real multi-line note survives.
      - "blank" → None: a note with no VISIBLE base char (only whitespace and/or combining marks) is
        not stored (L3). Combining marks attached to real text are kept."""
    if not isinstance(x, str):
        return None
    out = []
    for ch in x:
        if unicodedata.category(ch) == "Zs":          # any Unicode space separator → normalize, don't drop
            out.append(" ")
        elif ch.isprintable() or ch in "\t\n":        # real text + ordinary horizontal/line whitespace
            out.append(ch)
        # else: control / format / zero-width / Zl/Zp separators → dropped
    t = "".join(out).strip()
    # require at least one visible base char — a note of only whitespace and/or combining marks is blank
    if not any(not c.isspace() and not unicodedata.category(c).startswith("M") for c in t):
        return None
    return t[:_NOTE_MAX]


def _is_int(x) -> bool:
    # bool is an int subclass, so isinstance(True, int) is True — reject it explicitly so a JSON
    # `true`/`false` can't slip through a gut/n/id validator and log a degenerate read (L1/L2 LOW).
    return isinstance(x, int) and not isinstance(x, bool)


def _clean_pct(x) -> float | None:
    # a calibration value (gut or truth) must be a finite number in [0, 100]. NaN/Inf would poison
    # the durable JSON (Infinity/NaN are not valid JSON); out-of-range corrupts the trend average.
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v < 0 or v > 100:
        return None
    return v


def _derive_calibration(gut: int, truth: float) -> dict:
    """Every derived field is a pure function of (gut, truth). Banded from the FULL-PRECISION truth
    so the trend's verdict matches the drill reveal (which bands the unrounded equity), and computed
    by this SAME function at write AND load — so a hand-edited / corrupt derived field on disk can
    never reach the summary (it's always recomputed), and load == write by construction (L3 codex+kimi)."""
    gut_band, truth_band = equity.band(gut), equity.band(truth)
    return {"gap": round(gut - truth, 1), "abs_gap": round(abs(gut - truth), 1),
            "gut_band": gut_band, "truth_band": truth_band,
            "band_hit": bool(gut_band == truth_band)}

# Archetype keys a villain can be dealt. A villain's NAME is independent of its
# type (random pairing per session) — the whole point of the reading trainer.
ARCH_KEYS = ["rock", "reg", "station"]

# Neutral name pool — nothing here hints at a play style. Random 5 per session.
NAME_POOL = [
    "Max", "Ivy", "Leo", "Zoe", "Ray", "Cleo", "Gus", "Mae", "Otto", "Pip",
    "Vera", "Hank", "Lola", "Sid", "Nico", "Rosa", "Dex", "Tess", "Bo", "Juno",
    "Cyrus", "Wanda", "Felix", "Greta", "Moe",
]


def _action_phrase(token: str, facing: int) -> str:
    """A betting action rendered as English ('raises to 12' / 'calls 6' / 'checks').
    Shared by the live decision log and the review replay (one phrasing, both surfaces)."""
    if token == "f":
        return "folds"
    if token == "c":
        return "checks" if facing == 0 else f"calls {facing}"
    if token.startswith("r"):
        to = int(token[1:])
        return f"raises to {to}" if facing > 0 else f"bets {to}"
    return token


def _select_target(cand: list[dict], history: list[dict], live_villains: list[int]) -> dict:
    """Pick the single opponent to model live equity against: the LAST aggressor by ACTION order — the
    live pressure source — walking the betting history backwards (NOT seat order: BTN-opens / SB-3bets
    must target the SB, not the higher-seat BTN; codex+kimi L3). If nobody live has raised, the tightest
    live range (the clearest threat). `cand` entries carry a `seat`; `cand` is always non-empty here."""
    last_agg = next((e["seat"] for e in reversed(history)
                     if e["seat"] in live_villains and e["token"].startswith("r")), None)
    return next((c for c in cand if c["seat"] == last_agg), None) \
        or min(cand, key=lambda c: c["range_pct"])


class Game:
    def __init__(self, path: Path | None = None, career_path: Path | None = None,
                 reports_dir: Path | None = None, calib_path: Path | None = None,
                 history_db: Path | None = None, leak_store=None):
        self.path = path or paths.session_path()
        self.career_path = career_path or paths.career_path()
        self.reports_dir = reports_dir or paths.reports_dir()
        self.calib_path = calib_path or paths.calibration_path()
        # The canonical hand-history store. Injected like every other path so a test/smoke
        # isolates it to a tmp dir — the same explicit-paths contract that stops any test from
        # writing real state (test_paths.py); the per-hand history write must inherit it.
        self.history_db = history_db or paths.history_db()
        # P4 — the operator-owned, anneal-backed leak store (the moat). INJECTED, default None:
        # the 298 pre-P4 tests construct Game() with no leak store, so episode capture is OFF for
        # them (zero perturbation, zero new dep at the unit layer); the server wires a real
        # leakstore.LeakStore() on, and leak-specific tests inject one. A None / unavailable store
        # makes every read-capture call below a no-op — the game plays + calibrates exactly as before.
        self.leaks = leak_store
        self._career_recovered = False
        self.career = self._load_career()   # may set self._career_recovered
        self.calib = self._load_calib()
        self.s = self._load()
        if self._career_recovered and self.s.get("seated"):
            # the seated table's buy-in was debited from the career ledger that just
            # got reset (corrupt) — crediting its stack into a fresh 4000 bankroll
            # would CREATE money. The table is no longer trustworthy: discard it.
            self.s = self._fresh_unseated()
            self._save()
        elif (self.s.get("seated") and self.s.get("session") and
              any(x.get("id") == self.s["session"]["id"]
                  for x in self.career.get("sessions", []))):
            # idempotency: this session is ALREADY recorded in the career ledger (it
            # was cashed out) but a torn/reordered write left the table seated. Cashing
            # it out again would double-credit. Discard it — the cashout already landed.
            self.s = self._fresh_unseated()
            self._save()
        self._ensure_stats()

    # ---------------- persistence: career (durable) ----------------
    def _fresh_career(self) -> dict:
        return {"bankroll": DEFAULT_BANKROLL,
                "lifetime": {"hands": 0, "net": 0, "sessions": 0, "won": 0},
                "bankroll_history": [], "sessions": []}

    def _load_career(self) -> dict:
        if self.career_path.exists():
            try:
                c = json.loads(self.career_path.read_text())
                # tolerate older/partial files without losing the bankroll
                base = self._fresh_career()
                base.update({k: c[k] for k in base if k in c})
                base["lifetime"] = {**base["lifetime"],
                                    **(c.get("lifetime") or {})}
                return base
            except (json.JSONDecodeError, OSError, TypeError):
                # NEVER silently reset a career — back the bad file up so it's
                # recoverable, then start fresh. Flag it so __init__ can discard any
                # seated table whose buy-in was debited from this now-lost ledger.
                self._career_recovered = True
                try:
                    self.career_path.replace(
                        self.career_path.with_suffix(".corrupt.json"))
                except OSError:
                    pass
        return self._fresh_career()

    def _save_career(self) -> None:
        self._atomic_write(self.career_path, self.career, durable=True)

    def _bankroll_point(self, event: str) -> None:
        self.career["bankroll_history"].append(
            {"t": datetime.now().isoformat(timespec="seconds"),
             "bankroll": self.career["bankroll"], "event": event})
        # cap the series so the file can't grow without bound
        if len(self.career["bankroll_history"]) > 500:
            self.career["bankroll_history"] = self.career["bankroll_history"][-500:]

    # ---------------- persistence: calibration trend (the felt-sense read-accuracy log) ----------------
    def _fresh_calib(self) -> dict:
        return {"samples": [],
                "lifetime": {"exact":   {"n": 0, "band_hits": 0, "abs_gap_sum": 0},
                             "modeled": {"n": 0, "band_hits": 0, "abs_gap_sum": 0}}}

    @staticmethod
    def _clean_sample(s) -> dict | None:
        """Validate + NORMALIZE one persisted sample on load. Drop the row if gut/truth/ts are
        unusable (so a partial/old-schema/wrong-shaped record can't crash the summary or block
        __init__ — L1 MED-2/3); otherwise RECOMPUTE every derived field from gut+truth so a
        hand-edited / inconsistent derived value (e.g. abs_gap=9999 with band_hit=true) can never
        reach the summary (L3 codex+kimi MED). The returned dict is internally consistent."""
        if not isinstance(s, dict) or s.get("source") not in ("exact", "modeled"):
            return None
        gut, truth = _clean_pct(s.get("gut")), _clean_pct(s.get("truth"))
        if gut is None or truth is None or not isinstance(s.get("ts"), str):
            return None
        gut = int(round(gut))                     # gut is an integer percent
        lean = s.get("lean") if s.get("lean") in _LEANS else None
        street = s.get("street") if isinstance(s.get("street"), str) else None
        # normalize the join key on load: a hand-edited file must never publish a corrupt
        # (session_id, hand_no, n) to the coach as if it were authoritative — a wrong-typed value
        # reads back as "unknown" (None), never as a fabricated join (L3 codex; same anti-fab telos
        # as recomputing the derived fields below — corruption can't reach the coach's hot path).
        sid = s.get("session_id") if isinstance(s.get("session_id"), str) else None
        hand_no = s.get("hand_no") if _is_int(s.get("hand_no")) else None
        n = s.get("n") if _is_int(s.get("n")) else None
        return {"ts": s["ts"], "source": s["source"], "gut": gut, "truth": truth,
                **_derive_calibration(gut, truth), "lean": lean, "street": street,
                "session_id": sid, "hand_no": hand_no, "n": n,
                "note": _clean_note(s.get("note")),   # player's prose read; re-sanitized on load (hand-edit safe)
                "vs": s.get("vs"), "req_id": s.get("req_id")}

    def _load_calib(self) -> dict:
        base = self._fresh_calib()
        if self.calib_path.exists():
            try:
                c = json.loads(self.calib_path.read_text())
            except (json.JSONDecodeError, OSError):
                return self._fresh_calib()    # corrupt → start fresh (reflection data, not money)
            if not isinstance(c, dict):
                return self._fresh_calib()    # valid JSON but wrong shape ([]/"x"/5) — don't crash __init__
            raw = c.get("samples")
            if isinstance(raw, list):
                # drop malformed/old-schema rows rather than trust them into the summary's hot path
                base["samples"] = [s for s in (self._clean_sample(s)
                                               for s in raw[-CALIB_CAP:]) if s is not None]
            lt = c.get("lifetime") or {}
            for src in ("exact", "modeled"):
                s = lt.get(src)
                if isinstance(s, dict):
                    for k in ("n", "band_hits"):                  # counts must be non-negative ints
                        if _is_int(s.get(k)) and s[k] >= 0:
                            base["lifetime"][src][k] = s[k]
                    ags = s.get("abs_gap_sum")                    # a finite, non-negative number (Inf/NaN
                    if (isinstance(ags, (int, float)) and not isinstance(ags, bool)
                            and math.isfinite(ags) and ags >= 0):  # would re-emit as invalid JSON — L3 codex)
                        base["lifetime"][src]["abs_gap_sum"] = ags
                    # a hand-edited file could carry band_hits > n → a >100% rate; clamp (L1 LOW-3)
                    base["lifetime"][src]["band_hits"] = min(
                        base["lifetime"][src]["band_hits"], base["lifetime"][src]["n"])
        return base

    def _save_calib(self) -> None:
        # non-durable: a lost read is not a lost dollar, so no fsync ceremony (cf. the career ledger).
        self._atomic_write(self.calib_path, self.calib)

    def _record_calibration(self, source: str, gut, truth, lean, street,
                            session_id, hand_no, n, vs, req_id=None, note=None) -> dict | None:
        """Append one committed read to the trend, AFTER validating it — returns the sample, or None
        if the canonical truth is unusable (NaN/Inf/out-of-range), so a corrupt worker blob can never
        poison the durable JSON or skew the average (append-after-validate; L1 MED-5). Bands are
        recomputed from gut/truth via the SAME canonical equity.band() so the two judgments agree.
        Truth is supplied by the CALLER from a canonical source (recomputed exact equity / the
        worker's frozen modeled blob) — never trusted from the browser (the anti-fabrication invariant)."""
        truth_pct = _clean_pct(truth)     # finite, in [0,100] — append ONLY after this passes
        if truth_pct is None:
            return None
        gut = int(gut)
        d = _derive_calibration(gut, truth_pct)   # full-precision truth → bands match the drill reveal
        sample = {
            "ts": datetime.now().isoformat(timespec="seconds"),
            "source": source, "gut": gut, "truth": truth_pct,   # full precision; summary rounds for display
            **d, "lean": lean, "street": street, "session_id": session_id,
            "hand_no": hand_no, "n": n, "vs": vs, "req_id": req_id,   # (session_id, hand_no, n) =
            "note": _clean_note(note),                               # the join key the coach reads, never infers
        }                                                            # note = the player's prose read (optional)
        self.calib["samples"].append(sample)
        if len(self.calib["samples"]) > CALIB_CAP:
            self.calib["samples"] = self.calib["samples"][-CALIB_CAP:]
        lt = self.calib["lifetime"].setdefault(
            source, {"n": 0, "band_hits": 0, "abs_gap_sum": 0})
        lt["n"] += 1
        lt["band_hits"] += 1 if d["band_hit"] else 0
        lt["abs_gap_sum"] += abs(gut - truth_pct)   # float accumulator at full precision — the integer
        self._save_calib()                          # round floored sub-point error to "0.0 off" (L1 MED-4)
        # P4 — also record this read as an anneal episode in the operator's leak store (the citable
        # evidence a leak graduates from). Best-effort + archetype-free: ordered AFTER the durable
        # _save_calib so a leak-store hiccup can never lose the calibration sample just persisted
        # (leakstore swallows its own errors; the attr may be None on a no-anneal / test-built Game).
        leaks = getattr(self, "leaks", None)
        if leaks is not None and getattr(leaks, "available", False):
            leaks.record_read_episode(sample)
        return sample

    def record_review_read(self, session_id, hand_no, n, gut, lean, note=None) -> dict:
        """A read committed at the REVIEW drill → calibrate vs EXACT equity (the clean headline signal).
        Recomputes the canonical truth server-side from (session_id, hand_no) so a hand-edited report or
        a hostile client can't inject a fake truth into the trend. Re-reviewing a hand is a LEGITIMATE
        new rep (a fresh read at a fresh time), so this path is deliberately NOT deduped."""
        if not (isinstance(session_id, str) and _is_int(hand_no)
                and _is_int(n) and _is_int(gut) and 0 <= gut <= 100):
            return {"ok": False, "reason": "bad input"}
        built = self.coach_review_context(session_id, hand_no)
        if built is None:
            return {"ok": False, "reason": "hand not found"}
        ctx, _ = built
        dp = next((d for d in ctx.get("decision_points", []) if d.get("n") == n), None)
        if dp is None or dp.get("equity") is None:
            return {"ok": False, "reason": "no such decision / no equity"}
        lean = lean if lean in _LEANS else None
        s = self._record_calibration("exact", gut, dp["equity"], lean,
                                     dp.get("street"), session_id, hand_no, n, dp.get("vs"), note=note)
        if s is None:
            return {"ok": False, "reason": "unusable truth"}
        return {"ok": True, "truth": round(s["truth"], 1),   # stored full-precision; round for display
                "band_hit": s["band_hit"], "gap": s["gap"]}

    def record_live_read(self, gut, lean, truth_pct, street=None, vs=None, req_id=None,
                         hand_no=None, n=None, session_id=None, note=None) -> dict:
        """A read committed at the LIVE beat → calibrate vs MODELED equity (carries range-model error;
        tracked as a SEPARATE source so the headline exact-arc isn't entangled). The caller passes the
        worker's frozen modeled number (coach.db meta), never a browser-supplied value. A live coach id
        names ONE spot, so it logs ONCE — a double-submit (retry / double-click) is deduped (L1 MED-1)."""
        if not (_is_int(gut) and 0 <= gut <= 100) or truth_pct is None:
            return {"ok": False, "reason": "bad input"}
        lean = lean if lean in _LEANS else None
        # The join key is ALL-OR-NOTHING (the invariant that closes the cross-session wrong-join class):
        #  - a FROZEN session_id (from coach_decision_context) binds the WHOLE frozen tuple
        #    (session_id, hand_no, n) — the read is attributed to the session it was MADE in, so a
        #    delayed fire-and-forget commit after a get-up/sit-down can't join to a later session's hand.
        #  - WITHOUT a frozen session_id, the read takes the current session and CANNOT carry hand_no/n.
        # That second branch is what closes the residual: a stale/partial row (frozen hand_no/n but no
        # frozen session_id — e.g. a pre-P3a pending coach.db row) must NEVER mix a CURRENT session id
        # with a FROZEN hand ordinal (codex re-review HIGH). No frozen session_id ⇒ no joinable (hand_no, n).
        if isinstance(session_id, str):
            sid = session_id
            hand_no = hand_no if _is_int(hand_no) else None
            n = n if _is_int(n) else None
        else:
            sid = (self.s.get("session") or {}).get("id")
            hand_no = n = None      # unjoinable: a join key without a frozen session is never trusted
        # Dedup a double-commit of the SAME live decision — a re-click / re-open of the still-frozen spot
        # (the hand hasn't advanced) or a retried POST. One decision = one modeled sample; logging a second
        # would double-count a read whose estimate is already contaminated by the first reveal. Two
        # identities cover both transports:
        #   - req_id — the legacy coach-channel id (one live coach request = one spot).
        #   - the join key (session_id, hand_no, n) — the /api/read/commit code path carries NO req_id; the
        #     FULL frozen tuple names one decision. A preflop read (n=None) has no join key by design (the
        #     honest null), so it can't be deduped this way — accepted (the same edge the req_id scan has:
        #     it covers the retained CALIB_CAP window only — a single-operator hobby tool, not a migration).
        if any(s.get("source") == "modeled" and (
                   (req_id is not None and s.get("req_id") == req_id)
                   or (hand_no is not None and n is not None and s.get("session_id") == sid
                       and s.get("hand_no") == hand_no and s.get("n") == n))
               for s in self.calib["samples"]):
            return {"ok": False, "reason": "already logged"}
        s = self._record_calibration("modeled", gut, truth_pct, lean,
                                     street, sid, hand_no, n, vs, req_id=req_id, note=note)
        if s is None:
            return {"ok": False, "reason": "unusable truth"}
        return {"ok": True, "band_hit": s["band_hit"], "gap": s["gap"]}

    def calibration_summary(self) -> dict:
        """The trend view's data, split by source (exact = headline, modeled = flagged secondary).
        Two denominators, kept EXPLICIT so the UI never implies they share one (L1 NOTE / L2 MED):
        the all-time band_hit_rate/avg_abs_gap is the stable headline; the recent_* stats + the
        direction arrow describe the SAME window the sparkline draws, so arrow ⇔ chart."""
        out = {}
        for src in ("exact", "modeled"):
            lt = self.calib["lifetime"].get(src, {"n": 0, "band_hits": 0, "abs_gap_sum": 0})
            n = lt.get("n", 0)
            samples = [s for s in self.calib["samples"] if s.get("source") == src]
            recent = samples[-CALIB_RECENT:]
            rn = len(recent)
            direction = None
            if rn >= 10:                    # over the SAME window the sparkline shows (arrow ⇔ bars)
                half = rn // 2
                older, newer = recent[:half], recent[half:]
                o = sum(s["abs_gap"] for s in older) / len(older)
                nw = sum(s["abs_gap"] for s in newer) / len(newer)
                delta = nw - o              # negative = the gap is shrinking = sharpening
                direction = ("sharpening" if delta <= -3 else
                             "drifting" if delta >= 3 else "holding")
            out[src] = {
                "n": n,                     # all-time count (the headline denominator)
                "band_hit_rate": round(100 * lt.get("band_hits", 0) / n) if n else None,
                "avg_abs_gap": round(lt.get("abs_gap_sum", 0) / n, 1) if n else None,
                "recent_n": rn,             # the sparkline/arrow denominator (the recent window)
                "recent_band_hit_rate": (round(100 * sum(1 for s in recent if s["band_hit"]) / rn)
                                         if rn else None),
                "direction": direction,
                "recent": [{"abs_gap": s["abs_gap"], "band_hit": s["band_hit"],
                            "gut": s["gut"], "truth": round(s["truth"], 1),   # round for display
                            "street": s.get("street"), "ts": s["ts"],
                            "note": s.get("note"),   # the player's prose read — coach the prose-vs-digit gap
                            #   (their words are often sharper than their number; never let outs read as equity).
                            #   ⚠ player-supplied text: if ever rendered into the DOM, escapeHtml() it (the
                            #   trend view is innerHTML — an un-escaped note = stored XSS). Untrusted DATA to
                            #   the coach too (the operator's words, never instructions). [L1/L2 P3c]
                            # the join key — exposed so the coach reads which hand/decision a read came
                            # from instead of inferring it (anti-fab). null for PREFLOP/pre-P3a reads
                            # (live POSTFLOP reads now carry it → the modeled−exact reconciliation).
                            "session_id": s.get("session_id"), "hand_no": s.get("hand_no"),
                            "n": s.get("n")}
                           for s in recent],
            }
        return out

    def reconcile_live_reads(self, session_id=None) -> list[dict]:
        """P3a — the modeled−exact reconciliation. A LIVE read is graded vs MODELED equity (the
        villain's likely RANGE); once the hand completes, the EXACT equity (vs her ACTUAL cards)
        can diverge hugely — a station who flats the nuts collapses the range to a hand the model
        couldn't see. The trend stores the modeled '✓ nice read' and never reconciles. This joins
        each joinable live read to its decision via the (hand_no, n) key and surfaces the delta, so
        the coach can name 'you read her range right and were still crushed.'

        ``session_id`` scopes the join: a session-id STRING reconciles one session (the live coach +
        the hand-end range-collapse card); ``None`` reconciles EVERY joinable live read across all
        sessions (the cross-session range-collapse trend in 🎯 Reads). A read is joinable only with a
        FROZEN session_id of its own — the (hand_no, n) ordinal is meaningless without the session it
        was made in (hand_no COLLIDES across sessions), so the replay cache is keyed on the whole
        (session_id, hand_no) pair, NEVER hand_no alone (a hand_no-only cache would cross-join one
        session's read to another's exact equity and fabricate a delta).

        EXACT equity comes from the SAME oracle the review drill uses (coach_review_context's
        decision_points, sourced from history.db so it works mid-session) — never a second equity
        path, so the reconciliation can't disagree with the drill. Returns FACTS (modeled, exact,
        delta, felt bands); the COACH narrates them (code computes, the LLM never fabricates the
        number). Reads without a join key (preflop live reads, or pre-P3a reads) are SKIPPED — the
        join is read, never inferred."""
        if session_id is not None and not isinstance(session_id, str):
            return []
        reads = [s for s in self.calib["samples"]
                 if s.get("source") == "modeled" and isinstance(s.get("session_id"), str)
                 and (session_id is None or s.get("session_id") == session_id)
                 and _is_int(s.get("hand_no")) and _is_int(s.get("n"))]
        if not reads:
            return []
        dp_by_hand: dict[tuple, dict] = {}   # one replay per (session, hand) — hand_no collides across sessions
        out = []
        for s in reads:
            sid, hand_no, n = s["session_id"], s["hand_no"], s["n"]
            key = (sid, hand_no)
            if key not in dp_by_hand:
                built = self.coach_review_context(sid, hand_no)
                dp_by_hand[key] = ({d["n"]: d for d in built[0].get("decision_points", [])}
                                   if built else {})
            dp = dp_by_hand[key].get(n)
            if dp is None or dp.get("equity") is None:
                continue                     # hand not yet recorded / no exact equity → skip, don't fabricate
            modeled, exact = s["truth"], dp["equity"]
            out.append({
                "session_id": sid, "hand_no": hand_no, "n": n,
                "street": s.get("street"), "gut": s["gut"],
                "modeled": round(modeled, 1), "exact": round(exact, 1),
                "delta": round(exact - modeled, 1),   # negative = exact worse than you were graded against
                "modeled_band": equity.band(modeled), "exact_band": equity.band(exact),
                "ts": s["ts"],
            })
        return out

    # ---------------- persistence: active table ----------------
    def _atomic_write(self, path: Path, obj: dict, durable: bool = False) -> None:
        # Structural: every writer self-ensures its dir, so Game works no matter who constructed
        # it with what paths (covers career/session/reports even in separate dirs).
        path.parent.mkdir(parents=True, exist_ok=True)
        # pid-suffix the temp so two processes sharing one data dir can't interleave-corrupt the
        # SAME tmp before the atomic rename (the multi-writer footgun the data-dir consolidation
        # raised; last-writer-wins on the final file is the worst remaining case, not corruption).
        tmp = path.parent / f"{path.name}.{os.getpid()}.tmp"
        # allow_nan=False: NaN/Infinity are not valid JSON — fail LOUD on a non-finite value rather
        # than silently persisting an unparseable file that a strict reader chokes on (L3 codex). No
        # legitimate game/career/calibration value is non-finite, so this only ever catches corruption.
        data = json.dumps(obj, indent=2, allow_nan=False)
        if durable:
            # money ledger: flush to disk before the rename so a power loss can't
            # lose a committed bankroll change (worth the fsync on infrequent writes).
            with open(tmp, "w") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            tmp.replace(path)
            # also fsync the DIRECTORY so the rename itself is durable (a file fsync alone
            # leaves the new dir entry un-flushed). Best-effort: dir fsync is unsupported on
            # some platforms (e.g. Windows) — a failure there is not worth crashing a save.
            try:
                dfd = os.open(path.parent, os.O_RDONLY)
                try:
                    os.fsync(dfd)
                finally:
                    os.close(dfd)
            except OSError:
                pass
        else:
            tmp.write_text(data)
            tmp.replace(path)

    def _fresh_unseated(self) -> dict:
        return {"seated": False, "players": [], "hero_idx": 0, "hand_no": 0,
                "master_seed": 0, "current": None, "last_result": None,
                "last_table": None, "hero_busted": False, "session": None}

    def _load(self) -> dict:
        if self.path.exists():
            try:
                s = json.loads(self.path.read_text())
                s.setdefault("seated", bool(s.get("players")))
                s.setdefault("hero_busted", False)
                s.setdefault("session", None)
                # a seated table with no session block is a pre-0b / incompatible
                # file — don't trust it (get_up would crash on session["invested"]).
                if s.get("seated") and s.get("session") is None:
                    return self._fresh_unseated()
                return s
            except (json.JSONDecodeError, OSError):
                # truncated/corrupt table → start unseated. The career LEDGER is
                # untouched, but a buy-in already debited for the in-progress session
                # is forfeit (its `invested` lived only in the now-unreadable file).
                pass
        return self._fresh_unseated()

    def _new_stats(self) -> dict:
        return {"hands": 0, "vpip": 0, "pf_raises": 0, "showdowns": 0,
                "net": 0, "wins": 0}

    def _ensure_stats(self) -> None:
        for p in self.s.get("players", []):
            p.setdefault("stats", self._new_stats())

    def _save(self) -> None:
        self._atomic_write(self.path, self.s)

    # ---------------- table construction ----------------
    def _generate_villains(self, rng: random.Random) -> list[dict]:
        """Draw N distinct names, each assigned a RANDOM archetype. Name ⊥ type."""
        names = rng.sample(NAME_POOL, N_VILLAINS)
        return [{"name": nm, "type": rng.choice(ARCH_KEYS), "stack": BUYIN,
                 "stats": self._new_stats()} for nm in names]

    # ---------------- lifecycle: sit / reload / getup ----------------
    def sit_down(self, hero: str = "You", buyin: int = BUYIN) -> dict:
        if self.s.get("seated"):
            return self.view()
        if self.career["bankroll"] < buyin:
            return self.view()  # can't afford a seat
        self.career["bankroll"] -= buyin
        # Wide enough that an offline scan against the visible villain names is not worth running, and
        # narrow enough that master_seed + hand_no * 101 stays an exact JSON number (< 2**53) for an MCP
        # host that parses with doubles.
        master_seed = random.SystemRandom().randrange(10**6, 2**52)
        villains = self._generate_villains(random.Random(master_seed))
        players = [{"name": hero, "type": "hero", "stack": buyin,
                    "stats": self._new_stats()}] + villains
        # A random suffix keeps two sit-downs in the same second from colliding on a report filename
        # (datetime is only second-resolution). It must NOT derive from master_seed: the id reaches
        # every coach leg, and the master seed rebuilds the villains, the decks and the §3f roster.
        sid = f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
        self.s = {
            "seated": True, "players": players, "hero_idx": 0, "hand_no": 0,
            "master_seed": master_seed, "current": None, "last_result": None,
            "last_table": None, "hero_busted": False,
            "session": {"id": sid,
                        "started": datetime.now().isoformat(timespec="seconds"),
                        "buyin": buyin, "reloads": 0, "invested": buyin,
                        "hands": [], "hero_agg": {"raises": 0, "calls": 0}},
        }
        self._bankroll_point("sit")
        # persist the DEBIT (career) before the table (session). A crash between the
        # two writes then loses the buy-in rather than leaving a free un-debited table
        # — money can only be destroyed on a torn write, never created.
        self._save_career()
        self._save()
        return self.view()

    def reload_hero(self) -> dict:
        """Buy a fresh stack from bankroll after busting (a between-hands choice)."""
        if not self.s.get("seated") or not self.s.get("hero_busted"):
            return self.view()
        if self.s["current"] is not None:
            return self.view()  # explicit guard (hero_busted already implies this,
            #                      but don't depend on the coupling — invariant > trust)
        if self.career["bankroll"] < BUYIN:
            return self.view()
        self.career["bankroll"] -= BUYIN
        # ADD a buy-in to the residual, never overwrite — a hero busts at stack < BB,
        # so a leftover chip exists and overwriting it would destroy money (the
        # bankroll loses BUYIN but the stack only gains BUYIN-residual). += conserves.
        self.s["players"][self.s["hero_idx"]]["stack"] += BUYIN
        self.s["session"]["reloads"] += 1
        self.s["session"]["invested"] += BUYIN
        self.s["hero_busted"] = False
        self._bankroll_point("reload")
        self._save_career()   # debit before the stack credit — lose-safe on a torn write
        self._save()
        return self.view()

    def get_up(self) -> dict:
        """Cash out the table stack → bankroll, write a session report, unseat."""
        if not self.s.get("seated"):
            return self.view()
        if self.s["current"] is not None:
            return self.view()  # can't leave mid-hand
        hero_idx = self.s["hero_idx"]
        cashout = self.s["players"][hero_idx]["stack"]
        sess = self.s["session"]
        invested = sess["invested"]
        net = cashout - invested
        hands = self.s["hand_no"]
        bb100 = round(net / BB / hands * 100, 2) if hands else 0.0
        report = self._build_report(cashout, net, bb100)
        self._write_report(report)
        # career updates
        self.career["bankroll"] += cashout
        lt = self.career["lifetime"]
        lt["hands"] += hands
        lt["net"] += net
        lt["sessions"] += 1
        lt["won"] += 1 if net > 0 else 0
        self.career["sessions"].insert(0, {
            "id": report["id"], "started": report["started"],
            "ended": report["ended"], "buyin": report["buyin"],
            "reloads": report["reloads"], "invested": report["invested"],
            "cashout": cashout, "net": net, "hands": hands, "bb100": bb100})
        if len(self.career["sessions"]) > 200:
            self.career["sessions"] = self.career["sessions"][:200]
        self._bankroll_point("getup")
        self.s = self._fresh_unseated()
        # unseat (session) before crediting (career): a crash between writes loses the
        # cashout rather than letting a still-seated table be cashed out twice.
        self._save()
        self._save_career()
        v = self.view()
        v["report"] = report
        return v

    def reset_all(self) -> dict:
        """Wipe the career ledger + the live table back to fresh (in-memory, then saved). This clears
        the MONEY (career) and the table, but deliberately does NOT touch on-disk history.db / session
        reports / calibration.json — the same soft-reset semantics those already had. The HARD slate
        (also clears history, the coach channel, and reports) is the CLI `gutcheck --reset`
        (paths.reset_data). Consequence: after an in-app reset, /api/history can still surface hands
        from pre-reset sessions — orphaned but never corrupt (session ids are unique, no dedup clash)."""
        self.career = self._fresh_career()
        self.s = self._fresh_unseated()
        self._save()
        self._save_career()
        return self.view()

    # ---------------- session report ----------------
    def _compact_hand(self, r: dict) -> dict:
        """A replayable hand record: (seed, start_stacks, log) reconstructs it
        exactly via engine.Hand; assign maps seat→player for labeling."""
        return {"hand_no": r["hand_no"], "seed": r["seed"],
                "start_stacks": r["start_stacks"], "assign": r["assign"],
                "log": r["log"], "pot": r["pot"], "hero_net": r["hero_net"],
                "board": r["board"], "street": r["street"]}

    def _build_report(self, cashout: int, net: int, bb100: float) -> dict:
        sess = self.s["session"]
        hero_idx = self.s["hero_idx"]
        hands = self.s["hand_no"]
        recs = sess["hands"]
        biggest = sorted(recs, key=lambda r: r["pot"], reverse=True)[:5]
        notable = sorted(recs, key=lambda r: abs(r["hero_net"]), reverse=True)[:5]
        villains = [{"name": p["name"], "type": p["type"],
                     "archetype": ARCHETYPES.get(p["type"], "?"),
                     "stats": p["stats"]}
                    for i, p in enumerate(self.s["players"]) if i != hero_idx]
        agg = sess["hero_agg"]
        af = round(agg["raises"] / agg["calls"], 2) if agg["calls"] else float(agg["raises"])
        hs = self.s["players"][hero_idx]["stats"]
        vpip = round(100 * hs["vpip"] / hs["hands"], 1) if hs["hands"] else 0
        return {
            "id": sess["id"], "started": sess["started"],
            "ended": datetime.now().isoformat(timespec="seconds"),
            "buyin": sess["buyin"], "reloads": sess["reloads"],
            "invested": sess["invested"], "cashout": cashout, "net": net,
            "hands": hands, "bb100": bb100,
            "hero": {"vpip": vpip, "af": af, "raises": agg["raises"],
                     "calls": agg["calls"], "showdowns": hs["showdowns"],
                     "wins": hs["wins"]},
            "villains": villains,
            "biggest_pots": [self._compact_hand(r) for r in biggest],
            "notable_hands": [self._compact_hand(r) for r in notable],
        }

    def _write_report(self, report: dict) -> None:
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self._atomic_write(self.reports_dir / f"{report['id']}.json", report)

    # ---------------- seating / hand construction ----------------
    def _assignment(self) -> list[int]:
        # player index in each pokerkit seat (0=SB..n-1=BTN). Adding hand_no gives
        # the CANONICAL rotation: button moves clockwise/left, so the SB player gets
        # the button next hand → a player goes SB→BTN→CO→HJ→UTG→BB→SB. (Verified
        # against the real rule; L1's "SB→BB" review note was backwards — subtracting
        # would reverse the rotation.)
        n = len(self.s["players"])
        h = self.s["hand_no"]
        return [(seat + h) % n for seat in range(n)]

    def _build(self) -> tuple[Hand, list[int], int]:
        assign = self._assignment()
        stacks = [self.s["players"][assign[seat]]["stack"] for seat in range(len(assign))]
        seed = self.s["master_seed"] + self.s["hand_no"] * 101
        log = self.s["current"]["log"] if self.s["current"] else []
        h = Hand(seed=seed, small_blind=SB, big_blind=BB,
                 starting_stacks=stacks, action_log=list(log))
        hero_seat = assign.index(self.s["hero_idx"])
        return h, assign, hero_seat

    def _villain_token(self, h: Hand, assign: list[int], seat: int) -> str:
        ptype = self.s["players"][assign[seat]]["type"]
        rng = random.Random(h.seed * 1000 + len(h.action_log) * 7 + seat)
        lg = h.legal()  # compute once (was queried twice)
        ctx = dict(hole=h.hole(seat), board=h.board(), legal=lg, pot=h.pot(),
                   bb=BB, street_index=h.street_index, rng=rng,
                   facing_bet=lg["call_amount"] > 0, current_level=max(h.bets))
        return decide(ptype, ctx)

    def _advance(self, h: Hand, assign: list[int], hero_seat: int) -> list:
        """Run villains until hero's turn / hand end. Returns animation frames
        (one per villain action) for the client to play back — this is the
        betting progression the player needs to SEE to build reads."""
        frames = []
        safety = 0
        while not h.over and h.actor is not None and h.actor != hero_seat and safety < 80:
            safety += 1
            actor = h.actor
            h.record(self._villain_token(h, assign, actor))
            frames.append(self._frame(h, assign, hero_seat, actor))
        if safety >= 80 and not h.over and h.actor is not None and h.actor != hero_seat:
            import sys
            print(f"[gutcheck] WARNING: _advance hit safety cap; hand may be "
                  f"stranded (seed={h.seed}, log={h.action_log})", file=sys.stderr)
        self.s["current"]["log"] = list(h.action_log)
        return frames

    def _frame(self, h: Hand, assign: list[int], hero_seat: int, actor: int) -> dict:
        """Lightweight table snapshot after a villain action, for client playback."""
        n = len(assign)
        last = h.history[-1] if h.history else None
        label = ""
        if last:
            tok, facing = last["token"], last["facing"]
            if tok == "f":
                label = "folds"
            elif tok == "c":
                label = "checks" if facing == 0 else f"calls {facing}"
            elif tok.startswith("r"):
                to = int(tok[1:])
                label = f"raises to {to}" if facing > 0 else f"bets {to}"
        live = h.live_seats()
        seats = []
        for seat in range(n):
            p = self.s["players"][assign[seat]]
            is_hero = seat == hero_seat
            seats.append({
                "seat": seat, "position": position_name(seat, n),
                "name": p["name"], "type_label": "YOU" if is_hero else "",
                "stack": h.stacks[seat], "bet": h.bets[seat],
                "live": seat in live, "is_hero": is_hero, "is_actor": False,
                "hole": h.hole(seat) if is_hero else None,
            })
        return {"board": h.board(), "pot": h.pot(), "street": h.street_name(),
                "hero_seat": hero_seat, "players_total": n, "seats": seats,
                "just_acted": actor, "label": label}

    def _settle(self, h: Hand, assign: list[int]) -> None:
        net = h.net()
        result_rows = []
        for seat in range(len(assign)):
            self.s["players"][assign[seat]]["stack"] += net[seat]
            if net[seat] != 0:
                result_rows.append({"name": self.s["players"][assign[seat]]["name"],
                                    "amount": net[seat]})
        # Showdown participants = seats that never folded (pokerkit mucks losers'
        # status + cards at settle, so post-hand live_seats() is just the winner).
        n = len(assign)
        holes = self.s["current"].get("holes") or [h.hole(s) for s in range(n)]
        folded = {e["seat"] for e in h.history if e["token"] == "f"}
        showdown_seats = [s for s in range(n) if s not in folded]
        is_showdown = len(showdown_seats) > 1
        showdown = []
        if is_showdown:
            for seat in showdown_seats:
                showdown.append({"name": self.s["players"][assign[seat]]["name"],
                                 "hole": holes[seat]})
        self.s["last_result"] = {"board": h.board(), "net": result_rows,
                                 "showdown": showdown}
        self._update_stats(h, assign, net, showdown_seats, is_showdown)
        self.s["last_table"] = self._snapshot_final(h, assign, net, holes,
                                                    showdown_seats, is_showdown)
        self._accumulate_session(h, assign, net)
        self.s["current"] = None
        # hero bust check AFTER settle — villains rebuy at next deal, the hero must
        # choose reload-from-bankroll or get up (handled between hands).
        self.s["hero_busted"] = self.s["players"][self.s["hero_idx"]]["stack"] < BB

    def _accumulate_session(self, h: Hand, assign: list[int], net: list[int]) -> None:
        """Record this hand into the session accumulator: a replayable record +
        the hero's aggression tally. (Runs before current is cleared.)"""
        sess = self.s.get("session")
        if sess is None:
            return
        hero_seat = assign.index(self.s["hero_idx"])
        for e in h.history:
            if e["seat"] == hero_seat:
                if e["token"].startswith("r"):
                    sess["hero_agg"]["raises"] += 1
                elif e["token"] == "c" and e["facing"] > 0:
                    sess["hero_agg"]["calls"] += 1
        record = {
            "hand_no": self.s["hand_no"], "seed": h.seed,
            "start_stacks": list(h.starting_stacks), "assign": list(assign),
            "log": list(h.action_log), "pot": h.total_pot(),
            "hero_net": net[hero_seat], "board": h.board(),
            "street": h.street_name()}
        sess["hands"].append(record)
        # Incremental per-hand persistence to the CANONICAL history store (history.py — NOT
        # the coach channel) so the seam's /api/history, and the MCP adapter next, can pull a
        # complete cross-session stream. Best-effort: a history write must NEVER take down the
        # game loop, because the hand already lives in sess["hands"] + the get-up report (the
        # money truth) — this store is the secondary queryable projection. Failures are LOUD
        # (stderr), never silently swallowed (invisible_infrastructure_failure).
        sid = sess.get("id")
        if sid:
            try:
                history.record_hand(sid, record, path=self.history_db)
            except Exception as e:                               # noqa: BLE001 — non-fatal by design
                import sys                                       # local, matching this module's pattern
                print(f"[history] persist failed for hand {record['hand_no']}: {e}",
                      file=sys.stderr)

    def _snapshot_final(self, h: Hand, assign: list[int], net: list[int],
                        holes: list, showdown_seats: list[int],
                        is_showdown: bool) -> dict:
        """Freeze the finished hand so the table keeps rendering between hands.

        (Stacks are already settled here; we can't rebuild via replay without
        double-applying, so we snapshot the final view directly. Holes come from
        the deal-time capture since pokerkit has mucked folded/losing hands.)
        """
        n = len(assign)
        hero_seat = assign.index(self.s["hero_idx"])
        seats = []
        for seat in range(n):
            pidx = assign[seat]
            p = self.s["players"][pidx]
            is_hero = seat == hero_seat
            live = seat in showdown_seats          # didn't fold
            reveal = is_hero or (is_showdown and seat in showdown_seats)
            seats.append({
                "seat": seat, "position": position_name(seat, n),
                "name": p["name"], "type": p["type"],
                "archetype": ARCHETYPES.get(p["type"], "hero"),
                "type_label": "YOU" if is_hero else "",
                "stats": p["stats"], "stack": p["stack"], "bet": 0,
                "live": live, "is_hero": is_hero, "is_actor": False,
                "hole": holes[seat] if reveal else None,
                "won": net[seat] > 0,
            })
        return {"hero_seat": hero_seat, "players_total": n,
                "board": h.board(), "pot": 0, "seats": seats}

    def _update_stats(self, h: Hand, assign: list[int], net: list[int],
                      showdown_seats: list[int], is_showdown: bool) -> None:
        """Accumulate observed per-player stats from the hand's action history."""
        voluntary, raised = set(), {}
        for e in h.history:
            if e["street"] != 0:
                continue
            seat = e["seat"]
            if e["token"].startswith("r"):
                raised[seat] = raised.get(seat, 0) + 1
                voluntary.add(seat)
            elif e["token"] == "c" and e["facing"] > 0:  # a real call, not a free BB check
                voluntary.add(seat)
        for seat in range(len(assign)):
            st = self.s["players"][assign[seat]]["stats"]
            st["hands"] += 1
            if seat in voluntary:
                st["vpip"] += 1
            st["pf_raises"] += raised.get(seat, 0)
            if is_showdown and seat in showdown_seats:  # actually reached showdown
                st["showdowns"] += 1
            st["net"] += net[seat]
            if net[seat] > 0:
                st["wins"] += 1

    def _rebuy_villains(self) -> None:
        # rebuy any VILLAIN who can't post a full big blind (incl. busted) — avoids a
        # seat lingering perpetually all-in for sub-blind change. The hero is NEVER
        # auto-rebought here: a busted hero must choose reload-from-bankroll or get up.
        for i, p in enumerate(self.s["players"]):
            if i != self.s["hero_idx"] and p["stack"] < BB:
                p["stack"] = BUYIN

    # ---------------- commands ----------------
    def new_hand(self) -> dict:
        if not self.s.get("seated"):
            return self.view()
        if self.s["current"] is not None:
            return self.view()
        if self.s.get("hero_busted"):
            return self.view()  # must reload or get up first
        self._rebuy_villains()
        self.s["hand_no"] += 1
        self.s["current"] = {"log": []}
        self.s["last_result"] = None
        h, assign, hero_seat = self._build()
        # capture every seat's hole cards NOW — pokerkit mucks folded/losing hands
        # at settle, so this is the only point all holes are intact (for showdown
        # reveal + showing the hero their own folded cards). Deterministic per seed.
        self.s["current"]["holes"] = [h.hole(s) for s in range(len(assign))]
        frames = self._advance(h, assign, hero_seat)
        if h.over:
            self._settle(h, assign)
        self._save()
        v = self.view()
        v["frames"] = frames
        return v

    def act(self, action: str, amount: int | None = None) -> dict:
        if not self.s.get("seated") or self.s["current"] is None:
            return self.view()
        h, assign, hero_seat = self._build()
        if h.over or h.actor != hero_seat:
            return self.view()
        if action == "fold":
            if not h.legal()["fold"]:   # no open-folds (only when facing a bet)
                return self.view()
            tok = "f"
        elif action in ("call", "check"):
            # honor "check" only when nothing is owed — a stale "check" must not
            # silently become a paid call if state advanced to facing a bet
            if action == "check" and h.legal()["call_amount"] > 0:
                return self.view()
            tok = "c"
        elif action == "raise":
            if amount is None:
                return self.view()
            lg = h.legal()
            if not lg["raise"]:
                return self.view()
            try:
                amt = int(amount)
            except (TypeError, ValueError):
                return self.view()
            lo, hi = lg["min_raise_to"], lg["max_raise_to"]
            # clamp a stale/out-of-range size into the legal band; guard each bound
            # independently (either can be None without crashing the clamp)
            if lo is not None:
                amt = max(lo, amt)
            if hi is not None:
                amt = min(hi, amt)
            tok = f"r{amt}"
        else:
            return self.view()
        h.record(tok)
        self.s["current"]["log"] = list(h.action_log)
        frames = self._advance(h, assign, hero_seat)
        if h.over:
            self._settle(h, assign)
        self._save()
        v = self.view()
        v["frames"] = frames
        return v

    # ---------------- views ----------------
    def career_view(self) -> dict:
        lt = self.career["lifetime"]
        bb100 = round(lt["net"] / BB / lt["hands"] * 100, 2) if lt["hands"] else 0.0
        winrate = int(round(100 * lt["won"] / lt["sessions"])) if lt["sessions"] else 0
        return {"bankroll": self.career["bankroll"],
                "lifetime": {**lt, "bb100": bb100, "winrate": winrate},
                "history": self.career["bankroll_history"][-120:],
                "sessions": self.career["sessions"][:20]}

    def _session_running(self) -> dict | None:
        sess = self.s.get("session")
        if not sess:
            return None
        hands = self.s["hand_no"]
        stack = self.s["players"][self.s["hero_idx"]]["stack"]
        net = stack - sess["invested"]   # unrealized session P/L
        bb100 = round(net / BB / hands * 100, 2) if hands else 0.0
        return {"id": sess["id"], "hands": hands, "net": net, "bb100": bb100,
                "invested": sess["invested"], "reloads": sess["reloads"],
                "stack": stack}

    def view(self) -> dict:
        if not self.s.get("seated"):
            return {"seated": False, "in_hand": False, "over": True,
                    "blinds": [SB, BB], "buyin": BUYIN,
                    "bankroll": self.career["bankroll"],
                    "can_sit": self.career["bankroll"] >= BUYIN,
                    "career": self.career_view()}

        n = len(self.s["players"])
        base = {"seated": True, "hand_no": self.s["hand_no"], "blinds": [SB, BB],
                "buyin": BUYIN, "players_total": n,
                "bankroll": self.career["bankroll"],
                "session": self._session_running()}
        if self.s["current"] is None:
            base.update({"in_hand": False, "over": True,
                         "last_result": self.s.get("last_result"),
                         "table": self.s.get("last_table"),
                         "hero_busted": self.s.get("hero_busted", False),
                         "can_reload": self.career["bankroll"] >= BUYIN,
                         "hero_stack": self.s["players"][self.s["hero_idx"]]["stack"]})
            return base

        h, assign, hero_seat = self._build()
        seats = []
        for seat in range(n):
            pidx = assign[seat]
            p = self.s["players"][pidx]
            invested = h.starting_stacks[seat] - h.stacks[seat]
            is_hero = seat == hero_seat
            live = seat in h.live_seats()
            seats.append({
                "seat": seat,
                "position": position_name(seat, n),
                "name": p["name"],
                "type": p["type"],                       # key (hidden; for modal reveal)
                "archetype": ARCHETYPES.get(p["type"], "hero"),  # revealed on demand
                "type_label": "YOU" if is_hero else "",  # NO always-on archetype label
                "stats": p["stats"],
                "stack": p["stack"] - invested,
                "bet": h.bets[seat],
                "live": live,
                "is_hero": is_hero,
                "is_actor": (h.actor == seat) and not h.over,
                "hole": h.hole(seat) if is_hero else None,
            })
        hero_hole = h.hole(hero_seat)
        board = h.board()
        to_act = "hero" if (h.actor == hero_seat and not h.over) else (
            None if h.over else "villain")
        lg = h.legal()
        options = None
        pot_odds = None
        strength = None
        if to_act == "hero":
            options = {
                "fold": lg["fold"],
                "check_or_call": lg["check_or_call"],
                "call_amount": lg["call_amount"],
                "is_check": lg["call_amount"] == 0,
                "can_raise": lg["raise"],
                "min_raise": lg["min_raise_to"],
                "max_raise": lg["max_raise_to"],
            }
            if lg["call_amount"] > 0:
                pot = h.pot()
                call = lg["call_amount"]
                pot_odds = {"call": call, "pot": pot,
                            "breakeven_pct": round(100 * call / (pot + call), 1)}
            strength = (f"preflop: {preflop_bucket(hero_hole)}" if not board
                        else f"made hand: {postflop_strength(hero_hole, board)}")
        base.update({
            "in_hand": True, "over": h.over,
            "hero_seat": hero_seat,
            "hero_position": position_name(hero_seat, n),
            "board": board, "pot": h.pot(), "street": h.street_name(),
            "seats": seats,
            "hero": {"hole": hero_hole, "strength_hint": strength},
            "to_act": to_act,
            "options": options,
            "pot_odds": pot_odds,
            "log": list(h.action_log),
            "hero_stack": self.s["players"][self.s["hero_idx"]]["stack"],
        })
        if h.over:
            base["last_result"] = self.s.get("last_result")
        return base

    # ---------------- coach context (Slice 1) ----------------
    # These freeze a spot into a self-contained dict the coach worker reasons over.
    # The worker never touches game state — only the frozen snapshot — so it can run
    # in a separate process (a warm CC session now, a headless daemon next slice).

    def _readable_history(self, h: Hand, assign: list[int], n: int, name_for) -> list[str]:
        """The hand's action so far as English lines the coach can read directly,
        e.g. 'preflop: Gus (UTG) raises to 6'. name_for maps a player index to a label."""
        lines = []
        for e in h.history:
            who = f"{name_for(assign[e['seat']])} ({position_name(e['seat'], n)})"
            lines.append(f"{STREETS.get(e['street'], '?')}: {who} "
                         f"{_action_phrase(e['token'], e['facing'])}")
        return lines

    def _villain_read(self, name: str, position: str, stats: dict, ptype: str,
                      stack: int | None = None, in_hand: bool | None = None) -> dict:
        """One villain's OBSERVED profile + the hidden TRUTH (the `_truth` field).
        Per Phill's pedagogy call: the coach coaches FROM the observed read and uses
        `_truth` ONLY to validate/correct it — never to shortcut the read."""
        st = stats or {}
        vpip = round(100 * st.get("vpip", 0) / st["hands"], 1) if st.get("hands") else None
        read = {"name": name, "position": position,
                "observed": {"hands": st.get("hands", 0), "vpip_pct": vpip,
                             "pf_raises": st.get("pf_raises", 0),
                             "showdowns": st.get("showdowns", 0), "net": st.get("net", 0)},
                "_truth": {"type": ptype, "label": ARCHETYPES.get(ptype, "?"),
                           "use": "validation only — coach from the observed read"}}
        if stack is not None:
            read["stack"] = stack
        if in_hand is not None:
            read["in_hand"] = in_hand
        return read

    def coach_decision_context(self) -> dict | None:
        """Freeze the live decision in front of the hero. Returns None unless it is
        genuinely the hero's turn in a live hand (the coach has nothing to weigh else)."""
        if not self.s.get("seated") or self.s.get("current") is None:
            return None
        h, assign, hero_seat = self._build()
        if h.over or h.actor != hero_seat:
            return None
        n = len(self.s["players"])
        hero_idx = self.s["hero_idx"]
        name_for = lambda pidx: "you" if pidx == hero_idx else self.s["players"][pidx]["name"]
        hero_hole = h.hole(hero_seat)
        board = h.board()
        lg = h.legal()
        to_call = lg["call_amount"]
        pot = h.pot()
        pot_odds = None
        if to_call > 0:
            pot_odds = {"call": to_call, "pot": pot,
                        "breakeven_pct": round(100 * to_call / (pot + to_call), 1)}
        strength = (f"preflop {preflop_bucket(hero_hole)}" if not board
                    else f"made hand: {postflop_strength(hero_hole, board)}")
        villains = []
        reads = {}                                   # seat → read, reused by the modeled block (no recompute)
        for seat in range(n):
            if seat == hero_seat:
                continue
            p = self.s["players"][assign[seat]]
            reads[seat] = self._villain_read(
                p["name"], position_name(seat, n), p["stats"], p["type"],
                stack=h.stacks[seat], in_hand=seat in h.live_seats())
            villains.append(reads[seat])
        # Modeled-equity SPEC (LIVE path): we bake the target opponent's modeled RANGE here (fast — just
        # list-building) but DEFER the Monte-Carlo equity to the coach worker (coach_watch._ensure_modeled_
        # equity). The /api/coach/ask handler is single-threaded and meant to return the poll id instantly;
        # a ~1.5s MC here would stall every other request. The range is modeled from the OBSERVED read
        # (VPIP→width, position, raised-this-hand) — never the hidden type, so no truth-leak. HEADS-UP vs ONE
        # opponent (the aggressor, else the tightest/clearest-threat live range): multiway-vs-wide-ranges is
        # both pathologically slow AND a mushy number; pot_way + live_count let the coach discount for the field.
        modeled_spec = None
        live_villains = [s for s in h.live_seats() if s != hero_seat]
        if live_villains:
            cand = []
            for s in live_villains:
                obs = reads[s]["observed"]
                vp = obs.get("vpip_pct")
                raised = any(e["seat"] == s and e["token"].startswith("r") for e in h.history)
                pos = position_name(s, n)
                cand.append({"seat": s, "name": name_for(assign[s]), "position": pos, "vpip": vp,
                             "raised": raised, "hands": obs.get("hands", 0),
                             "range_pct": round(ranges.effective_width(vp, pos, raised)),
                             "range": ranges.villain_range(vp, pos, raised)})
            target = _select_target(cand, h.history, live_villains)
            modeled_spec = {
                "hero_hole": hero_hole, "board": board, "range": target["range"], "samples": 2500,
                "vs": {k: target[k] for k in ("name", "position", "vpip", "hands", "raised", "range_pct")},
                "live_count": len(live_villains),
                "heads_up": len(live_villains) == 1,
                "pot_way": len(live_villains) + 1,   # +1 for the hero (a 2-villain pot is 3-way)
            }
        # The live decision's ordinal among the hero's POSTFLOP decisions this hand — counted the
        # SAME way coach_review_context numbers decision_points (1-based, hero postflop, in action
        # order), so a live read can later JOIN to that hand's EXACT decision (the modeled−exact
        # reconciliation, P3a). None preflop: review tracks only postflop spots, so a preflop live
        # read has no exact decision to reconcile against (honest null, never inferred). The current
        # decision isn't in h.history yet (it's the hero's pending turn), so it's prior + 1.
        decision_n = None
        if h.street_index >= 1:
            prior = sum(1 for e in h.history if e["seat"] == hero_seat and e.get("street", 0) >= 1)
            decision_n = prior + 1
        return {
            "kind": "decision",
            "modeled_spec": modeled_spec,            # worker computes modeled_equity from this (off the ask path)
            # the FULL join key, FROZEN here so all three parts come from ONE source: a delayed
            # fire-and-forget live commit must not mix THIS hand's (hand_no, n) with a LATER session's
            # id read from current state — the cross-session wrong-join the apparatus reproduced (HIGH).
            "session_id": (self.s.get("session") or {}).get("id"),
            "hand_no": self.s["hand_no"], "n": decision_n,
            "street": h.street_name(), "blinds": [SB, BB],
            "hero": {"hole": hero_hole, "position": position_name(hero_seat, n),
                     "stack": h.stacks[hero_seat], "strength": strength},
            "board": board, "pot": pot, "to_call": to_call, "pot_odds": pot_odds,
            "legal": {"can_fold": lg["fold"], "can_check": to_call == 0,
                      "can_raise": lg["raise"], "min_raise_to": lg["min_raise_to"],
                      "max_raise_to": lg["max_raise_to"]},
            "action_this_hand": self._readable_history(h, assign, n, name_for),
            "villains": villains,
        }

    def _load_report(self, session_id) -> dict | None:
        """Read a session report by id, REJECTING any id that could escape the reports
        dir (the id arrives from the client). Strict charset + a resolved-path recheck."""
        if not isinstance(session_id, str) or not re.fullmatch(r"[0-9A-Za-z_-]+", session_id):
            return None
        path = (self.reports_dir / f"{session_id}.json").resolve()
        if path.parent != self.reports_dir.resolve():   # defense in depth vs traversal
            return None
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return None

    def _review_frame(self, h: Hand, assign: list[int], n: int, hero_seat: int,
                      name_for, holes: list) -> dict:
        """A playback frame for a reconstructed (review) hand — same shape the live 0c
        frames use, so the browser renders it with the existing path. Hero's hole is
        revealed throughout review; villain holes stay hidden until showdown (set by
        the caller on the final frame). Holes come from a capture taken BEFORE any
        replay action — pokerkit mucks folded/losing hands at hand-end, so reading
        h.hole() on the final frame would blank the hero's own cards (and most villains')."""
        last = h.history[-1] if h.history else None
        label = _action_phrase(last["token"], last["facing"]) if last else ""
        live = h.live_seats()
        seats = []
        for seat in range(n):
            is_hero = seat == hero_seat
            seats.append({
                "seat": seat, "position": position_name(seat, n),
                "name": "You" if is_hero else name_for(assign[seat]),
                "type_label": "YOU" if is_hero else "",
                "stack": h.stacks[seat], "bet": h.bets[seat],
                "live": seat in live, "is_hero": is_hero, "is_actor": False,
                "hole": holes[seat] if is_hero else None,
            })
        return {"board": h.board(), "pot": h.pot(), "street": h.street_name(),
                "hero_seat": hero_seat, "players_total": n, "seats": seats,
                "just_acted": last["seat"] if last else None, "label": label}

    def coach_review_context(self, session_id, hand_no):
        """Reconstruct a stored hand from (seed, log) for post-hand review. Returns
        (context, frames): the coach narrates from `context`; the browser steps `frames`
        as a visual replay (one reconstruction feeds both). None if not found.

        Source order: the get-up report's record (it carries villain metadata) when present,
        else the canonical history.db (EVERY played hand, written on settle). The history.db
        fallback DELIBERATELY widens the review surface from the report's top-5 to any stored
        hand — and makes review work MID-SESSION (before get-up) — which is what the P3a
        modeled−exact reconciliation needs. History-sourced hands have no villain names
        (villains=[] → labels degrade to P1/P2); the EXACT equity is cards-only, unaffected."""
        if not _is_int(hand_no):         # _is_int rejects bool (True/False ⊄ a real hand_no), unlike isinstance
            return None
        report = self._load_report(session_id)
        villains = report.get("villains", []) if report else []
        rec = None
        if report is not None:           # prefer the report's record (it travels with villain metadata)
            pool = {r["hand_no"]: r for r in
                    (report.get("biggest_pots", []) + report.get("notable_hands", []))}
            rec = pool.get(hand_no)
        if rec is None:                  # not a report top-5 hand (or mid-session, no report yet) —
            rec = history.get_hand(session_id, hand_no, path=self.history_db)  # the canonical store has EVERY hand
        if rec is None:
            return None
        # the report is server-authored, but guard a corrupt/hand-edited file: validate the
        # replay inputs up front (bounded log, well-formed tokens, a real seat permutation with
        # the hero at index 0) so a bad record returns a clean None→404 instead of leaking an
        # exception through the endpoint's generic 400 handler — or building a giant frame list.
        log = rec.get("log"); seed = rec.get("seed")
        starts = rec.get("start_stacks"); assign_raw = rec.get("assign")
        if not (isinstance(seed, int) and isinstance(starts, list) and starts
                and isinstance(assign_raw, list) and len(assign_raw) == len(starts)
                and sorted(assign_raw) == list(range(len(starts))) and 0 in assign_raw
                and isinstance(log, list) and len(log) <= 100
                and all(isinstance(t, str) and re.fullmatch(r"f|c|r\d+", t) for t in log)):
            return None
        # `villains` resolved above (report metadata, or [] for a history.db-sourced hand — names then
        # degrade to "P1"/"P2" via name_for; the exact equity the reconciliation needs is cards-only).
        # player index 0 == hero (sit_down always seats the hero at players[0]); the
        # villains follow in original player order, so index k maps to villains[k-1].
        def name_for(pidx):
            return "you" if pidx == 0 else (
                villains[pidx - 1]["name"] if 0 < pidx <= len(villains) else f"P{pidx}")
        h = Hand(seed=seed, small_blind=SB, big_blind=BB,
                 starting_stacks=list(starts), action_log=[])
        assign = list(assign_raw)
        n = len(assign)
        hero_seat = assign.index(0)
        holes = [h.hole(s) for s in range(n)]   # capture BEFORE replay — pre-muck, intact
        frames = []
        decision_points = []                 # the prediction-error drill: hero's postflop spots
        try:
            for tok in log:                  # step + snapshot — replays the 0c progression
                # Capture a hero decision BEFORE applying it. Postflop only — felt-sense board
                # reading lives postflop; preflop is a position/odds call. Equity is EXACT, computed
                # vs the live villains' ACTUAL hands (cards are revealed in review) — the unimpeachable
                # ground truth the gut is checked against (SCOPE decision 7, review-first drill).
                if not h.over and h.actor == hero_seat and h.street_index >= 1:
                    to_call = h.call_amount()
                    dboard = h.board()
                    live_villains = [s for s in h.live_seats() if s != hero_seat]
                    eq = (equity.hero_vs_known(holes[hero_seat],
                                               [holes[s] for s in live_villains], dboard)
                          if live_villains else None) or {}
                    decision_points.append({
                        "n": len(decision_points) + 1,
                        "frame_index": len(frames),   # browser pauses BEFORE this frame plays
                        "street": h.street_name(),
                        "board": dboard,
                        "hero_hole": holes[hero_seat],
                        "pot": h.pot(),
                        "to_call": to_call,
                        "can_raise": h.legal()["raise"],   # facing a bet ≠ can raise (all-in/capped) → gate the lean
                        "action": _action_phrase(tok, to_call),   # what the hero ACTUALLY did
                        "equity": eq.get("hero_pct"),             # exact hero equity, or None
                        "band": equity.band(eq.get("hero_pct")),  # felt-sense label ('behind' / 'crushing')
                        "exact": eq.get("exact", False),
                        "cards_to_come": eq.get("cards_to_come"),
                        "vs": len(live_villains),
                    })
                h.record(tok)
                frames.append(self._review_frame(h, assign, n, hero_seat, name_for, holes))
        except ValueError:
            return None                      # a corrupt log produced an illegal action sequence
        net = h.net()
        folded = {e["seat"] for e in h.history if e["token"] == "f"}
        showdown_seats = [s for s in range(n) if s not in folded]
        is_showdown = len(showdown_seats) > 1
        if frames and is_showdown:           # reveal showdown holes on the final frame
            for s in frames[-1]["seats"]:
                if s["seat"] in showdown_seats:
                    s["hole"] = holes[s["seat"]]
        review_villains = []
        for j, v in enumerate(villains):
            pidx = j + 1
            if pidx not in assign:
                continue
            review_villains.append(self._villain_read(
                v["name"], position_name(assign.index(pidx), n),
                v.get("stats", {}), v.get("type", "")))
        context = {
            "kind": "review", "hand_no": hand_no, "session_id": session_id,
            "blinds": [SB, BB],
            "hero": {"hole": holes[hero_seat], "position": position_name(hero_seat, n),
                     "net": net[hero_seat]},
            "final_board": h.board(), "pot": rec.get("pot"),
            "result": {
                "hero_net": net[hero_seat],
                "winners": [{"name": name_for(assign[s]), "amount": net[s]}
                            for s in range(n) if net[s] > 0],
                "showdown": ([{"name": name_for(assign[s]), "hole": holes[s]}
                              for s in showdown_seats] if is_showdown else []),
            },
            "action_by_street": self._readable_history(h, assign, n, name_for),
            "villains": review_villains,
            "decision_points": decision_points,   # exact equity at each hero postflop spot (the drill)
            "note": "villain stats are session-final (hindsight review)",
        }
        return context, frames

    def sizing_tells(self, session_id=None) -> dict:
        """Per-villain OBSERVED betting-behavior → shown-strength tell, from showdown-revealed hands
        (P3(c)). Replays every stored hand in the session and, for each villain BET/RAISE the operator
        could ALSO have seen down (the villain reached showdown → her holes were revealed), records her
        made-hand strength AT THAT STREET. SHOWDOWN-OBSERVED ONLY: a villain who folded out never
        contributes — the coach gets no privileged sim-truth, so the read stays earnable (§3f). Postflop
        only — preflop raises are range/position plays in a different vocabulary, not a made-hand tell.

        What the data SHOWS (measured 2026-06-23, not assumed): the v0 archetypes size ~constantly
        (≈⅔ pot), so bet MAGNITUDE barely separates strength — the real, strongly-modelable tell is
        AGGRESSION (when a villain bets/raises postflop she is overwhelmingly strong+). The output
        exposes BOTH dimensions (aggregate aggression + a per-size-bucket breakdown) as RAW FACTS; the
        doctrine teaches the coach to read aggression as the signal and call flat sizing flat. Pure data
        out — no coaching prose, no archetype name (the coach speaks in behavior, never a type).

        Keyed by the STABLE villain identity (player index — the seat rotates with the button, the
        villain keeps her name). ``session_id`` scopes the tell to one session (villains are per-session,
        regenerated each sit-down, so a cross-session tell is meaningless); ``None`` defaults to the live
        session. Names resolve via the session report, then the live table, then ``P{idx}`` — the same
        identity the operator watches move around the table."""
        sid = (session_id if isinstance(session_id, str) and session_id
               else (self.s.get("session") or {}).get("id"))
        if not sid:
            return {"session_id": None, "villains": [], "note": _SIZING_NOTE}
        # Reuse P3(d)'s session filter — pull only this session's stored hands (every played hand lives
        # in history.db, written on settle). PAGE THE WHOLE SESSION through the watermark: a single
        # capped read would silently truncate the OLDEST limit-N rows (id ASC), making the tell reflect
        # early-session behavior and drop the most recent, most-informative showdowns
        # (invisible_infrastructure_failure). A session is bounded, so this terminates quickly.
        hands = []
        since = 0
        while True:
            page = history.get_history(since, path=self.history_db, session_id=sid)
            if not page["hands"]:
                break
            hands.extend(page["hands"])
            since = page["watermark"]
        report = self._load_report(sid)
        report_villains = report.get("villains", []) if report else []
        live_players = self.s.get("players") or []
        live_sid = (self.s.get("session") or {}).get("id")

        def name_for(pidx):
            if 0 < pidx <= len(report_villains):
                return report_villains[pidx - 1].get("name") or f"P{pidx}"
            if sid == live_sid and 0 < pidx < len(live_players):   # live session, no report yet; 0<
                return live_players[pidx].get("name") or f"P{pidx}"  # excludes hero, matching the report path
            return f"P{pidx}"

        # Accumulate per player index: showdown count (the evidence base) + the (strength, size-frac)
        # of each postflop bet/raise made on a hand SHE took to showdown.
        agg: dict[int, dict] = {}
        for rec in hands:
            log = rec.get("log"); seed = rec.get("seed")
            starts = rec.get("start_stacks"); assign_raw = rec.get("assign")
            # Same replay-input validation as coach_review_context — a corrupt/hand-edited row is
            # skipped, never crashes the tell (one bad hand must not blank the whole session's read).
            if not (isinstance(seed, int) and isinstance(starts, list) and starts
                    and isinstance(assign_raw, list) and len(assign_raw) == len(starts)
                    and sorted(assign_raw) == list(range(len(starts))) and 0 in assign_raw
                    and isinstance(log, list) and len(log) <= 100
                    and all(isinstance(t, str) and re.fullmatch(r"f|c|r\d+", t) for t in log)):
                continue
            n = len(assign_raw); assign = list(assign_raw)
            h = Hand(seed=seed, small_blind=SB, big_blind=BB, starting_stacks=list(starts), action_log=[])
            holes = [h.hole(s) for s in range(n)]   # pre-muck holes, captured before replay
            actions = []                            # (pidx, strength, frac) — villain postflop bet/raises
            try:
                for tok in log:
                    if h.over:                       # a token AFTER the hand ended ⇒ corrupt/edited row;
                        raise ValueError("log continues past hand end")   # bail to the skip below [codex L3]
                    seat = h.actor                   # guard None: between-street / terminal states have no actor
                    if seat is not None and assign[seat] != 0 and tok.startswith("r") and h.street_index >= 1:
                        pot = h.pot()
                        raise_to = int(tok[1:])
                        # Coarse size proxy: raise-to over the pot BEFORE the action (≈ bet/pot for a
                        # lead; an over-read for a re-raise — but the buckets are coarse and size is
                        # not the live signal here, aggression is). Refine if a size-varying archetype
                        # ever ships (this method then catches it automatically).
                        frac = raise_to / pot if pot > 0 else 0.0
                        actions.append((assign[seat], postflop_strength(holes[seat], h.board()), frac))
                    h.record(tok)
            except (ValueError, TypeError, IndexError):
                continue                            # corrupt / illegal / malformed log — skip the hand
            if not h.over:                          # a syntactically-valid but INCOMPLETE log never reached
                continue                            # a terminal state — its "showdown" would be fabricated [codex L3]
            folded = {e["seat"] for e in h.history if e["token"] == "f"}
            showdown_seats = [s for s in range(n) if s not in folded]
            if len(showdown_seats) <= 1:
                continue                            # no showdown → no revealed holes → nothing observable
            showdown_pidx = {assign[s] for s in showdown_seats if assign[s] != 0}   # exclude hero
            for pidx in showdown_pidx:
                agg.setdefault(pidx, {"showdowns": 0, "actions": []})["showdowns"] += 1
            for pidx, strg, frac in actions:
                if pidx in showdown_pidx:           # count the action ONLY if she reached showdown (Opt A)
                    agg[pidx]["actions"].append((strg, frac))

        villains = []
        for pidx in sorted(agg):
            acts = agg[pidx]["actions"]
            strength_counts = collections.Counter(s for s, _ in acts)
            bucketed: dict[str, list] = {label: [] for label, _, _ in _SIZE_BUCKETS}
            for strg, frac in acts:
                bucketed[_size_bucket(frac)].append(strg)
            by_size = [{"size": label, "n": len(bucketed[label]),
                        "strength_counts": dict(collections.Counter(bucketed[label]))}
                       for label, _, _ in _SIZE_BUCKETS]
            villains.append({
                "name": name_for(pidx),
                "player_index": pidx,
                "showdowns": agg[pidx]["showdowns"],   # hands of hers seen down — the confidence base
                "aggressive_postflop": {
                    "n": len(acts),
                    "strong_or_better": sum(strength_counts[k] for k in _STRONG_PLUS),
                    "strength_counts": dict(strength_counts),
                },
                "by_size": by_size,
            })
        return {"session_id": sid, "villains": villains, "note": _SIZING_NOTE}


if __name__ == "__main__":
    import tempfile
    d = Path(tempfile.mkdtemp())
    g = Game(d / "session.json", d / "career.json", d / "sessions")
    print("fresh bankroll:", g.career["bankroll"], "| seated:", g.view()["seated"])
    g.sit_down(hero="Phill")
    print("after sit — bankroll:", g.career["bankroll"], "| seated:", g.view()["seated"])
    print("villains:", [(p["name"], p["type"]) for p in g.s["players"][1:]])
    for _ in range(3):
        v = g.new_hand()
        while v.get("in_hand") and v.get("to_act") == "hero":
            v = g.act("call")
    r = g.get_up()
    rep = r["report"]
    print(f"report — hands:{rep['hands']} net:{rep['net']} bb/100:{rep['bb100']} "
          f"biggest pot:{rep['biggest_pots'][0]['pot'] if rep['biggest_pots'] else 0}")
    print("after getup — bankroll:", g.career["bankroll"], "| sessions:",
          g.career["lifetime"]["sessions"])
