"""gutcheck coach worker — the read side of the coach channel.

The worker reasons over the FROZEN spot, never live game state, so it runs as a separate
process — a warm CC session (Slice 1, flavor a) or the headless daemon (Slice 2, flavor b).
It imports `coach` (the channel) and `equity` (→ pokerkit): the LIVE modeled-equity Monte-Carlo
+ the felt band run HERE, not on the single-threaded /api/coach/ask path (game.py bakes only the
fast range spec; the worker turns it into the equity — see `_ensure_modeled_equity`). It still
has NO dependency on game/engine — it works from the frozen spec, so the equity stack is the
only added runtime requirement vs the original coach-only worker.

    python -m gutcheck.coach_watch next         # print the next pending request, laid out to read
    python -m gutcheck.coach_watch stub         # auto-answer pending requests with a grounded
                                                 #   template (no LLM) — the end-to-end test harness
                                                 #   AND a usable fallback coach when no session is up
    python -m gutcheck.coach_watch stub --once  # single pass (answer what's pending, then exit)

THE WARM-SESSION LOOP (Slice 1, flavor a): a CC session on the subscription runs
`python -m gutcheck.coach_watch next`, reads the spot, coaches it, then
`python -m gutcheck.coach respond --id N --body "…"`, and repeats. The session stays warm, so it is
snappy — no `claude -p` cold start. Coaching is "me"; the villains stay coded.

Per Phill's pedagogy call: coach FROM the observed read + behavior; use `_truth` ONLY to
calibrate how much to trust that read and to avoid teaching a WRONG one — NEVER to name the
hidden type (the player earns the read by watching). The `next` layout and the `stub` model
exactly that: the truth surfaces as a confidence flag, never as an answer key.
"""

from __future__ import annotations

import argparse
import sys
import time

from . import coach
from . import equity   # the worker computes the live modeled-equity MC + the felt band; needs the equity stack
from . import paths
from . import settings   # facts-floor stat-tailoring prefs (slice b): equity format / pot-odds / observed-read

SUIT = {"s": "♠", "h": "♥", "d": "♦", "c": "♣"}


def _cards(cards) -> str:
    out = []
    for c in cards or []:
        r = "10" if c[0] == "T" else c[0]
        out.append(f"{r}{SUIT.get(c[1], c[1])}")
    return " ".join(out) if out else "—"


def _vpip_bucket(vpip):
    """The archetype FAMILY a VPIP reads as — ONE source of truth for both the inference
    line and the truth-check, so they can never disagree at a bucket boundary (15/45)."""
    if vpip is None:
        return None
    if vpip < 15:
        return "nit"
    if vpip > 45:
        return "station"
    return "TAG"


_BUCKET_PHRASE = {"nit": "a nit (ultra-tight)", "station": "a station (loose-passive)",
                  "TAG": "a TAG / balanced"}
_TRUTH_BUCKET = {"rock": "nit", "station": "station", "reg": "TAG"}


def _infer_from_vpip(vpip):
    """The read a player would make from the VISIBLE VPIP alone — what the coach reasons
    FROM (legitimate: the player can see VPIP too). NOT the hidden type."""
    return _BUCKET_PHRASE.get(_vpip_bucket(vpip), "unknown (no sample yet)")


def _truth_check(vpip, truth_type) -> str:
    """How much to TRUST the observed read — NOT an answer key. The hidden truth ONLY
    calibrates confidence and keeps the coach from teaching a WRONG read; it must NEVER
    name the type (the player earns that by watching). Coach from the observed behavior;
    this flag only says how far to commit to it."""
    b = _vpip_bucket(vpip)
    if b is None:
        return "no read yet — say so; coach the cards + position, and watch how they play"
    if b == _TRUTH_BUCKET.get(truth_type):
        return "observed read looks reliable — coach to it (describe the behavior, don't name a 'type')"
    return "don't lock this read — the small sample is misleading; stay cautious, don't commit yet"


# ---------------- human-readable layout (warm-session `next`) ----------------
def _format_villains(villains) -> str:
    lines = []
    for v in villains:
        o = v.get("observed", {})
        vp = o.get("vpip_pct")
        vp_s = f"{vp}%" if vp is not None else "—"
        extra = []
        if v.get("stack") is not None:
            extra.append(f"stack {v['stack']}")
        if v.get("in_hand") is not None:
            extra.append("in hand" if v["in_hand"] else "folded")
        extra_s = ("  [" + ", ".join(extra) + "]") if extra else ""
        lines.append(
            f"  {v['name']} ({v['position']}){extra_s}\n"
            f"      observed: {o.get('hands', 0)}h · VPIP {vp_s} · "
            f"{o.get('pf_raises', 0)} PFR · {o.get('showdowns', 0)} SD · net {o.get('net', 0)}\n"
            f"      read-confidence: {_truth_check(vp, v.get('_truth', {}).get('type'))}")
    return "\n".join(lines) if lines else "  (none)"


def _stored_equity(req_id: int, db) -> dict | None:
    """The modeled-equity blob already persisted for this request, or None if not ready. The channel is
    the single source of truth for the NUMBER, so both the worker (here) and the app read it from here —
    that's what keeps the drill reveal and the prose byte-identical."""
    pm = coach.poll_meta(req_id, db)
    return (pm.get("meta") or {}).get("modeled_equity") if pm.get("status") == "ready" else None


def _ensure_modeled_equity(ctx, req_id: int | None = None, db=None) -> None:
    """Worker-side: compute the LIVE modeled equity from the frozen range spec game.py baked. The Monte-
    Carlo lives HERE, off the single-threaded /api/coach/ask path (which returns the poll id instantly),
    not in coach_decision_context. Idempotent + fail-safe — a bad spot just yields no equity line.

    The number is CANONICALIZED through the channel so the drill reveal and the prose can never diverge,
    even under two workers or a restart (codex/kimi L3): with a req_id we (1) REUSE an already-persisted
    blob instead of recomputing, and (2) after a write-once `set_meta`, REREAD and adopt whatever is
    stored — so a worker that lost the write race still cites the winner's number in its prose. A bare
    call (req_id None — e.g. a unit test) just computes into ctx with no channel round-trip."""
    if not isinstance(ctx, dict) or ctx.get("modeled_equity") is not None:
        return
    spec = ctx.get("modeled_spec")
    if not spec:
        return
    dbp = db or paths.coach_db()
    if req_id is not None:                       # reuse a number already computed for this request
        me = _stored_equity(req_id, dbp)
        if me is not None:
            ctx["modeled_equity"] = me
            return
    me = equity.modeled_from_spec(spec)          # the SHARED code compute (server + worker, one path)
    if not me:
        return
    if req_id is not None:
        # best-effort: a lost meta write is a non-event. write-once means a concurrent worker may have
        # won — so REREAD and adopt the stored value (the loser uses the winner's number, never its own).
        try:
            coach.set_meta(req_id, {"modeled_equity": me}, dbp)
            stored = _stored_equity(req_id, dbp)
            if stored is not None:
                me = stored
        except Exception:
            pass
    ctx["modeled_equity"] = me


def _format_modeled_equity(me) -> str:
    """The LIVE modeled equity line: hero % vs the most relevant opponent's OBSERVED-read range (not
    their cards). The width is surfaced so the coach states the assumption; pot_way lets it discount for
    the field in a multiway pot; a thin VPIP sample is flagged so the coach treats the range as a guess."""
    if not me or me.get("hero_pct") is None:
        return ""
    v = me.get("vs") or {}
    hands = v.get("hands")
    thin = f", thin {hands}h sample — treat as a guess" if (hands is not None and hands < 10) else ""
    who = (f"{v.get('name')} (~{v.get('range_pct')}% range{', raised' if v.get('raised') else ''}{thin})"
           if v else "their range")
    eq_s = f"{me.get('band') or equity.band(me['hero_pct'])} (~{me['hero_pct']}%)"   # band leads; % calibrates
    if me.get("heads_up"):
        frame = f"{eq_s} heads-up vs {who}"
    else:
        pw = me.get("pot_way")
        frame = (f"{eq_s} vs just {who} (one of {me.get('live_count')} live)"
                 + (f" — it's a {pw}-way pot, so discount for the field" if pw else ""))
    return (f"\nModeled equity: {frame}. This is vs an OBSERVED-read range, NOT their actual cards — "
            "name the assumed range when you lean on this number; it's a read, not ground truth. Coach "
            "in the felt language (the band), not the bare number — the player is still learning to read equity.")


def _format_decision(ctx: dict, req_id: int | None = None, db=None) -> str:
    _ensure_modeled_equity(ctx, req_id, db)  # worker-side MC (off the ask path); idempotent + persists
    h = ctx.get("hero", {})
    po = ctx.get("pot_odds")
    po_s = (f"to call {po['call']} into {po['pot']} → need {po['breakeven_pct']}%"
            if po else "nothing to call — you can check")
    lg = ctx.get("legal", {})
    opts = []
    if lg.get("can_fold"):
        opts.append("fold")
    opts.append("check" if lg.get("can_check") else "call")
    if lg.get("can_raise"):
        opts.append(f"raise {lg.get('min_raise_to')}–{lg.get('max_raise_to')}")
    return (
        f"Your hand: {_cards(h.get('hole'))}  ({h.get('position')})  ·  {h.get('strength')}"
        f"  ·  stack {h.get('stack')}\n"
        f"Board: {_cards(ctx.get('board'))}   street: {ctx.get('street')}   pot: {ctx.get('pot')}\n"
        f"Pot odds: {po_s}"
        f"{_format_modeled_equity(ctx.get('modeled_equity'))}\n"
        f"Legal: {', '.join(opts)}\n"
        f"Action this hand:\n" +
        ("\n".join(f"  {ln}" for ln in ctx.get("action_this_hand", [])) or "  (you're first to act)") +
        f"\nVillains (observed reads | read-confidence — calibration only, NEVER volunteer a 'type'):"
        f"\n{_format_villains(ctx.get('villains', []))}")


def _format_decision_points(dps) -> str:
    """The hero's exact equity at each postflop decision (enumerated vs the opponents' ACTUAL cards
    — revealed in review). This is the prediction-error ground truth: the coach names where the gut
    and the real number diverged and WHY. The app shows the player the bare gap; the coach gives the why."""
    dps = [d for d in (dps or []) if d.get("equity") is not None]
    if not dps:
        return ""
    lines = []
    for d in dps:
        facing = d.get("to_call", 0) > 0
        spot = f"to call {d['to_call']} into {d['pot']}" if facing else f"checked, pot {d['pot']}"
        b = d.get('band') or equity.band(d['equity'])
        lines.append(f"  [{d['street']}] {_cards(d.get('board'))} — you were {b} ({d['equity']}%) "
                     f"vs {d.get('vs')} ({spot}); you {d.get('action')}")
    return ("\nYour EXACT equity at each decision (enumerated vs their actual cards — the ground "
            "truth; coach the gap between this and what they FELT, in the felt language [the band], not the "
            "bare number — the player is still learning to read equity):\n" + "\n".join(lines))


def _format_review(ctx: dict) -> str:
    h = ctx.get("hero", {})
    res = ctx.get("result", {})
    wins = ", ".join(f"{w['name']} +{w['amount']}" for w in res.get("winners", [])) or "—"
    sd = res.get("showdown") or []
    sd_s = ("\n  showdown: " + " · ".join(f"{s['name']} {_cards(s['hole'])}" for s in sd)) if sd else ""
    return (
        f"Your hand: {_cards(h.get('hole'))}  ({h.get('position')})   "
        f"final board: {_cards(ctx.get('final_board'))}\n"
        f"Result: you {res.get('hero_net', 0):+d}   pot {ctx.get('pot')}   winners: {wins}{sd_s}\n"
        f"How it went:\n" +
        ("\n".join(f"  {ln}" for ln in ctx.get("action_by_street", [])) or "  (no action)") +
        _format_decision_points(ctx.get("decision_points")) +
        f"\nVillains (session-final reads | read-confidence — calibration only, NEVER volunteer a 'type'):"
        f"\n{_format_villains(ctx.get('villains', []))}")


def _leak_line(leak: dict) -> str:
    """One graduated leak rendered for the coach: de-snaked name + level + the operator's own
    meaning summary. The name is de-snaked so the coach reads behavior, not an identifier. Both fields
    coerced through ``str(... or "")`` so a None/missing summary renders empty, never the literal
    'None' (the value is a str in the production path, but these are public helpers over a plain dict)."""
    name = str(leak.get("name") or "").replace("_", " ").strip()
    return f"  - {name} ({leak.get('level', 1)}x): {str(leak.get('summary') or '')}".rstrip()


def _format_leaks(ctx: dict) -> str:
    """P4b-i — render the operator's recalled graduated leaks (ctx['relevant_leaks'], attached
    server-side) for the coach prompt. EMPTY string when none are relevant (the common case — most
    spots match no leak; surface nothing rather than noise).

    HIGH-2: the recall surfaces a leak NAME + the operator's own graduated meaning — the AWARENESS half.
    The product's whole differentiator is the read→action IMPLICATION (doctrine: 'a read is incomplete
    until you've named what it forbids'), so the prompt makes the coach DERIVE that here: lead with the
    FORBIDDEN action for THIS spot, then the permitted one. The recalled summary often already carries it
    (the operator graduated the leak with its forbid/permit framing); where it doesn't, the coach draws it
    from the observed read in front of them — never free-associating from the bare name. Behavior language,
    never a 'type', and take-it-or-leave-it (scaffolding the gut absorbs, never a mid-hand checklist)."""
    leaks = ctx.get("relevant_leaks") or []
    if not leaks:
        return ""
    body = "\n".join(_leak_line(lk) for lk in leaks)
    return ("\n\n⚑ Read-leaks YOU'VE shown in spots like this (your own graduated read-memory). If one "
            "genuinely fits THIS spot, surface it as awareness and coach its read→action implication — "
            "LEAD with what it FORBIDS here, then what it permits, tied to the observed read in behavior "
            "language. If none fits this spot, say nothing about them. Never name a villain 'type':\n" + body)


def format_spot(req: dict, db=None) -> str:
    """The laid-out spot — head + body, NO respond instructions. Shared by the warm-session
    `next` view (which appends a respond hint) and the headless daemon (coach_daemon.py, which
    feeds this straight to `claude -p`). One layout, two readers — so the warm session and the
    daemon coach from byte-identical spots. `db` flows to the modeled-equity persist so the meta
    lands in the SAME channel the caller reads (None ⇒ the default data-dir coach.db, the production path)."""
    ctx = req.get("context", {})
    kind = req.get("kind")
    head = f"=== Coaching request #{req['id']} ({kind}) — hand #{ctx.get('hand_no', '?')} ==="
    body = _format_decision(ctx, req.get("id"), db) if kind == "decision" else _format_review(ctx)
    return f"{head}\n{body}{_format_leaks(ctx)}"


def format_request(req: dict, db=None) -> str:
    tail = f"\nRespond:  python -m gutcheck.coach respond --id {req['id']} --body \"<your coaching>\""
    return f"{format_spot(req, db)}\n{tail}"


def cmd_next() -> int:
    pend = coach.pending()
    if not pend:
        print("no pending coaching requests.")
        return 0
    print(format_request(pend[0]))
    if len(pend) > 1:
        print(f"\n(+{len(pend) - 1} more pending: "
              f"{', '.join('#' + str(p['id']) for p in pend[1:])})")
    return 0


# ---------------- facts-only answer: code-true facts, NO recommendation ----------------
# Two callers, two framings (facts_mode):
#   facts_mode=True  — coaching is DELIBERATELY off (the operator runs facts-only during interactive
#                      play, coaching their own read or a partner coaching in conversation). NO head,
#                      no scripted sign-off — just the numbers. It is NOT a degradation, so it must not
#                      announce itself or pump template tells.
#   facts_mode=False — the DEGRADED path (AI coach unavailable: it errored, the breaker is open, or the
#                      spot aged before it answered). A short, GENERIC head is the honest signal that
#                      this isn't the intended experience.
# Neither mode ever makes the bet/fold/call — a template recommendation taught with confidence trains
# WORSE reading (the disentangle-(2) call); the facts are ground truth, the read stays the operator's.
_DEGRADED_HEAD = "AI coach unavailable — facts only:"


# The facts-only floor can't JUDGE fit the way the LLM coach can (the coach is told "skip it if it
# doesn't fit"; the stub just prints). So the un-judged surface uses a TIGHTER relevance bar than the
# coach — it shows only leaks that match the spot STRONGLY (>= this many situational words), not the
# coach's looser min_overlap=2. (Phill's visual-gate call: keep the leak in the stub, but high-confidence
# matches only — so a borderline 2-word match reaches the coach who can weigh it, never the bare floor.)
_STUB_MIN_OVERLAP = 3


def _stub_leaks_line(ctx: dict) -> str:
    """P4b-i in the facts-only floor: the operator's OWN graduated read-leaks relevant to this spot,
    stated as FACT (their own graduated wording), not a fresh recommendation — honest in deliberate-OFF
    mode (self-coaching) the way the observed VPIP read is a fact, not a steer. '' when none clear the
    tighter floor bar (a strong, high-confidence match — the floor can't judge fit, so it only shows
    leaks that fit strongly). Behavior-language + archetype-free by construction (the leaks carry no 'type')."""
    leaks = [lk for lk in (ctx.get("relevant_leaks") or [])
             if lk.get("overlap", 0) >= _STUB_MIN_OVERLAP]
    if not leaks:
        return ""
    items = " · ".join(f"{str(lk.get('name') or '').replace('_', ' ').strip()} — {str(lk.get('summary') or '')}"
                       for lk in leaks)
    return f"\n⚑ Your read-leaks for spots like this: {items}"


def _stub_decision(ctx: dict, req_id: int | None = None, db=None, facts_mode: bool = False) -> str:
    """Facts-only decision answer: hand + board, the CODE-computed modeled equity (anti-fab — the same
    canonicalized number the live reveal shows), pot odds, and the VPIP read the operator can see too.
    No recommendation, no scripted sign-off. facts_mode=True omits the degraded head (deliberate OFF)."""
    _ensure_modeled_equity(ctx, req_id, db)  # worker-side MC (off the ask path); idempotent + persists
    prefs = settings.stat_prefs()            # which grounding facts to show (slice b — operator-tailored)
    h = ctx.get("hero", {})
    me = ctx.get("modeled_equity")
    eq_line = ""
    if me and me.get("hero_pct") is not None:
        v = me.get("vs") or {}
        field = "" if me.get("heads_up") else f", {me.get('pot_way')}-way (discount for the field)"
        b = me.get('band') or equity.band(me['hero_pct'])
        # equity_format=band → felt-sense only (drop the hero %); the villain range-width stays (a
        # different fact, the read's context). band_pct keeps the precise number too.
        pct = f" (~{me['hero_pct']}%)" if prefs["equity_format"] == "band_pct" else ""
        eq_line = (f"\nModeled equity: {b}{pct} vs {v.get('name', 'their')} "
                   f"(~{v.get('range_pct')}% range){field} — vs a read range, not their actual cards.")
    board = ctx.get("board")
    lines = [f"{h.get('strength')}{' on ' + _cards(board) if board else ''}.{eq_line}"]
    if prefs["show_pot_odds"]:
        po = ctx.get("pot_odds")
        po_s = (f"to call {po['call']} into {po['pot']} → you need {po['breakeven_pct']}%"
                if po else "nothing to call — you can check")
        lines.append(f"Pot odds: {po_s}.")
    if prefs["show_observed_read"]:
        # the read a player would make from the VISIBLE VPIP alone — a fact, not a steer. The hidden-truth
        # confidence check is the coach's private calibration; it has no place in a facts-only answer.
        villains = [v for v in ctx.get("villains", []) if v.get("in_hand")]
        reads = [f"{v['name']} reads as {_infer_from_vpip(v.get('observed', {}).get('vpip_pct'))}"
                 for v in villains]
        lines.append(f"Reads: {'  '.join(reads) if reads else 'no live reads yet'}.")
    head = "" if facts_mode else _DEGRADED_HEAD + "\n"
    return head + "\n".join(lines) + _stub_leaks_line(ctx)


def _stub_review(ctx: dict, facts_mode: bool = False) -> str:
    """Facts-only review answer: the result, and the EXACT equity at each decision (enumerated vs the
    villains' actual cards — the prediction-error ground truth). The numbers are the honest gap between
    gut and math; the lesson is the operator's. facts_mode=True omits the degraded head (deliberate OFF)."""
    res = ctx.get("result", {})
    net = res.get("hero_net", 0)
    verdict = ("won the pot" if net > 0 else "lost the pot" if net < 0 else "broke even")
    dps = [d for d in (ctx.get("decision_points") or []) if d.get("equity") is not None]
    lines = []
    for d in dps:
        b = d.get('band') or equity.band(d['equity'])
        facing = d.get("to_call", 0) > 0
        spot = f"to call {d['to_call']} into {d['pot']}" if facing else f"checked, pot {d['pot']}"
        lines.append(f"  [{d['street']}] {_cards(d.get('board'))} — {b} ({d['equity']}%) "
                     f"vs {d.get('vs')} ({spot}); you {d.get('action')}")
    eq_block = ("\nYour exact equity at each decision (vs their actual cards):\n" + "\n".join(lines)
                ) if lines else ""
    head = "" if facts_mode else _DEGRADED_HEAD + "\n"
    return (f"{head}"
            f"You {verdict} ({net:+d}) on a final board of {_cards(ctx.get('final_board'))}.{eq_block}"
            f"{_stub_leaks_line(ctx)}")


def stub_answer(req: dict, db=None, facts_mode: bool = False) -> str:
    ctx = req.get("context", {})
    return (_stub_decision(ctx, req.get("id"), db, facts_mode) if req.get("kind") == "decision"
            else _stub_review(ctx, facts_mode))


def cmd_stub(once: bool) -> int:
    answered = 0
    while True:
        for req in coach.pending():
            # per-row guard (mirrors coach_daemon.answer_one): a malformed spot must not crash the
            # whole fallback loop — drain the row with a safe line so it leaves the queue, then log.
            try:
                ok = coach.respond(req["id"], stub_answer(req))
            except Exception as e:
                print(f"stub: request #{req.get('id')} failed: {e!r}", file=sys.stderr)
                try:
                    coach.respond(req["id"], "(coach hit an error on this spot)")
                except Exception:
                    pass
                continue
            if ok:
                answered += 1
                print(f"stub answered #{req['id']} ({req['kind']})", file=sys.stderr)
        if once:
            print(f"stub: answered {answered} request(s).", file=sys.stderr)
            return 0
        time.sleep(1.0)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gutcheck coach worker")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("next", help="print the next pending request, laid out to read")
    st = sub.add_parser("stub", help="auto-answer with honest facts only — no recommendation "
                                     "(test harness / offline fallback)")
    st.add_argument("--once", action="store_true", help="single pass, then exit")
    args = ap.parse_args(argv)
    if args.cmd == "next":
        return cmd_next()
    if args.cmd == "stub":
        return cmd_stub(args.once)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
