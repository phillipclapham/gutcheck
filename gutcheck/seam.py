"""gutcheck seam — the transport-neutral coach-facing contract (Layer B).

The canonical-object / replaceable-surface law made structural: ONE seam over the engine,
N thin transports over the seam. server.py (HTTP) consumes it today; the MCP adapter (P2)
will be another thin consumer; a future host is just one more. Every transport reaches the
engine ONLY through here, so coaching logic can never get locked inside one transport and
go dark in the others — the guarantee behind "building an MCP server can't cut off Claude
Code" (productization design §4.0).

Pure data in / data out — no Request/Response objects, no auth, no serialization. Those
live at each transport's EDGE: server.py owns the host/CSRF/body-size guards and the JSON
encoding; the MCP adapter will own its own gating. The seam owns the CONTRACT and the
structural invariants every transport must inherit:

  * anti-fabrication — the read TRUTH is recomputed/frozen server-side inside the engine
    (Game.record_review_read / the worker's frozen modeled blob), never trusted from a
    client. A transport supplies only the operator's gut + lean + which spot.
  * pedagogy gating (§3f) — see the get_ground_truth slot below.

Deliberately thin: this is a renaming + gating layer over Game's already-clean methods,
not a re-implementation. Game stays the engine (Layer A); history.py is the canonical
hand-history store; this maps the productization vocabulary onto them so the contract is
decoupled from Game's internal method names.
"""

from __future__ import annotations

from . import history


class Seam:
    """A thin facade over a live Game instance plus the (stateless) history store."""

    def __init__(self, game):
        self._game = game
        # Read history from the SAME store the game WRITES to (game.history_db) — never re-resolve the
        # env-default independently. Otherwise an injected/isolated Game writes one db and the seam reads
        # another: silent empty/stale history AND a real-state isolation breach (a Seam-using test could
        # touch ~/.gutcheck). The seam PROJECTS the canonical object's store; it doesn't pick its own.
        # [codex L3 HIGH — the cross-substrate catch L1/L2 both missed]
        self._history_db = game.history_db

    # ---------------- reads ----------------
    def get_table_state(self) -> dict:
        """The FULL table view (the /api/state payload the BROWSER renders). It carries the hidden villain
        `type`/`archetype` per seat for the browser's click-to-reveal self-test — so it is NOT coach-safe.
        The coach must use get_coach_table_state (below), which strips them (§3f)."""
        return self._game.view()

    def get_coach_table_state(self) -> dict:
        """COACH-facing table state: the live table the operator sees, with the villain ARCHETYPE
        STRIPPED (§3f — the coach coaches from OBSERVED behavior; the hidden type is the operator's to
        earn). Strips `type` + `archetype` from the live `seats` AND the between-hands `table.seats`
        snapshot, keeping everything else incl. the OBSERVED `stats` (the read material). Deep-copied so
        the strip never mutates the stored last_table. [codex L3 HIGH — get_table_state leaked the gate]"""
        import copy
        v = copy.deepcopy(self._game.view())
        for s in (v.get("seats") or []):
            s.pop("type", None)
            s.pop("archetype", None)
        table = v.get("table")
        if isinstance(table, dict):
            for s in (table.get("seats") or []):
                s.pop("type", None)
                s.pop("archetype", None)
        return v

    def get_history(self, since: int = 0, session_id=None, limit: int = 500,
                    latest: bool = False) -> dict:
        """The replayable hand stream with id > since (half-open watermark). Stateless — read straight
        off the canonical history store the game writes to (self._history_db), independent of the live
        session, so a consumer can pull the full cross-session history without a live table.

        ``session_id`` optionally scopes the stream to one session ("replay session X"); ``None``/empty
        returns every session (the prior behavior). The global-id watermark is unaffected by the scope.

        ``limit`` caps the rows (the store default is 500); ``latest`` returns the most-recent ``limit``
        hands instead of the oldest (the coach's "what did I just play" window — see history.get_history).
        Both pass straight through to the store — the seam previously dropped ``limit`` on the floor, so
        a budget-bound consumer (the MCP coach surface) could not page a small recent window."""
        return history.get_history(since, limit=limit, path=self._history_db,
                                   session_id=session_id, latest=latest)

    def get_calibration(self) -> dict:
        """The gut-vs-truth read-accuracy trend (exact = headline, modeled = flagged secondary)."""
        return self._game.calibration_summary()

    def get_sizing_tells(self, session_id=None) -> dict:
        """P3(c) — per-villain OBSERVED betting-behavior → shown-strength tell, from showdown-revealed
        hands. RAW FACTS only (showdown count, aggressive-postflop strength breakdown, per-size buckets);
        no coaching prose and no archetype name (§3f — the coach reads aggression and speaks in
        behavior). ``session_id`` scopes to one session; ``None`` = the live session. The dominant signal
        is AGGRESSION (the v0 villains size ~constantly), which the doctrine teaches the coach to read."""
        return self._game.sizing_tells(session_id)

    def get_read_reconciliation(self, session_id=None) -> dict:
        """P3a — modeled−exact reconciliation for live reads ('read the range right, still
        crushed'). ``session_id`` scopes to one session (the live coach + the hand-end card);
        ``None`` returns every joinable read across all sessions (the cross-session collapse
        trend). Exact equity recomputed server-side from the canonical store (anti-fab). Wrapped
        in a dict so the transport returns an object, not a bare list."""
        return {"reads": self._game.reconcile_live_reads(session_id)}

    # ---------------- writes ----------------
    def commit_read(self, kind: str, **kw) -> dict:
        """Log a committed read against ground truth. ``kind`` selects the calibration source:

          * ``"review"`` -> record_review_read(session_id, hand_no, n, gut, lean): truth is
            recomputed server-side from (session_id, hand_no) — the EXACT enumerated equity.
          * ``"live"``   -> record_live_read(gut, lean, truth_pct, street=, vs=, req_id=): the
            caller passes the worker's FROZEN modeled number (the transport reads it off the
            coach channel; the seam never touches that plumbing), never a browser value.

        The anti-fabrication invariant lives below the seam (in the engine), so every transport
        inherits it for free — there is no truth value for a transport to forge here."""
        if kind == "review":
            return self._game.record_review_read(**kw)
        if kind == "live":
            return self._game.record_live_read(**kw)
        return {"ok": False, "reason": f"unknown read kind: {kind!r}"}

    def get_ground_truth(self, session_id, hand_no, reveal_archetype: bool = False) -> dict:
        """The coach's CALIBRATABLE ANSWER-KEY for a stored hand (P2 — the MCP coach is the consumer).

        ALWAYS exposed (the calibratable facts — the method's whole superpower is the sim handing the
        coach the answer key): the exact enumerated equity at each hero decision (`decision_points`),
        the showdown holes + winners (`result`), the `final_board`, and the hero's spot. These are what
        the coach uses to CONFIRM or CORRECT the operator's read.

        GATED behind ``reveal_archetype`` (default OFF) — the §3f STRUCTURAL pedagogy gate: the villain
        ARCHETYPE NAME (each villain's hidden ``_truth``) is withheld so a shipped MCP coach CAN'T spoil
        the read — not a coach-side discipline it could leak, but the seam simply not handing over the
        answer. The OBSERVED stats stay (those ARE the read material the coach works from). [§3f]"""
        built = self._game.coach_review_context(session_id, hand_no)
        if built is None:
            return {"ok": False, "reason": "hand not found"}
        ctx, _frames = built
        villains = []
        for v in ctx.get("villains", []):
            vv = {k: val for k, val in v.items() if k != "_truth"}   # drop the hidden archetype...
            if reveal_archetype and isinstance(v.get("_truth"), dict):
                vv["archetype"] = v["_truth"].get("label")            # ...unless explicitly opted in
            villains.append(vv)
        return {
            "ok": True, "session_id": session_id, "hand_no": hand_no,
            "final_board": ctx.get("final_board"),
            "decision_points": ctx.get("decision_points", []),   # exact equity per hero postflop spot
            "result": ctx.get("result"),                          # winners + showdown holes + hero_net
            "hero": ctx.get("hero"),
            "villains": villains,                                  # observed stats; archetype gated
            "archetype_revealed": bool(reveal_archetype),
        }

    # ---------------- leak memory (P4 — the operator-owned graduating-leak store, the moat) ----------------
    # The dead-store-vs-immune-system half: the coach names a RECURRING read-leak in observed-behavior
    # language, citing the (session_id, hand_no, n) join key of the reads that evidence it; anneal's
    # immune system graduates it 1x->2x->3x (and demotes a sycophantic rephrase). Read-based + archetype
    # gated by construction — the leak grounds on episodes the seam already wrote WITHOUT the type (§3f),
    # so the leak memory structurally cannot spoil. OPTIONAL: a clear marker when gutcheck[anneal] is absent.
    _LEAKS_UNAVAILABLE = {"ok": False, "reason": "leak memory unavailable (install gutcheck[anneal])"}

    def _leak_store(self):
        """The Game's leak store, or None when the [anneal] extra isn't installed / not wired on."""
        ls = getattr(self._game, "leaks", None)
        return ls if (ls is not None and getattr(ls, "available", False)) else None

    def prepare_leak_wrap(self, recover: bool = False) -> dict:
        """Open a leak-naming wrap — the coach compresses recent read-episodes into NAMED, cited leaks.
        On ok:True, returns the compression package + ``wrap_token``. On ok:False the ``reason`` says why:
        'no new reads to graduate' (empty store), or 'a leak wrap is already in progress' (a prior wrap
        is open — the seam NEVER auto-cancels it; pass ``recover=True`` to deliberately reclaim a
        KNOWN-abandoned one, e.g. after a crashed coach). [codex L3 HIGH-1]"""
        ls = self._leak_store()
        if ls is None:
            return dict(self._LEAKS_UNAVAILABLE)
        wrap = ls.prepare_leak_wrap(recover_abandoned=recover)
        if wrap is None:
            return {"ok": False, "reason": "leak wrap could not be prepared"}
        status = wrap.get("status")
        # A prior wrap is still open (single-writer). DON'T cancel it on a plain prepare — surface it as
        # recoverable so the operator/coach decides; recover=True does the deliberate reclaim. [codex HIGH-1/MED-2]
        if status == "wrap_in_progress":
            return {"ok": False, "reason": "a leak wrap is already in progress",
                    "status": status, "recoverable": True}
        # An "empty" wrap (no new reads since the last graduation) mints NO wrap_token and CANNOT be
        # committed — surface ok:False so the coach branches on it instead of discovering it via a
        # ValueError on commit_leaks(doc, None). [L2 M1]
        if status != "ready" or not wrap.get("wrap_token"):
            return {"ok": False, "reason": "no new reads to graduate",
                    "status": status, "episode_count": wrap.get("episode_count")}
        return {"ok": True, **wrap}

    def commit_leaks(self, continuity_text: str, wrap_token: str,
                     affect_tag: str | None = None, affect_intensity: float | None = None) -> dict:
        """Save the coach-named leaks doc through anneal's FULL immune system (citation-validated
        graduation, demotion of ungrounded/sycophantic leaks, Hebbian links). ``wrap_token`` must come
        from a preceding ``prepare_leak_wrap``. Returns the validation result, or a failure reason."""
        ls = self._leak_store()
        if ls is None:
            return dict(self._LEAKS_UNAVAILABLE)
        # leakstore deliberately lets ValueError propagate — a malformed wrap (bad/empty doc, no wrap in
        # progress, wrong token) OR anneal's catastrophic-shrink policy rejection. The SEAM is the
        # transport boundary, so normalize it into the leak API's {ok, reason} shape HERE rather than
        # leaking raw anneal text to the client through the server's generic handler. [L1 MED-1 / L2 M2]
        try:
            result = ls.commit_leaks(continuity_text, wrap_token,
                                     affect_tag=affect_tag, affect_intensity=affect_intensity)
        except ValueError as e:
            return {"ok": False, "reason": f"malformed or rejected leak wrap: {e}"}
        if result is None:
            return {"ok": False, "reason": "leak commit failed"}
        return {"ok": True, "result": result}

    def get_leaks(self) -> dict:
        """The current graduated-leaks doc — the leaks the coach recalls at the decision point. Returns
        ``{"ok": True, "leaks": <markdown or None>, "status": "ok"|"empty"|"load_error"}``. leaks=None
        with the status distinguishes a healthy-empty store from a SILENT read failure, so a client can't
        read 'no leaks learned yet' off a broken moat (complement L3 MED-2)."""
        ls = self._leak_store()
        if ls is None:
            return dict(self._LEAKS_UNAVAILABLE)
        leaks = ls.get_leaks()
        if leaks is None:
            status = "load_error" if getattr(ls, "last_load_error", False) else "empty"
            return {"ok": True, "leaks": None, "status": status}
        return {"ok": True, "leaks": leaks, "status": "ok"}

    def recall_relevant_leaks(self, query: str, limit: int = 2) -> dict:
        """P4b-i — recall the operator's GRADUATED leaks most relevant to ``query`` (a spot-derived
        string), for injection into the live coach context at the decision point. Returns
        ``{"ok": True, "leaks": [{"name","level","summary"}], "status": "ok"|"none"}`` — ``none`` when
        nothing cleared the relevance bar (a distinct, honest "no leak applies here", not an error).
        Archetype-free by construction (§3f). Unavailable (no ``[anneal]`` extra) → the unavailable
        marker, same as the other leak methods, so a transport branches cleanly."""
        ls = self._leak_store()
        if ls is None:
            return dict(self._LEAKS_UNAVAILABLE)
        leaks = ls.recall_relevant_leaks(query, limit=limit) or []
        return {"ok": True, "leaks": leaks, "status": "ok" if leaks else "none"}
