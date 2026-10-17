"""gutcheck leak memory — the operator-owned, anneal-backed graduating-leak store (P4, the moat).

The flat ``calibration.json`` is a DEAD store: it accumulates committed reads, but it never
graduates a *recurring* error into a NAMED leak the coach recalls next time you're in the spot.
That gap is the moat. This adapter wraps `anneal-memory` so the operator's leaks GRADUATE
(1x -> 2x -> 3x) through anneal's citation-validated immune system: the dead-store-vs-immune-system
difference, made felt and measured in a winrate.

The mapping (verified against anneal 0.8.5, and on 2026-10-17 against 1.0.0: test_leakstore.py,
test_leak_recall.py and test_coach_daemon.py pass on both):
  * calibration reads  -> anneal EPISODES (the citable evidence; the (session_id, hand_no, n)
    join key from P0 travels in the content so a leak cites the exact decision, never infers it).
  * leaks              -> NAMED PATTERNS in anneal's 4-section continuity doc
    (## State / ## Patterns / ## Decisions / ## Context), graduating via cited episode evidence.
  * graduation         -> anneal's immune system: >=2-word grounding to the cited episode, and a
    <3-word cross-session overlap gate that DEMOTES a rephrase of a prior session's explanation.

OPTIONAL BY CONSTRUCTION. The core game never imports this module; `anneal-memory` is an optional
extra (`pip install gutcheck[anneal]`). When anneal is absent, ``LeakStore`` is a graceful no-op —
the game plays + calibrates exactly as before, it just doesn't graduate leaks. The single-dep
`pip install gutcheck` artifact is unchanged.

OPERATOR-OWNED (§3e). The leak store lives in the data dir next to ``calibration.json`` — never
bundled, never synced without the operator's say.

READ-BASED, ARCHETYPE-GATED (§3f). The coach names leaks in OBSERVED-BEHAVIOR language; the hidden
villain archetype never enters an episode or a leak. The seam already withholds the type, so the
leak memory structurally *cannot* spoil the read — not a coach-side discipline it could leak.

Every store operation is GUARDED: a leak-store failure must never lose a calibration sample or
break play. The methods return ``None`` (not raise) on any anneal error, so the caller treats the
leak layer as best-effort enrichment over the durable calibration trend.
"""

from __future__ import annotations

import threading

from . import paths

# anneal-memory is an OPTIONAL dependency. Import it guarded so importing leakstore (and therefore
# constructing a Game with a leak store) never hard-fails on a single-dep install.
try:  # pragma: no cover - exercised by both branches across environments
    from anneal_memory import (
        Store,
        EpisodeType,
        AffectiveState,
        prepare_wrap,
        validated_save_continuity,
        extract_pattern_summaries,
        extract_keywords,
        AnnealMemoryError,
        WrapInProgressError,
    )

    _ANNEAL_AVAILABLE = True
except ImportError:  # pragma: no cover - the no-anneal branch
    _ANNEAL_AVAILABLE = False
    AnnealMemoryError = Exception        # so except-clauses below are importable without anneal
    WrapInProgressError = AnnealMemoryError  # never reached at runtime (methods gate on _ensure())


PROJECT_NAME = "gutcheck-operator"


def anneal_available() -> bool:
    """True iff `anneal-memory` is importable in this environment (the `[anneal]` extra is installed)."""
    return _ANNEAL_AVAILABLE


def _fmt_pct(x) -> str:
    """A read value rendered for prose as a WHOLE percent, or '?' if missing (never crash on a partial
    sample). Whole-percent is deliberate — the episode is evidence prose; the durable calibration trend
    keeps the full-precision value."""
    try:
        return f"{float(x):.0f}"
    except (TypeError, ValueError):
        return "?"


def _as_int(v):
    """Coerce an int / digit-string to int, else None (bool excluded). A sample rehydrated from JSON can
    carry ``hand_no``/``n`` as strings — coercing keeps the prose join key and the structured metadata
    join key in agreement on type (codex L3 LOW-1 / complement LOW-3)."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, str) and v.lstrip("-").isdigit():
        return int(v)
    return None


def _join_key(sample: dict):
    """The validated (session_id, hand_no, n) join key — the SINGLE source of truth shared by the prose
    (_episode_content) and the structured metadata (record_read_episode) so the two NEVER disagree
    (codex L3 LOW-1). All-or-nothing, mirroring record_live_read's discipline: a full tuple only when
    session_id is a str AND both ordinals coerce to int; else session-only; else all None."""
    sid = sample.get("session_id")
    if not isinstance(sid, str):
        return (None, None, None)
    hand_no, n = _as_int(sample.get("hand_no")), _as_int(sample.get("n"))
    if hand_no is not None and n is not None:
        return (sid, hand_no, n)
    return (sid, None, None)


def _as_dict(result):
    """Normalize a ``prepare_wrap`` result to a plain dict. anneal 0.8.5 returns a dict; the dep is
    floor-pinned (>=0.8.5) to track immune-system fixes, so guard against a future version returning a
    dataclass / Mapping — a bare ``wrap.get()`` downstream would AttributeError → 500 past the seam's
    normalization (complement L3 MED-1)."""
    if isinstance(result, dict):
        return result
    try:
        return dict(result)                       # dict-like / Mapping / (k, v) pairs
    except (TypeError, ValueError):
        return {k: getattr(result, k, None)
                for k in ("status", "wrap_token", "episode_count", "package", "message")}


def _episode_content(sample: dict) -> str:
    """Render one calibration sample as ARCHETYPE-FREE, behavior-language episode content.

    The content is what a leak explanation grounds against (anneal's >=2-word grounding check), so it
    reads as a hand the coach can cite — the read, the gut-vs-truth gap, the player's own words — plus
    the (session_id, hand_no, n) join key so the leak cites the exact decision instead of inferring it.
    It deliberately carries NO villain ``type``/``archetype`` (§3f): the read is the operator's to earn.
    """
    street = sample.get("street") or "preflop"
    lean = sample.get("lean") or "—"
    source = sample.get("source") or "modeled"           # exact (review) | modeled (live)
    gut, truth = _fmt_pct(sample.get("gut")), _fmt_pct(sample.get("truth"))
    band = "band hit" if sample.get("band_hit") else "band miss"
    gap = sample.get("gap")
    gap_str = f"{gap:+g}" if isinstance(gap, (int, float)) else "?"
    # Flatten the player's note to a single line AND strip leading '#' so it can't read as a Markdown
    # heading. It interpolates into episode content, and a future P4b coach that echoes episode prose
    # into the commit doc must not be able to smuggle a fake "## Patterns" / graduated-leak line — a
    # single-line "## Patterns - evil | 5x" still starts with '#' after the flatten, so strip it too.
    # Episodes are never parsed as continuity sections today; this is defense-in-depth for P4b. [L1/L3 LOW-2]
    note = " ".join((sample.get("note") or "").split()).lstrip("#").strip()

    sid, hand_no, n = _join_key(sample)                    # shared with the metadata key — never disagree
    if sid and hand_no is not None and n is not None:
        join = f"[session {sid} · hand {hand_no} · decision {n}]"
    elif sid:
        join = f"[session {sid}]"                          # preflop / pre-join read: honest partial key
    else:
        join = "[unjoined read]"

    head = (f"Read committed at {street} (lean {lean}) — gut {gut}% vs {source} truth {truth}%: "
            f"{band}, gap {gap_str}.")
    note_part = f' Read in your words: "{note}".' if note else ""
    return f"{head}{note_part} {join}"


# The spot query is built from a SMALL fixed vocabulary (server._spot_query), so a handful of frame
# nouns recur on essentially every query and carry ZERO discriminative signal about WHICH read-error a
# spot risks. Counting them as overlap is what let a leak clear the bar on boilerplate alone — the
# autoimmune false-positive both L1+L2 flagged (a leak firing on a spot it shares nothing real with).
# Strip them before scoring so a match must land on SITUATIONAL signal (street, strength, behavior, or
# the leak's own read-error words). KEEP value/bet/checked/loose/tight/etc. — those vary by spot and ARE
# how a leak's situation matches a spot's. (Heuristic v1; tune on real leak data once the corpus exists.)
_SPOT_SCAFFOLD = frozenset({"villain", "villains", "spot", "spots", "hand", "hands", "made", "facing"})

# Street is the single highest-signal discriminator and it is ALWAYS present — so it gates, not scores.
_STREETS = frozenset({"preflop", "flop", "turn", "river"})


def _leak_keywords(name: str, summary: str) -> set:
    """The keyword set for one leak — its de-snaked NAME (``barrel_into_the_caller`` -> ``barrel into
    the caller``, so the identifier's words count) plus its meaning summary. Reuses anneal's
    ``extract_keywords`` so the leak and the query tokenize identically (a mismatch would mis-score)."""
    return set(extract_keywords(f"{name.replace('_', ' ')} {summary or ''}"))


def _street_excluded(query_streets: set, leak_kw: set) -> bool:
    """A street-named leak on a DIFFERENT street is never this spot's leak (a river-overfold lesson has
    nothing to say on a turn value bet — the cross-street false positive L1+L2 both reproduced). A
    street-AGNOSTIC leak (no street in its text) always passes — it's situational across streets. The
    hard precision gate that the soft keyword overlap structurally couldn't enforce."""
    leak_streets = _STREETS & leak_kw
    return bool(leak_streets) and bool(query_streets) and leak_streets.isdisjoint(query_streets)


def _score_leak(query_kw: set, leak_kw: set, level: int) -> tuple[float, int]:
    """Rank ONE graduated leak against the spot keywords. Returns (score, meaningful_overlap_count).

    Relevance = the count of SITUATIONAL keywords shared (the raw overlap minus the recurring frame
    nouns), level-weighted so a 3x leak that matches outranks a 1x match (graduated leaks earned more
    evidence → more confidence the recall is worth surfacing). Scoring on the scaffold-stripped overlap
    is the precision fix: two frame words (``villain`` + ``value``) no longer constitute a 'match'."""
    meaningful = (query_kw & leak_kw) - _SPOT_SCAFFOLD
    if not meaningful:
        return (0.0, 0)
    # gentle level multiplier — enough to break ties toward the more-graduated leak, never enough to
    # let a 3x with one weak overlap beat a 1x with strong overlap. CLAMP to [1,3] (codex L3 LOW): the
    # graduation ladder is 1x->3x, but extract_pattern_summaries will parse an arbitrary '99x' off a
    # malformed doc line — unclamped, that would let a junk high-level leak dominate every real match.
    level = min(max(int(level), 1), 3)
    return (len(meaningful) * (1.0 + 0.15 * (level - 1)), len(meaningful))


class LeakStore:
    """Operator-owned, anneal-backed leak memory. A graceful no-op when `anneal-memory` is absent.

    Construction is cheap and never touches the filesystem; the underlying anneal ``Store`` opens
    lazily on first use (so importing/constructing in a no-leak context costs nothing). The DB path
    is injectable like every other gutcheck store, so a test isolates it to a tmp dir.

    THREADING (codex L3 HIGH-2 / complement LOW-1; the P4b-ii GATE, now CLOSED): anneal's ``Store`` is
    ``check_same_thread=True`` — a connection is bound to the thread that opened it. P4b-ii drives the
    WRITE lifecycle (``prepare_leak_wrap`` -> ``commit_leaks``) from the coach DAEMON thread while the
    serve thread still records reads + recalls — two threads on one db. So each thread gets its OWN
    ``Store`` via the thread-local below (anneal's own guidance: "create a Store per thread"); ``_store``
    is a thread-local-backed property, so a test can still inject a fake on its own thread. The RLock
    serializes every op AND every lazy open ACROSS threads: anneal opens its sqlite WITHOUT a
    ``busy_timeout``, so two connections writing at once would raise 'database is locked' — the lock
    prevents that (it is no longer merely the double-open guard it was pre-P4b-ii). The one slow step —
    the ``claude -p`` leak-authoring call — happens in the seed driver BETWEEN prepare and commit with NO
    lock held, so a seeding pass on the daemon thread never blocks the serve thread's read-recording. The
    wrap snapshot is db-persisted (anneal metadata), so prepare and commit on the daemon thread's one
    connection compose cleanly; ``record`` has no wrap-in-progress guard, so a read recorded mid-wrap
    simply falls into the NEXT snapshot.
    """

    def __init__(self, db_path=None):
        self.available = _ANNEAL_AVAILABLE
        self._db_path = str(db_path) if db_path is not None else None
        # Per-thread Store connections (the P4b-ii thread-safety fix): anneal's Store is
        # check_same_thread=True and P4b-ii adds a SECOND writing thread (the coach daemon). Each thread
        # lazily opens its OWN connection into this thread-local; the ``_store`` property below reads /
        # writes the CURRENT thread's slot, so every existing call site (and the test's fake-store
        # injection) is unchanged. The RLock serializes ops + opens across threads (see the class docstring).
        self._tls = threading.local()
        self._lock = threading.RLock()       # serializes store ops + opens ACROSS the serve + daemon threads
        self.last_load_error = False         # set on a swallowed get_leaks() error so the seam can tell
        #                                      a healthy-empty store from a broken read (complement MED-2)

    @property
    def _store(self):
        """The CURRENT thread's anneal Store, or None if not opened on this thread yet. Thread-local so
        each thread holds its own ``check_same_thread`` connection; exposed as ``_store`` so a test can
        inject a fake (``ls._store = _Boom()``) on its own thread exactly as before."""
        return getattr(self._tls, "store", None)

    @_store.setter
    def _store(self, value):
        self._tls.store = value

    # ---------------- lifecycle ----------------
    def _resolve_path(self) -> str:
        return self._db_path or str(paths.leaks_db())

    def _ensure(self):
        """Open the CURRENT thread's anneal store on first use; return it, or None if anneal is
        unavailable / open failed. ``_store`` is thread-local, so each thread opens its OWN connection
        (by design, post-P4b-ii) — two threads constructing their own Stores concurrently is fine. The
        lock still wraps the open because the open runs init WRITES (DDL / WAL pragma / a commit), which
        must not race another thread's in-flight write on the shared db file (anneal sets no busy_timeout).
        ``available`` is a process-global kill switch: a failed open disables the layer for ALL threads —
        deliberate, since a Store that can't open is a shared-filesystem problem, not a per-thread one."""
        if not self.available:
            return None
        if self._store is None:
            with self._lock:
                if self._store is None:                     # re-check under the lock (this thread's slot)
                    try:
                        paths.ensure_data_dir()             # the store sits in the data dir
                        self._store = Store(self._resolve_path(), project_name=PROJECT_NAME)
                    except (AnnealMemoryError, OSError):     # best-effort: a failed open disables the layer
                        self.available = False               # don't retry every read on a broken store
                        self._store = None
        return self._store

    def close(self) -> None:
        """Close THIS thread's connection (``_store`` is thread-local + ``check_same_thread=True``, so a
        thread can only close its own). Each thread that opened a connection must close its own; the
        daemon thread does so in ``coach_daemon.run``'s exit, a test on its single thread via the context
        manager / explicit close. A connection whose owning thread never calls close is reclaimed at
        process exit (its WAL recovers on the next open)."""
        if self._store is not None:
            try:
                self._store.close()
            except (AnnealMemoryError, OSError):
                pass
            self._store = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # ---------------- write: reads -> episodes (the citable evidence) ----------------
    def record_read_episode(self, sample: dict):
        """Record one committed read as an anneal episode; return its 8-char id, or None (no-op / error).

        Best-effort BY CONTRACT: episode capture is enrichment over the durable calibration trend the
        caller already persisted, so it must NEVER affect the read's success. We therefore catch
        ``Exception`` broadly — the one place a bare catch is correct (a bug in _episode_content or an
        unexpected anneal internal must not turn a successful, already-saved read into a caller error;
        L1 MED-2). Content is archetype-free (§3f); the (session_id, hand_no, n) join key is ALSO stored
        as structured ``metadata`` (not only in the prose) so P4b's decision-point recall can query it
        without regexing free text (L2 rec)."""
        store = self._ensure()
        if store is None or not isinstance(sample, dict):
            return None
        # the SAME validated join key the prose uses (never a type-mismatched / fabricated key in metadata).
        sid, hand_no, n = _join_key(sample)
        if sid and hand_no is not None and n is not None:
            meta = {"session_id": sid, "hand_no": hand_no, "n": n}
        elif sid:
            meta = {"session_id": sid}
        else:
            meta = None
        try:
            with self._lock:
                ep = store.record(_episode_content(sample), EpisodeType.OBSERVATION,
                                  source="read", metadata=meta)
            return getattr(ep, "id", None)
        except Exception:                        # best-effort: never break a read that already landed
            return None

    # ---------------- the leak-naming graduation path (coach-driven) ----------------
    def prepare_leak_wrap(self, recover_abandoned: bool = False):
        """Open a leak-naming wrap: returns anneal's compression package + ``wrap_token`` (normalized to a
        dict) for the coach to name/graduate leaks from the recent read-episodes, or None on a no-op /
        error.

        Single-writer lifecycle (codex L3 HIGH-1 / MED-2): anneal PERSISTS the wrap lock in the db, so a
        coach that prepared a wrap then crashed without committing/cancelling leaves it wrap-locked across
        restarts — bricking graduation while ``available`` still reads True (the dead-store failure the
        moat exists to prevent). The earlier fix auto-cancelled on every WrapInProgressError — but that
        OVER-CORRECTED: a second caller (a retry, a double-click, the P4b coach thread, or a plain prepare
        after a commit-rejected-but-RETRYABLE wrap) would then cancel a wrap the first caller legitimately
        holds. So recovery is now EXPLICIT: by default a WrapInProgressError returns
        ``{"status": "wrap_in_progress", "wrap_token": None}`` (the seam surfaces ok:False — never destroys
        the open wrap). Pass ``recover_abandoned=True`` for the DELIBERATE reclaim of a KNOWN-abandoned
        wrap: that cancels the open snapshot (no committed leak is lost) and retries once.
        WrapInProgressError ⊂ AnnealMemoryError, so its clause MUST precede the broad one below."""
        store = self._ensure()
        if store is None:
            return None
        with self._lock:
            try:
                return _as_dict(prepare_wrap(store))
            except WrapInProgressError:
                if not recover_abandoned:
                    return {"status": "wrap_in_progress", "wrap_token": None}
                try:
                    store.wrap_cancelled()       # DELIBERATE reclaim — discards only the in-progress snapshot
                    return _as_dict(prepare_wrap(store))
                except (AnnealMemoryError, ValueError, OSError):
                    return None
            except (AnnealMemoryError, ValueError, OSError):
                return None

    def commit_leaks(self, continuity_text: str, wrap_token: str,
                     affect_tag: str | None = None, affect_intensity: float | None = None):
        """Save the coach-named leaks doc through anneal's FULL immune system (citation-validated
        graduation, demotion of ungrounded/sycophantic leaks, Hebbian links). Returns the validation
        result dict, or None on error. ``wrap_token`` must come from a preceding ``prepare_leak_wrap``."""
        store = self._ensure()
        if store is None:
            return None
        affect = None
        if affect_tag is not None:
            try:
                affect = AffectiveState(tag=affect_tag,
                                        intensity=float(affect_intensity if affect_intensity is not None else 0.5))
            except (ValueError, TypeError):
                affect = None
        with self._lock:
            try:
                return validated_save_continuity(store, continuity_text, wrap_token=wrap_token,
                                                 affective_state=affect)
            except (AnnealMemoryError, OSError):
                return None
        # NOTE: a ValueError here is NOT swallowed — it signals either a MALFORMED wrap (bad/empty doc,
        # no wrap in progress, wrong token) OR anneal's catastrophic-shrink POLICY rejection (a commit
        # that would collapse a protected section below its retained-mass floor; L1 MED-1). Both are let
        # propagate (the ``with`` releases the lock on the way out) so the SEAM — the transport boundary —
        # normalizes them into the leak API's {ok, reason} shape (see seam.commit_leaks), rather than
        # silently dropping the operator's leaks.

    def cancel_leak_wrap(self) -> bool:
        """Release an in-progress leak wrap — discard the snapshot (no COMMITTED leak is lost, only the
        uncommitted prepare). The seed driver calls this when authoring fails, so a prepared-but-never-
        committed wrap can't brick the next pass (the single-writer dead-store failure). Returns True iff
        a wrap was ACTUALLY in progress and got cancelled, False on no-op / nothing-in-progress / error —
        ``Store.wrap_cancelled`` is a silent no-op when idle, so we check ``load_wrap_snapshot`` first to
        keep the return value honest (codex L3 LOW). Safe to call blind: the guard swallows any error."""
        store = self._ensure()
        if store is None:
            return False
        with self._lock:
            try:
                if store.load_wrap_snapshot() is None:   # nothing in progress → nothing to cancel
                    return False
                store.wrap_cancelled()
                return True
            except (AnnealMemoryError, ValueError, OSError):
                return False

    # ---------------- read: the graduated leaks + evidence recall ----------------
    def get_leaks(self):
        """The current graduated-leaks doc (anneal's continuity file) — the leaks the coach recalls at
        the decision point. Returns the markdown string, or None (no leaks yet / unavailable / error).
        On a SWALLOWED error sets ``last_load_error`` so the seam can tell a healthy-empty store from a
        broken read (complement L3 MED-2). Catches broadly so a read can never 500 ``/api/leaks`` — incl.
        a future sqlite cross-thread error from the P4b daemon path (codex L3 HIGH-2 defensive)."""
        store = self._ensure()
        if store is None:
            return None
        with self._lock:
            try:
                self.last_load_error = False
                return store.load_continuity()
            except Exception:                    # broad: a read must never crash /api/leaks
                self.last_load_error = True
                return None

    def new_read_count(self) -> int:
        """The count of un-graduated read-episodes (episodes since the last completed wrap) — the
        threshold signal the seed driver gates on. A PURE READ with NO wrap side effect, so the driver
        only opens a wrap when it's actually going to graduate (not a prepare-just-to-count probe that
        would churn the single-writer lock). 0 on no-op / unavailable / any error (best-effort, like
        every store op here)."""
        store = self._ensure()
        if store is None:
            return 0
        with self._lock:
            try:
                return len(store.episodes_since_wrap())
            except Exception:                    # broad: a count failure must never break the driver
                return 0

    def recall_leaks(self, keyword: str | None = None, limit: int = 10):
        """Recall read-episodes (the leak evidence) by keyword. Returns a list of episodes, or None.
        The decision-point-targeted recall-injection is P4b; this is the raw evidence read-side. Catches
        broadly for the same never-crash-the-caller reason as get_leaks (codex L3 HIGH-2 defensive)."""
        store = self._ensure()
        if store is None:
            return None
        with self._lock:
            try:
                result = store.recall(keyword=keyword, limit=limit)
                return list(getattr(result, "episodes", []))
            except Exception:
                return None

    def recall_relevant_leaks(self, query: str, *, limit: int = 2, min_overlap: int = 2):
        """P4b-i — the moat READ half: surface the operator's GRADUATED leaks most relevant to the
        current spot, for injection into the coach's decision/review context.

        Given a spot-derived ``query`` (street + observed villain behavior + the hero's spot — built
        from PLAYER-VISIBLE signals only, never the hidden archetype), parse the named leaks out of
        the continuity doc's ``## Patterns`` working set (where commit_leaks graduates them, 1x->2x->3x)
        and return the top ``limit`` that share at least ``min_overlap`` distinctive keywords with the
        spot. Each entry is ``{"name", "level", "summary"}``. Returns ``[]`` when nothing clears the bar
        ("surface nothing rather than noise" — anneal's precision-bias philosophy at the recall point),
        or on no-op / unavailable / any error: recall is pure enrichment over the coach prompt, so a
        miss or a failure must degrade to "no leak section", never break the coach.

        ARCHETYPE-FREE by construction (§3f): the leaks graduated from archetype-free episodes, so the
        recalled summaries carry no villain ``type`` — the moat memory structurally cannot spoil a read.
        We recall from the working-set (continuity-doc) tier, NOT a CrystalStore: that is where leaks
        currently graduate, and crystallizing out is a later slice the corpus does not yet need.

        Best-effort + never-raise + thread-guarded (the lazy-open + per-op serialization), like every
        other store op here."""
        store = self._ensure()
        if store is None or not isinstance(query, str) or not query.strip():
            return []
        try:
            # extract_keywords / extract_pattern_summaries are anneal's pure string helpers (tokenizer +
            # doc parser, NO Store/SQLite access), so calling them outside the lock is thread-safe — only
            # load_continuity (the actual DB read) is locked. [complement L3 LOW-2: assumption made explicit]
            query_kw = set(extract_keywords(query))
            if len(query_kw) < min_overlap:      # too thin a query to match anything meaningfully
                return []
            query_streets = _STREETS & query_kw
            with self._lock:
                doc = store.load_continuity()
            if not doc:
                return []
            ranked = []
            for s in extract_pattern_summaries(doc):
                # PER-ENTRY guard (complement L3 MED-1): one malformed leak line (a bad level/attr from a
                # pre-schema-change doc) must skip ONLY that entry — not crash the loop and let the outer
                # except below zero out the WHOLE recall (per-request blast radius). Degrade per-entry.
                try:
                    leak_kw = _leak_keywords(s.name, s.summary)
                    if _street_excluded(query_streets, leak_kw):   # hard gate: wrong street → never this leak
                        continue
                    score, overlap = _score_leak(query_kw, leak_kw, s.level)
                    if overlap >= min_overlap:    # >= min_overlap SITUATIONAL words (frame nouns stripped)
                        # carry `overlap` (the situational-match count) so each SURFACE can set its own
                        # confidence bar: the LLM coach shows all (it judges fit + can skip); the
                        # facts-only floor — which can't judge — shows only strong matches (coach_watch).
                        ranked.append((score, {"name": s.name, "level": s.level,
                                               "summary": s.summary, "overlap": overlap}))
                except Exception:                 # skip this leak, keep recalling the rest
                    continue
            # highest score first; cap to the decision-point nudge size (the relevant leak, NOT the
            # whole get_leaks list — the explicit P4b design line).
            ranked.sort(key=lambda r: r[0], reverse=True)
            return [leak for _score, leak in ranked[:max(0, limit)]]
        except Exception:                        # best-effort: recall is enrichment, never break a coach
            return []
