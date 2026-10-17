"""gutcheck hand-history store — incremental per-hand persistence.

The engine (game.py) records EVERY completed hand here as it settles — not just the
handful of notable hands the get-up report keeps — so the coach surface, and any
transport over the seam (server.py's /api/history today, the MCP adapter next), can
pull a clean, queryable, cross-session hand stream.

This is the CANONICAL OBJECT's own store. It is NOT the coach channel: coach.py /
coach.db carries coaching requests (a consumer's plumbing), and the engine deliberately
does not write its history there — that would couple the canonical record to one
consumer's transport. Kept separate so no transport owns the history (the seam-primary
law, §4.0 of the productization design).

Each row is fully replayable: (seed, start_stacks, log) reconstructs the hand via
engine.Hand — the same shape the session report's compact hands carry.

Half-open watermark: ``get_history(since)`` returns hands with ``id > since`` plus the
new watermark, so a consumer polls forward with the last id it saw. A harmless re-scan
on overlap beats a silent gap (watermark_is_a_start_cutoff_not_a_completion_time).

Stdlib sqlite3 only (zero-dep, like coach.py). WAL so the single-threaded game server
writes while any reader (a future live consumer / the MCP adapter) reads concurrently.
"""

from __future__ import annotations

import contextlib
import json
import sqlite3
import time
from pathlib import Path

from . import paths

# Fields stored as JSON text and re-parsed on read.
_JSON_FIELDS = ("start_stacks", "assign", "log", "board")

# The per-hand WRITE rides the single-threaded request thread (game.py _accumulate_session), so it
# must never freeze a settling hand if history.db is briefly locked (a WAL checkpoint, or a future
# second writer). A SHORT busy_timeout bounds the worst-case stall: under contention the write
# degrades to a logged miss — the hand still lives in sess["hands"] + the get-up report (the money
# truth), so a lost row degrades history, not play. Reads keep the default 5s (WAL readers don't
# block on the writer anyway). history.db is SINGLE-WRITER by design (only the server process writes
# it); this bound is the safety margin for when that assumption is stressed (e.g. the P2 MCP adapter).
_WRITE_BUSY_TIMEOUT_MS = 1000

# Paths whose schema/WAL this process has already ensured — keeps the once-per-process
# PRAGMA off the per-write path (CREATE TABLE IF NOT EXISTS still runs every connection
# so the store SELF-HEALS after reset_data() deletes history.db, across processes too).
_INITIALIZED: set[str] = set()


def _conn(path: Path | None = None, busy_timeout_ms: int = 5000) -> sqlite3.Connection:
    if path is None:
        path = paths.history_db()
    # Self-heal the data dir: a reader/writer can run before the server ever calls
    # ensure_data_dir(); connect() on a missing parent raises "unable to open database file".
    path.parent.mkdir(parents=True, exist_ok=True)
    busy_timeout_ms = int(busy_timeout_ms)
    c = sqlite3.connect(str(path), timeout=busy_timeout_ms / 1000)   # the writer bounds this short (see record_hand)
    c.row_factory = sqlite3.Row
    c.execute(f"PRAGMA busy_timeout={busy_timeout_ms}")   # how long to wait out a lock before erroring
    c.execute(
        """CREATE TABLE IF NOT EXISTS hands (
               id           INTEGER PRIMARY KEY AUTOINCREMENT,
               session_id   TEXT    NOT NULL,
               hand_no      INTEGER NOT NULL,
               seed         INTEGER NOT NULL,
               start_stacks TEXT    NOT NULL,   -- JSON: per-seat starting stacks
               assign       TEXT    NOT NULL,   -- JSON: seat -> player index
               log          TEXT    NOT NULL,   -- JSON: the action log (replays the hand)
               pot          INTEGER NOT NULL,
               hero_net     INTEGER NOT NULL,
               board        TEXT    NOT NULL,   -- JSON: community cards
               street       TEXT    NOT NULL,
               ts           REAL    NOT NULL,
               UNIQUE(session_id, hand_no)      -- one row per hand; a replayed settle is a no-op
           )"""
    )
    key = str(path)
    if key not in _INITIALIZED:
        # WAL is a persistent db property → set once per process per path. CHECK the result: on a
        # network mount (NFS/SMB) or a read-only fs it silently falls back to "delete" (rollback
        # journal), which breaks the WAL reader-doesn't-block-writer assumption the whole store rests
        # on — so fail LOUD rather than proceed unaware (invisible_infrastructure_failure). [complement L3]
        mode = c.execute("PRAGMA journal_mode=WAL").fetchone()
        if not mode or str(mode[0]).lower() != "wal":
            import sys
            got = mode[0] if mode else None
            print(f"[history] WARNING: WAL unavailable for {path} (journal_mode={got!r}) — reader/"
                  "writer concurrency degraded; is the data dir on a network or read-only mount?",
                  file=sys.stderr)
        _INITIALIZED.add(key)
        # NOTE (L3 LOW, latent): _INITIALIZED is keyed per-process + per-PATH, not per-file-identity. If
        # another process DELETES + recreates the db file while this one stays alive, this gate skips
        # re-running WAL on the new file (it would default to DELETE journal). Can't happen today —
        # --reset is a startup flag (fresh process) and /api/reset doesn't delete history.db; coach.db
        # carries the identical accepted trade-off. Revisit (track file identity, or run WAL every
        # _conn) if P2 ever adds a runtime reset with a long-lived reader attached.
    c.commit()
    return c


def record_hand(session_id: str, record: dict, path: Path | None = None) -> bool:
    """Persist one finished hand (the dict game.py builds in _accumulate_session). Returns
    True if a new row was written, False if this (session_id, hand_no) was already stored.

    Idempotent on (session_id, hand_no) via INSERT OR IGNORE — the FIRST write wins: a duplicate/
    replayed settle is a true no-op and can never crash, overwrite the stored row, or write a dupe.
    (Settle runs once per hand, so first-vs-last only matters as a guard against a future re-record
    bug; switch to ON CONFLICT DO UPDATE if last-write-wins is ever wanted.) The caller treats a
    write failure as non-fatal: the hand already lives in the in-memory session + the get-up report
    (the money truth), so a lost row degrades history, not play — hence the SHORT write busy_timeout
    (it would rather miss a row under contention than freeze a settling hand)."""
    if not session_id:
        raise ValueError("record_hand needs a session_id")
    with contextlib.closing(_conn(path, busy_timeout_ms=_WRITE_BUSY_TIMEOUT_MS)) as c, c:
        cur = c.execute(
            """INSERT OR IGNORE INTO hands
                   (session_id, hand_no, seed, start_stacks, assign, log,
                    pot, hero_net, board, street, ts)
               VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
            (str(session_id), int(record["hand_no"]), int(record["seed"]),
             json.dumps(record["start_stacks"]), json.dumps(record["assign"]),
             json.dumps(record["log"]), int(record["pot"]), int(record["hero_net"]),
             json.dumps(record["board"]), str(record["street"]), time.time()),
        )
        return cur.rowcount > 0


def get_history(since: int = 0, limit: int = 500, path: Path | None = None,
                session_id: str | None = None, latest: bool = False) -> dict:
    """Replayable hands with ``id > since`` (half-open), oldest first, capped at ``limit``,
    plus the new ``watermark`` to poll forward from.

    ``session_id`` optionally scopes the stream to ONE session ("replay session X"); absent/empty
    means the full cross-session stream (the prior behavior). The watermark stays the global
    AUTOINCREMENT id either way — ids are monotonic ACROSS sessions, so the half-open ``id > since``
    cursor composes cleanly with the filter (it only narrows WHICH rows in that id range return; a
    scoped consumer that resumes from its last watermark still skips nothing and never overlaps).

    ``latest`` flips the window to the MOST-RECENT ``limit`` hands (the coach's "what did I just
    play" need) instead of the oldest ``limit``: the rows are selected ``id DESC LIMIT`` then
    reversed back to ascending for presentation, so the caller still reads them oldest→newest. The
    returned ``watermark`` is the highest id in the window (= the newest hand), unchanged in meaning.
    NOTE: ``latest=True`` is a RECENT-WINDOW, not a gap-free forward cursor — it deliberately skips
    everything older than the window (with 97 hands, ``latest, limit=20`` returns hands 78-97 and a
    watermark of 97; a consumer that polls ``since=97`` next correctly sees only NEW hands). A
    cursor consumer that must never skip a row keeps ``latest=False`` and pages forward from ``since``.

    The watermark is the id of the LAST returned row (or ``since`` unchanged when empty), so a
    consumer that hit the limit simply polls again from there — no row is skipped, overlap is
    impossible because the cutoff is strict (``id > since``) — WITHIN ONE DB GENERATION.

    Cross-generation caveat (L3, 3-way converged): ``gutcheck --reset`` (paths.reset_data) DELETES
    history.db, so AUTOINCREMENT restarts at 1. A consumer that PERSISTED a watermark across that reset
    (e.g. since=80) would poll ``id > 80`` against the fresh db and silently miss new rows 1..80. No
    live consumer persists watermarks today (the HTTP client's ``since`` is in-memory, reset on refresh;
    --reset is a startup flag, so the server is a fresh process anyway). The P2 MCP adapter is the first
    persistent consumer — it needs a generation/epoch token in the cursor (see the productization design
    §6); add it WITH the adapter, not speculatively here."""
    try:
        since = int(since)
    except (TypeError, ValueError):
        since = 0
    if since < 0:
        since = 0
    try:
        limit = max(1, int(limit))
    except (TypeError, ValueError):
        limit = 500
    # Build the query with the optional session scope. Parameterized — session_id is a stored TEXT
    # value, never interpolated. An absent/empty session_id leaves the query at the cross-session form.
    sql = "SELECT * FROM hands WHERE id > ?"
    args: list = [since]
    if session_id is not None and str(session_id) != "":
        sql += " AND session_id = ?"
        args.append(str(session_id))
    # latest → grab the most-recent `limit` (id DESC) and reverse below to present oldest-first;
    # else the original oldest-first forward window. The LIMIT binds the same either way.
    sql += f" ORDER BY id {'DESC' if latest else 'ASC'} LIMIT ?"
    args.append(limit)
    with contextlib.closing(_conn(path)) as c:
        rows = c.execute(sql, args).fetchall()
    if latest:
        rows = list(reversed(rows))   # DESC fetch → ascending presentation; watermark stays hands[-1].id (newest)
    hands = []
    for r in rows:
        d = dict(r)
        for f in _JSON_FIELDS:
            try:
                d[f] = json.loads(d[f])
            except (json.JSONDecodeError, TypeError):
                pass  # leave the raw text rather than drop the row — a corrupt blob is still a hand
        hands.append(d)
    watermark = hands[-1]["id"] if hands else since
    return {"hands": hands, "watermark": watermark, "count": len(hands)}


def get_hand(session_id: str, hand_no: int, path: Path | None = None) -> dict | None:
    """Point lookup of ONE stored hand by (session_id, hand_no) → the replayable record
    (seed/start_stacks/assign/log + pot/result), or None. The UNIQUE(session_id, hand_no) index
    makes this exact. Unlike the get-up report (which keeps only the top-5 notable hands), the
    canonical store has EVERY played hand — so the coach can reconstruct ANY decision, MID-SESSION
    (history.db is written on hand-end, the report only at get-up). Used by the modeled−exact read
    reconciliation; the returned dict is the same shape coach_review_context replays from `rec`."""
    if not isinstance(hand_no, int) or isinstance(hand_no, bool) or not session_id:
        return None
    with contextlib.closing(_conn(path)) as c:
        row = c.execute(
            "SELECT * FROM hands WHERE session_id = ? AND hand_no = ?",
            (str(session_id), int(hand_no)),
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    for f in _JSON_FIELDS:
        try:
            d[f] = json.loads(d[f])
        except (json.JSONDecodeError, TypeError):
            pass  # a corrupt blob is still a hand; let the caller's replay-validation reject it
    return d
