"""gutcheck coach channel — a tiny local SQLite request/response bus.

Decouples the game app (server.py, which WRITES coaching requests) from the coach
worker (a warm CC session via coach_watch.py, or a headless daemon next slice, which
READS pending requests and WRITES prose responses). A plain inbox pattern, purely
local and zero-dep (stdlib sqlite3 only). WAL mode so the
single-threaded game server and a separate worker process read/write concurrently
without locking each other out.

The channel carries NO money and NO game authority — only a coaching request (a
frozen snapshot of one spot) and the coach's prose reply. So it deliberately has none
of career.json's durability/idempotency machinery: a lost request is a non-event (just
ask again), so there is nothing here a torn write can corrupt.

Importable API (server + worker) and a CLI (the worker / debugging):

    python -m gutcheck.coach ask --kind decision --context-file spot.json -> prints the new id
    python -m gutcheck.coach pending                              -> JSON list of unanswered
    python -m gutcheck.coach get --id 7                           -> JSON of one request
    python -m gutcheck.coach respond --id 7 --body-file reply.txt -> mark answered
    python -m gutcheck.coach poll --id 7                          -> {"status":..,"response":..}
    python -m gutcheck.coach clean --older-than 24                -> prune old rows
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sqlite3
import sys
import time
from pathlib import Path

from . import paths

# The coach channel db lives in the per-user data dir (paths.py / GUTCHECK_DATA_DIR),
# resolved lazily in _conn() so a test/smoke can redirect it off the real channel.

KINDS = ("decision", "review")
# A coaching request that no worker ever answered is junk after this long — a stale
# "flow is thinking…" the player long since clicked past. Pruned by clean().
_PENDING_TTL_H = 6
# Bound a runaway worker reply: the writer is trusted (your own CC session / a launchd
# daemon), but a giant response would bloat the row and ship whole to the browser on poll.
_MAX_RESPONSE = 16 * 1024
# Paths whose schema this process has already ensured — so the DDL stays OFF the hot
# poll path (poll fires every ~1.2s). WAL is a persistent DB property; the table is
# IF NOT EXISTS — both only need running once per path per process.
_INITIALIZED: set[str] = set()


def _conn(path: Path | None = None) -> sqlite3.Connection:
    if path is None:
        path = paths.coach_db()
    # Self-heal the data dir: the coach CLI and the launchd daemon can run BEFORE the server ever
    # calls ensure_data_dir(), and sqlite.connect() on a missing parent raises "unable to open
    # database file". Ensuring it here covers every coach entry point at the one chokepoint.
    path.parent.mkdir(parents=True, exist_ok=True)
    c = sqlite3.connect(str(path), timeout=5.0)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA busy_timeout=5000")     # wait out a concurrent writer rather than erroring
    # CREATE TABLE runs on EVERY connection (cheap — a no-op when the table exists) so the channel
    # SELF-HEALS after reset_data() deletes coach.db, ACROSS PROCESSES too: every coach op opens a
    # fresh per-call connection (no long-lived fd to a stale inode), so the L3-feared daemon
    # split-brain can't persist. The _INITIALIZED gate below covers only the once-per-process work.
    c.execute(
        """CREATE TABLE IF NOT EXISTS requests (
               id       INTEGER PRIMARY KEY AUTOINCREMENT,
               created  REAL NOT NULL,
               kind     TEXT NOT NULL,
               context  TEXT NOT NULL,                       -- JSON: the frozen spot
               status   TEXT NOT NULL DEFAULT 'pending',     -- 'pending' | 'answered'
               response TEXT,                                -- the coach's prose
               meta     TEXT,                                -- JSON: opaque worker-computed blob
               answered REAL                                 --   (the modeled-equity, for the drill reveal)
           )"""
    )
    key = str(path)
    if key not in _INITIALIZED:
        c.execute("PRAGMA journal_mode=WAL")  # persistent db property; set once per process per path
        # An older coach.db predates `meta` — add it (cheap, one-time). Guarded by a column probe
        # because ADD COLUMN errors if it already exists. The probe→ALTER is NOT atomic across
        # PROCESSES, so a concurrent racer can add the column between our probe and our ALTER —
        # swallow ONLY that duplicate-column error; a real lock/IO/read-only failure must propagate
        # (swallowing it would mark this path _INITIALIZED with no meta column, breaking set_meta).
        if "meta" not in {r[1] for r in c.execute("PRAGMA table_info(requests)")}:
            try:
                c.execute("ALTER TABLE requests ADD COLUMN meta TEXT")
            except sqlite3.OperationalError as e:
                if "duplicate column" not in str(e).lower():
                    raise
        _INITIALIZED.add(key)
    c.commit()
    return c


def _row_to_dict(r: sqlite3.Row) -> dict:
    d = dict(r)
    try:
        d["context"] = json.loads(d["context"])
    except (json.JSONDecodeError, TypeError):
        d["context"] = {}
    return d


def ask(kind: str, context: dict, path: Path | None = None) -> int:
    """Write a pending coaching request. Returns its id (the poll handle)."""
    if kind not in KINDS:
        raise ValueError(f"unknown coach request kind: {kind!r} (expected {KINDS})")
    # closing() closes the connection; the inner `c` is the transaction (commit/rollback).
    with contextlib.closing(_conn(path)) as c, c:
        cur = c.execute(
            "INSERT INTO requests (created, kind, context, status) VALUES (?,?,?,'pending')",
            (time.time(), kind, json.dumps(context)),
        )
        return int(cur.lastrowid)


def pending(path: Path | None = None) -> list[dict]:
    """All unanswered requests, oldest first (the worker's work queue)."""
    with contextlib.closing(_conn(path)) as c:
        rows = c.execute(
            "SELECT * FROM requests WHERE status='pending' ORDER BY id ASC"
        ).fetchall()
    return [_row_to_dict(r) for r in rows]


def get(req_id: int, path: Path | None = None) -> dict | None:
    with contextlib.closing(_conn(path)) as c:
        r = c.execute("SELECT * FROM requests WHERE id=?", (req_id,)).fetchone()
    return _row_to_dict(r) if r else None


def respond(req_id: int, body: str, path: Path | None = None) -> bool:
    """Attach the coach's reply and mark answered. Returns False if the id is unknown
    or already answered (so a double-respond — two workers, a retry — is a no-op, never
    an overwrite of a delivered answer)."""
    body = (body or "")[:_MAX_RESPONSE]
    with contextlib.closing(_conn(path)) as c, c:
        cur = c.execute(
            "UPDATE requests SET response=?, status='answered', answered=? "
            "WHERE id=? AND status='pending'",
            (body, time.time(), req_id),
        )
        return cur.rowcount > 0


def poll(req_id: int, path: Path | None = None) -> dict:
    """The app's read side: {'status': 'pending'|'answered'|'missing', 'response': ...}."""
    row = get(req_id, path)
    if row is None:
        return {"status": "missing", "response": None}
    return {"status": row["status"], "response": row["response"]}


def set_meta(req_id: int, obj: dict, path: Path | None = None) -> bool:
    """Attach an OPAQUE structured blob to a request, separate from the prose response.

    The worker computes the modeled equity exactly ONCE (coach_watch._ensure_modeled_equity
    early-returns if it is already on the ctx) and stores it here, so the app can REVEAL it at the
    gut-commit beat WITHOUT re-running the Monte-Carlo — and so the drill's number is byte-identical
    to the number the coach prose cites. (Byte-identical because it is the SAME computed blob, NOT
    because the MC is deterministic — it isn't; a recompute would wiggle ±1-2%. The single-compute
    early-return is what makes it stable.) The channel stays equity-agnostic: just JSON in, JSON out.

    `meta` (equity) and `response` (prose) are SEPARATE, non-atomic writes BY DESIGN: the equity is the
    ground-truth reveal (authoritative, available the moment it's computed), the prose is best-effort
    narration (may never arrive — no worker up, or one that dies between this write and `respond`). The
    app reveals the number regardless and only narrates when the prose lands.

    WRITE-ONCE (`AND meta IS NULL`): the FIRST worker to compute the equity wins and the number is FROZEN.
    A concurrent OR restarted worker that computes a second, MC-wiggled value cannot overwrite the number
    the browser may already have revealed — so the late/losing worker must REREAD the stored value and use
    THAT for its prose (coach_watch._ensure_modeled_equity does), keeping the drill reveal and the prose
    byte-identical even under two workers. Returns True iff THIS call set it (False if the id is unknown
    OR meta was already set)."""
    with contextlib.closing(_conn(path)) as c, c:
        cur = c.execute("UPDATE requests SET meta=? WHERE id=? AND meta IS NULL",
                        (json.dumps(obj), req_id))
        return cur.rowcount > 0


def poll_meta(req_id: int, path: Path | None = None) -> dict:
    """The app's read side for the opaque blob: {'status': 'ready'|'pending'|'no_equity'|'missing', 'meta': obj|None}.

    'ready'     = the blob is present and parseable (the drill reveals it).
    'pending'   = the request exists, is still UNANSWERED, and has no blob yet — keep polling.
    'no_equity' = TERMINAL: the worker finished (answered) but produced no usable number — equity
                  computation failed, the spot expired, or the stored blob is corrupt. The app stops
                  waiting, reveals "no number", but STILL surfaces the prose (it exists). This split
                  (consulting row['status']) is what stops an answered-but-equity-less row from looking
                  like "still computing" until timeout and then suppressing prose that's already there
                  (codex L3 HIGH).
    'missing'   = no such request (expired/never-was)."""
    row = get(req_id, path)
    if row is None:
        return {"status": "missing", "meta": None}
    raw = row.get("meta")
    if raw:
        try:
            return {"status": "ready", "meta": json.loads(raw)}
        except (json.JSONDecodeError, TypeError):
            return {"status": "no_equity", "meta": None}     # present-but-corrupt → terminal, not "computing"
    # no blob yet: still computing only if the request is unanswered; an answered row will never get one
    if row.get("status") == "answered":
        return {"status": "no_equity", "meta": None}
    return {"status": "pending", "meta": None}


def clean(older_than_hours: float = 24.0, path: Path | None = None) -> int:
    """Prune answered rows older than the cutoff, and pending rows older than the
    pending-TTL (a request no worker ever picked up). Returns rows deleted."""
    now = time.time()
    ans_cut = now - older_than_hours * 3600
    pend_cut = now - _PENDING_TTL_H * 3600
    with contextlib.closing(_conn(path)) as c, c:
        cur = c.execute(
            "DELETE FROM requests WHERE (status='answered' AND COALESCE(answered, created) < ?) "
            "OR (status='pending' AND created < ?)",
            (ans_cut, pend_cut),
        )
        return cur.rowcount


# ---------------- CLI ----------------
def _read_text_arg(direct: str | None, file_arg: str | None) -> str:
    if direct is not None:
        return direct
    if file_arg:
        return Path(file_arg).read_text()
    data = sys.stdin.read()
    if not data:
        raise SystemExit("no body: pass --body, --body-file, or pipe via stdin")
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gutcheck coach channel")
    ap.add_argument("--db", default=None, help="channel db path (default: <data-dir>/coach.db)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    a = sub.add_parser("ask", help="write a pending coaching request")
    a.add_argument("--kind", required=True, choices=KINDS)
    a.add_argument("--context", help="inline JSON context")
    a.add_argument("--context-file", help="JSON context file ('-' for stdin)")

    sub.add_parser("pending", help="list unanswered requests (JSON)")

    g = sub.add_parser("get", help="fetch one request (JSON)")
    g.add_argument("--id", type=int, required=True)

    r = sub.add_parser("respond", help="attach a coaching reply + mark answered")
    r.add_argument("--id", type=int, required=True)
    r.add_argument("--body", help="inline reply text")
    r.add_argument("--body-file", help="reply text file")

    p = sub.add_parser("poll", help="read a request's status/response (JSON)")
    p.add_argument("--id", type=int, required=True)

    cl = sub.add_parser("clean", help="prune old rows")
    cl.add_argument("--older-than", type=float, default=24.0, help="answered-row age cutoff (hours)")

    args = ap.parse_args(argv)
    db = Path(args.db) if args.db else paths.coach_db()

    if args.cmd == "ask":
        if args.context_file == "-" or (args.context is None and not args.context_file):
            raw = sys.stdin.read()
        elif args.context is not None:
            raw = args.context
        else:
            raw = Path(args.context_file).read_text()
        ctx = json.loads(raw)
        print(ask(args.kind, ctx, db))
        return 0
    if args.cmd == "pending":
        print(json.dumps(pending(db), indent=2))
        return 0
    if args.cmd == "get":
        row = get(args.id, db)
        if row is None:
            print(json.dumps({"error": "not found", "id": args.id}))
            return 1
        print(json.dumps(row, indent=2))
        return 0
    if args.cmd == "respond":
        body = _read_text_arg(args.body, args.body_file)
        ok = respond(args.id, body, db)
        print(json.dumps({"id": args.id, "answered": ok}))
        return 0 if ok else 1
    if args.cmd == "poll":
        print(json.dumps(poll(args.id, db)))
        return 0
    if args.cmd == "clean":
        print(json.dumps({"deleted": clean(args.older_than, db)}))
        return 0
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
