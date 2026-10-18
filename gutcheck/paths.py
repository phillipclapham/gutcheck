"""gutcheck writable-state location — the single source of truth.

Persisted state lives in a per-user data dir, NOT the repo:

    ~/.gutcheck/
        career.json          durable bankroll + lifetime ledger
        session.json         the active sit-down (seating, stacks, current hand)
        sessions/<id>.json   frozen session reports (the coach's review source)
        calibration.json     gut-vs-truth read-accuracy log (the felt-sense trend)
        coach.db             the coach request channel (+ -wal / -shm)

Why a data dir (Slice 6 / Stage 1): it stops the app littering the working directory,
and a first-run init + ``gutcheck --reset`` give a clean training slate at the
dev -> training handoff.

Override with the env var ``GUTCHECK_DATA_DIR`` (an absolute or ``~``-relative path). The
pre-rename ``FLOWPOKER_DATA_DIR`` is still honored when the new one is unset, so an older script
that points a smoke run at a scratch dir keeps pointing it there.
The test suite / any integration smoke sets it to a throwaway tmp dir so NO test can
touch the real bankroll — the structural fix for the 2026-06-18 smoke that ran
``Game()`` against the default paths and debited real play-money.

Resolution is LAZY (computed per call, never captured at import) so setting the env var
at any point before a ``Game()`` / coach connection is constructed redirects all state —
import order can't defeat it.

Migration from the private-build name: the first time a process resolves the DEFAULT dir and
``~/.gutcheck`` does not exist but ``~/.flowpoker`` does, the old dir is COPIED (never moved) to
the new one, so the bankroll, reads and leak memory carry over and the old dir stays as a backup.
"""

from __future__ import annotations

import os
import shutil
import sys
import threading
from pathlib import Path

ENV_VAR = "GUTCHECK_DATA_DIR"
LEGACY_ENV_VAR = "FLOWPOKER_DATA_DIR"
_DEFAULT = "~/.gutcheck"
_LEGACY_DEFAULT = "~/.flowpoker"

_migrate_lock = threading.Lock()
_migrate_checked = False


def _override() -> tuple[str, str] | tuple[None, None]:
    for var in (ENV_VAR, LEGACY_ENV_VAR):
        v = os.environ.get(var)
        if v:
            return var, v
    return None, None


class LegacyMigrationError(RuntimeError):
    """The one-time copy from ``~/.flowpoker`` failed. Raised instead of starting on an empty
    ``~/.gutcheck``: an empty dir would exist from then on, so every later start would skip the copy
    and the player's bankroll and reads would sit unseen in the old folder."""


def _migrate_legacy_once(new: Path) -> None:
    """Copy ``~/.flowpoker`` to ``new`` once per process, if ``new`` is absent and the old dir exists.
    Copy into a sibling temp dir, then rename it into place, so a crash mid-copy never leaves a
    half-populated ``new`` that would suppress the migration on the next start. Separate processes
    starting together (server, coach daemon, MCP proxy) each copy; the first rename wins and the
    rest find ``new`` in place and discard their copy."""
    global _migrate_checked
    with _migrate_lock:
        if _migrate_checked:
            return
        old = Path(_LEGACY_DEFAULT).expanduser()
        if new.exists() or not old.is_dir():
            _migrate_checked = True
            return
        tmp = new.with_name(f"{new.name}.migrating-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)   # a same-pid leftover from a killed earlier run
        try:
            shutil.copytree(old, tmp, symlinks=True)
            os.rename(tmp, new)
        except OSError as e:
            shutil.rmtree(tmp, ignore_errors=True)
            if new.is_dir():                     # another process's copy landed first: that is success
                _migrate_checked = True
                return
            raise LegacyMigrationError(
                f"gutcheck could not copy your data from {old} to {new} ({e}). Nothing was changed "
                f"in {old}. Fix the cause and start again, or set {ENV_VAR} to choose a data dir."
            ) from e
        _migrate_checked = True
        print(f"gutcheck: copied your data from {old} to {new} (the old folder is left as a backup)",
              file=sys.stderr)


def data_dir() -> Path:
    """The resolved data dir (``GUTCHECK_DATA_DIR``, else ``FLOWPOKER_DATA_DIR``, else ``~/.gutcheck``).
    Not created, except that resolving the default may run the one-time copy from ``~/.flowpoker``.

    The override MUST be ABSOLUTE: a relative value resolves against the current working directory,
    which differs between the interactive server and the launchd coach daemon — they'd silently
    diverge to different dirs and the coach channel wouldn't connect. Relative values are REJECTED
    (the default ~/.gutcheck is always absolute)."""
    var, val = _override()
    if val is None:
        p = Path(_DEFAULT).expanduser()
        _migrate_legacy_once(p)
        return p
    p = Path(val).expanduser()
    if not p.is_absolute():
        raise ValueError(
            f"{var}={p!r} is relative — set an ABSOLUTE path so the server and coach daemon "
            "resolve to the same directory.")
    return p


def ensure_data_dir() -> Path:
    """Resolve + create the data dir (and its ``sessions/`` subdir). Idempotent; returns it.
    Raises NotADirectoryError with a clear message if the override points at an existing FILE."""
    d = data_dir()
    if d.exists() and not d.is_dir():
        raise NotADirectoryError(
            f"{ENV_VAR}={d} points at a file, not a directory — set it to a directory path.")
    (d / "sessions").mkdir(parents=True, exist_ok=True)
    return d


def session_path() -> Path:
    return data_dir() / "session.json"


def career_path() -> Path:
    return data_dir() / "career.json"


def reports_dir() -> Path:
    return data_dir() / "sessions"


def calibration_path() -> Path:
    return data_dir() / "calibration.json"


def settings_path() -> Path:
    # Runtime user settings (the settings menu's backing store — coaching on/off today).
    # A PREFERENCE, not training data: deliberately NOT touched by reset_data() so a
    # clean-slate `--reset` can't silently flip a cost toggle back to spend.
    return data_dir() / "settings.json"


def secrets_path() -> Path:
    # The BYO-API-key store for the metered coach worker (provider → API key). A CREDENTIAL,
    # not training data: like settings.json it is deliberately NOT touched by reset_data() so a
    # clean-slate `--reset` can't silently wipe the operator's key. Written 0600 by keystore.py
    # (never world-readable); the key never round-trips back over any read API. (The MODULE is
    # keystore.py, not secrets.py — that name would shadow the Python stdlib `secrets` module
    # app-wide, since the repo dir is sys.path[0]; the on-disk file stays the descriptive secrets.json.)
    return data_dir() / "secrets.json"


def coach_db() -> Path:
    return data_dir() / "coach.db"


def history_db() -> Path:
    # The canonical hand-history store (every settled hand, incrementally). Deliberately
    # SEPARATE from coach.db: that is the coach channel (a consumer's plumbing); this is the
    # engine's own replayable record, read by the seam (/api/history, the MCP adapter next).
    return data_dir() / "history.db"


def leaks_db() -> Path:
    # The operator-owned, anneal-backed graduating-leak store (P4 — the moat). Lives in the data
    # dir next to calibration.json; OPTIONAL — only written when the `gutcheck[anneal]` extra is
    # installed (leakstore.py is a graceful no-op otherwise). anneal writes a sidecar continuity
    # file (leaks.continuity.md) + an audit chain (leaks.audit*) alongside it; reset_data() handles all three.
    return data_dir() / "leaks.db"


def reset_data() -> list[str]:
    """Clear persisted state for a clean training slate; returns the items cleared. The bankroll
    ledger is BACKED UP (career.json -> career.reset-backup.json) rather than hard-deleted, so an
    accidental reset is one-level recoverable; the active table, session reports, coach channel,
    and atomic-write temps are removed. Targeted — never an rm -rf of the dir; leaves the dir."""
    d = data_dir()
    removed: list[str] = []
    career = d / "career.json"
    try:
        career.replace(d / "career.reset-backup.json")   # one-level undo for the money ledger
        removed.append("career.json -> career.reset-backup.json")
    except FileNotFoundError:
        pass                                              # nothing to back up (fresh slate)
    try:
        # the read-accuracy trend is the second-most-precious durable artifact (after money) —
        # back it up too rather than hard-delete, so a clean-slate reset is one-level recoverable.
        (d / "calibration.json").replace(d / "calibration.reset-backup.json")
        removed.append("calibration.json -> calibration.reset-backup.json")
    except FileNotFoundError:
        pass
    # the graduated-leak memory (anneal, P4) is precious operator data — back up the db + the leaks
    # doc like calibration so a clean-slate reset is one-level recoverable; the metadata, audit chain,
    # and wal/shm are regenerable and get cleared below for a fresh slate (a fresh store = fresh chain).
    # COUPLING NOTE: the sidecar names below (leaks.continuity.md / .meta.json / -wal / -shm) are
    # anneal's on-disk layout. The dep is bounded (>=0.8.5,<2), and anneal 1.0 adds an audit manifest
    # and lock files beside leaks.audit.jsonl: a sidecar missed here is NOT cosmetic, a surviving
    # manifest remembers the deleted chain and every later audit append refuses. So the audit family is
    # cleared by the `leaks.audit*` glob below (locks excepted), not by a list of names. A blind `leaks.*` glob is
    # intentionally avoided: it would re-match the .reset-backup.* files this very block creates.
    # WAL COMPLETENESS (codex L3 MED-1): the leak db is WAL-mode, so committed rows can live in
    # leaks.db-wal until checkpointed (e.g. after a crashed prior run). Back up the WHOLE db set
    # (db + -wal + -shm) together so the backup isn't a stale, tail-missing copy — never delete the
    # -wal while keeping only the main file. sqlite ties the WAL to the db by name, so the renamed
    # backup set (leaks.reset-backup.db + -wal + -shm) reopens + checkpoints cleanly if restored.
    for src, dst in (("leaks.db", "leaks.reset-backup.db"),
                     ("leaks.db-wal", "leaks.reset-backup.db-wal"),
                     ("leaks.db-shm", "leaks.reset-backup.db-shm"),
                     ("leaks.continuity.md", "leaks.continuity.reset-backup.md")):
        try:
            (d / src).replace(d / dst)
            removed.append(f"{src} -> {dst}")
        except FileNotFoundError:
            pass
    for name in ("career.corrupt.json", "session.json",
                 "coach.db", "coach.db-wal", "coach.db-shm",
                 "history.db", "history.db-wal", "history.db-shm",
                 "leaks.continuity.meta.json"):                              # regenerable
        if (d / name).exists():                                     # training data → cleared, not backed up
            (d / name).unlink(missing_ok=True)            # tolerate a concurrent cleanup (TOCTOU)
            removed.append(name)
    for f in sorted(d.glob("leaks.audit*")):              # anneal's audit chain + manifest: fresh slate =
        if not f.is_file() or f.name.endswith(".lock"):    # fresh chain. Locks stay: they are coordination
            continue                                       # files, and unlinking a held one splits its mutex.
        f.unlink(missing_ok=True)
        removed.append(f.name)
    for tmp in sorted(d.glob("*.tmp")):                   # pid-suffixed atomic-write temps (career/session/
        tmp.unlink(missing_ok=True)                        # settings — settings.json ITSELF is never removed,
        removed.append(tmp.name)                           # but a crash-orphaned settings temp is fair to sweep)
    reps = d / "sessions"
    if reps.is_dir():
        # NOTE: glob "*.tmp" (not "*.json.tmp") — report temps are named <id>.json.<pid>.tmp, which
        # does NOT end in ".json.tmp"; the narrower glob silently missed them (3-way L3 catch).
        for f in sorted(list(reps.glob("*.json")) + list(reps.glob("*.tmp"))):
            f.unlink(missing_ok=True)
            removed.append(f"sessions/{f.name}")
    # the WAL pragma is re-set per process per path; drop this path so a recreated coach.db re-runs
    # it (CREATE TABLE already self-heals on every connection). Narrow catch: only the late-import
    # itself may legitimately fail (avoids the paths<->coach cycle); anything else must surface.
    try:
        from . import coach
    except ImportError:
        pass
    else:
        coach._INITIALIZED.discard(str(coach_db()))
    # Same for the history store: drop the once-per-process WAL guard so a recreated
    # history.db re-runs the pragma (CREATE TABLE already self-heals on every connection).
    try:
        from . import history
    except ImportError:
        pass
    else:
        history._INITIALIZED.discard(str(history_db()))
    return removed
