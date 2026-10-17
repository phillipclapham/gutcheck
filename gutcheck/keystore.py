"""gutcheck BYO-API-key store — the metered coach worker's credential backing.

The subscription/OAuth coach (``call_claude`` / ``call_codex``) needs no key — it rides the
Claude Code / Codex login. The METERED workers (``call_anthropic_api`` / ``call_openai_api``,
settings providers ``anthropic_api`` / ``openai_api``) are for the operator who has no Claude/Codex
subscription and brings their own API key, billed per call. That key lives HERE, NOT in
``settings.json`` (Anthropic is the worked example below; OpenAI follows the same disciplines):

- **A credential, never a preference.** ``settings.json`` round-trips through the settings
  menu's read API; a key must never appear in any read payload. So it gets its own file,
  written ``0600`` (owner-only), and the read surface exposes only ``{set: bool, hint: "…1234"}``
  — never the key itself.
- **Explicit-only, never the ambient ``ANTHROPIC_API_KEY``.** The official ``anthropic`` SDK
  auto-reads ``ANTHROPIC_API_KEY`` from the process environment — which, for the in-process
  coach daemon thread, is whatever launched ``server.py`` (on flow's own machine, possibly a
  Claude *subscription/work* key). Reading that ambient var would silently bill the wrong
  account. So this store reads ONLY ``GUTCHECK_ANTHROPIC_API_KEY`` (an explicit gutcheck
  opt-in) or the secrets file, and ``call_anthropic_api`` ALWAYS passes ``api_key=`` explicitly
  so the SDK can never fall back to the ambient key. Money gets explicit config, not implicit.

Resolution order for ``get_api_key`` (first hit wins): the ``GUTCHECK_<PROVIDER>_API_KEY`` env
var (lets a user keep the key off disk entirely), then the secrets file. Disciplines mirror
``settings.py``: reads tolerate a missing/corrupt file (→ no key, the worker degrades to the
facts floor, never crashes); the file write is atomic (temp + ``os.replace``) and created with
mode ``0600``.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading

from . import paths

# Providers that take a BYO metered key: Anthropic (call_anthropic_api) and OpenAI (call_openai_api).
# Both are metered BYO-key workers for an operator with no Claude/Codex subscription; the per-provider
# env var is GUTCHECK_<PROVIDER>_API_KEY (e.g. GUTCHECK_OPENAI_API_KEY). The store is fully
# provider-parameterized — adding a vendor here is the only keystore change a new metered worker needs.
_PROVIDERS = ("anthropic", "openai")

# Serializes every public read AND write across threads (L3 nemotron + codex). The coach daemon
# THREAD reads the key per request while the HTTP request thread writes it via POST /api/secrets,
# so without this they can overlap. Two hazards it closes: (1) set/clear's load→modify→save is a
# read-modify-write — two concurrent writers would lose an update; the lock makes it atomic. (2) on
# Windows, os.replace() over a path another thread has open for read raises (POSIX is fine); the lock
# guarantees no read is in flight during the replace, so the credential store is thread-safe on EVERY
# platform, not just POSIX — a structural guard, not a "happens to work on macOS" assumption.
# (`settings.py` ships the same un-locked pattern and notes the same caveat; locking it is a separate
# follow-up, out of this slice — but a CREDENTIAL store earns the stronger bar now.)
_LOCK = threading.Lock()


def _env_var(provider: str) -> str:
    """The explicit per-provider env override, e.g. anthropic → GUTCHECK_ANTHROPIC_API_KEY.
    Deliberately NOT the SDK's own ``ANTHROPIC_API_KEY`` — see the module docstring (we never
    read the ambient key, to avoid billing a shared/work account)."""
    return f"GUTCHECK_{provider.upper()}_API_KEY"


def _load() -> dict:
    """The on-disk secrets map (provider → key). Never raises: a missing file, unreadable file,
    malformed JSON, or non-object document all fall back to an empty map — the metered worker then
    has no key and degrades to the facts floor rather than taking the app down over a cosmetic read."""
    try:
        raw = json.loads(paths.secrets_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return {}
    return raw if isinstance(raw, dict) else {}


def get_api_key(provider: str = "anthropic") -> str | None:
    """The effective key for ``provider``: the ``GUTCHECK_<PROVIDER>_API_KEY`` env var if set
    (keeps the key off disk), else the secrets file, else None. A blank/whitespace value (env or
    file) is treated as unset — a present-but-empty key would otherwise make the SDK fail
    confusingly instead of cleanly degrading. Never raises."""
    env = os.environ.get(_env_var(provider))
    if env and env.strip():
        return env.strip()
    with _LOCK:                                   # serialize the file read vs a concurrent _save
        val = _load().get(provider)
    if isinstance(val, str) and val.strip():
        return val.strip()
    return None


def key_status(provider: str = "anthropic") -> dict:
    """The SAFE read surface for the UI: ``{"set": bool, "hint": "…1234" | "", "source": "env"|"file"|""}``.
    NEVER returns the key. ``hint`` is the last 4 chars (enough to recognise which key is loaded,
    useless to an attacker). ``source`` distinguishes an env-provided key (which ``clear`` can't
    remove) from a file-stored one, so the UI can tell the operator the truth."""
    env = os.environ.get(_env_var(provider))
    if env and env.strip():
        k = env.strip()
        return {"set": True, "hint": _hint(k), "source": "env"}
    with _LOCK:                                   # serialize the file read vs a concurrent _save
        val = _load().get(provider)
    if isinstance(val, str) and val.strip():
        return {"set": True, "hint": _hint(val.strip()), "source": "file"}
    return {"set": False, "hint": "", "source": ""}


def _hint(key: str) -> str:
    """The last-4 recognition hint, prefixed with an ellipsis. Short keys (≤4 chars — never a real
    API key) hint nothing, so a stray value can't leak in full."""
    return f"…{key[-4:]}" if len(key) > 4 else ""


def set_api_key(key: str, provider: str = "anthropic") -> None:
    """Persist ``key`` for ``provider`` to the secrets file (mode 0600), preserving other providers'
    keys. Raises ``ValueError`` on an unknown provider or a blank key (the HTTP boundary turns that
    into a 400) — storing an empty key is the footgun this guards. NOTE: an env-provided key still
    wins over the file (see get_api_key); the UI's status ``source`` surfaces that."""
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    if not isinstance(key, str) or not key.strip():
        raise ValueError("api key must be a non-empty string")
    with _LOCK:                                   # atomic read-modify-write (no lost update)
        data = _load()
        data[provider] = key.strip()
        _save(data)


def clear_api_key(provider: str = "anthropic") -> None:
    """Remove ``provider``'s key from the secrets file. Idempotent (no-op if absent). Does NOT and
    CANNOT unset a ``GUTCHECK_<PROVIDER>_API_KEY`` env var — if one is set, that key stays effective
    (key_status will report source "env"); honest, and the UI says so. Raises on an unknown provider."""
    if provider not in _PROVIDERS:
        raise ValueError(f"unknown provider: {provider}")
    with _LOCK:                                   # atomic read-modify-write (no lost update)
        data = _load()
        if provider in data:
            del data[provider]
            _save(data)


def _save(obj: dict) -> None:
    """Atomic, owner-only file write: serialize to a UNIQUE temp in the data dir, created mode 0600 by
    ``mkstemp`` (which uses O_CREAT|O_EXCL internally — symlink-safe, never reuses/follows a planted
    name), then ``os.replace`` onto the real path. The same-dir rename is atomic on POSIX and carries
    the temp's 0600 mode onto the final file (so even overwriting a world-readable secrets.json ends
    0600). A reader and a crash both see either the whole old file or the whole new one, never a
    half-written key. On any failure the temp is unlinked, never orphaned.

    Why mkstemp (a fresh random name) over a fixed ``secrets.json.<pid>.tmp`` (L2 LOW): a fixed pid-suffixed
    temp orphaned by a crash makes the next O_EXCL open raise FileExistsError — an OSError the /api/secrets
    handler doesn't catch, surfacing as a cryptic path-disclosing 400 that the operator can't clear without
    sweeping the orphan. mkstemp picks a fresh random name (62^6 space) per write, so a crash-orphaned temp
    is never the name THIS write reuses → the FileExistsError failure class is removed structurally, not
    caught. The ``secrets.<rand>.tmp`` shape still matches reset_data()'s ``*.tmp`` sweep, so a crash orphan
    is still cleaned on the next --reset."""
    d = paths.ensure_data_dir()
    path = paths.secrets_path()
    fd, tmp = tempfile.mkstemp(prefix="secrets.", suffix=".tmp", dir=str(d))   # 0600 + O_EXCL + unique name
    try:
        with os.fdopen(fd, "wb") as f:                # takes ownership of fd; closes it on exit
            f.write(json.dumps(obj, indent=2).encode("utf-8"))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
