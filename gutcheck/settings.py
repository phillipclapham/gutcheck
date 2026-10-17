"""gutcheck runtime user settings — the settings menu's backing store.

A tiny JSON document in the data dir (``~/.gutcheck/settings.json``). Today it holds
ONE knob: ``coaching_enabled`` — the cost×latency×fidelity floor. "off" means the
built-in coach serves the honest facts-only stub (band / equity / pot-odds / your read,
NO AI prose) and never spends a ``claude -p`` call. Stat-tailoring and provider/auth
choice (the other two settings-menu rows) land in this same document next.

Two disciplines, both load-bearing:
- **Reads tolerate a missing OR corrupt file** → defaults. A settings store that raised
  on a truncated write would take the whole app down over a cosmetic toggle.
- **The file write is atomic** (temp + ``os.replace``): a reader and a crash both see either
  the whole old file or the whole new one, never a half-written document — and for
  ``coaching_enabled`` that torn read would flip a cost toggle back to *spend*, the exact
  footgun the persistence exists to prevent. NOTE: this is atomic at the FILE level, not an
  atomic read-modify-write — ``_update`` is load→modify→save, so two concurrent writers would
  last-writer-win (a lost update, never corruption). Safe today because the single-threaded
  server serializes POSTs; add a lock here if a second writer is ever introduced.

Location resolves through ``paths`` (honors ``GUTCHECK_DATA_DIR``), so the server and any
worker that shares the data dir see the same settings.
"""

from __future__ import annotations

import json
import os

from . import paths

# The schema + its defaults. Unknown keys read from disk are PRESERVED (forward-compat:
# a newer build's knob survives an older build round-tripping the file), but every known
# key always has a default so a partial file is never missing one.
#   coaching_enabled    — cost toggle (slice a): on = AI coaching, off = zero-claude facts floor.
#   equity_format       — facts-floor stat-tailoring (slice b): "band" (felt-sense only) | "band_pct".
#   show_pot_odds       — facts-floor: show the pot-odds line.
#   show_observed_read  — facts-floor: show the VPIP-inferred read line.
#   coach_provider      — which worker runs the built-in coach: "claude" (claude -p, subscription) |
#                         "codex" (codex exec, OAuth) | "anthropic_api" (metered, BYO Anthropic key) |
#                         "openai_api" (metered, BYO OpenAI key) — both metered keys live in secrets.json
#                         (keystore.py), NOT here (a credential never round-trips through the settings API).
#   api_model           — the metered anthropic_api worker's model (the cost lever): a current Anthropic
#                         model id. Opus = sharpest/priciest, Haiku = cheapest. Operator's call.
#   api_effort          — the anthropic_api worker's effort dial (the fidelity×latency lever): "low" |
#                         "medium" | "high" | "xhigh". Each is a real, correctly-ordered Anthropic effort
#                         level (low = cheapest/fastest read → xhigh = deepest). No "off": OMITTING effort
#                         server-side means DEFAULT (high) effort, not low-cost (L3 codex), so an "off"
#                         that claimed cheapest would be a lie — "low" IS the cheapest meaningful setting.
#                         PER-MODEL gated in the worker: effort 400s on Haiku 4.5 (no-ops there), and "xhigh"
#                         is Opus-only (added in Opus 4.7) — on Sonnet 4.6 the worker degrades xhigh→high.
#   openai_model        — the metered openai_api worker's model (the cost lever): a current GPT tier
#   openai_effort       — the openai_api worker's reasoning_effort dial: "none"|"low"|"medium"|"high"|"xhigh"
#                         (cheapest→deepest). "none" is the fastest/cheapest floor — the gpt-5.4/5.5 era
#                         dropped "minimal" in favor of "none". Unlike the anthropic api_effort (no "off"),
#                         OpenAI's "none" is a REAL explicit level (no reasoning), so the cheapest dial is honest.
#                         PER-VENDOR by design: the two metered vendors have DISTINCT model namespaces
#                         (claude-* vs gpt-*), so switching provider can never strand a Claude id in an
#                         OpenAI call (and vice-versa) — each vendor's choice persists independently.
DEFAULTS: dict = {
    "coaching_enabled": True,
    "equity_format": "band_pct",
    "show_pot_odds": True,
    "show_observed_read": True,
    "coach_provider": "claude",
    "api_model": "claude-opus-4-8",
    "api_effort": "low",
    "openai_model": "gpt-5.5",
    "openai_effort": "low",
}
_EQUITY_FORMATS = ("band", "band_pct")
_PROVIDERS = ("claude", "codex", "anthropic_api", "openai_api")
# The metered-worker model choices surfaced in the UI (current vendor tiers, cheapest→sharpest is the
# operator's cost call). A power-user env override (GUTCHECK_API_MODEL / GUTCHECK_OPENAI_MODEL) can
# still force any id at the worker; these allowlists are what the SETTINGS surface accepts + the dropdowns
# offer. Anthropic = Opus/Sonnet/Haiku; OpenAI = GPT-5.5 (sharpest) / 5.4 (mid) / 5.4-mini (cheap).
_API_MODELS = ("claude-opus-4-8", "claude-sonnet-4-6", "claude-haiku-4-5")
# Anthropic effort levels, cheapest→deepest. "xhigh" is Opus-4.7+ only (Sonnet 4.6 tops out at "high", Haiku
# 4.5 rejects effort entirely). This flat allowlist is the SETTINGS surface (model-blind, like api_model) — any
# model can POST api_effort=xhigh; the ENFORCEMENT (the trust boundary) is the worker's per-model degrade
# (_XHIGH_EFFORT_MODELS), and the UI merely SURFACES xhigh only for Opus. VERIFY on an Anthropic-model change:
# re-confirm xhigh is still Opus-4.7+ (the worker's _XHIGH_EFFORT_MODELS literal is the authority).
_API_EFFORTS = ("low", "medium", "high", "xhigh")
_OPENAI_MODELS = ("gpt-5.5", "gpt-5.4", "gpt-5.4-mini")
# The OpenAI reasoning_effort range, cheapest→deepest. The full set the offered gpt-5.4/5.5 tiers accept
# (verified against the model docs 2026-06-27): "none" is the genuine fastest/cheapest floor — NOT "minimal",
# which the 5.4/5.5 generation REPLACED with "none" (sending "minimal" 400s). "xhigh" is the deepest tier.
# All three offered models (gpt-5.5 / 5.4 / 5.4-mini) accept the identical set, so no per-model gate is needed.
# VERIFY on any gutcheck[api] OpenAI-model change: re-confirm none/xhigh are still accepted by every offered
# model (the set is model-dependent; a future tier could drop one → reasoning_effort → 400 → facts floor).
_OPENAI_EFFORTS = ("none", "low", "medium", "high", "xhigh")


def load() -> dict:
    """The current settings (defaults merged under whatever is on disk). Never raises:
    a missing file, unreadable file, malformed JSON, or non-object document all fall back
    to a fresh copy of DEFAULTS."""
    try:
        raw = json.loads(paths.settings_path().read_text(encoding="utf-8"))
    except (FileNotFoundError, ValueError, OSError):
        return dict(DEFAULTS)
    if not isinstance(raw, dict):
        return dict(DEFAULTS)
    return {**DEFAULTS, **raw}


def _as_bool(val, default: bool) -> bool:
    """Honor only a real bool from disk; any other value (a hand-edited "false" string, a stray int)
    falls back to the default rather than being truthiness-coerced — for these toggles a "false"->True
    coercion is the wrong kind of clever."""
    return val if isinstance(val, bool) else default


def coaching_enabled() -> bool:
    """True = the built-in coach calls claude (falling back to facts on failure); False =
    serve the honest facts-only floor, zero claude spend."""
    return _as_bool(load().get("coaching_enabled"), DEFAULTS["coaching_enabled"])


def stat_prefs() -> dict:
    """The facts-floor stat-tailoring prefs (which grounding facts the floor shows), each validated
    against its type/enum with a fallback to the default — same tolerance as coaching_enabled()."""
    s = load()
    fmt = s.get("equity_format")
    return {
        "equity_format": fmt if fmt in _EQUITY_FORMATS else DEFAULTS["equity_format"],
        "show_pot_odds": _as_bool(s.get("show_pot_odds"), DEFAULTS["show_pot_odds"]),
        "show_observed_read": _as_bool(s.get("show_observed_read"), DEFAULTS["show_observed_read"]),
    }


def coach_provider() -> str:
    """Which worker runs the built-in coach: 'claude' | 'codex' | 'anthropic_api'. A bad on-disk value
    falls back to the default (claude) rather than starving the coach of a valid provider."""
    p = load().get("coach_provider")
    return p if p in _PROVIDERS else DEFAULTS["coach_provider"]


def api_model() -> str:
    """The metered anthropic_api worker's model (the cost lever). A bad on-disk value falls back to the
    default flagship rather than handing the SDK a junk id. (The worker still lets GUTCHECK_API_MODEL
    override this for power use; this getter is the UI-surfaced value.)"""
    m = load().get("api_model")
    return m if m in _API_MODELS else DEFAULTS["api_model"]


def api_effort() -> str:
    """The metered anthropic_api worker's effort dial: 'low' | 'medium' | 'high' | 'xhigh' (the fidelity×latency
    lever). Model-blind (like api_model) — 'xhigh' validates here for any model; the worker degrades it to 'high'
    on a non-Opus model. A bad on-disk value falls back to the default rather than sending the SDK a junk effort."""
    e = load().get("api_effort")
    return e if e in _API_EFFORTS else DEFAULTS["api_effort"]


def openai_model() -> str:
    """The metered openai_api worker's model (the cost lever). A bad on-disk value falls back to the
    default flagship rather than handing the SDK a junk id. (GUTCHECK_OPENAI_MODEL still overrides this
    for power use; this getter is the UI-surfaced value.)"""
    m = load().get("openai_model")
    return m if m in _OPENAI_MODELS else DEFAULTS["openai_model"]


def openai_effort() -> str:
    """The metered openai_api worker's reasoning_effort dial: 'none' | 'low' | 'medium' | 'high' | 'xhigh'
    (cheapest→deepest). A bad on-disk value falls back to the default rather than sending the SDK a junk
    effort (in particular 'minimal' — invalid on the offered gpt-5.4/5.5 tiers — falls back, never 400s)."""
    e = load().get("openai_effort")
    return e if e in _OPENAI_EFFORTS else DEFAULTS["openai_effort"]


def set_coaching_enabled(on: bool) -> dict:
    """Persist the coaching toggle; return the full updated settings. The LENIENT internal/test
    convenience setter (coerces with bool()); the HTTP POST boundary goes through apply_patch instead,
    which validates STRICTLY (rejects a non-bool). Two paths by intent: strict at the wire, lenient in-process."""
    return _update(coaching_enabled=bool(on))


def apply_patch(patch):
    """Validate + persist a partial settings update (the /api/settings POST surface). Returns
    (full_settings, None) on success or (None, error) on a bad key/value. STRICT at this boundary —
    the HTTP layer rejects junk loudly; the read-side getters stay tolerant (strict-in / tolerant-store)."""
    if not isinstance(patch, dict) or not patch:
        return None, "settings patch must be a non-empty object"
    clean = {}
    for k, v in patch.items():
        if k == "coaching_enabled" or k in ("show_pot_odds", "show_observed_read"):
            if not isinstance(v, bool):
                return None, f"{k} must be true or false"
            clean[k] = v
        elif k == "equity_format":
            if v not in _EQUITY_FORMATS:
                return None, f"equity_format must be one of {list(_EQUITY_FORMATS)}"
            clean[k] = v
        elif k == "coach_provider":
            if v not in _PROVIDERS:
                return None, f"coach_provider must be one of {list(_PROVIDERS)}"
            clean[k] = v
        elif k == "api_model":
            if v not in _API_MODELS:
                return None, f"api_model must be one of {list(_API_MODELS)}"
            clean[k] = v
        elif k == "api_effort":
            if v not in _API_EFFORTS:
                return None, f"api_effort must be one of {list(_API_EFFORTS)}"
            clean[k] = v
        elif k == "openai_model":
            if v not in _OPENAI_MODELS:
                return None, f"openai_model must be one of {list(_OPENAI_MODELS)}"
            clean[k] = v
        elif k == "openai_effort":
            if v not in _OPENAI_EFFORTS:
                return None, f"openai_effort must be one of {list(_OPENAI_EFFORTS)}"
            clean[k] = v
        else:
            return None, f"unknown setting: {k}"
    return _update(**clean), None


def _update(**patch) -> dict:
    """Read-modify-write a subset of keys, preserving the rest. Returns the full settings."""
    current = load()
    current.update(patch)
    _save(current)
    return current


def _save(obj: dict) -> None:
    """Atomic file write: serialize to a pid-suffixed temp in the data dir, then ``os.replace``
    onto the real path (same-dir rename is atomic on POSIX). The pid suffix distinguishes
    DIFFERENT-PROCESS writers (a future standalone writer won't collide); same-process writes are
    serialized by the single-threaded server, so they don't race for the temp either — were a second
    in-process writer THREAD added, the shared pid means it would need its own unique temp + a lock.
    ``os.replace`` makes the swap last-writer-win clean rather than corrupting. The filename comes
    from ``settings_path()`` (single source of truth — if it ever moves, this follows). On a failed
    write/replace the temp is unlinked rather than orphaned (the settings dir is the one place
    ``reset_data`` may not routinely sweep)."""
    paths.ensure_data_dir()                       # create the data dir (+ sessions/) if needed
    path = paths.settings_path()
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        tmp.write_text(json.dumps(obj, indent=2), encoding="utf-8")   # always UTF-8, never the locale default
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)               # don't leave a half-written temp behind
        raise
