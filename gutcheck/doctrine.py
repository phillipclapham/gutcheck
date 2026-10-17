"""gutcheck coaching doctrine — the single source every coach surface reads from.

The doctrine CONTENT lives in `doctrine.md` (the canonical, human-editable, swappable pack —
see productization_design.md §3). This module is the thin loader: it resolves the doc relative
to itself (so it works both in an editable install and from the repo) and hands it to each
coach surface — the headless daemon embeds `full()` in its prompt; the MCP server serves
`brief()` in its instructions. Author the doctrine in ONE place (the .md); both paths inherit.

FAIL-SAFE BY CONTRACT: the coach must NEVER crash because the doctrine file is missing or
unreadable (a broken wheel, a bad path, a permissions glitch, OR a non-UTF-8 / corrupted save —
the guard catches both OSError AND UnicodeError, since a decode failure is a ValueError, not an
OSError). Both accessors degrade to a minimal embedded stub that still carries the §3f gate and
the read→action core — so the coach keeps working at reduced fidelity rather than going dark.
The stubs are deliberately SHORT (not a copy of the doc): the full doctrine has one home, the .md.
"""

from __future__ import annotations

import os
import re
from pathlib import Path

# Markers in doctrine.md delimiting the concise read→action principle that length-bounded
# surfaces (the MCP `instructions` block) embed. Author the brief ONCE, inside the doc.
_BRIEF_START = "<!-- BRIEF:START -->"
_BRIEF_END = "<!-- BRIEF:END -->"

# Degraded stubs — used ONLY when doctrine.md can't be read. Short on purpose; they keep the
# coach honest (the gate) and pointed at the load-bearing principle (read→action), no more.
_FALLBACK_FULL = (
    "Coach FROM observed villain behavior only — NEVER name or reveal a hidden villain type "
    "(the operator earns the read). You are a feedback signal for the operator's felt-sense, "
    "not a rule-engine. A read is incomplete until you've named what it forbids: after a read, "
    "give the action it forbids and the action it permits — don't bluff a player who never "
    "folds (value-bet bigger instead). Coach the read's own implication; never grade EV.")
_FALLBACK_BRIEF = (
    "A read is incomplete until you've named what it forbids: translate the read into the "
    "action it forbids and permits (don't bluff a player who never folds — value-bet bigger). "
    "Coach the read's implication, never the action's EV. Speak in behavior, never a type.")


# A loaded doctrine is usable only if it is STRUCTURALLY COMPLETE. A non-atomic / truncated save
# mid-edit can leave valid-UTF-8 but PARTIAL text — and since the daemon reads per-request, a torn
# read would silently serve a doctrine missing the §3f gate or the read→action core. We treat "an
# intact, well-ordered BRIEF region" as the completeness sentinel (it sits late in the doc, in
# Section 3, so its presence means the file wasn't cut short) plus a minimum length. Deliberately
# CONTENT-agnostic: a swappable pack may rephrase the doctrine freely, but must carry that region.
_MIN_DOC_LEN = 500
# Upper bound on the brief that rides in the MCP `instructions` metadata block — a pathologically
# large marked region must not bloat the server instructions; over the cap, degrade to the stub.
_BRIEF_MAX = 1000


# --- doctrine-pack format (productization §6) ------------------------------------------------------
# A pack is markdown + an OPTIONAL leading '---'-fenced frontmatter block. We parse a FLAT key: value
# SUBSET ourselves (no PyYAML — doctrine.py is stdlib-only by contract; nested YAML is intentionally
# unsupported). The frontmatter is METADATA (name/version/read_based/…), STRIPPED from full()/brief()
# so the coach prompt never carries it. `read_based` is a LABEL for curation/discovery, NOT a guard —
# the §3f SEAM is what stops any doctrine from spoiling, so the loader never rejects a pack on it.
_DOCTRINE_ENV = "GUTCHECK_DOCTRINE"   # swap mechanism: an explicit pack path wins over the bundled flagship
_FM_FENCE = "---"
_DEFAULT_META = {"name": "gutcheck Flagship", "version": "0", "read_based": True}


def _doc_path() -> Path:
    # Swap order (productization §6): $GUTCHECK_DOCTRINE (set by the `--doctrine` flag / settings row)
    # wins; otherwise the bundled flagship pack #1. Path(__file__).parent, NOT .resolve() — .resolve()
    # can raise RuntimeError on a symlink loop, escaping the (OSError, UnicodeError) guards below;
    # doctrine.md sits next to this module on every supported install (editable + repo).
    override = os.environ.get(_DOCTRINE_ENV)
    if override:
        return Path(override)
    return Path(__file__).parent / "doctrine.md"


def _split_frontmatter(text: str) -> tuple[dict, str]:
    """Split an optional leading '---'-fenced frontmatter block from the markdown body → (meta, body).

    SEQUENTIAL + FAIL-CLOSED (both sacred invariants — no-leak + degrade-on-malformed — depend on it):
      * First line is NOT exactly '---'  -> ({}, text): a plain, frontmatter-less doctrine.md loads as
        the whole body (back-compat). A real no-frontmatter doc starts with '#', so a whole-line leading
        '---' is an UNAMBIGUOUS "this is a pack" signal (never matched by loose prefix — '----' /
        '---trailing' are NOT openers).
      * After the opener, a VALID block is only blank / '#'-comment / 'key: value' lines up to a clean
        closing '---'. The FIRST line that is none of those (body prose, OR a malformed closer like
        '--- nope') means the block is broken -> ({}, "") -> full() drops below the length floor -> stub.
        Parsing SEQUENTIALLY (NOT "scan for the next '---' anywhere") is what stops a later body
        horizontal-rule '---' from being 'rescued' as the closer and swallowing the real doctrine
        [codex L3 HIGH]. An opener that never closes is likewise ({}, "").
    Only the literal scalars true/false coerce to bool; every other value stays a string; the first ':'
    splits so a value may itself contain colons."""
    lines = text.split("\n")
    if not lines or lines[0].strip() != _FM_FENCE:        # whole-line opener, not a loose prefix
        return {}, text
    meta: dict = {}
    for i in range(1, len(lines)):
        s = lines[i].strip()
        if s == _FM_FENCE:                                # clean closer -> body is everything after it
            return meta, "\n".join(lines[i + 1:]).lstrip("\n")
        if not s or s.startswith("#"):                    # blank / comment lines are allowed in the block
            continue
        if ":" not in lines[i]:                           # a non-conforming line before the closer ->
            return {}, ""                                 # broken frontmatter -> fail closed [codex HIGH]
        k, _, v = lines[i].partition(":")                 # first ':' splits -> a value may contain colons
        k, v = k.strip(), v.strip().strip('"\'')
        meta[k] = (v.lower() == "true") if v.lower() in ("true", "false") else v
    return {}, ""                                         # opener but never closed -> fail closed


def meta() -> dict:
    """The ACTIVE pack's frontmatter metadata, over safe defaults. Returns defaults when the pack is
    unreadable OR structurally incomplete (the SAME guard full() applies, via _load_pack) — so meta()
    always reflects the doctrine actually in force, never a phantom 'Pack v9 loaded' beside a fallback
    coach. read_based defaults TRUE but is a LABEL, not a guard — the §3f seam is the boundary, so meta()
    never rejects a pack on it."""
    loaded = _load_pack()
    return {**_DEFAULT_META, **loaded[0]} if loaded else dict(_DEFAULT_META)


def _markers_ok(text: str) -> bool:
    """True iff the BRIEF region is present and well-ordered — exactly one START and one END, START
    before END. Rejects truncated (missing END) and malformed/duplicated (nested) marker sets."""
    return (text.count(_BRIEF_START) == 1 and text.count(_BRIEF_END) == 1
            and text.index(_BRIEF_START) < text.index(_BRIEF_END))


def _load_pack() -> tuple[dict, str] | None:
    """Read + split + STRUCTURALLY-VALIDATE the active pack ONCE — the single path full()/brief()/meta()
    all share, so they normalize IDENTICALLY (same BOM + leading-whitespace handling before the split,
    same frontmatter parse, same structural guard) [codex L3 HIGH: divergent normalization across the
    accessors was a leak/inconsistency path]. Returns (meta, body) ONLY when the pack is readable AND
    structurally complete (clears _MIN_DOC_LEN + the BRIEF markers); otherwise None, so every accessor
    degrades to its stub in lockstep.

    Scope of the lockstep guard = STRUCTURAL validity only (readable, long enough, markers present +
    ordered). It deliberately does NOT validate the brief REGION's content — an empty or oversized
    marked region is brief()'s OWN concern. So a structurally-valid doctrine with an unusable brief
    region correctly serves the full doctrine to the daemon (it's fine) and a stub brief to the MCP:
    we never throw away a valid doctrine over a brief-region authoring typo [codex re-verify MED —
    intentional per-surface degradation, not a lockstep violation]. Never raises."""
    try:
        text = _doc_path().read_text(encoding="utf-8-sig").strip()   # -sig strips a leading BOM; .strip()
    except (OSError, UnicodeError):                                  # normalizes leading whitespace — else
        return None                                                 # the '---' opener wouldn't match → leak
    meta, body = _split_frontmatter(text)
    body = body.strip()
    if len(body) < _MIN_DOC_LEN or not _markers_ok(body):
        return None
    return meta, body


def full() -> str:
    """The full flagship doctrine for surfaces with room for it (the daemon's claude -p prompt).
    Returns the minimal embedded stub if the active pack is unreadable OR structurally incomplete (a
    truncated/partial save, or broken frontmatter) — the daemon reads this per-request, so a torn read
    must degrade to the guard-carrying stub, never to partial text that could drop the §3f / read→action
    core. The frontmatter is stripped; the coach prompt is BODY only. Never raises."""
    loaded = _load_pack()
    return loaded[1] if loaded else _FALLBACK_FULL


def brief() -> str:
    """The concise read→action principle for length-bounded surfaces (the MCP `instructions` block).
    Extracted from the BRIEF-marked region of the active pack's BODY; falls back to a one-liner if the
    pack is unreadable / structurally incomplete (the SAME guard full() applies, via _load_pack — codex
    L3), the markers are absent/malformed, the region is empty, still contains a marker after extraction,
    or exceeds the metadata cap. The markers + any frontmatter are stripped. Never raises."""
    loaded = _load_pack()
    if not loaded:
        return _FALLBACK_BRIEF
    m = re.search(re.escape(_BRIEF_START) + r"(.*?)" + re.escape(_BRIEF_END), loaded[1], re.DOTALL)
    if not m:
        return _FALLBACK_BRIEF
    # collapse the markdown paragraph into a single clean instruction line for the metadata block
    region = re.sub(r"\s+", " ", m.group(1)).strip()
    # sanity: never leak a marker into the instructions, never let a huge region bloat them
    if not region or _BRIEF_START in region or _BRIEF_END in region or len(region) > _BRIEF_MAX:
        return _FALLBACK_BRIEF
    return region
