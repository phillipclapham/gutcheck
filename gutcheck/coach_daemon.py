"""gutcheck headless coach daemon — Slice 2, flavor (b).

The read side of the coach channel, as a STANDALONE process: it polls coach.py for pending
requests and answers each by firing `claude -p` on the Claude Code SUBSCRIPTION (OAuth — $0 against
the metered API) — so coaching works with NO open CC session. Slice 1 (flavor a) needed a warm
`coach_watch.py` session babysitting the channel; this is the no-babysitting path.

The channel seam (coach.py) is UNCHANGED and the app (server.py) is UNTOUCHED — exactly as the
Slice-1 design intended: Slice 2 swaps only the worker. Villains stay coded; only the coach is "me".

    python -m gutcheck.coach_daemon run            # poll + answer forever (the daemon)
    python -m gutcheck.coach_daemon run --once     # one pass over the current pending queue, then exit
    python -m gutcheck.coach_daemon answer --id N  # answer one specific pending request via claude -p
    python -m gutcheck.coach_daemon smoke          # fire ONE trivial claude -p call; print result + timing
                                                    #   (verify the subscription path on THIS machine)

WHY A STANDALONE PROCESS: coaching then needs no open CC session. The nesting gotcha is why the child
env matters: Claude Code refuses to run `claude` as a child of a session when it sees CLAUDECODE /
CLAUDE_CODE_* in its env, and the child dies silently. _claude_env strips those vars, so the daemon (and
the built-in worker) also works when started from inside a session; a gutcheck started from a Claude
Code session answered via claude on 2026-10-18. It also strips ANTHROPIC_API_KEY so the spawned `claude`
rides the Max subscription, not metered API billing — the whole point of the "coach on the subscription"
design.

GRACEFUL DEGRADATION (a hard floor): every request is ALWAYS answered and ALWAYS leaves the queue —
via claude -p, else the grounded stub (claude failed), else a safe fallback line (a malformed spot
that breaks even the stub). A request the worker touched but never answered would stay `pending`, get
re-fetched, and — on a poison payload — crash-loop the daemon forever; so answer_one can never let a
touched row stay pending. The player never gets a dead spinner.

LATENCY: `claude -p` cold-starts in ~seconds, so the daemon is the no-babysitting path, not the
snappiest. A warm `coach_watch.py` session (Slice 1) stays the low-latency option; this trades a few
seconds of cold start for needing no session at all.
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

from . import coach
from . import coach_watch
from . import doctrine
from . import keystore
from . import paths

HERE = Path(__file__).parent

# claude -p model: sonnet is the right default for an in-the-moment poker nudge — fast cold-start,
# plenty sharp for range-based coaching. Override via env for experiments.
COACH_MODEL = os.environ.get("GUTCHECK_COACH_MODEL", "sonnet")
# The metered Anthropic-API worker (provider 'anthropic_api' — call_anthropic_api) reads its MODEL and
# EFFORT live from the settings store (the in-UI cost×fidelity dials); GUTCHECK_API_MODEL still
# overrides the model for power use. These are real API model IDs, distinct from COACH_MODEL's
# `claude -p` alias ("sonnet") — the SDK needs the full id.
#
# Models that get the EFFORT + ADAPTIVE-THINKING pair (the two are always sent TOGETHER). The set is the
# INTERSECTION of "supports output_config.effort" AND "supports adaptive thinking" — NOT just effort:
# Opus 4.5 supports effort but uses a manual thinking budget (adaptive thinking is 4.6+), so sending it
# thinking={"type":"adaptive"} would be an invalid request (L3 codex) — it's excluded. Haiku 4.5 / pre-4.6
# Sonnet 400 on effort outright. A model the operator picks (via GUTCHECK_API_MODEL) that isn't here
# simply runs without effort/thinking (a plain call) — a structural gate, never a runtime surprise.
_EFFORT_MODELS = frozenset({
    "claude-opus-4-8", "claude-opus-4-7", "claude-opus-4-6", "claude-sonnet-4-6",
})
# The subset of _EFFORT_MODELS that ALSO accepts the "xhigh" effort level. xhigh was added in Opus 4.7
# (between high and max), so it's Opus-4.7+ only — Sonnet 4.6 and Opus 4.6 take low/medium/high but 400 on
# xhigh, and Haiku 4.5 (not in _EFFORT_MODELS) rejects effort entirely. A dialed xhigh on a non-xhigh model
# is degraded to "high" in the worker (the deepest level that model DOES support) — never sent as-is.
# DERIVED via intersection (not a bare literal) so `_XHIGH_EFFORT_MODELS ⊂ _EFFORT_MODELS` is STRUCTURAL,
# not a discipline a future contributor can violate: an xhigh-capable id missing from _EFFORT_MODELS is
# excluded here (→ degrades to high → plain call, fail-closed), never sent xhigh to a model the gate drops.
# VERIFY on an Anthropic-model change: re-confirm xhigh is still Opus-4.7+ (the literal below) per the docs.
_XHIGH_EFFORT_MODELS = _EFFORT_MODELS & frozenset({"claude-opus-4-8", "claude-opus-4-7"})
# The metered OpenAI worker (provider 'openai_api' — call_openai_api). reasoning_effort is gated on an
# EXPLICIT allowlist of the offered reasoning models — NOT a `gpt-5` prefix — mirroring the anthropic
# worker's _EFFORT_MODELS (L1+L2: a prefix would admit a non-reasoning gpt-5 chat variant via a
# GUTCHECK_OPENAI_MODEL override → reasoning_effort → 400; the allowlist gives any unknown/override
# model a PLAIN call instead, the safe direction). All three offered tiers take the SAME effort set —
# none/low/medium/high/xhigh (verified vs the model docs 2026-06-27) — so the level gate is the settings
# allowlist alone, no per-model effort matrix. (Note: "minimal" is NOT in that set — 5.4/5.5 replaced it
# with "none"; settings.openai_effort() can never emit it, so the worker never sends an effort that 400s.)
_OPENAI_EFFORT_MODELS = frozenset({"gpt-5.5", "gpt-5.4", "gpt-5.4-mini"})
# max_completion_tokens must be GENUINELY generous (L2 HIGH): OpenAI reasoning models bill AND consume
# hidden reasoning tokens out of THIS same budget, so a too-small cap can return EMPTY visible text you
# STILL PAID FOR — and the worst case is the operator picking the priciest dial ('high') and getting the
# facts floor + a bill. A high-effort gpt-5 read can burn many thousands of reasoning tokens before any
# prose, so 8192 was too small; 25000 matches OpenAI's documented "reserve ~25k for reasoning+output"
# guidance and sits well under the offered models' max-output limits. The ceiling is FREE (you pay per
# token actually generated, and a coach answer is short), so this only buys headroom. We additionally
# screen finish_reason != "stop" → '' so even a truncation degrades to the facts floor, never bad coaching.
_OPENAI_MAX_COMPLETION_TOKENS = 25000
# The anthropic worker's max_tokens. CRITICAL once adaptive thinking is on (the _EFFORT_MODELS): anthropic
# counts THINKING tokens against max_tokens, so a too-small cap lets high-effort thinking starve or TRUNCATE
# the visible answer (stop_reason='max_tokens') — the same starvation class _OPENAI_MAX_COMPLETION_TOKENS
# guards above. 2048 was sized for a NO-thinking nudge; 8192 gives thinking real headroom on a bounded poker
# prompt. The cap is FREE (you pay per token actually generated, and coach prose is short) — it only prevents
# truncation. What bounds this call is NOT an SDK max_tokens ceiling: the SDK's non-streaming long-request
# guard (_calculate_nonstreaming_timeout, which ValueErrors on a too-large cap) only fires when the client
# uses the DEFAULT timeout — messages.py:1012 `self._client.timeout == DEFAULT_TIMEOUT` — and the worker
# passes an explicit timeout=, so it's SUPPRESSED (codex/L1, verified against anthropic 0.112). The real
# bounds are the 90s wall-clock watchdog + the stop_reason screen below (a terminal that isn't 'end_turn',
# incl. 'max_tokens', degrades to the facts floor — never mid-sentence coaching). So there's non-streaming
# headroom: if play shows high effort routinely truncating, raise this modestly (still non-streaming) before
# reaching for .stream() — a future tuning item.
_ANTHROPIC_MAX_TOKENS = 8192
# Coaching prose is short; even a cold start finishes well inside this. On timeout we fall back.
CLAUDE_TIMEOUT = float(os.environ.get("GUTCHECK_COACH_TIMEOUT", "90"))
# How often the daemon re-checks the channel when idle. The poll is a cheap sqlite read; a
# claude -p call only fires when something is actually pending.
POLL_INTERVAL = float(os.environ.get("GUTCHECK_COACH_INTERVAL", "1.5"))
# Circuit breaker: after this many CONSECUTIVE claude failures, stop firing claude and serve
# grounded stubs directly for the cooldown — so a wedged / timing-out claude can't turn the daemon
# into a CLAUDE_TIMEOUT-per-request stall machine (launchd's respawn throttle can't help a process
# that stays alive but blocks). Recovery: after the cooldown, one real attempt; success resets the
# breaker, failure re-trips it.
BREAKER_THRESHOLD = int(os.environ.get("GUTCHECK_COACH_BREAKER", "3"))
BREAKER_COOLDOWN = float(os.environ.get("GUTCHECK_COACH_COOLDOWN", "60"))
# A poker DECISION older than this is moot — the player acted long ago, and no client will read a
# late answer (the app bumps its COACH_GEN epoch on every action). Expire it instead of spending a
# subscription call + blocking the queue on an answer nobody sees. Reviews are not time-sensitive.
STALE_DECISION_S = float(os.environ.get("GUTCHECK_COACH_STALE", "25"))
# coach.clean()'s TTL only fires if something calls it; the always-on daemon is its natural home.
CLEAN_INTERVAL_S = 60.0

# The last-resort answer when even the grounded stub can't be built (a malformed spot). It must never
# itself touch the request dict — it's the floor under graceful degradation.
_SAFE_FALLBACK = ("Couldn't read this spot cleanly — trust your gut, play it straightforward, and "
                  "we'll break it down after the hand.")

# The poker coaching contract = operator-specific frame + the DOCTRINE (loaded from doctrine.md via
# doctrine.full(), the single source every coach surface shares — see productization_design.md §3) +
# the transport instruction. The daemon does NOT run claude hermetically, so the player's own
# ~/.claude/CLAUDE.md loads: if it says who they are, the coach knows them. FLOW_HOOK_SUPPRESS is
# read only by the author's own session hooks; for anyone else it is an inert env var.
#
# The doctrine is operator-AGNOSTIC ("the operator") so it ships as a swappable pack; the frame
# names no particular player, so the same build coaches whoever sits down.
_OPERATOR_FRAME = """You're coaching a player at their 6-max No-Limit Hold'em cash table (a poker \
reading trainer). If you know this player (their play style, how they think, why they're working on \
this), bring all of it. The bots they're up against have HIDDEN archetypes; coach the spot in front of \
them. A "read-confidence" note in the spot tells you how far to trust the observed read. The doctrine \
below speaks of "the operator": that's the player you're coaching."""

_TRANSPORT = """Respond with ONLY the coaching prose — talk to the player directly, in your own voice."""


def _pedagogy() -> str:
    """The full daemon coaching prompt prefix: the operator frame, the shared doctrine (doctrine.md),
    and the transport instruction. Composed per call so an edit to doctrine.md takes effect without a
    daemon restart; doctrine.full() is fail-safe (never raises)."""
    return f"{_OPERATOR_FRAME}\n\n--- DOCTRINE ---\n{doctrine.full()}\n\n{_TRANSPORT}"


# ---------------- the claude -p invocation (the one piece that needs a real session to verify) ----
def _claude_env() -> dict:
    """Environment for spawning `claude`: strip the CC-nesting sentinels (so a child claude runs at
    all), strip ANTHROPIC_API_KEY (so it bills the Max subscription, not the API), and suppress flow's
    session hooks (this is a hermetic poker prompt, not a flow session). Mirrors the proven pattern in
    flow's publish_essay.py / delay.py."""
    # Fail-closed on the billing invariant: drop the ENTIRE ANTHROPIC_* namespace, not just the API
    # key — a stray ANTHROPIC_BASE_URL / proxy / auth var could silently route to metered billing.
    # OAuth subscription creds come from the login keychain, not env, so this never starves auth.
    env = {k: v for k, v in os.environ.items()
           if not k.startswith("ANTHROPIC_")
           and k != "CLAUDECODE"
           and not k.startswith("CLAUDE_CODE_")}
    env["FLOW_HOOK_SUPPRESS"] = "1"
    return env


def _claude_bin() -> str:
    """Resolve the claude binary: explicit override > PATH > the known macOS install location
    (MEMORY: ~/.local/bin/claude, NOT ~/.claude/local/claude)."""
    return (os.environ.get("CLAUDE_BIN")
            or shutil.which("claude")
            or str(Path.home() / ".local/bin/claude"))


def _claude_argv(prompt: str) -> list[str]:
    """The argv for one subscription-billed, tool-free, no-litter, JSON-output headless prose call.
    Flags verified against CLI 2.1.181 (claude-code-guide):
      -p / --output-format json / --model  : headless JSON prose on the chosen model
      --tools ""                            : disable ALL tools — pure text, can't touch disk
      --no-session-persistence              : no session file per call (valid with --print only)
      --max-turns 1                         : single model turn (matches publish_essay.py; with no
                                              tools there's nothing to loop on, but bounds it anyway)

    We deliberately do NOT pass --bare: --bare forces ANTHROPIC_API_KEY/apiKeyHelper auth and NEVER
    reads the OAuth keychain, which would bill the METERED API, not the subscription. Subscription
    billing = API key UNSET (see _claude_env) + no --bare.

    DESIGN (decided 2026-06-18): without --bare the player's own ~/.claude/CLAUDE.md (and anything it
    imports) loads, so a coach whose player has described themselves there already knows them. That is
    the FEATURE, not a leak. It was also the only working shape when measured on 2026-06-18: no flag then
    isolated the coach while keeping the subscription (--setting-sources/--safe-mode did not skip the
    user CLAUDE.md, and a HOME override was hermetic but broke OAuth login). (We do NOT cap MAX_THINKING_TOKENS: observed latency is 2-6s, fine for an on-demand nudge.)"""
    return [
        _claude_bin(), "-p", prompt,
        "--output-format", "json",
        "--model", COACH_MODEL,
        "--tools", "",
        "--no-session-persistence",
        "--max-turns", "1",
    ]


# The `claude -p` children in flight right now. A worker that is told to stop (the server's shutdown, a
# plain `kill`) reaps them: left alone they run on, re-parented to init, after the process that wanted
# their answer is gone.
_INFLIGHT: set = set()
_INFLIGHT_LOCK = threading.Lock()


_REAPED = False       # set by reap_inflight(): the process is shutting down, no new child may start


def _run_tracked(argv: list[str], timeout: float, input: str | None = None, **kw) -> subprocess.CompletedProcess:
    """subprocess.run, with the child registered in _INFLIGHT for its lifetime so reap_inflight() can
    reach it. Same contract: TimeoutExpired after killing the child; OSError if it can't spawn (and
    after reap_inflight(), so a call racing the shutdown degrades instead of starting an orphan)."""
    if input is not None:
        kw["stdin"] = subprocess.PIPE
    with _INFLIGHT_LOCK:
        if _REAPED:
            raise OSError("coach shutting down")
        proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, **kw)
        _INFLIGHT.add(proc)
    with proc:
        try:
            try:
                out, err = proc.communicate(input, timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()               # NOT communicate(): a grandchild holding the pipes would block it forever
                raise
            return subprocess.CompletedProcess(argv, proc.returncode, out, err)
        finally:
            with _INFLIGHT_LOCK:
                _INFLIGHT.discard(proc)


def reap_inflight() -> int:
    """Terminate every in-flight `claude -p` child (SIGTERM, then SIGKILL after a short grace). Returns
    how many were signalled. The call that owned each one sees a failed exit and degrades to the stub."""
    global _REAPED
    with _INFLIGHT_LOCK:
        _REAPED = True
        procs = list(_INFLIGHT)
    for proc in procs:
        if proc.poll() is None:
            proc.terminate()
    deadline = time.monotonic() + 2.0
    for proc in procs:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            proc.kill()
    return len(procs)


def call_claude(prompt: str, timeout: float = CLAUDE_TIMEOUT) -> str:
    """Fire one headless `claude -p` and return the response text, or '' on ANY failure. '' is the
    caller's signal to fall back to the grounded stub — so the coach never hangs. Never raises."""
    try:
        r = _run_tracked(
            _claude_argv(prompt),
            timeout=timeout,
            stdin=subprocess.DEVNULL,        # never block waiting on stdin (the codex-hang class)
            cwd=str(HERE), env=_claude_env(),
        )
    except subprocess.TimeoutExpired:
        print(f"[coach_daemon] claude -p timed out after {timeout}s", file=sys.stderr)
        return ""
    except Exception as e:                   # binary missing, OS error, etc. — degrade, don't crash
        print(f"[coach_daemon] claude -p spawn error: {e!r}", file=sys.stderr)
        return ""

    nonzero = r.returncode != 0
    if nonzero:
        print(f"[coach_daemon] claude -p exit {r.returncode}: {(r.stderr or '')[:300]}",
              file=sys.stderr)
    out = (r.stdout or "").strip()
    if not out:
        return ""
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        # plain-text stdout is only trustworthy on a CLEAN exit; on a failure exit it's almost
        # certainly an auth/nesting/usage ERROR message — never deliver that to the player as coaching.
        return "" if nonzero else out
    if not isinstance(data, dict) or data.get("is_error"):
        if isinstance(data, dict) and data.get("is_error"):
            detail = data.get("api_error_status") or data.get("result")
            print(f"[coach_daemon] claude reported is_error: {str(detail)[:300]}", file=sys.stderr)
        return ""
    result = data.get("result")
    return result.strip() if isinstance(result, str) else ""  # shape-drift → '' (never raises)


# ---------------- the codex -exec invocation (the second OAuth provider, slice c) ----------------
def _codex_env() -> dict:
    """Environment for spawning `codex`: strip the CC-nesting sentinels (so a child runs clean) and
    suppress flow's session hooks. Codex auths via its OWN login (~/.codex), not env, so nothing here
    starves it; we do NOT touch ANTHROPIC_* (irrelevant to codex)."""
    env = {k: v for k, v in os.environ.items()
           if k != "CLAUDECODE" and not k.startswith("CLAUDE_CODE_")}
    env["FLOW_HOOK_SUPPRESS"] = "1"
    return env


def _codex_bin() -> str:
    """Resolve the codex binary: explicit override > PATH > the known Homebrew install location."""
    return os.environ.get("CODEX_BIN") or shutil.which("codex") or "/opt/homebrew/bin/codex"


def call_codex(prompt: str, timeout: float = CLAUDE_TIMEOUT) -> str:
    """Fire one headless `codex exec` and return the response text, or '' on ANY failure ('' → stub).
    Mirrors call_claude's contract (never raises, never hangs) with codex's two specifics:
      - the prompt is PIPED on stdin (a positional prompt makes `codex exec` block reading stdin — the
        documented hang class); subprocess `input=` provides it + closes stdin (EOF) so codex proceeds,
      - the final message is read from a FRESH per-call `-o` temp file (a stale -o would deliver a PRIOR
        run's result), unlinked in finally. The subprocess timeout also covers codex's silent-hang class
        (PID alive, 0% CPU, network drop) — TimeoutExpired → '' → the grounded stub.
    `--ephemeral` keeps coach calls out of ~/.codex/sessions (matches claude's --no-session-persistence).
    OAuth via the codex login (whatever ~/.codex is set to — the UI says "your Codex login", not "no API
    key", because we don't enforce the auth mode here; the explicit OAuth-vs-token choice is the deferred
    metered-worker slice). read-only sandbox (the coach never writes).

    KNOWN ASYMMETRY (owned, deferred to the metered-worker slice): call_claude parses JSON + screens
    `is_error`; codex is screened only on EXIT CODE (non-zero → ''). A codex exit-0 whose final message is
    a refusal / usage-notice would be served as coaching. Low-probability (read-only + a self-contained
    poker prompt give little to refuse, and real auth/usage failures exit non-zero), but not screened."""
    out_path = None
    try:
        fd, out_path = tempfile.mkstemp(prefix="fp_codex_", suffix=".txt")
        os.close(fd)
        r = _run_tracked(
            [_codex_bin(), "exec", "--ephemeral", "--sandbox", "read-only", "-o", out_path],
            input=prompt, timeout=timeout,
            cwd=str(HERE), env=_codex_env(),
        )
        if r.returncode != 0:
            print(f"[coach_daemon] codex exec exit {r.returncode}: {(r.stderr or '')[:300]}",
                  file=sys.stderr)
            return ""
        return Path(out_path).read_text().strip()        # '' if codex wrote nothing → stub fallback
    except subprocess.TimeoutExpired:
        print(f"[coach_daemon] codex exec timed out after {timeout}s", file=sys.stderr)
        return ""
    except Exception as e:                                # binary missing, OS error, read error — degrade
        print(f"[coach_daemon] codex exec error: {e!r}", file=sys.stderr)
        return ""
    finally:
        if out_path:
            try:
                os.unlink(out_path)
            except OSError:
                pass


# ---------------- shared metered-client hardening: deny the ambient *_CUSTOM_HEADERS channel ----------
@contextlib.contextmanager
def _denied_ambient_env(*env_names: str):
    """Temporarily remove ambient credential/header env var(s) for the duration of the block, restoring
    them after. Named for its first use — the `<PROVIDER>_CUSTOM_HEADERS` channel — but it's a generic
    construction-window env denier: the anthropic worker also passes ANTHROPIC_AUTH_TOKEN through it (a
    credential vector the SDK likewise resolves at __init__), so any inherited value is invisible to the
    build. The shared requirement for everything passed here: the SDK reads it AT CONSTRUCTION (assumption
    (1) below). BOTH metered SDK clients read a *_CUSTOM_HEADERS var at construction and merge it into the
    request headers — SPREAD OVER the auth header (verified against anthropic 0.112 `_client.py:234` and
    openai `_client.py:217`: the env-parsed dict lands in `_custom_headers`, which the base client spreads
    `**self._custom_headers` LAST, after `**self.auth_headers`). There is no SDK flag to disable that
    channel, so an inherited env var (a proxy / gateway / another tool) could inject or override headers —
    including `x-api-key` — on the operator's BILLED call. Denying the var across the synchronous
    constructor window (the SDKs read + cache it at `__init__`; `create()` never re-reads it) closes the
    channel completely — the same hermetic posture `_claude_env()` takes for the subprocess path. Restore
    in finally so a construction error can't leak the stripped state. (codex, sibling-openai-worker review.)

    TWO load-bearing assumptions (L1) — re-verify on any gutcheck[api] SDK floor bump:
      (1) the SDK resolves *_CUSTOM_HEADERS at construction and CACHES it (request-time methods don't re-read
          the env) — true for anthropic 0.112 / openai 2.x; if a future SDK moved header resolution to
          request time, this deny-block would silently become a no-op AND the mocked test suite would stay
          green (the fakes can't model a request-time read). Re-check `_client.py` on a bump.
      (2) the constructor does NO network I/O (httpx builds a lazy pool, opens no socket) — that is what
          makes the strip window non-blocking and the os.environ mutation effectively single-threaded. TWO
          things keep it single-threaded: the serial coach loop AND the _METERED_INFLIGHT gate, which stops a
          NEW worker from spawning (and mutating os.environ) while an ABANDONED worker is still draining
          alongside the daemon thread. If a future SDK added an eager connect/auth probe to __init__, OR the
          gate were removed as "just a billing optimization" (L1), the env mutation would no longer be
          race-free."""
    saved = {n: os.environ.pop(n, None) for n in env_names}
    try:
        yield
    finally:
        for n, v in saved.items():
            if v is not None:
                os.environ[n] = v


# ---------------- shared metered-call runner: the wall-clock watchdog + the one-in-flight gate ----------
# At most ONE metered SDK call in flight across the process. The watchdog abandons an OVERRUNNING worker
# THREAD, not its in-flight HTTP request — that request keeps draining on the SDK's own per-op timeout (up to
# ~2×timeout) and still BILLS. Without this gate the serial coach loop would start a NEW metered call the
# instant the watchdog returned the facts floor, stacking a 2nd billable request on the still-draining one
# (and N stacking if every call overran). The lock is held for the worker thread's LIFETIME, so a draining
# abandoned call keeps the gate shut. (codex MED — sibling-worker review.)
_METERED_INFLIGHT = threading.Lock()


def _metered_call(thunk, label: str, timeout: float):
    """Run ONE metered SDK call under (a) the process-wide one-in-flight gate and (b) a hard wall-clock
    watchdog, returning (ok, resp). `thunk()` builds the client (under _denied_ambient_env) and returns
    the raw SDK response — it OWNS the client's `with` close, so the transport closes on every path including
    a watchdog abandon. ok=False (resp=None) → the caller serves the facts floor: gate contention, overrun,
    spawn failure, or an SDK error. ok=True → `resp` is the raw response for the caller to SCREEN (each worker
    keeps its own terminal/refusal screen).

    The watchdog: the SDK's own `timeout=` is HTTPX's PER-OPERATION inactivity limit, not total-elapsed (L3,
    3 substrates) — a slow-drip response would wedge the daemon past the budget. So the call runs in a daemon
    thread joined for `timeout`; an overrun is abandoned (it finishes off the loop's path) → facts floor.

    The gate (codex MED): captured LOCALLY so an abandoned thread releases the SAME lock it acquired even if
    the module global is later swapped (tests do that for isolation). Held until the thread truly finishes
    (its finally), so while an abandoned call drains, a new call can't acquire it → it degrades to the facts
    floor instead of starting a 2nd billable call. This is a COST LATCH, not a kill switch: the hold is
    ~2×timeout ONLY if the SDK's own per-op timeout unwinds the call; a wedged socket or an adversarial
    slow-drip (bytes arriving just under the per-op limit — realistic only via a custom GUTCHECK_*_BASE_URL
    proxy) can hold the gate longer, worst case until process restart, with metered coaching failing CLOSED to
    the facts floor the whole time (the safe direction — never a double-bill, and never a daemon hang since the
    calling thread's acquire is non-blocking; codex/complement/L1). It does NOT kill the abandoned call (you
    can't recall an in-flight HTTPS request from a thread; a killable subprocess is heavier than a local
    single-process coach toy warrants). Process-wide only — assumes the single launchd daemon owns metered
    calls (a concurrent `answer`/`smoke` invocation wouldn't share this gate)."""
    lock = _METERED_INFLIGHT                     # capture locally — see the gate note above
    if not lock.acquire(blocking=False):
        print(f"[coach_daemon] {label}: a prior metered call is still in flight — serving the facts floor "
              f"(not starting a 2nd billable call)", file=sys.stderr)
        return (False, None)
    box: dict = {}

    def _run() -> None:
        try:
            box["resp"] = thunk()
        except Exception as e:                   # auth/rate-limit/network/timeout/400 — degrade, don't crash
            # type NAME only (L3 complement/nemotron): the class is enough to diagnose, and dropping str(e)
            # removes the one place a future SDK could echo a key fragment into a log.
            box["err"] = type(e).__name__
        finally:
            lock.release()                       # release ONLY when the thread truly finishes (abandoned or not)

    t = threading.Thread(target=_run, name=f"gutcheck-coach-{label}", daemon=True)
    try:
        t.start()
    except Exception:                            # thread exhaustion etc. — _run never ran, so its finally
        lock.release()                           # won't fire; release here so the gate can't leak, then degrade
        return (False, None)
    t.join(timeout)
    if t.is_alive():                             # overran the budget → abandon it (drains off the loop's path)
        print(f"[coach_daemon] {label} exceeded {timeout:.0f}s wall-clock — abandoning the call, "
              f"serving the facts floor", file=sys.stderr)
        return (False, None)
    if "err" in box:
        print(f"[coach_daemon] {label} call failed: {box['err']}", file=sys.stderr)
        return (False, None)
    return (True, box.get("resp"))


# ---------------- the metered Anthropic-API invocation (BYO-key worker — provider 'anthropic_api') ----
def call_anthropic_api(prompt: str, timeout: float = CLAUDE_TIMEOUT) -> str:
    """Fire one Anthropic Messages-API call with the operator's OWN key and return the text, or '' on
    ANY failure ('' → the grounded stub). The third coach worker, and the first that is NOT a subprocess
    CLI: call_claude/call_codex ride an OAuth login (no key); THIS bills the operator's metered API key
    directly — the path for someone with no Claude subscription (and likely no `claude` CLI installed).

    The official `anthropic` SDK is imported LAZILY so it stays an OPTIONAL dependency (``pip install
    'gutcheck[api]'``) — sub-only users never pull it. Same contract as the other two runners: never
    raises, never hangs, '' on any failure.
      - **Four ambient vectors are pinned: the API key, the endpoint, the CUSTOM_HEADERS injection channel,
        and the AUTH_TOKEN credential.** (Not a blanket "no ambient env" — ambient ANTHROPIC_ORG_ID-style
        config that can't override auth or redirect the pinned endpoint stays honored.)
        keystore.get_api_key reads only GUTCHECK_ANTHROPIC_API_KEY / the secrets file, and we pass api_key=
        to the client — so the SDK can never silently fall back to a shared/work key in the daemon's inherited
        env (see keystore.py). We pass base_url= (GUTCHECK_ANTHROPIC_BASE_URL or the official endpoint), never
        the ambient ANTHROPIC_BASE_URL the SDK would otherwise honor — so a stray proxy/gateway env var can't
        redirect the metered key (the credential-exfil vector codex flagged HIGH on the sibling openai worker).
        And the constructor runs under _denied_ambient_env, denying TWO ambient vectors for the build
        window: ANTHROPIC_CUSTOM_HEADERS (which the SDK merges SPREAD OVER the auth header) and
        ANTHROPIC_AUTH_TOKEN. The AUTH_TOKEN deny is belt-and-suspenders: in anthropic 0.112 passing an
        explicit api_key= already makes the SDK skip ALL credential env vars ("explicit ctor args are total",
        _client.py:204-217 — verified), so an ambient AUTH_TOKEN is ALREADY ignored and self.auth_token stays None
        (no rogue Bearer header). The strip makes that hermeticity STRUCTURAL rather than dependent on the
        SDK's hand-patched precedence: if a future floor reverted "explicit args are total", an inherited
        AUTH_TOKEN would otherwise authenticate the operator's billed call as a DIFFERENT account. Re-verify
        on a gutcheck[api] floor bump (same assumption as _denied_ambient_env: read-at-construction).
        (Asymmetry by design — we deny AUTH_TOKEN but NOT ambient ANTHROPIC_API_KEY: an ambient API key would
        populate the SAME `x-api-key` header our explicit api_key= already sets [explicit param wins], so it
        can't redirect auth; AUTH_TOKEN is a SEPARATE `Authorization: Bearer` header the SDK emits ALONGSIDE
        x-api-key — the genuine dual-auth vector — which is why only it needs the strip.)
      - **Wall-clock bound + one-in-flight gate (both in _metered_call):** the SDK call runs in a daemon
        WATCHDOG thread joined for `timeout`; if it overruns, the loop abandons it and serves the facts floor.
        anthropic's own `timeout=` is HTTPX's PER-OPERATION inactivity limit, NOT total-elapsed (L3, 3
        substrates) — a slow-drip response would otherwise wedge the daemon past the budget. The watchdog is
        what makes 'never hangs' real; the shared process-wide gate then caps concurrent metered calls at one
        so a draining abandoned call can't be double-billed by the next request (codex MED). max_retries=0
        additionally stops the SDK's 429/5xx backoff from stacking attempts.
      - **Terminal-screened (an `end_turn` whitelist):** only a naturally-finished `stop_reason == "end_turn"`
        is served; `max_tokens` (a TRUNCATED answer — thinking is billed out of max_tokens), `refusal`
        (empty/partial content), and any unexpected terminal degrade to the facts floor (parity with the openai
        worker's `finish_reason != "stop"` screen). The _TRANSPORT 'ONLY the coaching prose' instruction in the
        prompt doubles as the final-answer constraint.
      - **Cost×fidelity dials (live from settings):** the operator picks the MODEL (Opus/Sonnet/Haiku — the
        cost lever) and EFFORT ('low'|'medium'|'high'|'xhigh' — the fidelity×latency lever). effort is sent as
        output_config.effort + adaptive thinking, and only for models that support BOTH (_EFFORT_MODELS); a
        non-supporting model (e.g. Haiku) → a plain call (the cheapest/fastest real-time nudge). 'xhigh' is
        Opus-4.7+ only (_XHIGH_EFFORT_MODELS) — on ANY non-xhigh model it's degraded to 'high' before the
        call; that 'high' is then SENT on Sonnet 4.6 but DROPPED on Haiku (Haiku ∉ _EFFORT_MODELS → plain)."""
    key = keystore.get_api_key("anthropic")
    if not key:
        print("[coach_daemon] anthropic_api selected but no API key is set — "
              "set one in Settings → Provider → Claude (API key), or export GUTCHECK_ANTHROPIC_API_KEY",
              file=sys.stderr)
        return ""
    try:
        import anthropic                       # lazy: optional dep (gutcheck[api]); absent → degrade
    except ImportError:
        print("[coach_daemon] anthropic_api selected but the `anthropic` package is not installed — "
              "run: pip install 'gutcheck[api]'", file=sys.stderr)
        return ""

    # Resolve the live cost×fidelity dials. Lazy + defensive: a settings read must NEVER break the worker —
    # on any read failure, fall back to the flagship + a PLAIN call (effort=None). GUTCHECK_API_MODEL still
    # overrides the model for power use.
    try:
        from . import settings as _settings
        model = os.environ.get("GUTCHECK_API_MODEL") or _settings.api_model()
        effort = _settings.api_effort()         # a valid level (low|medium|high|xhigh); the except below sets None
    except Exception:
        model, effort = (os.environ.get("GUTCHECK_API_MODEL") or "claude-opus-4-8"), None
    # xhigh is Opus-4.7+ only. The UI offers it only for Opus, but a persisted xhigh can outlive a model
    # switch (api_effort is model-blind) and GUTCHECK_API_MODEL can override the model entirely — so degrade
    # xhigh→high on any non-xhigh model here. That sends the deepest level the model DOES accept rather than a
    # 400 that silently collapses to the facts floor (the worst case: priciest dial, no coaching, still billed).
    if effort == "xhigh" and model not in _XHIGH_EFFORT_MODELS:
        effort = "high"
    # Endpoint is EXPLICIT, never the ambient ANTHROPIC_BASE_URL (mirrors the openai worker; L3 codex flagged
    # this HIGH there): the SDK silently falls back to ANTHROPIC_BASE_URL when base_url isn't passed, so a
    # daemon that inherited that env var (a proxy / gateway / another tool) would ship the operator's metered
    # key to a non-Anthropic endpoint. Same posture as the KEY: only an EXPLICIT gutcheck opt-in
    # (GUTCHECK_ANTHROPIC_BASE_URL — for a legit gateway/proxy) can redirect it, defaulting to the official
    # endpoint (no /v1 — the SDK appends the path). The ambient ANTHROPIC_BASE_URL is never honored.
    base_url = os.environ.get("GUTCHECK_ANTHROPIC_BASE_URL") or "https://api.anthropic.com"
    create_kwargs = {
        "model": model,
        "max_tokens": _ANTHROPIC_MAX_TOKENS,    # generous so adaptive THINKING can't starve the answer (see const)
        "messages": [{"role": "user", "content": prompt}],
    }
    # Send effort + adaptive thinking ONLY for models that support BOTH (output_config.effort 400s on Haiku
    # 4.5; adaptive thinking is 4.6+, so opus-4-5 is excluded too — see _EFFORT_MODELS). `effort` is a valid
    # level on the happy path and None only on the settings-read fallback → a plain call either way for a
    # non-supporting model. Paired so the level actually drives reasoning depth.
    if effort and model in _EFFORT_MODELS:
        create_kwargs["thinking"] = {"type": "adaptive"}
        create_kwargs["output_config"] = {"effort": effort}

    # Build + fire the call under the shared one-in-flight gate + wall-clock watchdog (_metered_call). The
    # thunk builds the client with api_key + base_url explicit and the ambient ANTHROPIC_CUSTOM_HEADERS
    # channel denied for the constructor (it spreads env headers OVER the auth header — codex), then
    # `with client:` closes the HTTPX transport on every path (incl. a watchdog abandon).
    def _thunk():
        with _denied_ambient_env("ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_AUTH_TOKEN"):
            client = anthropic.Anthropic(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        with client:
            return client.messages.create(**create_kwargs)

    ok, resp = _metered_call(_thunk, "anthropic_api", timeout)
    if not ok or resp is None:                  # contention / overrun / error / (shouldn't-happen) empty → floor
        return ""
    # Serve ONLY a naturally-finished answer (parity with the openai worker's finish_reason!='stop' screen).
    # Whitelisting 'end_turn' degrades EVERY non-success terminal to the facts floor: 'max_tokens' (a TRUNCATED,
    # mid-sentence answer — anthropic counts thinking against max_tokens, so a high-effort read can hit it),
    # 'refusal' (empty/partial content), and any unexpected terminal. We send no tools / stop sequences, so
    # 'end_turn' is the only success terminal for this plain text call.
    stop = getattr(resp, "stop_reason", None)
    if stop != "end_turn":                                # !r so a missing field prints None, not the str "None"
        print(f"[coach_daemon] anthropic_api stopped on {stop!r} (not a finished answer) — "
              f"falling back to the facts floor", file=sys.stderr)
        return ""
    try:                                        # text blocks only; shape-drift → '' (never raises)
        return "".join(b.text for b in (resp.content or []) if getattr(b, "type", None) == "text").strip()
    except Exception:
        return ""


# ---------------- the metered OpenAI-API invocation (BYO-key worker — provider 'openai_api') ----------
def call_openai_api(prompt: str, timeout: float = CLAUDE_TIMEOUT) -> str:
    """Fire one OpenAI Chat-Completions call with the operator's OWN key and return the text, or '' on ANY
    failure ('' → the grounded stub). The FOURTH coach worker, and the SIBLING of call_anthropic_api — the
    metered path for someone with an OpenAI key and no Claude/Codex subscription. Same contract as the
    other three runners: never raises, never hangs, '' on any failure.

    Mirrors call_anthropic_api structurally; the OpenAI specifics:
      - **Chat Completions, not the Responses API** (deliberate): a stateless single-turn text-in/text-out
        nudge maps 1:1 onto Chat (one user message, top-level `reasoning_effort=`), it parallels the
        anthropic worker for review, and its refusal/truncation screen is the VERIFIED one
        (`finish_reason` + `message.refusal`) where the Responses `incomplete_details` enum is not.
      - **Three ambient vectors are pinned: the API key, the endpoint, and the CUSTOM_HEADERS injection channel.**
        keystore.get_api_key('openai') reads only GUTCHECK_OPENAI_API_KEY / the secrets file, and we pass
        api_key= to the client — the SDK can never silently fall back to a shared/work key in the daemon's
        inherited env (see keystore.py). Likewise we pass base_url= (GUTCHECK_OPENAI_BASE_URL or the official
        endpoint), never the ambient OPENAI_BASE_URL the SDK would otherwise honor — so a stray proxy/Azure env
        var can't redirect the metered key (L3 codex). And the constructor runs under _denied_ambient_env,
        denying the ambient OPENAI_CUSTOM_HEADERS channel (which the SDK spreads OVER the auth header) the same
        way. NOT pinned — and intentionally honored — is ambient OPENAI_ORG_ID / OPENAI_PROJECT_ID: the SDK
        turns those into org/project headers, but they're legit operator config, can't override auth, and don't
        redirect the now-pinned endpoint, so they stay honored (codex — the docstring no longer claims a blanket
        "no ambient env").
      - **Wall-clock bound + one-in-flight gate (both in _metered_call):** the SDK call runs in a daemon
        WATCHDOG thread joined for `timeout`; an overrun is abandoned → facts floor. The client's own
        `timeout=` is HTTPX's per-operation limit, not total-elapsed; the watchdog is what makes 'never hangs'
        real. The shared process-wide gate caps concurrent metered calls at one (a draining abandoned call
        can't be double-billed by the next request — codex MED). max_retries=0 stops paid backoff.
      - **NO `temperature`** — gpt-5.x reasoning models 400 on any non-default temperature, so we never send
        it. **`max_completion_tokens`** (NOT the legacy `max_tokens`, which reasoning models reject), set
        generous so billed reasoning tokens can't starve the answer to empty (_OPENAI_MAX_COMPLETION_TOKENS).
      - **Screened both ways:** serve ONLY a finished, non-refused, non-empty answer — finish_reason must be
        'stop' (a 'length'/'content_filter' truncation is never served as coaching) AND message.refusal must
        be None. The _TRANSPORT 'ONLY the coaching prose' instruction doubles as the format constraint.
      - **Cost×fidelity dials (live from settings):** the operator picks the MODEL (gpt-5.5 / 5.4 / 5.4-mini
        — the cost lever) and EFFORT ('none'|'low'|'medium'|'high'|'xhigh' — reasoning_effort, the depth lever;
        'none' is the genuine fastest/cheapest, 'xhigh' the deepest). Effort is sent only for an offered
        reasoning model (_OPENAI_EFFORT_MODELS — an explicit allowlist); an unknown GUTCHECK_OPENAI_MODEL
        override → a plain call (the safe direction). 'none' is a non-empty string so it passes the truthy
        gate and is sent explicitly (distinct from omitting the param, which would default per-model).

    Contract caveat: `_metered_call` itself now catches a stray RuntimeError from `t.start()` under thread
    exhaustion (releasing the gate + returning the facts floor); `answer_one` remains the outer backstop, so a
    spawn failure lands on the facts floor, not the operator."""
    key = keystore.get_api_key("openai")
    if not key:
        print("[coach_daemon] openai_api selected but no API key is set — "
              "set one in Settings → Provider → OpenAI key, or export GUTCHECK_OPENAI_API_KEY",
              file=sys.stderr)
        return ""
    try:
        import openai                          # lazy: optional dep (gutcheck[api]); absent → degrade
    except ImportError:
        print("[coach_daemon] openai_api selected but the `openai` package is not installed — "
              "run: pip install 'gutcheck[api]'", file=sys.stderr)
        return ""

    # Resolve the live cost×fidelity dials. Lazy + defensive: a settings read must NEVER break the worker —
    # on any read failure, fall back to the flagship + a PLAIN call (effort=None). GUTCHECK_OPENAI_MODEL
    # still overrides the model for power use.
    try:
        from . import settings as _settings
        model = os.environ.get("GUTCHECK_OPENAI_MODEL") or _settings.openai_model()
        effort = _settings.openai_effort()      # always a valid level (none|low|medium|high|xhigh), never None
    except Exception:
        model, effort = (os.environ.get("GUTCHECK_OPENAI_MODEL") or "gpt-5.5"), None
    # Endpoint is EXPLICIT, never the ambient OPENAI_BASE_URL (L3 codex HIGH): the SDK silently falls back
    # to OPENAI_BASE_URL when base_url isn't passed, so a daemon that inherited that env var (a proxy /
    # Azure / another tool) would ship the operator's metered key to a non-OpenAI endpoint. Same posture as
    # the KEY: only an EXPLICIT gutcheck opt-in (GUTCHECK_OPENAI_BASE_URL — for legit Azure/proxy use) can
    # redirect it, defaulting to the official endpoint. The ambient OPENAI_BASE_URL is never honored.
    base_url = os.environ.get("GUTCHECK_OPENAI_BASE_URL") or "https://api.openai.com/v1"
    create_kwargs = {
        "model": model,
        "max_completion_tokens": _OPENAI_MAX_COMPLETION_TOKENS,   # NOT max_tokens (rejected by reasoning models)
        "messages": [{"role": "user", "content": prompt}],        # one user turn — parallel to the anthropic worker
    }
    # reasoning_effort only for an OFFERED reasoning model (explicit allowlist, not a prefix — an unknown
    # or non-reasoning override would 400 on it). `effort` is a valid level on the happy path, None only on
    # the settings-read fallback → a plain call either way.
    if effort and model in _OPENAI_EFFORT_MODELS:
        create_kwargs["reasoning_effort"] = effort

    # Build + fire under the shared one-in-flight gate + wall-clock watchdog (_metered_call). The thunk builds
    # the client with api_key + base_url explicit and the ambient OPENAI_CUSTOM_HEADERS channel denied for the
    # constructor (it spreads env headers OVER the auth header — codex), then `with client:` closes the HTTPX
    # transport on every path (incl. a watchdog abandon).
    def _thunk():
        with _denied_ambient_env("OPENAI_CUSTOM_HEADERS"):
            client = openai.OpenAI(api_key=key, base_url=base_url, timeout=timeout, max_retries=0)
        with client:
            return client.chat.completions.create(**create_kwargs)

    ok, resp = _metered_call(_thunk, "openai_api", timeout)
    if not ok or resp is None:                  # contention / overrun / error / (shouldn't-happen) empty → floor
        return ""
    try:                                        # shape-drift anywhere in here → '' (never raises)
        choice = (getattr(resp, "choices", None) or [None])[0]
        if choice is None:
            return ""
        if getattr(choice, "finish_reason", None) != "stop":   # truncation ('length') / 'content_filter'
            print(f"[coach_daemon] openai_api stopped on '{getattr(choice, 'finish_reason', None)}' "
                  f"(not a finished answer) — falling back to the facts floor", file=sys.stderr)
            return ""
        msg = getattr(choice, "message", None)
        if getattr(msg, "refusal", None):       # a refusal carries no usable coaching — never serve it
            print("[coach_daemon] openai_api refusal — falling back to the facts floor", file=sys.stderr)
            return ""
        return (getattr(msg, "content", None) or "").strip()
    except Exception:
        return ""


def _runner_for(provider_name: str):
    """Map the live provider setting → its runner. Looks up the module-level functions by name at call
    time (monkeypatchable in tests); an unknown provider falls back to claude.
      - 'codex'         → call_codex          (OAuth, codex login)         [slice c]
      - 'anthropic_api' → call_anthropic_api  (metered, BYO Anthropic key) [metered-worker slice]
      - 'openai_api'    → call_openai_api      (metered, BYO OpenAI key)    [openai metered-worker slice]
      - else (incl. 'claude') → call_claude   (OAuth, Claude subscription) [default]"""
    if provider_name == "codex":
        return call_codex
    if provider_name == "anthropic_api":
        return call_anthropic_api
    if provider_name == "openai_api":
        return call_openai_api
    return call_claude


# ---------------- the worker loop (fully testable without claude) ----------------
def build_prompt(req: dict, db=None) -> str:
    """The full claude -p prompt: the pedagogy contract + the laid-out spot (shared verbatim with the
    warm-session worker via coach_watch.format_spot). `db` flows through so the modeled-equity meta is
    persisted to the channel the daemon is actually serving (test isolation; the default is the data-dir coach.db)."""
    return f"{_pedagogy()}\n\n--- THE SPOT ---\n{coach_watch.format_spot(req, db)}"


def answer_one(req: dict, runner=call_claude, db=None, facts_mode: bool = False,
               success_source: str = "claude") -> tuple[bool, str]:
    """Coach one request and ALWAYS write a reply — claude -p, else the grounded stub, else a safe
    fallback line on ANY exception (a malformed spot that breaks even the formatter). The row MUST
    leave 'pending' no matter what: a touched-but-unanswered row gets re-fetched next pass and, on a
    poison payload, crash-loops the daemon. `runner` is injectable so the loop is testable without
    spawning claude. `facts_mode` forwards to the stub: True = deliberate facts-only (clean, no head),
    False = degraded (a generic 'AI coach unavailable' head). `success_source` labels a RUNNER success
    (the provider — 'claude' | 'codex') so logs/counts name the actual provider, not always 'claude'.
    Returns (responded, source) with source in success_source | 'stub' | 'error'; responded is False only
    if the row was already answered (idempotent race)."""
    db = db or paths.coach_db()
    source = success_source
    try:
        prose = (runner(build_prompt(req, db)) or "").strip()
        if not prose:
            prose, source = coach_watch.stub_answer(req, db, facts_mode=facts_mode), "stub"
    except Exception as e:                   # malformed spot, formatter KeyError, anything at all
        print(f"[coach_daemon] coaching #{req.get('id')} failed: {e!r}", file=sys.stderr)
        prose, source = _SAFE_FALLBACK, "error"
    responded = coach.respond(req["id"], prose, db)
    return responded, source


def _stub_only(_prompt: str) -> str:
    """A runner that returns nothing → forces answer_one's grounded-stub fallback. Used when the
    circuit breaker is open (claude is failing) so we skip the per-call timeout and serve stubs fast."""
    return ""


def _is_stale_decision(req: dict, now: float) -> bool:
    """A 'decision' request older than the staleness window is moot — the player has acted. Reviews
    are never stale (hindsight coaching is useful whenever it lands). Defensive: an unparseable
    `created` (NULL / corrupt row) is treated as NOT stale, so it's answered normally rather than
    raising in the (otherwise un-try/excepted) loop."""
    if req.get("kind") != "decision":
        return False
    try:
        return (now - float(req.get("created", now))) > STALE_DECISION_S
    except (TypeError, ValueError):
        return False


def _probe_subscription(runner=call_claude) -> bool:
    """One-shot startup health check (MED-1): is the claude/subscription path actually answering?
    Logs LOUDLY on failure so a silent all-stub session is visible instead of looking like coaching.
    Best-effort; never fatal — the daemon still runs and degrades to stubs."""
    try:
        out = (runner("Reply with exactly one word, nothing else: PONG") or "")
    except Exception as e:
        out = ""
        print(f"[coach_daemon] subscription probe error: {e!r}", file=sys.stderr)
    if "PONG" in out.upper():
        print("[coach_daemon] subscription path OK — claude -p answered.", file=sys.stderr)
        return True
    print("[coach_daemon] ⚠ SUBSCRIPTION PATH DOWN — claude -p is not answering; the coach will "
          "serve grounded STUBS only. Check: is the login keychain unlocked (GUI session)? "
          "Run `python -m gutcheck.coach_daemon smoke` in a plain terminal to diagnose.", file=sys.stderr)
    return False


class _Breaker:
    """Half-open circuit breaker for the claude path. CLOSED → (BREAKER_THRESHOLD consecutive claude
    failures) → OPEN (serve stubs WITHOUT calling claude, for the cooldown) → HALF_OPEN (cooldown
    elapsed: ONE real probe attempt) → CLOSED on success, or straight back to OPEN on failure. The
    half-open probe is the fix for a persistently-wedged claude: without it, every cooldown would
    re-incur a fresh THRESHOLD run of full-timeout stalls. Clock is injectable for testing."""

    def __init__(self, threshold: int, cooldown: float, clock=time.monotonic):
        self.threshold = threshold
        self.cooldown = cooldown
        self.clock = clock
        self.fails = 0
        self.open_until = 0.0        # > 0 while OPEN (cooling down)
        self.half_open = False       # True = the next real attempt is the recovery probe

    def trip_cold(self) -> None:
        """Start OPEN without a prior failure run — used when the startup probe fails, so the first
        requests serve stubs immediately instead of each paying a full claude timeout."""
        self.open_until = self.clock() + self.cooldown

    def should_call_claude(self) -> bool:
        """True = make a real claude attempt (CLOSED or HALF_OPEN). False = serve a stub without
        calling claude (OPEN). Transitions OPEN→HALF_OPEN once the cooldown elapses."""
        if self.open_until and self.clock() >= self.open_until:
            self.open_until = 0.0
            self.half_open = True
            self.fails = 0
        return self.open_until == 0.0

    def is_closed(self) -> bool:
        """Pure read (NO state transition): True ONLY in the genuine CLOSED state — the provider is
        trusted. The leak-seed driver gates on this (not on "not cooling down") so it never fires a
        provider call in the post-cooldown-but-unproven window: after the cooldown elapses, ``open_until``
        is still set until ``should_call_claude`` runs the half-open probe, and during HALF_OPEN
        ``half_open`` is set — seeding must wait for the request path to PROVE recovery, else an idle seed
        burns a full author timeout on a still-wedged provider (codex L3 HIGH-1). It also can't steal that
        half-open probe, since it mutates nothing."""
        return self.open_until == 0.0 and not self.half_open

    def record(self, source: str) -> str | None:
        """Advance the state from a real attempt's result. Call ONLY after should_call_claude()
        returned True. Any PROVIDER success ('claude' | 'codex' | …) = success; 'stub' = a provider
        failure; 'error' = a payload/formatter problem (NOT a provider-health signal — never moves the
        breaker). Returns 'open'/'reopen' on a transition (for logging), else None."""
        if source not in ("stub", "error"):   # any provider success (claude/codex/…) resets the breaker
            self.fails = 0
            self.half_open = False
            return None
        if source != "stub":                  # 'error' — leave the breaker untouched
            return None
        if self.half_open:                    # the recovery probe failed → straight back to OPEN
            self.open_until = self.clock() + self.cooldown
            self.half_open = False
            self.fails = 0
            return "reopen"
        self.fails += 1
        if self.fails >= self.threshold:
            self.open_until = self.clock() + self.cooldown
            self.fails = 0
            return "open"
        return None


def _tally(served: dict) -> str:
    """The running served-by-source tally for the log line — dynamic so a 'codex' (or any provider)
    source shows up, instead of the old hard-coded claude/stub/err triple."""
    return " ".join(f"{k}={v}" for k, v in served.items())


# ---------------- P4b-ii: the leak-seed driver (the moat WRITE half, daemon-driven) ----------------
# The coach DRIVES leak graduation mid-session: when enough un-graduated reads have accumulated, it
# authors a NAMED, cited leak doc from them and commits it through anneal's immune system. This runs on
# THIS (the daemon) thread — leakstore's thread-local store (P4b-ii-a) makes that safe alongside the
# serve thread's read-recording. The slow claude/codex authoring call holds NO store lock. Mid-session +
# threshold-gated (Phill's call 2026-06-24): seeding DURING play is what lets P4b-i recall a leak in the
# SAME session it was learned. The threshold naturally re-arms (episodes_since_wrap resets on a commit);
# the min-interval backstops the failure mode where a rejected commit leaves the count high and would
# otherwise re-fire the runner every poll.
LEAK_SEED_THRESHOLD = int(os.environ.get("GUTCHECK_LEAK_THRESHOLD", "6"))
LEAK_SEED_MIN_INTERVAL = float(os.environ.get("GUTCHECK_LEAK_INTERVAL", "120"))
# The author call runs SYNCHRONOUSLY in the daemon's poll loop (same as a coach answer does), so it
# blocks request-serving + stop-event response while it runs. We bound it BELOW the coach CLAUDE_TIMEOUT
# (90s) so a seed stalls the loop less than a live coach answer would, and only fire it on an IDLE poll
# (no pending coach requests) so a waiting player never sits behind a seed. (L1 MED-1 / L2 Finding 7.)
LEAK_SEED_AUTHOR_TIMEOUT = float(os.environ.get("GUTCHECK_LEAK_AUTHOR_TIMEOUT", "60"))

_LEAK_AUTHOR_FRAME = """You're maintaining a poker player's LEAK MEMORY — the graduating record of \
their recurring read-errors. This is NOT in-the-moment coaching; it's the periodic compression that turns \
their recent committed reads into NAMED, cited leaks their coach recalls next time they're in the spot. If \
you know this player, use it. The doctrine below is the lens; where it says "the operator", that's the \
player."""

_LEAK_AUTHOR_RULES = """--- HOW TO WRITE THE LEAK MEMORY ---
You're handed (1) the operator's RECENT COMMITTED READS — each line starts with an 8-char episode id in \
parentheses, your citation handle — and (2) the CURRENT leak memory (leaks already named, or "none yet"). \
Recompose the FULL leak-memory document, folding the recent reads in.

NAME LEAKS IN OBSERVED-BEHAVIOR LANGUAGE, NEVER A PLAYER TYPE. Write "barreled into a player who only \
calls", never "barreled into a station". You never see a villain's hidden type and must never invent one — \
a leak is a pattern in THEIR reads and THEIR actions, not a label on an opponent.

A leak is a RECURRING read-error — a read whose action-implication they keep missing (a right read played \
wrong, or a read taken backwards). Only name a pattern you can cite from TWO OR MORE reads. If nothing \
recurs, return the current memory UNCHANGED.

Each leak goes under "## Patterns" in EXACTLY this line format (the leading "! " is optional emphasis):
  - <snake_case_name> | <N>x (<today>) [evidence: <id>, <id> "<explanation>"] - <one line of felt prose>
- <N> = the graduation count: 1 the first time a leak is named; for a leak ALREADY in the current memory \
  that recurs in these reads, KEEP its name and INCREMENT N by one.
- Cite the real 8-char ids (at least two, comma-separated) of the reads that evidence THIS leak.
- The "<explanation>" must share at least TWO meaningful words with the cited reads' own text — ground it \
  in what actually happened, not a slogan.
- CROSS-SESSION RULE: when you increment a leak named in a PRIOR session, the explanation must be FRESHLY \
  grounded in THESE reads — do not reword the prior one (an explanation that mostly reuses the prior \
  session's words is rejected as ungrounded). Same name, new evidence, new words.

Output ONLY the document — the "# " title line then the four sections (## State, ## Patterns, ## Decisions, \
## Context). No preamble, no commentary, no code fences."""


def _leak_author_prompt(pkg: dict) -> str:
    """The leak-authoring prompt: the frame, the shared doctrine, the rules, the recent reads (with their
    citable ids), and the current leak memory to recompose. Composed per pass so a doctrine.md edit takes
    effect without a daemon restart; doctrine.full() is fail-safe (never raises)."""
    episodes = pkg.get("episodes") or "(no recent reads)"
    continuity = pkg.get("continuity") or "(none yet — this is the first leak-memory pass)"
    today = pkg.get("today") or ""
    return (f"{_LEAK_AUTHOR_FRAME}\n\n--- DOCTRINE ---\n{doctrine.full()}\n\n{_LEAK_AUTHOR_RULES}\n\n"
            f"Today is {today}.\n\n--- RECENT COMMITTED READS (cite these ids) ---\n{episodes}\n\n"
            f"--- CURRENT LEAK MEMORY ---\n{continuity}")


def _extract_doc(text: str) -> str:
    """Pull the continuity doc out of the author's response, anchored on the one REQUIRED section
    ("## Patterns"), NOT on fence position. From that anchor: walk UP to the highest heading line with NO
    code-fence between it and ## Patterns (so a ```md wrapper ABOVE the title is INCLUDED — its headings
    are legitimately inside it — but a SEPARATE leading junk fence is EXCLUDED, since its closing ``` sits
    between its content and ## Patterns), then walk DOWN to the first fence line after ## Patterns (the
    wrapper's close / a trailing outro) or EOF. This survives every common chatty-LLM shape a naive fence
    strip mangles: preamble, a full ```md wrapper, a leading non-doc fenced block (with OR without its own
    heading), and a trailing fence + commentary (L1 HIGH-1/HIGH-2; codex L3 MED-1 — whose "headings only
    outside fences" fix would have wrongly rejected the common whole-doc-in-```md case). Returns '' if
    there's no "## Patterns" heading line at all (nothing worth committing); anneal's immune system is the
    REAL validator (grounding / cross-session-overlap / shrink gate), this is only transport clean-up."""
    if not text:
        return ""
    lines = text.split("\n")
    pat = next((i for i, ln in enumerate(lines) if ln.lstrip().startswith("## Patterns")), None)
    if pat is None:                                    # no real ## Patterns section line — refuse
        return ""
    start = pat                                        # walk UP to the doc's first heading, not crossing a fence
    for i in range(pat - 1, -1, -1):
        s = lines[i].lstrip()
        if s.startswith("```"):                        # a fence boundary above ## Patterns — don't cross it
            break
        if s.startswith("#"):                          # a contiguous heading (# title / ## State)
            start = i
    end = len(lines)                                   # walk DOWN to the first fence after ## Patterns / EOF
    for i in range(pat + 1, len(lines)):
        if lines[i].lstrip().startswith("```"):        # the wrapper's closing fence / a trailing outro
            end = i
            break
    return "\n".join(lines[start:end]).strip()


# A leak pattern line: optional bullet/marker, snake_case name, " | ", the level digits + "x".
# Captures (1) the lead-in, (2) name, (3) the " | " separator, (4) the level — for a level-only rewrite.
_PATTERN_LEVEL_RE = re.compile(
    r"^(\s*(?:[-*]\s*)?(?:!!|!|\?|✓|\*)?\s*)([A-Za-z][A-Za-z0-9_.\-]*)(\s*\|\s*)(\d+)x")


def _enforce_level_policy(doc: str, current_doc: str) -> str:
    """Structurally cap author-asserted graduation levels — DON'T trust the prompt to set them
    (``structural_invariants_beat_discipline``): a NEW leak starts at 1x, an EXISTING leak increments by
    at most one, max 3x. The author is TOLD this, but an LLM that over-levels a leak would otherwise
    surface it with inflated recall weight — a level it never earned (codex L3 HIGH-2: anneal validates
    citation grounding but NOT level transitions, and the marked line still parses at the claimed level).
    Caps DOWN only, never raises, so a legitimate 1x->2x->3x is untouched. The prior levels come from
    anneal's OWN pattern parser (lazy + guarded import, so this module stays import-safe without the
    optional [anneal] extra; absent it, prior is empty → every leak treated as new → capped to 1x)."""
    prior = {}
    try:
        from anneal_memory import extract_pattern_summaries
        for s in extract_pattern_summaries(current_doc or ""):
            prior[s.name] = s.level
    except Exception:                                  # no anneal / malformed current → treat all as new
        prior = {}
    out = []
    for line in doc.split("\n"):
        m = _PATTERN_LEVEL_RE.match(line)
        if m:
            name, claimed = m.group(2), int(m.group(4))
            ceiling = min(prior.get(name, 0) + 1, 3) if name in prior else 1
            allowed = min(claimed, ceiling)            # cap DOWN only — never raise the author's level
            if allowed != claimed:
                line = f"{m.group(1)}{name}{m.group(3)}{allowed}x{line[m.end():]}"
        out.append(line)
    return "\n".join(out)


def seed_leaks(leak_store, runner=call_claude) -> dict:
    """Run ONE leak-seed pass: prepare the wrap -> author the leak doc via ``runner`` -> commit it through
    anneal's immune system (graduation + demotion of ungrounded/sycophantic leaks). Returns a summary dict
    for logging. NEVER raises; on ANY failure it CANCELS the open wrap so a prepared-but-uncommitted
    snapshot can't brick the next pass. The ``runner`` call (claude/codex — the slow step) holds NO store
    lock; each store op takes the lock internally."""
    try:
        wrap = leak_store.prepare_leak_wrap()
        # The embedded daemon is the SOLE in-process leak-wrapper (the serve thread only records reads;
        # the external CLI daemon never gets a leak_store). So a wrap already "in progress" here can ONLY
        # be our own abandoned wrap — e.g. a prior pass whose cancel_leak_wrap ALSO failed. Reclaim it,
        # else one swallowed cancel-failure bricks graduation for the rest of the process (the dead-store
        # mode this slice exists to prevent — L2 Finding 6). Safe precisely because there's no legitimate
        # concurrent wrapper to stomp.
        if isinstance(wrap, dict) and wrap.get("status") == "wrap_in_progress":
            wrap = leak_store.prepare_leak_wrap(recover_abandoned=True)
    except Exception as e:                          # prepare must never crash the daemon loop
        return {"seeded": False, "reason": f"prepare error: {e!r}"}
    if not wrap or wrap.get("status") != "ready" or not wrap.get("wrap_token"):
        # not ready = no new reads to graduate, or a wrap is already in progress (we never auto-cancel
        # someone else's) — either way there's nothing to author this pass.
        return {"seeded": False, "reason": (wrap or {}).get("status") or "no wrap prepared"}
    token, pkg = wrap["wrap_token"], (wrap.get("package") or {})
    try:
        doc = _extract_doc((runner(_leak_author_prompt(pkg)) or "").strip())
        if not doc:
            leak_store.cancel_leak_wrap()
            return {"seeded": False, "reason": "author produced no usable doc"}
        # structural level cap BEFORE commit — the author can't mint an unearned graduation (codex HIGH-2).
        # pkg["continuity"] is the current leak memory at prepare time = the prior levels to cap against.
        doc = _enforce_level_policy(doc, pkg.get("continuity") or "")
        res = leak_store.commit_leaks(doc, token, affect_tag="sharpening", affect_intensity=0.6)
    except ValueError as e:                         # malformed doc / catastrophic-shrink rejection
        leak_store.cancel_leak_wrap()
        return {"seeded": False, "reason": f"rejected: {e}"}
    except Exception as e:
        leak_store.cancel_leak_wrap()
        return {"seeded": False, "reason": f"error: {e!r}"}
    if res is None:                                 # a swallowed anneal/OS error inside commit_leaks
        leak_store.cancel_leak_wrap()
        return {"seeded": False, "reason": "commit failed"}
    return {"seeded": True, "result": res}


def run(once: bool = False, interval: float = POLL_INTERVAL, runner=None, db=None,
        startup_probe: bool = True, stop_event=None, install_signal_handlers: bool = True,
        coaching_enabled=None, provider=None, leak_store=None) -> int:
    """Poll the channel and answer pending requests. SIGTERM/SIGINT (launchctl unload, Ctrl-C) stops
    the loop cleanly BETWEEN requests; a `claude -p` call already in flight runs to completion or its
    timeout first (launchd may SIGKILL it after the grace window — the row stays pending, but the
    dead worker's claim holds it until the lease lapses, so the next run re-answers it after
    max(coach.CLAIM_LEASE_S, 2*CLAUDE_TIMEOUT+30) seconds, not at once). Carries a half-open circuit breaker
    (a wedged claude → serve stubs), stale-decision expiry, and periodic channel pruning.

    Embeddable (the in-process path — server.py runs this in a background thread): pass
    `install_signal_handlers=False` (signal.signal only works in the MAIN thread — registering off it
    raises) and a `stop_event` (a threading.Event set on shutdown) to end the loop. Signal
    registration is ALSO auto-skipped off the main thread (belt-and-suspenders), so a thread caller
    that forgets the flag won't crash — but then a `stop_event` is the ONLY way to stop the loop (with
    neither, it runs forever). The standalone process keeps the defaults — signals installed, no
    external event — so its behavior is unchanged."""
    db = db or paths.coach_db()
    # When coaching is toggled OFF, serve the honest facts-only floor (zero claude spend) —
    # the SAME path the breaker / stale-decision branches use. A None callable means always-on
    # (the standalone daemon's unchanged default); the embedded server injects
    # settings.coaching_enabled so the in-UI toggle reaches the worker without coupling this
    # module to the settings store. Read per-request, so a toggle takes effect on the next ask.
    is_enabled = coaching_enabled if coaching_enabled is not None else (lambda: True)
    # provider→runner (slice c): an explicitly-injected runner (tests) wins; else map the LIVE provider
    # setting → its runner per request, so switching provider in the UI takes effect on the next ask.
    # The embedded server injects provider=settings.coach_provider; None → claude (unchanged default).
    get_provider = provider if provider is not None else (lambda: "claude")
    def pick_runner():
        return runner if runner is not None else _runner_for(get_provider())
    stop = {"flag": False}

    def _sig(_signum, _frame):
        stop["flag"] = True

    # signal.signal raises off the main thread; gate on BOTH the flag and main-thread-ness so an
    # embedded caller that forgets install_signal_handlers=False still can't trigger that ValueError.
    if install_signal_handlers and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _sig)
        signal.signal(signal.SIGINT, _sig)

    def _stopping() -> bool:
        return stop["flag"] or (stop_event is not None and stop_event.is_set())

    claim_lease = max(coach.CLAIM_LEASE_S, 2 * CLAUDE_TIMEOUT + 30)   # outlasts any call this worker can make
    worker_id = f"{os.getpid()}-{threading.get_ident()}"      # unique per loop, even two in one process
    breaker = _Breaker(BREAKER_THRESHOLD, BREAKER_COOLDOWN)
    if startup_probe and not once and not _stopping():   # a stop set before we start → skip the probe's claude call
        if not _probe_subscription(pick_runner()):
            breaker.trip_cold()               # probe failed → start degraded (stubs, no per-req stall)

    served = {"claude": 0, "stub": 0, "error": 0}
    last_clean = time.monotonic()
    # P4b-ii leak-seed cooldown clock — start armed so the first seed waits one interval into the session
    # (the operator needs ~THRESHOLD reads to accumulate anyway; this avoids a startup thundering-herd if
    # a prior session left un-graduated reads). leak_store is None under --no-coach-style standalone runs
    # that don't pass it, and always None on the legacy external daemon CLI (the in-process server wires it).
    last_seed = time.monotonic()

    print(f"[coach_daemon] up — model={COACH_MODEL} interval={interval}s "
          f"{'(single pass)' if once else '(polling)'}", file=sys.stderr)
    while not _stopping():
        try:
            pending = coach.pending(db)
        except Exception as e:                 # transient sqlite error (locked/corrupt) — degrade, don't crash
            print(f"[coach_daemon] channel read failed: {e!r} — will retry", file=sys.stderr)
            pending = []
        for req in pending:
            if _stopping():
                break
            try:                               # one worker per request: the loser skips, never spends
                if not coach.claim(req["id"], worker_id, lease=claim_lease, path=db):
                    continue
            except Exception as e:             # transient sqlite error — leave the row for the next pass
                print(f"[coach_daemon] claim #{req.get('id')} failed: {e!r} — will retry", file=sys.stderr)
                continue
            try:
                coaching_off = not is_enabled()            # the in-UI toggle (cost floor)
                if coaching_off or _is_stale_decision(req, time.time()):  # per-request: a backlog can age a decision mid-pass
                    # coaching toggled OFF, or the decision aged past the live window: skip the
                    # claude SPEND either way, but still answer with the honest facts-only floor —
                    # a still-polling player (the browser polls ~60s) gets band / equity / pot-odds,
                    # never a dead placeholder. Neither is a claude-health signal, so the breaker is
                    # untouched (we `continue` past breaker.record). COUNT the stub: with coaching OFF
                    # this is the NORMAL path, so an uncounted answer would make the served tally read
                    # 0 while the floor is in fact serving every request. facts_mode=coaching_off →
                    # deliberate OFF serves CLEAN facts (no head); a STALE spot while coaching is ON
                    # keeps the degraded head (the coach was wanted, the spot just aged before it answered).
                    responded, source = answer_one(req, runner=_stub_only, db=db, facts_mode=coaching_off)
                    if responded:
                        served[source] = served.get(source, 0) + 1
                    reason = "coaching OFF" if coaching_off else "stale decision"
                    print(f"[coach_daemon] {reason} #{req['id']} → facts-only stub (no claude spend) "
                          f"[{_tally(served)}]", file=sys.stderr)
                    continue
                if breaker.should_call_claude():
                    responded, source = answer_one(req, runner=pick_runner(), db=db,
                                                   success_source=get_provider())
                    transition = breaker.record(source)
                    if transition == "open":
                        print(f"[coach_daemon] ⚠ CIRCUIT BREAKER OPEN — coach provider failing; serving stubs "
                              f"for {BREAKER_COOLDOWN:.0f}s", file=sys.stderr)
                    elif transition == "reopen":
                        print(f"[coach_daemon] ⚠ CIRCUIT BREAKER RE-OPEN — recovery probe failed; stubs "
                              f"for another {BREAKER_COOLDOWN:.0f}s", file=sys.stderr)
                else:
                    responded, source = answer_one(req, runner=_stub_only, db=db)
                if responded:
                    served[source] = served.get(source, 0) + 1
                    print(f"[coach_daemon] answered #{req['id']} ({req['kind']}) via {source} "
                          f"[{_tally(served)}]",
                          file=sys.stderr)
            except Exception as e:             # a row-level / transient error must NOT kill the daemon
                print(f"[coach_daemon] loop error on #{req.get('id')}: {e!r}", file=sys.stderr)
                try:                           # drain the row so a persistent error can't busy-spin it
                    if not coach.respond(req["id"], _SAFE_FALLBACK, db):
                        coach.release(req["id"], worker_id, db)
                except Exception:
                    try:                       # the drain failed too: hand the claim back, don't sit on it for the lease
                        coach.release(req["id"], worker_id, db)
                    except Exception:
                        pass
                continue

        if not once and (time.monotonic() - last_clean) >= CLEAN_INTERVAL_S:
            deleted = coach.clean(path=db)
            last_clean = time.monotonic()
            if deleted:
                print(f"[coach_daemon] pruned {deleted} old channel row(s)", file=sys.stderr)

        # P4b-ii: the leak-seed driver — mid-session, threshold-gated. The coach DRIVES graduation when
        # enough new reads have accumulated AND coaching is on AND the provider isn't wedged (breaker
        # closed — gated on the non-mutating is_closed, i.e. the provider is PROVEN-good, not merely
        # "not cooling down": seeding in the post-cooldown-unproven / half-open window would burn a full
        # author timeout on a still-wedged provider, codex L3 HIGH-1) AND this poll had NO pending requests
        # (idle): the author call is SYNCHRONOUS in this loop, so seeding only when idle keeps a waiting
        # player from sitting behind it (L1 MED-1 / L2 F7). last_seed is bumped whether the pass seeds,
        # no-ops, or fails: the min-interval is the anti-hammer backstop (a rejected commit leaves the count
        # high, which would otherwise re-fire the runner every idle poll). The pass itself never raises
        # (seed_leaks is fully guarded). The author runner gets a SHORTER timeout than a coach answer
        # (LEAK_SEED_AUTHOR_TIMEOUT) to bound the loop-block; pick_runner returns a real (prompt, timeout)-
        # accepting runner in production (the server injects no runner).
        if (leak_store is not None and not once and not pending and is_enabled()
                and breaker.is_closed()
                and (time.monotonic() - last_seed) >= LEAK_SEED_MIN_INTERVAL):
            last_seed = time.monotonic()
            try:
                count = leak_store.new_read_count()
                if count >= LEAK_SEED_THRESHOLD:
                    seed_runner = functools.partial(pick_runner(), timeout=LEAK_SEED_AUTHOR_TIMEOUT)
                    outcome = seed_leaks(leak_store, runner=seed_runner)
                    if outcome.get("seeded"):
                        r = outcome.get("result")
                        grad = r.get("graduations_validated") if isinstance(r, dict) else None
                        print(f"[coach_daemon] leak-seed: authored leaks from {count} new reads "
                              f"(graduations_validated={grad})", file=sys.stderr)
                    else:
                        print(f"[coach_daemon] leak-seed skipped at {count} reads: "
                              f"{outcome.get('reason')}", file=sys.stderr)
            except Exception as e:                 # the seed pass must NEVER take down the daemon loop
                print(f"[coach_daemon] leak-seed error: {e!r}", file=sys.stderr)

        if once:
            break
        # signal-responsive idle: short hops so SIGTERM / a stop_event lands within ~0.25s, not a full interval
        slept = 0.0
        while slept < interval and not _stopping():
            step = min(0.25, interval - slept)
            time.sleep(step)
            slept += step
    print("[coach_daemon] stopping.", file=sys.stderr)
    if leak_store is not None:
        # close THIS (daemon) thread's thread-local leak-store connection so its WAL handle is
        # checkpointed + released on clean shutdown (the serve thread owns + reclaims its own). close() is
        # thread-scoped, so this only touches the daemon connection — exactly right (L1 LOW-1 / L2 F1).
        try:
            leak_store.close()
        except Exception:                            # shutdown best-effort: never raise out of the loop
            pass
    return 0


def answer_id(req_id: int, db=None) -> int:
    """Answer one specific pending request by id (debugging / manual one-shot)."""
    db = db or paths.coach_db()
    req = coach.get(req_id, db)
    if req is None:
        print(json.dumps({"error": "not found", "id": req_id}))
        return 1
    if req.get("status") != "pending":
        print(json.dumps({"id": req_id, "skipped": "already answered"}))
        return 0
    responded, source = answer_one(req, db=db)
    print(json.dumps({"id": req_id, "answered": responded, "source": source}))
    return 0 if responded else 1


def smoke() -> int:
    """Fire ONE trivial claude -p call and report result + timing. The out-of-session verification
    path: run this to confirm the subscription call works on this machine before trusting the daemon."""
    t0 = time.time()
    out = call_claude("Reply with exactly one word, nothing else: PONG", timeout=CLAUDE_TIMEOUT)
    dt = time.time() - t0
    ok = "PONG" in out.upper()
    result = {"ok": ok, "elapsed_s": round(dt, 1), "response": out[:200]}
    if not out:
        result["reason"] = ("claude -p returned nothing — a timeout, spawn error, is_error, or a "
                            "logged-out claude. See stderr above.")
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="gutcheck headless coach daemon (Slice 2)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="poll the channel and answer pending requests via claude -p")
    r.add_argument("--once", action="store_true", help="single pass over the queue, then exit")
    r.add_argument("--interval", type=float, default=POLL_INTERVAL, help="idle poll interval (s)")

    a = sub.add_parser("answer", help="answer one specific pending request via claude -p")
    a.add_argument("--id", type=int, required=True)

    sub.add_parser("smoke", help="fire one trivial claude -p call; print result + timing")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        return run(once=args.once, interval=args.interval)
    if args.cmd == "answer":
        return answer_id(args.id)
    if args.cmd == "smoke":
        return smoke()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
