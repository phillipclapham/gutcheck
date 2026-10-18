"""gutcheck local web server — stdlib only, zero deps.

Serves the browser table and a small JSON API over game.py. Single-player; binds
to loopback by default. Pass --host <your Tailscale IP> to play on the iPad over
the tailnet (Slice 0d). The act/sit/reload/getup/reset endpoints MUTATE and have
NO auth, so a non-loopback bind is locked to a Tailscale CGNAT address
(100.64.0.0/10) — the WireGuard mesh is the only access control. Wildcard, public,
and shared-LAN binds are refused.

    gutcheck [--port 8777] [--host 100.x.y.z]      (installed console script)
    python -m gutcheck.server [--port 8777]         (module form, from a clone)
    open http://127.0.0.1:8777              (local)
    open http://<laptop-tailscale-ip>:8777  (iPad on the tailnet)
"""

from __future__ import annotations

import argparse
import ipaddress
import os
import json
import hashlib
import signal
import socket
import sys
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

from . import coach
from . import coach_daemon
from . import equity
from . import keystore
from . import leakstore
from . import paths
from . import settings
from .game import Game
from .seam import Seam

HERE = Path(__file__).parent
STATIC = HERE / "static"
# Built in main() AFTER the data dir is ensured (and an optional --reset applied), so a clean
# slate isn't loaded-then-overwritten. Handlers run only under serve_forever(), by which point
# main() has assigned it.
GAME: Game | None = None
# The seam (Layer B): the transport-neutral coach contract over GAME + the history store.
# Built right after GAME in main(); both readiness guards check GAME-or-SEAM, so neither the
# one-statement init window nor a future reorder can dereference a None seam.
SEAM: Seam | None = None

# Whether the BUILT-IN coach worker is running this process (False under --no-coach, where an
# external worker owns coaching). The settings surface reports it so the UI never offers a live
# coaching toggle that an external worker wouldn't honor. Set in main() when the thread starts.
COACH_BUILT_IN: bool = False

# RFC 6598 shared-address space — Tailscale's IPv4 (CGNAT) range. A non-loopback
# bind must land here: only this operator's own enrolled tailnet devices can route
# to it, which is the entire auth story for the unauthenticated, WRITABLE API.
_TAILSCALE_CGNAT = ipaddress.ip_network("100.64.0.0/10")

# Request bodies are tiny JSON ({"action": "...", "amount": N}); refuse anything that
# claims to be larger so a hostile Content-Length can't drive a huge or hanging read.
_MAX_BODY = 64 * 1024
# Set by main() after the bind succeeds — the canonical IP this server is actually on.
# The anti-DNS-rebinding Host check (Handler._host_ok) compares against it.
BOUND_HOST = "127.0.0.1"
# The machine's bare name == its Tailscale MagicDNS short label (Tailscale derives the
# device name from the hostname). Lets the iPad reach the box by short name, not just FQDN.
_LOCAL_HOSTNAME = socket.gethostname().split(".")[0].lower()


def _decision_spot(ctx: dict) -> dict:
    """The minimal spot the browser needs to render the live gut-commit drill: the board, the
    hero's hand, the price, and how many cards are still to come. NO equity (that's the worker's
    job, revealed only post-commit). `has_model` is True iff game.py baked a modeled range — i.e.
    there is a live opponent to model equity against; if not, the app skips the beat and just asks
    for prose."""
    spec = ctx.get("modeled_spec")
    board = ctx.get("board") or []
    hero = ctx.get("hero") or {}
    spot = {
        "has_model": spec is not None,
        "street": ctx.get("street"),
        "board": board,
        "hero_hole": hero.get("hole"),
        "pot": ctx.get("pot"),
        "to_call": ctx.get("to_call"),
        "cards_to_come": max(0, 5 - len(board)),
    }
    # the can_* booleans the live drill's lean buttons need — esp. can_raise: without it the drill
    # reads `spot.legal.can_raise` as undefined and silently offers only fold/call when facing a bet,
    # dropping the raise lean (the exact gap the raise-lean fix closed for the review drill but never
    # reached on the live surface — _decision_spot never carried `legal`). The raise-SIZING amounts are
    # deliberately NOT exposed: read-logging picks a lean, it does not size a bet.
    legal = ctx.get("legal") or {}
    spot["legal"] = {"can_fold": legal.get("can_fold"), "can_check": legal.get("can_check"),
                     "can_raise": legal.get("can_raise")}
    if spec:
        spot["live_count"] = spec.get("live_count")
        spot["vs"] = (spec.get("vs") or {}).get("name")
    return spot


def _spot_token(ctx: dict) -> str:
    """A stable identity for the EXACT decision a live drill rendered, round-tripped /api/read/spot →
    /api/read/commit so a read committed from a STALE drill is REJECTED, not logged against the wrong
    spot. The hole this closes: the app serves cross-device (iPad + laptop), and `syncState` is
    suppressed while a drill is open — so a second tab/device can advance the hand to a LATER hero
    decision in the SAME hand while this tab keeps showing the old drill. coach_decision_context() would
    then return a valid-but-DIFFERENT spot, and the read would log against it + wrong-join in P3a. This
    token makes "committing a read doesn't advance the hand" a STRUCTURAL guarantee, not a comment:
    covers everything that distinguishes two hero decisions in one hand — the frozen join key + street +
    board + price. Deterministic across calls on the same frozen decision (spec-building has no
    randomness; the MC is separate), so the same decision → the same token, ANY advance → a different
    one. Identity only (ordinals + the public board), never the truth — the server still recomputes the
    equity itself (anti-fab intact)."""
    board = ctx.get("board") or []
    key = (ctx.get("session_id"), ctx.get("hand_no"), ctx.get("n"), ctx.get("street"),
           tuple(board), ctx.get("to_call"), ctx.get("pot"))
    return hashlib.sha256(repr(key).encode()).hexdigest()[:16]


# ---------------- P4b-i: leak recall-injection (the moat READ half) ----------------
def _vpip_behavior(vpip) -> str:
    """One OBSERVED-BEHAVIOR phrase for a villain's player-visible VPIP, for the leak-recall query.

    §3f / MED-1: VPIP measures LOOSENESS ONLY (how often a seat voluntarily enters pots) — it says
    NOTHING about passive-vs-aggressive (that is PFR/AF). So this maps to looseness language only;
    asserting 'passive'/'calling' here would be an inference NOT derivable from VPIP, and worse, in the
    current 3-archetype roster (rock/reg/station) only the station is high-VPIP — so 'passive calling'
    would silently re-encode the hidden archetype the §3f gate exists to withhold (a spoiler the moment
    a loose-AGGRESSIVE type or a 9-max seat lands). Looseness IS player-visible from VPIP; behavior
    beyond it must come from the observable PFR/AF, not from VPIP. (Buckets mirror coach_watch's 15/45.)
    ⚠ If the archetype roster gains a loose-aggressive type, REVISIT — do not add passivity words here."""
    if vpip is None:
        return ""
    if vpip < 15:
        return "very tight villain"
    if vpip > 45:
        return "loose wide villain"
    return "balanced villain"


def _spot_query(ctx: dict) -> str:
    """Build the relevance query for a live DECISION spot — from PLAYER-VISIBLE signals ONLY (the
    street, the hero's strength, the facing-action shape, and observed villain behavior). It NEVER
    reads the hidden ``_truth`` archetype, so the recall query — like the leaks it matches — stays
    §3f-clean. Decision-only by design (codex L3 MED): the review ctx has no per-decision street/action,
    so it is not fed here; review recall is deferred until the replay UI signals an in-focus decision."""
    parts: list[str] = []
    if ctx.get("street"):
        parts.append(ctx["street"])
    hero = ctx.get("hero") or {}
    if hero.get("strength"):
        parts.append(hero["strength"])
    # facing-action shape: a price to call frames a value/bluff read; a checked line frames a different one.
    to_call = ctx.get("to_call")
    if isinstance(to_call, (int, float)) and to_call > 0:
        parts.append("facing a bet, value spot")
    elif to_call == 0:
        parts.append("checked to, thin value or bluff")
    # NOTE: the hand's OUTCOME (won/lost) is deliberately NOT in the relevance query (LOW-2). Outcome is
    # not read-quality — the doctrine trains the opposite (a right read can still get crushed), so leak
    # relevance must match the read-error SHAPE, not who won.
    # observed villain behavior (player-visible VPIP only); skip folded seats.
    for v in ctx.get("villains", []) or []:
        if v.get("in_hand") is False:
            continue
        phrase = _vpip_behavior((v.get("observed") or {}).get("vpip_pct"))
        if phrase:
            parts.append(phrase)
    return " ".join(p for p in parts if p)


def _attach_relevant_leaks(ctx: dict) -> dict:
    """Recall the operator's GRADUATED leaks relevant to this spot and attach them as
    ``ctx['relevant_leaks']`` (a possibly-empty list of ``{name, level, summary}``), so the coach
    worker renders them in the live context (the recall-injection that makes the leak memory an immune
    system, not a dead store). Best-effort: built server-side on the single serve thread where the leak
    store lives (so the in-process daemon thread never touches it — the P4a thread-safety gate stays
    closed), and any miss/unavailable just attaches ``[]``. Returns ctx for chaining.

    Fully guarded (MED-2): the recall is pure enrichment, so the WHOLE path — including the query
    builder, which runs BEFORE the seam's own error swallowing — degrades to no leak section on ANY
    exception, never 500-ing /api/coach/ask. (The serve-built ctx is well-formed today; this makes the
    never-break contract structural at the enrichment boundary, not dependent on every upstream ctx.)"""
    try:
        res = SEAM.recall_relevant_leaks(_spot_query(ctx))
        ctx["relevant_leaks"] = res.get("leaks") or [] if isinstance(res, dict) else []
    except Exception:
        ctx["relevant_leaks"] = []
    return ctx


def _run_coach(db: Path, stop: threading.Event, leak_store=None) -> None:
    """The built-in coach worker (a background thread): the FULL coach_daemon loop, in-process. It
    coaches each request as flow on the Claude subscription (`claude -p`) and degrades on a hard
    floor — claude → honest facts-only stub → safe fallback — so the felt-sense loop runs from
    `gutcheck` alone and a request is ALWAYS answered. Carries the daemon's circuit breaker,
    stale-decision expiry, and channel pruning. Signal handlers are OFF (the main thread owns those —
    registering off a thread raises); `stop` (set on shutdown) ends the loop within ~0.25s.

    Real coaching needs `claude` logged in. Starting `gutcheck` from inside a Claude Code session is
    fine: _claude_env strips the nesting sentinels from the child. --no-coach skips this entirely (run
    your own worker).

    A second worker (an external coach_daemon/launchd) on the same coach.db is safe: each request is
    claimed by one worker before it is coached (coach.claim)."""
    try:
        # startup_probe=False (embedded only): the standalone daemon probes because the operator starts
        # it deliberately; embedded, a slow/hung probe would block the loop before the first request is
        # served, so a hand-1 decision could age toward the stale window. The breaker trips on the first
        # real request instead.
        # coaching_enabled + provider wire the in-UI settings to the worker (both read per-request, so
        # toggling coaching off → zero-claude facts floor, and switching provider → claude|codex, are LIVE).
        # leak_store (P4b-ii): the daemon DRIVES leak-seeding mid-session on ITS thread — the same
        # LeakStore instance the serve thread records reads into, made cross-thread-safe by leakstore's
        # thread-local store (P4b-ii-a). None when the [anneal] extra is absent → seeding is a no-op.
        coach_daemon.run(db=db, stop_event=stop, install_signal_handlers=False, startup_probe=False,
                         coaching_enabled=settings.coaching_enabled, provider=settings.coach_provider,
                         leak_store=leak_store)
    except Exception as e:           # the worker thread must never take the server down with it
        # the loop died → coaching is genuinely OFF for this run. Tell the truth on the settings
        # surface (COACH_BUILT_IN means "the built-in worker is RUNNING", not "we tried to start it"):
        # leaving it True would offer a live toggle + "talk it through" prose that nothing can answer.
        global COACH_BUILT_IN
        COACH_BUILT_IN = False
        print(f"[coach] worker loop crashed: {e!r} — coaching is OFF for this run", file=sys.stderr)


def _all_key_status() -> dict:
    """The SAFE BYO-key status for BOTH metered vendors — ``{anthropic: {set,hint,source}, openai: {…}}``,
    never a key. The single source for the /api/secrets reads and the settings-payload fold-in, so the two
    can't drift. key_status enforces the never-the-key invariant per provider."""
    return {"anthropic": keystore.key_status("anthropic"), "openai": keystore.key_status("openai")}


def _settings_payload() -> dict:
    """The settings surface, shared by GET and POST /api/settings. Carries the coaching toggle, the
    facts-floor stat-tailoring prefs, and `coach_built_in` (False under --no-coach → the UI shows the
    coaching toggle read-only, since an external worker owns coaching). `api_key` / `openai_key` are the
    SAFE status of each metered worker's BYO key — ``{set, hint, source}`` ONLY, never the key (which lives
    in secrets.json and only ever moves through the dedicated /api/secrets write surface). Folding the
    read-only status in here lets the settings menu render the provider section in one fetch."""
    p = settings.stat_prefs()
    p["coaching_enabled"] = settings.coaching_enabled()
    p["coach_provider"] = settings.coach_provider()
    p["api_model"] = settings.api_model()          # the anthropic_api worker's cost dial
    p["api_effort"] = settings.api_effort()        # the anthropic_api worker's fidelity×latency dial
    p["openai_model"] = settings.openai_model()    # the openai_api worker's cost dial
    p["openai_effort"] = settings.openai_effort()  # the openai_api worker's fidelity×latency dial
    p["coach_built_in"] = COACH_BUILT_IN
    keys = _all_key_status()
    p["api_key"] = keys["anthropic"]               # back-compat name = the anthropic key status
    p["openai_key"] = keys["openai"]
    return p


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _json(self, obj: dict, code: int = 200) -> None:
        self._send(code, json.dumps(obj).encode(), "application/json")

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0) or 0)
        if length <= 0:  # <=0 (not just falsy): a negative Content-Length would make
            return {}     # rfile.read(-1) block reading to EOF — a hang now reachable off-box
        # malformed JSON must RAISE (caught by do_POST → 400) — swallowing it to {}
        # silently reset the game on a bad /api/reset.
        return json.loads(self.rfile.read(length).decode() or "{}")

    def _same_origin(self) -> bool:
        """Cross-origin write guard (CSRF). The POST endpoints mutate with no auth, so
        once 0d makes them reachable off-box, a page on ANY other origin a tailnet
        browser loads could drive them (e.g. fetch('…/api/reset')) — these are simple
        no-preflight POSTs, so CORS does not stop the mutation, only the response read.

        Defense: if an Origin header is present, its host:port must equal this request's
        Host (the request came from the page WE served, not a foreign one). A cross-origin
        browser fetch always carries Origin, so absent ⇒ not a browser CSRF vector ⇒ allow
        (covers curl and same-origin legacy clients). HOST-AGNOSTIC by construction: works
        whether the app was loaded by Tailscale IP or MagicDNS name, since both Origin and
        Host reflect whatever URL the app was actually served under. (Origin==Host alone
        does NOT catch DNS-rebinding — both sides become the attacker's name post-rebind —
        so `_host_ok` adds the Host-allowlist that closes it.)"""
        origin = self.headers.get("Origin")
        if not origin or not origin.strip():
            return True  # absent/empty Origin ⇒ not a browser CSRF vector (curl/legacy clients)
        return self._netloc_matches_host(origin)

    def _host_ok(self) -> bool:
        """Anti-DNS-rebinding: the Host header must NAME this server, not an attacker's
        domain that rebound to our IP. The Origin==Host check alone can't catch rebinding
        (post-rebind Origin and Host are both the attacker's name and match).

        Parse STRICTLY (Host is the security boundary): exactly one Host header, no
        userinfo, validated host:port. Then allow only: loopback (parsed as an IP — a
        ``startswith("127.")`` string test would admit ``127.0.0.1.evil.com``), the
        actually-bound IP, the machine's bare name (= the MagicDNS short label), or a
        Tailscale MagicDNS FQDN. The ``.ts.net`` suffix is DEFENSE-IN-DEPTH, not a
        standalone guarantee: those names resolve only inside a tailnet, and the CGNAT
        bind already limits reach to this operator's tailnet — the suffix backstops that
        boundary (it would admit ANY tailnet's name, but reaching the box needs ours)."""
        hosts = self.headers.get_all("Host", [])
        if len(hosts) != 1:                  # 0 (HTTP/1.0) or >1 (smuggling) — refuse
            return False
        raw = hosts[0].strip()
        if not raw or "@" in raw:            # a Host header never carries userinfo
            return False
        try:
            parsed = urlsplit("//" + raw)    # proper host:port parse (handles IPv6 brackets)
            _ = parsed.port                  # force port validation (raises on a bad port)
        except ValueError:
            return False
        name = (parsed.hostname or "").removesuffix(".").lower()
        if not name:
            return False
        if name == "localhost":
            return True
        try:
            if ipaddress.ip_address(name).is_loopback:   # 127/8 (and ::1) — structurally
                return True
        except ValueError:
            pass                             # a hostname, not an IP literal
        return name == BOUND_HOST or name == _LOCAL_HOSTNAME or name.endswith(".ts.net")

    def _netloc_matches_host(self, origin: str) -> bool:
        try:
            netloc = urlsplit(origin).netloc
        except ValueError:
            return False  # malformed Origin (e.g. unterminated IPv6 bracket) ⇒ treat as cross-origin
        # case-insensitive: hostnames (incl. MagicDNS names) are case-insensitive per RFC 3986;
        # the port is digits, so lowercasing the whole netloc is safe and avoids a false 403.
        return netloc.lower() == self.headers.get("Host", "").lower()

    def do_GET(self):  # noqa: N802
        # Safety net mirroring do_POST: a handler crash must return a clean JSON 500, NEVER drop the
        # connection. A dropped GET surfaces to the MCP proxy as RemoteDisconnected — misclassified as
        # "server down" — and bypasses the proxy's error sanitization (L1 H1). Real error -> stderr;
        # the response body stays generic so no internals leak even to a direct caller.
        try:
            self._route_get()
        except Exception as e:
            print(f"[server] GET {self.path} crashed: {type(e).__name__}: {e}", file=sys.stderr)
            try:
                self._json({"error": "internal error"}, code=500)
            except Exception:
                pass

    def _route_get(self):  # noqa: N802
        if not self._host_ok():
            self._send(403, b"host not allowed", "text/plain")
            return
        if GAME is None or SEAM is None:    # only if a handler runs before main() builds them (e.g. a
            self._send(503, b"server initializing", "text/plain")   # direct test dispatch) — never on serve
            return
        if self.path in ("/", "/index.html"):
            f = STATIC / "index.html"
            if f.exists():
                self._send(200, f.read_bytes(), "text/html; charset=utf-8")
            else:
                self._send(404, b"index.html missing", "text/plain")
            return
        if self.path == "/api/state":
            self._json(SEAM.get_table_state())
            return
        if self.path == "/api/coach/state":
            # the COACH-facing table state — the same live table with the villain ARCHETYPE STRIPPED
            # (§3f). The MCP get_table_state tool proxies THIS, never /api/state (which carries `type`
            # for the browser's click-to-reveal self-test). A harmless read. [codex L3 HIGH]
            self._json(SEAM.get_coach_table_state())
            return
        if self.path == "/api/career":
            self._json(GAME.career_view())
            return
        if self.path == "/api/settings":
            # the settings menu's read side (coaching on/off today). A pure read — the POST
            # twin mutates. GET (no body) routes here; the frontend api() helper sends a body
            # for the POST so it can't be misrouted to this read.
            self._json(_settings_payload())
            return
        if self.path == "/api/secrets":
            # both metered workers' BYO-key STATUS — ``{set, hint, source}`` per vendor, NEVER the key
            # (keystore.key_status enforces that). The canonical secrets read; the settings menu also
            # gets this status folded into /api/settings, so the UI doesn't call this directly. A pure
            # read with no secret in the payload — same loopback/Tailscale access boundary as every route.
            self._json(_all_key_status())
            return
        if self.path == "/api/calibration":
            # the read-accuracy trend (gut-vs-truth over time). A harmless GET — pure read.
            self._json(SEAM.get_calibration())
            return
        parsed = urlsplit(self.path)
        if parsed.path == "/api/history":
            # the replayable hand stream over the seam (kills the external poller's tail-gap; the
            # MCP adapter reads the same SEAM.get_history next). EXACT-path match (not startswith) so a
            # future /api/history/<sub> route isn't shadowed and garbage sub-paths 404. Harmless GET.
            q = parse_qs(parsed.query)
            try:
                since = int(q.get("since", ["0"])[0] or 0)
            except (ValueError, IndexError):
                self._send(400, b"bad since", "text/plain")
                return
            # ?limit=<n> caps rows (opt-in; default 500 keeps the prior behavior for forward-pollers).
            # Clamped [1, 1000] so a direct caller can't ask the seam to serialize an unbounded stream.
            try:
                limit = int(q.get("limit", ["500"])[0] or 500)
            except (ValueError, IndexError):
                self._send(400, b"bad limit", "text/plain")
                return
            limit = max(1, min(limit, 1000))
            # ?latest=1 returns the most-recent `limit` hands (the coach window) instead of the oldest;
            # absent/falsey → the original oldest-first forward window (default, unchanged).
            latest = q.get("latest", ["0"])[0].lower() in ("1", "true", "yes", "on")
            # ?session=<id> scopes to one session ("replay session X"); absent/empty → None → all
            # sessions (the cross-session stream), same convention as /api/read_reconciliation.
            sid = q.get("session", [None])[0] or None
            self._json(SEAM.get_history(since, session_id=sid, limit=limit, latest=latest))
            return
        if parsed.path == "/api/sizing_tells":
            # P3(c) — per-villain observed betting→shown-strength tell. A pure read off the showdown-
            # revealed history; the §3f archetype gate is in the seam (behavior facts only, no type).
            # ?session=<id> scopes to one session; absent/empty → None → the live session.
            q = parse_qs(parsed.query)
            sid = q.get("session", [None])[0] or None
            self._json(SEAM.get_sizing_tells(sid))
            return
        if parsed.path == "/api/ground_truth":
            # the coach's calibratable answer-key for a stored hand (exact equity + showdown). The
            # villain ARCHETYPE is GATED at the seam (default off — §3f structural pedagogy gate). A
            # read; the bind guard + loopback/Tailscale boundary is the access control, same as every
            # route. This hands TRUTH to the coach surface BY DESIGN — it is never wired into the
            # browser UI (the human earns the read; the coach calibrates it).
            q = parse_qs(parsed.query)
            sid = q.get("session", [None])[0]
            try:                                     # query params are STRINGS — parse, don't _is_int
                hand = int(q.get("hand", [""])[0])
            except (TypeError, ValueError):
                self._send(400, b"bad or missing hand", "text/plain")
                return
            if not sid:
                self._send(400, b"missing session", "text/plain")
                return
            # reveal_archetype is NOT exposed over HTTP — the §3f gate is not just MCP-only: a direct or Tailscale-bound HTTP caller must not be able to flip it. The seam keeps
            # the reveal capability for a future operator-only LOCAL path; no networked transport opts in.
            # [codex L3 MED1]
            self._json(SEAM.get_ground_truth(sid, hand, reveal_archetype=False))
            return
        if parsed.path == "/api/read_reconciliation":
            # P3a — modeled−exact reconciliation ("read the range right, still crushed"). Pure read;
            # the exact equity is recomputed server-side from the canonical store (never client-
            # supplied), so this inherits the anti-fab invariant like every route. ?session=<id> scopes
            # to one session (the live coach + the hand-end range-collapse card); OMIT it for the
            # cross-session collapse trend in 🎯 Reads (all joinable reads across sessions).
            q = parse_qs(parsed.query)
            sid = q.get("session", [None])[0] or None   # absent / empty → None → all sessions
            self._json(SEAM.get_read_reconciliation(sid))
            return
        if parsed.path == "/api/leaks":
            # P4 — the operator's graduated-leak memory (anneal-backed, the moat). A pure read: the
            # current leaks doc (the leaks the coach recalls at the decision point), or an unavailable
            # marker when gutcheck[anneal] isn't installed. Leak NAMES are behavior-language only — the
            # §3f archetype gate is upstream (episodes written WITHOUT the type), so no spoiler can
            # surface here by construction. Harmless GET.
            self._json(SEAM.get_leaks())
            return
        if self.path == "/favicon.ico":
            icon = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 16 16">'
                    '<rect width="16" height="16" rx="3" fill="#1d6b46"/>'
                    '<text x="8" y="12" font-size="11" text-anchor="middle" fill="#e8c372">♠</text>'
                    '</svg>')
            self._send(200, icon.encode(), "image/svg+xml")
            return
        if self.path.startswith("/api/coach/poll"):
            # the app's read side of the coach channel — a harmless GET (no mutation)
            q = parse_qs(urlsplit(self.path).query)
            try:
                rid = int(q.get("id", [""])[0])
            except (ValueError, IndexError):
                self._send(400, b"bad or missing id", "text/plain")
                return
            self._json(coach.poll(rid))
            return
        if self.path == "/api/read/spot":
            # the live drill's render payload — the spot in front of the hero (board / hand / price),
            # with NO equity (the number is computed + revealed only at commit, never before the read
            # is locked in). A harmless GET that dispatches NO coach request: LOGGING a read is pure
            # calibration, decoupled from ASKING flow for the prose ('talk it through' → /api/coach/ask).
            ctx = GAME.coach_decision_context()
            if ctx is None:
                self._json({"error": "no live decision to read"}, code=409)
                return
            # the token identifies THIS exact decision; the browser round-trips it on commit so a read
            # made on a stale drill (the hand advanced on another device) is rejected, not mis-logged.
            self._json({"spot": _decision_spot(ctx), "token": _spot_token(ctx)})
            return
        self._send(404, b"not found", "text/plain")

    def do_POST(self):  # noqa: N802
        if not self._host_ok():                 # anti-rebinding: Host must name this server
            self._send(403, b"host not allowed", "text/plain")
            return
        if GAME is None or SEAM is None:         # readiness guard (never hit on the normal path)
            self._send(503, b"server initializing", "text/plain")
            return
        if not self._same_origin():             # anti-CSRF: no cross-origin writes
            self._send(403, b"cross-origin request refused", "text/plain")
            return
        try:
            clen = int(self.headers.get("Content-Length", 0) or 0)
        except ValueError:
            self._send(400, b"bad Content-Length", "text/plain")
            return
        if clen < 0:                             # malformed (negative) Content-Length
            self._send(400, b"bad Content-Length", "text/plain")
            return
        if clen > _MAX_BODY:                     # body-size cap: bodies are tiny JSON
            self._send(413, b"request body too large", "text/plain")
            return
        try:
            if self.path == "/api/new":
                self._json(GAME.new_hand())
                return
            if self.path == "/api/act":
                body = self._read_body()
                self._json(GAME.act(body.get("action", ""), body.get("amount")))
                return
            if self.path == "/api/sit":
                self._json(GAME.sit_down(hero=self._read_body().get("hero", "You")))
                return
            if self.path == "/api/reload":
                self._json(GAME.reload_hero())
                return
            if self.path == "/api/getup":
                self._json(GAME.get_up())
                return
            if self.path == "/api/reset":
                # hard nuke: wipes the career ledger AND table (UI confirms first)
                GAME.reset_all()
                self._json(GAME.view())
                return
            if self.path == "/api/settings":
                # the settings menu's write side — a partial patch of known keys (coaching on/off +
                # the facts-floor stat prefs). apply_patch validates each key STRICTLY (unknown key or
                # bad value → 400) and persists; the read-side getters stay tolerant. The
                # `isinstance(body, dict)` guard keeps a null/list/str body a clean 400, not a 500.
                body = self._read_body()
                _updated, err = settings.apply_patch(body if isinstance(body, dict) else None)
                if err is not None:
                    self._json({"error": err}, code=400)
                    return
                self._json(_settings_payload())
                return
            if self.path == "/api/secrets":
                # a metered worker's BYO-key WRITE side — set or clear the operator's Anthropic OR OpenAI key.
                # The key is accepted HERE and stored to secrets.json (0600 via keystore); it NEVER comes
                # back in the response (only the safe {set, hint, source} status for BOTH vendors). Deliberately
                # SEPARATE from /api/settings so a credential can't ride the settings read payload. Body:
                # {key:"sk-..."} to set, {clear:true} to clear; provider defaults to "anthropic" (so the
                # OpenAI write MUST pass {provider:"openai"}). Inherits do_POST's host + same-origin (CSRF) +
                # body-size guards like every write.
                body = self._read_body()
                if not isinstance(body, dict):
                    self._json({"error": "secrets body must be an object"}, code=400)
                    return
                provider = body.get("provider", "anthropic")
                try:
                    if body.get("clear"):
                        keystore.clear_api_key(provider)
                    else:
                        keystore.set_api_key(body.get("key"), provider)
                except ValueError as e:           # unknown provider / blank key → a clean 400, not a 500
                    self._json({"error": str(e)}, code=400)
                    return
                self._json(_all_key_status())
                return
            if self.path == "/api/coach/ask":
                # build the spot SERVER-SIDE (authoritative — never trust client game
                # state), drop it on the coach channel, hand back the poll id. For a
                # review the frames are deterministic, so return them now to start the
                # visual replay immediately while the prose narration is still cooking.
                body = self._read_body()
                kind = body.get("kind")
                if kind == "decision":
                    ctx = GAME.coach_decision_context()
                    if ctx is None:
                        self._json({"error": "no live decision to coach"}, code=409)
                        return
                    eq = body.get("equity")
                    if isinstance(eq, dict) and eq.get("hero_pct") is not None:
                        # 'talk it through' — the read was already committed via /api/read/commit and the
                        # equity REVEALED. Pass that number back so the worker reuses it (its
                        # _ensure_modeled_equity early-returns on a populated modeled_equity) and the prose
                        # cites the SAME number the reveal showed, not a second MC that wiggles ±1-2% off it.
                        # GUARD it with the SAME spot token as the commit: the echoed equity is valid ONLY
                        # for the spot it was computed on. If the hand advanced (another device) between the
                        # reveal and this ask, injecting the stale number would make the worker narrate THIS
                        # (new) spot with the OLD spot's equity — clean calibration, but wrong coaching prose
                        # (and 'the conversation is the product'). Reject instead: the read's spot is gone.
                        if body.get("token") != _spot_token(ctx):
                            self._json({"error": "spot moved"}, code=409)
                            return
                        ctx["modeled_equity"] = eq
                    # P4b-i — recall the operator's relevant graduated leaks into the live context so the
                    # coach can surface "you've shown this read-leak in spots like this" at the decision
                    # point (the moat READ half; built server-side, off the daemon thread — §3f-clean).
                    _attach_relevant_leaks(ctx)
                    # id + the drill spot for the live gut-commit beat (board/hand/pot — all cheap, NO
                    # equity: the modeled-equity MC runs in code at /api/read/commit, never leaked here).
                    self._json({"id": coach.ask("decision", ctx), "spot": _decision_spot(ctx)})
                    return
                if kind == "review":
                    built = GAME.coach_review_context(
                        body.get("session_id"), body.get("hand_no"))
                    if built is None:
                        self._json({"error": "hand not found for review"}, code=404)
                        return
                    ctx, frames = built
                    # P4b-i recall is DECISION-ONLY (codex L3 MED): a review spans multiple decision
                    # points across streets, but the review ctx has only final_board (always 5 cards →
                    # always reads as 'river') and no per-decision street/action — so a single review
                    # query would mis-route (river leaks on a turn-reviewed mistake) AND drop the actual
                    # reviewed-decision signal. Recall belongs AT a decision point; per-DP review recall
                    # is a follow-up that needs the replay UI to signal the in-focus decision. So the
                    # review coach narrates the hand without leak-injection for now (renders no section).
                    # decision_points carry the exact-equity drill data to the browser (the
                    # commit-your-read beat); the coach worker gets them inside ctx for narration.
                    self._json({"id": coach.ask("review", ctx), "frames": frames,
                                "decision_points": ctx.get("decision_points", [])})
                    return
                self._json({"error": "unknown coach kind"}, code=400)
                return
            if self.path == "/api/calibration/review":
                # a read committed at the review drill → log it vs EXACT equity. The server
                # RECOMPUTES the canonical truth from (session_id, hand_no, n) — the client only
                # supplies its gut + lean + which decision, never a truth value to be trusted.
                body = self._read_body()
                self._json(SEAM.commit_read(
                    "review", session_id=body.get("session_id"),
                    hand_no=body.get("hand_no"), n=body.get("n"),
                    gut=body.get("gut"), lean=body.get("lean"),
                    note=body.get("note")))   # optional prose read — sanitized in Game._clean_note
                return
            if self.path == "/api/read/commit":
                # a read committed at the LIVE beat → compute the modeled equity in CODE right here and
                # log it. ANTI-FAB: the truth is recomputed from authoritative game state (never trusted
                # from the browser). NO coach worker required — logging a read is pure calibration,
                # decoupled from asking flow for the prose. Committing a read does NOT advance the hand,
                # so coach_decision_context() returns the SAME frozen spot the drill opened on; if the
                # hero has since acted (the spot moved) there is nothing left to calibrate.
                body = self._read_body()
                ctx = GAME.coach_decision_context()
                if ctx is None:
                    self._json({"ok": False, "reason": "spot moved"})
                    return
                # STRUCTURAL anti-stale guard: the read must be committed against the SAME decision the
                # drill rendered. A mismatch means the hand advanced to a different hero decision (another
                # tab/device) while this drill stayed open — log it and we'd grade the read against the
                # wrong spot + wrong-join P3a. Reject instead. (A missing/None token never matches a real
                # token, so a commit that never fetched a spot is refused too.)
                if body.get("token") != _spot_token(ctx):
                    self._json({"ok": False, "reason": "spot moved"})
                    return
                me = equity.modeled_from_spec(ctx.get("modeled_spec"))
                if not me or me.get("hero_pct") is None:
                    self._json({"ok": False, "reason": "no modeled equity for this spot"})
                    return
                res = SEAM.commit_read(
                    "live", gut=body.get("gut"), lean=body.get("lean"),
                    note=body.get("note"),   # optional prose read — sanitized in Game._clean_note
                    truth_pct=me["hero_pct"], street=ctx.get("street"),
                    # `vs` carries the live_count (provenance), matching the prior live route.
                    vs=(ctx.get("modeled_spec") or {}).get("live_count"),
                    # the FULL join key from the FRESH decision context — session_id included, frozen as
                    # one tuple so the modeled−exact reconciliation (P3a) joins to the right hand. The
                    # join key ALSO dedups a double-commit of the same spot (record_live_read).
                    session_id=ctx.get("session_id"),
                    hand_no=ctx.get("hand_no"), n=ctx.get("n"))
                # hand the computed equity back so the reveal narrates the SAME number that was logged.
                self._json({**res, "equity": me} if res.get("ok") else res)
                return
            self._send(404, b"not found", "text/plain")
        except Exception as e:  # never let a bad request kill the handler
            self._json({"error": str(e)}, code=400)

    def log_message(self, *args):  # quiet
        pass


def _bind_decision(host: str) -> tuple[str | None, str | None]:
    """Resolve a --host into (canonical_bind_ip, None) or (None, reason_to_refuse).

    The API is WRITABLE and UNAUTHENTICATED, so the bind address IS the security
    boundary. Two hardenings ride under the obvious wildcard/public refusal:

    - Canonicalize-or-reject (closes a parser-differential bypass): libc's
      getaddrinfo accepts legacy numeric IPv4 forms (``0``, ``0x08080808``,
      ``010.010.010.010``) that bind as 0.0.0.0 / 8.8.8.8 / 10.10.10.10, but Python's
      strict ``ipaddress`` rejects them. Validating with ``ipaddress`` while BINDING
      the raw string would let those slip past the guard — so we accept only
      ``localhost`` or a canonical literal, and bind the canonical form we return.
    - Tailscale-only, not any-private: a read-only mirror could safely bind any LAN
      IP, but a writable no-auth endpoint on a shared LAN (192.168/16, 10/8) is
      reachable by every device on that segment. So a non-loopback bind must be a
      Tailscale CGNAT address (100.64.0.0/10), where only the operator's enrolled
      devices can route.
    """
    h = host.strip().removesuffix(".").lower()  # tolerate one FQDN-style trailing dot, not garbage
    if h == "localhost":
        return "127.0.0.1", None
    try:
        ip = ipaddress.ip_address(h)
    except ValueError:
        return None, (f"'{host}' is not a canonical IP literal — pass your Tailscale IP "
                      "(e.g. 100.101.102.103) or 'localhost'. A hostname or legacy-numeric "
                      "form is refused.")
    if isinstance(ip, ipaddress.IPv6Address):
        # Refuse ALL IPv6 literals (incl. IPv4-mapped ::ffff:x forms): the server is
        # single-stack AF_INET, the only documented input is an IPv4 Tailscale IP, and
        # accepting ::ffff: forms would be pure attack surface with no legit use. Every
        # IPv6 form (mapped-public, mapped-wildcard, ULA) still ends up refused — cleaner.
        return None, "IPv6 binds aren't supported — pass your IPv4 Tailscale IP (100.x.y.z)."
    if ip.is_unspecified:  # kept first so 0.0.0.0 gets the precise wildcard message
        return None, "a wildcard (0.0.0.0) binds EVERY interface, not just the tailnet."
    # Explicit ALLOW-LIST before the global/LAN refusals: the CGNAT allow-path must NOT
    # depend on CPython classifying 100.64/10 as non-global. That special-range table has
    # shifted across point releases (CVE-2024-4032 reworked is_global/is_private for
    # several ranges); ordering the allow first means a future flip can't refuse a real
    # Tailscale IP. Loopback + Tailscale CGNAT are the only addresses we ever bind.
    if ip.is_loopback:
        return str(ip), None
    if ip in _TAILSCALE_CGNAT:
        return str(ip), None
    if ip.is_global:
        return None, "a public/global IP would expose this writable app to the internet."
    return None, (f"{ip} is a shared-LAN address — this app mutates with no auth, so a "
                  "non-loopback bind must be your Tailscale IP (100.64.0.0/10), never a "
                  "LAN address every device on the network can reach.")


def _coach_banner() -> str:
    """One startup line naming the coach provider, and saying so when its CLI is not installed (the
    coach then answers with facts only, which otherwise only shows up as a spawn error in the log)."""
    try:
        provider = settings.coach_provider()
    except Exception:                       # a corrupt settings file must not stop the server starting
        provider = "claude"
    clis = {"claude": coach_daemon._claude_bin, "codex": coach_daemon._codex_bin}
    if provider in clis:
        exe = clis[provider]()
        if not (Path(exe).is_file() and os.access(exe, os.X_OK)):
            return (f"no `{provider}` CLI found, so 'talk it through' gives facts only "
                    "(pick another provider in ⚙ Settings)")
    return f"built-in AI coach ({provider}) — falls back to facts-only if it can't answer"


def main() -> int:
    ap = argparse.ArgumentParser(description="gutcheck local web server")
    ap.add_argument("--port", type=int, default=8777, help="TCP port (default 8777)")
    ap.add_argument("--host", default="127.0.0.1",
                    help="bind address: 'localhost' (default) or your Tailscale IP "
                         "(100.x.y.z) to play on the iPad over the tailnet")
    ap.add_argument("--reset", action="store_true",
                    help="start fresh: delete the table, hand history, session reports and coach "
                         "channel; bankroll, read accuracy and leak memory are backed up first")
    ap.add_argument("--no-browser", action="store_true",
                    help="don't auto-open a browser (e.g. when serving the iPad over Tailscale)")
    ap.add_argument("--no-coach", action="store_true",
                    help="don't run the built-in coach — use this if you run your OWN worker "
                         "(`python -m gutcheck.coach_daemon run` / launchd) on the same channel.")
    args = ap.parse_args()

    if not 1 <= args.port <= 65535:
        print(f"--port must be 1-65535 (got {args.port}).", file=sys.stderr)
        return 1

    # Fail LOUD + early if the table UI is missing. static/ ships as package-data (both editable
    # and plain `pip install gutcheck` carry it), so a miss here means a genuinely broken install
    # (a corrupt wheel or a package-data regression), NOT an unsupported install mode. Beats a
    # silent 404.
    if not (STATIC / "index.html").exists():
        print(f"static assets not found at {STATIC} — the install is broken (static/ ships as "
              "package-data). Reinstall: `pip install --force-reinstall gutcheck` (or `pip "
              "install -e .` from a clone).", file=sys.stderr)
        return 1

    # NOTE: the data dir + the destructive --reset + Game construction happen LATER, only after
    # EVERY guard that can `return 1` (bind validation, the bind itself, the post-bind confirm) —
    # so a failed launch can never leave a wiped bankroll behind (the aborted-reset HIGH).
    bind_ip, reason = _bind_decision(args.host)
    if reason is not None:
        print(f"refused to bind --host {args.host!r}: {reason}", file=sys.stderr)
        return 1

    try:
        server = HTTPServer((bind_ip, args.port), Handler)  # single-threaded: avoids the
        # global-RNG / shared-session races (local single-player needs no concurrency)
    except OSError as e:
        # Errno 48 = port in use; Errno 49 = address not assigned to a local interface
        # (the common 0d case: Tailscale is down or the --host IP is wrong).
        print(f"could not bind {bind_ip}:{args.port} — {e}. Check the port is free, and "
              "that the --host IP is actually assigned to this machine (is Tailscale up?).",
              file=sys.stderr)
        return 1

    # Un-foolable backstop (bridge.py's reality-check pattern): re-run policy on the
    # address the OS ACTUALLY bound, not the requested host. If anything let a
    # disallowed address through, refuse-and-close before a single request is served.
    actual = str(server.server_address[0])
    confirmed, why = _bind_decision(actual)
    if why is not None or confirmed != actual:
        server.server_close()
        print(f"refusing to serve: bound a disallowed address ({actual}) — {why}", file=sys.stderr)
        return 1

    global BOUND_HOST          # the Host the anti-rebinding _host_ok check allows by IP
    BOUND_HOST = actual

    # All abort paths are behind us — NOW it's safe to touch state: ensure the data dir, apply an
    # optional (destructive) --reset, then construct the Game. Doing this only after the bind is
    # confirmed means a refused launch never wipes a bankroll (the aborted-reset HIGH).
    try:
        data = paths.ensure_data_dir()
    except paths.LegacyMigrationError as e:
        print(e, file=sys.stderr)
        return 1
    if args.reset:
        removed = paths.reset_data()
        print(f"--reset: cleared {len(removed)} item(s) from {data} "
              "(fresh 4000 bankroll; career.json backed up to career.reset-backup.json).",
              file=sys.stderr)
    global GAME, SEAM
    # P4 — wire the operator-owned, anneal-backed leak store. A no-op when the gutcheck[anneal] extra
    # isn't installed (LeakStore.available is False → every read-capture + seam leak call degrades), so
    # this is safe to construct unconditionally: the single-dep game is unchanged, the [anneal] install
    # gets graduating-leak memory.
    GAME = Game(leak_store=leakstore.LeakStore())
    SEAM = Seam(GAME)   # Layer B: the coach contract over GAME + history (every transport goes through it)

    # The built-in coach: a background thread runs the full coach_daemon loop (flow on the Claude
    # subscription, degrading to honest facts-only when that path is down) so the felt-sense loop
    # works from `gutcheck` alone. daemon=True → it dies with the process; coach_stop ends the LOOP
    # cleanly between requests, and the shutdown below reaps an in-flight `claude -p` child. --no-coach
    # turns it off when you run your own worker; a worker started anyway no longer double-fires,
    # because each request is claimed (coach.claim) before anyone spends on it.
    def _on_term(_signum, _frame):      # a plain `kill`: take the same clean exit as Ctrl-C, so the finally runs
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _on_term)
    coach_stop = threading.Event()
    coach_thread = None
    if not args.no_coach:
        global COACH_BUILT_IN
        COACH_BUILT_IN = True            # the settings surface reports this; the live toggle is built-in-only
        coach_thread = threading.Thread(target=_run_coach, args=(paths.coach_db(), coach_stop, GAME.leaks),
                                        daemon=True, name="coach")
        coach_thread.start()

    if ipaddress.ip_address(actual).is_loopback:
        reach = "localhost-only"
    else:
        reach = (f"reachable to ANY device on your tailnet at {actual} — this app MUTATES "
                 "with NO auth; the Tailscale WireGuard mesh is the only access control")
    # flush so the URL is visible immediately even when stdout is piped or run under
    # launchd (block-buffered, not line-buffered, off a tty).
    print(f"gutcheck → http://{actual}:{args.port}  (Ctrl-C to stop)", flush=True)
    print(f"  {reach}", flush=True)
    print("  coach: " + (_coach_banner() if not args.no_coach
                         else "OFF (--no-coach; run your own worker)"), flush=True)
    # The socket is already listening (HTTPServer bound above), so a browser opened now connects
    # even before serve_forever() drains the accept queue. Loopback only — a Tailscale bind is for
    # the iPad, not this box; --no-browser opts out for scripted/headless runs.
    if ipaddress.ip_address(actual).is_loopback and not args.no_browser:
        try:
            webbrowser.open(f"http://{actual}:{args.port}")
        except Exception:
            pass   # headless box with no browser is fine — the URL is printed above
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    finally:
        signal.signal(signal.SIGTERM, signal.SIG_IGN)   # a second kill must not interrupt the cleanup below
        coach_stop.set()        # signal the coach thread to exit cleanly (a daemon would die anyway)
        if coach_thread is not None:
            # bounded join so the loop can finish its current pass + close its leak-store connection
            # cleanly (a checkpointed WAL on the common idle-shutdown path). If a coach answer or a leak
            # seed author call is in flight it may outlast this — the daemon=True thread then dies with the
            # process and any open leak wrap is reclaimed by the next run's recover (F6). [codex L3 MED-2]
            coach_thread.join(timeout=3.0)
        coach_daemon.reap_inflight()    # the built-in coach's `claude -p` child must not outlive the server
        if coach_thread is not None:
            coach_thread.join(timeout=2.0)   # let the reaped call write its stub answer before we exit
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
