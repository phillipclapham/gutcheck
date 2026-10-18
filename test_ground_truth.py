"""Tests for the §3f structural pedagogy gate — seam.get_ground_truth + GET /api/ground_truth (P2).

The load-bearing claim: the coach's answer-key (exact equity + showdown) is ALWAYS exposed, but the
villain ARCHETYPE NAME is withheld by default so a shipped MCP coach CAN'T spoil the read — the gate
is structural (the seam doesn't hand over the answer), not a coach-side discipline it could leak.
Isolated via explicit Game paths (incl. history_db), no real ~/.gutcheck touch.
"""

import json
import random
import tempfile
import threading
import unittest
import unittest.mock
from http.server import HTTPServer
from pathlib import Path

from gutcheck import game as G
from gutcheck import seam as seam_mod
from gutcheck import server as S


def _play_session_with_report(d: Path):
    """Play a repeatable session (SystemRandom pinned below) to a written report, returning (game, session_id, report)."""
    random.seed(7)                                   # deterministic villains + deal
    g = G.Game(d / "session.json", d / "career.json", d / "sessions",
               d / "calibration.json", d / "history.db")
    # sit_down draws the master seed from SystemRandom, so random.seed(7) alone never made the deals
    # repeat; pin that draw so the fixture plays the same session every run.
    with unittest.mock.patch.object(G.random, "SystemRandom", lambda: random.Random(7)):
        g.sit_down(hero="Phill")
    sid = g.s["session"]["id"]
    for _ in range(25):                              # check/call down → postflop spots + a report
        g.new_hand()
        guard = 0
        while True:
            v = g.view()
            if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                break
            o = v.get("options") or {}
            if o.get("is_check"):
                g.act("check")
            elif o.get("check_or_call"):
                g.act("call")
            else:
                break
            guard += 1
            if guard > 30:
                break
    g.get_up()
    report = json.loads((d / "sessions" / f"{sid}.json").read_text())
    return g, sid, report


def _a_reviewable_hand(report: dict) -> int:
    pots = report.get("biggest_pots") or report.get("notable_hands") or []
    assert pots, "no reviewable hands in the report"
    return pots[0]["hand_no"]


class TestGroundTruthGate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        cls.g, cls.sid, report = _play_session_with_report(cls.d)
        cls.hand_no = _a_reviewable_hand(report)
        cls.seam = seam_mod.Seam(cls.g)

    def test_answer_key_always_exposed(self):
        gt = self.seam.get_ground_truth(self.sid, self.hand_no)
        self.assertTrue(gt["ok"])
        self.assertIn("decision_points", gt)             # exact equity per hero spot (may be [])
        self.assertIsNotNone(gt.get("result"))           # winners + showdown holes + hero_net
        self.assertIsNotNone(gt.get("final_board"))
        self.assertTrue(gt["villains"], "expected the villain roster in the truth")

    def test_archetype_gated_by_default(self):
        gt = self.seam.get_ground_truth(self.sid, self.hand_no)
        self.assertFalse(gt["archetype_revealed"])
        for v in gt["villains"]:
            self.assertNotIn("_truth", v)                # the hidden archetype is stripped...
            self.assertNotIn("archetype", v)             # ...and not revealed
            self.assertIn("observed", v)                 # the READ material (stats) stays

    def test_archetype_revealed_only_on_opt_in(self):
        gt = self.seam.get_ground_truth(self.sid, self.hand_no, reveal_archetype=True)
        self.assertTrue(gt["archetype_revealed"])
        self.assertTrue(any(v.get("archetype") for v in gt["villains"]),
                        "reveal=True must surface the archetype label")
        for v in gt["villains"]:
            self.assertNotIn("_truth", v)                # raw _truth never leaks even when revealed

    def test_hand_not_found(self):
        gt = self.seam.get_ground_truth("nonexistent-session", 1)
        self.assertFalse(gt["ok"])


class TestGroundTruthEndpoint(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d = Path(tempfile.mkdtemp())
        cls.g, cls.sid, report = _play_session_with_report(cls.d)
        cls.hand_no = _a_reviewable_hand(report)
        S.GAME = cls.g
        S.SEAM = seam_mod.Seam(cls.g)
        cls.httpd = HTTPServer(("127.0.0.1", 0), S.Handler)
        cls.port = cls.httpd.server_address[1]
        threading.Thread(target=cls.httpd.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.httpd.shutdown()
        cls.httpd.server_close()

    def _get(self, path):
        import http.client
        c = http.client.HTTPConnection("127.0.0.1", self.port, timeout=30)
        c.request("GET", path, headers={"Host": f"127.0.0.1:{self.port}"})
        r = c.getresponse()
        b = r.read()
        c.close()
        return r.status, b

    def test_endpoint_gated_by_default(self):
        st, b = self._get(f"/api/ground_truth?session={self.sid}&hand={self.hand_no}")
        self.assertEqual(st, 200)
        d = json.loads(b)
        self.assertTrue(d["ok"])
        self.assertFalse(d["archetype_revealed"])
        for v in d["villains"]:
            self.assertNotIn("archetype", v)

    def test_endpoint_never_reveals_over_http(self):
        # the §3f gate is seam-WIDE: /api/ground_truth IGNORES reveal_archetype=1 — a networked (incl.
        # Tailscale) caller cannot flip it; reveal is operator-LOCAL only. [codex L3 MED1]
        st, b = self._get(f"/api/ground_truth?session={self.sid}&hand={self.hand_no}&reveal_archetype=1")
        self.assertEqual(st, 200)
        d = json.loads(b)
        self.assertFalse(d["archetype_revealed"])
        for v in d["villains"]:
            self.assertNotIn("archetype", v)

    def test_endpoint_bad_input_400(self):
        st, _ = self._get("/api/ground_truth?session=&hand=abc")
        self.assertEqual(st, 400)


class TestCoachTableStateGate(unittest.TestCase):
    """get_coach_table_state strips the villain archetype that Game.view() (the browser /api/state)
    leaks — the codex L3 HIGH: get_table_state bypassed the §3f gate through the front door."""

    def _game_midhand(self, d):
        random.seed(7)
        g = G.Game(d / "session.json", d / "career.json", d / "sessions",
                   d / "calibration.json", d / "history.db")
        g.sit_down(hero="Phill")
        g.new_hand()
        return g

    def test_live_seats_archetype_stripped(self):
        g = self._game_midhand(Path(tempfile.mkdtemp()))
        s = seam_mod.Seam(g)
        full = s.get_table_state()          # browser view — DOES carry the archetype
        coach = s.get_coach_table_state()   # coach view — STRIPPED
        self.assertTrue(any("archetype" in seat or "type" in seat for seat in full.get("seats", [])),
                        "the browser view should carry the archetype (that's why the strip exists)")
        for seat in coach.get("seats", []):
            self.assertNotIn("type", seat)
            self.assertNotIn("archetype", seat)
        villains = [seat for seat in coach.get("seats", []) if not seat.get("is_hero")]
        self.assertTrue(villains and all("stats" in seat for seat in villains),
                        "the OBSERVED read material (stats) must survive the strip")

    def test_between_hands_table_seats_stripped_without_mutating_state(self):
        g = self._game_midhand(Path(tempfile.mkdtemp()))
        for _ in range(40):                 # finish the hand so view() returns the between-hands `table`
            v = g.view()
            if not v.get("in_hand") or v.get("over") or v.get("to_act") != "hero":
                break
            o = v.get("options") or {}
            if o.get("is_check"):
                g.act("check")
            elif o.get("check_or_call"):
                g.act("call")
            else:
                break
        coach = seam_mod.Seam(g).get_coach_table_state()
        table = coach.get("table")
        if not (isinstance(table, dict) and table.get("seats")):
            self.skipTest("hand still live; no between-hands table snapshot")
        for seat in table["seats"]:
            self.assertNotIn("type", seat)
            self.assertNotIn("archetype", seat)
        # the deep-copy must NOT have mutated the stored last_table — the browser view still has it
        full_table = (seam_mod.Seam(g).get_table_state().get("table") or {})
        self.assertTrue(any("archetype" in seat or "type" in seat for seat in full_table.get("seats", [])),
                        "the strip mutated the stored snapshot — browser would lose the archetype")


if __name__ == "__main__":
    unittest.main(verbosity=2)
