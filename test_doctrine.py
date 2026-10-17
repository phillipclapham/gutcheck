"""P3(b) — the shared coaching doctrine: single-source loading, fail-safe degradation, the
read→action principle reaching BOTH coach surfaces, and the §3f archetype gate holding in the
doctrine CONTENT (the doctrine must teach the coach to think in behavior, never type labels).
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from gutcheck import coach_daemon
from gutcheck import doctrine


def _pack(meta_lines):
    """Build a structurally-valid pack: a '---'-fenced frontmatter block + a body that clears the
    min length and carries a well-formed BRIEF region."""
    fm = "---\n" + "\n".join(meta_lines) + "\n---\n"
    body = ("# Test Pack\n\n## Section 1\n"
            + "filler doctrine prose line, long enough to clear the min length.\n" * 12
            + "\n<!-- BRIEF:START -->\nA read is incomplete until you've named what it forbids.\n"
              "<!-- BRIEF:END -->\n")
    return fm + body

HAS_MCP = importlib.util.find_spec("mcp") is not None

# Archetype TYPE LABELS as proper nouns — the doctrine must NOT hand these to the coach as
# labels (§3f: coach from observed behavior, the operator earns the type). Lowercase generic
# usage ("don't bluff a station") is fine; a capitalized PROPER-NOUN label is the leak shape.
_ARCHETYPE_LABELS = ["Station", "Nit", "Maniac", "Whale", "Rock", "Fish", "TAG", "LAG", "Calling Station"]


class TestDoctrineLoader(unittest.TestCase):
    def test_full_reads_the_real_doc(self):
        full = doctrine.full()
        # the real doc, not the stub (the stub is short + has no section headers)
        self.assertGreater(len(full), 2000)
        for header in ("Section 1", "Section 2", "Section 3"):
            self.assertIn(header, full)
        # the read→action principle (b) is present and first-class
        self.assertIn("incomplete until you've named what it forbids", full)
        # the §3f gate is stated in the doctrine itself
        self.assertIn("NEVER name or reveal", full)
        # the reconciliation beat survives the lift out of _PEDAGOGY
        self.assertIn("read her range right and were still crushed", full)
        # read-based-not-solver positioning is explicit (don't grade EV)
        self.assertIn("never grade", full.lower().replace("-", " "))

    def test_brief_is_one_line_stripped_and_carries_readaction(self):
        brief = doctrine.brief()
        self.assertNotIn("\n", brief)                      # one line for the metadata block
        self.assertNotIn("<!--", brief)                    # HTML markers stripped
        self.assertNotIn("BRIEF:", brief)
        self.assertIn("forbids", brief)                    # the load-bearing principle
        self.assertLess(len(brief), 700)                   # concise — it rides in MCP instructions

    def test_brief_is_a_region_of_full_not_a_separate_copy(self):
        # author-once: the brief's core sentence lives verbatim inside the full doc
        self.assertIn("A read is incomplete until you've named what it forbids.", doctrine.full())


class TestDoctrineFailSafe(unittest.TestCase):
    """The coach must NEVER crash because doctrine.md is unreadable — both accessors degrade."""

    def _point_at(self, path: Path):
        orig = doctrine._doc_path
        doctrine._doc_path = lambda: path
        self.addCleanup(lambda: setattr(doctrine, "_doc_path", orig))

    def test_full_falls_back_when_doc_missing(self):
        self._point_at(Path("/nonexistent/doctrine.md"))
        full = doctrine.full()
        self.assertEqual(full, doctrine._FALLBACK_FULL)    # degraded, not raised

    def test_brief_falls_back_when_doc_missing(self):
        self._point_at(Path("/nonexistent/doctrine.md"))
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)

    def test_brief_falls_back_when_markers_absent(self):
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write("# a doctrine with no brief markers\njust prose, no region to extract\n")
            tmp = Path(f.name)
        self.addCleanup(lambda: tmp.unlink(missing_ok=True))
        self._point_at(tmp)
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)

    def test_both_accessors_failsafe_on_non_utf8_doc(self):
        # the precise hole L1 found: a non-UTF-8 doctrine.md raises UnicodeDecodeError
        # (a ValueError subclass, NOT OSError) — it MUST still degrade, never raise, or it
        # aborts MCP build_server() at startup.
        with tempfile.NamedTemporaryFile("wb", suffix=".md", delete=False) as f:
            f.write(b"\xff\xfe# a doctrine saved in the wrong encoding \x80\x81\x82\n")
            tmp = Path(f.name)
        self.addCleanup(lambda: tmp.unlink(missing_ok=True))
        self._point_at(tmp)
        self.assertEqual(doctrine.full(), doctrine._FALLBACK_FULL)     # degraded, did NOT raise
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)

    def test_full_falls_back_on_truncated_partial_save(self):
        # codex MED: a non-atomic / truncated mid-edit save is valid UTF-8 but PARTIAL — it must
        # degrade to the guard-carrying stub, not serve a doctrine missing the §3f / read→action
        # core. A truncation before BRIEF:END leaves the completeness sentinel broken.
        partial = ("# gutcheck Doctrine\n" + "filler line\n" * 80 +
                   "<!-- BRIEF:START -->\nhalf-written, never closed")   # no BRIEF:END
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(partial)
            tmp = Path(f.name)
        self.addCleanup(lambda: tmp.unlink(missing_ok=True))
        self._point_at(tmp)
        self.assertEqual(doctrine.full(), doctrine._FALLBACK_FULL)
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)   # malformed markers → stub

    def test_brief_falls_back_on_oversized_region(self):
        # codex LOW: a pathologically large marked region must not bloat the MCP instructions
        huge = ("# doc\n" + "x" * 50 + "\n<!-- BRIEF:START -->\n" + ("word " * 400) +
                "\n<!-- BRIEF:END -->\n")
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False) as f:
            f.write(huge)
            tmp = Path(f.name)
        self.addCleanup(lambda: tmp.unlink(missing_ok=True))
        self._point_at(tmp)
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)   # over _BRIEF_MAX → stub
        self.assertGreater(len(doctrine.brief()), 0)

    def test_fallback_stubs_carry_gate_and_readaction_and_stay_short(self):
        # even degraded, the coach stays honest (gate) and pointed at the principle (read→action)
        for stub in (doctrine._FALLBACK_FULL, doctrine._FALLBACK_BRIEF):
            self.assertIn("forbids", stub)                 # the read→action core
            self.assertLess(len(stub), 700)                # a stub, never a copy of the doc
        self.assertIn("NEVER name or reveal", doctrine._FALLBACK_FULL)   # the gate


class TestDoctrinePackFormat(unittest.TestCase):
    """The doctrine-pack format (productization §6): markdown + optional flat frontmatter, one file =
    one pack. The frontmatter is metadata + stripped from the coach prompt; read_based is a LABEL, not
    a guard (the §3f seam is the real boundary); $GUTCHECK_DOCTRINE swaps the pack."""

    def _write_pack(self, text: str) -> Path:
        with tempfile.NamedTemporaryFile("w", suffix=".md", delete=False, encoding="utf-8") as f:
            f.write(text)
            p = Path(f.name)
        self.addCleanup(lambda: p.unlink(missing_ok=True))
        return p

    def _set_env_doctrine(self, path: Path):
        prev = os.environ.get(doctrine._DOCTRINE_ENV)
        os.environ[doctrine._DOCTRINE_ENV] = str(path)
        self.addCleanup(lambda: os.environ.__setitem__(doctrine._DOCTRINE_ENV, prev)
                        if prev is not None else os.environ.pop(doctrine._DOCTRINE_ENV, None))

    def test_flagship_frontmatter_parses_and_is_stripped(self):
        # the bundled flagship pack #1 carries frontmatter; meta() reads it, full()/brief() strip it
        m = doctrine.meta()
        self.assertEqual(m["name"], "gutcheck Flagship")
        self.assertTrue(m["read_based"])
        self.assertIn("version", m)
        full = doctrine.full()
        self.assertFalse(full.startswith("---"))           # no frontmatter fence
        self.assertNotIn("read_based:", full)              # no metadata key leaked into the coach prompt
        self.assertNotIn("author:", full)

    def test_plain_doctrine_without_frontmatter_still_loads(self):
        # back-compat: a pack with NO frontmatter loads as the whole body (and meta() = defaults)
        body = ("# Plain Pack\n" + "prose line that is plenty long for the floor.\n" * 14
                + "<!-- BRIEF:START -->\nA read is incomplete until you've named what it forbids.\n"
                  "<!-- BRIEF:END -->\n")
        self._set_env_doctrine(self._write_pack(body))
        self.assertIn("Plain Pack", doctrine.full())       # loaded, not the fallback
        self.assertEqual(doctrine.meta()["name"], doctrine._DEFAULT_META["name"])  # defaults, no frontmatter

    def test_env_override_swaps_the_pack(self):
        self._set_env_doctrine(self._write_pack(_pack(['name: Swapped Pack', 'version: "9"',
                                                       'read_based: true'])))
        self.assertEqual(doctrine.meta()["name"], "Swapped Pack")
        self.assertEqual(doctrine.meta()["version"], "9")
        self.assertIn("Test Pack", doctrine.full())        # the swapped body, frontmatter stripped
        self.assertNotIn("Swapped Pack", doctrine.full())

    def test_read_based_false_pack_still_loads_label_not_guard(self):
        # THE breakthrough embodied: the §3f seam is the security boundary, so the loader NEVER rejects
        # a pack on its read_based label — a read_based:false pack loads exactly the same. The flag is
        # for curation/discovery; it is not enforcement.
        self._set_env_doctrine(self._write_pack(_pack(['name: Solverish', 'read_based: false'])))
        self.assertFalse(doctrine.meta()["read_based"])
        self.assertIn("Test Pack", doctrine.full())        # loaded, NOT fallback — never rejected on the label

    def test_torn_frontmatter_fails_closed_no_leak(self):
        # L1 MED: a leading '---' with NO clean closing fence is a BROKEN pack, not a plain doc — it must
        # fail closed (empty body → full() degrades to the stub), NEVER leak frontmatter into the prompt.
        meta, body = doctrine._split_frontmatter("---\nname: secret\nauthor: leaked\nno closer\n# body\n")
        self.assertEqual((meta, body), ({}, ""))           # torn → fail closed at the unit level
        torn = ("---\nname: SECRET\nauthor: LEAKED\nversion: 9\n"   # opener, NEVER closed by a '---' line
                + "# Doctrine\n" + "prose line long enough for the floor.\n" * 14
                + "<!-- BRIEF:START -->\nA read is incomplete until you've named what it forbids.\n"
                  "<!-- BRIEF:END -->\n")
        self._set_env_doctrine(self._write_pack(torn))
        full = doctrine.full()
        self.assertEqual(full, doctrine._FALLBACK_FULL)    # degraded to the stub end-to-end
        for leaked in ("SECRET", "LEAKED", "name:", "author:"):
            self.assertNotIn(leaked, full)                 # no frontmatter text reached the coach prompt
        self.assertEqual(doctrine.meta(), dict(doctrine._DEFAULT_META))   # meta honest: defaults, not "SECRET"

    def test_loose_dash_lines_are_not_openers(self):
        # L1 LOW: only a WHOLE-LINE '---' opens frontmatter; '----' / '---x' is a plain doc left intact
        for opener in ("----", "---hr", "------"):
            meta, body = doctrine._split_frontmatter(f"{opener}\nname: x\n# content\n")
            self.assertEqual(meta, {})
            self.assertIn(opener, body)

    def test_malformed_closer_not_rescued_by_body_hr(self):
        # codex L3 HIGH: a malformed closer ('--- nope') must NOT be rescued by a LATER body '---' HR
        # that swallows the real doctrine + reports phantom metadata. Sequential parse fails closed at
        # the first non-conforming line, never scanning ahead for a stray fence.
        meta, body = doctrine._split_frontmatter(
            "---\nname: X\n--- nope\nphantom body\n---\nmore body that would survive\n")
        self.assertEqual((meta, body), ({}, ""))

    def test_full_and_brief_normalize_identically(self):
        # codex L3 HIGH: full()/brief()/meta() must normalize the same. A leading blank before the
        # frontmatter must not make brief()/meta() miss the frontmatter (a divergence full() wouldn't have).
        self._set_env_doctrine(self._write_pack("\n\n" + _pack(['name: Lead Blank', 'read_based: true'])))
        self.assertNotIn("name:", doctrine.full())
        self.assertNotIn("name:", doctrine.brief())        # brief strips the frontmatter too
        self.assertNotIn("Lead Blank", doctrine.brief())
        self.assertEqual(doctrine.meta()["name"], "Lead Blank")   # meta sees it after the same normalize

    def test_bad_brief_region_degrades_brief_only_not_full(self):
        # codex re-verify MED — INTENTIONAL per-surface degradation (NOT lockstep): a STRUCTURALLY valid
        # pack (readable, long enough, markers present) whose BRIEF REGION is unusable (here: empty) must
        # still serve the FULL doctrine to the daemon — never throw away a valid doctrine over a
        # brief-region typo — while brief() falls to the stub (which still carries the read→action core).
        body = ("# Pack\n## Section 1\n" + "long enough prose to clear the floor here.\n" * 14
                + "<!-- BRIEF:START --><!-- BRIEF:END -->\n")          # markers present, region EMPTY
        self._set_env_doctrine(self._write_pack("---\nname: EmptyBrief\nread_based: true\n---\n" + body))
        self.assertNotEqual(doctrine.full(), doctrine._FALLBACK_FULL)  # full doctrine still served
        self.assertIn("Section 1", doctrine.full())
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)   # brief degrades alone (empty region)
        self.assertEqual(doctrine.meta()["name"], "EmptyBrief")        # meta honestly reflects the loaded pack

    def test_brief_applies_length_floor_like_full(self):
        # codex L3 MED: a too-short pack with valid markers must degrade brief() too, not just full() —
        # all surfaces degrade in lockstep (the shared _load_pack guard).
        short = "---\nname: Tiny\n---\n# t\n<!-- BRIEF:START -->\nshort brief\n<!-- BRIEF:END -->\n"
        self._set_env_doctrine(self._write_pack(short))
        self.assertEqual(doctrine.full(), doctrine._FALLBACK_FULL)
        self.assertEqual(doctrine.brief(), doctrine._FALLBACK_BRIEF)    # was a gap — brief ignored the floor
        self.assertEqual(doctrine.meta(), dict(doctrine._DEFAULT_META))

    def test_value_may_contain_colon(self):
        meta, _ = doctrine._split_frontmatter("---\ndescription: trains reads: vs truth\n---\nbody\n")
        self.assertEqual(meta["description"], "trains reads: vs truth")   # only the first ':' splits

    def test_split_tolerates_stray_cr(self):
        # read_text normalizes CRLF, but be robust if a caller passes it directly — strip absorbs the \r
        meta, body = doctrine._split_frontmatter("---\r\nname: CR Pack\r\n---\r\n# body\r\n")
        self.assertEqual(meta.get("name"), "CR Pack")
        self.assertNotIn("name:", body)

    def test_bom_prefixed_pack_strips_frontmatter(self):
        # a pack saved with a UTF-8 BOM must still detect + strip its frontmatter (read via utf-8-sig),
        # never leak it — without -sig the '---' opener wouldn't match and the metadata would ride through
        with tempfile.NamedTemporaryFile("wb", suffix=".md", delete=False) as f:
            f.write(("\ufeff" + _pack(['name: BOM Pack', 'read_based: true'])).encode("utf-8"))
            p = Path(f.name)
        self.addCleanup(lambda: p.unlink(missing_ok=True))
        self._set_env_doctrine(p)
        self.assertEqual(doctrine.meta()["name"], "BOM Pack")
        self.assertIn("Test Pack", doctrine.full())
        self.assertNotIn("name:", doctrine.full())

    def test_frontmatter_meta_failsafe_when_doc_missing(self):
        self._set_env_doctrine(Path("/nonexistent/pack.md"))
        self.assertEqual(doctrine.meta(), dict(doctrine._DEFAULT_META))   # defaults, never raises


class TestThirdFGateInDoctrine(unittest.TestCase):
    """§3f at the doctrine-content layer: the doctrine must not hand the coach archetype LABELS."""

    def test_no_archetype_proper_noun_labels(self):
        for surface in (doctrine.full(), doctrine.brief(),
                        doctrine._FALLBACK_FULL, doctrine._FALLBACK_BRIEF):
            for label in _ARCHETYPE_LABELS:
                self.assertNotIn(label, surface,
                                 f"archetype label {label!r} leaked into doctrine content")


class TestDaemonDelivery(unittest.TestCase):
    """The daemon coach prompt inherits the doctrine from the single source."""

    def test_pedagogy_composes_frame_doctrine_transport(self):
        ped = coach_daemon._pedagogy()
        self.assertIn("coaching a player", ped)                              # operator frame
        self.assertIn("incomplete until you've named what it forbids", ped)  # the doctrine (b)
        self.assertIn("ONLY the coaching prose", ped)                     # transport instruction
        # the doctrine sits BETWEEN the frame and the transport (compose order is load-bearing)
        self.assertLess(ped.index("coaching a player"), ped.index("incomplete until"))
        self.assertLess(ped.index("incomplete until"), ped.index("ONLY the coaching prose"))


@unittest.skipUnless(HAS_MCP, "mcp SDK not installed (optional extra)")
class TestMcpDelivery(unittest.TestCase):
    """The MCP coach surface (where the live leak fired) inherits the read→action doctrine."""

    def test_instructions_carry_the_readaction_brief(self):
        from gutcheck import mcp_server
        instr = mcp_server.build_server(expose_ground_truth=True).instructions
        self.assertIn("forbids", instr)                          # the read→action principle
        self.assertIn("archetype is gated", instr)               # the gate, still stated
        for label in _ARCHETYPE_LABELS:
            self.assertNotIn(label, instr)                       # no label leak in instructions

    def test_get_table_state_docstring_closes_read_to_action(self):
        from gutcheck import mcp_server
        tools = {t.name: t for t in asyncio.run(
            mcp_server.build_server(expose_ground_truth=True).list_tools())}
        desc = tools["get_table_state"].description or ""
        self.assertIn("CLOSE READ→ACTION", desc)
        self.assertIn("never grade", desc.lower())               # read-trainer, not solver


if __name__ == "__main__":
    unittest.main()
