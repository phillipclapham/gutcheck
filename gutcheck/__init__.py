"""gutcheck — a felt-sense poker reading-trainer.

The importable package: the game engine, the local web table server, the MCP coach
proxy, and the operator-owned graduating-leak store. Submodules are imported lazily by
their consumers (``from gutcheck import server`` / ``from gutcheck.seam import Seam``);
this ``__init__`` deliberately imports nothing so a plain ``import gutcheck`` has no
side effects and can't trip the server→coach_daemon→… chain at package-load time.

All runtime state lives OUTSIDE the package, under ``~/.gutcheck`` (override with
``GUTCHECK_DATA_DIR``) — see ``gutcheck.paths`` — so a read-only wheel install is safe.
"""

__version__ = "0.1.3"
