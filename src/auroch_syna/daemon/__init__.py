"""auroch_syna.daemon — long-running engine service.

Speaks JSON over WebSocket. Clients (Winnie, command bar, viewer,
continuity service) connect, subscribe to events, dispatch commands.

Run with: ``auroch-syna daemon --port 8765``
"""
from __future__ import annotations

from .server import DaemonServer, run

__all__ = ["DaemonServer", "run"]
