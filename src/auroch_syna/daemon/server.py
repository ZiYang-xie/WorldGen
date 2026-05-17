"""WebSocket daemon — fronts the engine for external consumers.

Wire protocol (JSON, line-delimited per WebSocket frame):

    → from client:  {"type": "command", "command": {...}}
    → from client:  {"type": "subscribe", "kinds": ["pipeline.*", "selection.*"]}
    ← to client:    {"type": "event", "event": {"kind": "...", "payload": {...}}}
    ← to client:    {"type": "ack",   "op_id": "...", "ok": true, "result": {...}}
    ← to client:    {"type": "error", "op_id": "...", "message": "..."}

This file requires ``websockets`` (install via ``pip install '.[daemon]'``).
The engine binding itself is light — the daemon imports ``WorldGen``
lazily so that the daemon module can be imported in lightweight envs.
"""
from __future__ import annotations

import asyncio
import json
import signal
from dataclasses import asdict
from typing import Any, Awaitable, Callable, Optional

from auroch_syna.runtime import EventBus, default_bus, get_logger
from auroch_syna.scene.commands import Command

log = get_logger(__name__)


CommandHandler = Callable[[Command], Awaitable[Any]]


class DaemonServer:
    """JSON-over-WebSocket dispatch server.

    Construct with a command handler; the server multiplexes the event
    bus to all connected clients and forwards incoming commands to the
    handler.
    """

    def __init__(
        self,
        handler: CommandHandler,
        *,
        bus: Optional[EventBus] = None,
        host: str = "127.0.0.1",
        port: int = 8765,
    ) -> None:
        self.handler = handler
        self.bus = bus or default_bus()
        self.host = host
        self.port = port
        self._clients: set[Any] = set()
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    async def serve(self) -> None:
        try:
            import websockets
        except ImportError as e:  # pragma: no cover
            raise RuntimeError(
                "websockets is not installed. Run `pip install '.[daemon]'`."
            ) from e

        self._loop = asyncio.get_running_loop()
        unsub = self.bus.subscribe("*", self._on_bus_event)

        log.info("Daemon listening on ws://%s:%d", self.host, self.port)
        try:
            async with websockets.serve(self._client, self.host, self.port):
                stop = self._loop.create_future()
                for sig in (signal.SIGTERM, signal.SIGINT):
                    try:
                        self._loop.add_signal_handler(sig, stop.set_result, None)
                    except (NotImplementedError, RuntimeError):
                        # Windows / non-main thread
                        pass
                await stop
        finally:
            unsub()

    async def _client(self, ws) -> None:
        self._clients.add(ws)
        log.info("client connected (%d total)", len(self._clients))
        try:
            async for raw in ws:
                await self._dispatch(ws, raw)
        except Exception:  # noqa: BLE001
            log.exception("client error")
        finally:
            self._clients.discard(ws)
            log.info("client disconnected (%d total)", len(self._clients))

    async def _dispatch(self, ws, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            await ws.send(json.dumps({"type": "error", "message": "invalid json"}))
            return

        t = msg.get("type")
        if t == "subscribe":
            # We send everything to everyone today; selective subscription
            # is a future filter on the wire. (Kinds list ignored for now.)
            await ws.send(json.dumps({"type": "ack", "op_id": msg.get("op_id"), "ok": True}))
            return

        if t == "command":
            cmd = Command.from_dict(msg["command"])
            try:
                result = await self.handler(cmd)
                await ws.send(json.dumps({
                    "type": "ack", "op_id": cmd.op_id, "ok": True, "result": result,
                }))
            except Exception as e:  # noqa: BLE001
                log.exception("command failed: %s", cmd.kind)
                await ws.send(json.dumps({
                    "type": "error", "op_id": cmd.op_id, "message": str(e),
                }))
            return

        await ws.send(json.dumps({"type": "error", "message": f"unknown type: {t}"}))

    def _on_bus_event(self, event) -> None:
        """Bus → all-clients fanout.

        Bus subscribers run on the publisher's thread, so we schedule the
        fanout onto the event loop.
        """
        if self._loop is None:
            return
        payload = {
            "type": "event",
            "event": {
                "kind": event.kind,
                "payload": event.payload,
                "timestamp": event.timestamp,
                "id": event.id,
            },
        }
        encoded = json.dumps(payload)
        for ws in list(self._clients):
            asyncio.run_coroutine_threadsafe(self._safe_send(ws, encoded), self._loop)

    async def _safe_send(self, ws, raw: str) -> None:
        try:
            await ws.send(raw)
        except Exception:
            self._clients.discard(ws)


def run(host: str = "127.0.0.1", port: int = 8765) -> None:
    """Run the daemon with the default in-process engine handler."""
    asyncio.run(_run(host, port))


async def _run(host: str, port: int) -> None:
    from auroch_syna.daemon.handlers import default_handler

    server = DaemonServer(default_handler, host=host, port=port)
    await server.serve()
