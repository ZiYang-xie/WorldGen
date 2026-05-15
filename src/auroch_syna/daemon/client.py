"""WebSocket client for the Auroch Syna daemon.

Provides ``DaemonClient`` — an async-first JSON-over-WebSocket client
that mirrors the wire protocol in ``server.py``, and ``DaemonClient.sync``
for simple blocking callers.

Wire protocol (matches server.py):
    → send:  {"type": "command", "command": {...}}
    ← recv:  {"type": "ack",   "op_id": "...", "ok": true, "result": {...}}
    ← recv:  {"type": "error", "op_id": "...", "message": "..."}
    ← recv:  {"type": "event", "event": {...}}

Install deps: pip install '.[daemon]'
"""
from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any, Callable, Optional

from auroch_syna.runtime import get_logger
from auroch_syna.scene.commands import Command

log = get_logger(__name__)

EventCallback = Callable[[dict], None]


class DaemonClient:
    """Async WebSocket client for a running ``DaemonServer``.

    Usage (async)::

        async with DaemonClient("ws://127.0.0.1:8765") as client:
            result = await client.command("generate_scene", {"prompt": "a beach"})

    Usage (sync, blocks the calling thread)::

        result = DaemonClient.run_command(
            "ws://127.0.0.1:8765",
            "generate_scene",
            {"prompt": "a beach"},
        )
    """

    def __init__(
        self,
        endpoint: str = "ws://127.0.0.1:8765",
        *,
        on_event: Optional[EventCallback] = None,
        timeout: float = 120.0,
    ) -> None:
        self.endpoint = endpoint
        self.on_event = on_event
        self.timeout = timeout
        self._ws: Any = None
        self._pending: dict[str, asyncio.Future] = {}
        self._recv_task: Optional[asyncio.Task] = None

    # ---- context manager ----

    async def __aenter__(self) -> "DaemonClient":
        await self.connect()
        return self

    async def __aexit__(self, *_) -> None:
        await self.close()

    # ---- connection ----

    async def connect(self) -> None:
        try:
            import websockets
        except ImportError as e:
            raise RuntimeError(
                "websockets is not installed. Run `pip install '.[daemon]'`."
            ) from e

        self._ws = await websockets.connect(self.endpoint)
        loop = asyncio.get_running_loop()
        self._recv_task = loop.create_task(self._recv_loop())
        log.info("Connected to daemon at %s", self.endpoint)

    async def close(self) -> None:
        if self._recv_task:
            self._recv_task.cancel()
            try:
                await self._recv_task
            except asyncio.CancelledError:
                pass
        if self._ws:
            await self._ws.close()
        self._ws = None
        log.info("Disconnected from daemon")

    # ---- commands ----

    async def command(self, kind: str, params: Optional[dict] = None) -> Any:
        """Send a command and await the matching ack/error response."""
        if self._ws is None:
            raise RuntimeError("Not connected — call connect() or use async with.")

        cmd = Command(kind=kind, params=params or {})
        payload = json.dumps({"type": "command", "command": cmd.to_dict()})

        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending[cmd.op_id] = fut

        await self._ws.send(payload)
        try:
            return await asyncio.wait_for(fut, timeout=self.timeout)
        except asyncio.TimeoutError:
            self._pending.pop(cmd.op_id, None)
            raise TimeoutError(
                f"command {kind!r} (op_id={cmd.op_id}) timed out after {self.timeout}s"
            )

    async def subscribe(self, kinds: Optional[list[str]] = None) -> None:
        """Ask the server to deliver events (currently a no-op server-side)."""
        if self._ws is None:
            raise RuntimeError("Not connected.")
        msg = {"type": "subscribe", "kinds": kinds or ["*"], "op_id": uuid.uuid4().hex}
        await self._ws.send(json.dumps(msg))

    # ---- receive loop ----

    async def _recv_loop(self) -> None:
        try:
            async for raw in self._ws:
                self._dispatch(raw)
        except Exception:  # noqa: BLE001
            log.exception("receive loop error")
            for fut in self._pending.values():
                if not fut.done():
                    fut.set_exception(ConnectionError("daemon disconnected"))
            self._pending.clear()

    def _dispatch(self, raw: str) -> None:
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            log.warning("received invalid JSON from daemon")
            return

        t = msg.get("type")
        op_id = msg.get("op_id")

        if t in ("ack", "error") and op_id and op_id in self._pending:
            fut = self._pending.pop(op_id)
            if fut.done():
                return
            if t == "ack" and msg.get("ok"):
                fut.set_result(msg.get("result"))
            else:
                fut.set_exception(RuntimeError(msg.get("message", "daemon error")))

        elif t == "event":
            event = msg.get("event", {})
            if self.on_event:
                try:
                    self.on_event(event)
                except Exception:  # noqa: BLE001
                    log.exception("on_event callback raised")

    # ---- sync convenience ----

    @classmethod
    def run_command(
        cls,
        endpoint: str,
        kind: str,
        params: Optional[dict] = None,
        *,
        timeout: float = 120.0,
    ) -> Any:
        """Blocking wrapper — run a single command and return the result.

        Creates a temporary event loop. Don't call from inside an async
        context (use ``command()`` directly there).
        """

        async def _go():
            async with cls(endpoint, timeout=timeout) as client:
                return await client.command(kind, params)

        return asyncio.run(_go())

    @classmethod
    def ping(cls, endpoint: str = "ws://127.0.0.1:8765", *, timeout: float = 5.0) -> bool:
        """Return True if the daemon at ``endpoint`` responds to a noop."""
        try:
            cls.run_command(endpoint, "noop", timeout=timeout)
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Remote ModelClient — routes WorldGen calls to a daemon
# ---------------------------------------------------------------------------

class DaemonModelClient:
    """Drop-in replacement for ``ModelClient`` that calls a remote daemon.

    Instead of loading models locally, this client sends ``generate_scene``
    commands to a running ``DaemonServer`` and returns the result dict.

    Use this when you want the ML process isolated from the orchestrator
    process (e.g., keeping Winnie's process lean while inference runs in
    a background daemon).

    Example::

        client = DaemonModelClient("ws://127.0.0.1:8765")
        result = client.generate_scene(
            prompt="a mountain lake",
            mode="t2s",
            resolution=1600,
        )
        # result is the dict returned by the daemon handler
    """

    def __init__(
        self,
        endpoint: str = "ws://127.0.0.1:8765",
        timeout: float = 300.0,
    ) -> None:
        self.endpoint = endpoint
        self.timeout = timeout

    def generate_scene(
        self,
        *,
        prompt: str = "",
        mode: str = "t2s",
        use_sharp: bool = False,
        inpaint_bg: bool = False,
        resolution: int = 1600,
        return_mesh: bool = False,
        device: Optional[str] = None,
    ) -> dict:
        """Send a generate_scene command; return the daemon's result dict."""
        params = {
            "prompt": prompt,
            "mode": mode,
            "use_sharp": use_sharp,
            "inpaint_bg": inpaint_bg,
            "resolution": resolution,
            "return_mesh": return_mesh,
        }
        if device:
            params["device"] = device

        return DaemonClient.run_command(
            self.endpoint, "generate_scene", params, timeout=self.timeout
        )

    def is_available(self) -> bool:
        """Return True if the daemon is reachable."""
        return DaemonClient.ping(self.endpoint, timeout=5.0)
