"""In-process event bus.

The bus is the single way the ML pipeline communicates *outward* —
progress, lifecycle, telemetry, and (eventually) selection / command
results. The daemon, the CLI, and the viewer all subscribe to the same
bus; an out-of-process consumer (Winnie, the command bar) sees the same
event stream via the daemon's WebSocket transport.

Events are plain dataclasses for cheap serialization. Keep the payload
JSON-friendly.
"""
from __future__ import annotations

import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional


@dataclass(frozen=True)
class Event:
    """A single event published on the bus.

    ``kind`` is a dotted string like ``pipeline.progress`` or
    ``selection.changed``. Payload must be JSON-serializable.
    """

    kind: str
    payload: dict
    timestamp: float = field(default_factory=time.time)
    id: str = field(default_factory=lambda: uuid.uuid4().hex)

    def to_dict(self) -> dict:
        return asdict(self)


Subscriber = Callable[[Event], None]


class EventBus:
    """Tiny synchronous fan-out bus.

    - Subscribers are called on the publisher's thread; keep them fast.
    - For async / out-of-process consumers, use the daemon transport
      which adapts the bus onto a WebSocket.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._subscribers: dict[str, list[Subscriber]] = {}
        self._wildcard: list[Subscriber] = []

    def subscribe(self, kind: str, fn: Subscriber) -> Callable[[], None]:
        """Subscribe to a kind. Use ``"*"`` to subscribe to all events.

        Returns an unsubscribe callable.
        """
        with self._lock:
            if kind == "*":
                self._wildcard.append(fn)
            else:
                self._subscribers.setdefault(kind, []).append(fn)

        def _unsub() -> None:
            with self._lock:
                if kind == "*":
                    if fn in self._wildcard:
                        self._wildcard.remove(fn)
                else:
                    lst = self._subscribers.get(kind, [])
                    if fn in lst:
                        lst.remove(fn)

        return _unsub

    def publish(self, kind: str, payload: Optional[dict] = None) -> Event:
        event = Event(kind=kind, payload=payload or {})
        with self._lock:
            subs = list(self._subscribers.get(kind, []))
            wild = list(self._wildcard)
        for fn in subs:
            _safe_call(fn, event)
        for fn in wild:
            _safe_call(fn, event)
        return event

    def subscribers(self, kind: str) -> Iterable[Subscriber]:
        with self._lock:
            return list(self._subscribers.get(kind, [])) + list(self._wildcard)


def _safe_call(fn: Subscriber, event: Event) -> None:
    try:
        fn(event)
    except Exception:  # noqa: BLE001 — bus must never bring down the publisher
        from .logging import get_logger

        get_logger("auroch_syna.runtime.bus").exception(
            "subscriber raised on event %s", event.kind
        )


_default_bus = EventBus()


def default_bus() -> EventBus:
    """The process-wide event bus.

    Most callers should use this. Tests can instantiate their own
    ``EventBus`` for isolation.
    """
    return _default_bus
