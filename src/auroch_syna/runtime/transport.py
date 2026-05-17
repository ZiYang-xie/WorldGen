"""Model transport — the seam between callers and inference backends.

Callers receive a ``ModelHandle`` and call ``.infer(*args, **kwargs)``.
Implementations:

- ``InProcessHandle``: wraps a built pipeline + an inference function.
  Drop-in for today's "call the pipe directly" pattern.
- ``RemoteHandle``: stub. Speaks JSON over a transport (WebSocket / UNIX
  socket). The daemon serves the matching adapter.

The whole point of this module is that consumers of ``ModelClient`` no
longer reach into the raw pipeline object — they call ``.infer()``.
"""
from __future__ import annotations

from typing import Any, Callable, Protocol, runtime_checkable


@runtime_checkable
class ModelHandle(Protocol):
    """Abstract handle to a built (local or remote) model."""

    name: str

    def infer(self, *args: Any, **kwargs: Any) -> Any: ...

    def raw(self) -> Any:
        """Escape hatch: the raw underlying pipeline (in-process only).

        Out-of-process handles raise ``NotImplementedError``.
        """
        ...


class InProcessHandle:
    """Wraps a built pipeline plus an inference callable."""

    def __init__(
        self,
        name: str,
        pipeline: Any,
        infer_fn: Callable[..., Any] | None = None,
    ) -> None:
        self.name = name
        self._pipeline = pipeline
        # If no infer_fn is given, treat the pipeline as callable.
        self._infer_fn = infer_fn or (lambda *a, **kw: pipeline(*a, **kw))

    def infer(self, *args: Any, **kwargs: Any) -> Any:
        return self._infer_fn(self._pipeline, *args, **kwargs)

    def raw(self) -> Any:
        return self._pipeline


class RemoteHandle:
    """Stub for out-of-process inference.

    Real implementation will live in ``auroch_syna.daemon.client``.
    Today this is a placeholder so the rest of the system can be wired
    against the right type.
    """

    def __init__(self, name: str, endpoint: str) -> None:
        self.name = name
        self.endpoint = endpoint

    def infer(self, *args: Any, **kwargs: Any) -> Any:  # pragma: no cover
        raise NotImplementedError(
            "RemoteHandle is a stub; the daemon transport is not yet implemented. "
            "Use InProcessHandle for now."
        )

    def raw(self) -> Any:
        raise NotImplementedError("RemoteHandle has no in-process pipeline.")


__all__ = ["ModelHandle", "InProcessHandle", "RemoteHandle"]
