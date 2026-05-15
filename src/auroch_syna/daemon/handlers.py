"""Default command handlers for the daemon.

The handlers themselves are kept thin so the engine's logic stays in
``worldgen.py``. Most handlers are stubs today; they exist so the
daemon's wire protocol can be exercised end-to-end without an ML
install.
"""
from __future__ import annotations

import asyncio
from typing import Any

from auroch_syna.runtime import get_logger
from auroch_syna.scene.commands import Command

log = get_logger(__name__)


async def default_handler(cmd: Command) -> Any:
    """Dispatch ``cmd`` to the matching handler.

    Handlers are async; ML calls are run in a default executor to keep
    the WebSocket event loop responsive.
    """
    fn = _HANDLERS.get(cmd.kind, _noop)
    return await fn(cmd)


async def _noop(cmd: Command) -> dict:
    log.info("noop command kind=%s op_id=%s", cmd.kind, cmd.op_id)
    return {"ok": True, "kind": cmd.kind}


async def _generate_scene(cmd: Command) -> dict:
    """Run ``WorldGen`` once.

    Lazy-imports so the daemon module is importable without the [ml]
    extra. Runs inside the default executor.
    """
    loop = asyncio.get_running_loop()

    def _run():
        from auroch_syna import WorldGen  # heavy

        params = cmd.params
        wg = WorldGen(
            mode=params.get("mode", "t2s"),
            use_sharp=params.get("use_sharp", False),
            inpaint_bg=params.get("inpaint_bg", False),
            resolution=params.get("resolution", 1600),
            device=params.get("device", "auto"),
        )
        scene = wg.generate_world(
            prompt=params.get("prompt", ""),
            return_mesh=params.get("return_mesh", False),
        )
        return {"object_count": _count(scene)}

    return await loop.run_in_executor(None, _run)


def _count(scene) -> int:
    centers = getattr(scene, "centers", None)
    if centers is not None:
        try:
            return int(centers.shape[0])
        except Exception:
            return -1
    return -1


async def _select(cmd: Command) -> dict:
    log.info("selection: %s", cmd.params)
    return {"ok": True}


async def _set_affordance(cmd: Command) -> dict:
    log.info("set_affordance: %s", cmd.params)
    return {"ok": True, "note": "set_affordance is a stub; scene state mutation TBD"}


_HANDLERS = {
    "generate_scene": _generate_scene,
    "select": _select,
    "set_affordance": _set_affordance,
}
