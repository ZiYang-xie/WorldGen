"""Smoke import tests — no torch / open3d required.

These run in any environment that has just ``pip install .`` (no [ml] extras).
"""
import importlib
import sys


def test_top_level_import():
    import auroch_syna
    assert auroch_syna.__version__


def test_runtime_import():
    from auroch_syna.runtime import EventBus, get_logger, resolve_device, DeviceInfo


def test_scene_import():
    from auroch_syna.scene.ir import SceneSnapshot, SemanticObject, Pose, Affordance
    from auroch_syna.scene.commands import Command
    from auroch_syna.scene.events import ProgressEvent, SelectionEvent


def test_daemon_import():
    from auroch_syna.daemon.server import DaemonServer


def test_project_import():
    from auroch_syna.project.bundle import ProjectBundle


def test_cli_import():
    from auroch_syna.cli import main, _build_parser


def test_cli_help(capsys):
    from auroch_syna.cli import main
    try:
        main(["--help"])
    except SystemExit as e:
        assert e.code == 0
    out = capsys.readouterr().out
    assert "generate" in out


def test_worldgen_lazy():
    """WorldGen should NOT be imported at the top-level package level."""
    import auroch_syna
    assert "auroch_syna.worldgen.worldgen" not in sys.modules, (
        "worldgen.worldgen was imported eagerly — that pulls in torch at startup"
    )
