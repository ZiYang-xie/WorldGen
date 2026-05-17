**Auroch Syna — Architecture & Roadmap**

> See [REPORT.md](./REPORT.md) for the concrete bug-fix audit and
> resolution status of individual issues.  This document owns the
> higher-level design intent and migration plan.

---

## Purpose

Evolve this repo from a research prototype into a cross-platform,
context-aware world-building environment with:

- A clean separation between the deterministic runtime and the ML service
  layer.
- Semantic primitives for state, handoff, and collaborative editing.
- Practical on-device inference (CoreML / ONNX / TorchScript).
- Good developer ergonomics (CLI, typed API, CI, packaging).

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Desktop App  (apps/desktop — Tauri + React)                │
│  Web Viewer   (Viser — demo.py)                             │
│  CLI          (auroch-syna — src/auroch_syna/cli.py)        │
└────────────────────────┬────────────────────────────────────┘
                         │ Commands / Events (EventBus)
┌────────────────────────▼────────────────────────────────────┐
│  Runtime Layer  (src/auroch_syna/runtime/)                  │
│  • device.py     — resolve_device(), DeviceInfo             │
│  • policy.py     — select_policy(), RuntimePolicy           │
│  • transport.py  — ModelHandle protocol, InProcessHandle    │
│  • bus.py        — EventBus (publish/subscribe)             │
│  • logging.py    — get_logger(), configure()                │
│  • torch_compat.py — safe_autocast()                        │
└────────────────────────┬────────────────────────────────────┘
                         │ ModelHandle.infer()
┌────────────────────────▼────────────────────────────────────┐
│  ML Service Layer  (src/auroch_syna/worldgen/)              │
│  • model_client.py  — ModelClient (build + cache handles)   │
│  • worldgen.py      — WorldGen orchestrator                 │
│  • pano_gen.py      — FLUX panorama generation              │
│  • pano_depth.py    — DA-2 depth estimation                 │
│  • pano_seg.py      — OneFormer segmentation                │
│  • pano_inpaint.py  — LaMa inpainting                       │
│  • pano_sharp.py    — Apple ml-sharp (optional)             │
│  • backends/        — RPC client, MLX backend, mock server  │
└────────────────────────┬────────────────────────────────────┘
                         │ SplatFile / TriangleMesh
┌────────────────────────▼────────────────────────────────────┐
│  Scene Layer  (src/auroch_syna/scene/)                      │
│  • ir.py      — SceneSnapshot, SemanticObject, Affordance   │
│  • crdt.py    — LWWRegister, ORSet, SceneDelta, diff/merge  │
│  • commands.py — Command (serializable intent)              │
│  • events.py   — typed event definitions                    │
└────────────────────────┬────────────────────────────────────┘
                         │ ProjectBundle
┌────────────────────────▼────────────────────────────────────┐
│  Project Layer  (src/auroch_syna/project/)                  │
│  • bundle.py  — .aurochsyna directory (scene + assets +     │
│                 edit_log + manifest)                        │
└─────────────────────────────────────────────────────────────┘
```

---

## Semantic Primitives

| Primitive | Location | Purpose |
|-----------|----------|---------|
| `Affordance<T>` | `scene/ir.py` | Typed, minimal state unit (e.g. `Affordance(kind="surface", params={...})`) |
| `SceneSnapshot` | `scene/ir.py` | Immutable, JSON-round-trippable scene state |
| `SceneDelta` | `scene/crdt.py` | Compact diff between two snapshots |
| `LWWRegister` | `scene/crdt.py` | Last-Write-Wins register for scalar fields |
| `ORSet` | `scene/crdt.py` | Observed-Remove Set for object collections |
| `Command` | `scene/commands.py` | Serializable user intent |
| `ModelHandle` | `runtime/transport.py` | Protocol: `infer(*args) -> outputs` |
| `RuntimePolicy` | `runtime/policy.py` | Device + dtype + backend selection |
| `ProjectBundle` | `project/bundle.py` | On-disk project format (`.aurochsyna/`) |

---

## Performance & Local Edge

- **Model warm/cold policy** is implemented in `ModelClient` (lazy build,
  cached handles, optional CPU offload via `RuntimePolicy`).
- **Device resolution** is centralised in `runtime/device.py`
  (`resolve_device()`) and `runtime/policy.py` (`select_policy()`).
  All worldgen builders now route through these helpers; no module
  hard-codes `device='cuda'` as a default.
- **`safe_autocast`** in `runtime/torch_compat.py` no-ops on unsupported
  device types, fixing the CPU crash.
- **VAE tiling** is enabled by default in `build_pano_gen_model` and
  `build_pano_fill_model` for large panorama decoding.
- **MLX backend** for Apple Silicon is probed by `select_policy()` and
  loaded lazily by `backends/mlx_flux.py`.

---

## Service Integration

The `ModelHandle` protocol (`runtime/transport.py`) is the boundary
between callers and inference backends:

```python
class ModelHandle(Protocol):
    name: str
    def infer(self, *args, **kwargs) -> Any: ...
    def raw(self) -> Any: ...
```

`InProcessHandle` wraps a built pipeline + inference callable.
`RemoteHandle` is a placeholder that raises `NotImplementedError` until
the out-of-process transport is implemented.

`ModelClientRPC` (`worldgen/backends/rpc_client.py`) provides a thin HTTP
JSON client for development-time remote model service calls.

---

## Developer UX

- **CLI**: `auroch-syna {info,daemon,generate,generate-from}` — ML imports
  are lazy so `auroch-syna info` works without GPU or ML extras.
- **CI**: Three jobs — `lint` (ruff), `unit-tests` (pytest, no GPU),
  `smoke-import` (lazy-shim verification).
- **Tests**: `tests/` covers scene IR, CRDT primitives, project bundle,
  CLI, model client RPC, and worldgen math utilities.
- **Packaging**: `pip install .` for the runtime/scene/CLI; `pip install
  ".[ml]"` for the full ML pipeline.

---

## Roadmap

### Near-term (next 1–2 weeks)

- [ ] Mirror Sharp + LaMa model weights on HuggingFace with pinned sha256s.
- [ ] Promote FLUX monkey-patched VAE helpers to real subclass methods.
- [ ] Add `mypy` to CI once type annotations are more complete.

### Medium-term (1–2 months)

- [ ] Implement `RemoteHandle` transport (WebSocket / gRPC) so the ML
      service can run out-of-process on a GPU machine while the desktop
      app runs on CPU.
- [ ] Add vector clocks / hybrid logical clocks to `SceneSnapshot` objects
      for production-grade CRDT merging.
- [ ] Streaming panorama decode in the FLUX pipelines (tiling + streaming
      for large resolutions).

### Longer-term

- [ ] Native Rust/wgpu renderer for the desktop app.
- [ ] CoreML / ONNX / TorchScript export pipeline for on-device inference.
- [ ] Multi-user collaborative editing via the CRDT layer + daemon WebSocket.

---

## Where to Start in Code

| Goal | Entry point |
|------|-------------|
| Run the demo | `demo.py` |
| Generate programmatically | `src/auroch_syna/worldgen/worldgen.py` — `WorldGen` |
| Understand device selection | `src/auroch_syna/runtime/policy.py` — `select_policy()` |
| Understand model construction | `src/auroch_syna/worldgen/model_client.py` — `ModelClient` |
| Understand the scene format | `src/auroch_syna/scene/ir.py` — `SceneSnapshot` |
| Understand collaborative sync | `src/auroch_syna/scene/crdt.py` — `diff`, `merge` |
| Understand the project format | `src/auroch_syna/project/bundle.py` — `ProjectBundle` |
| Add a new ML backend | `src/auroch_syna/worldgen/backends/` |
