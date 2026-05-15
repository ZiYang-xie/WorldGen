Auroch Syna — Full Improvement Audit

Scope
- Goal: transform this repo from a single-process ML prototype into a cross-platform, context-aware world-building environment that can serve as Auroch’s OS layer.
- Deliverables here: concise assessment, inspected file list with links, prioritized recommendations, concrete next steps and quick fixes.

Quick summary
- The codebase is ML-first (PyTorch/diffusers) and contains heavy native/Python bindings (torch, pytorch3d, open3d, nunchaku). This makes cross-platform distribution and low-latency on-device execution difficult.
- Good strides already taken: `ModelClient` abstraction, `auroch_syna` compatibility shim, lazy import CI smoke test.

Key inspected files
- Core orchestrator: [src/auroch_syna/worldgen/worldgen.py](src/auroch_syna/worldgen/worldgen.py#L1)
- Pipeline adapters: [src/auroch_syna/worldgen/pano_gen.py](src/auroch_syna/worldgen/pano_gen.py#L1), [src/auroch_syna/worldgen/pano_fill.py](src/auroch_syna/worldgen/pano_gen.py#L63)
- Depth/inpaint/seg/sharp adapters: [src/auroch_syna/worldgen/pano_depth.py](src/auroch_syna/worldgen/pano_depth.py#L1), [src/auroch_syna/worldgen/pano_inpaint.py](src/auroch_syna/worldgen/pano_inpaint.py#L1), [src/auroch_syna/worldgen/pano_seg.py](src/auroch_syna/worldgen/pano_seg.py#L1), [src/auroch_syna/worldgen/pano_sharp.py](src/auroch_syna/worldgen/pano_sharp.py#L1)
- Model wrappers: [src/auroch_syna/worldgen/models/flux_pano_gen_pipeline.py](src/auroch_syna/worldgen/models/flux_pano_gen_pipeline.py#L1), [src/auroch_syna/worldgen/models/flux_pano_fill_pipeline.py](src/auroch_syna/worldgen/models/flux_pano_fill_pipeline.py#L1)
- Utilities: [src/auroch_syna/worldgen/utils/splat_utils.py](src/auroch_syna/worldgen/utils/splat_utils.py#L1), [src/auroch_syna/worldgen/utils/lora_utils.py](src/auroch_syna/worldgen/utils/lora_utils.py#L1)
- Packaging / metadata: [pyproject.toml](pyproject.toml#L1), [src/auroch_syna.egg-info/PKG-INFO](src/auroch_syna.egg-info/PKG-INFO#L1)
- Demo + visualization: [demo.py](demo.py#L1), `submodules/viser` (web visualizer)

Heavy dependencies and platform concerns
- Heavy native/Python dependencies visible in `pyproject.toml`: `torch`, `pytorch3d` (submodule), `open3d`, `nunchaku` (platform wheel), `viser` (submodule), `ml-sharp` (submodule). These require platform-specific builds and have large binary footprints.
- Code contains explicit device branching and CUDA checks: `torch.cuda.is_available()` and `torch.cuda.get_device_properties(0)` (affects default device decisions).
- `nunchaku` import guarded but used for low_vram; fallback logic exists (see `pano_gen.py`). `pytorch3d` is used in `pano_sharp.py`.
- Submodules are present and modified (`submodules/viser`, `submodules/DA-2`, `submodules/ml-sharp`) — these complicate clean upstream sync and packaging.

Primary risks / blockers
1. Monolithic coupling of orchestration + ML pipelines prevents shipping a lightweight runtime. Current `WorldGen` imports torch/open3d at module import time in some places.
2. Submodules and pinned external git/Wheel deps make reproducible builds harder.
3. No explicit runtime contract/API boundary for deterministic core vs ML services (ModelClient is a good start but needs to be the canonical boundary).
4. No tests or CI beyond the smoke-import; no automated validation of core behavior or cross-platform builds.

Priority recommendations (short)
1. Enforce the `ModelClient` boundary: extract model loads into a separate process/service with a small RPC (gRPC/HTTP/Unix socket) so the main runtime can run headless/native without torch.
2. Make ML deps optional extras in `pyproject.toml` (`[project.optional-dependencies]`) and keep the core package minimal.
3. Add small native/deterministic core library (Rust or C++) for heavy real-time work; implement a thin Python adapter. This enables on-device ports later (WASM, iOS, Android).
4. Convert inference models to deployable formats for each platform: TorchScript/ONNX for Linux/Windows, CoreML for iOS, and an MPS-friendly pipeline for macOS. Provide conversion scripts under `tools/`.
5. Improve packaging and CI: add wheels-building CI matrix, publish separate ML wheel (heavy) vs runtime wheel (light). Add unit/integration tests for the `WorldGen` orchestrator using a mocked `ModelClient`.
6. Replace direct submodule modifications with pinned package dependencies where possible; keep submodules only for tightly-coupled local dev targets.

Concrete near-term tasks (what I can implement now)
- Add a small test/mock harness that imports `auroch_syna` and instantiates `WorldGen` with a mocked `ModelClient` (CI friendly). (Next task)
- Make `pyproject.toml` list ML deps under `optional-dependencies = { ml = [...] }` so `pip install .[ml]` is explicit.
- Add `tools/convert_model.py` skeleton for TorchScript/ONNX conversions.
- Add CI matrix entries to build wheels and run smoke imports on multiple Python versions/platforms.

Files to refactor (high value)
- `src/auroch_syna/worldgen/worldgen.py` — split orchestration and model creation; avoid importing heavy libs at top-level.
- `src/auroch_syna/worldgen/*` pipeline modules — ensure `build_*_model` functions are purely factory functions and are only called by `ModelClient`.
- `demo.py` — move heavy imports behind command execution and provide a `--no-ml` mode that only demonstrates the web visualizer with static assets.

Next steps I can take now (choose one or more)
- Create `AUDIT_FULL.md` (done) and a `tests/test_smoke_import.py` that uses a mocked `ModelClient` and runs in CI.
- Convert heavy dependencies in `pyproject.toml` into an `ml` optional requirement group.
- Implement `tools/convert_model.py` skeleton and a `README.md` with conversion guidance.
- Start extracting ModelClient RPC skeleton (HTTP/gRPC) and a tiny local server that returns canned responses for unit tests.

Notes about submodules
- `submodules/viser` and `submodules/ml-sharp` are present and show local changes; decide whether to keep submodules or replace them with PyPI/gh packages. If you want to preserve exact local dev versions, keep them but commit the submodule states or vendor only the necessary parts.

If you'd like, I will now:
- Add `tests/test_smoke_import.py` (mocked `ModelClient`) and update CI to run it, or
- Convert `pyproject.toml` dependencies to include an `ml` extras group, or
- Begin implementing the ModelClient RPC skeleton.

Pick one and I will proceed and commit the change.
