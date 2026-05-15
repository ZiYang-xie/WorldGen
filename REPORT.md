Auroch Syna (Syna) — Improvement Audit (concise)

Summary
- Codebase is a Python, ML-first prototype centered on PyTorch and diffusion pipelines for panorama→scene reconstruction.
- Not currently architected for cross-platform, low-latency OS runtime: heavy Python/native PyTorch dependencies, no typed affordances, and no state sync primitives.

Top priorities
1. Split into: native deterministic core (Rust/C++, wgpu + ECS), ML services (model runners/microservices), and adapter layers (iOS/Android/Windows/Web).
2. Introduce an `Entity-Component` schema and `Affordance<T>` primitive library with sidecar metadata (protobuf/CBOR) for assets.
3. Add a versioned snapshot + CRDT diff sync system and CloudKit/pubsub adapters for low-latency handoff.
4. Move latency‑sensitive rendering and physics to native runtime; provide CoreML/MPS/ONNX runners for on‑device ML.
5. Add small IPC/API hooks for Winnie/orchestrator/command bar: `captureSelection()`, `invokeCommand()`, `registerContextListener()`.

Files inspected (key)
- src/worldgen/worldgen.py
- src/worldgen/pano_gen.py
- src/worldgen/pano_depth.py
- src/worldgen/pano_seg.py
- src/worldgen/pano_inpaint.py
- src/worldgen/pano_sharp.py
- src/worldgen/models/flux_pano_gen_pipeline.py
- src/worldgen/models/flux_pano_fill_pipeline.py
- src/worldgen/models/inpaint_model.py
- src/worldgen/utils/lora_utils.py
- src/worldgen/utils/general_utils.py
- src/worldgen/utils/splat_utils.py
- demo.py, README.md, worldgen.egg-info

Next actions
- I implemented a thin `ModelClient` wrapper and updated `WorldGen` to use it.
- I added `src/auroch_syna` re-export package and updated `demo.py` imports.
- I can now: run the automated rebrand (full move + import updates), or start extracting the ML client into a separate process/service.
