import re
import warnings
from typing import Dict, Tuple, List

import torch
import safetensors.torch

from auroch_syna.runtime import get_logger

try:
    from nunchaku.lora.flux.compose import compose_lora as _nunchaku_compose_lora
except Exception:
    _nunchaku_compose_lora = None

log = get_logger(__name__)


def get_block_number(key):
    match = re.search(r"single_transformer_blocks\.(\d+)", key)
    return int(match.group(1)) if match else None


def load_and_fix_lora(lora_path: str) -> Tuple[Dict[str, torch.Tensor], float]:
    if lora_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(lora_path)
    else:
        state_dict = torch.load(lora_path, map_location="cpu")

    first_key = next(iter(state_dict.keys()))
    first_tensor = state_dict[first_key]
    rank = first_tensor.shape[0]
    in_features = first_tensor.shape[1]

    single_block_components = {
        "attn.to_k.lora_A.weight": (rank, in_features),
        "attn.to_k.lora_B.weight": (in_features, rank),
        "attn.to_q.lora_A.weight": (rank, in_features),
        "attn.to_q.lora_B.weight": (in_features, rank),
        "attn.to_v.lora_A.weight": (rank, in_features),
        "attn.to_v.lora_B.weight": (in_features, rank),
        "norm.linear.lora_A.weight": (rank, in_features),
        "norm.linear.lora_B.weight": (in_features * 3, rank),
        "proj_mlp.lora_A.weight": (rank, in_features),
        "proj_mlp.lora_B.weight": (in_features * 4, rank),
        "proj_out.lora_A.weight": (rank, in_features * 5),
        "proj_out.lora_B.weight": (in_features, rank),
    }

    transformer_block_components = {
        "attn.add_k_proj.lora_A.weight": (rank, in_features),
        "attn.add_k_proj.lora_B.weight": (in_features, rank),
        "attn.add_q_proj.lora_A.weight": (rank, in_features),
        "attn.add_q_proj.lora_B.weight": (in_features, rank),
        "attn.add_v_proj.lora_A.weight": (rank, in_features),
        "attn.add_v_proj.lora_B.weight": (in_features, rank),
        "attn.to_add_out.lora_A.weight": (rank, in_features),
        "attn.to_add_out.lora_B.weight": (in_features, rank),
        "attn.to_k.lora_A.weight": (rank, in_features),
        "attn.to_k.lora_B.weight": (in_features, rank),
        "attn.to_out.0.lora_A.weight": (rank, in_features),
        "attn.to_out.0.lora_B.weight": (in_features, rank),
        "attn.to_q.lora_A.weight": (rank, in_features),
        "attn.to_q.lora_B.weight": (in_features, rank),
        "attn.to_v.lora_A.weight": (rank, in_features),
        "attn.to_v.lora_B.weight": (in_features, rank),
        "ff.net.0.proj.lora_A.weight": (rank, in_features),
        "ff.net.0.proj.lora_B.weight": (in_features * 4, rank),
        "ff.net.2.lora_A.weight": (rank, in_features * 4),
        "ff.net.2.lora_B.weight": (in_features, rank),
        "ff_context.net.0.proj.lora_A.weight": (rank, in_features),
        "ff_context.net.0.proj.lora_B.weight": (in_features * 4, rank),
        "ff_context.net.2.lora_A.weight": (rank, in_features * 4),
        "ff_context.net.2.lora_B.weight": (in_features, rank),
        "norm1.linear.lora_A.weight": (rank, in_features),
        "norm1.linear.lora_B.weight": (in_features * 6, rank),
        "norm1_context.linear.lora_A.weight": (rank, in_features),
        "norm1_context.linear.lora_B.weight": (in_features * 6, rank),
    }

    # Derive the block count from the loaded checkpoint rather than
    # hard-coding 29 (which is FLUX.1-dev-specific and will silently
    # produce wrong results for any other FLUX variant).
    single_block_nums = set()
    transformer_block_nums = set()
    for key in state_dict:
        m = re.search(r"transformer\.single_transformer_blocks\.([0-9]+)\.", key)
        if m:
            single_block_nums.add(int(m.group(1)))
        m = re.search(r"transformer\.transformer_blocks\.([0-9]+)\.", key)
        if m:
            transformer_block_nums.add(int(m.group(1)))

    # Fall back to the known FLUX.1-dev value when the checkpoint has no
    # block keys at all (e.g. a very sparse / partial LoRA).
    n_single = max(single_block_nums) + 1 if single_block_nums else 29
    n_transformer = max(transformer_block_nums) + 1 if transformer_block_nums else 29

    # Infer dtype from the existing tensors so zero-filled entries match
    # and don't require an implicit upcast on every block.
    ref_tensor = next(iter(state_dict.values()))
    zero_dtype = ref_tensor.dtype

    for block_num in range(n_single):
        for component, shape in single_block_components.items():
            key = f"transformer.single_transformer_blocks.{block_num}.{component}"
            if key not in state_dict:
                state_dict[key] = torch.zeros(shape, dtype=zero_dtype)

    for block_num in range(n_transformer):
        for component, shape in transformer_block_components.items():
            key = f"transformer.transformer_blocks.{block_num}.{component}"
            if key not in state_dict:
                state_dict[key] = torch.zeros(shape, dtype=zero_dtype)

    return state_dict, 1.0


def compose_lora_with_fixes(lora_paths: List[Tuple[str, float]]) -> Dict[str, torch.Tensor]:
    fixed_loras = [load_and_fix_lora(path) for path, weight in lora_paths]

    if _nunchaku_compose_lora is None:
        if len(fixed_loras) > 1:
            dropped = [p for p, _ in lora_paths[1:]]
            warnings.warn(
                "nunchaku is unavailable on this platform; multi-LoRA compose is "
                f"not supported. Using the first LoRA only and dropping {len(dropped)} "
                f"others: {dropped}.",
                RuntimeWarning,
                stacklevel=2,
            )
        elif fixed_loras:
            log.info("nunchaku unavailable; using first (and only) fixed LoRA directly.")
        if not fixed_loras:
            return {}
        first_state_dict, _weight = fixed_loras[0]
        return first_state_dict

    return _nunchaku_compose_lora(fixed_loras)
