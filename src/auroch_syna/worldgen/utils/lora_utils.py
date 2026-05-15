import re
from typing import Dict, Tuple, List

import torch
import safetensors.torch

try:
    from nunchaku.lora.flux.compose import compose_lora as _nunchaku_compose_lora
except Exception:
    _nunchaku_compose_lora = None


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

    for block_num in range(29):
        for component, shape in single_block_components.items():
            key = f"transformer.single_transformer_blocks.{block_num}.{component}"
            if key not in state_dict:
                state_dict[key] = torch.zeros(shape)

    for block_num in range(29):
        for component, shape in transformer_block_components.items():
            key = f"transformer.transformer_blocks.{block_num}.{component}"
            if key not in state_dict:
                state_dict[key] = torch.zeros(shape)

    return state_dict, 1.0


def compose_lora_with_fixes(lora_paths: List[Tuple[str, float]]) -> Dict[str, torch.Tensor]:
    fixed_loras = [load_and_fix_lora(path) for path, weight in lora_paths]

    if _nunchaku_compose_lora is None:
        print("[WorldGen] nunchaku unavailable on macOS; using first fixed LoRA fallback.")
        if not fixed_loras:
            return {}
        first_state_dict, _weight = fixed_loras[0]
        return first_state_dict

    return _nunchaku_compose_lora(fixed_loras)
