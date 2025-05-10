import os
import torch
import safetensors.torch
from nunchaku.lora.flux.compose import compose_lora
import re
from typing import Dict, Tuple, List

def get_block_number(key):
    """Extract block number from key if present."""
    match = re.search(r'single_transformer_blocks\.(\d+)', key)
    return int(match.group(1)) if match else None

def load_and_fix_lora(lora_path: str) -> Tuple[Dict[str, torch.Tensor], float]:
    """Load and fix LoRA weights, ensuring all required components are present."""
    # Load the state dict
    if lora_path.endswith(".safetensors"):
        state_dict = safetensors.torch.load_file(lora_path)
    else:
        state_dict = torch.load(lora_path, map_location="cpu")
    
    # Get reference shapes from the first key
    first_key = next(iter(state_dict.keys()))
    first_tensor = state_dict[first_key]
    rank = first_tensor.shape[0]
    in_features = first_tensor.shape[1]
    
    # Define required components for each block type with their specific shapes
    single_block_components = {
        "attn.to_k.lora_A.weight": (rank, in_features),
        "attn.to_k.lora_B.weight": (in_features, rank),
        "attn.to_q.lora_A.weight": (rank, in_features),
        "attn.to_q.lora_B.weight": (in_features, rank),
        "attn.to_v.lora_A.weight": (rank, in_features),
        "attn.to_v.lora_B.weight": (in_features, rank),
        "norm.linear.lora_A.weight": (rank, in_features),
        "norm.linear.lora_B.weight": (in_features * 3, rank),  # 9216 = 3072 * 3
        "proj_mlp.lora_A.weight": (rank, in_features),
        "proj_mlp.lora_B.weight": (in_features * 4, rank),  # 12288 = 3072 * 4
        "proj_out.lora_A.weight": (rank, in_features * 5),  # 15360 = 3072 * 5
        "proj_out.lora_B.weight": (in_features, rank)
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
        "ff.net.0.proj.lora_B.weight": (in_features * 4, rank),  # 12288 = 3072 * 4
        "ff.net.2.lora_A.weight": (rank, in_features * 4),  # 12288 = 3072 * 4
        "ff.net.2.lora_B.weight": (in_features, rank),
        "ff_context.net.0.proj.lora_A.weight": (rank, in_features),
        "ff_context.net.0.proj.lora_B.weight": (in_features * 4, rank),  # 12288 = 3072 * 4
        "ff_context.net.2.lora_A.weight": (rank, in_features * 4),  # 12288 = 3072 * 4
        "ff_context.net.2.lora_B.weight": (in_features, rank),
        "norm1.linear.lora_A.weight": (rank, in_features),
        "norm1.linear.lora_B.weight": (in_features * 6, rank),  # 18432 = 3072 * 6
        "norm1_context.linear.lora_A.weight": (rank, in_features),
        "norm1_context.linear.lora_B.weight": (in_features * 6, rank)  # 18432 = 3072 * 6
    }
    
    # Create missing weights for single transformer blocks (0-28)
    for block_num in range(29):  # Example LoRA has blocks 0-28
        for component, shape in single_block_components.items():
            key = f"transformer.single_transformer_blocks.{block_num}.{component}"
            if key not in state_dict:
                state_dict[key] = torch.zeros(shape)
    
    # Create missing weights for transformer blocks (0-28)
    for block_num in range(29):  # Example LoRA has blocks 0-28
        for component, shape in transformer_block_components.items():
            key = f"transformer.transformer_blocks.{block_num}.{component}"
            if key not in state_dict:
                state_dict[key] = torch.zeros(shape)
    
    # Return the state dict and weight (1.0 for now)
    return state_dict, 1.0

def compose_lora_with_fixes(lora_paths: List[Tuple[str, float]]) -> Dict[str, torch.Tensor]:
    """Compose multiple LoRAs after fixing any missing keys."""
    # Load and fix each LoRA
    fixed_loras = [load_and_fix_lora(path) for path, weight in lora_paths]
    
    # Compose the fixed LoRAs
    return compose_lora(fixed_loras) 