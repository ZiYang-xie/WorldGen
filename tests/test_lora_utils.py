"""Tests for lora_utils.py (CPU-only, no model weights required)."""
import sys
import os
import tempfile

import pytest
import torch
import safetensors.torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from auroch_syna.worldgen.utils.lora_utils import load_and_fix_lora


def _make_minimal_lora(path: str, rank: int = 4, in_features: int = 8):
    """Write a minimal safetensors file with one LoRA key."""
    state = {
        "transformer.single_transformer_blocks.0.attn.to_k.lora_A.weight": torch.zeros(rank, in_features),
    }
    safetensors.torch.save_file(state, path)


class TestLoadAndFixLora:
    def test_returns_state_dict_and_weight(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        _make_minimal_lora(path)
        state_dict, weight = load_and_fix_lora(path)
        assert isinstance(state_dict, dict)
        assert weight == 1.0
        os.unlink(path)

    def test_fills_missing_single_block_keys(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        _make_minimal_lora(path, rank=4, in_features=8)
        state_dict, _ = load_and_fix_lora(path)
        # Should have filled in all 29 blocks × 12 components
        single_block_keys = [k for k in state_dict if "single_transformer_blocks" in k]
        assert len(single_block_keys) == 29 * 12
        os.unlink(path)

    def test_filled_keys_are_zero_tensors(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors", delete=False) as f:
            path = f.name
        _make_minimal_lora(path, rank=4, in_features=8)
        state_dict, _ = load_and_fix_lora(path)
        # Block 1+ were not in original state — should be zeros
        key = "transformer.single_transformer_blocks.1.attn.to_k.lora_A.weight"
        assert torch.all(state_dict[key] == 0)
        os.unlink(path)
