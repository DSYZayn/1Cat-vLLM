# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from types import SimpleNamespace

import pytest
import torch

from benchmarks.kernels.benchmark_sm70_quasar_nvfp4_oracle import (
    _load_column_parallel,
)


@pytest.mark.parametrize("rank", range(4))
def test_gdn_qkv_shards_each_logical_projection(tmp_path, rank):
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "text_config": {
                    "linear_num_key_heads": 4,
                    "linear_key_head_dim": 1,
                    "linear_num_value_heads": 12,
                    "linear_value_head_dim": 1,
                }
            }
        )
    )
    prefix = "model.language_model.layers.0.linear_attn.in_proj_qkv"
    packed = torch.arange(20, dtype=torch.uint8)[:, None].expand(-1, 2)
    scales = torch.arange(20, dtype=torch.float32)[:, None]
    tensors = {
        prefix + ".weight_packed": packed,
        prefix + ".weight_scale": scales,
        prefix + ".weight_global_scale": torch.tensor(2.0),
        prefix + ".input_global_scale": torch.tensor(3.0),
    }
    checkpoint = SimpleNamespace(model=tmp_path, tensor=tensors.__getitem__)
    projection = _load_column_parallel(checkpoint, "gdn_qkv", (prefix,), rank, 4)
    expected_rows = torch.tensor([rank, 4 + rank, *range(8 + 3 * rank, 11 + 3 * rank)])
    assert torch.equal(projection.packed, packed[expected_rows])
    assert torch.equal(projection.scales, scales[expected_rows])
