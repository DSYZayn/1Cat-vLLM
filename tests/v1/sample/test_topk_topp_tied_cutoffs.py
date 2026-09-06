# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
from vllm.v1.sample.ops.topk_topp_triton import apply_top_k_top_p_triton
from vllm.v1.worker.gpu.spec_decode.dflash2.sparse_rejection import (
    _compact_target_requires_reference,
)


@pytest.mark.parametrize("top_p", [1.0, 0.95, 0.6])
@pytest.mark.parametrize("case", ["k_tie", "p_tie", "uniform", "unique"])
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_tied_cutoffs_match_full_vocabulary_reference(case, top_p):
    x = torch.full((8, 32768), -20.0, device="cuda")
    if case == "k_tie":
        x[:, :24] = 1.0
        x[:, :18] = 2.0
    elif case == "p_tie":
        x[:, :20] = 1.0
        x[:, :2] = 2.0
    elif case == "uniform":
        x.fill_(1.0)
    else:
        x[:, :32] = torch.arange(32, 0, -1, device="cuda") / 8
    k = torch.full((8,), 20, dtype=torch.int32, device="cuda")
    p = torch.full((8,), top_p, device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), k, p)
    actual = apply_top_k_top_p_triton(x.clone(), k, p)
    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_standalone_topp_ties_and_graph_capture():
    x = torch.full((2, 32768), -20.0, device="cuda")
    x[:, :20] = 1.0
    x[:, :2] = 2.0
    p = torch.full((2,), 0.95, device="cuda")
    expected = apply_top_k_top_p_pytorch(x.clone(), None, p)
    actual = apply_top_k_top_p_triton(x.clone(), None, p)
    assert torch.equal(actual, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = apply_top_k_top_p_triton(x.clone(), None, p)
    graph.replay()
    assert torch.equal(captured, expected)


def test_compact_guard_only_rejects_ambiguous_cutoffs():
    unique = torch.arange(21, 0, -1, dtype=torch.float32)[None] / 8
    assert not _compact_target_requires_reference(unique, 1.0, 0.95)
    k_tie = unique.clone()
    k_tie[:, -1] = k_tie[:, -2]
    assert _compact_target_requires_reference(k_tie, 1.0, 1.0)
    p_tie = torch.ones(1, 21)
    p_tie[:, :2] = 2.0
    p_tie[:, -1] = -20.0
    assert _compact_target_requires_reference(p_tie, 1.0, 0.95)
    assert not _compact_target_requires_reference(p_tie, 1.0, 1.0)
