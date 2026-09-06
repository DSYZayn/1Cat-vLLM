# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import vocab_parallel_embedding as vocab


@pytest.mark.parametrize("tp", [2, 4])
@pytest.mark.parametrize("enabled", [False, True])
def test_fp32_head_admission_without_qpn8_layout(monkeypatch, tp, enabled):
    layer = SimpleNamespace(
        tp_size=tp,
        weight=torch.empty((248320 // tp, 5120), device="meta"),
    )
    monkeypatch.setattr(vocab, "_is_sm70_lm_head_fastpath_eligible", lambda _: True)
    monkeypatch.setattr(vocab, "_sm70_lm_head_packed_layout_requested", lambda: False)
    monkeypatch.setattr(vocab.envs, "VLLM_SM70_DFLASH2_FP32_LOGITS", enabled)
    assert vocab.maybe_prepare_sm70_lm_head_top1(layer)
    assert getattr(layer, "_sm70_dflash2_fp32_logits", False) == enabled
    assert not getattr(layer, "_sm70_dflash2_qpn8_rerank_prepared", False)
