# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers import vocab_parallel_embedding as vocab


@pytest.mark.parametrize("enabled", [False, True])
def test_explicit_fp32_flag_independent_of_other_head_fastpaths(monkeypatch, enabled):
    layer = SimpleNamespace(
        prefix="language_model.lm_head",
        weight=SimpleNamespace(
            dtype=torch.float16,
            is_cuda=True,
            device=torch.device("cuda", 0),
            ndim=2,
            shape=(124160, 5120),
        ),
    )
    monkeypatch.setattr(vocab, "_sm70_env_bool", lambda *args: False)
    monkeypatch.setattr(vocab, "_sm70_dflash2_qpn8_rerank_requested", lambda: False)
    monkeypatch.setattr(vocab.current_platform, "is_cuda_alike", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_capability", lambda _: (7, 0))
    monkeypatch.setattr(vocab.envs, "VLLM_SM70_DFLASH2_FP32_LOGITS", enabled)
    assert vocab._is_sm70_lm_head_fastpath_eligible(layer) == enabled


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
