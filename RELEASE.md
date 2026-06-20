# 1Cat-vLLM 1.2.0

1Cat-vLLM 1.2.0 is built on upstream vLLM
`0.21.1rc1.dev438+g4ff865c38.d20260603.cu128`, with focused upgrades for
Tesla V100 / SM70 serving, FlashAttention, Marlin/MoE, long-context execution,
and memory policy.

1. Updated vLLM base and runtime architecture
   SM70/V100 optimizations are re-integrated with the updated backend,
   quantization, attention, CUDA graph, serving, OpenAI API, Qwen hybrid model,
   and MTP paths.

2. Improved Flash-V100 attention backend
   `FLASH_ATTN_V100` now covers prefill, decode, paged KV, CUDA graph, and
   long-context decode on V100/SM70, providing a modern FlashAttention path for
   Tesla V100.

3. Enabled V100 Marlin path
   In 0.0.3, Marlin required SM75+ and could not run on V100/SM70. 1.2.0 restores
   the SM70 Marlin path and allows Marlin to reuse TurboMind MoE capability.

4. Strengthened 35B MoE serving
   Qwen3.6-35B-A3B-AWQ is now usable on V100. Marlin TP2 decode has been measured
   at about `98-109 tok/s`.

5. Stabilized long-context decode
   Qwen3.6-27B-AWQ TP2 no-MTP decode remains nearly flat across 16K/32K/64K,
   around `58.46-58.48 tok/s`, shifting the main long-context bottleneck toward
   prefill/TTFT.

6. Optimized long-context dense prefill
   D=256 WMMA-QK is enabled by default. On 27B-AWQ TP2 with
   `max_num_batched_tokens=16384`:
   - 16K: `12.57s -> 10.33s`, about `+21.69%`
   - 64K chunked: `75.83s -> 74.18s`, about `+2.23%`

7. Optimized paged-prefix exact path
   D=256 paged-prefix low-smem reduces shared memory from about `96.5KB/CTA` to
   about `36KB/CTA`; 256K chunked prefill kernel-only improves from `59627ms` to
   `52882ms`, about `+12.8%`.

8. Reduced long-context scheduling and memory-access overhead
   Added page-id/page-offset cache to reduce repeated block-table loads and
   address calculations in the paged-prefix hot loop, improving chunked prefill
   efficiency.

9. Improved V100 long-context memory policy
   Public profiles target 256K context with more conservative
   `max_num_batched_tokens`, `max_num_seqs`, and KV cache allocation settings,
   reducing memory pressure for long-context V100 serving.

10. Added experimental BFLA sparse prefill
    `VLLM_FLASH_V100_BFLA_PREFILL` is added as a default-off experimental path
    for approximate sparse attention and ultra-long-context prefill acceleration:
    - 9B-AWQ TP1 256K: `499.10s -> 79.08s`, about `6.31x`
    - 27B-AWQ 128K: `244.01s -> 83.02s`, about `2.94x`
    - Current recommended experimental operating point: `MIN_KV=32768`

11. Consolidated SM70 route compatibility
    AWQ, FP8, MTP, CUDA graph, compact allreduce, and Qwen GDN paths have been
    tightened for V100 multi-quantization, graph compilation, and long-context
    serving.

# Releasing vLLM

vLLM releases offer a reliable version of the code base, packaged into a binary format that can be conveniently accessed via [PyPI](https://pypi.org/project/vllm). These releases also serve as key milestones for the development team to communicate with the community about newly available features, improvements, and upcoming changes that could affect users, including potential breaking changes.

## Release Cadence and Versioning

We aim to have a regular release every 2 weeks. Since v0.12.0, regular releases increment the minor version rather than patch version. The list of past releases can be found [here](https://vllm.ai/releases).

Our version numbers are expressed in the form `vX.Y.Z`, where `X` is the major version, `Y` is the minor version, and `Z` is the patch version. They are incremented according to the following rules:

* _Major_ releases are reserved for architectural milestones involving sweeping API changes, similar to PyTorch 2.0.
* _Minor_ releases correspond to regular releases, which include new features, bug fixes and other backwards-compatible changes.
* _Patch_ releases correspond to special releases for new models, as well as emergency patches for critical performance, functionality and security issues.

This versioning scheme is similar to [SemVer](https://semver.org/) for compatibility purposes, except that backwards compatibility is only guaranteed for a limited number of minor releases (see our [deprecation policy](https://docs.vllm.ai/en/latest/contributing/deprecation_policy) for details).

## Release Branch

Each release is built from a dedicated release branch.

* For _major_ and _minor_ releases, the release branch cut is performed 1-2 days before release is live.
* For _patch_ releases, previously cut release branch is reused.
* Release builds are triggered via push to RC tag like `vX.Y.Z-rc1`. This enables us to build and test multiple RCs for each release.
* Final tag: `vX.Y.Z` does not trigger the build but used for Release notes and assets.
* After branch cut is created, we monitor the main branch for any reverts and apply these reverts to a release branch.

### Cherry-Pick Criteria

After branch cut, we approach finalizing the release branch with clear criteria on what cherry picks are allowed in. Note: a cherry pick is a process to land a PR in the release branch after branch cut. These are typically limited to ensure that the team has sufficient time to complete a thorough round of testing on a stable code base.

* Regression fixes - that address functional/performance regression against the most recent release (e.g. 0.7.0 for 0.7.1 release)
* Critical fixes - critical fixes for severe issue such as silent incorrectness, backwards compatibility, crashes, deadlocks, (large) memory leaks
* Fixes to new features introduced in the most recent release (e.g. 0.7.0 for 0.7.1 release)
* Documentation improvements
* Release branch specific changes (e.g. change version identifiers or CI fixes)

Please note: **No feature work allowed for cherry picks**. All PRs that are considered for cherry-picks need to be merged on trunk, the only exception are Release branch specific changes.

## Manual validations

### E2E Performance Validation

Before each release, we perform end-to-end performance validation to ensure no regressions are introduced. This validation uses the [vllm-benchmark workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) on PyTorch CI.

**Current Coverage:**

* Models: Llama3, Llama4, and Mixtral
* Hardware: NVIDIA H100 and AMD MI300x
* _Note: Coverage may change based on new model releases and hardware availability_

**Performance Validation Process:**

**Step 1: Get Access**
Request write access to the [pytorch/pytorch-integration-testing](https://github.com/pytorch/pytorch-integration-testing) repository to run the benchmark workflow.

**Step 2: Review Benchmark Setup**
Familiarize yourself with the benchmark configurations:

* [CUDA setup](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/cuda)
* [ROCm setup](https://github.com/pytorch/pytorch-integration-testing/tree/main/vllm-benchmarks/benchmarks/rocm)

**Step 3: Run the Benchmark**
Navigate to the [vllm-benchmark workflow](https://github.com/pytorch/pytorch-integration-testing/actions/workflows/vllm-benchmark.yml) and configure:

* **vLLM branch**: Set to the release branch (e.g., `releases/v0.9.2`)
* **vLLM commit**: Set to the RC commit hash

**Step 4: Review Results**
Once the workflow completes, benchmark results will be available on the [vLLM benchmark dashboard](https://hud.pytorch.org/benchmark/llms?repoName=vllm-project%2Fvllm) under the corresponding branch and commit.

**Step 5: Performance Comparison**
Compare the current results against the previous release to verify no performance regressions have occurred. Here is an
example of [v0.9.1 vs v0.9.2](https://hud.pytorch.org/benchmark/llms?startTime=Thu%2C%2017%20Apr%202025%2021%3A43%3A50%20GMT&stopTime=Wed%2C%2016%20Jul%202025%2021%3A43%3A50%20GMT&granularity=week&lBranch=releases/v0.9.1&lCommit=b6553be1bc75f046b00046a4ad7576364d03c835&rBranch=releases/v0.9.2&rCommit=a5dd03c1ebc5e4f56f3c9d3dc0436e9c582c978f&repoName=vllm-project%2Fvllm&benchmarkName=&modelName=All%20Models&backendName=All%20Backends&modeName=All%20Modes&dtypeName=All%20DType&deviceName=All%20Devices&archName=All%20Platforms).
