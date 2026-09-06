# Reproduce the Qwen3.8 NVFP4 single-request baseline from source

This is the quality-repaired **no-MTP** lane, not the experimental FlashInfer,
E4M3-KV, batched, or projection/W13 candidates. The production kernels were
integrated by [PR #525](https://github.com/1CatAI/1Cat-vLLM/pull/525), including
the HC and QSA/router commits from #506/#507. A stacked PR remaining open is
not proof that its commits are absent from `main`.

## What this entry point fixes

Previous local launchers selected HC, NVFP4 W13/W2, QSA top-k, FlashQLA and
Flash-V100 binaries from old experiment directories. Source being merged did
not by itself make those launchers reproducible from a clean checkout.

The new builder compiles the production sources in the selected checkout and
records source/binary hashes. Public sidecar bindings replace the formerly
local HC/NVFP4 wrappers; they do not copy or modify kernel algorithms. The HC
communicator lifecycle and every operation using its opaque pointer stay in
the same DSO. The benchmark rejects changed sources, changed libraries, or a
library pointing outside its runtime bundle, and verifies worker mappings.

This is a **source-overlay** reproduction route. A compatible native SM70
vLLM installation is still required for the remaining standard operators;
this command is not a full wheel rebuild. The worker report records those
native dependency paths and hashes instead of concealing that dependency.
An ordinary editable/source build with native extensions in `vllm/` can omit
`--native-extension-dir`. The optional public bootstrap only attaches the
specified native extension directory; it does not patch model execution.

## Fixed contract

| Item | Value |
|---|---|
| Model | RadixArk/Qwen3.8-Flash-Next-NVFP4 |
| Hardware | Four peer-connected V100-SXM2-32GB; TP4/PP1, one request |
| Precision | FP16 activations/KV, checkpoint NVFP4 experts via TurboMind W4A16 |
| Recurrent state | `mamba_ssm_cache_dtype=auto`, resolves to native FP32 |
| Runner | V2, dynamic prefill and full static decode CUDA Graph |
| Context/chunk | 262144 total tokens; prefill chunk8192 |
| PLE | Disk mmap prefill, rank-local pinned-UVA decode |
| MTP/prefix cache | Both off |
| Short speed workload | Fixed8192-token prompt,513 generated tokens,512 decode intervals |
| Sampling | Greedy/ignore-EOS for timing only; official sampling/natural EOS for health |
| Excluded routes | Online QPN8, approximate LM-head, experimental batch sidecars |

The validated environment uses Torch2.10.0+cu128, CUDA compiler12.0.140,
Triton3.6 and NVIDIA driver580.173.02. The compiler/runtime minor versions are
different in this recorded environment; no toolkit or dependency upgrade is
implied by the reproduction recipe. Preserve this environment when comparing
against the recorded baseline. Compiler flags retain the existing production
build recipe; this change introduces no lower-precision kernel route.

## Build and run

Run from the source checkout, with the project Python environment. Replace
the example paths. The build needs no GPUs and does not install into or change
the shared Python environment.

```bash
CUDA_HOME=/usr TORCH_CUDA_ARCH_LIST=7.0 MAX_JOBS=2 \
  .venv/bin/python benchmarks/kernels/build_sm70_qwen38_runtime.py \
  --output-dir /path/to/task/runtime
```

Reserve four idle GPUs before running; do not preempt other jobs. The tested
hybrid PLE setup needs at least90GiB of available host RAM at admission. This
is not the model's total host-memory footprint. Keep the output/cache directory
private to this run.

```bash
CUDA_HOME=/usr CUDA_VISIBLE_DEVICES=0,1,2,3 \
  .venv/bin/python benchmarks/benchmark_sm70_qwen38_baseline.py \
  --model /path/to/Qwen3.8-Flash-Next-NVFP4 \
  --runtime-dir /path/to/task/runtime \
  --native-extension-dir /path/to/compatible/site-packages/vllm \
  --output /path/to/task/results/result.json \
  --repeats 3 --long-context
```

Without `--long-context`, only the short health and8192-token baseline run.
With it, the same instance also measures261631+513, checks retrieval of a
middle-position record with natural EOS, and exercises262143+1 at the exact
context boundary. This is a focused regression, not broad quality certification.
All workers shut down in `finally`; no API is kept resident. A caller may apply
an outer timeout, for example `timeout --kill-after=35s 1800s ...`.

`--reference-json /path/to/previous/result.json` accepts the retained length
sweep's `cases[].runs[]` format and requires identical complete output tokens
for every requested fixed-prompt case (missing cases are rejected). The warmup
and timed repeats must also agree.
The driver does not change sampling to conceal early EOS or numerical drift.

## Accepted historical evidence and fresh-build acceptance

The unprofiled `main` source95205a2d9952 sweep used physical GPUs4-7 with the
fixed contract above. Two warmed repeats per case measured:

| Input | Prefill tok/s | Decode tok/s | TPOT ms |
|---:|---:|---:|---:|
|8192|6936.60|97.826|10.222|
|65536|6245.95|90.729|11.022|
|131072|5830.36|83.814|11.931|
|261631|5137.10|73.977|13.518|

These are the previous frozen-library results, **not yet fresh-builder results**.
The source-only reproduction review must attach fresh build manifests, worker
route evidence, natural quality results, repeatability/reference comparisons,
and unprofiled timings before claiming the new entry point reproduces them.
Do not substitute an Nsight graph interval for endpoint TPOT, or call a
configured256K maximum a tested256K input.

The follow-up length trace attributed98.989% of the increase in graph kernel
service to QSA Top-K and compressed-key scoring. It did not implement a new
long-context optimization. The current Top-K still has its single-CTA long-row
fallback; these reproduction changes do not claim reduced context decay.

## Focused checks

Pure configuration/provenance tests do not require the GPU test conftest:

```bash
CUDA_VISIBLE_DEVICES= .venv/bin/python -m pytest -q \
  --confcutdir=tests/benchmarks tests/benchmarks/test_sm70_qwen38_baseline.py
```

Use the freshly built runtime paths when running existing HC raw-bit,
QSA/router, and page4 relocation tests. Keep numerical admission separate from
speed: a startup or nonempty response is not an output-quality gate.
