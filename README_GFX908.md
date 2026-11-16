# MI100 / gfx908 local port

This branch targets personal vLLM use on MI100. It starts from ROCm/aiter
`7c2fa4dae5084d0ab8dcc17064afc8f27604c492` (2026-11-16). It uses that
commit's Composable Kernel revision, `f93c029e09f67c983696eecac8e0e0350f3dbb54`.
The original source checkouts were left unchanged. The running vLLM service
now uses this port after the full-model serving validation below.

## Run it

Use the existing ROCm Python environment. No installation into that environment
is needed: the launcher puts this checkout first on `PYTHONPATH`. It keeps JIT,
Triton, and vLLM caches in this worktree. The existing ROCr runtime fix is loaded
when it is present in the sibling `vllm-mi100_support/rocr-fixed/` directory.
Set `MI100_ROCR_RUNTIME` to use another runtime file.

```bash
cd /home/john/git-projects/aiter-mi100
./scripts/run_gfx908.sh /home/john/git-projects/vllm-mi100_support/.venv/bin/python scripts/validate_gfx908.py
./scripts/run_gfx908.sh /home/john/git-projects/vllm-mi100_support/.venv/bin/python scripts/validate_gfx908_vllm.py --graphs
```

For normal vLLM commands, put `scripts/run_gfx908.sh` before the vLLM executable.
Use `--attention-backend TRITON_ATTN` for the baseline profile. The launcher
turns on AITER, Triton RoPE, Triton unquantized GEMM, and RMSNorm. The local
vLLM build routes gfx908 RMSNorm to the Triton implementation. Other vLLM
builds need equivalent gfx908 dispatch; enabling native CK norms is not covered.

The launcher leaves AITER quantized linear, MoE, native MHA/MLA, custom
all-reduce, and automatic unified-attention selection off. Environment variables
can override these defaults. This port does not establish support for those
other backends.

A separate launcher for the user's existing Qwen setup is at
`/home/john/git-projects/vllm-mi100_support/serve-qwen-aiter.sh`. It preserves the
current model and serving options and selects this checkout. It uses the same port and GPU allocation; do not run a second copy beside the
four-GPU service. `/home/john/start_vllm.sh` now delegates to this launcher, so
`vllm-server.service` keeps the tested profile across restarts. The original
startup script is backed up at
`/home/john/git-projects/vllm-mi100_support/start_vllm.before-aiter-20261116.sh`.

## What changed

- Add gfx908 build-target validation, MI100 naming, 120-CU metadata, 64 KiB LDS
  metadata, and legacy tuning-CSV architecture resolution.
- Migrate the gfx908 Triton configs from `aiter-gfx908` ref
  `origin/mi100-a8w8-2026-08` (`6200ca919`) to the current config layout. Remove
  old `kpack` entries. Retain historical shape configs as ported data, not as
  newly measured tuning results.
- Add conservative PA decode and unified-attention configs. Use one warp for
  split-attention reduction: the four-warp configuration aborted in this
  environment's Triton compiler on a tested long-context case.
- Accept vLLM's `gemma_norm` argument in the Triton fused RMSNorm wrapper.
  The default False path is unchanged. True computes FP32 `1 + weight` before
  the existing norm implementation.
- Port CK INT8 instance/dtype pruning and dispatch guards from the August fork.
  A gfx908-only build omits FP8 instances and retains FP16/BF16 output variants.
  Optional instance and dtype filters remain available through
  `AITER_CK_INSTANCE_LIST` and `AITER_CK_DTYPES`.
- Reject unsupported block-scale layouts and incompatible scale dtypes at the
  gfx908 CK INT8 API boundary. This kernel expects one scale per activation row
  and one per weight output channel. Bias must match the output dtype.
- Extend existing tests and add bounded validation scripts. The scripts call
  existing public wrappers and test suites; they do not add alternate kernels.

## Validation on 2026-11-16

Environment: four AMD Instinct MI100 GPUs. Initial operator tests used GPU 0
while the old server remained running; the subsequent full-model validation
used all four GPUs after the user authorized a restart. PyTorch `2.14.0+rocm7.14`, HIP `7.14.60850`, Triton
`3.8.0`, vLLM `0.33.0rc1.dev1351+g53c9679c4.fp16.rocm714`.

- Bounded Triton suite: **358 passed, 32 skipped**. Covers RMSNorm, fused add
  RMSNorm, Gemma scaling, RoPE, FP16/BF16 GEMM, fused activation/layout cases,
  INT8 Triton GEMM, paged attention, and unified attention. Decode cases include
  8,704-token contexts, uneven query/context lengths, shuffled caches, and
  sliding windows. Unsupported backends and large test allocations are excluded.
- Large-row-count RMSNorm forward/backward: **9 passed** across FP32, FP16,
  and BF16, including 32,768 rows and irregular widths. A separate gfx908 config
  covers the per-head norm path used during long-prompt prefill.
- Build-target/JIT guard tests plus CK INT8 tests: **70 passed**, plus **51
  subtests**. CK coverage includes FP16/BF16 outputs, bias, and scale-layout
  rejection. The CK extension built successfully against the current CK pin.
- GEMM code-generation/dispatch harness: **74 checks passed**. The forwarding
  test mocks the CK capability gate so CPU placeholders do not enter a GPU
  fallback on gfx908.
- Tiny two-layer Llama: vLLM matched Transformers' eight greedy output tokens on
  three requests in each of three runs: Triton attention eager, Triton attention
  with decode graphs, and AITER unified attention with decode graphs. Instrumented
  call counts confirm that AITER RMSNorm and fused add RMSNorm actually ran.
- Decode graph capture completed for request sizes 1 and 2.

The tiny Llama model is generated locally. It downloads no model or tokenizer.
Its token agreement checks integration; it is not a language-quality evaluation
or a benchmark of the user's full Qwen model.

Operator logs are under `/home/john/git-projects/vllm-mi100_support/aiter-port-*.log`.
Full-model results, outputs, timing, and launcher state are in
`/home/john/git-projects/vllm-mi100_support/aiter-serving-*.json` and `.log`.

## Limits and next work

The full Qwen model was restarted with this port on all four GPUs, with MTP
and decode graphs enabled. A 252-request run passed: seven baseline tasks,
220 arithmetic checks, 22 long-prompt retrieval checks, and three 192-token
streaming benchmarks. The seven baseline answers matched the AITER-disabled
run, including JSON, Python code, tool calls, and red-image recognition. The
service stayed healthy with no restarts. Long retrieval prompts were about
9,261 tokens; the 262,144-token configured limit was not stress-tested.

Decode benchmark rates were 74.71, 74.35, and 63.27 tokens/s without AITER,
and 72.85, 72.84, and 65.15 tokens/s with AITER. These few samples show no
clear speedup. Free-form outputs were coherent but were not byte-identical,
so this is a practical serving comparison, not a fixed-token microbenchmark.
No end-to-end speedup is claimed. The historical tuning files were not retuned.
Only the listed config families and cases have test evidence; presence of a
migrated JSON file does not establish support for every associated operator.

Native FP8/FP4 and FlyDSL are not enabled for gfx908. FlyDSL is absent from the
validated Python environment. The old custom-all-reduce patches and unvalidated
September barrier experiment were not carried over. Keep the current vLLM
collective path for multi-GPU use.

To exercise the experimental attention backend explicitly:

```bash
./scripts/run_gfx908.sh /home/john/git-projects/vllm-mi100_support/.venv/bin/python scripts/validate_gfx908_vllm.py --graphs --unified-attention
```

To rerun native INT8 and metadata tests:

```bash
./scripts/run_gfx908.sh /home/john/git-projects/vllm-mi100_support/.venv/bin/python -m pytest -q op_tests/test_gemm_a8w8_int8_gfx908.py op_tests/test_build_targets_architecture.py op_tests/test_jit_arch_guard.py
```
