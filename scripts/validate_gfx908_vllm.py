#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Compare a small local vLLM model with Transformers on an MI100."""

import argparse
import os
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("GPU_ARCHS", "gfx908")
os.environ.setdefault("AITER_JIT_DIR", str(ROOT / ".jit"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton-cache"))
os.environ.setdefault("VLLM_CACHE_ROOT", str(ROOT / ".vllm-cache"))
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_RMSNORM", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_CUSTOM_AR", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER_LINEAR", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER_MOE", "0")
os.environ.setdefault("VLLM_ROCM_USE_AITER_TRITON_GEMM", "1")
os.environ.setdefault("VLLM_ROCM_USE_AITER_TRITON_ROPE", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import sys

sys.path.insert(0, str(ROOT))

import torch
from transformers import LlamaConfig, LlamaForCausalLM
from vllm import LLM, SamplingParams

import aiter.ops.triton.normalization.rmsnorm as norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graphs", action="store_true")
    parser.add_argument("--unified-attention", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(10)
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
        attn_implementation="eager",
    )
    model = LlamaForCausalLM(config).eval().half()
    prompt = [1, 21, 35, 46]
    with torch.inference_mode():
        expected = model.generate(
            torch.tensor([prompt]),
            max_new_tokens=8,
            do_sample=False,
            eos_token_id=None,
            pad_token_id=0,
        )[0, len(prompt) :].tolist()
    calls = {"rms_norm": 0, "fused_add_rms_norm": 0}
    for name, key in (
        ("rms_norm", "rms_norm"),
        ("rmsnorm2d_fwd_with_add", "fused_add_rms_norm"),
    ):
        original = getattr(norm, name)

        def counted(*a, _key=key, _fn=original, **kw):
            calls[_key] += 1
            return _fn(*a, **kw)

        setattr(norm, name, counted)
    with tempfile.TemporaryDirectory(prefix="aiter-mi100-model-") as directory:
        model.save_pretrained(directory)
        del model
        compilation = {
            "mode": 0,
            "cudagraph_mode": "FULL_DECODE_ONLY",
            "decode_cudagraph_capture_request_sizes": [1, 2],
        }
        backend = "ROCM_AITER_UNIFIED_ATTN" if args.unified_attention else "TRITON_ATTN"
        llm = LLM(
            model=directory,
            skip_tokenizer_init=True,
            dtype="float16",
            max_model_len=128,
            max_num_seqs=2,
            max_num_batched_tokens=128,
            kv_cache_memory_bytes=64 * 1024**2,
            gpu_memory_utilization=0.025,
            enforce_eager=not args.graphs,
            compilation_config=compilation,
            attention_config={"backend": backend},
            kernel_config={
                "ir_op_priority": {
                    "rms_norm": ["aiter"],
                    "fused_add_rms_norm": ["aiter"],
                }
            },
            disable_log_stats=True,
            seed=10,
        )
        params = SamplingParams(
            temperature=0, max_tokens=8, ignore_eos=True, detokenize=False
        )
        for _ in range(3):
            outputs = llm.generate(
                [{"prompt_token_ids": prompt}], params, use_tqdm=False
            )
            actual = list(outputs[0].outputs[0].token_ids)
            assert actual == expected, (actual, expected)
        assert all(calls.values()), calls
        print(
            {
                "result": "PASS",
                "backend": backend,
                "graphs": args.graphs,
                "tokens": actual,
                "aiter_calls": calls,
            },
            flush=True,
        )
        llm.llm_engine.engine_core.shutdown()


if __name__ == "__main__":
    main()
