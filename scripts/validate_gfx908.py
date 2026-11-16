#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run bounded subsets of the existing tests on an MI100."""

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)
os.environ.setdefault("GPU_ARCHS", "gfx908")
os.environ.setdefault("AITER_JIT_DIR", str(ROOT / ".jit"))
os.environ.setdefault("TRITON_CACHE_DIR", str(ROOT / ".triton-cache"))
os.environ.setdefault("MAX_JOBS", "4")

import pytest
import torch

GROUPS = {
    "norm": [
        "normalization/test_rmsnorm.py::test_rmsnorm",
        "normalization/test_rmsnorm.py::test_fused_add_rmsnorm",
        "normalization/test_rmsnorm.py::test_fused_add_rmsnorm_gemma_norm",
    ],
    "rope": ["rope/test_rope.py::test_rope_sbhd_fwd"],
    "gemm": [
        "gemm/basic/test_gemm_a16w16.py::test_gemm_a16_w16",
        "gemm/basic/test_gemm_a16w16.py::test_gemm_a16_w16_activation",
        "gemm/basic/test_gemm_a16w16.py::test_gemm_a16_w16_layout",
    ],
    "int8": ["gemm/basic/test_gemm_a8w8.py::test_gemm_int8"],
    "attention": [
        "attention/test_pa_decode.py::test_paged_attn",
        "attention/test_pa_decode.py::test_paged_attn_v2",
        "attention/test_unified_attention.py::test_triton_unified_attn",
        "attention/test_unified_attention.py::test_triton_unified_attn_3d",
    ],
}


class BoundedCases:
    def pytest_collection_modifyitems(self, config, items):
        kept, dropped = [], []
        for item in items:
            p = item.callspec.params
            keep = p.get("backend", "triton") == "triton"
            keep &= p.get("impl", "triton") == "triton"
            keep &= p.get("kernel_type", "bandwidth_bound") == "bandwidth_bound"
            for key, limit in (
                ("M", 256),
                ("N", 4096),
                ("K", 8192),
                ("B", 4),
                ("S", 32),
                ("num_blocks", 8192 if item.originalname.endswith("_3d") else 2048),
                ("head_size", 128),
            ):
                keep &= p.get(key, 0) <= limit
            for key in ("m", "n", "k"):
                keep &= p.get(key, 0) <= 2048
            for key in ("q_dtype", "kv_dtype", "dtype", "kv_cache_dtype"):
                keep &= p.get(key, torch.float16) in (torch.float16, torch.bfloat16)
            if "seq_lens" in p:
                keep &= p["num_heads"] == (8, 1)
                keep &= p["block_size"] in (16, 64)
                keep &= p["soft_cap"] is None
                keep &= not p.get("force_fp8_pv", False)
            (kept if keep else dropped).append(item)
        config.hook.pytest_deselected(items=dropped)
        items[:] = kept


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("groups", nargs="*", help="Test groups: " + ", ".join(GROUPS))
    args = parser.parse_args()
    groups = args.groups or list(GROUPS)
    unknown = set(groups) - set(GROUPS)
    if unknown:
        parser.error(f"Unknown test groups: {', '.join(sorted(unknown))}")
    if not torch.cuda.is_available():
        parser.error("A ROCm PyTorch environment and an MI100 are required")
    arch = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
    if arch != "gfx908":
        parser.error(f"Expected gfx908, got {arch}")
    paths = [
        "op_tests/triton_tests/" + path for group in groups for path in GROUPS[group]
    ]
    return pytest.main(
        [*paths, "-q", "--disable-warnings", "--maxfail=3"], plugins=[BoundedCases()]
    )


if __name__ == "__main__":
    sys.exit(main())
