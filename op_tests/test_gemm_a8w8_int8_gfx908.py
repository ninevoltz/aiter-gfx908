# SPDX-License-Identifier: MIT
"""Validate the MI100 CK INT8 scale and output contracts."""

import pytest
import torch

from aiter import gemm_a8w8_CK
from aiter.ops.triton.utils._triton.arch_info import get_arch

pytestmark = pytest.mark.skipif(get_arch() != "gfx908", reason="MI100-specific build")


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "shape", [(1, 128, 128), (17, 256, 512), (64, 1024, 1024), (512, 128, 256)]
)
@pytest.mark.parametrize("has_bias", [False, True])
@pytest.mark.parametrize("scale_type", ["fp32", "output"])
def test_ck_int8_per_row_scales(dtype, shape, has_bias, scale_type):
    m, n, k = shape
    torch.manual_seed(0)
    x = torch.randint(-16, 16, (m, k), dtype=torch.int8, device="cuda")
    w = torch.randint(-16, 16, (n, k), dtype=torch.int8, device="cuda")
    scale_dtype = torch.float32 if scale_type == "fp32" else dtype
    xs = torch.rand(m, 1, dtype=scale_dtype, device="cuda") * 0.01
    ws = torch.rand(n, 1, dtype=scale_dtype, device="cuda") * 0.01
    bias = torch.randn(n, dtype=dtype, device="cuda") if has_bias else None
    expected = (x.float() @ w.float().T) * xs.float() * ws.float().T
    if bias is not None:
        expected += bias
    actual = gemm_a8w8_CK(x, w, xs, ws, bias, dtype)
    torch.testing.assert_close(actual, expected.to(dtype), atol=0.005, rtol=0.01)


def test_ck_int8_rejects_block_scales():
    x = torch.ones(17, 256, dtype=torch.int8, device="cuda")
    w = torch.ones(128, 256, dtype=torch.int8, device="cuda")
    xs = torch.ones(17, 2, dtype=torch.float32, device="cuda")
    ws = torch.ones(128, 2, dtype=torch.float32, device="cuda")
    with pytest.raises(ValueError, match="block scales are not supported"):
        gemm_a8w8_CK(x, w, xs, ws)


def test_ck_int8_rejects_incompatible_scale_dtype():
    x = torch.ones(1, 128, dtype=torch.int8, device="cuda")
    w = torch.ones(128, 128, dtype=torch.int8, device="cuda")
    xs = torch.ones(1, 1, dtype=torch.float16, device="cuda")
    ws = torch.ones(128, 1, dtype=torch.float16, device="cuda")
    with pytest.raises(ValueError, match="either FP32 or the output dtype"):
        gemm_a8w8_CK(x, w, xs, ws, dtype=torch.bfloat16)
