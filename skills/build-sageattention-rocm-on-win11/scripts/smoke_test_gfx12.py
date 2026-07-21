"""Smoke test for the gfx12 native SageAttention extension (RDNA4 / gfx1201).

Loads the compiled extension directly by path so it works without the
`triton` package (which sageattention/core.py imports at module level).
Compares the fp8 NHD short-MHA kernel against PyTorch SDPA (fp16).
"""
import importlib.util
import math
import os
import sys
import sysconfig

import torch
import torch.nn.functional as F

# Locate site-packages of the interpreter running this script, so it works
# regardless of where the script is placed.
SITE_PKG = sysconfig.get_path("purelib")


def load_ext(name):
    path = os.path.join(SITE_PKG, "sageattention", f"{name}.cp312-win_amd64.pyd")
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    assert torch.cuda.is_available(), "ROCm GPU not available"
    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} ({getattr(props, 'gcnArchName', '?')})")

    fused = load_ext("_fused")
    gfx12 = load_ext("_qattn_gfx12_native")
    print("extensions loaded OK")

    torch.manual_seed(0)
    ok = True
    for (b, s, h, d, causal) in [
        (2, 1024, 8, 128, False),
        (2, 1024, 8, 128, True),
        (1, 512, 4, 64, False),
        (2, 2048, 8, 128, False),
    ]:
        q = torch.randn(b, s, h, d, device=dev, dtype=torch.float16).contiguous()
        k = torch.randn(b, s, h, d, device=dev, dtype=torch.float16).contiguous()
        v = torch.randn(b, s, h, d, device=dev, dtype=torch.float16).contiguous()
        sm_scale = 1.0 / math.sqrt(d)

        out = gfx12.sage_fp8_nhd_short_mha(q, k, v, int(causal), sm_scale, 2.25)
        ref = F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
            is_causal=causal, scale=sm_scale,
        ).transpose(1, 2)

        diff = (out.float() - ref.float())
        rel_rmse = diff.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt()
        cos = F.cosine_similarity(out.float().flatten(), ref.float().flatten(), dim=0)
        status = "OK" if rel_rmse < 0.1 and cos > 0.995 else "FAIL"
        ok &= status == "OK"
        print(f"B{b} S{s} H{h} D{d} causal={causal}: rel_rmse={rel_rmse:.4f} cos={cos:.5f} [{status}]")

    print("SMOKE TEST", "PASSED" if ok else "FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
