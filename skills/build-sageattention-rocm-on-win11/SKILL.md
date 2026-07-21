---
name: build-sageattention-rocm-on-win11
description: Complete guide to building SageAttention 2.2 on Windows for AMD RDNA4 (gfx1201) with ROCm 7.2 (TheRock torch), including all known pitfalls and fixes
type: prompt
whenToUse: When the user wants to build or install SageAttention on Windows 11 (especially on AMD ROCm / RDNA4 / gfx1201), or troubleshoot its build errors (hipify, rocWMMA, cmath conflicts, etc.)
---

# Building SageAttention 2.2 on Windows (ROCm 7.2 / RDNA4 gfx1201)

Verified end-to-end on this exact stack; the resulting kernels match the accuracy
numbers reported by the PR #368 author (rel RMSE ≈ 0.037, cosine ≈ 0.9993 vs SDPA):

- GPU: AMD Radeon AI PRO R9700 (`gfx1201`; other RDNA4 cards should work the same)
- OS: Windows 11
- Python **3.12** (the ROCm Windows torch wheels are cp312-only)
- Package manager: **uv** (recommended; pip works too but see pitfall #4)
- torch **2.9.1+rocmsdk20260116** (ROCm 7.2, TheRock channel)
- Visual Studio with MSVC C++ build tools (verified: VS 2026 Community, toolset 14.51)
- SageAttention 2.2.0 source containing the gfx12 native backend (PR #368);
  **no source code changes are required**

## Step 1 — Create a clean environment

Create the venv **outside the SageAttention source tree** (critical, see pitfall #3),
e.g. as a sibling directory:

```powershell
cd C:\path\to\workspace
uv venv --python 3.12 sage-venv
```

Install the ROCm torch stack from AMD's TheRock release repo (flat file index, so use
`--find-links`; all rocm packages carry dev version numbers, so `--prerelease=allow`
is required; `UV_LINK_MODE=copy` avoids hardlink-related pitfalls, see pitfall #4):

```powershell
$env:UV_LINK_MODE = "copy"
uv pip install --python sage-venv\Scripts\python.exe --prerelease=allow `
  --find-links "https://repo.radeon.com/rocm/windows/rocm-rel-7.2/" `
  "torch==2.9.1+rocmsdk20260116" "torchaudio==2.9.1+rocmsdk20260116" `
  "torchvision==0.24.1+rocmsdk20260116" `
  rocm rocm-sdk-core rocm-sdk-devel rocm-sdk-libraries-custom
```

If you pin packages by direct URL instead of `--find-links`, always use the full
`https://repo.radeon.com/...` wheel URLs — resolving bare names can pick up a
same-named empty shell package from PyPI (pitfall #5).

Expand the ROCm SDK devel payload (the wheel ships a tar that must be unpacked once):

```powershell
sage-venv\Scripts\python.exe -m rocm_sdk init
```

Sanity check:

```powershell
sage-venv\Scripts\python.exe -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_properties(0).gcnArchName)"
# expect: 2.9.1+rocmsdk20260116 True gfx1201
```

## Step 2 — Patch the clang HIP wrapper header (MSVC >= 14.40 STL conflict)

Edit `<venv>\Lib\site-packages\_rocm_sdk_devel\lib\llvm\lib\clang\22\include\__clang_hip_runtime_wrapper.h`:
inside the `#if !defined(__HIPCC_RTC__)` block, insert this line **before** `#include <cmath>`:

```c
#include <__clang_cuda_math_forward_declares.h>
```

This mirrors the upstream fix llvm/llvm-project#201563 (see pitfall #2). The patch
must be re-applied whenever rocm-sdk-devel is reinstalled or re-expanded.

## Step 3 — Fetch rocWMMA headers

The TheRock Windows SDK does not ship rocWMMA, but the gfx12 kernels
`#include <rocwmma/rocwmma.hpp>`. rocWMMA is header-only; use the git tag matching
your ROCm version (here `rocm-7.2.0`):

```bash
curl -sL https://github.com/ROCm/rocWMMA/archive/refs/tags/rocm-7.2.0.tar.gz -o /tmp/rw.tar.gz
tar -xzf /tmp/rw.tar.gz -C /tmp "rocWMMA-rocm-7.2.0/library/include/rocwmma"
cp -r /tmp/rocWMMA-rocm-7.2.0/library/include/rocwmma \
    <venv>/Lib/site-packages/_rocm_sdk_devel/include/
```

## Step 4 — Build from a copy of the source tree (never the original root)

The venv must NOT be inside the directory being built (pitfall #3). Copy the source
(excluding `.git`, `build`, any venvs) into a clean directory and build from there:

```bash
mkdir -p _build_tree
tar --exclude=.git --exclude=.venv --exclude=build --exclude=_build_tree \
    --exclude=sageattention.egg-info -cf - . | tar -xf - -C _build_tree
```

Then run the build from an initialized Visual Studio shell. Reference script:
`${KIMI_SKILL_DIR}/scripts/build_rocm.bat`. Its essential settings:

- `call vcvars64.bat` first (VS environment is required even though clang is the compiler)
- `set ROCM_HOME=<venv>\Lib\site-packages\_rocm_sdk_devel` — **required**, because the
  `rocm-sdk.exe` entry point is broken in uv venvs (pitfall #1)
- `set PYTORCH_ROCM_ARCH=gfx1201`
- `set HIPCC_APPEND_FLAGS=-Wno-invalid-specialization` (pitfall #6; setup.py appends
  this env var to the HIP flags, no source edits needed)
- `uv pip install --python <venv>\Scripts\python.exe --no-build-isolation -v _build_tree`

A full build takes roughly 15–25 minutes.

## Step 5 — Verify the result (do not trust the exit code alone)

Judge success by the artifacts, not the process exit code (pitfall #7):

- `<build_tree>\build\lib.win-amd64-cpython-312\sageattention\` must contain both
  `_fused.cp312-win_amd64.pyd` and `_qattn_gfx12_native.cp312-win_amd64.pyd`
- `uv pip list | grep sageattention` shows the package installed
- Run the smoke test `${KIMI_SKILL_DIR}/scripts/smoke_test_gfx12.py` with the target
  venv's python. It loads the extensions directly (works without triton, pitfall #8)
  and compares the fp8 kernels against PyTorch SDPA

Expected smoke-test baseline (fp8 NHD short-MHA vs fp16 SDPA):
rel RMSE ≈ 0.034–0.038, cosine ≈ 0.9993 across shapes
(B2/S1024/H8/D128 causal & non-causal, B1/S512/H4/D64, B2/S2048/H8/D128).

The wheel produced during the build can be recovered from uv's cache if needed:
`find ~/AppData/Local/uv/cache -name "sageattention*.whl"`.

## Pitfall reference (symptom → root cause → fix)

### 1. `rocm-sdk.exe` fails with "uv trampoline failed to canonicalize script path"
The uv-installed entry-point trampoline is broken in this setup, so `setup.py`'s
`rocm-sdk path --root` probe silently returns nothing. Fix: set `ROCM_HOME`
explicitly before building; use `python -m rocm_sdk ...` instead of the `.exe`.

### 2. `__device__ function 'isgreater' cannot overload __host__ __device__ function`
MSVC STL >= 14.40 (VS 2022 17.10+, VS 2026) declares `isgreater`/`isless`/etc. as
`constexpr` under `#ifdef __clang__`; in HIP mode `constexpr` implies
`__host__ __device__`, colliding with clang's `__device__`-only forward declares in
`__clang_cuda_math_forward_declares.h`. Fixed upstream by llvm-project#201563, but
ROCm 7.2's bundled clang 22 predates it. Fix: the header patch in Step 2 (device
forward declares must come before `<cmath>`). Only TUs that pull `<cmath>` on the
device side trigger it, which makes it look intermittent.

### 3. `redefinition of 'Device'` (c10/core/Device_hip.h vs Device.h)
When the venv sits inside the build directory, torch's `BuildExtension` hipify pass
scans `includes=[<build_dir>/*]` — fnmatch `*` crosses directories, and
`matched_files_iter` excludes only `.git`/`build`/`third_party`, not `.venv` — so
torch's own headers inside the venv get rewritten **in place, but only partially**
(e.g. `HIPHooksInterface.h` ends up including `Allocator_hip.h` while `TensorBody.h`
still includes the original `Device.h`), producing duplicate definitions in one TU.
Fix: build from a directory that contains no venv (Step 4). Health checks:
`find <venv> -name "*_hip.h"` must be 0, and line 4 of
`torch/include/ATen/detail/HIPHooksInterface.h` must be `#include <c10/core/Allocator.h>`.

### 4. Pitfall 3 persists after reinstalling torch / `'c10/core/Allocator_hip.h' file not found`
uv installs via hardlinks by default: when hipify rewrites a venv header in place,
the write goes through the hardlink into uv's cache. Every later "reinstall" then
restores the poisoned header from cache, while the generated `*_hip.h` files (not in
the wheel RECORD) are lost. Fix: `uv cache clean torch`, reinstall with
`UV_LINK_MODE=copy`, verify with the checks in pitfall #3. Prevention: never run a
build from a directory containing the venv.

### 5. `_rocm_sdk_devel/` directory missing after reinstalling rocm-sdk-devel
The wheel is a thin wrapper; the actual SDK (10+ GB) lives in an embedded
`_devel.tar` that is expanded lazily. Deleting the expanded directory and
reinstalling only restores the wrapper. Fix: install from the direct wheel URL with
`--prerelease=allow`, then run `python -m rocm_sdk init`. Also beware: an
unversioned reinstall can resolve to a same-named empty package on PyPI instead of
the AMD repo.

### 6. rocWMMA errors (two kinds)
- `'rocwmma/rocwmma.hpp' file not found` → fetch the headers (Step 3); use the git
  tag matching the ROCm version.
- `'is_arithmetic' cannot be specialized: ... forbidden by N5014` → rocWMMA
  specializes std type traits, which the VS 2026 STL rejects. Fix:
  `HIPCC_APPEND_FLAGS=-Wno-invalid-specialization`.

### 7. Build "fails" but artifacts were produced (false negative)
If the build log is piped through `iconv -f GBK`, invalid byte sequences make iconv
exit 1 and mask a successful build. Judge by artifacts (Step 5), not the exit code.

### 8. `import sageattention` fails with `No module named 'triton'`
`core.py` imports its triton backends at module level; there is no official Windows
ROCm triton build (AMD's nightly index has Linux wheels only). The gfx12 native path
itself does not need triton. Workarounds: load the extension `.pyd` files directly
by path (see the smoke-test script), or install a Windows-ROCm-compatible triton.

## Wheel output constraints

The resulting `sageattention-2.2.0-cp312-cp312-win_amd64.whl` installs only on:
Python 3.12 + Windows x64 + the same ROCm 7.2 torch stack + gfx1201 GPUs
(the wheel contains gfx1201 ISA only).
