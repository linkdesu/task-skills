---
name: comfyui-minimax-h3-rocm-tuning
description: Complete tuning & troubleshooting guide for ComfyUI + MiniMax-H3 on AMD ROCm (RDNA4 gfx1201 reference): startup flags, multi-GPU text encoder offload, black-screen/NaN debugging, fp8 vs convrot model choice.
type: prompt
whenToUse: When running MiniMax-H3 (text-to-video) in ComfyUI on AMD GPUs (ROCm) and you need to tune startup parameters, split models across multiple GPUs, or debug intermittent black videos / audio NaN issues.
---

# ComfyUI + MiniMax-H3 on AMD ROCm — Tuning & Troubleshooting

End-to-end field notes from running MiniMax-H3 text-to-video (the 机智罗/XB_ToolBox
workflow family) on AMD GPUs with ROCm. Covers model selection, startup flags,
multi-GPU offload, and the full black-screen/NaN debugging journey.

> **Reference hardware**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4) × 2, 32 GB VRAM
> each, ROCm 7.14 (TheRock nightly torch), ComfyUI 0.31.0 in a distrobox container.
> **Numbers below are for this stack only** — RDNA3 cards (7900 XTX etc.) have
> different VRAM and kernel behavior; re-measure on your own hardware. See the
> "Architecture differences" note in each section.

## 0. TL;DR (verified conclusions)

1. **Prefer fp8_scaled models over convrot int4/int8.** Same VRAM footprint, same
   inference speed, but loading drops from 30–40 min (CPU dequant + inverse-Hadamard
   rotation) to 1–2 min (native format). fp8 also sidesteps the numerical instability
   that caused black-screen/NaN on convrot weights. (Convrot's 30–40 min cold start
   *is* fixed by `--enable-triton-backend` — GPU kernels take over the dequant —
   but fp8 still wins on simplicity and stability.)
2. **Startup flags that matter**: `--enable-triton-backend --disable-pinned-memory
   --disable-mmap`. On the reference stack, `--disable-dynamic-vram` made no
   difference, and three "optimization" knobs (below) actively caused NaN.
3. **Multi-GPU**: put the 32B text encoder on the second GPU via
   `CLIPLoaderMultiGPU` + `device="cuda:1"`; UNet stays on GPU0. Verified
   ~20.7 GB GPU0 + ~14.7 GB GPU1 with headroom for 1080p sampling (peak 27.6 GB).
4. **BlockSwap (XB_Sage_BlockSwap) gave no benefit** (measured on convrot int8:
   same 55 s at SW=0 and SW=25; SW=50 blew up system RAM). Keep it off — the
   same conclusion applies to fp8 (no manual swap when the model fits).

## 1. Model selection: convrot vs fp8

MiniMax-H3 is distributed in several quantized variants (check the model cards on
Hugging Face for what's available). They are **not equivalent** — the difference is
in the *loading path*:

| | int8/int4 convrot | fp8 (pruned_fp8_scaled) |
|---|---|---|
| Storage | int + scale, **Hadamard-rotated** | native float8 (e4m3) |
| Load | CPU unpack + dequantize + **inverse rotation** | memory copy |
| Load time (32B TE + 21GB UNet) | **30–40 min** CPU-only; **~85 s** with Triton backend (GPU dequant kernels) | **1–2 min** |
| VRAM | 1 byte/elem (half of bf16) | 1 byte/elem (same) |
| Inference | INT8/W4A4 TensorCore kernels | FP8 kernels |
| Quality | near-bf16 | near-bf16 (slightly better range) |
| Stability on ROCm RDNA4 | ⚠️ intermittent NaN/black (see §6) | ✅ stable |

Why convrot loads so slowly: it is a QuaRot variant — weights are group-wise
Hadamard-rotated before quantization to suppress outliers (that's what makes quality
near-bf16). The file stores the packed+rotated form, so loading must unpack (two
int4 per byte), dequantize (×scale), and **undo the rotation**. Without the Triton
backend this is a single/double-threaded CPU compute pass (~17.5 MB/s measured on
21 GB → 30–40 min for TE+UNet), a known convrot characteristic ("loading can take
forever"), not a bug. **With `--enable-triton-backend`, comfy-kitchen's HIP backend
provides GPU kernels (`dequantize_convrot_w4a4_weight`, `rotate_int8_convrot_weight`)
that take over the conversion — cold load drops to ~85 s, warm runs to 55 s.** fp8
remains the simplest choice (native format, no conversion at all).

> **Model acquisition**: use whatever download path you prefer (Hugging Face CLI,
> browser, your existing model stash). This guide deliberately does not recommend
> specific repos/files — MiniMax-H3 has multiple variants (text/image/first-last-frame
> to video) and the exact files depend on your workflow. Place them in the standard
> ComfyUI model folders (see §3), e.g. `diffusion_models/` for UNets and
> `text_encoders/` for the Qwen3-VL-32B-based text encoder. ComfyUI loads models from
> subdirectories (e.g. `MiniMax-H3/...`) without extra config.
>
> If you do download via HF CLI, run long pulls under `systemd-run --user`
> (transient unit) so the process survives SSH disconnects, and redirect logs
> *inside* the unit command (`bash -c "cmd >> /path/log 2>&1"`) — an outer `> log`
> only captures "Running as unit".

## 2. Environment bring-up: base image, ROCm, Triton & custom nodes

This section documents everything that is **not part of a stock ComfyUI install**
— the base container image, the ROCm/torch stack, Triton, and the custom nodes
added during debugging. If you are building a similar box, follow this order.

### 2.1 Base image: what we started from and what we changed

The container is built from a **self-maintained Dockerfile** (public repo
`github.com/kyuz0/amd-r9700-comfy`, adjusted for your own hardware), **not** from
ComfyUI's official Docker image. Key decisions vs a stock ComfyUI image:

- **Base**: `registry.fedoraproject.org/fedora:rawhide` (Fedora; ComfyUI's official
  images are Debian/Ubuntu-based). Fedora is used because TheRock ROCm wheels and
  the gfx1201 toolchain are validated on it, and distrobox (container host) plays
  well with it.
- **Full compiler toolchain kept in the image** (`gcc gcc-c++ binutils make
  libdrm-devel python3.13-devel`): normally you'd strip these for size, but
  **Triton JIT needs hipcc/clang at runtime**, so they must stay.
- **Python 3.13 venv at `/opt/venv`** with an auto-activate profile script
  (`/etc/profile.d/venv.sh`) — ComfyUI runs from this venv.
- **ROCm + torch installed via pip from AMD/PyTorch nightly indexes** (see 2.2),
  not via `amdgpu-install` — this is the "TheRock" track for gfx1201.
- **`transformers>=4.56.2` unpinned** — pinning an exact version breaks against
  newer torch builds.
- **Helper scripts baked in** (model downloaders, `model_manager.py`,
  `benchmark_workflows.py` under `/opt/`).
- **3 stock custom nodes baked in**: `ComfyUI_essentials`, `ComfyUI-AMDGPUMonitor`,
  `ComfyUI-GGUF` (more nodes were added later, see 2.4).
- A profile script sets `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` at login —
  **note: this exact env var later turned out to be one of the NaN culprits (§6)**;
  you may want to remove it from your image.
- Image size lands around **36 GB** (full ROCm stack + torch + venv) — normal.

Rebuild flow: `podman build -t localhost/amd-r9700-comfy:test -f Dockerfile .`
then `distrobox create --image localhost/amd-r9700-comfy:test --additional-flags
"--device /dev/kfd --device /dev/dri --group-add video --group-add render
--security-opt seccomp=unconfined"`. GPU is verified with a `verify-rocm.sh`
script (expect `ROCm available: True`, GPU count, arch list containing gfx1201).

### 2.2 ROCm & torch: exact versions and install procedure

Reference stack (2026-08, verified):

- torch **2.14.0.dev20260809+rocm7.14**, torchvision 0.29.0.dev+rocm7.14,
  torchaudio 2.11.0.dev+rocm7.14
- ROCm **7.14** (HIP 7.14.60850) via the `rocm-sdk-*` pip packages
- Native `gfx1201` arch support — **no `HSA_OVERRIDE_GFX_VERSION` needed**

Two install paths, both from pip (no system ROCm driver install; the container
shares the host's amdgpu kernel driver via /dev/kfd):

**A. At image build time (Dockerfile):**

```dockerfile
RUN python -m pip install \
    --index-url https://download.pytorch.org/whl/nightly/rocm7.14 \
    --extra-index-url https://rocm.nightlies.amd.com/v2/gfx120X-all/ \
    --pre \
    rocm-sdk-core rocm-sdk-libraries-gfx120x-all rocm-sdk-devel \
    torch torchaudio torchvision
```

- `--pre` is **mandatory**: TheRock nightly wheels are alpha-versioned
  (`7.14.0a...`) and pip skips them without it.
- `rocm-sdk-core` = runtime core; `rocm-sdk-libraries-gfx120x-all` = gfx120X
  optimized libs (rocBLAS/hipBLAS); `rocm-sdk-devel` = hipcc/clang for Triton JIT.

**B. In-place upgrade (7.13 nightly → 7.14 stable, measured 2026-08-11):**
install everything with `--no-deps` in stages to avoid rocm's self-referential
dependency cycle: torch family → `triton-rocm` → `rocm-sdk-core` +
`rocm-sdk-libraries` + `rocm-sdk-device-gfx1201` + `rocm-sdk-device-gfx1200` →
`rocm` (metapackage) → `rocm-sdk-devel` (direct wheel URL from
repo.amd.com/rocm/whl-multi-arch). Note **package names differ between nightly and
stable**: nightly = `rocm-sdk-libraries-gfx120x-all`, stable = `rocm-sdk-libraries`
+ `rocm-sdk-device-gfx120X` — uninstall the orphaned nightly package after the
upgrade.

Verification (4 checks, all must pass): ① ComfyUI starts with no ERROR/Traceback;
② all custom nodes import; ③ a real txt2img renders; ④ `system_stats` reports all
GPUs natively (2× R9700 + iGPU in our case).

### 2.3 Triton & triton-rocm (not part of stock ComfyUI)

`--enable-triton-backend` (used in §4 startup) requires Triton, which ComfyUI does
not ship:

- Install **`triton-rocm`** (ROCm build; e.g. 3.8.0+git...). The PyPI `triton`
  package is the CUDA build — installing it breaks the ROCm setup.
- `triton-rocm` JIT-compiles kernels at runtime, so it needs `rocm-sdk-devel`
  (hipcc/clang) — this is why the base image keeps the compiler toolchain.
- **Package-name collision pitfall**: PyPI `triton` and `triton-rocm` both provide
  a `triton/` module dir. If PyPI triton overwrites triton-rocm and you then
  uninstall it, the shared dir is left corrupted (`module 'triton' has no attribute
  'language'`). Fix: uninstall **both**, `rm -rf site-packages/triton*`, reinstall
  triton-rocm.
- Verify engagement from the startup log: `Found triton 3.8.0. Enabling
  comfy-kitchen triton backend.` / `comfy_kitchen backend triton: available True`.
- **Key payoff for convrot models**: with the Triton backend on, comfy-kitchen's
  HIP backend replaces the slow CPU dequant path — `dequantize_convrot_w4a4_weight`
  and `rotate_int8_convrot_weight` run as GPU kernels, cutting convrot cold load
  from 30–40 min to ~85 s (warm 55 s). This is the fix for pitfall #6.

### 2.4 Custom nodes added during debugging

Beyond the 3 baked-in nodes, these were installed afterwards (each is a git clone
into `custom_nodes/` plus its own requirements):

- `ComfyUI-Manager` — node manager. ⚠️ **It installs deps with uv and can pull the
  PyPI CUDA torch over the ROCm one** (see pitfall #1; fixed with a reinstall
  script).
- `ComfyUI-Custom-Scripts` (pysssss) — PlaySound etc.
- `ComfyUI-VideoHelperSuite` (VHS) — video combine/encode.
- `ComfyUI-FlashVSR` — video super-resolution.
- `ComfyUI-S3-IO`
- `XB_ToolBox` (机智罗) — the MiniMax-H3 workflow nodes (params, Sage/BlockSwap,
  VAEDecode).
- `ComfyUI-MultiGPU` — provides `CLIPLoaderMultiGPU` used in §5.
- `ComfyUI-INT8-Fast-ROCM` (patientx) — Triton-based loaders for convrot int8
  (optional; fp8 makes it moot).
- `sageattention` 2.2 with gfx12 native backend (PR #368) — built from source
  (ROCm-specific build; see the sibling skill `build-sageattention-rocm-on-win11`
  for the Windows build story — the Linux/ROCm one follows the same kernel path).

> **RDNA3 note**: triton-rocm coverage and the INT8-Fast-ROCM kernel selection
> differ by gfx arch. On RDNA3 the `rocm-sdk-device-gfx1201` packages and the
> gfx12-specific sageattention kernels don't apply — pick the matching
> `rocm-sdk-device-gfxXXX` and check each custom node's arch support.

## 3. Environment layout (containerized ComfyUI)

The reference stack runs ComfyUI inside a **distrobox container** (rootless podman
under the hood) on the host. Key facts about this setup:

- **Host/container share `$HOME`**: distrobox mounts the host home into the
  container, so paths like `~/comfyui/` are identical on both sides — no
  `podman cp` needed for file transfer; just move files on the host and they're
  visible inside.
- **Container networking is host mode**: `-p` port mappings are ignored; the
  container's port *is* the host port (e.g. ComfyUI on 8188 is reachable at the
  host's IP directly). A `--listen 127.0.0.1` instance is therefore only
  reachable from the host itself, not from other LAN machines.
- **GPU access** requires privileged mode + `--security-opt seccomp=unconfined`
  (distrobox flags); with that, all GPUs are visible inside — multi-GPU offload
  (§5) works without extra passthrough.
- **Non-interactive execution**: `distrobox enter <name> -- bash -c "cmd"`; in
  this mode `$PATH` often misses the venv, so use full paths
  (`/home/you/comfyui/.venv/bin/python ...`). Beware nested-quote blowups with
  `bash -c` — for anything non-trivial, write a script, scp it, run it.
- **Rootless quirks**: `py-spy` attach needs `sudo`; no `rg`/`sqlite3` CLI inside
  the container (use `grep -rn`, or the venv python for DB queries).

Reference layout (adjust paths to yours):

- ComfyUI source + uv venv: `~/comfyui/` (torch 2.14.0.dev+rocm7.14, Python 3.13)
- Model/workflow data: `~/comfyui-assets/` (`models/`, `workflows/`, `output/`, `input/`)
- Data is linked into the source tree via symlinks (`~/comfyui/models` →
  `~/comfyui-assets/models`, etc.) — no extra_model_paths.yaml, no directory flags.
- Workflows are **files**, not DB entries: `~/comfyui-assets/workflows/` (symlinked to
  `user/default/workflows`). `user/comfyui.db` only holds the assets library.

## 4. Startup tuning

### Verified-good startup (reference stack)

```bash
cd ~/comfyui && ./.venv/bin/python main.py \
  --enable-triton-backend \
  --disable-pinned-memory \
  --disable-mmap \
  --listen 0.0.0.0 --port 8188
```

No environment variables. (Earlier guides suggest
`TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` and
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` — **both were part of the NaN
culprit set, see §6.**)

Verify the Triton backend actually engaged (startup log):

```
Found triton 3.8.0. Enabling comfy-kitchen triton backend.
comfy_kitchen backend triton: available True, disabled False
```

If you see `triton unavailable` ("Neither CUDA nor XPU available"), your torch got
overwritten by a PyPI CUDA build — reinstall the ROCm wheels (see pitfall #1).

### Flags we measured and rejected

| Flag / knob | Result on R9700×2 (fp8) | Verdict |
|---|---|---|
| `--disable-dynamic-vram` | same 55 s as default | no difference (model > VRAM → NORMAL_VRAM either way) |
| `--disable-pinned-memory` | part of stable set | keep (harmless, recommended) |
| `--fast-disk` | no change | skip |
| `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` | NaN risk | **remove** |
| `PYTORCH_HIP_ALLOC_CONF=expandable_segments:True` | NaN risk | **remove** |
| `--use-pytorch-cross-attention` | NaN risk | **remove** |
| XB_Sage_BlockSwap SW=25 | same 55 s (convrot int8), +8 GB RAM | skip (keep SW=0) |
| XB_Sage_BlockSwap SW=50 | run1 168 s; run2 RAM thrash, timeout | never |

> **RDNA3 note**: on 24 GB cards (7900 XTX), smaller swap values *did* help
> (SW25/SW10 ≈ −15–21 % vs SW50) because the UNet doesn't fully fit in VRAM. The
> "swap less = faster" result above only holds when the model fits. Re-benchmark
> SW on your card.

## 5. Multi-GPU: offload the text encoder

A 32B text encoder (14 GB int4 / ~20 GB fp8-era) plus a 20 GB UNet barely fits one
32 GB card at 1080p. With two GPUs, put the TE on GPU1 and keep the UNet on GPU0.

Implementation: swap the `CLIPLoader` node class to `CLIPLoaderMultiGPU`
(provided by the **ComfyUI-MultiGPU** custom node) and set its `device` input:

```python
# in an API prompt (Python test harness)
api["130"]["class_type"] = "CLIPLoaderMultiGPU"
api["130"]["inputs"]["device"] = "cuda:1"
```

Or edit the workflow JSON node directly: replace node 130's `class_type` with
`CLIPLoaderMultiGPU` and set its `device` input to `cuda:1`. (The stock
`CLIPLoader` device COMBO only offers `default`/`cpu` — the MultiGPU wrapper
adds the physical device list, which on a 3-GPU box is `cuda:0/cuda:1/cuda:2`;
pick `cuda:1`, never the iGPU `cuda:2`.) GPU0 = card 0, GPU1 = card 1; with
`HIP_VISIBLE_DEVICES` set, the indices shift accordingly.

### Verify placement (don't trust it blindly)

```bash
# per-card VRAM (kernel sysfs — works inside container with host network)
for f in /sys/class/drm/card*/device/mem_info_vram_used; do
  echo "$f: $(( $(cat $f) / 1024**3 )) GB"
done
# or
rocm-smi --showmeminfo vram | grep -E "GPU\[|Used"
```

Measured after a 1080p generation on the reference stack:
`GPU0 ≈ 20.7 GB (UNet+VAEs) + GPU1 ≈ 14.7 GB (TE)` — GPU0 sampling peak 27.6 GB,
no OOM, system RAM 31–32 GB used of 59 GB.

> **RDNA3 note**: smaller VRAM cards benefit even more from splitting, but watch
> total system RAM — two processes (or one process holding two cards' weights +
> CPU copies) can exceed 64 GB. SW>0 adds CPU-side copies on top.

## 6. Troubleshooting: black videos / audio NaN

**Symptom**: intermittent full-black output videos (latent ≈ 0) and/or audio NaN,
independent of seed/prompt/sage switch. Status often reports `success` even for bad
output — a broken encode can flip history to `error`, but a "successful" render can
still be a black clip.

### Detection triad (verify every run)

1. **History status**: `GET /history/<prompt_id>` → `status_str`.
2. **Frame mean/std**: extract a frame, compute grayscale stats
   (black ≈ mean 2–3, healthy ≈ 80–115):
   ```bash
   ffmpeg -y -v error -ss 2 -i out.mp4 -frames:v 1 /tmp/an.png
   # then read mean/std via PIL/numpy
   ```
3. **Audio volume**: `ffmpeg -i out-audio.mp4 -af volumedetect -f null - 2>&1`
   — healthy ≈ −14 dB; NaN prints an aac "Input contains (near) NaN" error.

### Root cause (verified on R9700/ROCm 7.14/gfx1201, convrot models)

The culprit was **three knobs** added for "performance":
`--use-pytorch-cross-attention` + `TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL=1` +
`PYTORCH_HIP_ALLOC_CONF=expandable_segments:True`. Removal → 9 consecutive
successful renders (including runs 4–9 after warm-up); with them kept, 2/3 later
runs produced audio NaN. Pattern: **delayed degradation** — first 1–3 runs after
restart look fine, failures appear after the model stays resident. This is why
"restart then test twice" gives a false sense of stability.

> With fp8 models the issue was not reproducible at all — another strong argument
> for fp8 over convrot on ROCm RDNA4. Verified on the reference stack: fp8 UNet +
> TE on cuda:1 ran **8 consecutive successful renders** across the 5 s / 10 s /
> 15 s / 720p / 1080p matrix (§7, §7.2) with no black frame and no audio NaN.
> Convrot's CPU-side rotation kernels and ROCm's FP8 paths may differ numerically
> on other architectures; re-test.

### Debugging discipline that worked

- **Never judge stability from a fresh restart**: run ≥6–9 consecutive renders,
  watch the resident (4th+) runs.
- **Don't trust the exit code / status alone** — always run the detection triad.
- **Sampling speed is not a health signal**: broken runs showed identical
  ~8.5–8.9 s/it; it's a numeric issue, not a hang.
- **py-spy to find the hang point** (rootless container needs sudo):
  `sudo ~/comfyui/.venv/bin/py-spy dump --pid <pid>`.
- **Interrupt can't cancel a model-loading prompt** — `/interrupt` only logs
  "Global interrupt" until loading finishes; `/queue {"clear": true}` won't stop a
  running load either. Wait or kill main.py.

## 7. Performance reference (R9700×2, fp8, 124 frames @24fps, 4 steps)

All numbers below were measured with the bundled test workflows (see §7.1) on the
reference stack: fp8 UNet + text encoder on `cuda:1`, 124 frames @24fps (5 s), 4
steps, res_multistep/simple, LoRA 0.75, BlockSwap off, random seeds.

### 7.1 Bundled test workflows

Three reference workflows from the 机智罗/XB_ToolBox MiniMax-H3 family ship with
this skill under `workflows/` (same files as used for every measurement here):

- `58-MiniMax-H3 文-图-首尾帧生视频 切换工作流-Lora加速(1).json` — text / image /
  first+last-frame to video, mode switched by enabling/disabling the reference
  image groups.
- `59-MiniMax-H3 图片-视频-音频-多参生视频 切换工作流-Lora加速(1).json` — image-video
  with more params (audio etc.).
- `60-MiniMax-H3 文生视频测速工作流.json` — text-to-video **benchmark workflow**
  (used for all speed tests). Contains `RandomNoise` (node 131), the
  `XB_HailuoH3VideoParams` params node (161), `UNETLoader` (129),
  `CLIPLoader` (130), `LoraLoaderModelOnly` (181), `XB_Sage_BlockSwap` (136),
  `MiniMaxH3ImageToVideo` (133), VHS output (168).

To reproduce the tuning/troubleshooting flows from this guide:

1. Drop the JSON into your ComfyUI `user/default/workflows/` (or open it in the
   UI via Workflow → Open).
2. Point the model loaders at **your** model files: node 129
   `unet_name` (diffusion model), node 130 `clip_name` (text encoder), plus the
   VAE/LoRA nodes. Workflow defaults reference common MiniMax-H3 file names — the
   actual files are whatever you downloaded (§1).
3. Multi-GPU: replace node 130's `class_type` with `CLIPLoaderMultiGPU` and set
   `device` to `cuda:1` (§5).
4. For benchmark runs, randomize `noise_seed` per run (node 131) and read
   results from VHS output + `GET /history` (§6 detection triad).

Node IDs above are stable in the 60 benchmark workflow; the 58/59 variants share
the same core node set.

| Resolution | Area | Gen time | Audio | GPU0 peak | Note |
|---|---|---|---|---|---|
| 480×864 | 0.41 MP | **55 s** | −14.5 dB | 25.8 G | baseline (workflow default) |
| 1280×720 | 0.92 MP (2.2×) | **175 s** | −15.5/−17.3 dB | 25.9 G | best value point |
| 1920×1080 | 2.07 MP (5×) | **650 s** (10m50s) | −21.7/−19.6 dB | **27.6 G** | near VRAM ceiling |

- Generation time scales **super-linearly** (2.2× area → 3.2× time; 5× area →
  11.8× time) — spatio-temporal attention is ~quadratic in tokens. 1080p at 124
  frames is the practical limit for 32 GB; 4K would OOM.
- 1080p output lands at 1920×1072 (VAE 16-px alignment); 720p is natively aligned.
- For comparison: a 7900 XTX (24 GB, RDNA3) took ~190–280 s for the same 480×864
  job on convrot — R9700's RDNA4 int8/fp8 kernels are significantly faster, but
  your mileage varies by architecture.

### 7.2 Duration scaling (480×864, fp8, 24 fps, 4 steps, TE on cuda:1)

| Duration | Frames | Gen time (run1/run2) | Audio | System RAM | GPU0/GPU1 |
|---|---|---|---|---|---|
| 5 s | 124 | **55 s** (hot; +~110 s first fp8 load) | −14.5 dB | 28 G | 25.8 / 14.7 G |
| 10 s | 243 | **145–150 s** | −15.6/−14.0 dB | 29 G | 25.9 / 14.7 G |
| 15 s | 362 | **270 s** | −14.2/−18.5 dB | 29 G | 25.9 / 14.7 G |

Duration is set via the `XB_HailuoH3VideoParams` node (161) `duration` input
(seconds; frames = `round(duration×24)` then aligned up to a `%17` boundary —
5 s→124, 10 s→243, 15 s→362). Gen time scales roughly **linearly with frames**
(~55 s per 124-frame block), while VRAM/RAM stay flat — 15 s at 480×864 leaves
~30 GB RAM and ~6 GB GPU0 headroom, so frame count is the cheap axis to extend
once the model is split across two GPUs.

## 8. Pitfall reference (symptom → cause → fix)

1. **Startup: `Found no NVIDIA driver` / triton unavailable** → a custom-node
   install pulled PyPI CUDA torch over the ROCm build (`torch.__version__` shows
   `+cu130`). Fix: uninstall triton+triton-rocm, reinstall torch family with
   `--reinstall --no-deps --index-url https://download.pytorch.org/whl/nightly/rocm7.14`,
   reinstall triton-rocm, verify `import torch; torch.__version__` contains rocm.
2. **VHS encode: `Encoder not found` for libx264, prompt marked error** → distro
   `ffmpeg` is the patent-stripped `ffmpeg-free`. Install full ffmpeg (RPM Fusion
   on Fedora-based containers: `dnf install --allowerasing ffmpeg`).
3. **Long task killed when SSH drops** → bare `setsid`/nohup chains die with the
   SSH session scope. Use `systemd-run --user` transient units; redirect logs
   inside the unit command.
4. **`pkill -f "main.py"` kills your own SSH** → the pattern matches the SSH
   command line itself. Use `pgrep -f '\.venv/bin/python main\.py'` or `[m]ain.py`.
5. **Windows workflow JSON on Linux: wrong model loaded silently** → paths use
   backslashes; Linux treats `\` as a filename char, the COMBO validation swaps
   the value to something else (e.g. audio VAE replaced by video VAE → crash in
   VAEDecodeAudio `IndexError`). Normalize all paths to `/` (walk the parsed JSON,
   don't blind-replace `\n`/`\u`), re-open the workflow fresh in the UI.
6. **40-min cold start "hang"** → convrot CPU conversion is single/double
   threaded by design. **Fixed by `--enable-triton-backend`**: comfy-kitchen's
   HIP backend GPU kernels (`dequantize_convrot_w4a4_weight`,
   `rotate_int8_convrot_weight`) take over dequantization → cold load drops to
   ~85 s, warm runs to 55 s (measured). Keep the flag; without it, timeout tests
   must allow ≥7200 s, or switch to fp8 (1–2 min).
7. **`status=success` but black video** → always run the detection triad (§6);
   NaN audio often accompanies it.
8. **`/interrupt` seemingly ignored during load** → see §6; loading is
   uninterruptible.
9. **system_stats device list location** → `d["devices"]` (top level), not
   `d["system"]["devices"]` — a common porting bug in monitor scripts.
10. **Rerunning a workflow "instantly" succeeds with stale output** → the seed
    wasn't actually randomized: in API JSON, `RandomNoise` seed lives in
    `widgets_values[0]`, index [1] is the `"randomize"` control value — skip the
    wrong index and the cache returns the previous result.
11. **container has no `rg`/`sqlite3` CLI** → use `grep -rn`; for DB queries run
    python inside the container with the venv interpreter.

## 9. Files & resources

- **Workflows**: 机智罗/XB_ToolBox MiniMax-H3 family (58/59/60 variants for
  text/image/first-last-frame-to-video) — bundled in `workflows/` (§7.1).
- Custom nodes required: `XB_ToolBox` (params, Sage/BlockSwap, VAEDecode),
  `ComfyUI-MultiGPU` (CLIPLoaderMultiGPU), `ComfyUI-VideoHelperSuite` (VHS),
  `ComfyUI-INT8-Fast-ROCM` (optional; its Triton kernels help convrot GEMMs, but
  loading still goes through the slow native path — fp8 makes it moot).
- sageattention 2.2 with gfx12 native backend (PR #368) installed and working,
  but it was **not** the cause of black screens (disabled still produced them).
- Base image repo: `github.com/kyuz0/amd-r9700-comfy` (Dockerfile, §2.1);
  sibling skill `build-sageattention-rocm-on-win11` covers the Windows ROCm build.
