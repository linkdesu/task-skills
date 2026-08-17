---
name: vllm-r9700-rocm-tuning
description: "vLLM on AMD Radeon R9700 (gfx1201/RDNA4): source-build 0.27.x, AITER unified-attention env trap, GPU_MAX_HW_QUEUES, NCCL/HSA tuning, tuned FP8 GEMM configs, distrobox GPU passthrough."
type: prompt
whenToUse: When running vLLM on AMD RDNA4 GPUs (Radeon R9700 / gfx1201, ROCm 7.14 TheRock track) and you need to build from source, fix silent startup failures, or squeeze decode throughput out of AITER + MTP speculative decoding.
---

# vLLM on AMD Radeon R9700 (gfx1201) — Containerized Source Build & Tuning

Field-verified notes from serving Qwen3.8-27B-FP8 on 2× AMD Radeon AI PRO R9700
with ROCm 7.14, source-built vLLM. Covers the full journey: source build, the
AITER environment-variable trap that silently kills startup, RDNA4-specific
env tuning, tuned FP8 GEMM configs, cache persistence, and distrobox/rootless
podman GPU passthrough.

> **Reference hardware**: AMD Radeon AI PRO R9700 (`gfx1201`, RDNA4) × 2, 32 GB
> VRAM each, ROCm 7.14 (TheRock track), distrobox (rootless podman) container.
> **Numbers are for this stack only** — RDNA3 (7900 XTX), MI300X, or CUDA boxes
> will behave differently; re-measure on your own hardware.

## Scope: containerized source build, two complementary paths

This skill documents the **from-source build** of vLLM for gfx1201, done
entirely inside a container (distrobox / docker / podman). The container is
not just a convenience — the whole build, including the ROCm/torch toolchain
and every patch, is captured in image layers, so a failed or interrupted
build is a throwaway layer, never a change to your host system. Work happens
in a sandbox; only the finished image is promoted for use.

Two complementary references:

- **[kyuz0/amd-r9700-ai-toolboxes](https://github.com/kyuz0/amd-r9700-ai-toolboxes)** —
  pre-built "toolbox" containers (llama.cpp, and the vLLM TheRock image we
  started from). If you want a working R9700 inference container **without
  compiling anything**, start here: `distrobox create --image
  docker.io/kyuz0/vllm-therock-gfx1201:latest`. This is the fast path.
- **[prcoe1/r9700-serving](https://github.com/prcoe1/r9700-serving)** — the
  from-source vLLM build recipe (multi-stage Dockerfile, compose, tuned FP8
  configs, benchmarks) that this skill is based on. Use this when you need a
  newer vLLM than the prebuilt image ships, or want the tuning knobs.

**When to build from source**: the prebuilt kyuz0 image carries an older vLLM
(0.22.1rc1 in our case); source-building 0.27.1 was worth a substantial decode
gain (0.26→0.27 alone was ~+16% upstream). The cost is a ~hour-long network-
bound build (§2) — well contained inside the container.

## 0. TL;DR (verified conclusions)

1. **The #1 silent killer: AITER env vars default to `True`.** vLLM's
   `envs.py` defaults `VLLM_ROCM_USE_AITER_MHA/MLA/MOE/LINEAR/FP8BMM/FP4BMM/
   TRITON_GEMM/RMSNORM` to **True**. If you set `VLLM_ROCM_USE_AITER=1` +
   `VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1` but do NOT explicitly set the
   other eight to `0`, vLLM starts and dies instantly with no useful output —
   AITER's MoE/FP8 backends abort on gfx1201 at startup. **Always ship the full
   11-var block** (see §4).
2. **`GPU_MAX_HW_QUEUES=1` is required on RDNA4** — multiple hardware queues
   cause a 55-63% decode throughput regression. One queue per process avoids
   kernel-launch scheduling overhead. Zero cost, mandatory.
3. **Source-build vLLM 0.27.1** (prcoe1's Dockerfile recipe) beats the
   prebuilt kyuz0 0.22.1rc1 image substantially on decode; the 0.26→0.27 jump
   alone was ~+16% (75→87 tg32) upstream. 27B dense FP8, TP=2, AITER unified
   attention + MTP3 lands ~57.6 tok/s tg32 on the upstream bench.
4. **Tuned w8a8 block-FP8 GEMM configs are a real win** (27B tg32 +19%,
   pp2048 +3% upstream A/B). vLLM matches config files by
   `N=<out>,K=<in>,device_name=<gpu>,dtype=fp8_w8a8,block_shape=[128,128].json`
   in `.../quantization/utils/configs/`. Same-model GPU ⇒ reuse the upstream
   configs directly instead of re-running the 576-config sweep.
5. **Rootless podman needs the distrobox parameter combo for GPU access** —
   bare `--device /dev/kfd --device /dev/dri` leaves torch with "No CUDA GPUs
   available". The working combo: `--userns=keep-id --privileged
   --annotation run.oci.keep_original_groups=1 --ipc host --pid host
   --network host`. Easiest path: just use distrobox, it wires this up for you.
6. **Persist compile caches** (`AITER_JIT_DIR`, `TRITON_CACHE_DIR`,
   `TORCHINDUCTOR_CACHE_DIR`, `TILELANG_CACHE_DIR` → `~/.cache/*`) — aiter JIT
   becomes an `import` of a prebuilt `.so` instead of a rebuild; but note
   torch.compile's cache key changes when env changes, so the first launch
   after a config change still recompiles (expect a long cold start with a
   66-shard 27B on this box — most of it safetensors disk I/O, not
   compilation).
7. **MTP on Qwen3.8-27B accepts drafts poorly**: startup-time acceptance can
   read ~40%, then settles to 50-90%, dropping to 40-50% at 80K context.
   Upstream found MTP3 optimal for 3.8 (MTP4 wastes compute past position 3).
   On this model, weigh MTP against no-speculative for long-context agent
   loads — earlier controlled A/B on 0.22 showed MTP net-negative at 18K+
   context. Measure, don't assume.

## 1. Version stack (reference)

| component | version |
|---|---|
| ROCm | 7.14.0 (`rocm/dev-ubuntu-24.04:7.14.0-full`) |
| PyTorch | 2.13.0+rocm7.14.0 (from `repo.amd.com/rocm/whl-multi-arch/`) |
| vLLM | 0.27.1 (source build, `gfx1201`) |
| AITER | v0.1.19.post2 |
| flash-attention | ROCm @ `1cc7ff67` |
| Triton | 3.8.0+git (ROCm) |

ROCm 7.14 is on AMD's TheRock technology-preview stream (7.9/7.13/7.14); the
production 7.2.x line lacks RDNA4/gfx1201 support. vLLM ≥0.26 on gfx1201
requires source builds.

## 2. Source build (vLLM 0.27.1)

Use the upstream repo's multi-stage `Dockerfile.fullbuild` (framework-base →
flash-attention → aiter → vllm → runtime) — it handles gfx1201 arch flags,
patches AITER, and builds AMD torch from the official wheel index.

> **Build environment**: the build is meant to run inside a container
> (distrobox / docker / podman) — see Scope above. Use the image built here
> with `distrobox create --image localhost/vllm-fullbuild:latest` (or run it
> directly with podman/docker), so the ROCm/torch toolchain and all patches
> live in the image layers, not on your host. Build time is dominated by
> network (base image ~10 GB+, AMD torch wheels, FetchContent git clones), so
> it varies a lot by link; expect roughly an hour on a decent connection and
> budget more on slow/proxied links. Run it in a server-side tmux session so
> SSH drops don't kill it.

```bash
# on the target host (rootless podman; docker also works)
git clone https://github.com/prcoe1/r9700-serving.git ~/r9700-serving
cd ~/r9700-serving
cp .env.example .env
# edit: RENDER_GID=$(getent group render | cut -d: -f3) — not the repo default!
#       MAX_JOBS=12 is safe on a 16-core/59 GB box (avoid OOM during HIP builds)
podman build \
  --build-arg ROCM_IMAGE=docker.io/rocm/dev-ubuntu-24.04:7.14.0-full \
  --build-arg GPU_ARCH=gfx1201 \
  --build-arg MAX_JOBS=12 \
  --build-arg TORCH_VERSION=2.13.0+rocm7.14.0 \
  --build-arg TORCHVISION_VERSION=0.28.0+rocm7.14.0 \
  --build-arg VLLM_REF=v0.27.1 \
  --build-arg VLLM_VERSION=0.27.1 \
  --build-arg AITER_REF=v0.1.19.post2 \
  --build-arg FLASH_ATTN_REF=1cc7ff67cbc5685046c75183e8defecca3e35d5c \
  -t localhost/vllm-fullbuild:latest -f Dockerfile.fullbuild .
```

- **Stalls that look like hangs but aren't**: cmake FetchContent full-clones
  ROCm/triton (~400 KB/s over a fake-ip DNS proxy; transient 0-byte periods
  are normal). HIP "loop not unrolled" warnings are benign.
- **No `podman compose` plugin?** The repo's `just` recipes target docker
  compose; on a podman-only host just run `podman build` directly with the
  build-args above (from `.env`).

## 3. Runtime environment — the full working env block

Mirror the upstream `env/2xr9700.vllm.common` + `env/aiter-unified-attention.env`
+ compose `environment:`. Everything below is verified to work together:

```bash
# --- AITER: ONLY unified attention. The other 8 MUST be 0 (default True = startup abort on gfx1201)
export VLLM_ROCM_USE_AITER=1
export VLLM_ROCM_USE_AITER_MHA=0
export VLLM_ROCM_USE_AITER_MLA=0
export VLLM_ROCM_USE_AITER_MOE=0
export VLLM_ROCM_USE_AITER_LINEAR=0
export VLLM_ROCM_USE_AITER_FP8BMM=0
export VLLM_ROCM_USE_AITER_FP4BMM=0
export VLLM_ROCM_USE_AITER_TRITON_GEMM=0
export VLLM_ROCM_USE_AITER_RMSNORM=0
export VLLM_ROCM_USE_AITER_UNIFIED_ATTENTION=1
export VLLM_ROCM_SHUFFLE_KV_CACHE_LAYOUT=0

# --- RDNA4 / NCCL / ROCm
export GPU_MAX_HW_QUEUES=1            # REQUIRED: multi-queue = 55-63% decode regression on RDNA4
export NCCL_P2P_DISABLE=1             # two GPUs on separate PCIe root ports
export NCCL_MIN_NCHANNELS=4
export NCCL_MAX_NCHANNELS=4           # bandwidth sweet spot for TP=2, P2P off
export NCCL_PROTO=Simple              # needed for TP=2 on this box
export HSA_ENABLE_IPC_MODE_LEGACY=1   # required by the ROCm stack
export HSA_NO_SCRATCH_RECLAIM=1       # avoid scratch reallocation stalls
export SAFETENSORS_FAST_GPU=1         # faster safetensors load on GPU
export PYTORCH_NVML_BASED_CUDA_CHECK=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export HIP_VISIBLE_DEVICES=0,1
export ROCR_VISIBLE_DEVICES=0,1

# --- persistent compile caches (restart fast; keep them user-owned)
export AITER_JIT_DIR=$HOME/.cache/aiter/jit
export TRITON_CACHE_DIR=$HOME/.cache/triton
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor
export TILELANG_CACHE_DIR=$HOME/.cache/tilelang
```

Already baked into the fullbuild image (do NOT re-export): `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`,
`HIP_FORCE_DEV_KERNARG=1`, `LD_PRELOAD=/opt/rocm/lib/libamd_smi.so`,
`TORCH_BLAS_PREFER_HIPBLASLT=1`, `TOKENIZERS_PARALLELISM=false`,
`AMDGPU_TARGETS=gfx1201`, `PYTORCH_ROCM_ARCH=gfx1201`.

**⚠️ Do not set `HSA_OVERRIDE_GFX_VERSION`** — the TheRock stack patches
gfx1201 natively; overriding breaks HSA init.

## 4. The AITER trap (startup dies silently) — details

Symptom: `vllm serve ... --attention-backend ROCM_AITER_UNIFIED_ATTN` with a
partial env block (just `VLLM_ROCM_USE_AITER=1` + `UNIFIED_ATTENTION=1`)
produces a banner, then nothing — no error, no listen, process exits. Checked
in `vllm/envs.py`: every `VLLM_ROCM_USE_AITER_*` component defaults to `True`,
so setting only the two switches enables **all** AITER backends, and AITER's
MoE/FP8 kernels abort on gfx1201 at startup (upstream README: "AITER MoE/FP8
backend on gfx1201: vLLM aborts at startup").

Fix: always export the full 11-var block (§3) — `VLLM_ROCM_USE_AITER=1` plus
all eight component `=0` plus `UNIFIED_ATTENTION=1`.

## 5. Tuned FP8 GEMM configs (the +19% decode win)

vLLM looks up per-shape Triton kernel configs in
`.../vllm/model_executor/layers/quantization/utils/configs/N=<out>,K=<in>,
device_name=<gpu>,dtype=fp8_w8a8,block_shape=[128,128].json`. For a dense 27B
at TP=2 the per-GPU shapes are: `(17408,5120) (5120,3072) (5120,8704)
(7168,5120) (8192,5120)` — the upstream repo ships tuned JSONs for exactly
these (device_name=`AMD_Radeon_R9700`).

- **Same-model GPU → copy the upstream JSONs, don't re-tune.** The
  `tools/tune_fp8_dense.py` sweep (576 Triton tile configs × shape with fp32
  numeric gating) takes a long time and yields the same result on identical
  hardware.
- Mount/copy them into the configs dir inside the container (ro bind-mount per
  file, or `cp` into site-packages for a persistent container).
- **Verify engagement** — the startup log must show, per worker:
  `Using configuration from ...N=8192,K=5120,device_name=AMD_Radeon_R9700...json
  for W8A8 Block FP8 kernel.` If you don't see it, the configs aren't being
  read (wrong path, wrong device_name, or silent stock fallback).

## 6. Container deployment: distrobox vs bare podman

**Use distrobox.** GPU passthrough under rootless podman is fragile; distrobox
creates the container with the working combo automatically:

```bash
distrobox create --name vllm-r9700-v2 --image localhost/vllm-fullbuild:latest --yes
# verify GPU is visible (do NOT use `distrobox enter` for one-shots — first
# entry runs init and is slow; use podman exec against the created container):
podman start vllm-r9700-v2
podman exec vllm-r9700-v2 /opt/venv/bin/python -c \
  "import torch; print(torch.cuda.get_device_name(0), torch.cuda.device_count())"
```

- **Bare `podman run` GPU gotcha**: `--device /dev/kfd --device /dev/dri
  --group-add video --group-add render` is NOT enough — torch reports "No CUDA
  GPUs available" even with `--privileged`. The full working combo is
  `--userns=keep-id --privileged --annotation run.oci.keep_original_groups=1
  --ipc host --pid host --network host`. The `keep_original_groups` annotation
  is what passes host supplementary groups (render) into the container so it
  can open `/dev/kfd`.
- **distrobox mounts `$HOME`** — `~/r9700-serving` and `~/.cache` are visible
  inside; overlay files (protocol.py, fp8 configs) copied once into
  site-packages persist across restarts.
- Pre-create `~/.vllm-workspace` (and `~/.cache/*` dirs) before starting, or
  podman errors `statfs ... no such file or directory` on the bind mount.

## 7. Serving command (Qwen3.8-27B-FP8, TP=2, verified)

```bash
distrobox enter vllm-r9700-v2 -- /opt/venv/bin/vllm serve /home/link/models/unsloth/Qwen3.8-27B-FP8 \
  --served-model-name qwen3.8-27b --host 0.0.0.0 --port 9527 \
  --tensor-parallel-size 2 \
  --attention-backend ROCM_AITER_UNIFIED_ATTN \
  --max-num-seqs 4 --max-num-batched-tokens 8192 \
  --max-model-len auto \
  --gpu-memory-utilization 0.92 \
  --kv-cache-dtype fp8 --calculate-kv-scales \
  --enable-chunked-prefill --enable-prefix-caching \
  --dtype auto --trust-remote-code \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser qwen3_coder \
  --chat-template /home/link/r9700-serving/chat-templates/qwen.jinja \
  --speculative-config '{"method": "mtp", "num_speculative_tokens": 3, "attention_backend": "ROCM_AITER_UNIFIED_ATTN"}'
```

- `--chat-template`: use froggeric's Qwen-Fixed-Chat-Templates v22 — fixes
  KV-cache invalidation, token waste, and agentic stalls in stock Qwen
  templates; pairs with `--reasoning-parser qwen3` for proper thinking/content
  splitting.
- `--kv-cache-dtype fp8 --calculate-kv-scales`: fine (upstream prefers bf16 KV
  + AITER LDS-fit patch, ~88 t/s — but fp8 keeps more KV headroom at 262K ctx;
  A/B on your load).
- Harmless warnings to ignore: `CUDA_VISIBLE_DEVICES ... deprecated` (use
  HIP_VISIBLE_DEVICES; only a conflict raises), `min_frames/max_frames ...
  not documented` (transformers docstring lint), `all_gather_into_tensor is
  deprecated` (torch rename), `Triton kernel JIT compilation during inference
  ... _w8a8_triton_block_scaled_mm` (one-time warmup miss; hits cache after).

## 8. MTP speculative decoding notes (Qwen3.8-27B)

- Upstream: MTP3 is the 3.8 optimum — tg32 57.6 (MTP2 56.0 / MTP4 49.2 /
  no-MTP 32.0). The 3.8 MTP head accepts drafts poorly past position 3
  (per-position ≈0.66/0.44/0.31/0.22).
- Field: acceptance fluctuates 50-90% after startup, dropping to 40-50% at
  ~80K context. Startup-time metrics can read ~40% for the first sample —
  don't tune on the first SpecDecoding metrics line.
- vLLM warns `num_speculative_tokens > 1 will run multiple times of forward on
  same MTP layer, which may result in lower acceptance rate` — expected.
- For long-context agent workloads, A/B MTP vs none before trusting it
  (earlier 0.22-era controlled tests showed MTP net-negative at 18K+ ctx on
  this box; 0.27 may differ).

## 9. Cache / startup-time expectations

| stage | cold (first launch after env change) | warm (cached) |
|---|---|---|
| safetensors load (66 shards, 27B fp8) | several minutes (disk I/O bound — cache can't help) | same |
| aiter JIT core | build once | `import module_aiter_core.so` (~0) |
| torch.compile graph | rebuild (new env → new cache key) | load from `~/.cache/vllm/torch_compile_cache/` |
| total to `GET /v1/models` 200 | long (model load + compile) | noticeably shorter |

> Actual wall-clock depends on your disk, link, and model size — the takeaway
> is *relative*: caches remove the compile part, never the safetensors I/O.

Changing env vars (e.g. adding NCCL/HSA vars) changes the torch.compile cache
key — the old `torch_compile_cache/<hash>` stays on disk but won't be hit.
Old hashes accumulate (GBs after many configs); prune
`~/.cache/vllm/torch_compile_cache/*` for a config you've abandoned.

## 10. Pitfall reference (symptom → cause → fix)

1. **vllm starts then dies silently** → partial AITER env block; components
   default `True` → abort on gfx1201. Export the full 11-var block (§3, §4).
2. **"No CUDA GPUs available" in torch under rootless podman** → missing
   `--annotation run.oci.keep_original_groups=1` (and keep-id/privileged/host
   ns). Use distrobox (§6).
3. **`statfs .../.vllm-workspace: no such file or directory`** → pre-create
   the bind-mount source dirs (§6).
4. **Decode throughput half of expectations** → `GPU_MAX_HW_QUEUES` unset
   (RDNA4 multi-queue regression). Set `=1`.
5. **SSH drop kills the build** → run builds in server-side tmux, tee the log
   to a file (§2).
6. **`podman compose` not found** → no compose plugin on host; run `podman
   build` directly with build-args (§2).
7. **Cold start still slow after "caching"** → safetensors disk I/O is the
   bulk of it and is unavoidable; aiter/triton cache only removes the compile
   part. Also: new env block = new torch.compile cache key (first launch after
   config change recompiles).
8. **Fake-ip DNS (198.18.x.x) stalls git clone during build** → transparent
   proxy on the gateway; transient 0-byte stalls are normal, don't kill the
   build (§2).
9. **MTP acceptance looks terrible on first metrics line** → startup transient;
   let it settle (50-90%) before tuning (§8).
10. **`--calculate-kv-scales` deprecated warning** → benign on 0.27; hybrid
    models auto-disable it with default scale 1.0.
11. **Chat template not applied** → must pass `--chat-template` AND keep
    `--reasoning-parser qwen3` together (§7).

## 11. Files & resources

- Pre-built containers (fast path, no compile): `github.com/kyuz0/amd-r9700-ai-toolboxes`
- From-source build recipe: `github.com/prcoe1/r9700-serving` — Dockerfile.fullbuild,
  compose.yaml, `env/` files, `fp8_configs/` (tuned JSONs), `chat-templates/`
  (froggeric v22), `tools/tune_fp8_dense.py`, `benchmarks/`.
- froggeric Qwen-Fixed-Chat-Templates: `huggingface.co/froggeric/Qwen-Fixed-Chat-Templates`
- AITER upstream: `github.com/ROCm/aiter` (v0.1.19.post2; PR #4593 fixes a
  cosmetic teardown crash, not yet in a release).
- Reference hardware/stack this was verified on: 2× R9700, ROCm 7.14,
  vLLM 0.27.1+rocm714.gfx1201, Qwen3.8-27B-FP8.
