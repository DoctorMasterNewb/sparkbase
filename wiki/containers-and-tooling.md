# Container images & tooling

> **area:** containers
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-forum-vllm-claude, S-forum-btop, S-forum-model-manager, S-forum-sparkdash, S-forum-tool-eval, S-forum-thunderkittens, S-forum-driver610, S-forum-flux2-nunchaku, S-forum-comfyui-container, S-forum-llamacpp-container, S-forum-sage-attn, S-forum-vllm-2606-broken, S-forum-gemma4-qat, S-forum-mistral-s4-nvfp4, S-forum-qwen-tts-arm64, S-forum-llama-benchy, S-forum-cluster-dashboard, S-forum-sunshine-rdp, S-forum-flux2-nvfp4-compute, S-forum-nvidia-vfx, S-forum-easy-vllm, S-forum-spark-studio, S-forum-comfyui-optimized, S-forum-litellm-orchestrator, S-forum-nemo-rt, S-forum-vllm025-nccl, S-forum-sparkdash-mia
> **updated:** 2026-07-22

Which image loads which arch is the whole game on GB10 — vLLM moves fast and arch support is
image-specific. Probe before you download; a model is only as serveable as the image that knows its
arch + quant.

## Known images (arm64 / sm_121-built)

- **[proven]** These images were pulled and run on real GB10; the "Loads" column is what each was
  confirmed to serve. (Image landscape changes monthly — re-probe rather than trusting this table
  blindly.)

| Image | Engine / build | Loads | Notes |
|---|---|---|---|
| `vllm/vllm-openai:gemma4-unified` | vLLM `0.1.dev17235…d20260603`, tfm 5.10.1 | `gemma4_unified`, ModelOpt-NVFP4 qwen3.5 MoE, FP8 | most capable on-box image for Jun-2026 arches; arm64 manifest confirmed; ~19.5 GB |
| `ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2` | vLLM dev d20260418, tfm 5.5.0 | qwen3.5/3_vl omni, **block-scale FP8** | omni/vision; **cannot** load ModelOpt-NVFP4 MoE (old loader — `KeyError w2_weight_scale`); stock NVIDIA passthrough entrypoint (supply full `vllm serve …`) |
| `ghcr.io/aeon-7/vllm-nemotron-omni-aeon-ultimate:v1` | vLLM 0.20+ omni | Nemotron-3-Nano omni (text+image+audio) | the worker vision/omni unit |
| `vllm-node-mimo-dev39` | vLLM 0.22.1rc1.dev305 (PR#41797 merged) | MiMo-V2.5 NVFP4/MXFP8 DiffKV | pinned for the MiMo mods; do NOT swap for newer dev309 |
| `avarok/dgx-vllm-nvfp4-kernel:v23` | vLLM + NVFP4 marlin kernel | Qwen3-Coder-Next NVFP4 | the head text unit; env `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_TEST_FORCE_FP8_MARLIN=1` |
| `avarok/atlas-gb10:latest` | Atlas (Rust+CUDA) | Qwen3.5/3.6/Coder/VL, Gemma-4, Nemotron-3, Mistral-Small-4, MiniMax-M2 | entrypoint `["spark"]`; see `[[wiki/engines.md]]` |

## Probing tricks (verify before a 20–35 GB download)

- **[proven]** **Arch support, no GPU/model needed:**
  `docker run --rm --entrypoint python3 <image>` → check `transformers.CONFIG_MAPPING` and
  `vllm.ModelRegistry.get_supported_archs()` for the model's `model_type` / arch class.
- **[proven]** **Quant format:** fetch the model's `config.json` over HTTP, read
  `quantization_config.quant_method`/`.format` → `modelopt` (works) vs `compressed-tensors` block-scale
  (Marlin fallback) vs `nvfp4-pack` (broken). See `[[wiki/quantization-on-gb10.md]]`.
- **[proven]** **GGUF arch:** ~2 MB HTTP range read of the header → `general.architecture`; confirm
  llama.cpp fork/upstream registers it. See `[[wiki/llama-cpp-rpc.md]]`.
- **arm64 manifest:** `docker manifest inspect <image>` — not every tag publishes arm64.

## Standard env / run

`--gpus all --ipc=host --network host`, cache mounts `~/.cache/{huggingface,vllm,flashinfer}`, and
env `TORCH_CUDA_ARCH_LIST=12.1a`, `VLLM_SKIP_P2P_CHECK=1`, `FLASHINFER_JIT_LOG_LEVEL=ERROR`,
`HF_HUB_OFFLINE=1` (when cached). Multi-node adds `NCCL_IB_HCA=<both twins>`, `NCCL_CUMEM_ENABLE=0`,
`NCCL_NVLS_ENABLE=0` (`[[wiki/multinode-tp-and-networking.md]]`).

## Download & filesystem gotchas

- **[proven]** **HF Xet finalization hangs forever** on this box even with all shards on disk (cost a
  7-hour overnight once). Always download with **`HF_HUB_DISABLE_XET=1`**, and **never gate a watcher
  on process-exit** — gate on `:8888` ready OR a fail marker OR a hard timeout.
- **[proven]** **HF cache permission-denied** (`.locks/…lock` root-owned from prior
  `--rootful`/container runs): `mv` the model dir **and** its `.locks/models--…` subdir aside (no sudo
  needed), or delete via a throwaway root container mounting the cache (`docker run --rm -v
  ~/.cache/huggingface:/c busybox rm -rf /c/…`). Passwordless sudo is not available on these boxes.
- **[proven]** **io_uring blocked by docker seccomp** (`Operation not permitted`, e.g. Atlas
  high-speed-swap): `--security-opt seccomp=unconfined`.
- **[proven]** **ComfyUI on GB10** (image-gen, GPU-shared with LLMs): flags `--disable-pinned-memory
  --dont-upcast-attention --disable-dynamic-vram` (Grace-Blackwell-optimized ComfyUI variants exist).
  Stop the LLM during heavy ComfyUI gen — single-tenant GPU.

## See also
`[[wiki/engines.md]]` · `[[wiki/quantization-on-gb10.md]]` · `[[wiki/platform-gb10.md]]`

## Forum ingest: community tools & images (2026-07-08)

- **[conjecture]** **Docker image: NVIDIA vLLM 0.23.0 with Claude Code compatibility**
  (S-forum-vllm-claude): community Docker image that updates NVIDIA's official 26.05.post1 vLLM
  image to work with Claude Code 2.1.195+.
- **[conjecture]** **btop for DGX Spark** (S-forum-btop): modified btop fork that displays GPU
  resource info on GB10 (stock btop doesn't show unified memory GPU stats).
- **[conjecture]** **DGX Spark Model Manager** (S-forum-model-manager): open-source web UI for
  managing Ollama, SGLang & LiteLLM models on Spark — single browser tab to control everything.
- **[conjecture]** **sparkdash** (S-forum-sparkdash): monitoring/control dashboard for sparkrun
  DGX Spark clusters — Ray/vLLM/recipe status and per-node vitals.
- **[conjecture]** **Tool Eval Bench CLI** (S-forum-tool-eval): benchmark tool for evaluating model
  performance on DGX Spark — complements llama-benchy with tool-calling/task-based metrics.
- **[conjecture]** **ThunderKittens 2.0** (S-forum-thunderkittens): tile primitives library with
  Blackwell support — useful for custom kernel development on sm_121.

### Batch 3 forum ingest (2026-07-09)

- **[conjecture]** **Vitoom Nunchaku for DGX Spark** (S-forum-flux2-nunchaku, tonera): optimized
  image inference library for DGX Spark. Flux.2 Klein 9B: 2.5× faster inference (10s→4s for 8 steps),
  59% lower peak VRAM (37.14→15.21 GB) with quantized transformer + text encoder. `pretouch` improves
  model load time 15.6× (249s→16s). Wheel: `tonera/vitoom-nunchaku` on HuggingFace. Also supports
  Qwen-Image, Chroma1-HD, SVDQ.
- **[conjecture]** **ComfyUI container for DGX Spark** (S-forum-comfyui-container, martial):
  `ComfyUI-Nvidia-Docker` with SageAttention2+3, ONNX Runtime, uid/gid config, Comfy Kitchen (fp16→
  NVFP4 conversion). Prebuilt images on Docker Hub with `compose-dgx_spark.yaml`. Requires
  userscript_dir setup before first start.
- **[conjecture]** **llama.cpp container build for Spark/GB10** (S-forum-llamacpp-container, cosinus):
  `nvidia/cuda:13.0.2-devel-ubuntu24.04` base needs `LD_LIBRARY_PATH=/usr/local/cuda-13/compat` (not
  the default `/usr/local/cuda/lib64` — linker can't find `libcuda.so.1`). CMAKE_CUDA_ARCHITECTURES
  must be set explicitly (`121a-real`) since Docker build has no GPU access. Community Dockerfile:
  stelterlab gist.
- **[conjecture]** **nvcr.io/nvidia/vllm:26.06-py3 broken** (S-forum-vllm-2606-broken): every API
  request returns HTTP 500 (`'_IncludedRouter' has no attribute 'path'`). `prometheus-fastapi-instrumentator`
  incompatible with `fastapi >= 0.137`. Use 26.02-py3 instead. See `[[wiki/platform-gb10.md]]`.
- **[conjecture]** **Gemma4 official QAT models** (S-forum-gemma4-qat, jwarner): Google released
  official Quantization Aware Training (QAT) versions including W4A16:
  `google/gemma-4-31B-it-qat-w4a16-ct` (23.3 GB, 4 GB larger than Intel AutoRound but potentially
  better fidelity). The 26B-A4B has no official W4A16 but has an unquantized version for custom
  quantization. QAT models work with Gemma4 MTP assistants.
- **[conjecture]** **Mistral-Small-4-119B-2603-NVFP4 OOM** (S-forum-mistral-s4-nvfp4): OOM during
  safetensors parse on 2× GB10 with `gpu_memory_utilization: 0.8`. Fix: bump to 0.9 + enable swap.
  Single-node NVFP4 variant reportedly works fine.
- **[conjecture]** **torchaudio unavailable on ARM64/CUDA 13** (S-forum-qwen-tts-arm64): no
  ABI-compatible wheel for DGX Spark — blocks Qwen3-TTS and audio models. `torchaudio` deprecated,
  not in NGC containers. Workaround: use PyTorch from pytorch.org. See `[[wiki/platform-gb10.md]]`.

### Batch 4 forum ingest (2026-07-10)

- **[conjecture]** **llama-benchy** (S-forum-llama-benchy, eugr): CLI benchmarking tool that brings
  llama-bench-style context-depth sweep measurements to ANY OpenAI-compatible endpoint (vLLM,
  SGLang, llama.cpp, etc.). Measures prompt processing (pp) and token generation (tg) speeds at
  different context depths (`--depth`), concurrency (`--concurrency`), configurable prompt/gen
  lengths, multiple iterations with mean ± std. Uses HuggingFace tokenizers for accurate token
  counts. Downloads a Project Gutenberg book as source text (important for spec-decode/MTP
  benchmarking — random tokens don't exercise draft acceptance properly). JSON/CSV output.
  Available via `uvx llama-benchy`. Demo numbers on dual Spark cluster:
  - MiniMax-M2.1-AWQ-4bit (2× Spark, 100K ctx): pp2048 ~3544 t/s, tg32 ~36 t/s; degrades with depth
    (tg32 @ 100K not shown but pp drops from ~3544 to ~2832 @ 8K).
  - GLM-4.7-Flash-AWQ-4bit (1× Spark, util 0.7, max_model_len 202752): pp2048 ~5326 t/s c1,
    tg32 ~41.75 t/s c1; c2 tg32 aggregate 73.74 (37.38/req); c10 tg32 aggregate 87.65 (15.33/req).
    KV cache 1,239,088 tokens, max concurrency 6.11× for 202K tokens/req.
- **[conjecture]** **DGX Spark Cluster Dashboard** (S-forum-cluster-dashboard, paul.aviles):
  web-based btop-inspired dashboard for multi-node Spark monitoring — replaces running separate
  SSH/btop sessions on each node. GitHub: paul-aviles/NVIDIA-DGX-Spark-Dashboard. Runs on a
  separate management system (not on the Sparks themselves). Note: enP7s7 utilization bar may
  show full when interface not at 100% (known display bug).
- **[conjecture]** **Headless Sunshine remote desktop for DGX Spark** (S-forum-sunshine-rdp):
  community setups for extending Spark beyond SSH/CLI to native desktop via Sunshine+Moonlight
  streaming. Repos: eelbaz/dgx-spark-headless-sunshine (automated setup),
  seanGSISG/dgx-spark-sunshine-setup (4K variant). GB10 display controller 165 MHz pixel clock
  limits resolution (see `[[wiki/platform-gb10.md]]`).
- **[conjecture]** **FLUX.2-dev as headless OpenAI Images-API server** (S-forum-flux2-nvfp4-compute,
  vr8vr8): first image-generation model added to eugr/spark-vllm-docker (PR #313). Uses torchao
  NVFP4 W4A4 on-the-fly quantization for real FP4 compute (~3× speedup over BF16, ~66 GB VRAM).
  See `[[wiki/quantization-on-gb10.md]]` for the quant details.
- **[conjecture]** **`DIFFUSERS_ATTN_BACKEND=_native_cudnn`** env var is a significant GB10
  diffusion-model speedup (S-forum-diffusion-speeds, ijontichy): Krea2-Turbo 39.3→13.9s,
  ERNIE-Image-Turbo 11.2→8.8s, no effect on Z-Image-Turbo. Combined with
  `torch.set_float32_matmul_precision('high')`. All via `diffusers` library (not ComfyUI).
  Second source (vasimv): 15-17s for Krea2-Turbo FP16 on ComfyUI + 610 drivers + CUDA 13.3.
- **[conjecture]** **Image diffusion model benchmarks on GB10** (S-forum-diffusion-speeds): single-node,
  1024×1024, `diffusers` library, post torch-compile. FLUX.2-klein-9B (4 steps) 4.4s → 3.3s NVFP4;
  Z-Image-Turbo (9 steps) 7.2s → 5.6s NVFP4; SDXL 1.0 (30 steps) 11.3s; Qwen-Image-2512 (50 steps) 61s;
  ERNIE-Image-Turbo (8 steps) 11.2→8.8s (cudnn attn) → 6.4s NVFP4; Krea2-Turbo (8 steps) 39.3→13.9s
  (optimized) → 12.4s NVFP4. See `[[wiki/benchmarks.md]]` for full table.

 ### Batch 10 forum ingest (2026-07-13)

 - **[reported]** **No nvidia-vfx (Maxine VFX SDK) aarch64 wheel for DGX Spark** (S-forum-nvidia-vfx):
 ComfyUI's NVIDIA RTX upscale node depends on `nvidia-vfx`, which only ships x86 wheels — the
 package cannot be pip-installed on GB10 (aarch64). The NVIDIA VFX / Maxine VFX SDK supported GPU
 list (A40/L40/L4/A30/B200/A2/H100/A10/T4/B100/A16/A100/B40) **does not include GB10**.
 NVIDIA (aniculescu, official) confirmed: **"There is currently no plan to add NVIDIA VFX support
 on Spark."** Multiple community users requested aarch64 wheels and source access for
 self-compilation — no response on either. **Status:** `open` — ComfyUI RTX upscaler nodes are
 broken on DGX Spark with no fix path. This is a broader pattern of aarch64 wheel gaps on GB10
 (cf. torchaudio, S-forum-qwen-tts-arm64).

 ### Batch 11 forum ingest (2026-07-13)

 - **[conjecture]** **easy-vllm code-agent harness** (S-forum-easy-vllm, sh.ahn): open-source
   Claude Code-based meta-harness that automates vLLM build → serve → adversarially verify →
   self-improve loops on DGX Spark. Uses deterministic scripts for version resolution, VRAM
   estimation, KV-clamp math (not probabilistic LLM guessing). Enforces HW homogeneity before
   multi-node (mixed clusters intentionally unsupported). Includes optional `mem_watchdog` +
   `earlyoom` host safety stack (kills container before UMA OOM = whole-host down). GitHub:
   `tbvjvsladla/easy_vllm_simulator`. See `[[wiki/engines.md]]` for the DSV4-Flash bring-up
   findings from this tool.

### Batch 12 forum ingest (2026-07-14)

- **[conjecture]** **Spark Studio** (S-forum-spark-studio, TheAwakenOne): open-source inference
  dashboard purpose-built for DGX Spark — MIT licensed. One-command install, launches
  vLLM/SGLang/llama.cpp/sparkrun recipes from a web UI. Key GB10-specific features: **live
  unified-memory monitor** (critical for UMA — models, background processes, and KV cache all
  share the same pool); **pre-launch memory guard** that stops models, waits for reclaim, and
  refuses launches that won't fit (prevents the OOM-crash cycle on 128 GB setups); **agent
  auto-fix** loop using local Claude Code / Codex CLIs — logs, patches, retries automatically
  (uses your own subscription, no extra setup); **Optimize Speed** strictly measurement-based
  (≥10% or it rolls back); multi-node cluster view with no node limit. Recipes pulled from
  sparkrun community recipes and spark-arena. GitHub: `TheAwaken1/Spark-Studio`.

### Batch 22 forum ingest (2026-07-19)

- **[conjecture]** **harinezumigel-llm-stack — LiteLLM + NVIDIA vLLM Docker orchestrator for
  single-Spark multi-model management** (S-forum-litellm-orchestrator, HarinezumIgel): a thin
  management layer around LiteLLM and vLLM for users who need to switch between models
  (inference, prompt guard, coding, RAG) on one Spark where memory precludes running them all
  simultaneously. Defines models once in `config.yaml` + `.env`, starts/stops model containers
  consistently, reuses existing containers, exposes a single OpenAI-compatible LiteLLM
  endpoint, keeps secrets separate from model params. Not a production orchestrator — a
  convenience tool for local inference workflows on GB10 where `--gpu-memory-utilization`
  makes co-hosting impractical. Thread also surfaces two related multi-model management tools:
  **Spark Studio** (TheAwakenOne, already registered S-forum-spark-studio) and
  **sparkstation** (`kshetrajna12/sparkstation` — unified LLM orchestration/gateway for
  vLLM, SGLang, and TensorRT-LLM backends under a single OpenAI-compatible API). Reinforces
  the existing [proven] single-tenant-per-node constraint: multi-model on one Spark is
  lifecycle-management (start/stop/swap), not co-residency.
- **[conjecture]** **Nemo-RT Community — real-time bilingual ES/EN voice agent co-located on
  one GPU, OpenAI Realtime API-compatible** (S-forum-nemo-rt, InfinitoCloud): full pipeline
  (VAD → STT NeMo Conformer → LLM Qwen3-8B-FP8 via vLLM → TTS NeMo FastPitch + HiFi-GAN)
  on a single GPU, speaks the OpenAI Realtime API protocol (drop-in `ws://your-box:8000/v1/realtime`
  for `wss://api.openai.com/v1/realtime`). Ships an Asterisk/SIP bridge (ARI + external-media
  RTP) validated on a live call. **On DGX Spark (GB10, 128 GB unified):** ~20 concurrent
  calls, sub-second TTFA. Rationale: 128 GB unified means memory stops being the ceiling on
  concurrent sessions; GB10 has **native FP8** (which the default Qwen3-8B-FP8 model wants);
  arm64 build means no cross-compile. Apache-2.0. One caveat: the measured Spark was already
  provisioned, so the one-command `setup.sh` hasn't been exercised against a fresh Spark OS.
  **GB10-relevant bits:** native-FP8 path for the LLM stage, vLLM as the LLM backend, unified
  memory as the concurrency enabler. Reference perf (RTX 4090, 24 GB): full stack ~21.5 GB,
  live voice TTFA 0.17–0.59 s, LLM 52 tok/s single-stream. Single source → [conjecture].
  GitHub: `infinitocloud/nemo-rt-community`.

### Batch 16 forum ingest (2026-07-16)

- **[conjecture]** **ComfyUI Docker optimized for DGX Spark** (S-forum-comfyui-optimized,
  luix93): a community Docker setup targeting ComfyUI on GB10. Key GB10-specific features:
  - **CUDA 13.1 base** with full `nvcc` support for sm_121 (enables CUDA extension compilation).
  - **PyTorch cu130** prebuilt ARM64 wheels from PyTorch's cu130 index.
  - **SageAttention 2** compiled from source directly against sm_121 for full hardware
    attention acceleration (cf. S-forum-sage-attn — the silent-fallback risk).
  - **Comfy Kitchen** (`comfy_kitchen`) for NVFP4 quantization support on Blackwell.
  - **`--disable-dynamic-vram`** — ComfyUI's dynamic VRAM management doesn't work properly on
    the Spark; with it disabled, models that fit in memory stay resident (faster prompt-to-image).
  - **Double-VRAM bug fix** — patches `comfy/utils.py` to set `copy=False` in `tensor.to()`,
    fixing double memory usage on unified memory systems with `--disable-mmap`. On UMA,
    ComfyUI's default `copy=True` duplicates tensor data in the same physical pool.
  - **`CUDA_MODULE_LOADING`** — a typo in the original compose (`CUDA` is not a valid value)
    silently defaulted to `LAZY`, which performed better than `EAGER` in testing (S-forum-comfyui-
    optimized, AoE). LAZY module loading is the accidental winner on GB10.
  - Repo: `luix93/DGX-Spark-ComfyUI`. Multiple users confirm stability improvements (Schordan,
    Zhiqing Yu Cn). ComfyUI v0.27 broke fp8/fp4 precision support — Comfy Kitchen needs
    updating to 0.2.61 (report from post #10, status open).
- **[conjecture]** **`cudaMemGetInfo` under-reports free memory on unified memory when a
  co-resident CUDA process holds allocation** (S-forum-comfyui-optimized, Haidij):
  **Symptom:** ComfyUI alone runs fine (~19-25s/image at 1080×1350, 5 steps, Flux2). With vLLM
  co-resident (34 GB bound, `gpu_memory_utilization=0.3`), every ComfyUI job shows text encoder
  *partially offloaded* and per-image time grows from ~20s to 100-330s. `/system_stats`
  reveals: `vram_free: 6.2 GB` (what CUDA reports) vs `ram_free: 46.3 GB` (what the host
  actually has free).
  **Root cause:** `comfy/model_management.py::get_free_memory()` uses
  `torch.cuda.mem_get_info()` (wrapping `cudaMemGetInfo`) to decide whether to keep a model
  resident vs offload. On unified-memory systems, `cudaMemGetInfo` reports only memory **not
  currently allocated by any CUDA process** on the same device — not the true free pool. When
  vLLM has 34 GB bound, `cudaMemGetInfo` returns ~6 GB free even though 40+ GB of unified
  memory is actually available. ComfyUI then offloads a ~7.7 GB text encoder to "CPU" (which
  on UMA is the *same physical RAM* it just decided it couldn't keep on GPU), paying the
  partial-offload penalty every forward pass.
  **Fix (verified by testing):** replace the CUDA memory query with
  `psutil.virtual_memory().available` in the CUDA branch of `get_free_memory()`. On GB10
  these are semantically the same pool; psutil's view correctly accounts for system reality
  including other processes' allocations. Patch (Dockerfile RUN):
  ```python
  # Replace: mem_free_cuda, _ = torch.cuda.mem_get_info(dev)
  # With:    import psutil as _psutil; mem_free_cuda = _psutil.virtual_memory().available
  ```
  This is a GB10-specific UMA finding that generalizes beyond ComfyUI — any application using
  `cudaMemGetInfo` to make memory decisions will under-report free memory when another CUDA
  process is resident on the same unified device. See `[[wiki/platform-gb10.md]]`.

### Batch 26 forum ingest (2026-07-21)

- **[conjecture]** **Community Spark Docker images lag behind upstream vLLM/NCCL releases**
  (S-forum-vllm025-nccl, Hunlx): as of 2026-07-19, standard community Spark Docker images (eugr/
  spark-vllm, sparkrun-vllm-ds4-gb10, vllm/vllm-openai:nightly, etc.) ship **vLLM 0.23** and an
  **old NCCL**, not the latest vLLM 0.25.1 or NCCL 2.30.7. The latest versions are needed for
  **pipeline parallel support for MiniMax M3** and **NCCL mesh networking** improvements. A GitHub
  image with vLLM 0.25.1 + NCCL 2.30.7 was found but not from the standard Spark image ecosystem.
  This is a known gap in the Spark container ecosystem: community images are built for stability
  with specific model recipes, not bleeding-edge upstream. Users needing latest vLLM features
  must either build their own image or find non-standard builds. Single source → [conjecture].

### Batch 28 forum ingest (2026-07-22)

- **[conjecture]** **sparkDash (MiaAI-Lab) — open-source multi-DGX Spark monitoring dashboard**
  (S-forum-sparkdash-mia, MiaAI_Lab): a second independent community dashboard (distinct from
  the earlier sparkdash by brainchillz, S-forum-sparkdash). Features: live overview of multiple
  Sparks in one browser tab (GPU/CPU/unified memory/storage/network), local LLM status for
  llama.cpp/vLLM/sglang with tok/s when a server is up, add/edit/remove Sparks from the UI (local
  + remote over SSH), power controls (graceful shutdown over SSH + Wake-on-LAN with
  auto-detected MAC), optional "Worker node" flag to hide local LLM panels for distributed-only
  workers. Intended for trusted LAN (no built-in auth — put behind your own proxy). GitHub:
  `MiaAI-Lab/sparkDash`. Single source → [conjecture]. Reinforces the existing pattern of
  community-built multi-Spark dashboards (cf. S-forum-sparkdash, S-forum-cluster-dashboard,
  S-forum-spark-studio).
