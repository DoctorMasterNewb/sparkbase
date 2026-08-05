# Container images & tooling

> **area:** containers
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-forum-vllm-claude, S-forum-btop, S-forum-model-manager, S-forum-sparkdash, S-forum-tool-eval, S-forum-thunderkittens, S-forum-driver610, S-forum-flux2-nunchaku, S-forum-comfyui-container, S-forum-llamacpp-container, S-forum-sage-attn, S-forum-vllm-2606-broken, S-forum-gemma4-qat, S-forum-mistral-s4-119b, S-forum-qwen-tts-arm64, S-forum-llama-benchy, S-forum-cluster-dashboard, S-forum-sunshine-rdp, S-forum-flux2-nvfp4-compute, S-forum-nvidia-vfx, S-forum-easy-vllm, S-forum-spark-studio, S-forum-comfyui-optimized, S-forum-litellm-orchestrator, S-forum-nemo-rt, S-forum-vllm025-nccl, S-forum-sparkdash-mia, S-forum-spark-vllm-rebuild, S-forum-vllm-containers, S-forum-qwen3tts-ggml, S-forum-vllm-stock-hang, S-forum-locateanything, S-forum-sparkctl, S-forum-whisper-docker, S-forum-llamacpp-fastest, S-forum-comfyui-crash, S-forum-cuda-mps, S-forum-model-storage, S-forum-acer-thermal, S-forum-vllm-2607-xgrammar, S-forum-depfree-dashboard, S-forum-comfyui-triplany, S-forum-acestep-v15-comfyui
> **updated:** 2026-08-05

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

### Batch 31 forum ingest (2026-07-23)

- **[conjecture]** **spark-vllm-docker build flags** (S-forum-spark-vllm-rebuild,
  elvis.dowson/eugr): `./build-and-copy.sh --rebuild-vllm` forces a local vLLM rebuild
  instead of pulling the pre-built Docker image; `--use-wheels` uses prebuilt wheels
  instead of compiling vLLM from source. The repo always builds from `main` — there is
  no pinned vLLM version tag, so images track vLLM HEAD at build time. Useful for users
  who need to force a fresh build or avoid source compilation. Single source →
  [conjecture]. See also `[[wiki/models/mistral-small-4.md]]` for the spark-vllm-docker
  base image description.

- **[conjecture]** **NGC vs community vLLM containers on Spark** (S-forum-vllm-containers,
  eugr, joshua.dale.warner): the NVIDIA NGC container lags ~2 versions behind the community
  spark-vllm-docker. Since Spark-specific optimizations (FlashInfer, CUTLASS MoE, Marlin)
  are not yet fully merged in mainline vLLM, the NGC container won't run as well as the
  community version. The community docker downloads pre-built FlashInfer and vLLM wheels by
  default (built nightly on a Spark cluster, run through regression testing — if degradation
  detected, the build fails). `--vllm-ref <ref>` builds from source using the specified ref
  even without `--rebuild-vllm`. For multi-container co-hosting, use `--name <container_name>`
  to avoid container name conflicts. Single source → [conjecture].

- **[conjecture]** **VRAM is soldered, only SSD is replaceable** (S-forum-vllm-containers,
  eugr): confirmed that GB10 (V)RAM cannot be upgraded — it's soldered. Only the SSD can be
  replaced. Relevant for users asking about hardware upgrades for larger models. [conjecture]
  (single forum source, but consistent with known GB10 hardware specs).

### Batch 33 forum ingest (2026-07-24)

- **[conjecture]** **Qwen3-TTS on DGX Spark — force `torch` backend to bypass GGML CUDA crash**
  (S-forum-qwen3tts-ggml, swann.schilling): the default GGML backend in `faster-qwen3-tts[ggml]`
  crashes on first inference on GB10 due to CUDA 12.8 / sm_120 kernel mismatch (see
  `[[wiki/platform-gb10.md]]` for the root-cause analysis). The fix is to use the `torch` backend:
  (1) drop the `[ggml]` extra in `pyproject.toml` (`"faster-qwen3-tts>=0.3.2"` instead of
  `"faster-qwen3-tts[ggml]>=0.3.2"`); (2) force the backend in docker-compose:
  `--qwen3_tts_backend torch`; (3) rebuild. The torch backend uses CUDA-graph-accelerated
  PyTorch (no GGML, no Flash Attention), same approach proven working in
  `martinb78/faster-qwen3-tts-dgx-spark`. Confirmed working: CUDA graph capture succeeds,
  **TTFA 2.65s**, first request RTF 0.54 (includes one-time graph capture), **steady-state
  RTF ~1.7** (faster than real-time). Additional tip (Drew_the_AI_Guy): on GB10 watch CPU-GPU
  migration for audio tensors — UMA helps but Qwen3-TTS chunks can end up CPU-bound if the torch
  backend doesn't pin input buffers; `torch.cuda.synchronize()` and `non_blocking=True` on
  tensor moves matter. This complements the existing torchaudio ARM64 gap
  (S-forum-qwen-tts-arm64) — both are GB10 audio-stack issues, but the GGML crash is a
  different failure mode (kernel dispatch, not wheel availability). Single source + one
  corroborating reply → [conjecture].

### Batch 34 forum ingest (2026-07-25)

- **[conjecture]** **Stock `vllm/vllm-openai:latest` hangs silently during model load on GB10 —
  no SM121 support** (S-forum-vllm-stock-hang, dotrantrung2003, Drew_the_AI_Guy): on an ASUS
  Ascent GX10 (GB10), serving `nvidia/Qwen3.6-35B-A3B-NVFP4` with the upstream
  `vllm/vllm-openai:latest` Docker image, the container starts and logs reach backend
  selection (`FLASHINFER` attention, `FlashInferFP8ScaledMMLinearKernel`, `MARLIN` NvFp4 MoE)
  but never progress to "Application startup complete." All API requests return
  `curl: (56) Recv failure: Connection reset by peer`. The container stays `Up` (it doesn't
  crash) — it simply hangs during initialization. **Root cause:** the upstream
  `vllm/vllm-openai` image does not include Blackwell/SM121 support out of the box. This
  corroborates the existing pattern documented across multiple batches: stock upstream
  vLLM images lack sm_121/CUDA 13 support; use a GB10-tuned build
  (`eugr/spark-vllm-docker` with `--tf5`, or a vLLM wheel built for CUDA 13 / SM121). The
  user's config flags (`--kv-cache-dtype fp8 --attention-backend flashinfer --moe-backend
  marlin --gpu-memory-utilization 0.7 --max-model-len 262144 --max-num-seqs 4
  --max-num-batched-tokens 8192 --enable-chunked-prefill --async-scheduling
  --enable-prefix-caching --speculative-config '{"method":"mtp",...}'
  --load-format fastsafetensors --reasoning-parser qwen3 --tool-call-parser qwen3_xml
  --enable-auto-tool-choice`) are a reasonable NVFP4+MTP recipe — the failure is the image,
  not the flags. Single source (OP resolved after switching to a GB10-tuned build; the
  fix was confirmed but the specific image used wasn't stated). Reinforces the existing
  `[conjecture]` community-image-lag finding (S-forum-vllm025-nccl) and the easy-vllm
  harness's identification of stock-vLLM-on-sm_121 as a "double hard wall"
  (S-forum-easy-vllm).

- **[conjecture]** **LocateAnything-3B bring-up on DGX Spark — ARM64 wheel gaps and
  `device_map='auto'` UMA pitfall** (S-forum-locateanything, swann.schilling):
  `nvidia/LocateAnything-3B` (visual grounding, not a vLLM-served text model) deployed as a
  standalone FastAPI server inside `vllm-node-tf5` (eugr/spark-vllm-docker `--tf5` build,
  CUDA 13 / SM121). Four GB10-specific bring-up findings:
  1. **`decord` has no ARM64 wheel** — transformers' `check_imports` statically scans for
     it before any code runs, so a `sys.modules` stub doesn't work. Fix: build a minimal
     local stub package (`VideoReader` no-op class + `setup.py`) and `pip install` it.
  2. **`deepspeed`, `bitsandbytes`, `liger_kernel` have no ARM64 wheels** — either no
     wheel or compilation fails on GB10. Since this is inference-only, install deps
     manually with `pip install --no-deps -e .` and skip the unneeded training-only
     packages. This corroborates the broader aarch64 wheel gap pattern on GB10
     (cf. torchaudio S-forum-qwen-tts-arm64, nvidia-vfx S-forum-nvidia-vfx, GGML
     qwentts-cpp S-forum-qwen3tts-ggml).
  3. **`device_map='auto'` is very slow on 128 GB unified memory** — runs a metadata
     analysis pass that can appear frozen for many minutes. Fix: use `.to(device)`
     directly (loads from cache in <1 s on GB10). Related to the existing
     `[conjecture]` UMA mmap double-allocation finding (S-forum-qwen35-lora-uma) — both
     are UMA-specific pitfalls in HuggingFace's device-mapping logic.
  4. **MoonViT sub-model (`moonshotai/MoonViT-SO-400M`) downloaded separately from HF
     Hub** — without authentication, the download hangs silently on rate limiting. Pass
     `HF_TOKEN` via both `-e HF_TOKEN=` and `-e HUGGING_FACE_HUB_TOKEN=` env vars.
  The bring-up also documents the pattern for non-vLLM models on Spark: use
  `--entrypoint /bin/bash` + inline `git clone` + `pip install` in the `docker run`
  command, expose a task-specific REST API (not `/v1/chat/completions`), and use
  `--shm-size=16g --ipc=host`. `vllm-node-tf5` is confirmed as a known-good base image for
  non-vLLM workloads on DGX Spark / ThinkStation PGX. Single source → [conjecture].

### Batch 37 forum ingest (2026-07-27)

- **[conjecture]** **sparkctl — config-driven model serving CLI for DGX Spark nodes and clusters**
  (S-forum-sparkctl, bradodarb): a CLI and orchestration layer for managing model serving on
  single Sparks and Spark clusters. Config-driven (YAML), multi-provider — supports vLLM,
  Ollama, llama.cpp, and other serving backends. Key differentiators vs sparkrun: (1) assumes
  networking is already completed (no bootstrapping/SSH mesh — sparkrun does that); (2)
  k8s/devops-inspired reproducible deployments via YAML configs; (3) load balancing for
  clusters where the same model is deployed across several nodes (API gateway spreads load
  evenly); (4) contextual data plane (metrics, LiteLLM proxy) that can run on host, node, or
  k8s cluster. Tutorial at bradmurry.com. Community feedback (mrDragonFox): "you pretty much
  reinvented sparkrun" — overlap acknowledged, but sparkctl targets users wanting config-driven
  multi-provider deployments rather than sparkrun's bootstrap-and-serve model. GitHub:
  `bradodarb/sparkctl`. Single source → [conjecture]. Reinforces the pattern of community-built
  Spark orchestration tools (cf. sparkrun, Spark Studio, harinezumigel-llm-stack).

### Batch 39 forum ingest (2026-07-28)

- **[conjecture]** **whisper.cpp STT server on DGX Spark via Docker** (S-forum-whisper-docker,
  swann.schilling): a complete Docker recipe for running whisper.cpp v1.8.4 as an STT server
  on GB10. Several GB10-specific build gotchas:
  1. **No pre-built ARM64+CUDA binaries** in whisper.cpp releases — must build from source
     inside Docker.
  2. **Use `docker.io/nvidia/cuda`, not `nvcr.io/nvidia/cuda`** — nvcr.io has no ARM64 tags.
     Base: `nvidia/cuda:13.0.3-devel-ubuntu24.04`.
  3. **Ubuntu 24.04, not 22.04** — DGX OS ships GLIBC 2.38; Ubuntu 22.04 containers only have
     GLIBC 2.35 and fail with `version GLIBC_2.38 not found`.
  4. **`CMAKE_CUDA_ARCHITECTURES="120;121"`** — must target both 120 and 121. Using only 120
     compiles for `sm_120a` which is not compatible with GB10 (sm_121). This is the same
     arch-targeting gap documented for vLLM/llama.cpp builds (cf. S-forum-llamacpp-container
     `121a-real`, S-forum-sm121-kernel-guide).
  5. **CUDA stubs for linking** — `libcuda.so.1` is a driver library not available at image
     build time. Use `-DCMAKE_EXE_LINKER_FLAGS="-L/usr/local/cuda/lib64/stubs -lcuda"`.
  6. **Only build `whisper-server` target** — building everything fails because `libcuda.so.1`
     is unavailable at build time.
  7. **`deploy.resources` GPU access, not `runtime: nvidia`** — `runtime: nvidia` throws
     `unknown or invalid runtime name: nvidia` on DGX OS. Use the `deploy.resources.reservations.
     devices` style with `driver: nvidia, count: all, capabilities: [gpu]`.
  Model: `ggml-large-v3-turbo.bin` (1623.92 MB, 4-layer decoder). Flash attention enabled
  (`--flash-attn`). Confirmed: `Device 0: NVIDIA GB10, compute capability 12.1, VMM: yes`.
  An alternative approach (ajvazan) uses `mekopa/whisperx-blackwell` (WhisperX built for
  SM_121) via faster-whisper with a FastAPI wrapper — `COMPUTE_TYPE=float16`, OpenAI-compatible
  `/v1/audio/transcriptions` endpoint. Single source + one corroborating reply → [conjecture].
  Reinforces the recurring ARM64 wheel/binary gap pattern on GB10 (cf. torchaudio, nvidia-vfx,
  decord, GGML/qwentts — all require building from source with explicit sm_121 targeting).

- **[conjecture]** **Official llama.cpp Docker image matches custom builds on GB10; `--mmap 0`
  mandatory on UMA** (S-forum-llamacpp-fastest, pontostroy/knitvoger1): the official
  `ghcr.io/ggml-org/llama.cpp:full-cuda13` image performs identically to a custom-optimized
  build on DGX Spark — both achieve **72.28 tok/s tg128** (Qwen3.5-35B-A3B NVFP4,
  `s-batman/Agents-A1-NVFP4-MTP-GGUF`, 19.84 GiB, 35.51B params). Key findings:
  - **`--mmap 0` is mandatory on UMA** — without it, mmap and CUDA compete for the same
    unified pool (same pattern as the ComfyUI double-VRAM bug, S-forum-comfyui-optimized).
  - **`-fa 1` (flash attention)** enabled in all runs.
  - **Performance degradation 40→67 tok/s fixed by system update** — a user getting 40 tok/s
    (vs the expected ~72) was advised to power-cycle the Spark (unplug power adapter 3–5 min,
    the known power-controller wedge fix, see `[[wiki/platform-gb10.md]]`). After a system
    update + reboot, speed jumped to 67 tok/s. This corroborates the existing [reported]
    power-controller wedge pattern: unexplained performance degradation → power-cycle.
  - **scitrera/dgx-spark-llama-cpp image is slower** (30 tok/s vs 72 tok/s) — community image
    not optimized for the current CUDA 13 / sm_121 stack.
  - Vulkan backend loaded alongside CUDA (NV_coopmat2 matrix cores) but CUDA is the primary
    compute path.
  Single source (two users in one thread) → [conjecture]. Corroborates the power-controller
  wedge pattern documented across multiple batches.

### Batch 40 forum ingest (2026-07-29)

- **[conjecture]** **ComfyUI hard-crash fix on DGX Spark — clock cap + swapoff + async offload**
  (S-forum-comfyui-crash, jas.burton): a detailed diagnosis and fix for ComfyUI causing hard
  system reboots on GB10. The root cause is a **GPU power spike** (14→85 W instantaneous) tripping
  overcurrent protection, not thermal or OOM. See `[[wiki/platform-gb10.md]]` for the full
  platform-level finding. ComfyUI-specific takeaways:
  - **Working flags:** `python main.py --listen 0.0.0.0 --bf16-unet --bf16-vae --bf16-text-enc
    --use-sage-attention` — no `--highvram`, no `--disable-async-offload`, no `--gpu-only`.
    On UMA, ComfyUI's async weight offloader is near-free (pointer update, not a real
    copy). `--highvram` forces all models pinned simultaneously → OOM with multi-model stacks.
  - **`CUDA_CACHE_MAXSIZE=4294967296`** (4 GB) — 3× rerun speedup from larger PTX→SASS cache.
  - **`NCCL_P2P_DISABLE=1`** — single GPU, skip NCCL overhead.
  - **Avoid:** `CUDA_CACHE_DISABLE=1` (kills kernel cache, 3× slower reruns),
    `PYTORCH_NO_CUDA_MEMORY_CACHING=1` (causes fragmentation → OOM).
  - Full stack demonstrated: ComfyUI (LTX Video) + llama-server (Qwen3-VL-8B Q6_K) + FastAPI
    web app on one Spark, peak ~93 GB / 119 GB, 68-74 °C, 45-51 W with clock cap — stable.
  - A second user (frozenace88) confirms `nvidia-smi -lgc 300,2100` stabilized their system.
    A third (knitvoger1) reports the lock command succeeds but the clock still shows 2418 MHz
    — may not take effect on all firmware versions. See `[[wiki/platform-gb10.md]]`.
  - **ComfyUI has no multi-GPU/multi-node support** (gpieceoffice, same thread): the
    `worksplit-multigpu` branch loads models on each GPU in parallel (not split), abandoned
    ~end of 2025. Multi-node model loading not feasible in ComfyUI.
  Reinforces existing ComfyUI UMA findings (S-forum-comfyui-optimized, S-forum-comfyui-container).

- **[conjecture]** **CUDA MPS for multiple vLLM instances on single DGX Spark**
  (S-forum-cuda-mps, shahizat): an experiment running 2+ vLLM servers on one GB10 via CUDA
  Multi-Process Service. MPS allows multiple CUDA processes to share scheduling resources,
  reducing context-switch overhead. Setup:
  1. `sudo nvidia-smi -i 0 -c EXCLUSIVE_PROCESS` (set compute mode)
  2. `mkdir -p /tmp/nvidia-mps /var/log/nvidia-mps`
  3. `export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps CUDA_MPS_LOG_DIRECTORY=/var/log/nvidia-mps`
  4. `nvidia-cuda-mps-control -d` (start MPS controller daemon)
  5. Each vLLM instance: `--gpu-memory-utilization 0.45` (split the UMA pool)
  **Result:** latency increased significantly, throughput improved modestly. The main advantage
  is serving **multiple independent models** on the same GPU (e.g. Qwen3.5-4B + another model),
  not raw performance. This is an alternative to the lifecycle-swap pattern (start/stop one
  model at a time) — MPS enables true co-residency at the cost of per-instance latency. A
  second user (MadsRotwitt) asked about MPS with Docker/NGC vLLM containers — not addressed in
  the thread. Single source → [conjecture]. Relevant to the existing [proven] single-tenant
  constraint: MPS is a potential workaround but trades latency for co-residency. See
  `[[wiki/platform-gb10.md]]` → operating constraints, `[[wiki/engines.md]]`.

## Model storage strategies (2026-07-30 ingest)

- **[conjecture]** **Model storage is a first-class problem on 1TB Sparks**
  (S-forum-model-storage, starkrun et al.): `/var/lib/docker` alone can reach
  ~390 GB (images + build cache, no models). LLM models are 170 GB each
  (DSV4-Flash, MiMo 2.5, Hy3). Community approaches on single and dual Sparks:
  - **4TB NVMe upgrade** (Corsair MP700 Micro Gen5 2242): danielgbates and
    robert287 both swapped the internal 1TB for a 4TB Gen5 NVMe — the most
    straightforward solution. Custom inference swapping script stages/unstages
    models between nodes + starts/stops vLLM containers.
  - **NFS share over 10GbE**: FlossingEnthusiast reports ~1.1 GB/s from a
    UGREEN DXP480T Plus all-flash NAS; ajvazan provides a complete NFS fstab
    + rsync backup script pattern (`/etc/fstab` with `rsize=1048576,
    wsize=1048576, x-systemd.automount`); domrockt uses Unraid NAS over 2.5GbE.
    See also the existing NFS model-share source S-forum-nfs-modelshare
    (~7 Gbit/s over CX-7).
  - **NVMe-oF over 400G fabric switch**: robert287 runs 12 TB of NVMe-oF
    attached to the 400G fabric switch, shared across all nodes — useful for
    training checkpoints.
  - **Cron-based model offloading**: VCR runs cron jobs that offload models
    unused >1 week to attached storage, delete if >1 month, clear caches.
  - **USB SSD**: starkrun reports 2TB USB SSD at 775 MB/s but randomly drops
    to 20 MB/s (see platform-gb10.md USB SSD speed finding). nightonthesun
    recommends USB 3.2 Gen2 2x2 SSD, noting USB bus bandwidth is the limit.
  - **modelctl tool** (piresbruno): `github.com/piresbruno/modelctl` — downloads
    models from HF to a local NAS, then syncs on-demand to the inference server.
    Single source → [conjecture].
  - **Key GB10-specific gotcha**: external USB drives connected at boot get
    stuck at USB2 speed (unmount/disconnect/reconnect/mount fixes it — see
    `[[wiki/platform-gb10.md]]` → USB3 fallback, now [reported] across 8 users
    / 4 OEM SKUs). Symlinking `.cache/huggingface` to an external drive is
    the common pattern (gaborm).

- **[conjecture]** **spark_hwmon — Linux hwmon driver for DGX Spark system power
  telemetry** (S-forum-acer-thermal, azampatti): `antheas/spark_hwmon` is a
  community Linux hwmon driver for the GB10 SoC that exposes full system power
  telemetry via standard `sensors` / sysfs interfaces. Useful for detailed
  thermal and power monitoring beyond what `nvidia-smi` reports (which shows
  GPU power only, not total SoC). Single source → [conjecture]. Relevant to
  the existing power/thermal monitoring findings on platform-gb10.md.

## Forum ingest: nvcr.io/nvidia/vllm:26.07-py3 tool-calling 500 — xgrammar dependency mismatch (2026-07-31)

- **[conjecture]** **nvcr.io/nvidia/vllm:26.07-py3 tool-calling returns HTTP 500 —
  xgrammar version mismatch** (S-forum-vllm-2607-xgrammar, rp_37716): the 26.07
  NGC container ships vLLM `0.24.0+092c4842.nv26.7` with `xgrammar==0.2.0`, but
  the vLLM build calls `xgrammar.normalize_tool_choice` which was only added in
  **xgrammar 0.2.4**. The container's vLLM has outrun its own bundled dependency.
  **Symptom:** any request with `tools`/`tool_choice` set returns:
  `{"error": {"message": "cannot import name 'normalize_tool_choice' from 'xgrammar'",
  "type": "InternalServerError", "code": 500}}`. Requests without tools work fine.
  **Workaround (verified on DGX Spark GB10):** two-line derived Dockerfile:
  ```dockerfile
  FROM nvcr.io/nvidia/vllm:26.07-py3
  RUN pip install -q -U xgrammar && pip install -q transformers==5.6.1
  ```
  `pip install -U xgrammar` bumps to 0.2.4 (fixes the import) but silently
  downgrades `transformers` from 5.6.1 to 4.57.6 (xgrammar 0.2.4 declares
  `transformers<5,>=4.38.0`). Re-pinning `transformers==5.6.1` is required —
  it's technically outside xgrammar's declared support, but basic tool-calling,
  `response_format: {"type": "json_schema"}`, and `guided_regex` all tested clean
  post-patch. `guided_grammar` (CFG) and `guided_choice` not tested. NVIDIA
  (Neill) confirmed an internal ticket and acknowledged the workaround but
  cannot officially validate it due to the dependency-constraint override.
  **Ask:** bump xgrammar to ≥0.2.4 in the next 26.xx container build with
  transformers re-verified against it. Tested on both
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` and `Qwen/Qwen3.6-35B-A3B-FP8`.
  Single source + NVIDIA staff confirmation → [conjecture]. This is the same
  class of NGC-container dependency mismatch as the 26.06-py3 FastAPI break
  (S-forum-vllm-2606-broken) — the NGC vLLM container line has a pattern of
  bundled dependencies lagging behind the vLLM build's actual requirements.

## Forum ingest: dependency-free DGX Spark dashboard (2026-08-02)

- **[conjecture]** **DGX-Spark-Dashboard — lightweight single-node monitoring
  dashboard** (S-forum-depfree-dashboard, angads25): a third independent
  community-built monitoring dashboard for DGX Spark (after sparkDash by
  brainchillz / S-forum-sparkdash and sparkDash by MiaAI-Lab /
  S-forum-sparkdash-mia). Distinct design philosophy: **dependency-free** —
  FastAPI + vanilla HTML/CSS/JS, no database, no agent daemon, no frontend
  framework, no CDN. One Docker Compose service.
  - **Footprint (measured on GB10):** ~190 MB image, ~42 MiB RAM, ~0.2% of one
    core when idle. Author measured the standard DCGM Exporter + Prometheus +
    Grafana stack on the same Spark at ~600 MiB RAM across 3 always-on
    containers and ~2.5 GB of images — roughly 14× the memory and 13× the
    disk. The full stack does more (history, alerting, full DCGM field set);
    this dashboard is a live-only glance at a single box.
  - **Demand-driven:** runs no background collector; reads metrics only when a
    browser asks. Idle overhead is effectively zero. Any category can be
    disabled in Settings.
  - **NVML, not nvidia-smi** (durable GB10 finding): the dashboard switched
    from `nvidia-smi` polling to **NVML** for GPU data after community
    feedback (elsaco) that `nvidia-smi` polling is a "performance killer" for
    a monitoring tool. On GB10, where the GPU and CPU share a unified memory
    pool, constant `nvidia-smi` subprocess spawning adds measurable overhead;
    NVML (the C library behind `nvidia-smi`) is the correct data source for
    low-overhead monitoring. GitHub: `singhangadin/DGX-Spark-Dashboard`.
  - **Security model:** read-only host `/proc` mounts, read-only Docker socket,
    host networking for real interface counters; runs non-root,
    `cap_drop: ALL`, `no-new-privileges`, read-only root filesystem.
  - **Single-node only:** CX-7 ports not monitored (author lacks a multi-node
    setup to test).
  Single source → [conjecture]. Reinforces the pattern of community-built
  Spark dashboards (cf. S-forum-sparkdash, S-forum-cluster-dashboard,
  S-forum-spark-studio, S-forum-sparkdash-mia). The NVML-over-nvidia-smi
  finding is the most broadly applicable takeaway — relevant to any GB10
  monitoring tooling. See also `[[wiki/platform-gb10.md]]` → operating
  constraints (nvidia-smi shows GPU power only, not total SoC).

### Batch 51 forum ingest (2026-08-04)

- **[conjecture]** **ComfyUI setup & patches for DGX Spark — UMA memory management
  fixes and benchmarks** (S-forum-comfyui-triplany, Triplany): a community install
  script + launch script + patch set (`Triplany/comfyui-dgx-spark`) that addresses
  the recurring ComfyUI UMA problems on GB10: double memory usage, not seeing all
  free VRAM and aborting (Wan 2.2 and Flux1 at full quant), models not unloading
  from VRAM when switching workflows, opposite problem of unloading after every
  run (every run cold), huge memory spikes on load, OOMs that brick the system.
  The setup claims stable, consistent memory usage across workflow switches
  (Flux2 → Wan2.2 properly evicts old models).
  - **Benchmark (stock ComfyUI templates, cold/warm, single Spark):**

    | Workflow | Quant | Mem | Resolution | Cold | Warm |
    |---|---|---|---|---|---|
    | Z-Image t2i | bf16 | 43.5 GB | 1024² | 96.17s | 43.73s |
    | Flux2-dev t2i | fp8mixed | 68 GB | 1024² | 300.38s | 50.14s |
    | LTX 2.3 t2v | fp8 | 44.73 GB | 1280×720, 5s | 179.55s | 81.83s |
    | Wan2.2 14b t2i | fp8 | 18 GB | 640², 5s | 644.75s | 565.24s |
    | Flux2-dev (full) + mistral3_small bf16 | full | 93.80 GB | 1024² | 407.52s | 80.25s |
    | Flux1-dev (full) + t5xxl fp16 | full | 32.16 GB | 1024² | 113.17s | 32.61s |

  - **comfy-aimdo 0.3.0 ARM compile fix** (AoE, joey28): `comfy-aimdo` 0.3.0
    compiles on ARM and fixes most model-loading-related problems on GB10.
    Multiple users confirm significant memory usage improvement after updating.
  - **LTX 2.3 22B NVFP4** (TheAwakenOne): `ltx-2.3-22b-dev-nvfp4.safetensors`
    runs ~12 min for a 20-second video — quality "not too bad." First reported
    LTX 2.3 22B NVFP4 data point on GB10.
  - **Docker alternative** (jd36): `jdaln/ComfyUI-DGX-Spark-Docker-opinionated`
    — a fork of an existing repo with auto-provisioning and workflow support.
  - Reinforces existing ComfyUI UMA findings (S-forum-comfyui-optimized,
    S-forum-comfyui-crash, S-forum-comfyui-container): the UMA memory management
    problems (double-VRAM, model eviction, memory spikes) are a persistent theme
    across independent ComfyUI setups on GB10. Single source for the benchmark
    numbers + multiple corroborating users for the comfy-aimdo fix → [conjecture].

### Batch 54 forum ingest (2026-08-05) — ACE-Step v1.5 + LTX-2.3 audio VAE flag

- **[conjecture]** **`--no-bf16-vae` required for LTX-2.3 audio workflows on GB10**
  (S-forum-acestep-v15-comfyui, Turrican): LTX-2.3's audio VAE never casts the incoming
  waveform to the VAE dtype, so under `--bf16-vae` every audio workflow dies with
  `Input type (float) and bias type (c10::BFloat16) should be the same`. This breaks
  stock ComfyUI LTX-2.3 audio templates and even a plain LoadAudio node. `--no-bf16-vae`
  takes only the VAE off bf16 while keeping the unet and text-encoder speedups. Passing
  `--fp32-vae` instead does **not** work — ComfyUI's VAE precision flags are mutually
  exclusive and one has already been added by the spark-comfyui build by then. Single
  source → [conjecture]. This is a GB10-specific gotcha for the spark-comfyui ComfyUI
  stack (bjarkebolding/spark-comfyui).

- **[conjecture]** **ACE-Step v1.5 + LTX-2.3 full lip-synced music video on single Spark**
  (S-forum-acestep-v15-comfyui, Turrican): end-to-end ComfyUI workflow on a single Spark
  generates a 2:28 music video (1280×704, 24fps, 3552 frames, 37 chained segments) —
  ACE-Step 1.5 XL for the song (2m05s), LTX-2.3 22B dev fp8 + distilled LoRA for video
  (30m11s, ~49s per 4s segment), total 32m16s. Uses the spark-comfyui build (cu130
  PyTorch, SageAttention for sm_121, GPU onnxruntime). Nine model files totaling ~59 GB.
  Single source → [conjecture]. Outside core LLM-inference scope (music + video
  generation, not vLLM/llama.cpp/sglang), but the `--no-bf16-vae` flag and the
  spark-comfyui stack details are GB10-specific. Source registered for provenance.
