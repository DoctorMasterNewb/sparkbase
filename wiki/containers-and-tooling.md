# Container images & tooling

> **area:** containers
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-forum-vllm-claude, S-forum-btop, S-forum-model-manager, S-forum-sparkdash, S-forum-tool-eval, S-forum-thunderkittens, S-forum-driver610, S-forum-flux2-nunchaku, S-forum-comfyui-container, S-forum-llamacpp-container, S-forum-sage-attn, S-forum-vllm-2606-broken, S-forum-gemma4-qat, S-forum-mistral-s4-nvfp4, S-forum-qwen-tts-arm64, S-forum-llama-benchy, S-forum-cluster-dashboard, S-forum-sunshine-rdp, S-forum-flux2-nvfp4-compute
> **updated:** 2026-07-10

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
