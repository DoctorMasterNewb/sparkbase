# Container images & tooling

> **area:** containers
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun5, S-sess-jun4, S-mimo-results, S-mimo-doc, S-forum-vllm-claude, S-forum-btop, S-forum-model-manager, S-forum-sparkdash, S-forum-tool-eval, S-forum-thunderkittens, S-forum-driver610
> **updated:** 2026-07-08

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
