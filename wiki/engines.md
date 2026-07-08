# Inference engines on GB10: vLLM vs Atlas vs llama.cpp

> **area:** containers
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-nemotron-rpc, S-mimo-results, S-forum-atlas, S-forum-ds4-cuda, S-forum-dflash-qwen122
> **updated:** 2026-07-08

Three engines run on the Spark pair; pick by arch support and quant.

| | vLLM | Atlas | llama.cpp |
|---|---|---|---|
| Strength | broadest arch/quant support, JIT (handles new arches) | fastest for *supported* MoE families, AOT kernels, tiny image | GGUF + archs others lack (hybrid SSM) |
| Weakness | slower than Atlas on its turf; cudagraph walls | brittle: no AOT kernel = can't run; loader bugs | pipeline RPC = lower throughput than TP |
| Multi-node | TP=2 (no-ray) / EP | `--tp 2` over InfiniBand | RPC (pipeline) |
| When to reach for it | anything new, NVFP4/MXFP8/AWQ MoE, vision | a known-good Qwen3.x/Gemma MoE you want fast | GGUF-only model, or unsupported arch |

- **[proven]** **Default decision:** new model → **vLLM** (it'll at least load). Known Qwen3.5/3.6 MoE
  you want fast → **Atlas**. GGUF or an arch vLLM/Atlas reject → **llama.cpp** (`[[wiki/llama-cpp-rpc.md]]`).

## Atlas specifics (the non-obvious engine)

- **[proven]** Image `avarok/atlas-gb10:latest`, **entrypoint `["spark"]`** (pass `serve …`, NOT
  `spark serve …`), OpenAI API on `:8888`, ~2.5 GB, Rust+CUDA, zero Python deps. CLI: **`sparkrun`**
  (uv tool). The Atlas recipe registry is hidden — list with `--registry atlas -a`.
- **[proven]** **AOT-compiled kernels:** Atlas is fast because kernels are pre-compiled for specific
  shapes. A model whose shapes aren't in the target list **silently can't run** (Gemma-4-12B
  hidden_size 3840 had no kernel). vLLM is JIT → the fallback for any new arch.
- **[proven]** **KV pool sizing ignores the obvious knobs.** Atlas sizes KV as
  `free_system_RAM − ~6–7 GB` at build, **independent of `gpu_memory_utilization`, `oom_guard_mb`, and
  `block_size`** (none bind). Real caps: `--high-speed-swap*` (NVMe-evict KV; **halves throughput** —
  usually not worth it) or a docker `--memory` cgroup limit (the free-RAM probe respects it; sparkrun
  has no docker-arg passthrough so you need a custom launch wrapper).
- **[proven]** **`max_model_len` only shrinks the KV pool — ZERO effect on single-stream decode
  tok/s.** Use `-o max_model_len=8192` to make a tight model fit without paying speed.
- **[proven]** **Recipe "fit: YES" can still OOM** — the estimate ignores the runtime working set
  (buffer arena + MTP cudagraphs peaked ~86 GB during a load that "had 81 GB free for KV").
- **[proven]** **MTP:** `--speculative --mtp-quantization bf16` (K=2), **only** with separate-file MTP
  checkpoints (`model_mtp.safetensors`). Inline-MTP checkpoints crash the dense loader
  (`cuMemcpyDtoDAsync_v2 status 1`, double FP32 promotion at layer-0 SSM).
- **[proven]** **Known Atlas loader bugs:** inline-MTP NVFP4 (llmfan46 Qwen3.5/3.6) and dense-VL NVFP4
  (croll83 Qwopus-27B) both crash; the **MoE loader handles VL fine, only the dense loader is buggy**.
  → no Qwopus3.6 currently loads on Atlas; fall back to AEON-7 NVFP4 VL-MoE, llama.cpp GGUF, or
  self-quant.
- **[proven]** **Scheduling:** `--scheduling-policy slai` (TBT deadline 100 ms) default; chunked
  prefill on.
- **Validation ladder:** `sparkrun recipe validate` → `recipe vram` (reads HF config, prints Spark
  fit) → `run --dry-run` → smoke serve → `benchmark run` (llama-benchy, `spark-arena-v2`).

## Durable serving pattern (any engine)

- **[proven]** systemd **user** service, foreground `docker run --rm`, `Restart=always`/`RestartSec=10`,
  `TimeoutStartSec=600`, an `ExecStartPre` docker-daemon-readiness loop, `Conflicts=<other :8888 unit>`,
  and `loginctl enable-linger` to boot without login. The production version of this is a serving
  supervisor / model-swapper that adds demand-driven swapping + a real-inference watchdog on top of the
  same unit pattern.

## See also
`[[wiki/containers-and-tooling.md]]` · `[[wiki/multinode-tp-and-networking.md]]` · `[[wiki/quantization-on-gb10.md]]`

## Forum ingest: Atlas, ds4, DFlash engines (2026-07-08)

- **[reported]** **Atlas engine** (S-forum-atlas, tbraun96/AzeezIsh): pure Rust LLM inference engine
  with 20+ custom kernels compiled directly for SM121. **82 tok/s** Qwen3-Next-80B on a single Spark,
  **2.8× faster than NVIDIA's stock vLLM image**, no speculative decoding. Source-to-first-token in
  under 2 min (vLLM takes 40+). 32/32 benchmarks beat PyTorch baselines (18× faster RoPE, 8× Gated
  Delta Rule, 3.9× MoE W4A16). No Python/PyTorch dependencies. Now an NVIDIA partner (eugr joined
  NVIDIA). See `[[wiki/containers-and-tooling.md]]` for the Atlas image.
- **[conjecture]** **antirez/ds4 (DwarfStar 4)** (S-forum-ds4-cuda, entrpi): fully custom CUDA-native
  inference engine for DeepSeek-V4-Flash, optimized for 128 GB systems (originally Mac M-series).
  Builds in ~8 s on Spark (`CUDA_ARCH=sm_121`), cold load ~20 s, ~28 tok/s Q2 decode single-stream
  (pp2048 ~365 tok/s). Q2 GGUF ~81 GiB. OpenAI v1-compatible API on :8000. A fourth engine option
  alongside vLLM/Atlas/llama.cpp — model-specific, not general-purpose.
- **[conjecture]** **DFlash block-speculative decoding** for Qwen3.5-122B-A10B on 1x Spark
  (S-forum-dflash-qwen122, entrpi): ~81 tok/s on agent/tool-call traffic (accept len ~8.3), ~59 tok/s
  on albond's e2e harness. DFlash block-drafts ~12 tokens in one parallel forward (MTP-N needs N
  sequential head passes). Built on albond's INT4+MTP recipe, stacks community patches for vLLM 0.23.
  Baseline INT4 no-spec = 28.2 tok/s; MTP-2 = 51.6; DFlash n=12 = 53.7 (unpatched) → 59.0 (dense
  levers) → ~81 (real agent turns).
