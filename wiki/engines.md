# Inference engines on GB10: vLLM vs Atlas vs llama.cpp

> **area:** containers
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-nemotron-rpc, S-mimo-results, S-forum-atlas, S-forum-ds4-cuda, S-forum-dflash-qwen122, S-forum-ddtree-dflash, S-forum-stream-loading, S-forum-turboquant, S-forum-vllm-019-vs-023, S-forum-sm121-kernel-guide, S-forum-easy-vllm
> **updated:** 2026-07-13

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

## Forum ingest: DDTree, STREAM LOADING, SM121 kernel guide, vLLM regression (2026-07-09)

- **[conjecture]** **DDTree + DFlash** (S-forum-ddtree-dflash, joshua.dale.warner): DDTree (Diffusion
  Draft Tree) builds a tree with probabilities at every position rather than one fixed DFlash sequence,
  enabling far higher acceptance rates and resilience for unusual vocabularies. Verification budget
  remains the constraint. Proof of concept in `liranringel/ddtree`. **[conjecture]** Community claims
  80+ tok/s with Qwen3.5-27B AWQ on GB10 (Mitko Vasilev, vllm-turboquant repo). Qwen3.6-35B-A3B NVFP4
  + DFlash claimed 91–97 tok/s single-stream. Costs a small amount of extra VRAM for the drafted tree.
  Lucebox-hub is a lightweight harness that supports consumer Blackwell with DFlash+DDTree but has
  low context cap due to shared DFlash attention.
- **[conjecture]** **STREAM LOADING engine mod** (S-forum-stream-loading, amasawa_seiji): custom
  vLLM 0.17.1 that reads only needed expert/layer chunks from storage and on-the-fly quantizes to 4-bit,
  eliminating the simultaneous BF16+quant memory requirement. Enables running BF16/FP8 models that
  would otherwise need pre-quantized checkpoints. Confirmed: Qwen3.5-397B-A17B-FP8 (TP=2),
  Nemotron3-120B-BF16, Qwen3.5-122B-A10B. NF4 sub-mode for better quality than pure MXFP4. GitHub:
  `namake-taro/vllm-custom`. See `[[wiki/quantization-on-gb10.md]]`.
- **[conjecture]** **Native SM121 kernel build guide** (S-forum-sm121-kernel-guide, troy.e.davis):
  stock vLLM Docker images have **zero Blackwell cubins** — the `cuda_archs_loose_intersection` "12.0f"
  family pattern doesn't actually compile for SM121. A multi-stage Docker build that compiles SM121
  kernels from source and injects only the `.so` files (`_C.abi3.so` + `_moe_C.abi3.so`) into the
  stock image takes Qwen3.5-35B-A3B FP8 from **13.3 → 48.6 tok/s** (3.65×) — no model/driver/hardware
  changes. Full `pip install .` rebuild risks dependency drift (transformers version mismatch loads
  wrong model class). The `.so` injection approach preserves the curated dependency tree.
  **[conjecture]** `ptxas error: Instruction 'cvt with .e2m1x2' not supported on .target 'sm_121'` —
  SM121 lacks the microscaling instructions that SM120 (datacenter Blackwell) has; the CMake guard
  incorrectly includes SM121 in NVFP4 compilation.
- **[conjecture]** **vLLM 0.19 → 0.23 regression** (S-forum-vllm-019-vs-023): Qwen3.5-122B AutoRound
  on same Spark: 37→32.5 tok/s (~12% speed regression), 104→120 GB unified RAM (~15% memory regression).
  Tag working images before upgrading. See `[[wiki/quantization-on-gb10.md]]`.

## Forum ingest: easy-vllm harness, DSV4-Flash GB10 bring-up (2026-07-13)

- **[conjecture]** **easy-vllm code-agent harness** (S-forum-easy-vllm, sh.ahn): an open-source
  Claude Code-based meta-harness that automates vLLM build → serve → verify → improve loops on
  DGX Spark. Uses deterministic scripts for VRAM estimation, KV-clamp math, and version resolution.
  Verifies HW homogeneity before multi-node (cpu_arch/gpu_model/GPU count must match — mixed
  clusters intentionally unsupported). Includes a `mem_watchdog` + `earlyoom` host safety stack
  (kills container before UMA OOM = host-down). **[conjecture]** ib_write_bw measured 208–218 Gb/s
  (~90% of 200G link) on 2× DGX Spark — corroborates proven fabric measurements. (S-forum-easy-vllm)
- **[conjecture]** **DSV4-Flash on GB10 via jasl/vllm SM12x fork** (S-forum-easy-vllm): stock vLLM
  hit a double hard-wall on DeepSeek-V4-Flash at sm_121: (1) sparse-MLA attention allows
  `major ∈ [9,10]` only, (2) MXFP4 MoE auto-quant → MARLIN repack → unified-memory OOM → host down.
  Fix: `jasl/vllm` fork PR#41834 pinned by SHA `c766cbc6` (force-push-immune) via
  `VLLM_REPO`/`VLLM_REF` build-args, plus `--moe-backend humming` and an NVML patch for GB10
  clock telemetry. Succeeded on attempt #8: KV 386,512 tokens, reasoning intact, zero host-downs.
  Corroborates the MXFP4→MARLIN→UMA-OOM pattern (see `[[wiki/quantization-on-gb10.md]]`).
- **[conjecture]** **torch 2.11+ ABI wall** (S-forum-easy-vllm): vLLM versions pin a torch
  version that determines the build track. torch 2.10.x → easy road (prebuilt wheel, e.g. vLLM
  0.18.0 → torch 2.10.0 → NGC 26.01-py3, `pip install --no-deps`). torch 2.11+ → hard road
  (source build): NGC's torch alpha C++ ABI clashes with prebuilt `_C` extension — wheel builds
  but breaks silently at serve time. When a base-image guess fails, grep candidate headers to
  prove the missing symbol, then override one line (real case: vLLM 0.23.0 failed on NGC 26.03
  `torch::stable::Tensor has no member "layout"` → override to 26.05 → built).
