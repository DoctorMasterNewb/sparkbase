# Inference engines on GB10: vLLM vs Atlas vs llama.cpp

> **area:** containers
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-nemotron-rpc, S-mimo-results, S-forum-atlas, S-forum-ds4-cuda, S-forum-dflash-qwen122, S-forum-ddtree-dflash, S-forum-stream-loading, S-forum-turboquant, S-forum-vllm-019-vs-023, S-forum-sm121-kernel-guide, S-forum-easy-vllm, S-forum-tokenspeed, S-forum-dsv4-vision, S-forum-llm-comfyui
> **updated:** 2026-07-15

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

## Forum ingest: TokenSpeed SM12x engine for DSV4-Flash (2026-07-14)

- **[conjecture]** **TokenSpeed `sm12x-stable`** (S-forum-tokenspeed, jasl): a fifth engine option
  alongside vLLM/Atlas/llama.cpp/ds4 — purpose-built inference engine with a clean architecture that
  is "easy to hack and maintain." jasl spent two weeks adding SM12x support. The SM12x path lands
  in `jasl/tokenspeed` (`sm12x-stable` branch), and the vLLM fork (`jasl/vllm`) remains maintained.
  Build on 2× GB10: torch 2.13, `TOKENSPEED_CUDA_ARCH=121`, FlashInfer CUTLASS MXFP4 MoE backend,
  `nvidia-nccl-cu13==2.30.4` (mandatory on multi-node — see below), `flashinfer-jit-cache==0.6.14+cu130`
  (skips 10–30 min cold CUTLASS-MoE JIT on first boot). Serve with `--moe-backend flashinfer_cutlass`
  + `--grammar-backend xgrammar` for tool-calling. (S-forum-tokenspeed)
- **[conjecture]** **Prefill wins, decode behind** (S-forum-tokenspeed): on the same 2× Spark pair,
  same fabric, same `llama-benchy` (MTP2 + fp8 KV + prefix cache, C=1 × 3 runs), TokenSpeed vs the
  jasl vLLM fork:
  | depth | ctx_pp t/s (TS / vLLM) | pp2048 (TS / vLLM) | tg128 peak (TS / vLLM) |
  |---|---|---|---|
  | 8192 | **2057 / 1866** | 1404 / 1406 | 30.3 / 41.5 |
  | 16384 | **2062 / 1825** | 1329 / 1354 | 28.7 / 41.3 |
  | 32768 | **1979 / 1737** | 1149 / 1224 | 33.3 / 45.3 |
  Cold-context prefill leads by ~10–14% (the dominant cost in long-context / 1M scenarios).
  pp2048-at-depth is at parity (94–100%). Decode is behind ~70–74% — the CUTLASS MoE that wins
  prefill has a weaker small-M decode GEMM; a hybrid path (CUTLASS prefill + Triton decode,
  single weight residency) is in progress. If decode-heavy at low concurrency, the vLLM fork is
  faster today. (S-forum-tokenspeed)
- **[conjecture]** **KV capacity +25%** — TokenSpeed fits 1.90M vs vLLM's 1.52M KV tokens at
  `--max-model-len 131072` in the same config. (S-forum-tokenspeed)
- **[conjecture]** **Tool calling & stability** — 45/45 tool-calling requests engine-clean, zero
  HTTP 500s (the intermittent MTP + thinking + `tool_choice` 500s that affect upstream vLLM's
  reasoning-boundary bug are absent). Long-gen GSM8K 0.96, zero illegal-memory-access. MTP
  acceptance rate higher than vLLM. (S-forum-tokenspeed)
- **[conjecture]** **NCCL 2.30.4 mandatory on multi-node** (S-forum-tokenspeed): 2.28.9 (torch's
  default), 2.29.7, and 2.30.7 all hit an NCCL graph-replay proxy-progress wedge that presents as
  serve hangs / late-wave request timeouts. Pin with `--no-deps` **after** the kernel build (torch's
  dep re-resolves over manual pins on every editable install). A minimal repro is headed to
  NVIDIA/nccl. This likely explains several "2-node decode hang" reports in the community.
  Corroborates existing [reported] NCCL 2.30.4 finding (S-forum-ds4f-4x-vllm).
- **[conjecture]** **Build traps** (S-forum-tokenspeed): `TOKENSPEED_CUDA_ARCH` must be set for
  the kernel build (no auto-detect → "no kernel image"); `rm -rf tokenspeed-kernel/python/objs`
  after changing it. `fast_hadamard_transform` must be built from the GitHub repo (PyPI sdist is
  missing its `csrc/`). Uninstall torchvision/torchaudio (unused; its torch-2.13 ABI trips
  transformers).

## Forum ingest: Multi-model co-hosting — vision + LLM on 2× Spark (2026-07-15)

- **[conjecture]** **Co-hosting a vision model alongside DSV4 on 2× Spark is memory-starved**
  (S-forum-dsv4-vision, cerchez07): running DeepSeek V4 + Qwen3-VL on the same 2-node cluster requires
  cutting DSV4 to 256K context at `--gpu-memory-utilization 0.73` to free enough VRAM for the vision
  model — and it's still "teetering on the edge of OOM." The single-tenant-per-node constraint
  (`[[wiki/platform-gb10.md]]`) makes this fundamentally difficult on 2× 121 GB UMA.
- **[reported]** **Recommended pattern: offload vision to a separate machine** (StarChickenXVII, 0rand):
  host the vision model (Gemma-4-12B, Qwen3.5-4B with vision tower) on a separate device (MacBook Pro,
  etc.) and expose it as a tool/API to the LLM running on the Spark cluster. This keeps the 2× cluster
  fully dedicated to the text LLM. Multiple users independently arrived at this approach.
- **[conjecture]** **Multimodal front-end + text reasoning pipeline** (gpieceoffice): runs
  `RedHatAI/Gemma-12B-NVFP4` or `RedHatAI/Qwen3.5-9B-FP8-dynamic` as a multimodal front-end (vision +
  audio analysis) alongside DSV4 with MTP=3 as the text reasoning engine on 2× Spark — combined
  ~35–40 tok/s. The vision model extracts structured information from visual/audio inputs and passes
  it to DSV4 for reasoning. Key insight: "simply connecting the models is not enough" — the multimodal
  analysis prompt and information transfer pipeline design is the most important aspect.
- **[conjecture]** **Qwen3.5-4B with vision tower fits in worker node spare memory** (0rand): 2–3 GB
  more room available on a worker node for a small vision model. Lower quality but sufficient for
  debugging / screenshot analysis.

## Forum ingest: LLM + ComfyUI co-hosting on 2× Spark (2026-07-15)

- **[conjecture]** **vLLM's KV cache reservation starves co-hosted workloads on unified memory**
  (S-forum-llm-comfyui, Alexander-F): vLLM pre-allocates unified memory for KV cache
  aggressively, leaving insufficient headroom for ComfyUI image generation on the same node.
  This is a known UMA constraint, not a configuration bug — the 121 GB pool is shared by
  weights, KV cache, and any co-hosted process. The practical solution on a 2× Spark setup is
  to dedicate one Spark to the LLM and the other to ComfyUI.
- **[conjecture]** **`--gpu-memory-utilization` 0.7–0.8 enables co-hosting** (S-forum-llm-comfyui,
  clawdiusmaximus, C_G): reducing vLLM's memory utilization from 0.9 to 0.8 (or 0.72–0.75)
  frees enough UMA headroom to run ComfyUI on the head node alongside a clustered LLM.
  C_G runs Qwen3.6-27B clustered on both nodes at low util with ComfyUI on node 2 (configured
  for low memory, models loaded on-demand), peaking at 114 GB RAM during 1024×1024 SDXL
  generation. clawdiusmaximus got ComfyUI running on the head node of a DSV4 cluster at
  0.8 util — "performance is slow, but output was fine."
- **[conjecture]** **llama.cpp is better than vLLM for co-hosting with ComfyUI**
  (S-forum-llm-comfyui, vasimv): vLLM's memory management is "really bad with unified memory"
  — it often OOMs because buffers/cache memory fill up. llama.cpp handles co-hosting better:
  Qwen3.6-35B in llama.cpp runs alongside ComfyUI without problem, though image generation
  is 5–6× slower under LLM load. This corroborates the existing finding that vLLM's
  pre-allocation behavior is aggressive on UMA (see `[[wiki/platform-gb10.md]]`).
- **[conjecture]** **Disable swap before loading large models** (S-forum-llm-comfyui,
  AakankshaS): `sudo swapoff -a` before loading large models — when swap is active, massive
  allocations force the OS to thrash data onto storage, creating kernel lockups. Documented
  in DGX Spark troubleshooting.
