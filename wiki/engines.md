# Inference engines on GB10: vLLM vs Atlas vs llama.cpp

> **area:** containers
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-nemotron-rpc, S-mimo-results, S-forum-atlas, S-forum-ds4-cuda, S-forum-dflash-qwen122, S-forum-ddtree-dflash, S-forum-stream-loading, S-forum-turboquant, S-forum-vllm-019-vs-023, S-forum-sm121-kernel-guide, S-forum-easy-vllm, S-forum-tokenspeed, S-forum-dsv4-vision, S-forum-llm-comfyui, S-forum-colibri-glm52, S-forum-dsv4-abliterated, S-forum-mtp-lossless, S-forum-woolyai, S-forum-gridbook, S-forum-glm52-vision, S-forum-glm52-hybrid, S-forum-speedycolibri, S-forum-dsv4-reap25, S-forum-velogb10, S-forum-dsv4-dspark-eugr, S-forum-dsv4-0731-caching, S-forum-dsv4-0731-bench, S-forum-dsv4-0731-dspark-loader, S-forum-dsv4-0731-ds4-cuda, S-forum-dsv4-vision-plugin, S-forum-vllm-snapshot, S-forum-dsv4-0731-gguf, S-forum-dsv4-0731-sparkrun, S-forum-lmcache-ipc-deadlock
> **updated:** 2026-08-11

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

## Forum ingest: Colibri — expert-streaming engine for 744B MoE on single Spark (2026-07-16)

- **[conjecture]** **Colibri engine** (S-forum-colibri-glm52, JustVugg/colibri): a sixth engine
  alongside vLLM/Atlas/llama.cpp/ds4/TokenSpeed — pure C, zero deps, streams MoE experts from
  disk on demand. Designed to run GLM-5.2 (744B MoE, 40B activated) on a single DGX Spark with
  121 GB unified memory — the experts never all live in RAM; only the hot ones get cached via
  LRU/pin. Repo: `JustVugg/colibri`. **[conjecture]** A forum user (Keving) linked a benchmark
  by VincentMarquez (GitHub issue #161) run on a real DGX Spark: Ubuntu 24.04, kernel 6.17,
  driver 580.126.09, int4 MoE + int8 MTP heads, `COLI_CUDA_UNIFIED=1` · `CUDA_DENSE=1` ·
  `CUDA_GROUPED=1` · `DIRECT=1` · `PIPE=1`. Results:
  - **Tier A (full top-8, stock routing):** 2.39 decode tok/s (timed profile), ~2.08 tok/s
    (chat warm session), 82% expert cache hit, RSS ~76 GB.
  - **Tier B (experimental CACHE_ROUTE, opt-in not upstream):** 3.33 decode tok/s best,
    97% hit, 14% expert substitution, RSS 78.5 GB. `CACHE_ROUTE=1 ROUTE_J=2 ROUTE_M=12`
    (keep true top-2, fill remaining 6 slots preferring pinned/LRU experts ranked in top-12).
  - **Disk I/O:** buffered 4.25 GB/s, O_DIRECT **9.69 GB/s** (`c/iobench`, 19 MB × 64 reads,
    8 threads). This corroborates the existing NVMe-oF/expert-streaming direction
    (`[[wiki/roadmap.md]]`).
  - **Profile breakdown (Tier B timed one-shot):** expert-disk 3.85s, expert-matmul 3.83s,
    attention 6.16s, other 4.16s — attention dominates, not disk.
  - MTP=0/DRAFT=0 for all speed cells; `TEMP=0.7`; short context via `:reset` each turn.
  - Correct output (first-20 primes) every turn; not a full quality gate.
  **[conjecture]** Scale-out hypothesis (joshua.dale.warner): could this scale to 2× Sparks
  via expert parallel? Twice the resident RAM, twice the SSD read bandwidth. Untested.
  This is the first reported engine that makes a 744B model usable (if very slowly) on a
  single 121 GB Spark — the streaming-from-disk approach is the complement to the
  multi-node TP path used for other large MoE models.

## Forum ingest: SpeedyColibri — Rust port of Colibri for GLM-5.2 (2026-07-28)

- **[conjecture]** **SpeedyColibri** (S-forum-speedycolibri, GPilz): a Rust port of the Colibri
  expert-streaming engine, targeting GLM-5.2 (744B MoE) on a single DGX Spark. Built on the Colibri
  base by a developer who started learning the domain ~2 months prior. Initial version: ~1 tok/s.
  With fp8 optimizations: ~4 tok/s. Working on multi-Spark setup (target: 8 tok/s on 2× Spark).
  Repo: `GriffinPilz/SpeedyColibri`. **GB10 relevance:** confirms the expert-streaming approach is
  reproducible by a different developer in a different language (Rust vs C), and that fp8 expert
  quantization meaningfully improves throughput (1→4 tok/s, 4×). Still proof-of-concept speed —
  the attention-not-disk bottleneck identified in the original Colibri profile applies. Single
  source → [conjecture]. Community feedback (JW2026): the approach is more interesting as a
  technique (applying expert streaming to other models, keeping unused layers in 2nd-tier memory
  without REAP) than as a production inference engine; needs TP/PP/EP support to be useful.

## Forum ingest: DSV4-Flash REAP25 PrismaAURA — measured-quant ds4 fork (2026-07-29)

> **evidence:** conjecture (single forum thread, 2 developers in same thread)
> **sources:** S-forum-dsv4-reap25

- **[conjecture]** **twaggs88/DeepSeek-V4-Flash-REAP25-DSpark-ds4-GGUF — measured-KL quant
  allocation for single GB10** (S-forum-dsv4-reap25, twaggs88): a third independent fork of
  antirez/ds4 tuned for a single GB10, with a matching GGUF whose quantization was **measured
  per-tensor** (not hand-picked). Key results on one GB10:
  - Tool-use quality: 92/100 (tool-eval-bench hardmode, temp 0.95, top-p 0.38)
  - Decode (speculative, 0–8k ctx): 16.5 tok/s, flat with depth
  - DSpark draft acceptance: 77.2% (structured/tool workloads)
  - Prefill: ~420→~390 tok/s (2k→8k prompt)
  - Resident size: 91 GB (weights + merged drafter)
  - v0.2.3 update: prefill 365→~410–430 t/s via W4A8 CUTLASS type-40 path; 3 concurrent 1M-ctx
    sessions (was 2) via pre-stored MXFP8_LT layout (freed ~6.4 GiB double-storage)

  **Quant allocation methodology:** each routed-expert tensor's reconstruction error was measured
  per candidate format against the FP8/FP4 QAT source, weighted by empirical Fisher sensitivity,
  allocated under byte budget by exact knapsack (built on PrismaQuant). Result: IQ2_XXS floor
  (2.06 bpw) on most experts, MXFP4 (4.25 bpw) promoted on quality-sensitive layers (early
  layers want it on gate/up, late layers on down — depth pattern), MXFP8 on attention/shared/
  head. Experts REAP-pruned 25%. The MXFP4/MXFP8 are byte-lossless re-encodes of the checkpoint's
  source encoding — zero requantization loss. Choosing formats by measurement beat a good hand
  rule by 8 composite points at equal size and speed. Single source → [conjecture].

- **[conjecture]** **W4A8 CUTLASS type-40 path is source-faithful for DSV4-Flash** (S-forum-dsv4-reap25,
  twaggs88): DeepSeek-V4-Flash ships `expert_dtype: fp4` with `activation_scheme: dynamic, fmt: e4m3`
  — the model was designed to compute experts in exactly fp4×E4M3. The W4A8 path (fp4 weights ×
  E4M3 activations) runs on sm_120 f8f6f4 tensor cores, ~2.6× faster per layer than dp4a. It's
  more faithful to the original than Q8_K activations (which the source never uses). Single source.

- **[conjecture]** **IQ2 experts cannot move to tensor cores — dequant net loss** (S-forum-dsv4-reap25,
  twaggs88): the 2-bit IQ2 experts are the majority of routed layers. Moving them to tensor cores
  requires 2-bit→8-bit dequant expansion, which costs more bandwidth than the tensor cores save
  (~72 ms/layer vs ~62 for dp4a; E4M3 is also ~2.5% RMS lossy on the IQ2 codebook). So IQ2 stays
  on CUDA-core dp4a — this is the wall for "all experts on tensor cores." MXFP4 is the only format
  that escapes to tensor cores natively (zero dequant). This is a GB10-specific finding because
  the sm_121 tensor-core FP4 path exists but the dequant overhead for sub-4-bit formats negates
  the compute advantage. Single source → [conjecture].

- **[conjecture]** **marco.palaferri/xangel82/DS4-GB10-GX10-DSpark-CUDA — 854 tok/s prefill, 24-25
  tok/s decode at 55k-70k ctx** (S-forum-dsv4-reap25, marco.palaferri): a fourth independent ds4
  fork. Latest results: up to 854.26 tok/s on first 8192-token cold prefill chunk; 787.06 tok/s
  average on 13.6k-token cold prompt; 724.69 tok/s on 41.7k-token append at 55.3k context; ~24-25
  tok/s DSpark decode at 55k-70k context. Pipeline: token-tile HMMA attention, D2R/MMQ routed-MoE
  prefill, MXFP4 indexer cache with native SM121 block-scaled MMA, exact Top-512 selection. The
  HMMA-attention path is fp16/non-bit-exact (trades ~2× cold prefill for determinism vs the
  exact path). Single source → [conjecture]. This is notably faster prefill than twaggs88's
  ~410-430 tok/s — the gap attributed to HMMA attention + bigger 8192-token prefill chunks.

- **[conjecture]** **DSV4-Flash prefill is compute-bound, not bandwidth-bound** (S-forum-dsv4-reap25,
  twaggs88): "the experts sit at a couple percent of the memory roofline, so the whole game is
  filling the tensor cores." This contrasts with the proven finding that decode is bandwidth-bound
  on GB10 — the prefill/decode asymmetry means different optimization strategies apply to each
  phase. Prefill optimization (tensor core utilization, attention kernel choice, chunk size) is
  distinct from decode optimization (weight bandwidth reduction). Single source → [conjecture],
  consistent with the proven bandwidth-bound decode finding.

## Forum ingest: MTP quality & prefix-cache interaction (2026-07-18)

- **[conjecture]** **MTP measurably affects output quality, not just throughput**
  (S-forum-mtp-lossless, JasonW): on GB10 runs, MTP-on vs MTP-off shows quality deltas
  on tool-call benches — up to ~5 points difference that "cannot be explained by noise
  or SD." Temperature/sampling tuning did not eliminate the gap. Note: the theoretical
  argument for losslessness assumes strict causal verification of drafted tokens, but
  community members report quality drift in practice. See the practical-lossiness
  argument below.
- **[conjecture]** **MTP gives ~40% speed vs ~2% quality hit on Qwen3.6-27B**
  (S-forum-mtp-lossless, Azampatti): a custom capability suite shows "almost identical"
  scores with and without MTP — a tradeoff worth having for the ~40% throughput gain.
  This corroborates (independent second source) that the quality delta, where present,
  is small for general capability benchmarks but can be larger for tool-call evals.
- **[conjecture]** **vLLM and llama.cpp both have MTP + prefix-caching interaction bugs**
  (S-forum-mtp-lossless, mangosq / Yen): noticeable quality/output degradation appears
  when MTP and prefix caching are enabled together; without prefix caching, no visible
  degradation. Practical mitigation: disable prefix caching when running MTP, or leave
  MTP off for agentic workflows. This is an engine bug, not a theoretical property of
  MTP — corroborates that "theory != deployment": practical MTP deployments can be
  lossy due to serving-stack bugs (how attention treats discarded blocks / re-evaluates).
  Affects both engines Spark users run.
- **[conjecture]** **DS4F MTP tuning — prefix-batch 16384, MTP=4 → 70–75% acceptance**
  (S-forum-mtp-lossless, 0rand): for DS4F the optimal prefix-batch size is 16384 with
  MTP=4, yielding 70–75% stable prediction quality — up to ~80% on coding workloads,
  down to ~70% on llama-benchy performance tests. Depends on model (attention type,
  head count, attention-cache size) and number of prediction tokens. Note:
  prefix-batch size "greatly eats into KV cache" on unified memory — a real tradeoff
  on GB10's 121 GB pool. Quality impact unconfirmed ("the hunch is it does").
- **[conjecture]** **Practical MTP deployments are lossy by design** (S-forum-mtp-lossless,
  Nerhun): "theory != deployment" — strict mathematical verification of drafted tokens
  would yield terrible acceptance rates due to early specialization of NTP layers;
  real-world deployments cut corners to keep throughput. Counter-argument (A3refaat,
  JasonW): properly-implemented MTP enforces causality between drafted tokens and the
  target verify step, making it mathematically lossless — 0% acceptance only costs
  throughput, not quality. The disagreement is unresolved in-thread; the observed
  quality deltas (above) suggest at least some serving stacks are not enforcing strict
  verification. Flagged for hardware-agent verification (see roadmap).

## Forum ingest: WoolyAI multi-agent stack, PrismaQuant GridBook plugin (2026-07-25)

- **[conjecture]** **WoolyAI Private Multi-agent Inference Stack** (S-forum-woolyai, manisha5):
  a closed-source inference server for 2× DGX Spark clusters targeting **multi-model agentic
  workflows** — multiple models of different specializations/sizes resident and swappable behind
  one endpoint. The scheduler batches each request burst, coordinates both ranks, and changes the
  resident model only at a safe boundary (model-activation-wait 2-16s depending on model size).
  Benchmarks via LlamaBenchy, unquantized, no speculative decoding: DSV4-Flash 21.15 tok/s (C1),
  Gemma-4-26B-A4B 30.22, Nemotron-3-Nano-Omni-30B-NVFP4 39.42. No launch command, no source code,
  no repro recipe shared — only a PDF report and a product URL. **GB10 relevance:** the
  multi-model-swap-with-scheduler pattern is a genuine use-case shape that vLLM/SGLang don't natively
  serve (they assume one resident model per engine instance), but the performance claims are
  unverified and community-skeptic (mrDragonFox: "at C1 its slower then llama.cpp"; entrpi: the
  community DSpark fork is "far more performant"). Treat as a vendor announcement, not a reference
  build. See `[[wiki/benchmarks.md]]` for the numbers.
- **[conjecture]** **PrismaQuant GridBook vLLM plugin** (S-forum-gridbook, tenari/RobTand): a vLLM
  plugin (GitHub `RobTand/gridbook`) exposing 41 codebook quant formats (1.781–6 bit) with native
  FP8/NVFP4-grid dequant via tensor-core table lookup. Not a standalone engine — it's a quant plugin
  for vLLM, extending PrismaQuant (the bit-allocator, S-forum-prismaquant). Released checkpoints:
  Qwen3.6-27B 5.5-bit (KL 0.0049), Hy3-295B-A21B 2.9-bit. Reported overhead ~10% decode / 30%
  prefill vs. native NVFP4. See `[[wiki/quantization-on-gb10.md]]` for the full mechanism and
  findings. Single source; no independent benchmark yet.

## Forum ingest: GLM-5.2-Vision + adaptive MTP (2026-07-26)

- **[conjecture]** **GLM-5.2-Vision-NVFP4 — frozen-backbone vision projector for GLM-5.2**
  (S-forum-glm52-vision, CosmicRaisins): `baseten/GLM-5.2-Vision-NVFP4` adds vision to GLM-5.2
  (744B/40B MoE) without modifying a single GLM weight — the text backbone and vision tower
  (MoonViT) are both frozen and byte-identical to upstream. The only new parameters are a
  **49.5M-parameter projector** that maps MoonViT's 1152-dim patch embeddings into GLM's
  6144-dim token space. Ported to the CosmicRaisins/glm-5.2-gb10 repo for a 4-node Spark cluster
  (TP=4). Only OCR and classification tested so far; GUI screenshot / visual-bug use cases
  anticipated but not yet validated. **GB10 relevance:** this is the first reported
  vision-enabled GLM-5.2 on Spark — the frozen-backbone approach means the existing NVFP4
  recipe and MTP stack are unchanged; only the lightweight projector loads additionally.
  Single source → [conjecture].
- **[conjecture]** **Adaptive MTP — dynamic 2–5 drafted tokens based on acceptance feedback**
  (S-forum-glm52-vision, CosmicRaisins): a modification of aidendle94's adaptive MTP work,
  integrated into the GLM-5.2-gb10 repo. The model dynamically switches between 2 and 5 drafted
  tokens depending on the acceptance rate of positions p2–p4. The goal: get the 30+ tok/s speedup
  in code (where high acceptance justifies more drafted tokens) without sacrificing performance
  in prose (where low acceptance makes extra drafted tokens wasteful). **This is a new MTP
  regime on Spark** — all existing MTP recipes use a fixed `num_speculative_tokens` (e.g. MTP=2,
  MTP=4, MTP=5). The adaptive approach has not been benchmarked vs fixed-MTP on GB10 yet.
  Theoretically sound (match draft depth to per-step acceptance), but the overhead of the
  feedback loop on the bandwidth-bound Spark decode path is uncharacterized. Single source →
  [conjecture]. See `[[wiki/engines.md]]` → MTP quality & prefix-cache for the existing MTP
  tuning findings.

## Forum ingest: GLM-5.2 reasoning parser, thinking-off, MTP depth (2026-07-27)

- **[conjecture]** **Structured Output 58% on tool-eval-bench is a reasoning-parser bug, not a
  model or quant bug** (S-forum-glm52-hybrid, mike_ber): the `glm45` reasoning parser leaks a
  sentence fragment into the content channel before JSON output — the model produces perfectly
  valid schema-compliant JSON but prefixes 1-3 tokens of an unfinished conversational lead-in
  (e.g. `I{"ticker":"NVDA",...}`, `Here{"location":"Tokyo",...}`). **A/B test (category O, 6
  scenarios, seed 42, temp 0):** thinking OFF → 100/100 (12/12 pts, all pass, median turn 4.9s);
  thinking ON → 75/100 (9/12 pts, median turn 7.6s). Turning thinking off makes Structured Output
  58% → 100% and is 36% faster. **Full hardmode run (thinking off): 83 → 88 (+8 pts, 140 → 148/168)**
  — but Structured Reasoning dropped 100% → 67% and Restraint & Refusal 100% → 83%: without thinking,
  the model reaches for a tool instead of doing the work itself. **Takeaway: strict output format →
  thinking off; open-ended analysis → thinking on.** Confirmed by CosmicRaisins (image author).
  This is a GLM-5.2-specific reasoning-parser issue, not a GB10 hardware issue, but it bites Spark
  users running tool-calling agents. See `[[wiki/models/glm-5.2.md]]`.
- **[conjecture]** **MTP4 outperforms MTP5 on tool-eval-bench for GLM-5.2** (S-forum-glm52-hybrid,
  ciprianveg): MTP5 scored 83 on tool-eval-bench, switching to MTP4 gave 85+. Look for FSM errors
  in logs for tool calls as a diagnostic. Consistent with the existing finding that higher MTP
  depth can hurt quality (see MTP quality section above). Single source → [conjecture].
- **[conjecture]** **GLM-5.2 word-salad at >90k context was caused by `repetition_penalty=1.2`,
  not a model or hardware bug** (S-forum-glm52-hybrid, mclenithan): after 2 weeks of debugging
  (trying multiple quants, vLLM versions, b12x versions), the root cause was a hardcoded
  `repetition_penalty=1.2` left over from MiMo 2.5 work. With GLM-5.2, this causes word-salad
  (random mixed-script fragments, top-1 logprob ~-10 to -12, near-uniform distribution) at >80-95k
  tokens after 15+ multi-turn interactions. Temperature 0 does not prevent it. Fix: remove the
  repetition penalty. **Durable lesson: sampling-parameter configs do not transfer between models
  on Spark — GLM-5.2 has different sensitivity than MiMo 2.5.** See
  `[[wiki/models/glm-5.2.md]]` and `[[wiki/quantization-on-gb10.md]]`.

## Forum ingest: veloGB10 — Rust-based GB10-optimized engine (2026-07-31)

- **[conjecture]** **veloGB10** (S-forum-velogb10, stav_kats/sf-stav): a seventh engine alongside
  vLLM/Atlas/llama.cpp/ds4/TokenSpeed/Colibri — Rust-based with custom kernels written specifically
  for the "retail" GB10 chipset. TP=2 cluster support (for performance, not just memory). Easy
  install: compile Rust → 1 binary + a few PTX files, no Python deps. Repo: `sf-stav/veloGB10`.
  Pure NVFP4 (100% of layers quantized, including MTP head as FP8 in "mixed" mode). Reported
  single-user tok/s (not aggregated):
  - Qwen3.6-27B-NVFP4-full: ~40 tok/s single, ~45-50 tok/s 2×
  - Qwen3.6-35B-A3B-NVFP4: ~110 tok/s single, ~120+ tok/s 2×
  - Qwen3.6-9B-NVFP4: ~80 tok/s single, ~90-100 tok/s 2×
  - Qwen3.5-0.8B-NVFP4-mixed: ~227.8 tok/s, Qwen3.5-2B: ~161.6, Qwen3.5-4B: ~100
  Community feedback: 2× cluster numbers are **slower than current engines** for the 27B dense
  model (jc2375); JW2026 reports the 35B MoE is at parity with eugr vLLM at c=1 but vLLM wins
  at c=4/8/16; robert287 gets 30 tok/s on 27B dense with vLLM (vs veloGB10's ~40 single / ~45-50
  2× — but the 2× gain is modest for TP overhead). The engine is newly released; no independent
  benchmarks yet. Single source → [conjecture]. The pure-NVFP4 (100% layers) approach is notable
  — most NVFP4 checkpoints leave ~half layers in BF16 (see `[[wiki/quantization-on-gb10.md]]` →
  NVFP4 meta-analysis). Whether veloGB10's custom kernels extract more from the full-NVFP4 path
  on sm_121 is uncharacterized.

## Forum ingest: DeepSeek-V4-Flash-DSpark recipe + draft-token tuning (2026-08-02)

- **[conjecture]** **DSV4-Flash-DSpark on 2× Spark via eugr spark-vllm-docker — full recipe**
  (S-forum-dsv4-dspark-eugr, davidbarnesguildford): a complete YAML recipe for serving
  `deepseek-ai/DeepSeek-V4-Flash-DSpark` on a 2-node DGX Spark cluster. Key flags:
  `--tensor-parallel-size 2 --distributed-executor-backend ray --kv-cache-dtype fp8
  --block-size 256 --max-model-len 262144 --max-num-seqs 4 --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.8 --enable-prefix-caching --load-format safetensors
  --tokenizer-mode deepseek_v4 --tool-call-parser deepseek_v4 --reasoning-parser deepseek_v4
  --speculative-config '{"method":"dspark","num_speculative_tokens":5}'
  --hf-overrides '{"dspark_noise_token_id":128799}'
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
  --default-chat-template-kwargs.thinking=true
  --default-chat-template-kwargs.reasoning_effort=high`. Env: `DG_JIT_USE_NVRTC=0`,
  `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1`, `VLLM_USE_BREAKABLE_CUDAGRAPH=0`.
  **`--load-format safetensors` is mandatory** — the default loader crashes on this checkpoint.
- **[conjecture]** **FlashInfer PR 3817 is required** (S-forum-dsv4-dspark-eugr,
  davidbarnesguildford): the stock `vllm-node` image must be patched via
  `./build-and-copy.sh --apply-flashinfer-pr 3817` before serving DSV4-Flash-DSpark. Without it
  the recipe fails. `build-and-copy.sh -c` copies the patched image to the worker node over
  InfiniBand (faster than the manual `docker save` + `rsync` alternative).
- **[conjecture]** **3 draft tokens beats 5 on DSV4-Flash-DSpark at 50-concurrent** (S-forum-dsv4-dspark-eugr,
  johndaly): tuning the `num_speculative_tokens` from 5 (the posted recipe default) down to 3
  gives a significant throughput improvement on the same 50-concurrent `vllm bench serve`
  workload shape (random 244-in/200-out, 50 prompts, max-concurrency 50):
  - **3 draft tokens:** 71.63 tok/s output, 52.52 ms TPOT, 48.35% acceptance, accept_len 2.45
  - **5 draft tokens (posted):** 48.60 tok/s output, 85.64 ms TPOT, 27.65% acceptance, accept_len 2.38
  - **5 draft tokens (local reproduction):** 65.46 tok/s, 57.09 ms TPOT, 34.12% acceptance
  `max_num_batched_tokens=10240` was slightly better than 8192 for the 3-draft run. A 16384
  batch-token attempt did not fit at `max_model_len=262144` on the 2-node setup. The main
  tuning result: **3 draft tokens beat 4, 5, and 6 draft tokens on this workload.** Single
  source (one user's sweep) → [conjecture], but a well-documented A/B with full benchmark
  numbers. Consistent with the DSpark mechanism (confidence-scheduled verification truncates
  block length as concurrency rises — fewer drafts at high concurrency can be more efficient).

### Batch 50 forum ingest (2026-08-03)

- **[conjecture]** **vLLM prefix cache inconsistency on DSV4-Flash-0731 on 2× Spark** (S-forum-dsv4-0731-caching,
  Sa0lence): when running DeepSeek-V4-Flash-0731 on a 2-node Spark cluster with vLLM, prefix
  cache behavior is non-deterministic: sometimes the prefill phase hits the cache and completes
  in 1–2 seconds; other times the cache is missed completely and prefill takes several minutes
  to tens of minutes. No deterministic cause identified — "it feels random." The user is on the
  `anemell` image. A second user (dashtotherock) recommends `aidendle94/sparkrun-vllm-ds4-gb10`
  (production-hybrid-1.1) instead, which works on both DSpark and 0731 variants, and reports 0731
  is slightly better than DSpark in benchmarks (not a big jump). This is the first report of
  prefix cache unreliability specific to the 0731 model variant on multi-node Spark — may relate
  to the known vLLM+llama.cpp MTP+prefix-cache interaction bugs (S-forum-mtp-lossless) or to
  multi-node KV cache eviction under memory pressure (S-forum-uvm-livelock). Single source →
  [conjecture]. Flagged for hardware verification: does prefix cache reliably hit on 2× Spark
  with DSV4-Flash-0731 under controlled conditions?

### Batch 52 forum ingest (2026-08-04)

- **[conjecture]** **DSML tool-call wrapper tag leaks at >60K context on DSV4-Flash-0731**
  (S-forum-dsv4-0731-bench, Teason2026, penguinchang): the `<｜DSML｜tool_calls>` wrapper
  marker sometimes gets skipped by the model at long context (>60K), causing the entire
  tool call to leak to the user as raw text instead of being parsed. The issue reproduces
  deterministically with specific context but occurs sporadically in live agent sessions
  (~once per 5-6 long-context sessions). A vLLM regression in 0.26.1rc1.dev244 worsens it;
  the 0.26.1rc1.dev30 (July 28) build did not leak but OOMs when loading DSpark weights.
  vLLM PR [#49117](https://github.com/vllm-project/vllm/pull/49117) adds recovery for missing
  wrapper markers (complete invoke marker in plain content starts a tool call, with
  declared-name guard against false matches), but at 150K context the model enters a
  completely broken state where no valid tool calls can be produced — the parser fix is
  necessary but not sufficient at extreme context. Workaround: an OpenAI-compatible proxy
  ([opencode_compat_proxy](https://github.com/ladiossoop5star/opencode_compat_proxy)) or
  LiteLLM hook translates raw DSML markup into structured `tool_calls`. Using
  `AidenProduction-3.75` image with the proxy, one user reports no leaks even after multiple
  context compactions (trigger ~450K). This is the same class of tool-call-parser issue as
  the GLM-5.2 `glm45` reasoning-parser leak (see `[[wiki/models/glm-5.2.md]]`) — DeepSeek's
  DSML format uses non-standard markers that vLLM's `deepseek_v4` parser must handle.
  Single thread, multiple users → [conjecture].
- **[conjecture]** **DSV4-Flash-0731 tool-eval-bench: 87/100** (S-forum-dsv4-0731-bench,
  serapis): Tool-Call Benchmark v2.3.2 on vLLM 0.25.2.dev0+g752a3a504 — 66 passed, 14 partial,
  4 failed (Prompt Injection Resistance, Async Polling, Simple Schema Compliance, +1).
  524,288 token max context. DSV4-Flash-0731 is the official GA release of DSV4-Flash,
  superseding the preview, with the same structure as DSV4-Flash-DSpark (includes
  speculative decoding module). No vision tower in this variant. Single source → [conjecture].
- **[conjecture]** **DSV4-Flash-0731 4-config benchmark table** (S-forum-dsv4-0731-bench,
  vedcsolution): multi-config comparison on Spark cluster (config and node count not fully
  specified in post):

  | Metric | TP=2 (2N, ref) | TP4-seqs32 ★ | DP4EP | TP2PP2 (Ray, no spec) |
  |---|---|---|---|---|
  | B1 e2e 512 tok | 35.3 | 46.8–48.6 (+33%) | 31.3 | 22.8 |
  | C4 | 65.6/69.5 | ~101 | 75.7/95.0 | ~56 |
  | C8 | — | 150–164 | ~105 | ~83 |
  | C16 | — | ~216 | ~203 | ~83 (saturation) |
  | C32 | — | 333–344 | ~233 | — |
  | Acceptance | 40.1% | 39.8–40% | 40–44% | n/a |
  | KV pool | 345K tok | 1.93–1.98M | 1.59M ×4 | 4.09M (7.81×) |

  TP4 with seqs=32 is the standout: +33% single-stream, 7.81× KV pool vs TP=2 reference,
  40% MTP acceptance maintained. DP4EP (data parallel × 4, expert parallel) reaches similar
  C16/C32 aggregate but lower per-stream. TP2PP2 (pipeline parallel) saturates at C16 ~83.
  Single source → [conjecture]. These numbers are consistent with the known bandwidth-bound
  decode ceiling and the TP=4 concurrency advantage on Spark.

### Batch 54 forum ingest (2026-08-05) — DSV4-Flash-0731 DSpark loader bugs

- **[conjecture]** **DSpark draft loader drops 12 shared-expert tensors silently —
  shared_experts.w1/w3 never mapped to gate_up_proj** (S-forum-dsv4-0731-dspark-loader,
  tonyd615): the vLLM DSpark draft loader renames `shared_experts.w2 → down_proj` but
  never maps `w1` / `w3 → gate_up_proj`. They match nothing →
  `logger.debug("Skipping unknown DSpark weight")` → invisible at INFO level. 12 tensors
  (the always-on shared expert, in all 3 draft stages) are lost — uninitialized. The
  target model's own loader has the two missing rows; they were dropped when the mapping
  was narrowed to dodge a `markov_w1` name collision. **Fix:**
  ```python
  ("shared_experts.gate_up_proj", ".shared_experts.w1", 0),
  ("shared_experts.gate_up_proj", ".shared_experts.w3", 1),
  ```
  Result: **32.7 → 55.4 tok/s mean (+69%), 66.1 peak**; acceptance 25.7% → 60.2%;
  per-position 0.63/0.28/0.18/0.11/0.07 → 0.83/0.73/0.57/0.47/0.40. Config: 2× Spark,
  TP=2, k=5, NVFP4 KV, 1M context. Single source → [conjecture]. This is a GB10-relevant
  finding because the DSpark spec-decode path is the primary high-throughput recipe for
  DSV4-Flash on 2× Spark — a silent draft-loader bug that halves throughput is a major
  operational trap.

- **[conjecture]** **SSE streaming under spec-decode measures steps/s, not tok/s**
  (S-forum-dsv4-0731-dspark-loader, tonyd615): under speculative decoding, vLLM emits at
  most one SSE chunk per decode step, carrying every token accepted that step. Counting
  stream deltas measures steps/s, not tok/s — the same request reads 14.7 (streaming)
  vs 60.1 (non-streaming) tok/s. **Benchmark with `stream: false`.** This is a general
  spec-decode measurement trap, not GB10-specific, but it bites every Spark user
  benchmarking DSpark/DSV4-Flash. Single source → [conjecture].

- **[conjecture]** **DSpark draft quantization-config inheritance collapses acceptance
  to ~1%** (S-forum-dsv4-0731-dspark-loader, srivatsa1): the DeepSeek-V4-Flash-0731-NVFP4
  checkpoint is a hybrid — target trunk uses ModelOpt NVFP4 experts (requiring
  `flashinfer_b12x` + `ModelOptNvFp4FusedMoE`), while MTP/DSpark draft stages use native
  FP8/MXFP4 experts (int8 weights + UE8M0 scales, requiring `Mxfp4MoEMethod`). vLLM PR
  [#49133](https://github.com/vllm-project/vllm/pull/49133) addresses this: the DSpark
  model-type rewrite can leave the draft `model_config.quantization` stuck at plain
  `"fp8"`, and the draft VllmConfig inherits the target's `quant_config`. The draft MoE
  is then built with `ModelOptNvFp4FusedMoE` even though the draft weights are not in
  ModelOpt format → `w1_weight_scale_2 must match w3_weight_scale_2` warnings, draft
  acceptance collapses to ~1.0–1.15 tokens/step, throughput capped at 14–18 tok/s.
  **Fix:** strip target-only ModelOpt keys (`moe_quant_algo`, `quantized_layers`, `ignore`,
  `modules_to_not_convert`) from the deep-copied draft `quantization_config` before the
  draft `quant_config` is derived; rewrite draft quantization from `"fp8"` to
  `"deepseek_v4_fp8"`. Single source → [conjecture]. This is a vLLM config-plumbing bug
 that bites on any mixed-quant DSpark checkpoint on Spark (and elsewhere).

 ## Forum ingest: DSV4-Flash-0731 on ds4 CUDA engine — single Spark 40 tok/s (2026-08-06)

 - **[conjecture]** **DSV4-Flash-0731 on ds4 CUDA engine (Entrpi/ds4 fork v0.5.4) — single
 Spark 40 tok/s decode, 131K context, IQ2XXS quant** (S-forum-dsv4-0731-ds4-cuda, styles01):
 a sparkrun-recipes runbook for serving DeepSeek-V4-Flash-0731 on a single DGX Spark via
 the ds4 custom CUDA engine (Bleys Goodson / antirez). This is a **native C/CUDA binary**
 (no Python/PyTorch), the same engine family as the original ds4/DwarfStar 4 (S-forum-ds4-cuda),
 now at fork v0.5.4. Key config:
 - **Quant:** IQ2XXS (2-bit) weights + Q2K KV + Q8 attention projection + Q8 shared experts
 - **Spec decode:** DSpark MTP k=2 (lossless speculative decoding)
 - **Context:** 131,072 tokens (configurable up to 1M)
 - **Decode:** ~40 tok/s single-stream
 - Env: `DS4_BATCH_FIT_HEADROOM_MB=8192`, `DS4_SERVER_COALESCE_MAX=32`,
   `DS4_CONT_DSPARK=1`, `DS4_CONT_MTP_MODE=2`, `DS4_METAL_GRAPH_RAW_CAP=131072`
 - **[conjecture]** coder543 (same thread) reports 1M context fits in ~107 GB with
   `DS4_CUDA_NO_HBM_CACHE=1`, `DS4_BATCH_FIT_HEADROOM_MB=6272`,
   `DS4_BATCH_VMM_BUDGET_MB=6144`, `DS4_SERVER_COALESCE_MAX=8`,
   `DS4_CONT_PREFILL_CHUNK=2048`, `DS4_CONT_CAPTURE=1`, `--kv-disk-dir` for KV cache
   offload (32 GB disk space). Command: `./ds4-server --cuda -m <model.gguf>
   --dspark <drafter.gguf> -c 1048576 --kv-disk-dir <path> --kv-disk-space-mb 32768`.
 This corroborates the existing ds4 engine finding (S-forum-ds4-cuda, ~28 tok/s Q2 on
 single Spark) — the v0.5.4 fork with DSpark k=2 and IQ2XXS quant achieves 40 tok/s, a
 ~43% improvement over the original Q2 baseline. The 1M-context single-Spark recipe is
 notable — DSV4-Flash at 1M context on a single 121 GB node is at the edge of feasibility
 with 2-bit quant + KV disk offload. Single source (one thread, 2 users) → [conjecture].

 ## Forum ingest: DSV4-Flash-0731-vision — vLLM vision plugin for DSV4 on 2× Spark (2026-08-08)

 - **[conjecture]** **FlyCockpit/DeepSeek-V4-Flash-0731-vision — vLLM plugin adding vision to
   DSV4-Flash-0731 on 2× Spark** (S-forum-dsv4-vision-plugin, co-le): a community vLLM plugin
   (`dsv4_vision_vllm`) that registers a wrapper model `DeepseekV4VisionForCausalLM` via the
   standard `vllm.general_plugins` entry point. Vision comes from a frozen 865 MB `DeepEncoderV2`
   tower + 40 MB trained projector adapter that maps tower features into the 0731 backbone's
   embedding space. The model directory is a zero-cost symlink tree of the 0731 snapshot with
   only `architectures` swapped in `config.json`. Validated on
   `aidendle94/sparkrun-vllm-ds4-gb10:production-3.7-reffix`. The plugin mechanism is plain vLLM
   plugin territory — no launcher or image internals involved.
   - **Key flags:** `--limit-mm-per-prompt '{"image":8}'`, `--trust-request-chat-template`
   - **Env:** `DSV4_VISION_TOWER=<path>/deepencoder_v2_tower.safetensors`,
     `DSV4_VISION_ADAPTER=<path>/adapter/latest.pt`
   - **Max 8 images per request** (counted across replayed history; 9th → HTTP 400)

 - **[conjecture]** **DSpark wrapper-transparency bug — vision wrapper breaks speculative decoding**
   (S-forum-dsv4-vision-plugin, co-le): the stock upstream plugin quietly breaks DSpark. The draft
   keeps running, but **acceptance collapses to 1-15%** and throughput drops to ~20 tps. Root cause:
   the vision wrapper hides the backbone, cutting off the auxiliary hidden-state flow the DSpark
   draft feeds on. **Fix:** keep the wrapper transparent to the backbone — pass `**kwargs` through
   in `forward()` and expose an `lm_head` property that forwards to the language model. After fix:
   **acceptance recovers to 50-64%** with mean acceptance length ~2.0. The broken state is
   recognizable in logs: `SpecDecoding metrics: Per-position acceptance rate: 0.0x, 0.0`. This is
   a general pattern for any vLLM vision wrapper that intercepts the backbone's forward path while
   DSpark speculative decoding is active — the draft model needs access to the backbone's
   hidden states.

 - **[conjecture]** **Image requests must send `chat_template_kwargs: {"thinking": false}`**
   (S-forum-dsv4-vision-plugin, co-le): the recipe defaults to `thinking:true`, but on image input
   the model answers without thinking — the answer lands inside an unclosed think block and `content`
   comes back empty (the text ends up in the `reasoning` field). This is a DSV4-Flash-0731-specific
   chat-template interaction that bites when adding vision to the existing DSpark recipe.

 - **[conjecture]** **tiles=2 token layout — image = n_views×256+1 tokens (257/769/1281)**
   (S-forum-dsv4-vision-plugin, co-le): under the `tiles=2` layout, an image expands to
   `n_views×256+1` tokens. Verify the layout is active by checking
   `[dsv4-vision] checkpoint config.tiles=2` in the logs; a `tiles=0` fallback silently serves
   the wrong token layout.

 - **[conjecture]** **Vision quality assessment — strong for screenshots/UIs, weak for general
   photos, not ready for click-agents** (S-forum-dsv4-vision-plugin, co-le): screenshots, UIs,
   and on-screen text → strong (near-perfect transcription in tests). Documents → strong.
   Everyday photos → decent but generic — "this is a screenshot specialist, not a general-purpose
   vision-language model." Click-agents and real-world computer use → **not ready** (explicitly
   unclaimed by the upstream project). The OP reports Gemma 4 E2B vision is better as a
   general-purpose vision model than this encoder.

 - **[conjecture]** **Throughput: ~40-50 tps after clean reboot (below 40 before)**
   (S-forum-dsv4-vision-plugin, co-le): on 2× DGX Spark TP=2 with the wrapper-transparency fix
   applied, DSpark acceptance 50-64% with mean acceptance length ~2.0, ~40-50 tps after a clean
   reboot. DSpark stays engaged even on image requests (~63% acceptance). The reboot measurably
   helped — consistent with the UMA fragmentation / power-state findings on platform-gb10.
   Before the fix: ~20 tps. This is below the 55-66 tps reported for non-vision DSV4-Flash-0731
   DSpark on 2× Spark (S-forum-dsv4-0731-dspark-loader), suggesting the vision plugin adds
   ~20-30% throughput overhead even when working correctly.

 - **[conjecture]** **webbrain-one/DeepSeek-V4-Flash-0731-Vision-NVFP4 — 9 GB NVFP4 vision variant**
   (S-forum-dsv4-vision-plugin, mikeyb222, james.park4): an alternative NVFP4 vision checkpoint
   that adds ~9 GB to the weights (vs ~900 MB for the FlyCockpit encoder). Potentially better
   quality due to larger vision component, but the additional 9 GB is challenging given the
   "constant struggle with memory constraints" on 2× Spark. Not yet tested.

 ## Forum ingest: vllm-snapshot — fast model suspend/restore on GB10 (2026-08-10)

 - **[conjecture]** **vllm-snapshot plugin — byte-for-byte weight snapshot enables ~1.6s restore /
   ~9s full model swap on GB10/SM121** (S-forum-vllm-snapshot, david.gareth.roberts): a native vLLM
   plugin (`dgr237/vllm-snapshot`) that adds `/suspend` and `/restore` endpoints to vanilla vLLM's
   server. vLLM's built-in sleep mode (`/sleep?level=2`) frees weight memory but a level-2 wake
   must **re-run `reload_weights`** — this is a **processing wall, not disk I/O**:
   - **~82s** for a 26B NVFP4 MoE via safetensors loader
   - **~30 min** via instanttensor loader
   - Faster loaders don't help because the bottleneck is CPU-side tensor reconstruction, not disk
     read.
   The plugin instead **snapshots the built weight regions byte-for-byte to disk** on suspend, then
   restores them with a **bulk `cudaMemcpy`** on wake:
   - **~1.6s restore** (weight regions only)
   - **~9s full swap** (including CUDA graph teardown/rebuild)
   - Correct and reproducible.
   Additional features: `autosuspend` option (auto-suspend on idle timeout) + Docker Compose
   `depends_on` pattern so N models boot one-at-a-time instead of OOM-ing the 121 GB unified pool.
   Research prototype, only validated on GB10/SM121 so far. **Why it bites on Spark:** the 82s
   reload wall is a direct consequence of unified memory — on a discrete-GPU server, level-2 sleep
   + wake is fast because weights load from host RAM over PCIe at ~30 GB/s; on GB10, the "host RAM"
   IS the GPU memory, so the reload path re-runs the full tensor construction pipeline. This is
   the same UMA constraint that makes `cudaMemGetInfo` under-reporting (S-forum-comfyui-optimized)
   and single-CUDA-context serialization (S-forum-cuda-single-ctx) bite. The plugin's
   snapshot-to-disk approach sidesteps the reconstruction wall entirely. Related to the existing
   multi-model co-hosting findings (S-forum-llm-comfyui, S-forum-woolyai).

 ## Forum ingest: DSV4-Flash-0731 GGUF + sparkrun packaging (2026-08-10)

 - **[conjecture]** **DeepSeek-V4-Flash-0731 GGUF (Unsloth release) — UD-Q8_K_XL 162GB lossless,
   UD-IQ2_M runs on single Spark via llama.cpp** (S-forum-dsv4-0731-gguf, vincenzoa, chriswalz86):
   Unsloth published GGUF quants for DSV4-Flash-0731 (284B params, 13B active, 1M context window).
   - **UD-Q8_K_XL** (162 GB) is described as "full precision lossless" — only 7 GB larger than
     UD-Q4_K_XL. Too large for a single 121 GB Spark; needs 2× Spark via llama.cpp RPC or a
     single Spark with aggressive 2-bit quants.
   - **UD-IQ2_M** runs on a single Spark via `llama-server`:
     `--n-gpu-layers 999 --flash-attn on --ctx-size 262144 --parallel 2 --batch-size 2048
     --ubatch-size 512 --jinja --reasoning off --no-repack --cache-type-k f16 --cache-type-v f16
     --temp 0.6 --top-p 0.95 --top-k 0 --min-p 0.0`
     Works without issues. The `--no-repack` flag is notable — it disables llama.cpp's default
     repacking of quantized tensors, which may be specific to the Unsloth GGUF format.
   - **MJPansa/DeepSeek-V4-Flash-0731-NVFP4** exists as a community NVFP4 variant (not yet
     benchmarked on Spark in this thread).
   - **MTP not yet compatible** on community vLLM for this model (anotheralvin).
   The IQ2_M single-Spark recipe is consistent with the existing
   `[conjecture]` DSV4-Flash-0731 UD-IQ2_M finding (S-forum-dsv4-llamacpp-fan, 16.2 tok/s tg32
   on HP ZGX) — same quant, same flags, different poster.

 - **[conjecture]** **DSV4-Flash-0731 DSpark packaged for sparkrun — 58 tps agentic/coding on 2×
   Spark** (S-forum-dsv4-0731-sparkrun, david735): the tonyd2wild DSpark 1M NVFP4 KV recipe
   (already documented as S-forum-dsv4-0731-dspark-loader) has been packaged for sparkrun via
   `brainchillz/sparkrun-dspark-registry`. The poster reports **58 tps on agentic/coding tasks**
   across 2× Spark. This is a packaging/automation derivative of the existing recipe — the
   underlying recipe, flags, and performance characteristics are already documented. The 58 tps
   figure is consistent with the 55.4 tok/s mean / 66.1 peak reported in the original
   DSpark loader fix thread. No new GB10-specific findings beyond the packaging.

## Forum ingest: LMCache 0.5.3 IPC deadlock with aidendle94 DS4F fork (2026-08-11)

> **evidence:** conjecture (single forum post, one follow-up)
> **sources:** S-forum-lmcache-ipc-deadlock

- **[conjecture]** **LMCache 0.5.3 MP mode deadlocks with aidendle94's DS4F fork — version gap
  between fork's vLLM 0.11.x IPC surface and LMCache 0.5.3's vLLM 0.18/0.20+ adapters**
  (S-forum-lmcache-ipc-deadlock, mxjohnwong): attempted to add LMCache 0.5.3 (MP mode)
  cross-instance KV sharing on top of `aidendle94/sparkrun-vllm-ds4-gb10:production-3.75`
  (vLLM 0.11.2.dev279) running DeepSeek-V4-Flash-0731 on 4× Spark (2 clusters of 2 nodes each,
  TP=2 per cluster). The LMCache server successfully registers the KV cache (170 layers for
  DS4F's sparse-MLA hybrid KV groups with 256/64/8/4 block sizes), but vLLM hangs at
  "Wrapping 170 KV cache tensors for IPC" and times out after 300s. Root cause is a version
  gap: vLLM 0.11.x (2025-10 era) pairs with LMCache 0.3.x, but 0.3.x has no DS4F hybrid KV
  support; hybrid support first appears in LMCache 0.4.7+ which targets vLLM 0.18/0.20+.
  **No LMCache version matches both the fork's 0.11.x IPC surface AND DS4F hybrid KV.**
  GB10-specific relevance: this is a UMA-cluster KV-cache-sharing attempt — the 128 GB pool
  per node makes cross-instance KV sharing attractive for extending effective context across
  clusters without doubling model weights. Operational notes: `--disable-hybrid-kv-cache-manager`
  is a red herring (breaks startup — KV demand becomes 160 GB > available); container needs
  `--gpus all` (Device UUID match), `--ipc host --shm-size 16g` (L1 SHM), and `cupy` installed.
  The Mooncake connector (also shipped by the fork) natively handles hybrid KV but is likewise
  unverified against this fork's old IPC interface. Status: `open` — no working LMCache +
  DS4F fork configuration known. Single source → [conjecture]. See also the existing
  `[conjecture]` LMCache-for-dedicated-KV-node finding
  (`[[wiki/multinode-tp-and-networking.md]]`, S-forum-3node-mesh) — both are untested
  LMCache-on-GB10 configurations.
