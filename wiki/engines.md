# Inference engines on GB10: vLLM vs Atlas vs llama.cpp

> **area:** containers
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-nemotron-rpc, S-mimo-results, S-forum-atlas, S-forum-ds4-cuda, S-forum-dflash-qwen122, S-forum-ddtree-dflash, S-forum-stream-loading, S-forum-turboquant, S-forum-vllm-019-vs-023, S-forum-sm121-kernel-guide, S-forum-easy-vllm, S-forum-tokenspeed, S-forum-dsv4-vision, S-forum-llm-comfyui, S-forum-colibri-glm52, S-forum-dsv4-abliterated, S-forum-mtp-lossless, S-forum-woolyai, S-forum-gridbook, S-forum-glm52-vision, S-forum-glm52-hybrid, S-forum-speedycolibri, S-forum-dsv4-reap25, S-forum-velogb10, S-forum-dsv4-dspark-eugr, S-forum-dsv4-0731-caching
> **updated:** 2026-08-03

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
