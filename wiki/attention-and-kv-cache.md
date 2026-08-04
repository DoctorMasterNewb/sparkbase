# Attention backends & KV cache on GB10

> **area:** attention
> **status:** stable
> **evidence:** proven
> **sources:** S-m3-vision, S-mimo-results, S-mimo-doc, S-sess-jun5, S-sess-jun4, S-dflash-nvfp4, S-forum-mimo-2x-opt, S-forum-dsv4-kvcache, S-forum-inkling-nvfp4, S-forum-flashinfer-livelock, S-forum-solar-open2-nvfp4, S-forum-glm52-hybrid, S-forum-nvfp4-kv, S-forum-glm52-3x-aqlm
> **updated:** 2026-08-04

Which `--attention-backend` to pass is decided by the model's attention type, not preference. Get it
wrong and KV-cache init fails or numerics are subtly off.

## Backend selection

| Backend | Use when | Notes on GB10 |
|---|---|---|
| **TRITON_ATTN** | sparse / MSA / block_size 128 archs (MiniMax-M3 MSA) | **required**, not optional, for MSA — supports `block_size=128` and is numerically correct on Blackwell. |
| **TRITON_ATTN_DIFFKV** | DiffKV archs where `v_head_dim ≠ head_dim` (MiMo-V2.5: v=128, head=192) | the MiMo path; from vLLM PR #41797, native in dev39-era images. |
| **FLASHINFER** | standard dense/MoE attention (Qwen3.x, Holo, Gemma-4) | the common fast path; default for ModelOpt NVFP4. |
| FlashAttention (ViT) | vision towers | first image triggers a one-time JIT autotune (~20 s) — see below. |

## Hard rules

- **[proven]** **`--block-size 128` + `TRITON_ATTN` for MSA / sparse-attention models.** FLASHINFER for
  the dense layers has **no common block size** with the MSA sparse/index cache → KV init fails ("No
  common block size for 128"). TRITON_ATTN supports 128. (MiniMax-M3.)
- **[proven]** **DiffKV needs TRITON_ATTN_DIFFKV.** `v_head_dim ≠ head_dim` checkpoints (MiMo-V2.5) won't
  run on a vanilla backend.
- **[proven]** **KV-cache dtype `fp8` (e4m3) is the default and is precise enough.** bf16 KV is *more*
  precise but rarely needed; on GB10 KV is cheap for sparse/hybrid archs anyway (few KV heads). MiMo uses
  `--kv-cache-dtype fp8`; M3's MSA + 4 KV heads make even 40k+ ctx cheap.
- **[proven]** **Sparse/hybrid attention makes long context cheap.** MSA (M3, 4 KV heads), Mamba-2
  hybrids (Nemotron-3: most layers SSM, few attn layers w/ 2 KV heads) → huge context for little memory
  (Nemotron-3 ran the model's full 1M ctx × 4 slots). The KV pool is rarely the constraint; **weights
  are**.

## Gotchas

- **[proven]** **First image request = ~20 s ViT JIT autotune** (`AttentionBackendEnum.FLASH_ATTN for vit
  attention`). Short client timeouts make the first call look like it "returned nothing." Use a long
  timeout (>150 s) or send a warmup image; every request after is fast. (M3 vision, Holo vision.)
- **[proven]** **flashinfer gemma-rmsnorm CUTLASS-DSL ICE.** Archs with `use_gemma_norm=true` call
  `flashinfer.norm.gemma_rmsnorm`, whose CUTLASS-DSL kernel can fail MLIR verification on cu130
  (`'llvm.mlir.global_dtors' requires attribute 'data'`). Fix: swap in pure-torch norms (verified
  coherent). Was misdiagnosed as a vision bug — it's a text-path norm. (MiniMax-M3.)
- **[proven]** **`--dtype bfloat16` may be mandatory** even when the checkpoint config says `float32`
  (Holo FP8/NVFP4 configs declare float32 — force bf16 activations or the launch is wrong).
- **[proven]** **Atlas KV flags differ** from vLLM (no `--kv-cache-gb`): `--max-seq-len`, `--block-size`,
  `--kv-cache-dtype`, `--kv-high-precision-layers`, `--ssm-cache-slots`, etc. See `[[wiki/engines.md]]`.

- **[proven]** **Spec-decode drafter + QUANTIZED target KV = page-unification wall.** A native spec
  proposer (DFlash/EAGLE3/MTP draft-model) registers the drafter's KV layers into the *global* paged
  allocator, which requires ONE physical page size across all layers. A bf16 drafter page can't unify with
  a 4-bit **nvfp4** target page (non-integer ratio; and NO GB10 backend implements the strided-padded-read
  `indexes_kv_by_block_stride` path — it's `False` on every backend) ->
  `NotImplementedError: The page size of the layer is not divisible by the maximum page size` in
  `unify_kv_cache_spec_page_size`. **Workaround:** run the drafter as a **`custom_class`** proposer so its
  KV is never registered globally (drafter runs standalone) -> the quantized target keeps its full pool.
  `fp8` (8-bit) target unifies more cleanly (2x bf16 ratio) so the native path can work there. Status:
  `workaround` — see `[[wiki/models/mimo-v2.5.md]]` (DFlash). Reusable for any spec-decode + quantized-KV
  combo on GB10.
- **[proven]** **Deep-context coherence tests need REALISTIC text.** Degenerate/repetitive filler prompts
  make models emit immediate-EOS at depth (a real behavior on pathological input) — this looks exactly like
  KV-quant corruption but is NOT. nvfp4 KV is coherent to 89k+ with varied content. Always test depth with a
  real corpus (`llama-benchy` uses a Gutenberg book for this).

## Forum ingest: hybrid-linear attention & FP8-KV capacity (2026-07-26)

- **[conjecture]** **FP8 KV on hybrid-linear models is a capacity lever, not a speed lever**
  (S-forum-solar-open2-nvfp4, danielgbates): on Solar Open2 250B (36/48 KDA linear-attn layers),
  FP8 KV is speed-neutral vs bf16 KV (15.8 vs 15.8 tok/s decode c1 d0) because only 12/48 layers
  touch KV — attention is a thin slice of decode time. What FP8 KV buys instead is 2× pool:
  2,665,802 tokens vs ~1.33M at 262K max-len (10.17× concurrency). This **contrasts** with the
  proven finding that fp8 KV is a decode-speed lever at depth on full-attention models
  (S-dflash-nvfp4: ~2× vs 4-bit KV at 200K depth on MiMo). The distinction: on full-attention
  archs, KV grows with context and dominates decode bandwidth → halving KV bytes halves that
  cost; on hybrid-linear archs, the linear layers don't materialize per-token KV, so the
  attention cost is already small and halving it changes nothing. **GB10 rule of thumb:** choose
  FP8 KV for *speed* on full-attention models at long context; choose FP8 KV for *capacity* on
  hybrid-linear models. vLLM handles the mixed page layout (mamba + attention pages) by padding
  the mamba page size 0.38% to keep both equal. Single source → [conjecture].
- **[conjecture]** **Hybrid linear attention makes decode ~flat with context depth on Spark**
  (S-forum-solar-open2-nvfp4, danielgbates): Solar Open2 decodes at 15.4 tok/s at 32K context depth
  vs 15.8 tok/s at depth 0 — only ~2.5% degradation. Every full-attention model on the same 2×
  Spark pair decays hard with context (a 310B MoE with NVFP4 KV drops to ~9 tok/s by 100K).
  This generalizes the proven finding that sparse/hybrid attention makes long context cheap
  (S-sess-jun5: Nemotron-3 Mamba-2 hybrid ran full 1M ctx × 4 slots) to a 4th architecture class
  (KDA linear attention, after MSA sparse, Mamba-2 SSM, and Holo's hybrid linear+full). **Why it
  bites on Spark:** the proven bandwidth-bound decode ceiling (~270 GB/s) means full-attention
  KV grows linearly with context → decode slows; hybrid-linear architectures that don't
  materialize per-token KV sidestep the wall entirely. See `[[wiki/benchmarks.md]]` → Batch 36.

## Forum ingest: NVFP4 vs FP8 KV cache capacity on GB10 (2026-07-29)

- **[conjecture]** **NVFP4 KV cache gives 1.68× more capacity than FP8 on DGX Spark**
  (S-forum-nvfp4-kv, shahizat): using SGLang with `Qwen/Qwen3-4B` on a single Spark, NVFP4 KV
  cache (`--kv-cache-dtype nvfp4`) allocates **2,309,504 tokens** vs FP8 (`--kv-cache-dtype
  fp8_e4m3`) at **1,371,456 tokens** — a 1.68× capacity increase. The NVFP4 KV dtype is
  `torch.float4_e2m1fn_x2`. On RTX PRO 6000 Blackwell the ratio is similar (1,808,192 vs
  1,067,328 = 1.69×). Launch config: `--prefill-attention-backend flashinfer
  --decode-attention-backend trtllm_mha --disable-radix-cache`. Production deployments should
  validate model quality and task-specific accuracy before enabling aggressive KV cache
  quantization. Single source → [conjecture]. This corroborates the existing finding that
  lower-bit KV cache increases token capacity on GB10 — the same principle as fp8 vs bf16 KV
  (2× capacity), now extended to NVFP4 vs fp8 (1.68×). See `[[wiki/benchmarks.md]]` for the
  full benchmark numbers.

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/platform-gb10.md]]` · `[[wiki/models/minimax.md]]` · `[[wiki/models/mimo-v2.5.md]]`

## Forum ingest: TRITON_ATTN_DIFFKV quantized KV guard (2026-07-11)

- **[conjecture]** **`TRITON_ATTN_DIFFKV` raises `NotImplementedError` on quantized
  `kv_cache_dtype`** (S-forum-mimo-2x-opt, renek): the backend's guard rejects any quantized KV
  cache dtype, but the underlying store kernel already accepts dtype + scales. The guard is
  defensive — disabling it allows `--kv-cache-dtype fp8_e4m3`, which roughly doubles the KV pool.
  This conflicts with first-party findings where fp8 KV works fine with TRITON_ATTN_DIFFKV
  (proven on dev39+), suggesting the guard was relaxed in later vLLM versions. If you hit this
  error on an older build, patching the guard is the workaround.

## Forum ingest: DSV4-Flash KV cache sizing on Spark (2026-07-14)

- **[conjecture]** **DSV4-Flash KV cache ~15 GB/1M tokens/node on 2× Spark** (S-forum-dsv4-kvcache,
  paxren2020): user running DeepSeek on a dual-node cluster reports ~15 GB VRAM per 1M context
  tokens per node (~30 GB total across 2 nodes, ~15 GB allocated on each). vLLM logs show
  `Available KV cache memory: 26.15 GiB`, `GPU KV cache size: 1,687,476 tokens`, `Maximum
  concurrency for 500,000 tokens per request: 3.37x` at `--gpu-memory-utilization 0.90`. This is
  significantly larger than online KV cache calculators (kvcache.ai) and Reddit formulas predict
  (~5× smaller). Possible explanations: vLLM overhead/fragmentation, CUDA graph memory
  profiling reservation, or the calculators modeling a different attention variant. (S-forum-dsv4-kvcache)
- **[conjecture]** **CUDA graph memory profiling overhead** (S-forum-dsv4-kvcache): vLLM v0.21.0+
  enables CUDA graph memory profiling by default. The log message shows that
  `--gpu-memory-utilization=0.9000` is equivalent to `0.8943` without profiling — the profiling
  reserves ~0.6% of usable memory. To maintain the same effective KV cache size, increase to
  `--gpu-memory-utilization 0.9057`. Disable profiling with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. This overhead is on top of the standard
  cudagraph capture memory cost. (S-forum-dsv4-kvcache)

## Forum ingest: tml_fa4 Sm120 path has no paged-KV (2026-07-20)

- **[conjecture]** **The `tml_fa4` Sm120/Sm121 cute FlashAttention path has no paged-KV support**
  (S-forum-inkling-nvfp4, greg190): vLLM's paged cache cannot feed the `tml_fa4` Sm120 kernel. The
  Inkling 8× Spark bring-up worked around this by **re-gathering the whole KV history into contiguous
  buffers every decode step** (patched `fa4_rel_attention` to gather paged KV → contiguous per call).
  This workaround is O(context) per token → decode caps at ~24 tok/s aggregate at real context
  regardless of concurrency, producing a steep long-context decode cliff (25 tok/s @ ~100 tok →
  13.5 tok/s @ 2048, c1). **Why it bites on Spark:** until paged-KV lands natively in the Sm120/Sm121
  cute FA4 path, any model routing through `tml_fa4` on GB10 will hit this cliff. This is the
  load-bearing blocker for Inkling (and likely other rel-bias / FA4-arch models) on sm_121a. The OP
  also notes the Sm80-inherited rel-bias path **discards the relative-position bias** (no bias
  parameter) → plausible-but-wrong outputs; the **score-mod `vllm_flash_attn/cute` path is the
  intended sm12x route.** See `[[wiki/models/inkling.md]]`. Single source → [conjecture], but a
  filed upstream issue (vllm#49049) and public patch set accompany it.

## Forum ingest: FlashInfer sparse-MLA mbarrier livelock on GB10 (2026-07-21)

> **[conjecture]** **Symptom → Root cause → Workaround → Status: `open` (upstream)** — A major
> kernel bug on GB10. The FlashInfer `sparse_mla_sm120` prefill/decode kernels hard-wedge one rank
> GPU under cold-prefill load. The workaround (Triton sparse-MLA) is validated in production.

- **[conjecture]** **FlashInfer `sparse_mla_sm120` kernels livelock in mbarrier TRYWAIT on GB10/sm_121
  under cold-prefill load** (S-forum-flashinfer-livelock, msunner): on a 4-node DGX Spark (GB10,
  NVRM 580.159.03, kernel 6.17.0-1026-nvidia, vLLM 0.23.1rc1.dev893, flashinfer JIT `sparse_mla_sm120`
  kernels), serving GLM-5.2 (DeepSeek-V4-class sparse MLA, `fp8_ds_mla` cache, TP=4), any
  cold-prefill-heavy request probabilistically **hard-wedges one rank GPU**: the device spins forever
  inside a sparse-MLA kernel, the host launch queue fills, and every host thread ends blocked in
  `cuLaunchKernel`.
  - **Probability scales with cold-prefill size**: ≥60K-token cold prefills wedged ~always (8/8
    observed over one day, two served-context configs 200K and 120K); 120-token cold prefills (no
    prefix-cache hit) wedged occasionally (1/3); per-step decode work (M≤24 tokens/step) **never
    wedged** (>120,000 rank-steps fleet-wide across boots). Prefix-cache-warm prefills (small
    computed suffix) never wedged.
  - **Root cause (cuda-gdb attach on live wedge, rank 0)**: a single resident block spin-looping on
    an mbarrier phase check (`SYNCS.PHASECHK.TRANS64.TRYWAIT`) whose expected phase never arrives —
    consistent with a TMA/cp.async.bulk expect-tx accounting race or a producer warp/block that
    already exited. The spinning block shows 96% GPU utilization, 0% memory utilization, ~18 W,
    clocks P0/2535 MHz — a spin loop, not compute. All host threads blocked in `cuLaunchKernel`
    (queue full behind the spinning kernel). NCCL RAS collective counts freeze (pending TP
    all-reduce never launches). RoCE exonerated: all hardware counters zero during wedge.
  - **Independent of**: free UMA (wedges with 5–7 GB and ~1 GB avail), `max_num_batched_tokens`
    (8192 and 2048), served ceiling (200K and 120K), `max_num_seqs` (6 and 3), victim rank
    (rank-agnostic), drafter-gate instrumentation. A second wedge occurred through the FP8 decode
    kernel of the same family (mixed-batch mode) — the race is not exclusive to the BF16 prefill
    kernel.
  - **Workaround (validated in production)**: route the main model's attention to the **portable
    Triton sparse-MLA implementations** (`--attention-backend FLASHMLA_SPARSE` + the sm12x Triton
    drop-in patch stack from the jasl/vllm `deepseek_v4` path). The Triton kernels have **no
    inter-block mbarrier/TMA dependencies** — each program is self-contained — so this livelock
    class has no mechanism there. **560+ context-ceiling sessions clean** post-workaround (including
    500 consecutive at seq 119,997–120,000, plus a ~15 h unattended overnight), plus a clean cold
    staged climb to 199,872 tokens and boundary completion at exactly 200,000 tokens — the same
    cold-prefill workload that wedged 8/8 on flashinfer completes routinely on Triton, at **no
    decode-throughput cost** (~25–27 tok/s at 120K boundary, ~26 tok/s at 200K, vs ~23 tok/s
    flashinfer baseline).
  - **Asks**: review `sparse_mla_sm120` prefill/decode mbarrier expect-tx logic for sm_121/GB10
    (source: `data/csrc/sparse_mla_sm120*.cu`,
    `include/flashinfer/attention/sparse_mla_sm120/prefill_kernel.cuh`). A separately-filed GB10
    0x51 UMA memdesc leak remains unfixed and independent. Evidence pack at
    `marksunner/glm52-dgx-spark-deadlock-evidence` (public excerpts).
  - Single source, but exceptionally well-evidenced: cuda-gdb device-side receipt, journaled
    per-rank engine-step totals (~30,000+ steps/rank), multiple capture bundles, sanitized
    excerpts. → **[conjecture]** (no hardware verification in sparkbase), but the evidence quality
    is close to what would justify `[reported]` if a second independent source confirms.
  - **Why it bites on Spark**: this is a GB10-specific kernel livelock in the `sparse_mla_sm120`
    path (the sm_121 JIT-compiled FlashInfer kernel for sparse/MLA attention). Any model using
    sparse MLA on GB10 through FlashInfer (GLM-5.2, DeepSeek-V4-class, future MLA models) under
    cold-prefill workloads is at risk. The Triton workaround is drop-in and has no throughput
    penalty, making it the recommended path until the upstream mbarrier bug is fixed.

## Forum ingest: FlashInfer sparse-MLA decode dispatch table — head-count tiling (2026-08-04)

- **[conjecture]** **FlashInfer `_DECODE_DSV3_2_DISPATCH` only instantiates specific head counts;
  non-matching counts fall through to generic paged-attention tiled in groups of 16**
  (S-forum-glm52-3x-aqlm, karol.spark + MiaAI-Lab): the FlashInfer sparse-MLA decode dispatch
  table carries only `{8, 16, 32, 64, 128} × {128, 512, 1024, 2048}` (head count × head dim).
  Local head counts not in this table (e.g. 22 at TP=3, 13 at TP=5) fall through to the generic
  `sparse_mla_sm120_paged_attention` kernel, which **tiles heads in groups of 16**. The effective
  attention cost is therefore `ceil(local_heads/16)` tiles, not `local_heads` — so `ceil(22/16) ==
  ceil(32/16) == 2` means 22 padded heads cost the same attention time as 32, while the
  q_b/kv_b/o_proj GEMMs shrink by 31%. **Why it bites on Spark:** this dispatch rule determines
  the cost of non-power-of-2 TP padding for any sparse-MLA model on GB10. At TP=3, padding to 66
  (22/rank) instead of 96 (32/rank) saves 31% on GEMMs at zero attention cost. At TP=5, 13/rank
  means `ceil(13/16) = 1` tile (even cheaper attention) but 9.4% MoE padding waste. Padding to 80
  (16/rank) would land on the fast specialized kernel — but at 25% ghost heads. Which wins (fewer
  ghost heads + generic kernel vs more ghost heads + fast kernel) is unmeasured. See
  `[[wiki/models/glm-5.2.md]]` → NVFP4+AQLM 3× section for the full TP padding table.
  Single source → [conjecture].

## Forum ingest: b12x sparse-MLA KV format constraint (2026-07-27)

- **[conjecture]** **b12x `B12X_MLA_SPARSE` kernel only reads packed `fp8_ds_mla` KV pages — bf16 KV
  returns immediate EOS** (S-forum-glm52-hybrid, mclenithan): on a 6× GB10 cluster (TP=6, DCP varied)
  running GLM-5.2 with the b12x sparse-MLA backend, the kernel only understands the packed fp8 DSA
  page format (`fp8_ds_mla`). With bf16 KV pages, the kernel misreads memory and every request returns
  an immediate EOS. The alternative backend that would accept non-packed KV (`FLASHMLA_SPARSE`) has
  **no shipped sm12x sparse kernels**, so the fp8-vs-bf16 KV comparison could not be completed.
  This is the same kernel maturity gap as the FlashInfer livelock finding above: the sparse-MLA
  attention path on sm_121 has limited backend options and KV format support. See
  `[[wiki/models/glm-5.2.md]]` → KV cache kernel constraint.
