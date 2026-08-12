# GLM-5.2 (Zhipu AI) — on DGX Spark

> **area:** model
> **status:** evolving
> **evidence:** conjecture
> **sources:** S-forum-glm52-4x, S-forum-glm52-mtp-fix, S-forum-glm52-1bit, S-forum-glm52-reapless, S-forum-glm52-800k, S-forum-glm52-iq4xs-4x, S-forum-glm52-8x, S-forum-glm52-vision, S-forum-glm52-hybrid, S-forum-flashinfer-livelock, S-forum-colibri-glm52, S-forum-6x-cluster, S-forum-glm52-3x-aqlm, S-forum-sparkring, S-forum-glm52-8x-nvfp4
> **updated:** 2026-08-12

**GLM-5.2** is a 744B-parameter / ~40B-active MoE with sparse-MLA (DeepSeek-V4-class) attention and
MTP speculative decoding support. It is one of the most-discussed large models on the DGX Spark forums
because it pushes the limits of what 4–8× GB10 clusters can serve. This page consolidates the durable
GB10-specific findings across all community threads. All claims are **[conjecture]** (single source) or
**[reported]** (multiple independent sources) — no first-party hardware verification exists in sparkbase.

## Why it bites on Spark

GLM-5.2 exercises three GB10-specific pressure points simultaneously:
1. **Sparse-MLA attention** — the `sparse_mla_sm120` FlashInfer kernel path has a known livelock bug
   on sm_121 under cold-prefill (see `[[wiki/attention-and-kv-cache.md]]`).
2. **744B total at 4-bit ≈ ~370–460 GB** — needs 4× Spark minimum (TP=4) or 8× (TP=8) for headroom.
3. **MTP + quantized weights** — the draft model must use the same quant mapping or acceptance
   collapses silently (see `[[wiki/quantization-on-gb10.md]]`).

## Hybrid FP8+NVFP4+MXFP4 quant recipe (2026-07-27 ingest)

- **[conjecture]** **Hybrid-precision GLM-5.2 checkpoint mixes FP8, NVFP4, and MXFP4 across layers**
  (S-forum-glm52-hybrid, aidendle94): a community-built checkpoint (`aidendle94/GLM-5.2-Hybrid-FP8-MXFP4`)
  that assigns different quant formats to different layer groups — FP8 for some layers (from RedHat
  AI), NVFP4 for others, and MXFP4 for the experts that would otherwise be FP3 (MXFP4 is more compact,
  taken from AMD). The goal: more weight savings than pure FP8 without the precision hit of pure
  NVFP4/FP3. Published on HuggingFace with a custom Docker image
  (`aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.2`, later `1.3`).
  - **Reported perf (OP, 4× Spark):** ~800 tok/s prefill at ~100k depth, ~25 tok/s decode on prose,
    ~800K context.
  - **Second reporter (CosmicRaisins):** ~800 tok/s prefill, ~20 tok/s decode — slightly lower decode
    attributed to prose-content difference and lower llama-benchy acceptance on default prose. Also
    noted the image includes **adaptive speculative depth** (draft depth adapts to workload).
  - **Benchmark table (alexander.korolev.germany, llama-benchy v0.4.0, 4× Spark TP4+DCP4):**

    | test | conc | pp tok/s | tg tok/s | TTFT (ms) |
    |---|---|---|---|---|
    | pp2048 tg128 @ d0 | c1 | 1,605 | 20.1 | 3,355 |
    | pp2048 tg128 @ d0 | c2 | 861 | 14.9 | 3,182 |
    | pp2048 tg128 @ d0 | c4 | 324 | 17.8 | 8,854 |
    | pp2048 tg128 @ d4096 | c1 | 887 | 19.8 | 7,836 |
    | pp2048 tg128 @ d4096 | c2 | 792 | 12.6 | 10,491 |
    | pp2048 tg128 @ d4096 | c4 | 564 | 12.7 | 20,087 |
    | pp2048 tg128 @ d8192 | c1 | 933 | 18.6 | 10,877 |
    | pp2048 tg128 @ d8192 | c2 | 851 | 11.7 | 16,536 |
    | pp2048 tg128 @ d8192 | c4 | 661 | 9.9 | 30,397 |

    Engine: vLLM 0.11.2.dev279+eldritch.final (b12x, CUDA 13.2, 2026-06-26 build). Decode ~20 tok/s
    single-stream at shallow context, degrading with depth and concurrency. Prefill strong at shallow
    depth (1,605 tok/s c1) but drops sharply with concurrency (324 tok/s at c4).
  - **Two independent reporters** (OP + CosmicRaisins) agree on ~800 tok/s prefill and ~20-25 tok/s
    decode → decode figure approaches **[reported]**, but both are in the same thread and using the
    same image → stays **[conjecture]** per the independence requirement.

- **[conjecture]** **Custom NVFP4 KV cache implementation with scaling and calibration**
  (S-forum-glm52-hybrid, aidendle94): the Docker image includes a custom NVFP4 KV cache implementation
  designed to retain precision, built on top of work by Koush and Dooner. Per their test results, the
  NVFP4 KV degradation is "almost noise." The OP reports clean 90k needle tests even at FP4 KV.
  This is distinct from the fp8_ds_mla packed page format used by the b12x sparse-MLA kernel path.

- **[conjecture]** **Docker entrypoint `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` unset workaround**
  (S-forum-glm52-hybrid, alexander.korolev.germany): the production-hybrid Docker image bakes
  `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` into the environment, which causes issues for some users.
  The workaround is a wrapper entrypoint that unsets the var before handing off to the original
  `nvidia_entrypoint.sh`:
  ```dockerfile
  FROM aidendle94/sparkrun-vllm-ds4-gb10:production-hybrid-1.2
  RUN printf '#!/bin/bash\nunset VLLM_PREFIX_CACHE_RETENTION_INTERVAL\nexec /opt/nvidia/nvidia_entrypoint.sh "$@"\n' > /entrypoint-wrapper.sh && chmod +x /entrypoint-wrapper.sh
  ENTRYPOINT ["/entrypoint-wrapper.sh"]
  ```
  This is a GB10-specific operational gotcha for anyone using this community image.

- **[conjecture]** **V3 model: GPTQ applied on top of MXFP4 experts — "noticeably smarter"**
  (S-forum-glm52-hybrid, aidendle94): a v3 model (`aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`) applies
  GPTQ quantization on top of the MXFP4 expert layers, reportedly improving quality over v2. The OP
  rented H200s to perform the GPTQ calibration. Tool-eval-bench v3 score: 85/100 (vs v2: 86/100) —
  slightly lower overall, but the OP attributes this to the reasoning-parser issue (see below) and
  reports subjective quality improvement in code-review tasks. Single source → [conjecture].

## Tool-eval-bench quality results

- **[conjecture]** **GLM-5.2 Hybrid FP8+MXFP4 tool-eval-bench: 86/100 (v2), 85/100 (v3-GPTQ)**
  (S-forum-glm52-hybrid, alexander.korolev.germany): full hardmode 84-scenario benchmark on 4× Spark.
  v2: 65 passed, 14 partial, 5 failed (144/168 pts, quality 86/100, responsiveness 26/100 median
  turn 5.9s). v3-GPTQ: 64 passed, 14 partial, 6 failed (142/168 pts, quality 85/100, responsiveness
  28/100 median turn 5.7s). Weakest category: Structured Output 58% (v2) / Toolset Scale 62% (v3).
  v3 single-stream throughput: 2,814 pp tok/s, 20.6 tg tok/s, TTFT 10,828ms. Engine: vLLM
  0.11.2.dev279+eldritch.final, 262K ctx.

- **[conjecture]** **Structured Output 58% is a reasoning-parser bug, not a model/quant bug**
  (S-forum-glm52-hybrid, mike_ber): the 58% Structured Output score is caused by the `glm45`
  reasoning parser leaking a sentence fragment into the content channel before JSON output. The model
  produces perfectly valid schema-compliant JSON — it just prefixes 1-3 tokens of an unfinished
  conversational lead-in (e.g. `I{"ticker":"NVDA",...}` or `Here{"location":"Tokyo",...}`).
  **A/B test (thinking off vs on, category O only, 6 scenarios, seed 42, temp 0):**
  - Thinking OFF: 100/100, 12/12 pts, all pass, median turn 4.9s
  - Thinking ON: 75/100, 9/12 pts, TC-64 fails, TC-67 partial, median turn 7.6s
  - **Turning thinking off makes Structured Output 58% → 100% and is 36% faster (4.9s vs 7.6s).**
  - Full hardmode run with thinking off: 83→88 (+8 pts, 140→148/168). But Structured Reasoning
    dropped 100%→67% and Restraint & Refusal 100%→83% — without thinking, the model reaches for a
    tool instead of doing the work itself. **Takeaway: strict output format → thinking off;
    open-ended analysis → thinking on.** This is a config flag, not a model problem.
  - Confirmed by CosmicRaisins (the image author): "You're absolutely right. It's the reasoning
    parser, not the model."
  - **Two independent users agree** → approaches [reported], but both are in the same thread using
    the same image → stays [conjecture] per independence rules.

- **[conjecture]** **MTP4 outperforms MTP5 on tool-eval-bench** (S-forum-glm52-hybrid, ciprianveg):
  MTP5 scored 83, switching to MTP4 gave 85+. Look for FSM errors in logs for tool calls as a
  diagnostic. Consistent with the general finding that higher MTP depth can hurt quality
  (see `[[wiki/engines.md]]` → MTP quality section).

## Long-context word-salad bug — root cause: repetition penalty

- **[conjecture]** **Word-salad at >90k context after multi-turn was caused by hardcoded
  `repetition_penalty=1.2`** (S-forum-glm52-hybrid, mclenithan): after 2 weeks of debugging, the
  OP found that a hardcoded repetition penalty of 1.2 (left over from MiMo 2.5 work) was causing
  "word slop" — random mixed-script fragments (Latin/Cyrillic/CJK/Thai/code tokens) at >80-95k
  tokens and 15+ turns. The failure is unmistakable: mid-response the output turns into garbage
  with top-1 logprob around -10 to -12 (near-uniform distribution, temperature 0 doesn't prevent it).
  **Fix: remove the repetition penalty.** This is not a GLM-5.2 or GB10 bug — it's a config mistake,
  but it's a durable lesson: **GLM-5.2 is sensitive to repetition penalty in a way MiMo 2.5 is not.**
  Single source → [conjecture], but the root cause (config) is well-characterized.

## KV cache kernel constraint (b12x sparse-MLA)

- **[conjecture]** **b12x sparse-MLA kernel only reads packed fp8 KV pages — bf16 KV returns
  immediate EOS** (S-forum-glm52-hybrid, mclenithan): the b12x sparse-MLA kernel (`B12X_MLA_SPARSE`)
  only understands the packed `fp8_ds_mla` page format. With bf16 KV pages, the kernel misreads
  memory and every request returns an immediate EOS. The alternative backend that would accept
  non-packed KV (`FLASHMLA_SPARSE`) has **no shipped sm12x sparse kernels**, so the fp8-vs-bf16 KV
  comparison could not be completed. This is a GB10-specific kernel gap: the sparse-MLA attention
  path (used by GLM-5.2 and DeepSeek-V4-Flash) has limited KV format support on sm_121.
  - Related to the existing finding: FlashInfer `sparse_mla_sm120` livelock on GB10
    (S-forum-flashinfer-livelock, `[[wiki/attention-and-kv-cache.md]]`).
  - The OP's word-salad issue was initially investigated as a KV cache problem before being traced
    to the repetition penalty (above). The KV format constraint is real but was not the root cause
    in this case.

## NVIDIA official GLM5 NVFP4 on 4× Spark

- **[conjecture]** **NVIDIA official GLM5 NVFP4 works on 4× Spark — ~115 GB/node weights**
  (S-forum-glm52-hybrid, kevin.wu07): the official NVIDIA GLM5 NVFP4 checkpoint runs on 4 Sparks.
  The HF repo size includes ~20 GB of optional MTP weights (off by default), so the effective weight
  footprint is ~460 GB total, ~115 GB per node. With ~5 GB for a float8 KV cache of decent size plus
  other buffers/OS, it "pushes right up against the limits" but works. Memory utilization ~98%.
  - Skepticism (p33zy): 98% utilization "will be next to impossible to boot, let alone run without
    swaps." Single source + one skeptic → [conjecture].
  - Consistent with the existing S-forum-glm52-4x finding (~22 tok/s at TP=4 with AWQ-INT4 + pruning).

## NVFP4+AQLM hybrid checkpoint on 3× Spark (2026-08-04 ingest)

- **[conjecture]** **GLM-5.2 full 753B (no expert pruning) fits 3× Spark via NVFP4+AQLM hybrid
  checkpoint** (S-forum-glm52-3x-aqlm, karol.spark): a community checkpoint by jarrelscy
  (`jarrelscy` NVFP4+AQLM, 272 GB total, ~3.1 bits/param) combines NVFP4 weights with AQLM
  codebook compression on the expert layers, making the unpruned 753B model fit across 3× 121 GB
  nodes (~91 GB/node). Built on MiaAI-Lab's recipe + container images + vLLM fork (SM120 sparse-MLA
  path). TP=3, DCP=1, RoCE mesh. Full top-8 of 256 experts — no pruning.
  - **Benchmark (OP, code-like content, temp 0, thinking off, 400 tokens, n=8, 3 warmups):**
    decode 15.2–15.4 tok/s (LRU prompt), 13.5 tok/s (asyncio prompt); prefill 357 tok/s @31k,
    366 tok/s @98k; TTFT 270s @98k; context 235,392 tokens.
  - **After v3 kernel optimizations (see below):** decode 16.13 tok/s (+5.5%), prefill 367.9
    @31k / 379.5 @98k (+3.1–3.7%), TTFT 258s (−12s), context 215,040 (−8.6% due to vision graft).
  - **Methodology note:** OP measures `t/s ÷ acceptance` (normalized decode) rather than raw
    tok/s — `VLLM_MARLIN_USE_ATOMIC_ADD=1` means no bitwise reproducibility even at temp 0, so
    acceptance wanders 2.8–3.6 between server instances. Raw t/s can show ±8% where the normalized
    figure is unchanged. Also: measure decode *before* prefill (decode after 100k-token prefills
    reads ~1.7% low with 2× sigma). See `[[wiki/engines.md]]` → MTP quality section.
  - Single source (one thread, multiple users in same thread using same image) → [conjecture].

- **[conjecture]** **Virtual head padding for TP=3: 66 heads (22/rank), not 96 — FlashInfer
  dispatch table determines the cost** (S-forum-glm52-3x-aqlm, karol.spark + MiaAI-Lab): GLM-5.2
  has 64 attention heads — not divisible by 3. The prior TP=3 technique (S-forum-mimo-3x) padded
  to 96 (32/rank), costing 32 ghost heads per step. MiaAI-Lab found that padding to **66 (22/rank)**
  is sufficient because FlashInfer's `_DECODE_DSV3_2_DISPATCH` table only carries
  `{8, 16, 32, 64, 128} × {128, 512, 1024, 2048}` — local head count 22 is not in the table, so
  it falls through to the generic `sparse_mla_sm120_paged_attention` which tiles heads in groups
  of 16. Since `ceil(22/16) == ceil(32/16) == 2`, the attention cost is identical to 32 heads,
  while the q_b/kv_b/o_proj GEMMs shrink by 31%. **This is the key GB10 kernel-dispatch insight:**
  the FlashInfer specialized decode kernel only instantiates specific head counts; non-matching
  counts fall through to a generic paged-attention path tiled in groups of 16, so the effective
  attention cost is `ceil(local_heads/16)` tiles, not `local_heads`.
  - **TP padding table (from `pad_attention_heads()` in `glm_tp_pad.py`):**

    | TP | padded heads | local | MoE interm. | per-rank pad loss |
    |---|---|---|---|---|
    | 3 | 66 | 22 | 2112 | +3.1% |
    | 4 | 64 | 16 | 2048 | 0% |
    | 5 | 65 | 13 | 2240 | +9.4% |
    | 7 | 70 | 10 | 2240 | +9.4% |

  - **TP=4 is the only clean split** — 64/4=16 heads/rank, 2048/4=512 MoE intermediate, no waste,
    and local 16 lands exactly on the FlashInfer specialized decode kernel.
  - **TP=5 is feasible but meaningfully weaker than 3→4:** 65 heads (13/rank) passes the
    wrapper-pad admissibility rule (`local ≤ 32`), but reintroduces 9.4% MoE padding (2240 vs
    2048). Padding to 80 (16/rank) instead would hit the fast kernel at the cost of 25% ghost
    heads — whether 65 or 80 wins is unmeasured. The dense MLP `intermediate_size=12288` is not
    divisible by 5 (12288/5=2457.6) → hard exception at startup; needs padding to 12480 at
    `lcm(5,64)=320` (per-rank 2496). The pattern (`pad_moe_intermediate()` +
    `copy_tp_shard_with_pad`/`copy_gateup_tp_shard_with_pad`) already exists for MoE — ~1 day of
    work to apply to the dense layers. Zero-padding is mathematically exact (SiLU(0)×0=0).
  - **GB10 relevance:** the FlashInfer dispatch table head-count tiling rule is a durable kernel
    insight that applies to any sparse-MLA model on GB10 when considering non-power-of-2 TP.
    See `[[wiki/attention-and-kv-cache.md]]`.
  - Single source → [conjecture].

- **[conjecture]** **v3 kernel L1/L2 stream optimizations: +6.2% normalized decode** (S-forum-
  glm52-3x-aqlm, karol.spark): MiaAI-Lab's v3 (`k12l1`) image shipped three bit-exact, env-gated
  kernel changes:
  1. `GLM_MOE_AQLM_CB=l1` + `GLM_MOE_AQLM_STREAM=1` — routes AQLM codebook gathers through L1,
     marks code/activation streams evict-first. Microbench: w13 +2.7%, w2 +22.5% sector throughput.
  2. `GLM_NVFP4_STREAM=1` — stops the large NVFP4 weight stream thrashing the 1 MB codebook out of L2.
     Microbench: w13 +7%.
  3. **Draft cudagraph capture fix** — capture lists sized for the target (4, 8, 12 … for MTP k=3)
     never produced a pure `decode_query_len=1` shape, so MTP draft steps stayed eager. The fix
     unions the pure-decode shapes in automatically.
  - **Measured (same protocol):** normalized decode 4.520 ± 0.054 vs 4.257 ± 0.039 baseline = +6.2%
    (~5 combined standard deviations). Raw decode 14.78 vs 14.35 t/s (+3.0%). Acceptance unchanged
    (3.27 vs ~3.3) → improvement is in raw step time. Caveat: three changes landed at once; not
    attributed individually.
  - Single source → [conjecture]. The AQLM L1/L2 streaming finding is GB10-specific (L2 cache size
    and AQLM codebook working set interact on sm_121).

- **[conjecture]** **v4 vision graft: MoonViT tower + patch-merger projector on unchanged text
  backbone** (S-forum-glm52-3x-aqlm, karol.spark): MiaAI-Lab's v4 adds a MoonViT vision tower plus
  patch-merger projector grafted onto the otherwise-unchanged GLM-5.2 text backbone. Same image
  serves text-only and vision — switching is 3 files (`config.json`, `chat_template.jinja`,
  `model.safetensors.index.json`), not an image change. Vision index is a superset (5824 vs 5489
  keys) pointing at `vision_tower.safetensors` + `mm_projector.safetensors` (~934 MB together).
  - **MM_ENCODER_TP_MODE=data** — MoonViT has 16 attention heads, not divisible by 3, so it
    cannot be sharded at TP=3. This mode replicates the vision tower per rank (~0.93 GiB/node).
    The KV pin was reduced (12→11 GiB) to pay for it → context drops 235,392 → 215,040.
  - **Tag to avoid:** `:20260725-vision` has a bug where flat `--hf-overrides` were written only
    to the `Glm5vConfig` wrapper while nested `text_config` kept the checkpoint default, silently
    running ~2× routed-expert traffic and losing ~20% text decode.
  - This is a **second independent vision graft** for GLM-5.2 (cf. S-forum-glm52-vision,
    CosmicRaisins' `baseten/GLM-5.2-Vision-NVFP4` 49.5M-param projector). The approaches are
    similar (frozen backbone + lightweight projector) but from different authors. The vision
    graft finding (MoonViT 1152-dim → GLM 6144-dim mapping) is now [conjecture] approaching
    [reported] — two independent implementations agree on the frozen-backbone projector approach.
  - Single source for the TP=3 vision specifics → [conjecture].

- **[conjecture]** **16 GB swap mandatory for 235K-context 3× recipe** (S-forum-glm52-3x-aqlm,
  karol.spark): the recipe sits at ~119 GiB used per node and will touch swap. Without it, a
  deterministic kernel OOM occurs ~3 minutes after engine starts: weights load fine (~22s),
  compilation takes ~3.5 GiB and never returns it, then KV allocation fails. Fix: 16 GB swapfile
  per node (`fallocate -l 16G /swapfile && mkswap && swapon`). This is a GB10 UMA-specific
  finding — the unified memory pool means compilation workspace and KV compete for the same 121 GB.

- **[conjecture]** **NCCL ≥2.30.7 at a hardcoded path — `start.sh` expects `$HOME/nccl-2.30.7`**
  (S-forum-glm52-3x-aqlm, karol.spark): the worker branch of MiaAI-Lab's launcher does not read
  `NCCL_HOST_DIR`; it hardcodes `$HOME/nccl-2.30.7`. A version mismatch here produces confusing
  NCCL failures much later. Verify with `md5sum ~/nccl-2.30.7/libnccl.so.2*` across all nodes.
  Corroborates the existing [conjecture] NCCL 2.30.4+ mandatory finding
  (S-forum-ds4f-4x-vllm, S-forum-tokenspeed).

## SparkRing — switchless 4-node ring with custom RDMA collectives (2026-08-07 ingest)

- **[conjecture]** **SparkRing: 4× DGX Spark switchless ring using custom SIRCL RDMA collective
  layer** (S-forum-sparkring, FujitsuPolycom): an experimental inference stack that runs GLM-5.2
  across four directly-cabled DGX Sparks arranged as a physical ring — **no Ethernet switch in the
  inference path**. Four direct 200GbE ConnectX-7 links; management backplane is WiFi/USB ethernet.
  The core innovation is **SIRCL** (Switchless Inference RDMA Collective Layer), a custom low-level
  transport that uses direct-neighbor RDMA RC links, mapped pinned-memory arenas on GB10, explicit
  sequence/doorbell ordering, and inference-specific collective plans. It implements:
  - Custom TP4 all-reduce
  - Custom DCP query and fused output/LSE combine
  - Custom vocabulary and all-gather paths
  - **CUDA-graph-aware command rings**
  - Explicit software decomposition/relay for communication that would otherwise require a
    non-adjacent connection (the ring's non-adjacent node pairs are not L2 neighbors)
  - A patched ring-only NCCL fallback for operations not yet moved onto the custom path
  - This is a **fundamentally different approach** from the S-forum-6x-ring-rdma NCCL env-var workarounds:
    SIRCL replaces NCCL's collective algorithms entirely for inference-critical paths, rather than
    trying to make stock NCCL topology-aware. The NCCL approach hit the L2-adjacency wall for
    non-adjacent ring pairs; SIRCL sidesteps it with software relay. See
    `[[wiki/multinode-tp-and-networking.md]]`.

- **[conjecture]** **GLM-5.2 MXFP4-Experts-GPTQ on SparkRing: 19-20 tok/s C1, 500K KV capacity**
  (S-forum-sparkring, FujitsuPolycom): checkpoint `aidendle94/GLM-5.2-MXFP4-Experts-GPTQ`, TP4/DCP4/MTP4
  using `ag_rs`, dynamic MTP2/4, `nvfp4_ds_mla` KV with per-token scaling, RoPE fp8. Config:
  4,000,000,000 KV bytes per rank, 500,224-token measured logical KV pool, 458,752-token request
  ceiling, 4,096 maximum batched tokens, 8 sequence slots, FULL_AND_PIECEWISE CUDA graphs.

  | Context | Prefill tok/s | C1 | C2 agg | C4 agg | C8 agg |
  |---|---|---|---|---|---|
  | 8K | 844 | 20.3 | 27.1 | 40.5 | 49.2 |
  | 16K | 876 | 19.0 | 26.4 | 37.9 | 53.3 |
  | 32K | 830 | 20.3 | 27.6 | 38.6 | 51.9 |
  | 64K | 832 | 20.3 | 27.0 | 39.4 | 50.9 |
  | 128K | 796 | 19.7 | 26.3 | 37.2 | 47.7 |

  Short workload-dependent C8 windows reached 66.3 aggregate tok/s. Consistent with the 20-25 tok/s
  C1 range across other GLM-5.2 4× Spark recipes (see cross-thread summary below).

- **[conjecture]** **GLM-5.2 MXFP8-NVFP4-NF3 hybrid on SparkRing: 40-50 tok/s C4, 875K KV**
  (S-forum-sparkring, FujitsuPolycom): checkpoint `madeby561/GLM-5.2-MXFP8-NVFP4-NF3-Hybrid`, TP4
  DCP4 AMTP2-4 (adaptive MTP 2-4 tokens), KV format NVFP4 MLA + FP8 RoPE + per-token scaling,
  max 8 sequences, 4096 batched tokens. Weight mix: 64 NVFP4 + 192 NF3 experts. Total KV: 875,520.
  Max model length: 458,752. Reported 40-50 tok/s range for 4-concurrent shared-context coding
  sessions. This is a **third independent hybrid quant checkpoint** for GLM-5.2 (cf.
  S-forum-glm52-hybrid `aidendle94/GLM-5.2-Hybrid-FP8-MXFP4` and S-forum-glm52-3x-aqlm
  `jarrelscy` NVFP4+AQLM) — the frozen-backbone + mixed-precision expert approach is now
  well-established across multiple authors.

- **[conjecture]** **GLM-5.2 EXL3 3.25bpw working — 1M KV room, decode parity but prefill penalty**
  (S-forum-sparkring, FujitsuPolycom): checkpoint `willfalco/GLM-5.2-EXL3-TR3-3.25bpw` (from the
  RTX6000 Discord quant community). Decode "just as good or better" than the NF3 hybrid quant, but
  prefill stuck in the 500s tok/s (vs ~800-880 for other quants) — possibly config or calculation
  loss. Room for 1M KV capacity. The EXL3 format (EXL3 = ExLlamaV3) is a newer quant format not
  previously documented in sparkbase; its GB10 compatibility via SparkRing's custom transport
  is a new data point.

- **[conjecture]** **GLM-5.2 indexer weights missing for 57/78 layers — silent garbage above
  index_topk=2048** (S-forum-sparkring, Terry01): GLM-5.2 ships indexer weights for only **21 of 78
  layers** (layout is in `config.json` under `indexer_types`, which vLLM doesn't read). The guard
  in `deepseek_v2.py` ~line 1166 never fires, so **57 layers run top-k on uninitialized memory**.
  This is silent because the uninitialized-weights check is disabled when quantization is set.
  Below `index_topk=2048`, top-k picks everything so garbage is a no-op. Above it: **word salad
  with CJK**. `acceptance_gate.py` can't catch it. **Fix:** pattern from SETUP.md:881 — also
  gained ~30% throughput (acceptance 3.14→4.17, step 354→260ms, KV 465,663→575,232). **Check:**
  `docker logs <rank> | grep -c "skip sparse MLA indexer computation"` should be 57/rank, not 0.
  This is a GB10/vLLM-specific GLM-5.2 bug that affects any sparse-MLA deployment, not just SparkRing.

- **[conjecture]** **SparkRing launcher peer ordering bug — XOR schedule vs rank sort mismatch**
  (S-forum-sparkring, Terry01): `sparkring_launcher.py` ~line 164 sorts `transport_peers` by rank
  number, but `tp4_schedule.cpp` pairs by XOR (round0: rank^1, round1: rank^3). This is correct by
  luck for ranks 0/1 but inverted for 2/3, so both ends pick the same listen/connect role and every
  rendezvous times out (`timed out connecting/accepting control peer`, `ibv_modify_qp(RTR)`).
  **Fix:** `[by_rank[id^1], by_rank[id^3]]`. After the fix, every rank has peer0 on cage0 / peer1
  on cage1, matching the cable plan. This is a SparkRing-specific bug, but the underlying insight
  (RDMA peer pairing must match the physical cable plan) generalizes to any multi-node ring.

- **[conjecture]** **VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1 missing from example launch config**
  (S-forum-sparkring, Terry01): documented in SETUP.md 8.4 but not the example launch config.
  Without it, you're on the legacy static-outer-scale KV record. On MXFP4-Experts-GPTQ, **448 of
  2528 latent groups sit under the E4M3 subnormal floor**, layers 0-7 entirely. This env var is
  GLM-5.2-specific on GB10 and critical for NVFP4 MLA KV cache correctness.

- **[conjecture]** **VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096 causes engine death on GLM-5.2**
  (S-forum-sparkring, Terry01): the image bakes this env var, but pinned vLLM rejects it for
  GLM-5.2 (no sliding-window/Mamba KV group) — engine dies in `kv_cache_coordinator` right after
  KV allocation. No value fixes it since `envs.py` tests membership; needs `docker run --env KEY`
  with no `=`. Corroborates the existing S-forum-glm52-hybrid finding about the same env var
  causing issues — now a second independent report → approaches [reported].

- **[conjecture]** **CUDA graphs produce single-token lock on GLM-5.2 SparkRing** (S-forum-sparkring,
  Terry01): with full 8.4 env, `MAX_QUERY_ROWS=40`, CPU pins, graph ports — the engine boots and
  serves, then answers a 13k-token prompt with a single token "lock" (output stalls at 1 token).
  DCP1 graph arm produced identical lock. Suspected workspace issue from TESTING_HISTORY.
  Eager mode works: 18.3 tok/s median on 500-token code completions, TP4/DCP4 on
  MXFP4-Experts-GPTQ, clean at 13K context — ~92% of the published eager number.

- **[conjecture]** **SparkCache: DCP4-sharded persistent KV context cache from local NVMe**
  (S-forum-sparkring, FujitsuPolycom): an experimental DCP4-sharded persistent context cache that
  can snapshot and restore very large KV contexts from each Spark's local NVMe shard with **zero
  cross-node KV traffic** — only tiny quorum metadata is exchanged. Current limitation: the
  snapshot/store path can still interfere with unique-context decode slightly. This is a novel
  approach to persistent KV caching on GB10 — distinct from vLLM prefix caching or LMCache.

- **[conjecture]** **Terry01 reproduced SparkRing: 18.3 tok/s eager, ~92% of published number**
  (S-forum-sparkring, Terry01): after fixing 6 setup issues (indexer weights, peer ordering,
  per-token scale env var, prefix cache retention interval, download script, tool/reasoning parser),
  achieved 18.3 tok/s median on 500-token code completions in eager mode, TP4/DCP4 on
  MXFP4-Experts-GPTQ, clean at 13K context. This is the **first independent reproduction** of a
  SparkRing deployment — ~92% of the published ~20 tok/s eager number. However, both reporters are
  in the same thread using the same image → stays [conjecture] per independence rules.

## Official NVIDIA GLM-5.2 NVFP4 on 8× Spark (2026-08-12 ingest)

- **[conjecture]** **Official `nvidia/GLM-5.2-NVFP4` on 8× Spark TP=8 — 25 tok/s decode, 256K
  context, tool-eval-bench v2.5.1 score 93** (S-forum-glm52-8x-nvfp4, hypermac.6502): served via
  eugr's `spark-vllm-docker` with the included `8x-spark-cluster/glm-5.2-nvfp4.yaml` recipe.
  Weights distributed to each worker from HuggingFace; "everything worked right away with the
  included recipe." This is the **official NVIDIA NVFP4 checkpoint** (not a community re-quant)
  and the first reported 8× Spark run of the official checkpoint with a tool-eval-bench score.
  - **tool-eval-bench v2.5.1 score: 93/100** — the highest reported tool-eval-bench score for
    GLM-5.2 on GB10. For comparison: the hybrid FP8+MXFP4 community quant scored 86/100 (v2) /
    85/100 (v3-GPTQ) on 4× Spark (S-forum-glm52-hybrid). The 8-point gap may reflect the official
    checkpoint's higher fidelity, the 8× cluster's headroom, or tool-eval-bench version differences
    (v2.5.1 vs the version used in the hybrid thread). Single source → [conjecture].
  - **25 tok/s decode at TP=8** is consistent with the existing 8× TP=8 Int4-Int8 mix result
    (33-54 tok/s, S-forum-glm52-8x) — the official NVFP4 checkpoint at 25 tok/s is at the lower
    end of that range, consistent with the [reported] finding that NVFP4 decode is bandwidth-bound
    and quant-format matters less than the sparse-MLA attention + bandwidth ceiling. The 8×
    Int4-Int8 mix may benefit from the b12x W4A8 backend's kernel efficiency (same pattern as
    AWQ-beats-NVFP4 for MiniMax M2.7).
  - **256K context** — lower than the 800K achieved by the hybrid quant on 4× Spark
    (S-forum-glm52-hybrid), likely because the official NVFP4 checkpoint's larger weight footprint
    (~460 GB total, ~115 GB/node per S-forum-glm52-hybrid) leaves less room for KV at 8× than the
    hybrid's mixed-precision approach. However, 8× gives 2× the compute/memory of 4×, so the
    KV pool per node is less constrained — the 256K may be a recipe default, not a hard limit.
  - This corroborates the existing [conjecture] finding that the official NVIDIA GLM-5.2 NVFP4
    checkpoint runs on 4× Spark at ~115 GB/node (S-forum-glm52-hybrid, kevin.wu07). At 8× the
    per-node weight footprint drops to ~58 GB, leaving ample room for KV.

## Performance across configurations (cross-thread summary)

All numbers are **[conjecture]** or **[reported]** as noted. See `[[wiki/benchmarks.md]]` for full rows.

|| Config | Quant | Nodes | Decode tok/s | Ctx | Source |
|---|---|---|---|---|---|
| TP=3, NVFP4+AQLM, DCP=1 | NVFP4+AQLM (~3.1 bpw) | 3 | 15.2–16.1 | 215K–235K | S-forum-glm52-3x-aqlm |
| TP=4, AWQ-INT4 + 15% expert prune + MTP | AWQ-INT4 | 4 | ~22 | 256K | S-forum-glm52-4x |
| TP=4, NVFP4, MTP4 fix | NVFP4 | 4 | 24 | 128K | S-forum-glm52-mtp-fix |
| TP=4, Hybrid FP8+MXFP4 (v2) | Hybrid | 4 | 20-25 | 800K | S-forum-glm52-hybrid |
| TP=4, Hybrid FP8+MXFP4 (v3-GPTQ) | Hybrid+GPTQ | 4 | 20.6 | 262K | S-forum-glm52-hybrid |
| TP=6, b12x | (Int4-family) | 6 | ~30 | — | S-forum-6x-cluster |
| TP=8, Int4-Int8 mix, b12x W4A8 | Int4-Int8 | 8 | 33-54 | 200K | S-forum-glm52-8x |
|| TP=4, MXFP4-Experts-GPTQ, SparkRing SIRCL | MXFP4-Experts-GPTQ | 4 | 19-20 (C1) / 50-63 (C8) | 500K | S-forum-sparkring |
|| TP=4, MXFP8-NVFP4-NF3 hybrid, SparkRing | MXFP8+NVFP4+NF3 | 4 | 40-50 (C4) | 875K | S-forum-sparkring |
|| TP=4, IQ4_XS GGUF, llama.cpp RPC | IQ4_XS | 4 | 6.28 | 1M | S-forum-glm52-iq4xs-4x |
|| 1× Spark, Colibri expert streaming | int4 MoE + int8 MTP | 1 | 2.4-3.3 | short | S-forum-colibri-glm52 |
|| TP=8, official NVFP4 | NVFP4 (official nvidia) | 8 | 25 | 256K | S-forum-glm52-8x-nvfp4 |

**[reported]** GLM-5.2 decode on 4× Spark is consistently in the 20-25 tok/s range across multiple
independent threads and quant formats (AWQ-INT4, NVFP4, Hybrid FP8+MXFP4) — the bottleneck is the
sparse-MLA attention + bandwidth-bound decode, not the quant choice. The 8× TP=8 run (33-54 tok/s)
is the outlier, attributed to the DCP collectives + b12x W4A8 backend + 2× the nodes. The 3× TP=3
run (15.2–16.1 tok/s, NVFP4+AQLM) continues the sublinear scaling trend — fewer nodes, lower
throughput, same bandwidth-bound regime.

## See also

`[[wiki/attention-and-kv-cache.md]]` (sparse_mla_sm120 livelock, KV format constraints) ·
`[[wiki/quantization-on-gb10.md]]` (NVFP4/MXFP4/AWQ, b12x W4A8, Int4-Int8 mix, quantized NextN mapping) ·
`[[wiki/engines.md]]` (MTP quality, adaptive MTP, Colibri expert streaming) ·
`[[wiki/multinode-tp-and-networking.md]]` (TP=4/6/8, DCP, NCCL buffer sizing) ·
`[[wiki/cudagraphs-and-compile.md]]` (MoE cudagraph wall, MTP-needs-cudagraphs) ·
`[[wiki/platform-gb10.md]]` (unified memory, no GPUDirect, bandwidth-bound decode) ·
`[[wiki/benchmarks.md]]` (full benchmark rows) ·
`[[wiki/roadmap.md]]` (open problems: paged-KV for Sm120, adaptive MTP overhead)