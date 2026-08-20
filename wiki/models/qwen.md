# Qwen (3.5 / 3.6 / Coder-Next)

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-swapper, S-mimo-doc, S-forum-unsloth-qwen36, S-forum-qwen397-arch, S-forum-bonsai27b, S-forum-qwen36-fp8-2x, S-forum-vllm-stock-hang, S-forum-qwen122-king, S-forum-qwen122-v26-dflash, S-forum-unsloth-b12x, S-forum-vllm-2607-xgrammar, S-forum-qwen36-draft-train, S-forum-moe-lora-vllm, S-forum-qlora-coding, S-forum-macaron-v1-tall, S-forum-qwen36-tp2-stall, S-forum-kat-coder-autoround, S-forum-qwen38-27b-mixedint4, S-forum-qwen38-nvfp4-vs-fp8, S-forum-ds4f-qwen38-orchestration
> **updated:** 2026-08-20

The best-supported family on GB10 — both Atlas (AOT kernels for the MoE variants) and vLLM serve it.
The recurring lesson: **MoE-A3B NVFP4 + MTP is the fastest regime on Spark; the dense variant of the
same size is ~4–5× slower.**

## Qwen3.6-35B-A3B (MoE) — the fast regime

- `RedHatAI/Qwen3.6-35B-A3B-NVFP4`: 10 attn + 30 SSM + 256 experts (~3B active), `qwen3_5_moe`
  loader, **fp8 KV**. Ships **`model_mtp.safetensors` + `model_visual.safetensors`** → **MTP + vision
  both work via the MoE loader**.
- **Atlas production config:** `--gpu-memory-utilization 0.40 --max-seq-len 262144 --kv-cache-dtype
  fp8 --max-batch-size 2 --scheduling-policy slai --oom-guard-mb 8000 --mtp-quantization bf16
  --enable-prefix-caching --speculative`.
  - **[proven]** **~142 tok/s benchmark / ~76–96 tok/s in service** (90 @ batch 4). The regime Atlas
    is built for. (`[[wiki/engines.md]]`)
- **[proven]** vLLM serves the family too (a TP=2 Ray serving stack): bf16 safetensors,
  `--tool-call-parser qwen3_xml --reasoning-parser qwen3 --chat-template fixed_chat_template.jinja`
  (+ a chat-template fixup mod), `--kv-cache-dtype fp8 --load-format fastsafetensors
  --attention-backend flashinfer`.

## Qwen3.6-35B-A3B HauhauCS Uncensored NVFP4 — MTP sweep [proven]

> **evidence:** proven (benchmarked on real GB10, 2026-07-14)
> **sources:** S-sess-jul14, local llama-benchy runs
> **container:** vllm-node-v0240 (v0.24.1.dev0), flashinfer-cutlass NVFP4, fp8 KV, flashinfer attn

Model: `lyf/Qwen3.6-35B-A3B-Uncensored-HauhauCS-Aggressive-NVFP4` — ~23.4 GB NVFP4 W4A4
compressed-tensors, 35B MoE / 3B active, 256 experts (8 routed + 1 shared), hybrid GDN + full
attention.

**MTP self-speculative decoding is a net negative at every depth on this quant.** Draft
acceptance is effectively zero — each speculative token adds one full forward pass with no
accepted tokens in return. The degradation is linear:

| Config (single-node TP=1, c1 decode) | tok/s | Prefill t/s (pp2048) |
|---|---|---|
| MTP off | 41.8 | 5,400–6,250 |
| MTP-1 | 31.7 | 4,900–5,960 |
| MTP-2 | (pending) | (pending) |
| MTP-3 | 22.3 | 4,550–5,090 |

For comparison, **TP=2 no-MTP on both GB10s = 67 tok/s decode, 8,400 t/s prefill** — 1.6x faster
than TP=1 no-MTP. Even with NCCL host-bounce overhead, splitting MoE expert compute across both
GB10s wins. This 3B active param model is compute-bound, not bandwidth-bound.

### Key findings

- **[proven]** MTP self-spec on NVFP4 quants with poor draft acceptance is strictly worse than
  no-MTP. Each spec token costs ~10 tok/s. This contradicts the spark-arena 100+ tok/s claims,
  which all use single-node TP=1 + MTP — but likely with quants that have real draft acceptance
  (e.g., Unsloth NVFP4-Fast with flashinfer_b12x, or external drafters like DFlash).
- **[proven]** TP=2 beats TP=1 by 1.6x for this model even without MTP. The compute parallelism
  on MoE expert evaluation outweighs NCCL overhead for 3B active params.
- **[proven]** Cudagraphs capture cleanly on TP=1 with MTP (PIECEWISE mode, 11 graphs, 3s).
  The cross-node cudagraph wall (see `[[wiki/cudagraphs-and-compile.md]]`) does NOT apply to
  Qwen3.6 — it uses standard piecewise compile, closer to MiMo than M3.
- **[proven]** vLLM #41190 (TP=2 + spec-decode crash) does NOT reproduce on v0.24.0. MTP loads
  and runs on TP=2 — it just doesn't help (21 tok/s, same regression).
- **[proven]** `moe_backend='triton'` is rejected for NVFP4 MoE. Must use `flashinfer_cutlass`
  (or `flashinfer_b12x`, `marlin`, etc.).
- **[proven]** Depth 4096 shows no degradation — decode is stable at all MTP depths tested.

### Operational notes

- The HF cache `trees/` directory can get root-owned by sparkrun's distribution step. If rsync
  to the worker fails with "permission denied" on `trees/*.json`, run
  `sudo chown -R daniel:daniel ~/.cache/huggingface/hub/models--*/trees/` on the head node.
- Cold start (TP=1): ~2 min weight load + ~2 min compile/autotune/cudagraph = ~5 min total.
- Cold start (TP=2): ~9 min weight load (per rank) + ~4 min compile = ~13 min total.

## Unsloth Qwen3.6 NVFP4 quants on GB10 (2026-07-15 forum ingest)

> **evidence:** reported (3 independent forum benchmarks agree Unsloth is slower than nvidia NVFP4 on GB10)
> **sources:** S-forum-unsloth-qwen36

Unsloth released `unsloth/Qwen3.6-35B-A3B-NVFP4` and `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`,
claiming 2.5× speedup on B200. On GB10 the reality is different:

- **[reported]** **Unsloth NVFP4 decodes ~15% slower than nvidia NVFP4 on GB10** — confirmed by 3
  independent forum benchmarks:
  - hedelyuk.alexandr: nvidia 103.4 avg tok/s vs Unsloth 87.6 avg tok/s (−15.2%), across 4 workload types
  - J-R: nvidia ~90 tok/s vs Unsloth ~75 tok/s at c=1 (MTP=2); gap narrows at higher concurrency
    (c=16: 420 vs 410 tok/s)
  - TheAwakenOne: Unsloth NVFP4-Fast 75→97 tok/s after tuning, still behind nvidia
  All measured single-node on DGX Spark, vLLM 0.24, MTP enabled.
- **[conjecture]** **`flashinfer_b12x` backend unavailable on stock vLLM builds** (jbourny): Unsloth's
  optimum-performance recipe requires `--moe-backend flashinfer_b12x --linear-backend flashinfer_b12x`,
  but stock vLLM reports `ValueError: no 'flashinfer_b12x' kernel exists for this layer type` — despite
  `has_flashinfer_b12x_gemm()` returning True on sm_121. Working fallback: `--moe-backend marlin`
  (the same path as nvidia NVFP4). JW2026 reports b12x flags give <3% difference (within margin of error).
- **[conjecture]** **Working spark-vllm-docker recipe** (emX0r): Marlin MoE backend + MTP nst=3,
  fp8 KV, flashinfer attention, `VLLM_MARLIN_USE_ATOMIC_ADD=1`. `--load-format fastsafetensors`.
  `moe_backend:'triton'` in MTP speculative-config is required (not `marlin`). Some users must drop
  `moe_backend` from speculative-config or MTP config is rejected. `--reasoning-parser qwen3
  --tool-call-parser qwen3_coder --enable-auto-tool-choice`.
- **[conjecture]** **Unsloth "speedup" is a W4A16 bypass, not NVFP4** (robert287): Unsloth's performance
  boost on Qwen models may come from bypassing the NVFP4 bottleneck on weights by using W4A16 internally
  — which would explain why it's slower on GB10 (AWQ/Marlin W4A16 vs NVFP4 Marlin MoE is a wash or slight
  regression, consistent with the AWQ-vs-NVFP4 findings on `[[wiki/quantization-on-gb10.md]]`).
- **[reported]** **Quality parity with nvidia NVFP4** (azampatti): tool-eval-bench scores approximately
  equal, but with more variance between runs. "Equally good" for tool-calling workloads.
- **[conjecture]** **Qwen3.6-27B NVFP4 fits in 24 GB VRAM** (Unsloth claim, untested on GB10): the dense
  27B at NVFP4 would be ~7 GB weights — fits easily on a single Spark. No GB10-specific benchmark
  reported; the dense regime is ~30 tok/s regardless (bandwidth-bound, see below).

### Paths to 100+ tok/s (not yet attempted)

1. ~~**Unsloth NVFP4-Fast quant** — tested, see section above: ~15% slower than nvidia NVFP4 on GB10.~~
2. **External drafter (DFlash)**: AEON-7 container with `z-lab/Qwen3.6-35B-A3B-DFlash` drafter,
   num_speculative_tokens=10, ~46-48% acceptance reported.
3. **AEON-7 heretic build**: `AEON-7/Qwen3.6-35B-A3B-heretic-NVFP4` + custom container
   (ghcr.io/aeon-7/vllm-spark-omni-q36:v1.2), 97-120 tok/s reported.

## Qwen3.6-27B (dense) — the slow regime

- `qwen35_dense` loader, 64 layers (16 attn + 48 SSM), **forced bf16 KV**, head_dim 256.
  - **[proven]** Activates all params → **~30 tok/s** (FP8+MTP). Needs **`-o max_model_len=8192`** to
    fit (lowering it shrinks the KV pool with **zero** decode-speed cost). Dense ~106 GB → concurrency
    needs a 2nd node.

## Qwen3-Coder-Next-NVFP4

- `saricles/Qwen3-Coder-Next-NVFP4-GB10` — the head single-node unit. Image
  `avarok/dgx-vllm-nvfp4-kernel:v23`, env `VLLM_NVFP4_GEMM_BACKEND=marlin`,
  `VLLM_TEST_FORCE_FP8_MARLIN=1`; `--kv-cache-dtype fp8 --attention-backend flashinfer
  --tool-call-parser qwen3_coder`. 79.7B MoE (512 exp/10 active), 262k ctx.

## Loader landmines (Atlas)

- **[proven]** **Inline-MTP NVFP4 checkpoints crash the dense loader** (`cuMemcpyDtoDAsync_v2 status
  1`, double FP32 promotion at layer-0 SSM `A_log`/`dt_bias`) — use checkpoints with a **separate**
  `model_mtp.safetensors` (stock RedHatAI), not "MTP-preserved" inline ones (llmfan46).
- **[proven]** **Dense-VL NVFP4 double-promotion** (croll83 Qwopus-27B): the dense-VL loader
  mishandles the `language_model.* + visual.*` dual prefix; **the MoE loader handles VL fine, only
  dense is buggy.**
- **[proven]** ⟹ **No Qwopus3.6 currently loads on Atlas.** Fallbacks: AEON-7 NVFP4 VL-MoE, llama.cpp
  GGUF, or self-quantize the MoE to NVFP4 with modelopt.

## Forum ingest: Bonsai 27B — binary/ternary Qwen3.6-27B (2026-07-17)

- **[conjecture]** **Bonsai 27B (Prism-ML): 1-bit and ternary builds of Qwen3.6-27B** (S-forum-bonsai27b,
  nerhun): Prism-ML released "Bonsai 27B" — binary (1-bit) and ternary quantization of Qwen3.6-27B,
  the largest binary/ternary attempt so far. Claimed ~94% of full quality at a much smaller memory
  footprint. Collection: `huggingface.co/collections/prism-ml/bonsai-27b`.
  - **[conjecture]** **Hypothesis: faster decode on Spark due to smaller footprint** (m0l0):
    Qwen3.6-27B dense is naturally bandwidth-limited on Spark for token generation (~10 tok/s
    ceiling, see `[[wiki/quantization-on-gb10.md]]` dense MTP math). Bonsai's much smaller footprint
    should leave prompt processing unaffected but significantly speed up token generation —
    especially with MTP. This is consistent with the proven "fewer bytes = faster decode" rule
    for bandwidth-bound dense models. However, **no GB10 benchmarks have been posted yet** — the
    1-bit/ternary kernel path on sm_121 is unverified. A user (m0l0) plans to benchmark Bonsai
    vs their daily driver (Qwen3.6-27B PrismaScout) and vs Qwen3.6-9B (~same footprint).
  - **[conjecture]** **Not much use on a Spark for the binary build** (stu.miller): "we can obviously
    do much better" — the binary/ternary footprint advantage matters most on laptops/phones, not
    on 121 GB unified memory. There could be merit for co-hosting a small model alongside other
    workflows where less memory is available without giving up too much quality.
  - **GB10-specific unknowns:** whether 1-bit/ternary kernels have a working sm_121 path (Marlin
    does not natively support 1-bit/ternary — would need a custom Triton or CUTLASS kernel), and
    whether the ~94% quality claim holds on agentic/tool-calling workloads. Queue for hardware
    agent verification.

## Qwen3.6-35B-A3B FP8 on 2× Spark (2026-07-23 forum ingest)

> **evidence:** conjecture (single forum source)
> **sources:** S-forum-qwen36-fp8-2x

- **[conjecture]** **75-80 tok/s output via spark-vllm-docker run-recipe.sh** (S-forum-qwen36-fp8-2x,
  gary100): `Qwen/Qwen3.6-35B-A3B-FP8` through vLLM with TP=2 across two GX10 nodes, no-ray,
  FlashInfer attention, FP8 KV cache, 262K max model len, prefix caching enabled, qwen3_xml tool
  parser. Results via direct vLLM API:
  - Small context (5,098 prompt tokens): cold TTFT 0.68s, cold input speed ~7,487 tok/s, output
    ~79.9 tok/s
  - Medium context (40,474 tokens): cold TTFT 5.23s, input ~7,740 tok/s, output ~78.5 tok/s
  - Large context (80,890 tokens): cold TTFT 8.49s, input ~9,522 tok/s, output ~75.3 tok/s
  - Prefix cache: 2nd-run TTFT drops to 0.47s (medium) / 0.99s (large)
  - Hardware: 2× ASUS GX10 4TB, ~258 GB combined unified memory, 200GbE RoCE direct link, MTU 9000
  - Recipe: `./run-recipe.sh qwen3.6-35b-a3b-fp8 --no-ray`
  This is an FP8 (not NVFP4) checkpoint — native FP8 compute on GB10. The 75-80 tok/s decode is
  notably lower than the 142 tok/s proven on Atlas with NVFP4+MTP (different engine, different
  quant, different serving stack). Single source → [conjecture].

- **[conjecture]** **Stock `vllm/vllm-openai:latest` hangs silently loading Qwen3.6-35B-A3B-NVFP4
  on GB10** (S-forum-vllm-stock-hang, dotrantrung2003): on an ASUS Ascent GX10, the upstream
  `vllm/vllm-openai:latest` image reaches backend selection (FlashInfer attention, MARLIN NvFp4
  MoE) but never reaches "Application startup complete" — the container stays `Up` but hangs
  during initialization; all API requests return `Connection reset by peer`. Root cause is the
  image, not the flags: stock upstream vLLM lacks SM121/Blackwell support. Fix: use a GB10-tuned
  build (spark-vllm-docker `--tf5` or a CUDA 13 / SM121 wheel). See
  `[[wiki/containers-and-tooling.md]]` for full details. Single source → [conjecture].

## Forum ingest: Qwen3.5-122B "king model" daily-driver consensus + sparkrun-recipes (2026-07-27)

> **evidence:** reported (4 independent users agree 122B is the best single-Spark daily driver)
> **sources:** S-forum-qwen122-king

A forum thread (378066) on the best daily-driver model for a single DGX Spark drew 4
independent users confirming **Qwen3.5-122B-A10B-int4** as the "king model" — the largest
actually-capable model that fits comfortably in 121 GB unified memory at high context.

- **[reported]** **Qwen3.5-122B-A10B-int4 is the community consensus single-Spark daily driver**
  (S-forum-qwen122-king, Styles01 + Josephbreda + 0rand + Rerollingingenshitimpactsucks): 4
  independent users with different workloads (agentic/tool-calling, vision/image
  interpretation, 100k+ context, coding) all keep coming back to 122B-int4 as the default.
  Key reasons cited: largest model that fits in memory at high context, capable vision tower
  for image interpretation/translation, good tool-calling quality, and speed that holds
  linearly through 100k+ context with AutoRound int4. 4 independent sources agreeing →
  [reported].

- **[conjecture]** **AutoRound int4 Qwen3.5-122B-A10B on 2× Spark: ~65 tok/s, holds linearly
  over 100k context** (S-forum-qwen122-king, Josephbreda): "I get 65 t/s with Intel AutoRound.
  Most importantly it holds fairly linearly through context over 100k." Also notes needing the
  vision tower. Single-source number → [conjecture].

- **[conjecture]** **FP8 122B on 1× Spark: ~35 tok/s at best** (S-forum-qwen122-king, 0rand):
  "Fp8 122b is about 35 t/s at best times, mtp is multi pass very old style, run on marlin or
  deepgemm no chance of flashinfer." Also reports "no tool call quality improvement over int4,
  most likely due to use of suboptimal kernels." On a vanilla stack, 0rand reports "haven't
  seen more than 40-45 from the memory" for int4 on 2 nodes. Single-source → [conjecture].

- **[conjecture]** **sparkrun-recipes: patched vLLM v26 build, 5 lanes at 256K ctx, 40+ tok/s
  decode on single Spark** (S-forum-qwen122-king, Styles01): `styles01/sparkrun-recipes` GitHub
  repo — "massively increases KV cache and overhead optimization on the qwen 122b hybrid model
  by blesyg." Requires "a lot of patching of the unpublished latest VLLM v26 build." Claims
  "5 lanes at 256kb at 40+ t/s decode is pretty phenomenal" and "the absolute maximum
  squeezing we can do of qwen 3.5 122b." Single-source recipe claim → [conjecture].

- **[conjecture]** **Intel AutoRound int4 quant has a real tendency to loop** (S-forum-qwen122-king,
  Rerollingingenshitimpactsucks): "IME this intel 4 bit autoround quant still has a very real
  tendency to loop, and I'm not sure whether it's actually the best in terms of fidelity."
  Suggests NVFP4 variants of 122B might perform better in output quality. Also notes "The
  official Nvidia repo version of this loops because they apparently screwed up the
  quantization themselves." Single-source quality observation → [conjecture].

- **[conjecture]** **DSV4-Flash on single Spark: 45-50 tok/s, up to 240 tok/s at 16 concurrent
  reqs, coherent to 1M tokens** (S-forum-qwen122-king, 0rand): reported as the strongest
  alternative to 122B for 2-Spark clusters. Single-source number → [conjecture].

- **[conjecture]** **Laguna is very slow / not good on this user's setup**
  (S-forum-qwen122-king, Rerollingingenshitimpactsucks): "my results haven't been good."
  Corroborates the first-party retirement finding on `[[wiki/models/laguna-s-2.1.md]]`.

- **[conjecture]** **Mistral 119B used as a daily-driver replacement for Qwen 3.5**
  (S-forum-qwen122-king, gaburko via topic 378131): a brewing-agent use case switched from
  Qwen3.5 to Mistral 119B as the locally hosted LLM on DGX Spark. No tok/s or recipe details —
  confirms the model is in daily use on Spark but adds no durable technical finding beyond
  the existing Mistral-Small-4 page (`[[wiki/models/mistral-small-4.md]]`).

## Forum ingest: Qwen 122B vLLM v26 + fp8 KV + DFlash + int8 lm-head (2026-07-28)

> **evidence:** conjecture (single forum source)
> **sources:** S-forum-qwen122-v26-dflash

A forum thread (378167, styles01) documents the **first working fp8 KV + DFlash implementation
on GB10** for a hybrid quantization model (Qwen3.5-122B-A10B). Three custom patches on top of
an unreleased vLLM v26 build unlock fp8 KV for `inc_hybrid` quant models, free ~1.4 GB from the
lm-head via int8 GEMV, and fix prefix-cache alignment for DFlash.

- **[conjecture]** **vLLM v26 (main, commit 318b527) adds native fp8 KV support for hybrid
  quantization models** (S-forum-qwen122-v26-dflash, styles01): previous vLLM versions (0.23–0.25)
  were architecturally blocked — the `inc_hybrid` quant method didn't support fp8 KV, and the
  lm-head projection (248K vocab × 3072 hidden = ~1.4 GB bf16) consumed memory that could go to KV
  cache. Built from main with `--build-arg torch_cuda_arch_list='12.1'`, build time 3–5 hours on
  GB10 (FA2/FA3 CUDA kernels are the slow part, ~85s each, 400 total). Single source → [conjecture].

- **[conjecture]** **Three custom patches:**
  1. **inc_hybrid** — enables fp8 KV cache for the hybrid quant model. Without it, the model
     fails to load with `weight_scale_inv` errors.
  2. **int8_lmhead_v3** — converts the 122B lm-head from bf16 (~1.4 GB) to int8 w8a16 GEMV
     (~175 MB), freeing ~1.4 GB for KV cache. Hooks into `LogitsProcessor._apply_head` (v26's
     new logits path). Bonus: recovers decode speed lost to fp8 KV overhead (45.98 vs 43.6 tok/s).
  3. **prefix_align** — fixes prefix caching alignment issues that caused cache corruption with
     DFlash speculative decoding.

- **[conjecture]** **Benchmark results (single Spark, pp512/tg128, 3 runs):**

  | Metric | bf16 KV (vLLM 0.23) | fp8 KV (v26 patched) | Improvement |
  |---|---|---|---|
  | KV cache | 549K tokens | 1,372,342 tokens | 2.6× |
  | Concurrency @ 256K | 2.09× | 5.24× | 2.5× |
  | Decode speed | 50.2 tok/s | 45.98 tok/s | recovered via int8 lm-head |
  | Prefill speed | 726 tok/s | 957 tok/s | +32% |

  Launch config: `--kv-cache-dtype fp8 --gpu-memory-utilization 0.85 --max-num-seqs 3
  --max-num-batched-tokens 8192 --enable-prefix-caching --enable-chunked-prefill
  --attention-backend FLASHINFER --speculative-config '{"method":"dflash","model":
  "z-lab/Qwen3.5-122B-A10B-DFlash","num_speculative_tokens":7}'`. Model:
  `bleysg/Qwen3.5-122B-A10B-int4-fp8-hybrid`. Single source → [conjecture].

- **[conjecture]** **vLLM v26 natively supports FlashInfer on SM121** — lower overhead than
  flash_attn for long contexts. (S-forum-qwen122-v26-dflash)

This corroborates the existing sparkrun-recipes finding (S-forum-qwen122-king, Styles01) — same
author, same "vLLM v26 patched" approach, now with full technical detail on the three patches and
the fp8 KV + DFlash + int8 lm-head combination. The 45.98 tok/s decode at 256K context on a single
Spark is a notable result for the 122B model, and the 1.37M-token KV pool (2.6× the bf16 baseline)
is a significant capacity gain. The int8 lm-head technique (~1.4 GB → ~175 MB) is a GB10-specific
memory-reclamation approach that could generalize to other large-vocab models on Spark.

## Forum ingest: Unsloth vs nvidia Qwen3.6-35B-A3B-NVFP4 with flashinfer_b12x (2026-07-29)

> **evidence:** conjecture (single forum source, same author as prior benchmarks)
> **sources:** S-forum-unsloth-b12x

A benchmark thread (376703, shahizat) directly compares `unsloth/Qwen3.6-35B-A3B-NVFP4-Fast`
with `--moe-backend flashinfer_b12x` vs `nvidia/Qwen3.6-35B-A3B-NVFP4` with default Marlin
backend on vLLM 0.25.0, 100 concurrent requests, random 1000-in/1000-out.

- **[conjecture]** **Unsloth+b12x ~8% faster than nvidia+Marlin on Spark at 100 concurrency**
  (S-forum-unsloth-b12x, shahizat): Unsloth aggregate output 435.84 tok/s vs nvidia 404.24 tok/s
  on DGX Spark — a ~8% Unsloth lead. This **reverses direction** from the prior [reported]
  finding (S-forum-unsloth-qwen36) where Unsloth was ~15% *slower* than nvidia on GB10. The
  key difference: the prior benchmarks used Marlin backend for both, while here Unsloth uses
  `flashinfer_b12x` and nvidia uses default Marlin. The b12x backend appears to be the lever,
  not the quant itself. Single source → [conjecture]. TPOT: Unsloth 212.83 ms vs nvidia ~228 ms
  (estimated from 404 tok/s aggregate / 100 concurrent).

- **[conjecture]** **Working flashinfer_b12x recipe on DGX Spark** (S-forum-unsloth-b12x,
  TheAwakenOne citing Unsloth blog): the recipe for enabling b12x on Spark:
  ```
  export CUTE_DSL_ARCH=sm_121a
  vllm serve unsloth/Qwen3.6-35B-A3B-NVFP4-Fast --moe-backend flashinfer_b12x
  ```
  Prerequisites: `vllm>=0.25.0`, `flashinfer-python>=0.6.13`, `nvidia-cutlass-dsl>=4.5.2`,
  installed via `uv pip install ... --torch-backend=auto`. A capability check snippet confirms
  b12x availability: `has_flashinfer_b12x_gemm()` and `has_flashinfer_b12x_moe()` must both
  return True on sm_121. If b12x is unavailable, serving degrades to Marlin W4A16 (~2× slower).
  This corroborates the existing [conjecture] finding (S-forum-unsloth-qwen36, jbourny) that
  b12x is not available on stock vLLM — it requires the Unsloth-recommended install path.

- **[conjecture]** **vLLM 0.25.x startup hang on GB10** (S-forum-unsloth-b12x, rtamax): vLLM
  0.25.x hangs at "Waiting for 1 local, 0 remote core engine proc(s) to start" — both in Python
  env and Docker. No resolution posted in thread. Single source → [conjecture].

This finding **qualifies but does not overturn** the existing [reported] finding that Unsloth
NVFP4 is ~15% slower than nvidia NVFP4 on GB10 (S-forum-unsloth-qwen36, 3 independent sources).
The prior benchmarks all used Marlin backend for both quants; this benchmark uses b12x for
Unsloth only, suggesting the b12x backend — not the quant — may be the performance lever. A
controlled comparison (nvidia+b12x vs Unsloth+b12x) is needed to isolate the variable.

## Forum ingest: Qwen3.6-35B-A3B-FP8 DeepGEMM assertion + draft model training (2026-07-31)

> **evidence:** conjecture (single forum source for each finding)
> **sources:** S-forum-vllm-2607-xgrammar, S-forum-qwen36-draft-train

- **[conjecture]** **Qwen3.6-35B-A3B-FP8 fails to load on patched 26.07 container with
  DeepGEMM assertion** (S-forum-vllm-2607-xgrammar, rp_37716): on the xgrammar-patched
  `nvcr.io/nvidia/vllm:26.07-py3` derivative image, `Qwen/Qwen3.6-35B-A3B-FP8` fails to
  load entirely with `RuntimeError: Assertion error .../layout.hpp:59: Unknown SF
  transformation` during FP8 MoE weight-layout conversion in DeepGEMM. This is a separate
  issue from the xgrammar tool-calling bug — it's a DeepGEMM kernel/layout incompatibility
  with the Qwen3.6-35B-A3B FP8 checkpoint on GB10. The same patched image loads and serves
  `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` without issue. Single source → [conjecture].
  This is a new GB10-specific model-load failure — the error string
  `layout.hpp:59: Unknown SF transformation` is a DeepGEMM internal assertion during
  weight layout conversion, suggesting the FP8 MoE weight format in the Qwen3.6-35B-A3B-FP8
  checkpoint uses a scaling-factor transformation that DeepGEMM's GB10 path doesn't
  recognize. Corroborates the pattern of FP8 model-load fragility on sm_121 (cf. S-forum-
  sm121-4bugs, S-forum-nvfp4-broken). Flagged for hardware-agent verification.

- **[conjecture]** **~50 tok/s sustained decode with MTP nst=3 on vLLM for Qwen3.6-35B-A3B
  on DGX Spark** (S-forum-qwen36-draft-train, colizu2020): a forum user reports running
  Qwen3.6-35B-A3B on vLLM with MTP (`num_speculative_tokens=3`) for everyday coding and
  agentic tasks (2-3 subagents in Claude Code CLI), sustaining ~50 tok/s decode. This is
  a single-user self-reported number in the context of a draft-model training question, not
  a controlled benchmark → [conjecture]. The ~50 tok/s is consistent with the proven
  bandwidth-bound decode range for 3B-active MoE on single Spark (cf. the 41.8 tok/s
  no-MTP proven result and the 142 tok/s Atlas NVFP4+MTP result on a different stack).
  Single source → [conjecture].

- **[conjecture]** **Community advice: use existing DFlash drafter over training custom DSpark
  for Qwen3.6-35B-A3B** (S-forum-qwen36-draft-train, alexander.kachur): when asked whether
  training a personal draft model from the vLLM speculators GitHub repo is worth the effort,
  the advice was to use the already-existing DFlash drafter (`z-lab/Qwen3.6-35B-A3B-DFlash`)
  instead — DSpark is "only marginally better" and custom training "will cost a couple
  hundred [dollars] with uncertain benefits." This corroborates the existing DFlash ecosystem
  finding (S-forum-dflash-qwen122) and the DFlash drafter approach documented on
  `[[wiki/engines.md]]`. Single source → [conjecture]. No new durable technical finding
  beyond reinforcing the existing DFlash-over-custom-draft recommendation.

## Forum ingest: MoE LoRA training + vLLM serving (2026-08-01)

> **evidence:** conjecture (single forum source)
> **sources:** S-forum-moe-lora-vllm

- **[conjecture]** **Unsloth LoRA format incompatible with vLLM fused MoE expert
  tensors** (S-forum-moe-lora-vllm, haidij): LoRA adapters trained via Unsloth for MoE models
  (Qwen3.5-35B-A3B, Gemma-4-26B-A4B) fail to load in vLLM due to a mismatch between Unsloth's
  fused expert tensor format and vLLM's fused MoE weight loading. Training only attention
  layers (skipping experts) produces high-loss adapters that don't meaningfully affect output.
  Unsloth Studio itself OOMs on these MoE models on GB10. Single source → [conjecture].

- **[conjecture]** **NVIDIA AutoModel/NeMo provides official MoE LoRA recipes servable via
  vLLM** (S-forum-moe-lora-vllm, aniculescu/NVIDIA): the `NVIDIA-NeMo/Automodel` repo includes
  ready-to-run Gemma-4-26B-A4B (`gemma4_26b_a4b_moe_peft.yaml`) and Qwen3.5-35B-A3B
  (`qwen3_5_35b.yaml`) MoE LoRA fine-tuning recipes. The LoRA `peft:` block applies to
  language-side modules with the vision tower frozen. AutoModel saves PEFT checkpoints as
  HF-compatible `adapter_config.json` + `adapter_model.safetensors`, servable via:
  `vllm serve <base-model> --enable-lora --max-lora-rank 16
  --lora-modules automodel-adapter=/path/to/checkpoint/model`
  This is the official path for MoE LoRA training on GB10 that produces vLLM-compatible
  adapters. Single source → [conjecture].

## See also
`[[wiki/engines.md]]` · `[[wiki/quantization-on-gb10.md]]` · `[[wiki/models/holo-3.1.md]]` (Qwen3.5 VL MoE)

## Forum ingest: QLoRA fine-tuning Qwen3.6-35B-A3B on single Spark (2026-08-05)

> **evidence:** conjecture (single forum source)
> **sources:** S-forum-qlora-coding

- **[conjecture]** **Train bf16, serve NVFP4 with --enable-lora hot-attach** (S-forum-qlora-coding,
  jake.w.sims): NVFP4 (`compressed-tensors` / `nvfp4-pack-quantized`) is a post-training
  quantization format with **no gradient path** — you cannot fine-tune the NVFP4 checkpoint
  directly. Instead: train against the bf16 base, then serve the NVFP4 base with
  `--enable-lora`, hot-attaching the night's adapter. vLLM reports "MoE model detected.
  Using fused MoE LoRA implementation" and serves both the base (`Qwen3.6-35B-A3B-NVFP4`)
  and base+adapter (`Qwen3.6-35B-A3B-Coder-NVFP4`) simultaneously. Attaching an adapter
  takes seconds and costs a ~27 MB file per night. This avoids the nightly
  train→merge→re-quantize cycle (hours + hundreds of GB scratch). Single source → [conjecture].

- **[conjecture]** **flash-linear-attention gives 2.52× QLoRA throughput win on GB10**
  (S-forum-qlora-coding, jake.w.sims): adding `flash-linear-attention` (FLA, v0.5.1) to
  the training stack reduced per-step time from ~1700 s to **611 s/step** — a **2.52×**
  throughput improvement. The user ran without it for a month due to a missing package.
  Training stack: torch 2.10.0+cu128, transformers 5.5.0, triton 3.6.0, unsloth 2026.6.9,
  FLA 0.5.1. Single source → [conjecture].

- **[conjecture]** **QLoRA on MoE at batch_size=1 is severely compute-underutilized on GB10**
  (S-forum-qlora-coding, jake.w.sims, emptysands): with `per_device_train_batch_size=1`
  and `GA=16`, each 256-expert MoE expert sees only a handful of tokens per step.
  Effective throughput is ~5.3 TFLOP/s on hardware capable of well over 100 TFLOP/s.
  At a 2.5% trained-token fraction, the setup produces ~5 tokens of real loss signal per
  second. A 4B dense model with a real batch size would train in hours instead of weeks.
  The user kept batch_size=1 from an early bitsandbytes OOM workaround and never revisited
  it. Single source → [conjecture].

- **[conjecture]** **Claude Code session logs → SFT data pipeline for coding agents**
  (S-forum-qlora-coding, jake.w.sims, emptysands): agentic coding session transcripts
  (Claude Code `~/.claude/projects/`) can be parsed into supervised fine-tuning data —
  one example per assistant turn carrying a tool call or final answer. Key gotcha:
  `cleanupPeriodDays` in Claude Code `settings.json` defaults to 30 days, silently
  pruning old session transcripts. Set to 365 to avoid data loss. Mean context ~6.3k–9.5k
  tokens (with tools prefix), mean completion ~230 tokens. 97% of each forward/backward
  pass is masked context. The user's parser had a hardcoded `MAX_CTX_CHARS=24000` that
  truncated context at ~half the budget the tokenizer allowed. Community advice: most
  "behaviors" worth teaching (tool conventions, file layout habits) are better expressed
  as CLAUDE.md rules or hooks than fine-tuning — fine-tuning's real job is whatever's
  left after everything writable is written down. Single source → [conjecture].

## Forum ingest: Qwen3.5-397B architecture & 8× GB10 cluster benchmark (2026-07-15)

- **[conjecture]** **Qwen3.5-397B-A17B on 8× GB10: 31–35 tok/s FP8** (S-forum-qwen397-arch,
  raphael.amorim): the largest DGX Spark cluster reported in the forums (8× GB10) runs
  Qwen3.5-397B-A17B FP8 inference at 31–35 tok/s. MoE scaling gains flatten past TP=4 —
  the all-to-all interconnect overhead dominates at higher parallelism. This is a single
  forum reference; no configuration details or reproduction provided. See
  `[[wiki/multinode-tp-and-networking.md]]` for the interconnect bottleneck analysis.
- **[conjecture]** **Architecture comparison: Qwen3.6 dense vs MoE vs 397B**
  (S-forum-qwen397-arch, vedcsolution): parameter math validated from HF config.json files:
  - Qwen3.6-27B (dense): 64 layers, hidden 5120, 24 attn heads / 4 KV heads, intermediate
    17408, ~27B total
  - Qwen3.6-35B-A3B (MoE): 40 layers, hidden 2048, 16 attn / 2 KV heads, 256 experts (8
    routed + 1 shared), moe_intermediate 512, ~35B total / ~3B active
  - Qwen3.5-397B-A17B (MoE): 60 layers, hidden 4096, 32 attn / 2 KV heads, 512 experts
    (10 routed + 1 shared), moe_intermediate 1024, ~397B total / ~17B active
  All share: head_dim 256, vocab 248320, max_position 262144, rope_theta 10M,
  full_attention_interval 4, MTP 1 layer, vision encoder 27 layers/1152 hidden.
  "Qwen3.6-397B" (proposed upcycle) would require matching the 397B's expert count with
  the 3.6 architecture — feasibility is constrained by the interconnect, not memory.

## Forum ingest: Macaron-V1-Tall — Qwen3.6-35B-A3B base + LoRA specialists (2026-08-05)

> **evidence:** conjecture (single forum thread, multiple users in same thread)
> **sources:** S-forum-macaron-v1-tall

A forum thread (378436) on `mindlab-research/Macaron-V1-Tall` — a 50B-parameter model
composed of a 35B Qwen3.6-35B-A3B base and four 3.7B Rank-64 LoRA specialists (L0
general/chat, L1 personal-agent/tool, L2 coding, L3 UI/A2UI). Designed to fit a single
GB10 box at bf16 (~110 GB).

- **[conjecture]** **Macaron-V1-Tall on single Spark: 25-27 tok/s bf16, fp8 KV**
  (S-forum-macaron-v1-tall, TheAwakenOne): working spark-vllm-docker `vllm-node` recipe,
  TP=1, `--gpu-memory-utilization 0.7 --max-model-len 229376 --max-num-batched-tokens
  16384 --max-num-seqs 128 --kv-cache-dtype fp8 --enable-prefix-caching
  --tool-call-parser qwen3_coder --reasoning-parser qwen3`. Single source → [conjecture].
  Speed is ~half of Qwen3.6-35B-A3B-NVFP4 (which is ~50-90+ tok/s depending on quant /
  MTP), because Macaron runs bf16 (no NVFP4 quant). No lower-bit quants exist yet.

- **[conjecture]** **MTP nst=3 on Macaron: 71.5% acceptance but only +2% throughput**
  (S-forum-macaron-v1-tall, TheAwakenOne): adding `--speculative-config
  '{"method":"mtp","num_speculative_tokens":3}'` gives 71.5% draft acceptance (pos0
  84.7%, pos1 70.6%, pos2 59.2%), but actual throughput improvement is only +2% (41.93
  → 42.79 tok/s avg). The main benefit is reduced latency variance (std dev 5.67 → 4.30,
  -24%). The implied speedup from acceptance (~3.1×) is much higher than actual —
  because prefill cost is unchanged, MTP overhead adds forward passes, and acceptance
  <100% means many tokens still need full decode. Single source → [conjecture]. This
  corroborates the existing finding that MTP on Qwen3.6-35B-A3B can be a net negative
  or marginal depending on the quant and draft acceptance (see the proven MTP sweep
  above where NVFP4 with poor acceptance was strictly worse).

- **[conjecture]** **Macaron tool-eval: base Qwen 90/100, full Macaron router 82/100**
  (S-forum-macaron-v1-tall, jetspark): the Macaron routing system (L0 → specialist →
  answer → hidden summary) scores *lower* than the bare Qwen base on tool-eval-bench
  because most requests are routed to L0 (general chat, 91/105 routing decisions) rather
  than the tool specialist L1 (7/105). Direct base Qwen without LoRA = 90/100; full
  Macaron proxy = 82/100. The LoRA specialists add overhead without improving
  tool-calling. Single source → [conjecture]. Also: the `mods/fix-qwen3.6-chat-template`
  mod and `--chat-template fixed_chat_template.jinja` improve output quality (emX0r,
  jomark). Running all LoRA specialists simultaneously causes OOM on a single Spark
  (emX0r).

- **[conjecture]** **bf16 Macaron at ~110 GB leaves no room for lower quants or
  co-hosting** (S-forum-macaron-v1-tall, 0rand): at bf16 the model consumes ~110 GB of
  the 121 GB pool. 8-bit (FP8) Qwen3.6-35B-A3B gives 93/100 on tool-eval hardmode —
  "difference to bf16 is expected to be minimal if noticeable at all, but 4 bit is
  significant downgrade from 8 bits." No FP8 or NVFP4 Macaron checkpoint exists yet.
  Single source → [conjecture].

## Forum ingest: Qwen3.6-35B-A3B TP=2 Ray decode stall (2026-08-05)

> **evidence:** conjecture (single forum post, no replies)
> **sources:** S-forum-qwen36-tp2-stall

- **[conjecture]** **Qwen3.6-35B-A3B bf16 on 2× Spark TP=2 Ray: decode collapses to
  0.1-0.2 tok/s under concurrent requests** (S-forum-qwen36-tp2-stall, ammarabbaxi13):
  serving `Qwen/Qwen3.6-35B-A3B` (bf16, not NVFP4) on 2× Spark with TP=2, Ray executor,
  `--gpu-memory-utilization 0.85 --max-num-seqs 8 --max-num-batched-tokens 16384
  --attention-backend flashinfer --enable-prefix-caching --tool-call-parser qwen3_xml
  --reasoning-parser qwen3 --distributed-executor-backend ray`. Both GPUs consume 105 GB
  each. vLLM logs show generation throughput dropping to 0.1-0.2 tok/s with KV cache
  usage <12% — the model is alive but producing almost no tokens. The initial burst hits
  32.4 tok/s, then collapses. Single post, no replies → [conjecture]. This may be a
  Ray + cross-node scheduling issue, a UMA memory pressure stall (cf. S-forum-uvm-livelock),
  or a bf16 MoE load issue on 2× Spark. The bf16 (non-quantized) checkpoint at 105 GB/node
  leaves only ~16 GB for KV cache + workspace — a very tight margin on 121 GB UMA.
  Flagged for hardware-agent verification: does bf16 Qwen3.6-35B-A3B on 2× Spark TP=2
  Ray reliably stall under concurrency, and does NVFP4 avoid it?

## Forum ingest: KAT Coder v2.5 Dev (Qwen3.6 MTP) AutoRound quant (2026-08-11)

> **evidence:** conjecture (single forum thread, mixed benchmark signals)
> **sources:** S-forum-kat-coder-autoround

- **[conjecture]** **KAT-Coder-V2.5-Dev-MTP-int4-AutoRound-SAR — Qwen3.6 MTP headers grafted
  onto KAT Coder v2.5 Dev, quantized via Spark AutoRound** (S-forum-kat-coder-autoround,
  SlopOps): KAT Coder v2.5 Dev is a model trained for efficient thinking and fewer tool
  calls. The OP grafted Qwen3.6 MTP headers onto it and quantized to int4 using the Spark
  AutoRound method (S-forum-spark-auto-round). On Asus GB10 via vLLM with MTP: 85+ t/s
  accepted. Tool-eval 84/100 vs Ornith-1.0-35B-int4-AutoRound 87/100. OP prefers Ornith
  for coding; DannyTup found it "pretty bad" on bfcl/bigcodebench/ifevalcode and noted
  benchmark validation issues. The MTP-header grafting pattern (adding Qwen3.6 MTP headers
  to a non-Qwen base) is notable — it extends the MTP drafter approach to models not
  originally trained with MTP. See `[[wiki/quantization-on-gb10.md]]` for the Spark
  AutoRound tool context.

## Forum ingest: Qwen3.8-27B-MixedInt4-AutoRound — single-Spark dense (2026-08-19)

> **evidence:** conjecture (single forum thread, multiple users in same thread)
> **sources:** S-forum-qwen38-27b-mixedint4

A 41-post thread (380248, 3971 views) on Qwen3.8-27B quantized for single DGX Spark.
Qwen3.8-27B is a **dense** model (not MoE) — the same architecture class as Qwen3.6-27B
above, which is proven bandwidth-bound at ~30 tok/s on Spark. The thread confirms this
regime while adding quality and recipe data.

- **[conjecture]** **Qwen3.8-27B-MixedInt4-AutoRound by PILCOTHINK — mixed 4-bit quant,
  20.8 GB, MMLU recovery 99.32%** (S-forum-qwen38-27b-mixedint4, PILCOTHINK): mixed
  4-bit AutoRound quantization with sensitive layers kept FP8/FP16 and vision layers
  unquantized. Model: `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound`. MMLU: 83.49% →
  82.92% (-0.57 pp, 99.32% recovery rate). Category breakdown: Humanities -0.68 pp,
  STEM -0.82 pp, Social Sciences -0.33 pp, Other -0.38 pp. Seven additional quality
  metrics planned. Single source → [conjecture].

- **[conjecture]** **vLLM recipe: TP=1, fp8 KV, MTP nst=3, 1.01M max context, 2.56M-token
  KV pool (2.54× concurrency)** (S-forum-qwen38-27b-mixedint4, PILCOTHINK):
  ```
  vllm serve Qwen3.8-27B-MixedInt4-AutoRound \
    --gpu-memory-utilization 0.9 --trust-remote-code --tensor-parallel-size 1 \
    --reasoning-parser qwen3 --tool-call-parser qwen3_coder \
    --enable-auto-tool-choice --enable-prefix-caching \
    --max-model-len 1010000 --max-num-batched-tokens 8192 \
    --hf-overrides '{"text_config": {"max_position_embeddings": 1010000}}' \
    --kv-cache-dtype fp8 --max-num-seqs 10 \
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
  ```
  KV cache: 2,561,684 tokens → 2.54× concurrency at 1,010,000-token max context on a
  single Spark. The 20.8 GB model weight footprint leaves ample room for the large KV
  pool. Single source → [conjecture].

- **[conjecture]** **llama-benchy decode: 21.86 tok/s @ d0, 17.04 @ d4096, 17.82 @ d8192;
  prefill 828-877 tok/s** (S-forum-qwen38-27b-mixedint4, PILCOTHINK): llama-benchy
  pp2048/tg1024, c=1, --exact-tg. Decode is consistent with the proven bandwidth-bound
  dense regime (~30 tok/s ceiling for 27B dense on Spark at FP8+MTP; the MixedInt4
  variant's mixed-precision overhead may explain the lower number). Prefill is stable
  across depths (828→877 tok/s). Single source → [conjecture].

- **[conjecture]** **tool-eval-bench: 91/100 normal, 92/100 hardmode (v2.5.1)** 
  (S-forum-qwen38-27b-mixedint4, PILCOTHINK): hardmode run on vLLM 0.24.0+092c4842.dev,
  tool-eval-bench v2.5.1.dev29. Score 92/100 (73 pass, 8 partial, 3 fail), deployability
  82/100, responsiveness 59/100 (median turn 2.4s). Weakest category: M Autonomous
  Planning (67%). Single source → [conjecture]. Note: tool-eval-bench v2.5.0+ scores
  5-8 pts lower than v2.0.1 (S-forum-dragonscale) — cross-version scores are not
  directly comparable.

- **[conjecture]** **Spark AutoRound (SAR) variant: 88/100 hardmode v2.1.0, MTP nst=3
  15.08 tok/s / nst=4 15.67 tok/s** (S-forum-qwen38-27b-mixedint4, SlopOps):
  `slopops/Qwen3.8-27B-int4-AutoRound-SAR` quantized via Spark AutoRound method
  (S-forum-spark-auto-round). Tool-eval 88/100 hardmode v2.1.0 (69 pass, 10 partial,
  5 fail). MTP benchmarks: nst=3 → 15.08 tok/s (peak 23.67), nst=4 → 15.67 tok/s
  (peak 28.0). Prefill 403-439 tok/s — notably lower than PILCOTHINK's 828-877
  (different quant: SAR is uniform int4 vs MixedInt4). The SAR variant is slower in
  decode (15 vs 21.86 tok/s) — the mixed-precision approach preserves more speed
  despite keeping some layers at higher precision. Single source → [conjecture].

- **[conjecture]** **co-le: 35-40 tok/s on 2× Spark** (S-forum-qwen38-27b-mixedint4,
  co-le): "I tried it today, good drop-in replacement for the FP8, faster and still
  very good. Speeds were around 35-40 tps" on 2× Spark. The ~2× speedup over
  single-Spark (21.86 tok/s) is consistent with TP=2 splitting the dense model's
  compute across 2 nodes. Single source → [conjecture].

- **[conjecture]** **0rand: DSpark 28-35 tok/s 8-bit on single Spark; bare 17-18 tok/s,
  MTP3 ~26 tok/s** (S-forum-qwen38-27b-mixedint4, 0rand): platform-independent DSpark
  for 27B tested with ml-dspark: 38-50 tok/s on M5, should be ~28-35 tok/s on single
  DGX Spark. Bare version (no drafter) 17-18 tok/s, regular MTP3 ~26 tok/s. Notes
  hardmode tool-eval below Qwen3.6-27B 8-bit so far. Single source → [conjecture].

- **[conjecture]** **dean.grande: 27B as planner + 35B as coder dual-role approach**
  (S-forum-qwen38-27b-mixedint4, dean.grande): "3.8 27b is the king no doubt" for
  planning/one-shotting difficult briefs that take 3× iterations on 3.6 35B MoE.
  Recommends dual-role: 27B planner + 35B implementer. Zoo Code can auto-handoff
  plan → review → implement → review. Single source → [conjecture].

- **[conjecture]** **stu.miller: dense 27B is slow vs MoE 35B — 27B not a good first
  impression for new Spark users** (S-forum-qwen38-27b-mixedint4, stu.miller): "much
  faster MoE models exist that are almost as proficient and run at 50-80 t/s." 3.6
  35B is 2×+ faster than 27B dense. Important context: the 21.86 tok/s decode is
  bandwidth-bound dense (all 27B params per token), not a recipe failure. Consistent
  with the proven Qwen3.6-27B dense ~30 tok/s ceiling. Single source → [conjecture].

This thread reinforces the proven finding that **dense 27B on Spark is bandwidth-bound**
(~17-30 tok/s depending on quant and MTP), while MoE 35B-A3B is 2-4× faster (~50-90+
tok/s) because only 3B params are active per token. The MixedInt4 approach (sensitive
layers at higher precision) achieves 99.32% MMLU recovery — a quality-first quant
strategy for dense models where speed is already capped by bandwidth. The 2.56M-token
KV pool at 1.01M context is notable — the 20.8 GB weight footprint leaves most of the
121 GB UMA for KV cache, enabling very long context on single Spark.

## Forum ingest: Qwen3.8-27B NVFP4 vs FP8 A/B benchmark (2026-08-20)

> **evidence:** conjecture (single forum source, well-controlled A/B)
> **sources:** S-forum-qwen38-nvfp4-vs-fp8

A clean A/B benchmark (380258, 2514 views, 8 posts) comparing two quantizations of
Qwen3.8-27B (dense) on a single DGX Spark, vLLM 0.27.1, 16 concurrent prompts,
identical flags:

- **Qwen/Qwen3.8-27B-FP8** — fine-grained FP8 (block 128, e4m3), official Qwen FP8 build
- **unsloth/Qwen3.8-27B-NVFP4** — Unsloth Dynamic NVFP4 (4-bit MLP + 8-bit attention + FP8 KV cache)

- **[conjecture]** **Unsloth Dynamic NVFP4 is 30-34% faster than Qwen official FP8 on
  dense 27B at 16 concurrent** (S-forum-qwen38-nvfp4-vs-fp8, shahizat): across all three
  workload types (prompt-heavy, decode-heavy, balanced), the NVFP4 model consistently
  outperforms the FP8 model on aggregate output throughput:

  | Scenario (16 concurrent) | FP8 tok/s | NVFP4 tok/s | NVFP4 gain |
  |---|---|---|---|
  | Prompt-heavy (8k→1k) | 65.58 | 87.91 (peak 128) | +34% |
  | Decode-heavy (1k→8k) | 99.47 | 132.07 (peak 144) | +33% |
  | Balanced (1k→1k) | 104.44 | 134.41 (peak 144) | +29% |

  NVFP4 also wins on latency: TTFT prompt-heavy 29,815 ms vs 36,904 ms; TPOT decode-heavy
  120.60 ms vs 160.19 ms. Model size: 23.4 GB NVFP4 vs 30.9 GB FP8. KV-cache usage at
  decode-heavy: ~5.6% NVFP4 vs ~15% FP8. Single source → [conjecture].

  **GB10 relevance:** this is a dense model (all 27B params/token). The proven bandwidth-
  bound decode ceiling for 27B dense at FP8 is ~10 tok/s single-stream (measured 7.8
  baseline, S-forum-qwen36-27b-fp8). At 16 concurrent the aggregate numbers reflect batch-
  amortized bandwidth: NVFP4's smaller weight footprint (23.4 vs 30.9 GB) means fewer bytes
  per token, confirming the proven "fewer weight bytes = faster decode" rule. The NVFP4
  model is 4-bit MLP + 8-bit attention — fewer bytes on the bandwidth-dominant MLP layers.
  Single-stream decode from racerdude: ~10-12 tok/s on real coding tasks (medium thinking),
  consistent with the bandwidth-bound ceiling.

- **[conjecture]** **FP8 recipe with MTP nst=2 on single Spark** (S-forum-qwen38-nvfp4-vs-fp8,
  racerdude): working Docker recipe for Qwen3.8-27B-FP8:
  ```
  docker run -it --gpus all --ipc=host -p 8000:8000 \
    -e HF_TOKEN="${HF_TOKEN}" -e MAX_JOBS=4 \
    -e VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS="0" \
    -e VLLM_USE_DEEP_GEMM="1" -e CUTE_DSL_ARCH=sm_121a \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    vllm/vllm-openai:latest Qwen/Qwen3.8-27B-FP8 \
    --host 0.0.0.0 --port 8000 --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 --safetensors-load-strategy lazy \
    --enable-auto-tool-choice --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 --served-model-name qwen \
    --speculative-config '{"method":"mtp","num_speculative_tokens":2}' \
    --max-num-seqs 4 --max-model-len 212992 --max-num-batched-tokens 8192 \
    --gpu-memory-utilization 0.88 --enable-chunked-prefill --trust-remote-code
  ```
  Uses stock `vllm/vllm-openai:latest` with `VLLM_USE_DEEP_GEMM=1` and
  `CUTE_DSL_ARCH=sm_121a`. MTP nst=2, fp8 KV, lazy safetensors loading. Single source.

- **[conjecture]** **Jetson Thor shows similar NVFP4 advantage** (S-forum-qwen38-nvfp4-vs-fp8,
  shahizat): the same A/B on NVIDIA Jetson Thor shows NVFP4 winning on every metric —
  TPOT reductions of 26-30%, total throughput gains of 36-46%. This suggests the NVFP4
  advantage is not GB10-specific but extends to other memory-bandwidth-bound ARM+NVIDIA
  platforms. Single source → [conjecture], but the consistency across two platforms
  strengthens the directional finding.

  **Note on Unsloth NVFP4 context:** the existing [reported] finding (S-forum-unsloth-qwen36,
  3 independent benchmarks) shows Unsloth NVFP4 is ~15% *slower* than nvidia NVFP4 for the
  **MoE** Qwen3.6-35B-A3B. This new benchmark compares Unsloth NVFP4 against Qwen **FP8**
  (not nvidia NVFP4) for the **dense** Qwen3.8-27B. The comparison is cross-quant (NVFP4 vs
  FP8), not cross-vendor (Unsloth vs nvidia NVFP4). The result is consistent with the proven
  "fewer bytes = faster" rule — NVFP4's 4-bit MLP weights are smaller than FP8's 8-bit, and
  for a bandwidth-bound dense model that's the dominant factor. No contradiction with the
  Unsloth-vs-nvidia NVFP4 finding.

## Forum ingest: Multi-model orchestration on 3+ DGX Sparks (2026-08-20)

> **evidence:** conjecture (single forum thread, multiple users sharing deployment patterns)
> **sources:** S-forum-ds4f-qwen38-orchestration

A forum thread (380426, 1586 views, 11 posts) on combining DeepSeek-V4-Flash with
Qwen3.8-27B across multiple DGX Sparks under a Hermes agent harness. While primarily
an orchestration discussion, it contains several durable deployment-pattern findings.

- **[conjecture]** **"Virtual MoE" pattern: DS4F as architect + Qwen 27B as reviewer**
  (S-forum-ds4f-qwen38-orchestration, ajvazan): on 3× Spark, the OP runs DS4F on 2 Sparks
  as the main model, Qwen3.6-35B-A3B on the 3rd for subagents/small vision, and Qwen3.8-27B
  in low-reasoning mode for code review and backup vision. The pattern: DS4F as
  architect/planner/coder, Qwen 27B as reviewer for deep logic error detection after DS4F's
  output. The OP reports Qwen 27B finds deep logic errors that 122B misses during code
  review. Single source → [conjecture].

- **[conjecture]** **Qwen3.8-27B too talkative for Hermes agent harness — Qwen3.6-27B better
  auxiliary** (S-forum-ds4f-qwen38-orchestration, ajvazan): the new Qwen3.8 model is "very
  talkative and goes beyond the standard limits of Hermes," making it less suitable as an
  auxiliary model in agent frameworks with token limits. Qwen3.6-27B is recommended as
  the additional model instead. This is a model-behavior finding relevant to Spark users
  running multi-model agent setups (e.g. Hermes, Claude Code with custom endpoints).
  Single source → [conjecture].

- **[conjecture]** **122B on 3rd Spark beats 35B for coding aux role** (S-forum-ds4f-qwen38-
  orchestration, stu.miller): stu.miller uses DS4F on 2 Sparks + Qwen3.5-122B on the 3rd
  Spark for coding/vision (via a Hermes core hack for delegation). The 35B-A3B "ended up
  taking longer than 122b on most tasks due to looping or failing the adversarial review
  step." This contradicts the speed-first intuition (35B MoE is faster per-token) —
  for agentic workloads with review loops, the larger 122B's higher quality reduces total
  iterations. Single source → [conjecture]. See the existing 122B "king model" finding
  (S-forum-qwen122-king, [reported]).

- **[conjecture]** **3-Spark resource allocation pattern: 2× for main LLM, 1× for aux**
  (S-forum-ds4f-qwen38-orchestration, ajvazan + Ama5u): common deployment topology for
  3× Spark: 2 Sparks for the main large model (DS4F TP=2), 1 Spark for auxiliary models
  (subagents, vision, cron tasks, code review). With 4 Sparks, the 4th can be dedicated
  to media generation (e.g. MiniMax-H3). This is a practical GB10-specific allocation
  pattern — each 121 GB node can serve one model at a time (single-tenant constraint),
  so multi-model setups require multi-node. Single source → [conjecture].
