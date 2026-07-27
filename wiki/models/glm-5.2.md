# GLM-5.2 (Zhipu AI) — on DGX Spark

> **area:** model
> **status:** evolving
> **evidence:** conjecture
> **sources:** S-forum-glm52-4x, S-forum-glm52-mtp-fix, S-forum-glm52-1bit, S-forum-glm52-reapless, S-forum-glm52-800k, S-forum-glm52-iq4xs-4x, S-forum-glm52-8x, S-forum-glm52-vision, S-forum-glm52-hybrid, S-forum-flashinfer-livelock, S-forum-colibri-glm52, S-forum-6x-cluster
> **updated:** 2026-07-27

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

## Performance across configurations (cross-thread summary)

All numbers are **[conjecture]** or **[reported]** as noted. See `[[wiki/benchmarks.md]]` for full rows.

| Config | Quant | Nodes | Decode tok/s | Ctx | Source |
|---|---|---|---|---|---|
| TP=4, AWQ-INT4 + 15% expert prune + MTP | AWQ-INT4 | 4 | ~22 | 256K | S-forum-glm52-4x |
| TP=4, NVFP4, MTP4 fix | NVFP4 | 4 | 24 | 128K | S-forum-glm52-mtp-fix |
| TP=4, Hybrid FP8+MXFP4 (v2) | Hybrid | 4 | 20-25 | 800K | S-forum-glm52-hybrid |
| TP=4, Hybrid FP8+MXFP4 (v3-GPTQ) | Hybrid+GPTQ | 4 | 20.6 | 262K | S-forum-glm52-hybrid |
| TP=6, b12x | (Int4-family) | 6 | ~30 | — | S-forum-6x-cluster |
| TP=8, Int4-Int8 mix, b12x W4A8 | Int4-Int8 | 8 | 33-54 | 200K | S-forum-glm52-8x |
| TP=4, IQ4_XS GGUF, llama.cpp RPC | IQ4_XS | 4 | 6.28 | 1M | S-forum-glm52-iq4xs-4x |
| 1× Spark, Colibri expert streaming | int4 MoE + int8 MTP | 1 | 2.4-3.3 | short | S-forum-colibri-glm52 |

**[reported]** GLM-5.2 decode on 4× Spark is consistently in the 20-25 tok/s range across multiple
independent threads and quant formats (AWQ-INT4, NVFP4, Hybrid FP8+MXFP4) — the bottleneck is the
sparse-MLA attention + bandwidth-bound decode, not the quant choice. The 8× TP=8 run (33-54 tok/s)
is the outlier, attributed to the DCP collectives + b12x W4A8 backend + 2× the nodes.

## See also

`[[wiki/attention-and-kv-cache.md]]` (sparse_mla_sm120 livelock, KV format constraints) ·
`[[wiki/quantization-on-gb10.md]]` (NVFP4/MXFP4/AWQ, b12x W4A8, Int4-Int8 mix, quantized NextN mapping) ·
`[[wiki/engines.md]]` (MTP quality, adaptive MTP, Colibri expert streaming) ·
`[[wiki/multinode-tp-and-networking.md]]` (TP=4/6/8, DCP, NCCL buffer sizing) ·
`[[wiki/cudagraphs-and-compile.md]]` (MoE cudagraph wall, MTP-needs-cudagraphs) ·
`[[wiki/platform-gb10.md]]` (unified memory, no GPUDirect, bandwidth-bound decode) ·
`[[wiki/benchmarks.md]]` (full benchmark rows) ·
`[[wiki/roadmap.md]]` (open problems: paged-KV for Sm120, adaptive MTP overhead)