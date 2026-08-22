# Kimi K3 (Moonshot AI) — on DGX Spark

> **area:** model
> **status:** evolving
> **evidence:** conjecture
> **sources:** S-forum-kimi-k3-coder-reap, S-sm121-nvfp4
> **updated:** 2026-08-22

**Kimi K3** is Moonshot AI's large MoE model. The REAP-320 variant is a pruned/expert-reduced
version that fits on 8× GB10 (968 GB combined). The full K3 model (~2.8T parameters, see
[[wiki/models/glm-5.2.md]] → "cluster ceiling" discussion) requires ~16× GB10 at 4-bit.

## Why it bites on Spark

- **2.8T total parameters** — the largest model community users have attempted on GB10 clusters.
  At 4-bit, ~115 GB usable per node → 16 nodes needed for the full model (~$100k, 2000-3200W).
- **MoE with same active expert count as full model** — the REAP-320 prune reduces total
  parameters but keeps the same number of active experts, so decode bandwidth is comparable
  to the full model.
- **MXFP4 quant** — the REAP variant ships in MXFP4 format, which dispatches via Marlin
  decompression on GB10 (~~no native FP4 compute~~ — corrected 2026-08-22: sm_121 *has*
  `mma…kind::mxf4.block_scale`; MXFP4 is **undispatched** here, not impossible.
  See [[wiki/quantization-on-gb10.md]]). (S-sm121-nvfp4)

## Kimi K3 Coder REAP-320 MXFP4 on 8× GB10 (2026-08-08 ingest)

- **[conjecture]** **Kimi K3 Coder REAP-320 MXFP4 on 8× GB10** (S-forum-kimi-k3-coder-reap,
  ciprianveg): first reported 8× GB10 run of a Kimi K3 variant. Measured with `llama-bench`
  (llama.cpp):

  | Test | Throughput (tok/s) | Peak (tok/s) | TTFR (ms) | Est. PPT (ms) | E2E TTFT (ms) |
  |---|---|---|---|---|---|
  | pp2048 | 685.74 | — | 2,702.15 | 2,699.29 | 2,702.15 |
  | tg1500 | 23.79 | 35.00 | — | — | — |
  | pp2048 @ d4000 | 541.09 | — | 10,171.28 | 10,168.42 | 10,171.28 |
  | tg1500 @ d4000 | 29.69 | 35.00 | — | — | — |
  | pp2048 @ d32000 | 678.43 | — | 45,261.54 | 45,258.68 | 45,269.10 |
  | tg1500 @ d32000 | 21.29 | 34.00 | — | — | — |

  Key observations:
  - **Decode 21-30 tok/s (tg1500), peak 35 tok/s** — consistent with bandwidth-bound decode for
    a large MoE at MXFP4 on 8× GB10. The ~24 tok/s average is in the same range as GLM-5.2
    (744B) on 8× Spark at Int4-Int8 mix (33-54 tok/s, S-forum-glm52-8x) — though K3 REAP-320 is
    a pruned variant with fewer total parameters.
  - **Prefill 541-686 tok/s** — strong prefill, consistent with multi-node TP prefill scaling.
    Prefill at d4000 (541) is lower than d0 (686) and d32000 (678) — the d4000 dip is unusual
    and may be a warmup/caching artifact.
  - **Decode holds across context depth**: 23.79 (d0) → 29.69 (d4000) → 21.29 (d32000) —
    relatively flat, suggesting the attention kernel is not the bottleneck at these depths.
  - **REAP variant "loops a lot"** — the OP notes the REAP pruned variant produces repetitive
    output, a known quality issue with expert pruning. The OP does not recommend this REAP
    variant and plans to try the full model on 16× GB10 or a better REAP on 8×.
  - **Engine**: `llama-bench` (llama.cpp), not vLLM — likely GGUF-converted or native MXFP4
    GGUF format. The OP references "dspark" (DSpark speculative decoding) in the thread title
    but the benchmark table is from `llama-bench`.

- **[conjecture]** **Full Kimi K3 requires 16× GB10** (S-forum-kimi-k3-coder-reap): the OP states
  the full K3 model needs 16× GB10 nodes. This is consistent with the cluster-sizing math in
  S-forum-kimi-k3-ceiling (~115 GB usable/node → 16 nodes for ~2.8T at 4-bit).

## See also

- [[wiki/models/glm-5.2.md]] — GLM-5.2 (744B) on 4-8× Spark, the closest comparable large-MoE
  benchmark data
- [[wiki/quantization-on-gb10.md]] — MXFP4 dispatch via Marlin on GB10 (a kernel-coverage gap; the FP4 MMA exists)
- [[wiki/benchmarks.md]] — collated decode tok/s table
- [[wiki/roadmap.md]] — open problems including cluster-scaling limits