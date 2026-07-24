# Laguna-S-2.1-NVFP4

> **area:** model
> **status:** retired
> **evidence:** proven
> **sources:** S-laguna-v251-bench, S-forum-laguna-dflash, S-forum-laguna-quality
> **updated:** 2026-07-24

Laguna-S-2.1 — 117.6B MoE, 8.5B active, 256 experts (top-10 + 1 shared), 48 layers (36 SWA +
12 global), `sliding_window=512`, 256K context. NVFP4 W4A4 ~67 GB. Custom architecture
(`LagunaForCausalLM`, `trust_remote_code=True`, `compressed-tensors` quant). Paired DFlash
draft model (`Laguna-S-2.1-DFlash-NVFP4`, ~2 GB, 5 files). Requires vLLM 0.25.0+.

**[proven] Retired 2026-07-22 — not viable for current workflows.** Output quality is on par
with Qwen3.6-35B-A3B for prose and chart processing: completes long-context chart processing
with accurate results, but writing quality is lower than MiMo-V2.5 or DeepSeek-V4-Flash.
At 22.6 tok/s decode it offers no speed advantage over Qwen3.6 TP=2 (67 tok/s) while being a
larger model (69.3 GiB vs 23.4 GiB). Weights deleted, recipe removed, no swapper recipe retained.

## Working config

- **[proven]** Single-node TP=1 on vLLM 0.25.1 stock image (`vllm/vllm-openai:v0.25.1`) + FlashInfer
  nightly 0.6.15.dev20260712 (python + cubin + jit-cache+cu130) installed via bootstrap script
  with `--no-deps`. The stock 0.25.1 image ships FlashInfer 0.6.13; Laguna's NVFP4 attention path
  needs the nightly. (S-laguna-v251-bench)
- **[proven]** MoE backend auto-selects **FLASHINFER_CUTLASS** (same proven path as MiMo/Qwen NVFP4).
  Attention backend: **FLASHINFER**. (S-laguna-v251-bench)
- **[proven]** DFlash speculative decoding: `num_speculative_tokens=7`, `method='dflash'`.
  `max_num_seqs=4` (model card warns DFlash crashes at default 256). (S-laguna-v251-bench)
- **[proven]** Cudagraph: **PIECEWISE** only — FULL_AND_PIECEWISE not supported with DFlash spec-decode
  on FlashInfer backend (`UNIFORM_SINGLE_TOKEN_DECODE` limitation). 11 graphs, 0.43 GiB. (S-laguna-v251-bench)
- **[proven]** `reasoning_parser='poolside_v1'` — reasoning output goes to `reasoning` field, not
  `content`. `tool_call_parser='poolside_v1'`. (S-laguna-v251-bench)
- **[proven]** `gpu_memory_utilization=0.85` → 69.3 GiB model, 30.3 GiB KV cache (828K tokens).
  `max_num_batched_tokens=8192`, `max_num_seqs=4`, chunked prefill enabled. (S-laguna-v251-bench)
- **[proven]** `override_generation_config`: `temperature=0.7, top_p=0.95, top_k=20`. (S-laguna-v251-bench)

## Boot timeline

- **[proven]** Cold start on single-node: FlashInfer bootstrap ~2 min → weight load 69.3 GiB in
  12.5 min (746s) → cudagraph + KV cache profiling 1.7 min (99.6s, compilation 36.2s) → total
  ~15 min. (S-laguna-v251-bench)

## Benchmarks (llama-benchy, pp2048/tg128, 3 runs, depth sweep)

- **[proven]** First-party single-node TP=1 GB10 measurement. vLLM 0.25.1, DFlash spec=7,
  PIECEWISE cudagraph, FlashInfer nightly 0.6.15.dev20260712. (S-laguna-v251-bench)

| test | t/s | peak t/s | est_ppt (ms) | e2e_ttft (ms) |
|---|---|---|---|---|
| pp2048 @ d0 | 3215.7 ± 67.2 | — | 637.5 ± 13.4 | 923.9 ± 12.7 |
| tg128 @ d0 | 22.6 ± 1.6 | 32.7 ± 3.1 | — | — |
| pp2048 @ d4096 | 3903.4 ± 15.1 | — | 1574.7 ± 6.2 | 1862.3 ± 4.4 |
| tg128 @ d4096 | 19.4 ± 1.2 | 27.0 ± 2.8 | — | — |
| pp2048 @ d8192 | 3598.1 ± 18.2 | — | 2846.8 ± 14.3 | 3136.7 ± 14.1 |
| tg128 @ d8192 | 22.4 ± 1.7 | 30.0 ± 5.9 | — | — |
| pp2048 @ d16384 | 3472.4 ± 5.3 | — | 5308.9 ± 8.0 | 5592.4 ± 8.8 |
| tg128 @ d16384 | 20.5 ± 0.8 | 29.0 ± 0.8 | — | — |

Key findings:
- **[proven]** Decode 20–23 tok/s across all depths — flat, no context-depth degradation (unlike
  MiMo which drops to 8.8 @ 100k). The SWA architecture (36 SWA + 12 global layers, sliding_window=512)
  keeps attention O(1) per layer for decode.
- **[proven]** Prefill 3.2K–3.9K tok/s — strong for a 117B MoE single-node. Slight bump at d4096
  (3903) vs d0 (3216) likely due to SWA window warmup.
- **[proven]** Peak throughput 27–33 tok/s — the DFlash draft is occasionally accepted but mean
  decode (22.6) is well below peak (32.7), indicating **low draft acceptance** on prose workload.
  This is consistent with forum reports (S-forum-laguna-dflash): DFlash acceptance 18-40% with 7
  tokens on structured/agentic, 2-3% with 15 tokens. The benchmark uses War & Peace prose, which
  sits at the low end of acceptance.
- **[proven]** TTFT scales linearly with depth: 924ms @ d0 → 5592ms @ d16384. Prefill time
  dominates TTFT (637ms → 5309ms est_ppt).

## Comparison to forum reports

- **[reported]** Forum 377663 (vr8vr8): DFlash 7 spec tokens gives 40-50 tok/s on structured/agentic
  workloads; 20-36 tok/s without spec. Our 22.6 tok/s on prose is at the low end of the no-spec range,
  suggesting DFlash is providing minimal uplift on this workload. (S-forum-laguna-dflash)
- **[conjecture]** The gap between our 22.6 tok/s mean and forum's 40-50 tok/s is likely workload
  dependent. Structured/agentic prompts get higher DFlash acceptance (18-40%) than prose. Running
  a coding/JSON benchmark would likely close this gap. (S-forum-laguna-dflash)
- **[conjecture]** Forum 377674 (alperen.duru17): Laguna-S-2.1 on single Spark gets ~20-30 tps for
  reasoning — quality "as good as DeepSeek V4 Flash over two Sparks" for document reasoning with
  tool calls, but **fails single-shot generation tasks** (HTML game/simulation). This corroborates
  the first-party retirement finding: Laguna matches Qwen3.6-class quality for structured tasks but
  underperforms on generative tasks. The ~20-30 tps figure is consistent with the first-party 22.6
  tok/s decode measurement. (S-forum-laguna-quality)

## Operational notes

- Bootstrap script installs FlashInfer nightly trio with `--no-deps` (avoids vLLM 0.25.1
  dependency resolver conflict — vLLM pins flashinfer 0.6.13, but the nightly is needed for NVFP4).
  The `vllm 0.25.1 requires flashinfer-cubin==0.6.13 which is incompatible` warning is benign.
- FlashInfer JIT cache persisted at `~/.cache/flashinfer` via volume mount — avoids re-JIT on
  restart. First cold start compiles ~36s of inductor; subsequent restarts should skip this.
- `rope_parameters` unrecognized keys warning (`sliding_attention`, `full_attention`) is benign —
  Laguna's custom RoPE uses these fields internally; transformers doesn't recognize them but vLLM
  handles them natively.
- `reasoning_token IDs` auto-init failure is expected — `poolside_v1` parser handles reasoning inline.
- `max_num_scheduled_tokens` warning (8168 vs 8192) is a standard DFlash warning — the draft model
  reserves slots from the batch token budget.

## See also
- [[wiki/quantization-on-gb10.md]] — NVFP4 FLASHINFER_CUTLASS path
- [[wiki/cudagraphs-and-compile.md]] — PIECEWISE fallback with spec-decode
- [[wiki/models/mimo-v2.5.md]] — DFlash spec-decode acceptance patterns on prose vs structured