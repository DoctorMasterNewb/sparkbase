# K-EXAONE-236B-A23B (LG AI Research) — on DGX Spark

> **area:** model
> **status:** evolving
> **evidence:** conjecture
> **sources:** S-forum-kexaone-236b
> **updated:** 2026-08-19

**K-EXAONE-236B-A23B** is a 237B-parameter / 23B-active MoE from LG AI Research with 128 routed
experts across 47 MoE layers, a shared expert, a dense layer 0, and an MTP block (blk.48). It is
notable as the **largest reported unpruned model to fit and serve on a single DGX Spark** at its
full 262,144-token context — no expert pruning, no layer truncation, no distillation. All claims
are **[conjecture]** (single forum source).

## Why it bites on Spark

K-EXAONE-236B exercises two GB10-specific levers:
1. **LLLG sliding-window attention schedule** — only 12 of 48 layers hold the full context, so KV
   costs 48 KiB/token instead of the 192 KiB/token a fully global GQA stack would need. This is the
   key architectural feature that makes 262K context affordable on 121 GB UMA.
2. **Mixed-quant GGUF on ds4 engine** — per-tensor precision assignment (routers/norms/attention/
   shared expert/dense layer 0 at 8-bit or F32, routed experts at IQ2_XXS/Q3_K) compresses 441.63
   GiB BF16 → 85.56 GiB (5.16×), fitting within the 121.6 GiB UMA pool with room for KV + workspace.

## Single-Spark serving via ds4 engine

- **[conjecture]** **K-EXAONE-236B-A23B fits unpruned on single DGX Spark at full 262K context**
  (S-forum-kexaone-236b, Baekpica): the full model — all 128 of 128 routed experts, shared expert,
  dense layer 0, and original MTP block — runs on one GB10 via the ds4 engine (antirez/ds4 fork,
  `make cuda-spark` with `CUDA_ARCH=sm_121`). 781 tensors, identical to the BF16 source. Mixed-quant
  GGUF: `Baekpica/K-EXAONE-236B-A23B-MXQ-IQ2XXS-Q3K-Q4Edge-Q8Dense-MTPQ8-v1` (3 shards).
  - **Memory breakdown:** 84.48 GiB weights, 12.30 GiB KV at 262,144 tokens, 1.60 GiB workspace =
    103.95 GiB of 121.6 GiB resident.
  - **Cold start:** ~4 minutes (dominated by one-time aligned repack).
  - **Quant assignment:** routers, norms, attention, shared expert, and dense layer 0 stay at 8-bit
    or F32; first/last sparse blocks' experts at Q4_K; middle routed experts at IQ2_XXS (gate/up)
    + Q3_K (down). All 128 experts within a block use the same precision. Heuristic, not
    auto-optimized.
  - Single source → [conjecture].

- **[conjecture]** **LLLG sliding-window schedule makes 262K context affordable: 48 KiB/token KV**
  (S-forum-kexaone-236b, Baekpica): K-EXAONE uses an LLLG (layer-wise local-global) sliding-window
  schedule where only 12 of 48 layers hold the full context. This reduces KV cost to 48 KiB/token
  vs 192 KiB/token for a fully global GQA stack — a 4× KV savings that is the enabling factor for
  fitting 262K context in 12.30 GiB. This is a durable architectural insight for single-Spark
  large-model serving: sliding-window / local-global attention schedules dramatically reduce KV
  pressure on 121 GB UMA.

## Throughput

- **[conjecture]** **Decode and prefill scale with context as expected for bandwidth-bound MoE**
  (S-forum-kexaone-236b, Baekpica): greedy, thinking off, one cold prompt per measurement via
  `/v1/chat/completions` with streaming:

  | Prompt tokens | Prefill tok/s | Decode tok/s | TTFT |
  |---|---|---|---|
  | 1,451 | 53.0 | 10.51 | 27.4 s |
  | 3,941 | 51.6 | 9.05 | 76.4 s |
  | 8,222 | 47.9 | 7.38 | 171.6 s |
  | 16,376 | 42.3 | 5.42 | 387.1 s |

  Both curves fit cleanly and were validated predictively before extrapolating — decode predicted
  3.58 t/s at 32K against 3.56 measured. The OP emphasizes: "This is not a speed post. It is a
  'the whole model is here, and it serves' post." The throughput is consistent with a 237B MoE
  at mixed 2-4 bit on a single bandwidth-bound GB10 node.

## MTP — executes but net loss on GB10

- **[conjecture]** **MTP block (blk.48) executes but is a net loss on this hardware**
  (S-forum-kexaone-236b, Baekpica): the MTP block ships inside the same GGUF (no separate draft
  model). llama.cpp stores those tensors and ignores them; ds4 executes them. Every draft is
  verified against the target's own argmax and committed only on an exact token-ID match. However,
  MTP is a net loss on single GB10 for this model — the draft overhead exceeds the acceptance
  benefit at these throughput levels. Consistent with the general finding that MTP on
  bandwidth-bound single-node decode can be marginal (see `[[wiki/engines.md]]` → MTP quality).

## Multi-turn prefix-resume divergence

- **[conjecture]** **Multi-turn chat does not re-pay its prefill — re-tokenization mismatch**
  (S-forum-kexaone-236b, Baekpica): a chat client replays the assistant's previous reply as text,
  and re-tokenizing it does not reproduce the token IDs the model sampled — so an is-a-prefix KV
  test fails on a continuation that shares 98.6% of its tokens. **Workaround:** resume at the
  divergence point instead of full re-prefill. Result: turn 2 (same history + reply + follow-up)
  TTFT 5.9s with 6,992 tokens reused vs 165.7s cold for turn 1 — 24× speedup. Unrelated prompts
  are not falsely matched onto a live session (turn 3 = 137.0s cold, 0 reused). This is a durable
  finding for any GGUF/ds4 multi-turn serving: token-ID-level prefix matching is needed, not
  text-level.

## Validation

- **[conjecture]** **OpenAI-compatible API: 16/16 validation checks pass**
  (S-forum-kexaone-236b, Baekpica): streaming, `stream_options.include_usage`, thinking-mode
  `reasoning_content` separation from `content`, stop reasons, no state bleed across concurrent
  requests, greedy reproducibility. Long-prompt recall checked with a planted needle at 10%, 50%,
  and 90% depth of a 7K-token document — 3/3. Coherence verified.

## See also

`[[wiki/llama-cpp-rpc.md]]` (ds4 engine, GGUF on Spark, --no-mmap) ·
`[[wiki/quantization-on-gb10.md]]` (mixed-quant GGUF, IQ2_XXS, Q3_K, Q4_K) ·
`[[wiki/attention-and-kv-cache.md]]` (sliding-window attention, KV budget) ·
`[[wiki/platform-gb10.md]]` (121 GB UMA, bandwidth-bound decode) ·
`[[wiki/benchmarks.md]]` (full benchmark rows)