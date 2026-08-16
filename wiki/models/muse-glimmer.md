# Muse Glimmer 30B

> **area:** model
> **status:** evolving
> **evidence:** conjecture
> **sources:** S-forum-muse-glimmer, S-forum-muse-glimmer-nvfp4-w4a4
> **updated:** 2026-08-16

Meta's Muse Glimmer 30B — a 30B dense model with DFlash speculative decoding support. Community
tested it on DGX Spark across llama.cpp and vLLM. Early findings: throughput is reasonable for a
30B dense model on Spark (especially with DFlash), but tool-calling quality is poor and the model
is very sensitive to reasoning budget truncation.

## Architecture

- **[conjecture]** 30B dense model. Has an official DFlash assistant model
  (`meta-models/Muse-Glimmer-30B-assistant`) for block-speculative decoding. Also has an
  ExecuTorch variant with DFlash support. Runs on ~18 GB RAM (per Meta's announcement).
  (S-forum-muse-glimmer)

## llama.cpp on DGX Spark

- **[conjecture]** **UD-Q6_K_XL + DFlash benchmarks** (S-forum-muse-glimmer, pontostroy): using
  `unsloth/Muse-Glimmer-30B-GGUF:UD-Q6_K_XL` with llama.cpp (`b10354-d2f83055d`) + DFlash
  speculative decoding, `llama-benchy` pp2048/tg128 c1, 3 runs:
  - d0: **44.6 tok/s** decode, 673 pp t/s, TTFT 3408 ms
  - d4096: **35.0 tok/s** decode, 685 pp t/s, TTFT 8719 ms
  - d8192: **26.7 tok/s** decode, 682 pp t/s, TTFT 14174 ms

  Decode degrades with context depth (44.6 → 26.7 at 8K), consistent with bandwidth-bound dense
  decode on GB10. Prefill stays flat (~680 tok/s). ~31 GiB system memory used, no swap/OOM/crash.

- **[conjecture]** **coder543 llama.cpp field notes** (S-forum-muse-glimmer, coder543): on a
  DGX Spark with DFlash:
  - No DFlash: **13 tok/s** decode
  - DFlash prose: **25 tok/s**
  - DFlash code: **35 tok/s**
  - Prefill without DFlash: ~1000 tok/s; with DFlash: ~725 tok/s (DFlash hurts prefill)
  - DFlash works fine in llama.cpp (vLLM DFlash is broken — see below)

## vLLM on DGX Spark

- **[conjecture]** **NVFP4 + DFlash on vLLM** (S-forum-muse-glimmer, gaborm): using
  `Preyazz/Muse-Glimmer-30B-NVFP4` with patched spark-vllm-docker + DFlash:
  - BF16 + DFlash: **7.72 tok/s** aggregate (8.19 mean/turn)
  - NVFP4 + DFlash: **18.65 tok/s** aggregate (19.66 mean/turn) — **2.42×** over BF16
  - Recipe: `--gpu-memory-utilization 0.45`, `--max-model-len 131072`, `--max-num-seqs 16`
    (DFlash-safe house rule: ≤32; 256 crashes vLLM under DFlash), `--max-num-batched-tokens 8192`,
    `--enable-prefix-caching`, `--enable-chunked-prefill`, `--enable-auto-tool-choice`,
    `--tool-call-parser muse_glimmer`, `--reasoning-parser muse_glimmer`,
    `--speculative-config '{"model":"meta-models/Muse-Glimmer-30B-assistant","num_speculative_tokens":15,"method":"dflash"}'`
  - **Caveat:** `reasoning_content` — keep `max_tokens ≥ 1500` or a mid-CoT cutoff blanks both
    `content` and `reasoning_content` fields
  - Required cherry-picked patches from a vLLM PR to fix DFlash support in spark-vllm-docker

- **[conjecture]** **vLLM DFlash broken for Muse Glimmer** (S-forum-muse-glimmer, DannyTup):
  the vLLM container for Muse Glimmer fails to start, complaining that
  `DFlashMuseGlimmerAssistantModel` doesn't exist. vLLM PR #51655 adds Muse Glimmer support but
  DFlash integration is not working. llama.cpp DFlash works fine. SGLang DFlash also works
  (faster than vLLM). As of the thread date, vLLM is not the best place to run Muse Glimmer
  with spec decode.

- **[conjecture]** **Inferact/Muse-Glimmer-30B-NVFP4-W4A4 — 52.55 tok/s on DGX Spark with vLLM**
  (S-forum-muse-glimmer-nvfp4-w4a4, kuscsik): a Spark Arena Benchmark submission reports
  **52.55 tok/s** text generation for the `Inferact/Muse-Glimmer-30B-NVFP4-W4A4` model
  (NVFP4 W4A4 activation-quantized variant) on DGX Spark with vLLM. Single-post benchmark
  report with no recipe details (no flags, env vars, context length, or concurrency info
  provided). The number is higher than the previously reported 18.65 tok/s for Preyazz NVFP4
  + DFlash — the W4A4 activation quant path (real FP4 compute via Triton, similar to
  S-forum-flux2-nvfp4-compute) and absence of DFlash overhead may explain the difference,
  but without recipe details this remains an unverified single-source claim.

## Tool-calling quality

- **[conjecture]** **BFCL benchmark scores very low (10-12%)** (S-forum-muse-glimmer, DannyTup):
  the model scores only 10-12% on BFCL across both vLLM and SGLang. The primary failure mode is
  **multi-tool calls**: the model only calls the first tool when multiple are expected. Other
  models (Gemma 4) handle multi-tool fine. This reproduces across both vLLM and SGLang, suggesting
  it may be a model limitation rather than a runtime/parser bug. Temperature/top_p/top_k values
  in the model config may not be reaching vLLM (not in the JSON config file).

- **[conjecture]** **Controlled A/B: single-tool calls pass, multi-tool fails**
  (S-forum-muse-glimmer, contact.cgp.mgm): on a DGX Spark (GB10) with UD-Q6_K_XL, 8K context,
  CUDA 13/sm121a, llama.cpp built from Muse parser PR #26849 head:
  - 20/20 forced single-tool calls passed (both plain Q6 and Q6+DFlash)
  - 5/5 tool-error recovery cases passed
  - 5/5 JSON cases passed
  - 5/5 bounded coding cases passed
  - DFlash preserved scores while reducing the 40-case run from 858.5s to 210.4s — **4.08×
    end-to-end speedup**. Tool decoding increased from 8.18 to 34.81 tok/s, 35.9% draft acceptance.
  - **However**: a smaller token-budget profile collapsed to 8/20 tool calls with DFlash.
    The model is very sensitive to reasoning truncation and parser/template correctness.
  - Exact-format French instruction following: only 3/5 — not generally reliable yet.
  - The 20 tool tests force exactly one tool call per request, so they do NOT contradict the
    BFCL multi-call failure. The model can serialize one forced call but fails multi-tool
    planning/serialization.
  - Hypothesis: three separate things need evaluation: (1) correct Muse
    reasoning/tool parser and chat template, (2) enough completion budget to avoid cutting
    reasoning, (3) genuine multi-tool planning and serialization. Since both vLLM and SGLang
    reproduce the multi-call issue, point 3 may be a model limitation.

- **[conjecture]** **0rand: poor tool-call quality across quants** (S-forum-muse-glimmer, 0rand):
  Apple/Metal quants show same poor multi-tool performance. In Opencode, single tool call failed
  (called `read.read()` instead of `read()`). Tested 4, 6, and 8 bit quants — all performed with
  generally same poor tool-call quality (77/100 on 2.0.1 old-style tool eval bench where good
  numbers go 90+). "Very slow" on Metal. Does not attribute to parser quality — rather model/quant
  property.

## Assessment

- **[conjecture]** Muse Glimmer 30B on DGX Spark: throughput is reasonable (13-44 tok/s depending
  on quant/DFlash/context), and DFlash provides a significant speedup (4× on controlled tool-call
  suite, 2.4× NVFP4-vs-BF16 on vLLM). However, tool-calling quality is not production-ready:
  multi-tool serialization fails across all tested runtimes (vLLM, SGLang, llama.cpp) and quants.
  The model is willing to call tools (unlike some competitors) but cannot reliably plan multi-tool
  sequences. Very sensitive to reasoning budget truncation. Verdict from the community: "very
  promising for local, on-demand batch/coding work, especially with DFlash, but not production-ready
  tool calling yet." Community attention expected to shift to Qwen 3.8 27B. (S-forum-muse-glimmer)

## See also
`[[wiki/engines.md]]` · `[[wiki/benchmarks.md]]` · `[[wiki/quantization-on-gb10.md]]`