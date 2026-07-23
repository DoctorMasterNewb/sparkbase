# Mistral Small 4 (119B NVFP4)

> **area:** model
> **status:** evolving
> **evidence:** reported
> **sources:** S-forum-mistral-s4-119b, S-forum-mistral-s4-nvfp4
> **updated:** 2026-07-23

Mistral Small 4 (119B, released 2026-03-16) uses **Multi-head Latent Attention (MLA)** with
`kv_lora_rank=256`, `qk_nope_head_dim=64`, `qk_rope_head_dim=64`, `v_head_dim=128`, giving an
effective KV head size of **320**. This non-standard head size is the central GB10-specific
obstacle: standard vLLM MLA backends reject `head_size=320` on sm_121.

## The MLA head_size=320 wall on GB10

- **[reported]** **All stock MLA backends reject `head_size=320` on SM 12.1a** (S-forum-mistral-s4-119b,
  mrDragonFox, chuckchambersdev): the error from vLLM's `AttentionSelectorConfig` lists every backend
  failing:
  - `FLASH_ATTN_MLA`: head_size not supported, compute capability not supported
  - `FLASHMLA`: head_size not supported, compute capability not supported, `vllm._flashmla_C` not
    compiled (insufficient nvcc / arch not in target list)
  - `FLASHINFER_MLA`: requires `qk_nope_head_dim == 128`, but got 64
  - `TRITON_MLA`: head_size not supported (initially)
  - `FLASHMLA_SPARSE`: head_size not supported, non-sparse not supported, compute capability not supported
- **[reported]** **TRITON_MLA resolves head_size=320 on SM121** (S-forum-mistral-s4-119b, eugr,
  chuckchambersdev): with eugr's spark-vllm-docker (nightly vLLM build with SM121 patches), the
  TRITON_MLA backend handles `head_size=320` without issue. **No `VLLM_MLA_DISABLE=1` needed.**
  This was the breakthrough that made Mistral Small 4 work on GB10. The earlier workaround
  (`VLLM_MLA_DISABLE=1` on avarok v23) disabled MLA entirely — functional but suboptimal.

## Working recipe (single-node, GB10)

- **[reported]** **Proven config on vLLM 0.21.0 (native arm64 image)** (S-forum-mistral-s4-119b,
  0rand, chuckchambersdev):
  ```
  docker run --gpus all --ipc=host --shm-size 4g \
    -v ~/.cache/huggingface:/root/.cache/huggingface \
    -e HF_TOKEN="$HF_TOKEN" -p 8000:8000 \
    vllm/vllm-openai:v0.21.0-ubuntu2404 \
    --model mistralai/Mistral-Small-4-119B-2603-NVFP4 \
    --served-model-name mistral-small4 \
    --attention-backend TRITON_MLA \
    --enable-auto-tool-choice \
    --host 0.0.0.0 --port 8000 \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --tokenizer-mode mistral --config-format mistral --load-format mistral \
    --pipeline-parallel-size 1 --trust-remote-code \
    --gpu-memory-utilization 0.85 \
    --max-num-seqs 4 \
    --max-model-len 256000 \
    --max-num-batched-tokens 4096 \
    --override-generation-config '{"temperature":0.6,"top_p":0.9,"top_k":20,"repetition_penalty":1.05}' \
    --reasoning-parser mistral --tool-call-parser mistral
  ```
  - **Critical:** `--shm-size 4g` — **do NOT raise to 16g; that causes a kernel crash** on GB10,
    independent of gpu-memory-utilization. Must be 4g with batch 4096.
  - `--max-num-seqs 4` (not 8 — 4 is the stable ceiling for 256K context at 0.85 util)
  - `--max-num-batched-tokens 4096` (not 16384 — paired with the 4g shm-size constraint)

- **[reported]** **vLLM 0.25.1 publishes native linux/arm64 images** (S-forum-mistral-s4-119b,
  chuckchambersdev): `vllm/vllm-openai:v0.25.1-ubuntu2404` has arm64 manifests. **No custom
  Avarok/eugr image is required merely to get an ARM64 vLLM base anymore.** v0.25.1 adds further
  Blackwell/NVFP4 and speculative-decoding work — clean upgrade candidate but not yet benchmarked
  end-to-end on Mistral Small 4.

## Benchmark numbers

- **[reported]** **~28-30 tok/s single-stream decode** — corroborated by 5 independent forum users:
  - mrDragonFox (vLLM 0.17.2rc1, TRITON_MLA, FLASHINFER_CUTLASS MoE, NVFP4, fp8 KV): 33.2 tok/s @ 2K
    ctx, 31.7 @ 32K, 17.7 @ 60K; 10 concurrent → 100.3 tok/s aggregate (10.0/req)
  - cosinus (llama-benchy, tg32): 30.18 tok/s @ d0, 28.32 @ d4096, 26.84 @ d8192, 24.12 @ d16384
  - tenari (llama-benchy, tg32): 28.84 @ d0, 26.81 @ d4096, 25.02 @ d8192, 22.10 @ d16384,
    16.65 @ d32768
  - 0rand (vLLM 0.21, bench serve): 28.76 tok/s output (peak 30.00), 20 prompts @ concurrency 1
  - chuckchambersdev: 28.0 tok/s sustained with MLA enabled (patched image)
  All single-node DGX Spark. Context-depth degradation follows the expected bandwidth-bound curve.

- **[reported]** **Prefill 2,600-3,900 tok/s** (mrDragonFox): 2,600 tok/s prefill at 2K-8K context,
    dropping to ~2,560 @ 16K depth (cosinus), ~3,922 (tenari @ d0). Context-prefill (ctx_pp) is
    faster: 5,351-6,195 tok/s at 4K-8K depth (tenari).

## Known issues & workarounds

### reasoning_effort bug (vLLM 0.17.2rc1)

- **[reported]** **vLLM 0.17.2rc1 unconditionally passes `reasoning_effort` to
  `apply_chat_template`, but `mistral_common` 1.10.0 doesn't support it** → 400 error on every
  request: `Kwargs ['reasoning_effort'] are not supported by MistralCommonTokenizer.apply_chat_template`
  (S-forum-mistral-s4-119b, drew22, chuckchambersdev). Fix: patch
  `vllm/tokenizers/mistral.py` to only pass `reasoning_effort` when it's not None (drew22's diff).
  PR #37081 (juliendenize) is the upstream fix but doesn't apply cleanly to current main.

### Tool-calling leaks

- **[reported]** **`[TOOL_CALLS]` tokens leak into content without PR #39217** (S-forum-mistral-s4-119b,
  drew22): streaming tool calls broken for post-v15 Mistral tokenizers — `[TOOL_CALLS]` tokens
  leak into content instead of being parsed. PR #39217 (Mistral Grammar) fixes this via Lark
  grammar-based structured output. Drew22 published a working Docker image:
  `androiddrew/mistral4-vllm-spark:26-04-14` (built on eugr's spark-vllm-docker with PR #39217 +
  two additional patches: a Triton MLA decode kernel fix for `Lk > Lv` and the reasoning_effort fix).

### Eagle/MTP speculative decoding

- **[conjecture]** **Eagle MTP does not work** (S-forum-mistral-s4-119b, drew22, 0rand):
  `mistralai/Mistral-Small-4-119B-2603-eagle` eagle head is not loaded by vLLM. MTP/speculative
  decoding is non-functional for this model on GB10 as of vLLM 0.21. v0.25.1 may improve this
  (untested).

### Tokenizer v15 requirement

- **[reported]** **Mistral Small 4 requires tokenizer v15 (`mistral_common` ≥ 1.10.0)** (S-forum-mistral-s4-119b,
  chuckchambersdev): the avarok v23 stock image tops out at tokenizer v13. Fix:
  `pip install --upgrade mistral_common` inside the container. The `mistralllm/vllm-ms4:latest`
  image from Mistral is **x86-only** — no arm64 build (cosinus, chuckchambersdev confirmed).

### Two-node performance

- **[conjecture]** **Speed was unimpressive on two nodes** (S-forum-mistral-s4-119b, josephbreda):
  one user reported running it on 2× Spark with disappointing speed. No specific tok/s numbers
  given for the 2-node config. The model fits comfortably on a single node (119B NVFP4 ≈ 60 GB),
  so TP=2 is not memory-driven and the cross-node overhead may negate gains.

## Community Docker images

- **[reported]** **eugr/spark-vllm-docker** — the foundational base (S-forum-mistral-s4-119b,
  eugr, chuckchambersdev): precompiled SM121 wheels, nightly vLLM, FlashInfer. Build:
  `./build-and-copy.sh` (~10 min on single Spark). Supports `--apply-vllm-pr <N>` to layer PRs.
- **[reported]** **androiddrew/mistral4-vllm-spark:26-04-14** (S-forum-mistral-s4-119b, drew22):
  eugr's base + PR #39217 + Triton MLA decode fix + reasoning_effort fix. Working tool-calling
  on GB10.
- **[reported]** **vllm/vllm-openai:v0.21.0-ubuntu2404** (S-forum-mistral-s4-119b, 0rand): upstream
  native arm64 image — works for Mistral Small 4 without custom builds. v0.25.1 also available
  with arm64.

## Model quality assessment (forum)

- **[conjecture]** **Faster but less accurate than Nemotron-3-Super / Qwen 3.5-122B** (S-forum-mistral-s4-119b,
  tenari): "Quality wise I think nemotron is slightly more accurate on coding than mistral small,
  but it's just so much less verbose. On my spark they both go about the same speed but nemotron
  uses 2-3× the tokens." Also noted: after-market NVFP4 quants of Qwen may contribute to Qwen's
  verbosity vs author-published NVFP4 (Mistral, Nemotron). Single user opinion → [conjecture].

## See also

`[[wiki/attention-and-kv-cache.md]]` (MLA backend selection) · `[[wiki/quantization-on-gb10.md]]`
(NVFP4 on GB10) · `[[wiki/containers-and-tooling.md]]` (spark-vllm-docker) ·
`[[wiki/benchmarks.md]]`