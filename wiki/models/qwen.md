# Qwen (3.5 / 3.6 / Coder-Next)

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-swapper, S-mimo-doc, S-forum-unsloth-qwen36, S-forum-qwen397-arch
> **updated:** 2026-07-15

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

## See also
`[[wiki/engines.md]]` · `[[wiki/quantization-on-gb10.md]]` · `[[wiki/models/holo-3.1.md]]` (Qwen3.5 VL MoE)

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
