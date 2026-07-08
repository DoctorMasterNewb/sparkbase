# Qwen (3.5 / 3.6 / Coder-Next)

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-sess-jun4, S-swapper, S-mimo-doc
> **updated:** 2026-07-08

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
