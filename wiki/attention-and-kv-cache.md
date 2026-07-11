# Attention backends & KV cache on GB10

> **area:** attention
> **status:** stable
> **evidence:** proven
> **sources:** S-m3-vision, S-mimo-results, S-mimo-doc, S-sess-jun5, S-sess-jun4, S-dflash-nvfp4, S-forum-mimo-2x-opt
> **updated:** 2026-07-11

Which `--attention-backend` to pass is decided by the model's attention type, not preference. Get it
wrong and KV-cache init fails or numerics are subtly off.

## Backend selection

| Backend | Use when | Notes on GB10 |
|---|---|---|
| **TRITON_ATTN** | sparse / MSA / block_size 128 archs (MiniMax-M3 MSA) | **required**, not optional, for MSA — supports `block_size=128` and is numerically correct on Blackwell. |
| **TRITON_ATTN_DIFFKV** | DiffKV archs where `v_head_dim ≠ head_dim` (MiMo-V2.5: v=128, head=192) | the MiMo path; from vLLM PR #41797, native in dev39-era images. |
| **FLASHINFER** | standard dense/MoE attention (Qwen3.x, Holo, Gemma-4) | the common fast path; default for ModelOpt NVFP4. |
| FlashAttention (ViT) | vision towers | first image triggers a one-time JIT autotune (~20 s) — see below. |

## Hard rules

- **[proven]** **`--block-size 128` + `TRITON_ATTN` for MSA / sparse-attention models.** FLASHINFER for
  the dense layers has **no common block size** with the MSA sparse/index cache → KV init fails ("No
  common block size for 128"). TRITON_ATTN supports 128. (MiniMax-M3.)
- **[proven]** **DiffKV needs TRITON_ATTN_DIFFKV.** `v_head_dim ≠ head_dim` checkpoints (MiMo-V2.5) won't
  run on a vanilla backend.
- **[proven]** **KV-cache dtype `fp8` (e4m3) is the default and is precise enough.** bf16 KV is *more*
  precise but rarely needed; on GB10 KV is cheap for sparse/hybrid archs anyway (few KV heads). MiMo uses
  `--kv-cache-dtype fp8`; M3's MSA + 4 KV heads make even 40k+ ctx cheap.
- **[proven]** **Sparse/hybrid attention makes long context cheap.** MSA (M3, 4 KV heads), Mamba-2
  hybrids (Nemotron-3: most layers SSM, few attn layers w/ 2 KV heads) → huge context for little memory
  (Nemotron-3 ran the model's full 1M ctx × 4 slots). The KV pool is rarely the constraint; **weights
  are**.

## Gotchas

- **[proven]** **First image request = ~20 s ViT JIT autotune** (`AttentionBackendEnum.FLASH_ATTN for vit
  attention`). Short client timeouts make the first call look like it "returned nothing." Use a long
  timeout (>150 s) or send a warmup image; every request after is fast. (M3 vision, Holo vision.)
- **[proven]** **flashinfer gemma-rmsnorm CUTLASS-DSL ICE.** Archs with `use_gemma_norm=true` call
  `flashinfer.norm.gemma_rmsnorm`, whose CUTLASS-DSL kernel can fail MLIR verification on cu130
  (`'llvm.mlir.global_dtors' requires attribute 'data'`). Fix: swap in pure-torch norms (verified
  coherent). Was misdiagnosed as a vision bug — it's a text-path norm. (MiniMax-M3.)
- **[proven]** **`--dtype bfloat16` may be mandatory** even when the checkpoint config says `float32`
  (Holo FP8/NVFP4 configs declare float32 — force bf16 activations or the launch is wrong).
- **[proven]** **Atlas KV flags differ** from vLLM (no `--kv-cache-gb`): `--max-seq-len`, `--block-size`,
  `--kv-cache-dtype`, `--kv-high-precision-layers`, `--ssm-cache-slots`, etc. See `[[wiki/engines.md]]`.

- **[proven]** **Spec-decode drafter + QUANTIZED target KV = page-unification wall.** A native spec
  proposer (DFlash/EAGLE3/MTP draft-model) registers the drafter's KV layers into the *global* paged
  allocator, which requires ONE physical page size across all layers. A bf16 drafter page can't unify with
  a 4-bit **nvfp4** target page (non-integer ratio; and NO GB10 backend implements the strided-padded-read
  `indexes_kv_by_block_stride` path — it's `False` on every backend) ->
  `NotImplementedError: The page size of the layer is not divisible by the maximum page size` in
  `unify_kv_cache_spec_page_size`. **Workaround:** run the drafter as a **`custom_class`** proposer so its
  KV is never registered globally (drafter runs standalone) -> the quantized target keeps its full pool.
  `fp8` (8-bit) target unifies more cleanly (2x bf16 ratio) so the native path can work there. Status:
  `workaround` — see `[[wiki/models/mimo-v2.5.md]]` (DFlash). Reusable for any spec-decode + quantized-KV
  combo on GB10.
- **[proven]** **Deep-context coherence tests need REALISTIC text.** Degenerate/repetitive filler prompts
  make models emit immediate-EOS at depth (a real behavior on pathological input) — this looks exactly like
  KV-quant corruption but is NOT. nvfp4 KV is coherent to 89k+ with varied content. Always test depth with a
  real corpus (`llama-benchy` uses a Gutenberg book for this).

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/platform-gb10.md]]` · `[[wiki/models/minimax.md]]` · `[[wiki/models/mimo-v2.5.md]]`

## Forum ingest: TRITON_ATTN_DIFFKV quantized KV guard (2026-07-11)

- **[conjecture]** **`TRITON_ATTN_DIFFKV` raises `NotImplementedError` on quantized
  `kv_cache_dtype`** (S-forum-mimo-2x-opt, renek): the backend's guard rejects any quantized KV
  cache dtype, but the underlying store kernel already accepts dtype + scales. The guard is
  defensive — disabling it allows `--kv-cache-dtype fp8_e4m3`, which roughly doubles the KV pool.
  This conflicts with first-party findings where fp8 KV works fine with TRITON_ATTN_DIFFKV
  (proven on dev39+), suggesting the guard was relaxed in later vLLM versions. If you hit this
  error on an older build, patching the guard is the workaround.
