# Grafting a vision adapter onto a text-only LLM (GB10)

> **area:** model
> **status:** open-problem
> **evidence:** mixed
> **sources:** S-dsv4-vision, S-m3-vision, S-sm121-nvfp4
> **updated:** 2026-08-24

A growing class of community checkpoints bolts a **frozen vision tower + a small trained
projector** onto a text-only LLM (Baseten's GLM-5.2-Vision-NVFP4 method; webbrain-one's
DeepSeek-V4-Flash-0731-Vision). They are attractive on a Spark pair — the adapter is
under 1 GB on top of a checkpoint you already have — and they are the easiest place to
waste a night, because **an adapter that is wired correctly and an adapter that works
look identical until you measure the embeddings.**

This page is the GB10-specific trap list and the verification method. Model-specific
results live on the model pages.

## Verify with numbers, not with output

**[proven]** A grafted adapter can run end to end — tower executes, projector executes,
placeholders expand, requests return fluent text — while the model is functionally
blind. Fluent, plausible, image-shaped answers are **not** evidence of sight: a text LLM
handed a constant vector still writes a confident description. (S-dsv4-vision)

Three cheap tests, in order. Run them before tuning anything.

1. **Solid-colour discrimination.** Send pure red, pure blue, pure green and ask for the
   colour in one word. This is the easiest visual task that exists. A working adapter is
   perfect here; ours answered "Blue" three times.
2. **A no-image control.** Ask the same question with no image. If the model's answer
   with an image matches its blind prior, the image contributed nothing — a "correct"
   answer can be the prior leaking through.
3. **Embedding-norm ratio.** Compare the projector's output norm against the LLM's own
   token-embedding norm (`embed.weight`, per-row L2). These must be the same order of
   magnitude. This one test would have saved most of a night.

**[proven]** Stage-isolate before blaming the projector: compute relative L2
(`||a-b|| / ||a||`) for the *same pair of images* at pixel, tower-output and
projector-output stages. It separates "preprocessing is wrong" from "tower is dead" from
"projector is miscalibrated". Include an `image vs itself` row (must be exactly 0) and an
`image vs noise` row (must be large) to bracket the scale. (S-dsv4-vision)

**[proven]** Do not judge by cosine of mean-pooled embeddings. Deep ViT features share a
large common component: two images that a stage clearly distinguishes can still pool to
cosine **+0.9999**, and the same pair shows a 16% relative L2 difference. Centre the
features (subtract the per-dimension mean across images) or use relative L2.
(S-dsv4-vision)

## The scale trap

**[proven]** The published DeepSeek-V4-Flash-0731-Vision projector emits per-token norms
of **~127–136**, while the LLM's token embeddings sit at **~7.3** (dealignai CRACK) and
**~4.9** (webbrain's own NVFP4 text half — *the weights the projector was trained
against*). That is **26× oversized against its own training target**. Over 98% of each
image embedding is a constant offset shared by every image; only ~1–2% varies with
content, so the constant swamps the signal in the residual stream. (S-dsv4-vision)

**[proven]** Rescaling image embeddings to the model's own embedding norm is what makes
answers vary by image at all — but rescaling alone does not buy comprehension. A sweep
from 0.005 to 0.5 found no setting that reliably names a solid colour: too small and the
image is ignored entirely, too large and answers collapse to a constant again.
(S-dsv4-vision)

**[conjecture]** The most likely cause is a **projector/text-half pairing** failure — a
projector trained against one quantisation of a base model does not transfer to a
different derivative of it. The decisive A/B (same adapter, the author's own text half)
was blocked on GB10 by the NVFP4 MoE gap below, so this stays conjecture.

## GB10 traps, in the order they bite

**[proven]** **The ViT attention backend must be overridden.** vLLM's multimodal encoder
defaults to a prebuilt `flash_attn_varlen_func`, whose PTX this GB10 driver rejects the
moment the encoder first runs:

```
CUDA error: the provided PTX was compiled with an unsupported toolchain
```

Fix: `--mm-encoder-attn-backend TORCH_SDPA` (a supported CUDA path in
`mm_encoder_attention.forward_cuda`, no JIT). Confirm the engine logs
`Using AttentionBackendEnum.TORCH_SDPA for MMEncoderAttention` — if it still says
`FLASH_ATTN`, the flag did not land. A 27-layer tower over ≤2048 patches is not where the
time goes. (S-dsv4-vision; see `[[wiki/attention-and-kv-cache.md]]` for the ViT JIT note.)

**[proven]** **Cudagraphs must be off.** Capture succeeds; the first real request then
dies with an illegal memory access inside a *text* attention FP8 GEMM. Runs clean with
`--enforce-eager`. Upstream's own SGLang serve config sets `disable-cuda-graph: true` for
the same adapter, so treat eager as the model's documented state, not a local workaround.
Root cause unresolved. (S-dsv4-vision; `[[wiki/cudagraphs-and-compile.md]]`)

**[proven]** **The image placeholder must be an atomic token, or nothing matches.** If
the placeholder is ordinary text, BPE merges its trailing character with whatever
follows: `<image>` alone tokenises to `[30, 10253, 32]`, but before a newline in a real
prompt it becomes `[30, 10253, 1018]` — the `>` fuses with `\n`. vLLM then cannot locate
the placeholder and every image request fails with:

```
Expected there to be 1 prompt placeholders corresponding to 1 image items,
but instead found 0 prompt placeholders!
```

Fix: register the placeholder as an **added special token** in the served model directory
(`tokenizer.add_special_tokens({"additional_special_tokens": ["<image>"]})`, then
`save_pretrained`). That is exactly the mechanism that prevents BPE merging. Verify it is
atomic *in context*, not just standalone. Checkpoints whose placeholder is already a real
vocab token (Kimi's `<|media_pad|>`) never hit this. (S-dsv4-vision)

**[proven]** **A model with no jinja chat template silently drops the image.** DeepSeek
V4 ships none — `tokenizer-mode deepseek_v4` uses a custom text-only encoder. With the
default `auto` content format the chat layer hands that encoder structured content parts,
it renders only the text, and the placeholder never reaches the prompt (same "found 0
placeholders" error, different cause). Fix:
`--chat-template-content-format string`, which flattens parts to text *with* the
placeholder. Check with `parse_chat_messages(..., "string")` offline before blaming the
processor. (S-dsv4-vision)

**[proven]** **An out-of-vocabulary placeholder id will fault via the padding rows.** These
adapters often set `image_token_id == vocab_size` (deliberately one past the end). vLLM
masks such ids only where its `is_multimodal` mask says so — that mask covers the
scheduled rows, **not** the cudagraph padding rows of the runner's static id buffer, which
still hold whatever profiling and capture left there (dummy image prompts). Embedding
those rows is an out-of-bounds gather. It surfaces asynchronously, typically as an illegal
memory access in an unrelated attention GEMM on a *text* request. Sanitise every id
`>= vocab_size` to 0 before embedding. (S-dsv4-vision)

**[proven]** **Hash-routed MoE architectures need the raw token ids.** DeepSeek V4's first
`num_hash_layers` layers select experts through `gate.tid2eid`, a `[vocab_size,
num_experts_per_tok]` table indexed by raw token id, and raise outright if `input_ids` is
absent — which is exactly what vLLM passes a multimodal model by default. Set
`requires_raw_input_tokens = True` on the model class, and substitute in-vocabulary ids at
the image positions (these checkpoints ship a routing palette for this) before the lookup.
(S-dsv4-vision)

## The NVFP4 text-half dead end

**[proven]** An adapter is only as serveable as the text half it was trained against. On
this runtime, MJPansa's DeepSeek-V4-Flash-0731-**NVFP4** conversion cannot be served at
all:

- native path → `NotImplementedError: No NvFp4 MoE backend supports the deployment
  configuration`
- `VLLM_TEST_FORCE_FP8_MARLIN=1` fallback → the same
  `the provided PTX was compiled with an unsupported toolchain`, this time in the Marlin
  FP4 **repack** (`marlin_utils_fp4.prepare_nvfp4_moe_layer_for_marlin`).

Two independent kernel-coverage gaps, so the author's own pairing could not be tested.
Budget for this before downloading 165 GB. (S-dsv4-vision;
`[[wiki/quantization-on-gb10.md]]`)

## Porting an SGLang adapter to vLLM

**[proven]** These adapters usually ship as a source-patched **SGLang** extension. The
port is often mostly wiring: a MoonViT tower and a Kimi PatchMerger projector already
exist in recent vLLM (`model_executor/models/kimi_k25_vit.py`), and published tensor names
(`encoder.blocks.N.wqkv`, `proj.0`/`proj.2`) may match vLLM's directly — SGLang's copy
needs a rename, vLLM's does not. Verify by loading with `strict=True`. (S-dsv4-vision)

**[proven]** Watch the config-key translation. vLLM's `KimiK25MultiModalProjector` uses
`mm_hidden_size` as the projector's **output** width, but these checkpoints write
`mm_hidden_size` = the *tower* width and put the real output width in `text_hidden_size`.
Check against the published `proj.2` weight shape (`[4096, 4608]` ⇒ output 4096) rather
than trusting either key. (S-dsv4-vision)

**[proven]** Keep the architecture string unchanged (`DeepseekV4ForCausalLM`). It drives
attention selection, the FP4 expert layout, the quantization-config lookup and the
spec-decode drafter's target lookup; renaming it breaks all four. Bind the *name* to the
wrapper class instead. (S-dsv4-vision)

**[proven]** Spec-decode survives the graft when the drafter's layers sit past the hash
layers (DSpark stages 40/41/42 vs `num_hash_layers=3`), and vLLM's drafter logs
`Draft model does not support multimodal inputs, falling back to text-only mode` — correct
behaviour, not a defect: images are consumed at prefill and the drafter only proposes text
tokens for the target to verify. The drafter does read `config.image_token_index` (HF's
spelling) for multimodal targets, so alias it if your config uses `image_token_id`.
(S-dsv4-vision)

## See also
`[[wiki/attention-and-kv-cache.md]]` · `[[wiki/quantization-on-gb10.md]]` ·
`[[wiki/cudagraphs-and-compile.md]]` · `[[wiki/models/holo-3.1.md]]` (a VLM that works —
native arch, no graft) · `[[wiki/roadmap.md]]`
