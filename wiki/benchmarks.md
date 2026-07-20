# Benchmarks (GB10 / DGX Spark)

> **area:** benchmarks
> **status:** evolving
> **evidence:** proven
> **sources:** S-sess-jun4, S-sess-jun5, S-m3-20tps, S-nemotron-rpc, S-mimo-doc, S-minimax-sweeps, S-swapper-sweep, S-dgxspark-report, S-diffusiongemma, S-forum-dsv4-flash, S-forum-dsv4-dspark, S-forum-glm52-4x, S-forum-mimo-2x, S-forum-mimo-3x, S-forum-m3-llamacpp-2x, S-forum-m3-awq-4x, S-forum-mxfp4-patches, S-forum-qwen122, S-forum-mimo-dflash-22-67, S-forum-glm47-full-2x, S-forum-ds4f-4x-vllm, S-forum-nemotron-super-mtp, S-forum-nemotron-ultra-4x, S-forum-m25-sglang-4x, S-forum-glm47-rdma, S-forum-glm52-iq4xs-4x, S-forum-roce-397b-mtp, S-forum-gemma4-mtp-4x, S-forum-qwen122-nvfp4-redhat, S-forum-qwen122-nvfp4-quant, S-forum-nemotron-super-abi, S-forum-ds4f-hybrid-1x, S-forum-step37-llamacpp, S-forum-gemma4-assistant, S-forum-qwen36-27b-fp8, S-forum-llama-benchy, S-forum-flux2-nvfp4-compute, S-forum-mimo-sglang-4x, S-forum-m27-recipe, S-forum-diffusion-speeds, S-forum-mimo-2x-opt, S-forum-4node-crs504, S-forum-tokenspeed, S-forum-unsloth-qwen36, S-forum-qwen397-arch, S-forum-colibri-glm52, S-forum-nvfp4-broken, S-forum-dsv4-abliterated, S-forum-nemotron-ollama, S-forum-glm52-8x, S-forum-6x-cluster
> **updated:** 2026-07-20

Single-stream decode unless noted. All on the 2× GB10 pair. Numbers anchor the rules on
`[[wiki/platform-gb10.md]]` (bandwidth-bound) and `[[wiki/quantization-on-gb10.md]]` (MoE-NVFP4 wins).
Append rows as new models are benched; cite the source.

- **[proven]** Every row below is a first-party measurement on a real GB10 pair **except** rows marked
  `(vendor)`, which are **[reported]** (vendor-claimed, not independently reproduced here).

| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Mem/node | Notes |
|---|---|---|---|---|---|---|---|
| gemma-4-12B-it | bf16 | vLLM | 1 | 7.6 | 32k | 22.3 GiB | bandwidth-bound baseline |
| gemma-4-12B-it | FP8 online-dynamic | vLLM | 1 | **15.8** | 32k | 12.9 GiB | 2.08× bf16, native CUTLASS, no quality loss |
| gemma-4-31b | NVFP4 | Atlas | 1 | ~9 | 16k | — | dense (vendor) — [reported] |
| gemma-4-26b-a4b | NVFP4 | Atlas | 1 | ~67 | — | — | hybrid attn+MoE (vendor) — [reported] |
| Holo-3.1-35B-A3B | FP8 block-scale | vLLM (omni img) | 1 | 38.3 | 32k | 35.6 GiB | Marlin FP8 fallback |
| Holo-3.1-35B-A3B | NVFP4 (modelopt) | vLLM (gemma4-unified) | 1 | **76.9** | 32k | 20.4 GiB | 2× FP8; aggregate 899 tok/s @128 conc |
| Qwen3.6-35B-A3B (MoE) | NVFP4 + MTP, fp8 KV | Atlas | 1 | **142.3** (peak 144) | 256k pool | ~43–64 GB | the fast regime; service ~76–96 |
| Qwen3.6-27B (dense) | FP8 + MTP, bf16 KV | Atlas | 1 | 30.2 (peak 49.7 MTP) | 8k | ~106 GB | dense = all params/token |
| MiniMax-M2.7-abliterated | AWQ 4-bit | vLLM | 2 (EP=2) | **23.9** (peak 25 MTP) | — | — | durable default; prefill pp2048 ~1100, pp8192 ~1268 tok/s |
| MiniMax-M2.7 (cyankiwi) | AWQ 4-bit | vLLM | 2 (EP=2) | ~22 (peak 23) | — | — | prefill pp2048 ~930–1120 |
| MiniMax-M2.7 | NVFP4 (FlashInfer-CUTLASS) | vLLM | 2 (EP=2) | **16.5** (peak 18) | — | — | **decode slower than AWQ**, prefill higher (pp2048 ~1320–1500). EP=2 raw tg32 ~15.3 |
| MiniMax-M3 | AutoRound-3.2bit | vLLM | 2 (TP=2) | ~5 (prefill ~74) | 40k | 88.4 GiB | eager+cross-node bound; superseded by EAGLE3 row |
| **MiniMax-M3 PRODUCTION** | AutoRound + EAGLE3, **nvfp4 KV**, inductor compile | vLLM dev537 | 2 (TP=2 EP=2) | **12.6 prose / 15.5 code (peak 19)** (prefill 308) | **262k** (pool 359k tok) | ~91 GiB | 2026-07-04 image :nvfp4kv-compile; clean reasoning field (m3_mmthink), tools, vision, multi-turn; CG replay open |
| MiniMax-M3 + EAGLE3 + compile (fp8) | AutoRound + draft, fp8 KV, inductor | vLLM dev537 | 2 | ~13 prose / 16.3 code (peak ~20) (prefill 321) | 102k | 90.7 GiB | image :compile; slightly faster, less ctx |
| MiniMax-M3 + EAGLE3 (eager) | AutoRound-3.2bit + Inferact draft, **fp8 KV** | vLLM dev537 | 2 (TP=2 EP=2) | **13.3 prose / 15.3 code (peak 20)** (prefill 310; 340 @ 68k depth) | **102k** (pool 244k tok) | 90.7 GiB | 2026-07-03; vision + tools working; nst=3, util 0.90, mnbt 4096, video-mm off; 68k needle PASS; nst=5/draft_tp=1 WORSE (9.6/12.9); reasoning-parser OFF (−30% streaming); stack `minimax-m3-eagle3` |
| **MiniMax-M3-W4A16-GPTQ (b12x)** | GPTQ W4A16 + nvfp4 KV + EAGLE3 | vLLM b12x (a3refaat) | 2 (TP=2) | **36.3 tg32 / 34.7 tg128 (peak 41)** (pp2048 **1028**) | 32k (desktop-head; 196k headless) | ~104 GiB | 2026-07-05; forum 375595 reproduced EXACTLY (35.5 — [reported] origin); **VISION WORKS** (drop --language-model-only: 32.9 tg32 w/ ViT, exact OCR, 113k pool); needs healthy GPU clocks (wedge→20) + warm triton cache both nodes |
| Nemotron-3-Super-120B-A12B | Q8_0 GGUF | llama.cpp RPC | 2 | ~10.5 | 1M | ~61 GB | 128 GB model, fits neither node alone |

## Swapper menu — full sweep (2026-06-30, post power-cycle + cross-node fixes)

Single-methodology sweep of every validated recipe on a serving supervisor / model-swapper,
back-to-back via a bench-recipe harness (short-prompt g256 single-stream = decode rate,
`/v1/completions`). Measured **after** the DGX Spark power-controller wedge was cleared
(`[[wiki/platform-gb10.md]]`) and after fixing 4 sparkrun cross-node bugs (below) — so these
**supersede earlier suppressed numbers** (e.g. minimax **43.8** now vs the 23.9 recorded while the
head was likely wedged / not truly cross-node).

- **[proven]** All med/max tok/s below are first-party GB10 measurements. The role column names which
  node role served the recipe.

| Recipe (role) | Model | Quant | Engine | Nodes | med/max tok/s | TTFT | conc4 / conc8 agg |
|---|---|---|---|---|---|---|---|
| head text | Qwen3-Coder-Next-NVFP4-GB10 | NVFP4 | vLLM | 1 (head) | **64.0** / 64.3 | 93 ms | 44 / 306 |
| qwen36-uncensored | Qwen3.6-35B-A3B-Uncensored-Aggressive | bf16 MoE + fp8-KV | vLLM TP=2 (Ray) | 2 | **50.5** / 50.7 | 84 ms | 145 / 131 |
| minimax-m2.7-ablit | MiniMax-M2.7-abliterated-heretic-AWQ | AWQ 4-bit | vLLM TP=2 (mp) | 2 | **43.8** / 43.9 | 52 ms | 172 / **257** |
| deepseek-v4-flash | DeepSeek-V4-Flash-DSpark | NVFP4 (`nvfp4_ds_mla` KV) | vLLM+DSpark self-spec-decode TP=2 | 2 | **43.7** / 54.3 | 188 ms | 79 / 88 |
| mimo-v25 | MiMo-V2.5-NVFP4 | NVFP4 + NVFP4-KV | vLLM TP=2 (Ray) | 2 | **32.3** / 32.6 | 203 ms | 68 / 94 |
| worker omni | Nemotron-3-Nano-Omni-AEON | BF16 | vLLM (omni) | 1 (worker) | **27.3** / 27.4 | 185 ms | 114 / 114 |

Reads: single-node head-text is fastest (no host-staged cross-node all-reduce). DeepSeek's max 54.3
approaches the **[reported]** forum number ~56.7; its flat conc scaling is `max_num_seqs=6` (1M-ctx
profile). minimax scales best under load (AWQ + EP). deepseek/mimo have higher TTFT (spec-decode draft
/ 1M-ctx setup). The Nemotron omni is BF16 61.6 GiB and takes ~20 min to start (8 min load + 322 s
warmup + capture).

- **[reported]** **External corroboration (S-dgxspark-report):** a published DGX-Spark optimization
  writeup reports the same models in the same range — DeepSeek-V4-Flash-DSpark gen ~37 (c1) → 54 (c2)
  t/s (their "60–67" is the optimistic peak), MiMo-V2.5-NVFP4 gen ~34 (c1) → 63 (c2), Qwen3.5-122B-A10B-int4
  gen ~27/req @ c4/4k. Our numbers matching theirs is the evidence our fabric is *not* hit by the
  kernel-6.17 RoCE regression they flag (`[[wiki/multinode-tp-and-networking.md]]`). Their numbers are
  external, not independently reproduced here.

**4 sparkrun cross-node bugs fixed this run** (in `orchestration/infiniband.py` + `scripts/ray_head.sh`;
**revert on sparkrun upgrade**) — without them the TP=2 vLLM/Ray recipes hang at cross-node init:
1. vLLM ignores deprecated `HOST_IP` → set **`VLLM_HOST_IP=<fabric>`** (else `mq_connect_ip` = mgmt).
2. Forced global **`NCCL_IB_GID_INDEX=3`** → empty (per-NIC auto: idx3 NIC0 / idx5 NIC1).
3. **`NCCL_SOCKET_IFNAME`** listed mgmt NIC first → fabric NICs only.
4. **`ray_head.sh`** echoed the mgmt IP as `head_ip` (worker `ray start --address` → mgmt) → fabric.
See `[[wiki/multinode-tp-and-networking.md]]`.

## DiffusionGemma-26B-A4B — bf16 vs NVFP4 (2026-07-01, TP=2, `llama-benchy` pp512/tg128, 3 runs)

- **[proven]** First-party TP=2 GB10 measurements.

| Quant | prefill pp512 c1 | e2e TTFT | "decode" tg128 c1 | prefill c4 agg | "decode" c4 agg | Mem/node |
|---|---|---|---|---|---|---|
| NVFP4 (marlin MoE) | **287.5** t/s | ~1.8 s | 92.3 ± 29.8 | 282.5 | 262.3 | ~18 GB total |
| bf16 (triton MoE) | 159.2 t/s | ~3.4 s | 128.0 ± 0.0 | 251.1 | 223.0 | ~52 GB total |

> **[proven] Caveat — don't trust the decode column for block-diffusion.** DiffusionGemma denoises a
> whole 256-token canvas and emits in a burst, so `llama-benchy`'s per-token tg metric is an artifact
> (bf16's `128.0 ± 0.0` = it just returned the 128 requested tokens ~all at once; NVFP4's high
> variance is the same effect, noisier). The **trustworthy** comparisons are **prefill (NVFP4 ~1.8×
> faster)** and **memory (NVFP4 ~1/3)** — by those NVFP4 was the stronger serving option. We deploy
> **bf16** as the full-precision choice regardless (GB10 has no native FP4 → NVFP4 = weight-only
> marlin decompress). NVFP4 retired + weights deleted. See `[[wiki/models/diffusiongemma.md]]`.
> Bench needs `--skip-coherence` (short-probe empty-output quirk).

## Concurrency (where measured)

- **[proven]** **Holo NVFP4 text** (256 tok/req, util 0.60): 1→76, 8→295, 32→575, 64→758, **128→899
  tok/s (ceiling, 11.8×)**. Saturates ~96–128 (compute-bound Marlin FP4). Interactive op point 32–64.
- **[proven]** **Holo NVFP4 vision** (1280×800 screenshot/req, grounding): peaks ~1.69 steps/s @ 32;
  prefill-bound; interactive sweet spot **4–8 concurrent agents**.
- **[proven]** **Holo thinking ON vs OFF** (conc 8): 1.13 → 4.74 steps/s (**4.2×**), 190 → 14 output
  tok/step, no grounding loss. See `[[wiki/models/holo-3.1.md]]`.
- **[proven]** **MiniMax-M2.7-abliterated AWQ decode scaling** (tg128, EP=2): c1 **23.9** (per-req
  23.9), c2 35.6 (18.5/req), c4 53.6 (14.1/req), c8 **77.0** (10.2/req). With MTP/spec the peak is 25 /
  40 / 61 / 96. Knee ~c4. (NVFP4 c8 aggregate ~58, peak 77 — consistently below AWQ.)

> **[proven] Durable finding (S-minimax-sweeps):** for MiniMax-M2.7 on GB10, **AWQ-4bit decodes ~1.4×
> faster single-stream than NVFP4** (~24 vs ~16.5 tok/s) despite both being ~4-bit — the AWQ/Marlin
> path is more decode-efficient than NVFP4 FlashInfer-CUTLASS for this MoE here, while NVFP4 wins
> prefill. This is *why* the AWQ checkpoint is the durable default. Note this does **not** contradict
> "fewer bytes = faster" (`[[wiki/quantization-on-gb10.md]]`): AWQ and NVFP4 move ~equal weight bytes,
> so the delta is kernel efficiency, not bandwidth. Pick quant by *measuring decode on the actual
> model* — don't assume NVFP4 > AWQ. (Both EP=2 cross-node; ignore the bogus 303410 tok/s and 9.4 ms
> prefill rows = prefix-cache-inflated.)

## Method notes
Single-stream code/prose at temp 0 (first-party bench scripts; Atlas: llama-benchy `spark-arena-v2`,
prefill 2048 / depth 0 / concurrency 1). Vendor-claimed numbers tagged "(vendor)" — not independently
reproduced. `max_model_len` does **not** affect single-stream decode (only KV pool). First-shape
cudagraph capture inflates first-run TTFT — re-run warm.

## MiMo-V2.5-NVFP4 + DFlash spec-decode (2026-07-06, custom_class, TP=2, S-dflash-nvfp4)

- **[proven]** Decode tok/s single-stream, temp 0. Spec accept is **workload-dependent** (structured
  >> prose):

| workload | tok/s | accept tok/block | note |
|---|---|---|---|
| math | 38.5 | 3.29 | short ctx, num_spec=5 |
| json | 42.2 | 3.23 | short ctx, num_spec=5 |
| code | 29.9 | 1.87 | short ctx, num_spec=5 |
| prose | 23.7 | 1.31 | short ctx, num_spec=5 |
| prose @ 100k depth | 8.8 | 0.86 | llama-benchy pp256/tg128, War&Peace |
| prose @ 200k depth | 4.3 | 0.87 | target-attention-bound (O(ctx)) |

- **[proven]** **Tuning:** `num_speculative_tokens=5` beats 7 by +15-20% tok/s (drafter block_size=8;
  the accept profile decays so positions 6-7 add ~2% accept but cost 2 verify positions/step).
  Deep-context decode is target-bound — nvfp4 diffkv attention is ~2x slower than fp8 at 200k (cf
  HeNryous fp8 200k@10 vs our 4.3); that gap is KV-dtype/kernel, not the drafter.
- **[proven]** Deep context: coherent + correct needle recall verified at **18k/45k/89k** (real varied
  content). Full **1M** context fits at GMU 0.88 (2.1M-token pool); 1M *prompt* prefill is ~30 min
  (O(context)) — 256k+ is the practical range. Drafter is eager PyTorch (per-request loop) — tok/s has
  headroom vs a fused/cudagraph path.
- **[reported]** cf forum HeNryous fp8: 100K@12 tok/s (fp8 decodes a bit faster than nvfp4) — external
  forum number, origin HeNryous.

## Forum-reported benchmarks (2026-07-08 ingest)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Tagged `[forum]` to distinguish from the proven rows above.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| DeepSeek-V4-Flash (official) | FP8 (E4M3 128×128 block) | vLLM (mp, no-ray) | 2 (TP=2) | ~44 | 200K | MTP nst=2, block-size 256, cudagraph FULL_AND_PIECEWISE | S-forum-dsv4-flash ||
|| DeepSeek-V4-Flash-DSpark | NVFP4 (`nvfp4_ds_mla` KV) | vLLM+DSpark self-spec | 2 (TP=2) | ~60-67 (code) / ~40 (mixed) | 1M | c=6 ~182 tok/s agg; c=16@200K ~315 agg | S-forum-dsv4-dspark ||
|| GLM-5.2 (744B/40B MoE) | AWQ-INT4 + 15% expert prune | vLLM | 4 (TP=4) | ~22 | 256K | MTP gave biggest uplift; cudagraph only ~3% | S-forum-glm52-4x ||
|| MiMo-V2.5-NVFP4 | NVFP4 + fp8 KV | vLLM (Ray) | 2 (TP=2) | 36-42 | 131K | Q&A 37, code 40, JSON 42, math 35 (run 2) | S-forum-mimo-2x ||
|| MiMo-V2.5-Omni | NVFP4 | vLLM (Ray) | 3 (TP=3) | 38.8 (eff 35.1) | 1M | Virtual-head padding; thinking-OFF > ON | S-forum-mimo-3x ||
|| MiniMax-M3 426B | UD-IQ4_XS GGUF | llama.cpp RPC | 2 (layer-split) | ~10.7 | 65K | ~590 prefill; tool-calling via hybrid template | S-forum-m3-llamacpp-2x ||
|| MiniMax-M3-AWQ | AWQ 4-bit + fp8 KV | vLLM | 4 (TP=4) | ~30 | 262K | Adaptive reasoning | S-forum-m3-awq-4x ||
|| Qwen3.5-35B-A3B | MXFP4 (patched) | vLLM 0.17.0 (patched) | 2 (TP=2) | 70.68 | — | +65% over vanilla 42.85 | S-forum-mxfp4-patches ||
|| gpt-oss-120b | MXFP4 (patched) | vLLM 0.17.0 (patched) | 2 (TP=2) | 80.88 | — | +56% over vanilla 51.82 | S-forum-mxfp4-patches ||
|| Qwen3.5-122B-A10B | int4 | vLLM | 1 | up to 51 | — | eugr patches + quick-start | S-forum-qwen122 ||
|| Qwen3.5-122B-A10B | int4 + DFlash n=12 | vLLM 0.23 (patched) | 1 | ~81 (agent) / 59 (e2e) | — | block-spec decode, accept len ~8.3 | S-forum-dflash-qwen122 ||
|| Qwen3-Next-80B | native (Atlas) | Atlas | 1 | 82 | — | 2.8× vLLM, no spec decode, Rust+CUDA | S-forum-atlas ||
|| DeepSeek-V4-Flash Q2 | Q2 GGUF | ds4 (DwarfStar 4) | 1 | ~28 | — | custom CUDA engine, 81 GiB | S-forum-ds4-cuda ||
|| Hy3 (295B/21B MoE) | NVFP4-W4A16 | vLLM 0.23.1 (Ray) | 2 (TP=2) | 21.8 / 59.7 agg@c6 | 128K | MTP nst=1 (pos-2 only 20%), enforce-eager wins | S-forum-hy3 ||
|| GLM-5.2 NVFP4 | NVFP4 | vLLM | 4 (TP=4) | 24 | 128K | MTP4 fix: config plumbing bug, accept ~0.84 pos-4 | S-forum-glm52-mtp-fix ||
|| GLM-5.2 AWQ-INT4 (pruned) | AWQ-INT4 + 15% prune | vLLM | 4 (TP=4) | 22 | 256K | MTP biggest uplift, cudagraph ~3% | S-forum-glm52-4x ||
|| GLM-5.2 1-bit | UD-IQ1_S GGUF | llama.cpp RPC | 2 (layer) | 8 | 256K | toy experiment, 1-bit quant | S-forum-glm52-1bit ||
|| MiniMax-M3-AWQ | AWQ-INT4 + fp8 KV | vLLM (Ray) | 4 (TP=4) | 33 | 262K | EAGLE3, 5 GB10 build fixes (CUDA 13.0 mismatch) | S-forum-m3-awq-tp4 ||
|| MiniMax-M3-AWQ | AWQ-INT4 + nvfp4 KV | vLLM | 4 (TP=4) | 25 | 1M | nvfp4 KV inline-dequant fused, 1M pool | S-forum-m3-awq-1m ||
|| MiniMax-M3-MXFP4 | MXFP4 + bf16 KV | vLLM nightly | 4 (TP=4) | 35 | 262K | EAGLE3 k=2, no fp8 KV (crashes), ~70 tok/s@c5 | S-forum-m3-mxfp4-4x ||
|| MiniMax-M3-W4A16-GPTQ | W4A16 + nvfp4 KV | vLLM b12x | 2 (TP=2) | 33 (+vision) | 113K | OCR-grade multimodal, vision reproduced | S-forum-m3-vision-b12x ||
| Qwen3.6-27B (dense) | FP8 + MTP nst=3 | vLLM (spark-vllm-docker) | 1 | 15.2 | 32K | 7.8 baseline → 1.94× with MTP; bandwidth-bound ~10 tok/s ceiling | S-forum-qwen36-27b-fp8 ||
| Gemma-4-31B-IT | NVFP4 + MTP=7 (assistant drafter) | vLLM (eugr fork) | 2 (TP=2) | 24.78 (peak) / 14.1 (tg1024) | auto | fp8 KV, MTP=7 optimal for 31B, MTP=4 for 26B-A4B | S-forum-gemma4-assistant ||
| GLM-4.7 (355B full) | NVFP4 | vLLM (spark-vllm-docker) | 2 (TP=2) | 17.5 | 64K | NCCL_NET_GDR_LEVEL=0 mandatory; --no-ray; 4 documented walls | S-forum-glm47-full-2x ||
| DeepSeek-V4-Flash | FP8 + MTP | vLLM (jasl fork) | 4 (TP=4) | 49.4–54.4 (single) / 180 (n8 agg) | 384K | NCCL 2.30.4 critical (2.28.9 wedges); sm12x_deep_gemm_fallbacks.py | S-forum-ds4f-4x-vllm ||
| Nemotron-3-Super-120B | NVFP4 + MTP | SGLang 0.5.13-dev | 4 (TP=4) | 1.70× single-stream (over no-spec) | 524K | accept_len ≈2.7; 3/4 depth beats NVIDIA cookbook 5/5; Mamba state pool limits single-node | S-forum-nemotron-super-mtp ||
| Nemotron-3-Ultra-550B | NVFP4 (modelopt_mixed) | SGLang 0.5.12 | 4 (TP=4, EP=4) | 42–43 (n8 peak) / 5.3 (per-req n8) | 512K | 83.7 GB/GPU weights; LatentMoE 512+1 experts, Mamba2+MoE+attn hybrid | S-forum-nemotron-ultra-4x ||
| MiniMax-M2.5 | NVFP4 | SGLang | 4 (TP=4, EP=4) | 25.5 (single) / 124 (n8 agg) | — | MAX_JOBS=1 fixes CUTLASS MoE compile OOM; RDMA enabled | S-forum-m25-sglang-4x ||
| GLM-4.7-FP8 | FP8 | SGLang | 4 (TP=4) | 25.1 (RDMA) / 8.2 (socket) | — | 2.5× speedup just from RDMA enable; SGLang container needs --device=/dev/infiniband | S-forum-glm47-rdma ||
| GLM-5.2 (744B) | IQ4_XS GGUF | llama.cpp RPC | 4 | 6.28 (C=1) / 9.07 (C=4 agg) | 1M | DSA active; ngram self-spec →24 tok/s structured; Q4_K_S has more headroom | S-forum-glm52-iq4xs-4x ||
| Qwen3.5-397B-A17B | NVFP4 + MTP | SGLang 0.5.10 | 4 (TP=4) | 40.0 (n1, MTP) / 110.9 (n8) | 524K | MTP +86% single-stream; cutlass MoE + triton attn + fi_cutlass FP4 + CG on wins | S-forum-roce-397b-mtp ||
| Gemma-4-31B (dense) | BF16 + MTP (assistant) | SGLang 0.5.11 | 4 (TP=4) | 26.68 (n1, MTP=6) / 153.24 (n8) | 262K | FROZEN_KV_MTP drafter; +154% @ n1, +80% @ n8 over baseline | S-forum-gemma4-mtp-4x ||
| Qwen3.5-122B-A10B | NVFP4 (RedHatAI) | vLLM | 1 | 16.15 | 262K | Quality close to FP16; moe-backend flashinfer_cutlass | S-forum-qwen122-nvfp4-redhat ||
| Qwen3.5-122B-A10B | NVFP4 (community) | vLLM | 1 | — | — | 234GB→75.6GB, fits 128GB; DeltaNet+vision, routers/lm_head kept BF16 | S-forum-qwen122-nvfp4-quant ||
| Nemotron-3-Super-120B | NVFP4 | vLLM TP=2 | 2 | 24 | — | ABI fix for cu130/cu132 mismatch in Dockerfile (cu132 wheel + cu130 PyTorch) | S-forum-nemotron-super-abi ||
| DeepSeek-V4-Flash | hybrid 2-bit (IQ2_XXS+Q2_K+FP8) | vLLM (custom) | 1 | — | — | antirez MLX recipe ported; ~85 GiB, coherent output on single Spark | S-forum-ds4f-hybrid-1x ||
|| Step-3.7-Flash | IQ4_XS GGUF | llama.cpp (stepfun fork) | 1 | 31 (short ctx) / 11 (max ctx) | 262K | Only path for Step-3.7 on Spark; prefill poor vs vLLM | S-forum-step37-llamacpp ||

## Forum-reported benchmarks (2026-07-10 ingest)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Measured with `llama-benchy` (context-depth sweep tool, see
`[[wiki/containers-and-tooling.md]]`).

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| MiniMax-M2.1-AWQ-4bit | AWQ 4-bit | vLLM | 2 | ~36 (tg32, c1) | 100K | pp2048 ~3544 t/s; degrades with depth (pp @ 8K ~2832); measured via llama-benchy | S-forum-llama-benchy ||
|| GLM-4.7-Flash-AWQ-4bit | AWQ 4-bit | vLLM | 1 | ~41.75 (tg32, c1) | 202K | pp2048 ~5326 t/s c1; c2 tg32 agg 73.74 (37.38/req); c10 tg32 agg 87.65 (15.33/req); KV cache 1.24M tokens, util 0.7 | S-forum-llama-benchy ||
||| FLUX.2-dev (image gen) | NVFP4 W4A4 (torchao) | torchao/diffusers | 1 | — | — | 28 steps @ 1024²: ~45s NVFP4 vs ~2.3 min BF16 (~3×); ~66 GB vs ~112 GB VRAM; edit ~1m51s vs ~4m20s (~2.3×) | S-forum-flux2-nvfp4-compute ||

## Forum-reported benchmarks (2026-07-10 ingest, Batch 5)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||| 
||| MiMo-V2.5 (native FP8) | FP8 (310B) | SGLang | 4 (TP=4) | 31.5 | 256K | EAGLE disabled (OOM); TTFT 0.46s; fp8 KV; tool eval 89/100; vision+audio+video | S-forum-mimo-sglang-4x ||
||| MiMo-V2.5-NVFP4 (renek recipe) | NVFP4 + fp8 KV | vLLM (Ray) | 2 (TP=2) | 30-33 (single) / 57-63 agg@c3 | 160K | enforce-eager, MTP=2 (86%/45% accept), util=0.89, triton_attn_diffkv; 38 tok/s claimed by tonyd615 (non-eager) | S-forum-mimo-2x-opt ||
||| MiniMax-M2.7-NVFP4 | NVFP4 (FlashInfer-CUTLASS) | vLLM | 2 (TP=2) | 24.12 (tg128) | 225K | FlashInfer-CUTLASS + throughput backend; no-Ray slightly better; CUTLASS baseline ~22 | S-forum-m27-recipe ||
||| MiniMax-M2.7-AWQ-4bit | AWQ 4-bit | vLLM | 2 (TP=2) | 39.4 (tg128) / 41.6 (tg32) | 196K | Clear decode winner — ~1.5× NVFP4; 3 independent reporters agree | S-forum-m27-recipe ||
||| MiniMax-M2.7 (Unsloth FP8) | FP8 | vLLM | 4 (TP=4) | 36–37 | — | No degradation vs NVFP4, slight increase; cache hit 53.6 @ 2 concurrent | S-forum-m27-recipe ||

## Forum-reported benchmarks (2026-07-13 ingest, Batch 11)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Measured on a 4-node cluster via CRS504 switch (100G). `llama-benchy` c=1, pp2048.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| DeepSeek-V4-Flash | FP8 KV, MTP k=2 | vLLM (aidendle B12X) | 4 (TP=4) | 52.0–53.6 (TG) | 512K | PP 2236–2452; no decode loss vs 200G; 100G link sufficient | S-forum-4node-crs504 ||
||| DeepSeek-V4-Flash | FP8 KV, MTP k=2 | vLLM (aidendle B12X) | 2 (TP=2) | 29.9–36.8 (TG) | 256K | PP 1612–2025; measured pre-4-node baseline | S-forum-4node-crs504 ||
||| MiniMax-M3-AWQ | AWQ-INT4 + bf16 KV + EAGLE | vLLM TP=4 | 4 (TP=4) | 27.7–35.4 (TG c=1) | 262K | PP 1684–2211; mns=4; 4-node CRS504 switch | S-forum-4node-crs504 ||

> **[conjecture]** 4-node CRS504 (100G switch) vs direct 200G: PP loss only 5–10%, decode
> unchanged. Measured traffic ~13 Gb/s — well below 100G. TP=4 DSV4-Flash decode 52–53.6 tok/s
> (vs TP=2's 29.9–36.8) shows near-linear scaling from 2→4 nodes. M3-AWQ+EAGLE on 4-node
> (28–35 tok/s) is consistent with existing [reported] M3-AWQ TP=4 benchmarks (S-forum-m3-awq-tp4
> 33 tok/s, S-forum-m3-awq-4x ~30 tok/s). (S-forum-4node-crs504)

## Forum-reported benchmarks (2026-07-14 ingest, Batch 12)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Measured with `llama-benchy` (MTP2 + fp8 KV + prefix cache, C=1 × 3 runs).

|||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
||||---|---|---|---|---|---|---|---|||
|||| DeepSeek-V4-Flash | MXFP4 MoE, fp8 KV, MTP2 | TokenSpeed `sm12x-stable` | 2 (TP=2) | 30.3–33.3 (tg128 peak) | 131K | PP 1979–2062 @ 8K-32K depth (+10–14% vs vLLM); KV 1.90M tokens (+25%); tool 45/45, GSM8K 0.96 | S-forum-tokenspeed |||
|||| DeepSeek-V4-Flash | MXFP4 MoE, fp8 KV, MTP2 | vLLM (jasl fork) | 2 (TP=2) | 41.3–45.3 (tg128 peak) | 131K | PP 1737–1866 @ 8K-32K depth; KV 1.52M tokens; decode ~30% faster than TokenSpeed | S-forum-tokenspeed |||

> **[conjecture]** TokenSpeed `sm12x-stable` vs jasl/vllm fork on the same 2× Spark pair: TokenSpeed
> leads cold-context prefill by ~10–14% but decode is behind ~70–74%. The CUTLASS MoE backend that
> wins prefill has a weaker small-M decode GEMM; a hybrid CUTLASS-prefill + Triton-decode path is in
> progress. TokenSpeed also fits +25% more KV cache (1.90M vs 1.52M tokens) and has zero tool-calling
> HTTP 500s. Both are single-source (jasl). (S-forum-tokenspeed)

## Forum-reported benchmarks (2026-07-15 ingest, Batch 14)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Single-node DGX Spark, vLLM 0.24, MTP enabled.

||||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
|||||---|---|---|---|---|---|---|---|||
||||| Qwen3.6-35B-A3B (nvidia) | NVFP4 + MTP=2, fp8 KV | vLLM 0.24 | 1 | ~90 (c=1) / ~420 (c=16 agg) | 262K | nvidia/Qwen3.6-35B-A3B-NVFP4; Marlin MoE; baseline for Unsloth comparison | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (Unsloth) | NVFP4 + MTP=2, fp8 KV | vLLM 0.24 | 1 | ~75 (c=1) / ~410 (c=16 agg) | 262K | unsloth/Qwen3.6-35B-A3B-NVFP4; ~15% slower than nvidia; gap narrows at high concurrency | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (nvidia) | NVFP4 + MTP, fp8 KV | vLLM 0.24 | 1 | 103.4 avg (4 workloads) | — | hedelyuk.alexandr benchmark: short_ok 107.2, coding 112.3, reasoning 93.2, agentic 100.8 | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (Unsloth) | NVFP4 + MTP, fp8 KV | vLLM 0.24 | 1 | 87.6 avg (4 workloads) | — | hedelyuk.alexandr: short_ok 92.9, coding 95.7, reasoning 77.3, agentic 84.6; −15.2% avg | S-forum-unsloth-qwen36 |||
||||| Qwen3.6-35B-A3B (Unsloth-Fast) | NVFP4 + MTP, fp8 KV | vLLM 0.24 | 1 | 97 (tuned) / 75 (initial) | 262K | unsloth/Qwen3.6-35B-A3B-NVFP4-Fast; TheAwakenOne tuned recipe; still behind nvidia | S-forum-unsloth-qwen36 |||

> **[reported]** Unsloth NVFP4 is consistently ~15% slower than nvidia NVFP4 on GB10 across 3
> independent benchmarks (hedelyuk.alexandr, J-R, TheAwakenOne). The "2.5× faster" claim is B200-only.
> At c=16 the gap narrows to ~2% (420 vs 410 tok/s aggregate). (S-forum-unsloth-qwen36)

## Forum-reported benchmarks (2026-07-15 ingest, Batch 15)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| Qwen3.5-397B-A17B | FP8 | — | 8 (TP=8) | 31–35 | — | largest reported DGX Spark cluster; MoE gains flatten past TP=4; no config details provided | S-forum-qwen397-arch ||

> **[conjecture]** Single forum reference (raphael.amorim) citing the 8× GB10 cluster result.
> No engine, flags, or configuration details provided — treat as a data point only.

## Forum-reported benchmarks (2026-07-16 ingest, Batch 16)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
|||---|---|---|---|---|---|---|---||
||| GLM-5.2 (744B/40B MoE) | int4 MoE + int8 MTP | Colibri (expert streaming) | 1 | 2.39 (full top-8) / 3.33 (CACHE_ROUTE) | short | Experts streamed from disk; O_DIRECT 9.69 GB/s; 82-97% expert cache hit; RSS 76-78.5 GB; `COLI_CUDA_UNIFIED=1`; single-Spark 744B | S-forum-colibri-glm52 ||

> **[conjecture]** Colibri is the first reported engine to run a 744B MoE on a single 121 GB Spark,
> streaming experts from disk (only hot experts cached via LRU/pin). 2.4-3.3 tok/s is very slow but
> coherent — the streaming-from-disk approach is fundamentally different from multi-node TP. Attention
> dominates the profile (6.16s of 18s), not disk I/O. Experimental CACHE_ROUTE (cache-aware routing,
> ~14% expert substitution) raises hit 82→97% and tok/s 2.4→3.3; not upstream default. (S-forum-colibri-glm52)

## Forum-reported benchmarks (2026-07-17 ingest, Batch 18)

All rows below are **[conjecture]** — single-source community-reported numbers (two forum users in one
thread, not independent threads), not first-party.

||||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
|||||---|---|---|---|---|---|---|---|||
||||| GLM-5.2 (744B/40B MoE) | Int4-Int8 mix (QuantTrio GLM-5.2-Int4-Int8Mix) | vLLM v16-unified + b12x W4A8 | 8 (TP=8, DCP=1) | 33–54 avg (33–39 prose, 40–55 code, peak 54.5–58) | 200K | ~1,200 t/s prefill (1,000→1,200 vs older branch); TTFR 1.7s→198s @ 0–200K depth; 2 conc → 50 (prose) / 60–70 (code); Snake game 54.16 tok/s; tool-eval 91/100 | S-forum-glm52-8x |||
||||| GLM-5.2 (744B/40B MoE) | Int4-Int8 mix | vLLM v16-unified | 8 (TP=4+PP2, experimental) | ~12 | — | ~1,800 t/s prefill but MTP acceptance collapses to ~8% → decode drops to ~12 t/s; NOT viable | S-forum-glm52-8x |||
||||| GLM-5.2 (744B/40B MoE) | Int4-Int8 mix | vLLM v16-unified (decode-aware scheduler) | 8 (TP=8, DCP=4, MTP3) | ~2.74 (under prefill pressure) / 26 (post-prefill) | 320K (×10 seq) | DCP4 gives 3.2M KV tokens; decode starves to ~0.0–0.2 tok/s during prefill; scheduler patch ENABLE_DECODE_AWARE_PREFILL=1 limits stalls to ~1.6s | S-forum-glm52-8x |||

> **[conjecture]** **GLM-5.2-Int4-Int8Mix on 8× GB10 is the largest reported DGX Spark cluster run**
> (S-forum-glm52-8x, ciprianveg + penguinchang, single thread). Key numbers from the OP benchmark
> (llama-benchy, tg=1500, single stream):
> - **Prefill stays >1,000 t/s all the way to 200K context** (1,211 at depth 0, 1,019 at 200K).
>   The v16-unified branch (local-inference-lab/vllm @ 5dffea8, branch
>   `codex/fathomless-firmament-v16-unified-20260712`) is the single biggest prefill lever — older
>   branch capped ~1,000, v16 climbs to ~1,200.
> - **Single-stream avg decode: 33–39 t/s on coherent prose (stable across 0–200K context),
>   40–55 t/s on coding/structured** (peak 53.5–58 t/s). Two concurrent: ~50 (prose) / 60–70 (code).
>   Snake game generation (temp=0, thinking=off): 54.16 tok/s.
> - **Stack:** vLLM v16-unified fork + b12x W4A8 MoE (lukealonso/b12x @ 97b3d64, unified SM120
>   sparse MLA + PCIe DCP collectives) + DCP1 patches from CosmicRaisins/glm-5.2-gb10. CUDA 13.2.0,
>   PyTorch 2.11.0, NCCL 2.30.4 (custom aarch64), transformers ≥5.0 (--tf5 build flag), prebuilt
>   sm_121 FlashInfer wheels.
> - **DCP1 knobs (3 vs CosmicRaisins base):** `VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1` (keeps B12X
>   indexer path consistent), `draft_tensor_parallel_size=1` (unsharded drafter avoids TP8
>   collectives on every draft step), `NCCL_BUFFSIZE=16777216` (16 MB, up from 8 MB — bigger
>   NCCL buffer for gen speed at high context; 8 MB starts bottlenecking allreduce on long-context
>   decode).
> - **TP4+PP2 is not viable for MTP:** prefill climbs to ~1,800 t/s but MTP acceptance collapses to
>   ~8% on the pipeline split → decode drops to ~12 t/s. Staying on TP8+PP1 for production. (OP)
> - **DCP4 decode starvation (penguinchang):** on TP8 DCP4 with MTP3, concurrent prefill requests
>   starve decode to ~0.0–0.2 tok/s until prefill completes. A custom "decode-aware prefill"
>   scheduler (ENABLE_DECODE_AWARE_PREFILL=1, DECODE_PREFILL_TOKEN_BUDGET=1024,
>   IDLE_PREFILL_TOKEN_BUDGET=16384, MAX_LONG_PREFILLS_PER_STEP=1) limits stalls to ~1.6s max
>   but decode still falls to ~2.74 tok/s under pressure. DCP1 would get ~30% faster prefill and
>   ~60% faster gen per the OP, but DCP4 enables 320K×10 context (3.2M KV tokens).
> - **Four production patches:** 01 (DCP config → draft model, prevents MTP collapse under DCP>1),
>   03 (quantized NextN draft token mapping — without it quantized drafts silently build
>   unquantized and MTP acceptance collapses), 04 (DeepSeekMTP SupportsPP + stale topk_indices_buffer
>   in flashinfer SM120 sparse MLA PR#46994 + MTP embed_tokens loading under PP), 06 (b12x stale
>   topk buffer PR#46994 Fix #4 — without it _maybe_share_lm_head swaps the indexer's buffer but the
>   backend keeps a stale ref → garbage DSA attention and ~30% acceptance instead of ~85%).
>
> This corroborates: (1) NCCL 2.30.4 mandatory on multi-node (S-forum-ds4f-4x-vllm, S-forum-tokenspeed),
> (2) MoE gains flatten / interconnect-bound at scale (S-forum-qwen397-arch), (3) pipeline parallelism
> is latency-sensitive and can wreck MTP on Spark (S-forum-2d-parallel). All numbers are from one
> thread with two active users — `[conjecture]`, not `[reported]` (the two posters are in the same
> thread and not independent).

## Forum-reported benchmarks (2026-07-16 ingest, Batch 17)

All rows below are **[reported]** — community-reported numbers from the NVIDIA DGX Spark forums, not
first-party. Numbers from a meta-analysis thread (S-forum-nvfp4-broken) citing multiple independent
forum sources, plus an abliterated model variant (S-forum-dsv4-abliterated).

||||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source |||
|||||---|---|---|---|---|---|---|---|||
||||| Llama-3.3-70B-Instruct-NVFP4 | NVFP4 | TensorRT-LLM | 1 | 5 (reporter 1) / 2.5 (reporter 2) | — | NVIDIA's own NVFP4 model on NVIDIA's TRT-LLM; slower than GGUF Q4_K_M via LM Studio (4.6-4.9 tok/s) on same Spark | S-forum-nvfp4-broken |||
||||| Llama-3.3-70B-Instruct | GGUF Q4_K_M | LM Studio (llama.cpp) | 1 | 4.6–4.9 | — | Non-NVIDIA quant on non-NVIDIA tooling beats NVIDIA NVFP4 on TRT-LLM | S-forum-nvfp4-broken |||
||||| Nemotron-3-Super-120B-A12B | NVFP4 | vLLM | 1 | 19–22 | — | 42-48% of theoretical bandwidth ceiling (~45 tok/s); 12B active @ 0.5 bytes = ~6 GB/token vs 273 GB/s | S-forum-nvfp4-broken |||
||||| Nemotron-3-Super-120B-A12B | NVFP4 | vLLM | 2 (TP=2) | 24 | — | ~200 GB/s cluster bandwidth; 71% of ~34 tok/s theoretical at that BW | S-forum-nvfp4-broken |||
||||| DeepSeek-V4-Flash-DSpark-Abliterated | NVFP4 (DSpark recipe) | vLLM (DSpark) | 2 (TP=2) | 50–60 | — | Abliterated (uncensored) variant; fork of DS4 DSpark recipe with model swapped in | S-forum-dsv4-abliterated |||

> **[reported]** The NVFP4 meta-analysis (S-forum-nvfp4-broken) aggregates multiple independent forum
> measurements confirming that NVFP4 on GB10 achieves only 42–48% of the bandwidth-limited theoretical
> ceiling — a software/kernel gap, not a hardware limitation. TRT-LLM NVFP4 (NVIDIA's own stack) is
> slower than GGUF Q4_K_M (community tooling) for the same 70B model. Multiple sources agree.

## Forum-reported benchmarks (2026-07-20 ingest, Batch 24)

All rows below are **[conjecture]** — single-source community-reported numbers, not first-party.

|||| Model | Quant | Engine | Nodes | Decode tok/s | Ctx | Notes | Source ||
||---|---|---|---|---|---|---|---||
|| GLM-5.2 (744B/40B MoE) | (unspecified, likely Int4-family) | vLLM + b12x | 6 (TP=6) | ~30 (single-stream) | — | 6× GB10 via MikroTik CRS812 (768 GB); b12x enables non-power-of-2 TP; cluster 800-1180 W peak; no config/YAML shared | S-forum-6x-cluster ||

> **[conjecture]** GLM-5.2 at TP=6 (~30 tok/s) sits between the TP=4 (~22-24 tok/s,
> S-forum-glm52-4x / S-forum-glm52-mtp-fix) and TP=8 (33-54 tok/s, S-forum-glm52-8x) results —
> consistent with sublinear scaling as interconnect overhead grows with node count. Quant format
> unspecified by the poster; "b12x" usage and ~30 tok/s range are consistent with the Int4-Int8 mix
> used in the 8× run. Single source (mclenithan), no benchmarking methodology described.

## Image generation benchmarks (diffusion models on GB10, 2026-07-10)

**[conjecture]** — single-source forum benchmarks for diffusion models on DGX Spark, all 1024×1024,
single-node, via `diffusers` library (not ComfyUI). Generation time post-compile (torch compile needs
~2 warmup runs). See `[[wiki/quantization-on-gb10.md]]` for NVFP4 W4A4 FLUX.2 details.

||| Model | Steps | Time (BF16) | Time (NVFP4) | Notes | Source ||
|||---|---|---|---|---|---|||
||| FLUX.2-klein-9B | 4 | 4.4s | 3.3s | distilled; NVFP4 via torchao W4A4 | S-forum-diffusion-speeds ||
||| Z-Image-Turbo | 9 | 7.2s | 5.6s | BF16 mostly; torch compile 8.1s (default attn) | S-forum-diffusion-speeds ||
||| ERNIE-Image-Turbo | 8 | 8.8s (BF16) / 11.2s (orig) | 6.4s | `DIFFUSERS_ATTN_BACKEND=_native_cudnn` improved 11.2→8.8 | S-forum-diffusion-speeds ||
||| SDXL 1.0 | 30 | 11.3s | — | BF16; no NVFP4 tested | S-forum-diffusion-speeds ||
||| Krea2-Turbo | 8 | 13.9s (BF16+optimized) | 12.4s | Default 39.3s; `torch.set_float32_matmul_precision('high')` → 32s; `DIFFUSERS_ATTN_BACKEND=_native_cudnn` → 13.9s | S-forum-diffusion-speeds ||
||| Qwen-Image-2512 | 50 | 61.0s | — | ~50% memory used; no NVFP4 tested yet | S-forum-diffusion-speeds ||

- **[conjecture]** **`DIFFUSERS_ATTN_BACKEND=_native_cudnn`** is a significant GB10 diffusion speedup
  env var (S-forum-diffusion-speeds, ijontichy): improved Krea2-Turbo 39.3→13.9s, ERNIE-Image-Turbo
  11.2→8.8s, no effect on Z-Image-Turbo. Combined with `torch.set_float32_matmul_precision('high')`.
  Second source (vasimv) reports 15-17s for Krea2-Turbo FP16 on ComfyUI with 610 drivers + CUDA 13.3.
- **[conjecture]** **NVFP4 W4A4 quantization gives modest speedups on diffusion models** (S-forum-diffusion-speeds):
  FLUX.2-klein 4.4→3.3s (~1.3×), Z-Image-Turbo 7.2→5.6s (~1.3×), ERNIE 8.8→6.4s (~1.4×), Krea2 13.9→12.4s
  (~1.1×, marginal). Smaller gains than the ~3× seen on FLUX.2-dev (which uses activation-quantized
  real FP4 compute via Triton — see `[[wiki/quantization-on-gb10.md]]`). The difference: these are
  weight-only NVFP4 vs FLUX.2-dev's activation-quantized path.
