# MiniMax (M2.x AWQ, M3)

> **area:** model
> **status:** stable
> **evidence:** proven
> **sources:** S-m3-vision, S-m3-20tps, S-sess-jun11, S-minimax-sweeps, S-forum-m3-nvfp4-4x, S-forum-m3-awq-4x, S-forum-m3-llamacpp-2x, S-forum-m3-quad, S-forum-m3-w4a16-gptq, S-forum-m25-sglang-4x, S-forum-m27-recipe, S-forum-4node-crs504
> **updated:** 2026-07-13

Two very different MiniMax stories on GB10: **M2.7 AWQ** = the fast, durable daily-driver default;
**M3** = a 428B research/long-context/vision endpoint that's structurally slow here.

## MiniMax-M2.7 (AWQ) — the durable default

- **[proven]** The daily driver: **~24 tok/s** decode, served TP=2 (EP=2 expert-parallel) over the
  fabric. AWQ 4-bit → Marlin, stable.
- The durable default is deployed as a swapper stack `minimax-m2.7-ablit`
  (`lhca521/MiniMax-M2.7-abliterated-heretic-ara-AWQ`), with a real-inference watchdog (a worker-reboot
  orphan answers `/v1/models` but hangs real TP=2 inference — the watchdog probes actual completion).
- **[proven]** **AWQ beats NVFP4 on decode here (measured, S-minimax-sweeps).** Single-stream tg128: AWQ
  **23.9 tok/s** (peak 25 w/ MTP) vs NVFP4 FlashInfer-CUTLASS **16.5** (peak 18) — ~1.4× — even though
  both are ~4-bit. NVFP4 wins *prefill* (pp2048 ~1320–1500 vs ~1100). Concurrency (AWQ, EP=2): c1 24 →
  c4 54 → c8 77 aggregate (peak 96 w/ MTP), knee ~c4. **This is why AWQ is the default** — the delta is
  kernel efficiency (AWQ/Marlin vs NVFP4 FIC for this MoE), not bytes. Don't assume NVFP4 > AWQ; measure.
  See `[[wiki/benchmarks.md]]`. (cyankiwi AWQ ≈ abliterated AWQ; an NVFP4 FlashInfer-CUTLASS recipe exists
  for the NVFP4 path.)

## MiniMax-M3 — 428B MoE + MSA + vision, cross-node

**[proven]** Community AutoRound-mixed quant `aquaman164/MiniMax-M3-AutoRound-3.2bit-longctx` (188.6 GB)
served **cross-node TP=2 with MSA active and working vision** — the quant author had only run it
single-node, text-only. Footprint 88.4 GiB/node; MSA + 4 KV heads make KV very cheap (40k ctx easily).

**[proven]** **M3 is *the* cudagraph-wall model.** Its fused `fused_allreduce_gemma_rms_norm` +
auto-enabled breakable_cudagraph is the specific path that crashes cross-node capture (vllm#46253) —
unlike the live MiMo-V2.5, which captures cross-node fine. Upstream **PR #46372** targets exactly this; a
GB10 test is staged (`[[wiki/cudagraphs-and-compile.md]]` → Upstream status).

**[proven]** **Why it's ~5 tok/s bare (structural):** 427B MoE needs both nodes → cross-node TP=2
mandatory → eager-forced (`[[wiki/cudagraphs-and-compile.md]]` both walls) + ~120 host-bounced
all-reduces/token. The June "get to 20 tok/s" mission concluded **not reachable**: cudagraphs walled, MTP
weights absent (`num_mtp_modules=0`), PP unsupported by the model class *on dev197*.

### EAGLE3 spec decode — the fix that landed (2026-07-03)
**`Inferact/MiniMax-M3-EAGLE3`** (1-layer dense-Llama draft, 3.3B, hidden 6144 / vocab 200064,
trained on M3-regenerated data, benchmarked upstream WITH `--enforce-eager`) works on our AutoRound
quant + shim on **dev537** (`SupportsEagle3` on both M3 classes): draft shares target embeddings,
keeps own lm_head. **[reported]** Same mechanism as DSpark's ~35 tok/s: **spec decode amortizes the
host-staged all-reduce storm across accepted tokens.**
- **[proven]** **Final measured (TP=2 EP=2 eager, vision ON, nst=3, mnbt=4096): 13.72 ± 0.63 tok/s tg128
  prose (peak 20.0), ~15.1 tok/s code-gen (timed 500-tok), prefill 310 tok/s pp2048** — **2.7× the bare
  5 tok/s**, vision fully working. Acceptance: ~2.4–2.8 prose, ~3.1–3.2 code (nst=3).
- **[proven]** **Tuning A/B (measured, don't redo):** `nst=5 + draft_tensor_parallel_size=1` is a
  **regression** — prose 9.56, code 12.88. Prose accepts ~0 at positions 4–5 (p4 0.17/p5 0.11) so the
  extra drafts are waste, and draft_tp=1 idles rank1 during drafting + adds a sync per cycle, eating the
  code-side acceptance win (4.6–4.8/5 accepted!). `max_num_batched_tokens: 4096` (vs 2048) is a genuine
  win (part of 11.98→13.72). nst=3 + draft on both ranks is the optimum found.
- **[proven]** Memory: model+draft+ViT = 90.68 GiB/node loaded; **`gpu_memory_utilization` 0.85 → 0.88
  required** (0.85 leaves 2.83 GiB KV < 3.53 needed @ 40k ctx). KV 57,472 tokens @ 0.88.
- **Deployed as a swapper stack `minimax-m3-eagle3`**, registered on the serving supervisor. After any
  weight re-download run the idempotent config-surgery script on both nodes.

### fp8 KV cache — 102k context at zero speed cost (2026-07-03)
**[proven]** `--kv-cache-dtype fp8` works with MSA/TRITON_ATTN/EAGLE3 (90→52 KB/token): with
`--limit-mm-per-prompt '{"image":4,"video":0}'` (the 102k **video** profiling alone eats ~6 GiB) and
util 0.90, the pool is **243,712 tokens / 13.3 GiB**, `max_model_len` **102400**. Measured: prose
13.27±0.76 (peak 20.0) / code 15.34 / prefill 310 — identical to bf16-KV within noise; **68k-token
needle recall PASS** (prefill ~340 tok/s at 68k depth). This is the deployed config. **[conjecture]** Next
context step: nvfp4 KV kernels (forum 375372, `tonyd2wild/MiniMax-M3-AWQ-1M-NVFP4-KV-4x-DGX-Spark`) —
would give ~300k+ here.

### Serving-layer gotchas (2026-07-03, all measured)
- **[proven]** **Tool calling:** dev537's `minimax_m3` tool parser needs the missing
  `vllm._rust_tool_parser` PyO3 ext (not in aarch64 nightlies); `minimax_m2` parses invoke-names but
  returns **empty args** (M3 args are nested XML). Fix = our tool-parser plugin
  (`m3_tool_parser_plugin.py`, via `--tool-parser-plugin` + `--tool-call-parser minimax_m3_xml`),
  streaming + non-streaming verified.
- **[proven]** **Reasoning parser: use OUR `m3_mmthink`** (in the same `m3_tool_parser_plugin.py`,
  text-based, fuzz-tested). The BUILT-IN `minimax_m3` parser leaks `<mm:think>` AND costs ~30%
  throughput — ours strips cleanly into the `reasoning` field at no measured cost (12.6 prose w/ parser
  vs 12.3-13.3 without). Needed for agent clients (Hermes) that can't strip think tags. Plugin must load
  on ALL ranks: the shim imports it + registers `sys.modules["m3_parsers"]` (headless workers validate
  the parser name at config time; the `--*-parser-plugin` flags are API-server-only).
- **[proven]** **`--enable-auto-tool-choice` shifts generation** (template renders tool instructions), so
  temp-0 outputs/timings differ vs no-tools configs — compare benchmarks like-for-like.
- **[proven]** **Launcher rank-fallback trap:** on a worker-ssh blip the multi-node launcher can land
  **rank 1 on the head** (both `node_*` containers on one node) → two 90 GiB loads on one GPU → bogus
  KV-fit failures / EngineDead. After ANY launch anomaly verify placement: head has `node_0` only, worker
  has `node_1` (`docker ps` on both; no `--node-rank 1` process on head).

### PP=2 — landed upstream but a trap here (2026-07-03)
dev537's M3 classes have `SupportsPP` (dev197 didn't). **[proven] But**: (1) **PP2+EAGLE3 impossible** —
the *draft* (`Eagle3LlamaForCausalLM`) lacks `SupportsPP`, `verify_with_parallel_config` raises; EAGLE3's
aux-hidden taps (layers 2/30/57) would straddle the stage split anyway. (2) **PP2 layer-split is
byte-UNEVEN for this mixed-bit quant**: head stage loaded 69.4 GiB → worker stage ~119 GiB **→
unified-mem OOM, node unreachable, hard power-cycle** (the llama-cpp-rpc page's OOM warning applies
to vLLM PP too). If PP2 is ever retried: set `VLLM_PP_LAYER_PARTITION` to balance *bytes*, not layers.

Treat M3 as: a real daily-usable long-context multimodal endpoint — ~14 tok/s prose / ~15 code
(peaks 20) with EAGLE3, no longer a 5 tok/s research curiosity.

### Config surgery to load the AutoRound-mixed quant
- **[proven]** **`Unsupported weight_bits: 16`** → register OneCompression `autoround_mixed` quant method
  and rename `quant_method` → `autoround_mixed` in `config.json` (16-bit base w/ per-module low-bit
  overrides).
- **[proven]** **`Unsupported activation: silu`** → bake missing attrs into `config.json` **top-level AND
  under `text_config`**: `hidden_act=swigluoai`, `n_shared_experts=1` (critical — else shared expert
  silently dropped → garbage), `use_gemma_norm=true`, `rope_theta=5e6`, etc.
- Plugin loaded via a `vllm.general_plugins` entry-point shim (no vLLM fork).

### Engine args (GB10 specifics)
`--tensor-parallel-size 2 --enable-expert-parallel --quantization autoround_mixed --block-size 128
--attention-backend TRITON_ATTN --enforce-eager --disable-custom-all-reduce --dtype bfloat16
--kv-cache-dtype auto --gpu-memory-utilization 0.85 --max-model-len 40960`, native multi-node (no
Ray). **[proven] `--block-size 128` + TRITON_ATTN are mandatory** for MSA (FLASHINFER has no common
block size → KV init fails). See `[[wiki/attention-and-kv-cache.md]]`.

### Vision (the part that wasn't supposed to work)
**[proven]** Use the full `MiniMaxM3SparseForConditionalGeneration` arch (not the text-only
`ForCausalLM`). The checkpoint ships the vision tower in **both** namings — pass the vLLM-named copy
straight through `hf_to_vllm_mapper` (vision loads nearly free, only 3.37 GB). Fix the **bare top-level
`lm_head.weight`** → `language_model.lm_head.weight` (miss it = silent random logits). First image =
~20 s ViT JIT.

### dev197 blockers fixed (also generic)
- **[proven]** **SemLock spawn race** (`context.Lock()` pickled to spawned worker → `SemLock._rebuild`
  FileNotFoundError → NCCL rendezvous hang): override mp-context `.Lock()` to a picklable no-op.
  Persistent, not "transient, just relaunch."
- **[proven]** **flashinfer gemma-rmsnorm CUTLASS-DSL ICE** on cu130 → swap pure-torch norms. (Text-path,
  not vision.) See `[[wiki/attention-and-kv-cache.md]]`.

## Result (see log 2026-07-03 for the EAGLE3/nvfp4/cudagraph era)
M3 text: coherent w/ `<mm:think>` reasoning. Vision: correctly read shapes/colors/positions/text from
a test image. ~5 tok/s decode, ~74 tok/s prefill, 40k ctx.

## M3-W4A16-GPTQ via the a3refaat b12x stack — 36 tok/s (2026-07-05, forum 375595)

**[proven]** The community **b12x** backend (a3refaat/spark-vllm-docker `minimax-m3-4bit-w4a16` +
`Sebesky/MiniMax-M3-W4A16-GPTQ` + `Sebesky/MiniMax-M3-EAGLE3-RTN-INT4`) reproduces the forum's
**36.25 tok/s tg32 (peak 37.5), 34.7 tg128, 1028 pp2048** on our 2× GB10 — **exactly** matching their
35.5. **VISION WORKS** (2026-07-05, verified): drop `--language-model-only` and the full 32-layer ViT
loads + runs on b12x — read a synthetic image (shapes/colors/text) AND a text-heavy report image
(**exact OCR**: title, three `$1,240,000`-style values, footer) perfectly, coherent & consistent.
**Text tg32 = 32.9 with the vision tower loaded** (vs 36 text-only — only ~3 t/s cost), KV pool
**113,152 tokens** @ util 0.92 headless + `--limit-mm-per-prompt '{"image":4,"video":0}'` (video
profiling eats ~6 GB — same fix as our EAGLE3 stack; note recipe templating needs `{{...}}` double
braces). This **beats our own `minimax-m3-eagle3` vision stack** (12.6/15.5 tok/s) by ~2.2× on speed;
tradeoff is context (b12x-vision ~113k vs our 262k). Both are viable; b12x-vision = fast, ours = long-ctx.

- **Deploy:** image `vllm-node-minimax-m3-b12x` (build via `spark-vllm-b12x/build-deploy.sh -c
  <worker>`, from-source vLLM @ pinned commit + fused-fp8-kv patch + `--build-rust`; the pinned commit
  isn't a branch tip → Dockerfile needs a `git fetch origin <ref>` fallback). Launch via the b12x
  `run-recipe.sh --no-ray` against the W4A16-GPTQ + nvfp4-EAGLE3 recipe. b12x attention backend, nvfp4
  target + nvfp4 DRAFT KV, cudagraph `FULL_DECODE_ONLY`, marlin int4 MoE.
- **[proven]** **The two things that make-or-break it** (both software, per
  `[[wiki/platform-gb10.md]]` parity tenet):
  1. **Warm the Triton cache on BOTH nodes.** The b12x prewarm mod (`fix-minimax-m3-b12x-triton-prewarm`)
     silently no-ops here (warms different shapes than inference hits; its except is swallowed unless
     `B12X_TRITON_PREWARM_STRICT=1`), so spec/decode kernels (`eagle_prepare_next`, `_nvfp4_write`,
     `update_regular_decode`, `rejection_sample`) JIT during cudagraph *replay*. On our no-GPUDirect
     cross-node setup the worker cache started nearly empty (88K vs head 6.7M) → worker JITs while head
     cache-hits → **rank desync → shm_broadcast deadlock → EngineDead / garbage output.** Fix:
     `rsync -a ~/.triton/ <worker>:~/.triton/` (identical sm_121 → portable kernels; chown via a busybox
     container first if perms block). `/root/.triton` is a host mount, so it persists across restarts.
     Once both caches are warm the deadlock is gone (one slow ~130s first-request JIT, then steady).
  2. **Healthy GPU clocks.** The wedge (611 MHz) caps it at 20 tok/s; cleared (2405 MHz) → 36. The wedge
     was the ENTIRE 20→36 gap — see the wedge section on `[[wiki/platform-gb10.md]]`.
- **[proven]** **Memory on a desktop head:** their util 0.93 / 196k ctx needs headless (desktop eats
  ~10 GB → free-mem check fails). We ran util 0.91 / ctx 32k (desktop back after reboot). Persistent
  headless (`systemctl set-default multi-user.target`) restores util 0.93 / 196k. The
  `fix-prometheus-fastapi-routing` mod must be made skip-graceful (version differs → hard-fails the whole
  launch otherwise).

## See also
`[[wiki/cudagraphs-and-compile.md]]` · `[[wiki/attention-and-kv-cache.md]]` · `[[wiki/benchmarks.md]]` · `[[wiki/platform-gb10.md]]`

## Forum ingest: 4× Spark recipes, llama.cpp RPC, AWQ-INT4 (2026-07-08)

- **[conjecture]** **MiniMax-M3-NVFP4 on 4× DGX Spark** (S-forum-m3-nvfp4-4x, OllieJW): uses the
  chthonic vLLM base (`vllm-m3-chthonic`), which bundles ModelOpt NVFP4 runtime + b12x sparse
  attention + b12x NVFP4 MoE path. Requires NCCL 2.30.7 forced via
  `VLLM_NCCL_SO_PATH=/opt/nccl230/build/lib/libnccl.so.2` + `LD_PRELOAD` (default NCCL in chthonic
  doesn't work for multi-node TP). Launch: `--no-ray` (PyTorch distributed, not Ray — chthonic build
  doesn't init M3 parsers cleanly with Ray). Key flags: `--quantization modelopt_fp4
  --attention-backend B12X_ATTN --moe-backend b12x -cc.mode=VLLM_COMPILE
  -cc.cudagraph_mode=FULL` (slightly better per-req throughput than PIECEWISE, longer warmup),
  `--max-model-len 524288`. **[conjecture]** Forum reports ~9-10 tok/s TG without a drafter on TP=4;
  community consensus is M3 is too large even on TP=4 for good throughput without spec decode.
- **[conjecture]** **MiniMax-M3-AWQ on 4× GB10** (S-forum-m3-awq-4x, Sebesky): fp8 KV, 262k context,
  adaptive reasoning, ~30 tok/s. This is the AWQ counterpart to the NVFP4 4× recipe — AWQ Marlin
  may be more decode-efficient (consistent with the M2.7 AWQ-vs-NVFP4 finding).
- **[conjecture]** **MiniMax-M3 426B via llama.cpp RPC on 2 nodes** (S-forum-m3-llamacpp-2x,
  karol.spark): UD-IQ4_XS GGUF (~194 GiB, ~97 GiB/node), `--split-mode layer`, **~10.7 tok/s**
  decode, ~590 tok/s prefill, 65k context (configurable; KV q8_0 ≈ 45 KB/token). Tool-calling works
  via a **hybrid chat template** — M3 native body + M2 tool-call format (llama.cpp PR #24523's
  tool-call parser can't read M3's native format; M2's template parses but corrupts M3's generation).
  First load ~13–25 min (RPC streams worker's layers; cached after). This is the llama.cpp
  alternative for users who can't run vLLM on 2 nodes. See `[[wiki/llama-cpp-rpc.md]]`.
- **[conjecture]** **MSA architecture overview** (S-forum-m3-quad, eh17): MiniMax Sparse Attention
  preserves raw uncompressed KV (not lossy compression like DeepSeek MLA), partitions KV-cache into
  fixed blocks, uses lightweight Top-K router — cuts compute to 1/20th. Smart KV-Block-Major memory
  layout optimizes SRAM locality. Claims 9.7× prefill speedup, 15.6× decode speedup vs dense
  attention. (Marketing/spec claims, not independently verified.)

### Batch 3 forum ingest (2026-07-09)

- **[reported]** **MiniMax-M3-W4A16-GPTQ 2×GB10 deployment** (S-forum-m3-w4a16-gptq, a3refaat):
  community GPTQ checkpoint (`Sebesky/MiniMax-M3-W4A16-GPTQ`) + EAGLE3 draft
  (`Sebesky/MiniMax-M3-EAGLE3-RTN-INT4`) + b12x + vllm. **36 tok/s** on 2× GB10. KVarN KV-cache
  quantization integrated for extended context (262K+ with KVarN, up to 370K observed). fp8 and nvfp4
  KV variants also tested. **4× Spark variant**: dropping KVarN and using NVFP4 quant gives flat 40
  tok/s decode throughout ctx depth, ~1200 tok/s prefill (S-forum-m3-w4a16-gptq, CosmicRaisins
  reply). Calibration dataset included simulated agentic trajectories. Vision not tested but should
  work in theory. Note: all recipes push the absolute limit of system memory — run headless and clear
  page caches before launch. This **corroborates** the first-party b12x reproduction (36.25 tok/s,
  S-forum-m3-vision-b12x) — now [reported] with two independent sources.
- **[conjecture]** **MiniMax-M2.5-NVFP4 on 4× Spark SGLang** (S-forum-m25-sglang-4x, Verel-lab):
  25.5 tok/s single-stream, 124 tok/s aggregate @ n8 (vs 70.7 in the published forum number). TP=4,
  EP=4, RDMA enabled, `lukealonso/MiniMax-M2.5-NVFP4` (~126 GB, ~32 GB/Spark for weights). For
  agentic workloads with many parallel agent threads, M2.5 outperforms GLM-4.7-FP8 and
  Qwen3.5-397B-A17B-NVFP4 on the same cluster at concurrency. CUTLASS MoE compile OOM fix:
  `MAX_JOBS=1 NVCC_THREADS=1` (see `[[wiki/multinode-tp-and-networking.md]]`).

## Forum ingest: M2.7 NVFP4/AWQ/FP8 recipes on 2×/4× Spark (2026-07-10)

- **[reported]** **MiniMax-M2.7-NVFP4 on 2× Spark (FlashInfer-CUTLASS)** (S-forum-m27-recipe,
  serapis + ekkis): `lukealonso/MiniMax-M2.7-NVFP4`, vLLM via eugr spark-vllm-docker TF5, TP=2, Ray.
  FlashInfer-CUTLASS backend beats CUTLASS on both latency and throughput: best config
  `VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass`, `VLLM_USE_FLASHINFER_MOE_FP4=1`,
  `VLLM_FLASHINFER_MOE_BACKEND=throughput`, `VLLM_FLOAT32_MATMUL_PRECISION=high`, no-Ray slightly
  better overall. Measured (2× Spark, FlashInfer-CUTLASS + throughput, no-Ray): pp2048 3065 t/s,
  **tg128 24.12 tok/s**, 32k TTFT 14.3 s. CUTLASS-only baseline: tg128 ~22 tok/s. Context up to
  ~225K (auto-calculated 196K). `--mamba_ssm_cache_dtype float32`, `--kv-cache-dtype fp8`,
  `--quantization modelopt_fp4`, `--load-format fastsafetensors`. Context degrades gracefully:
  tg128 at 131K depth ~11.3 tok/s. **This corroborates** the first-party AWQ-beats-NVFP4 finding
  (AWQ ~24 vs NVFP4 ~16.5 single-stream here — FlashInfer-CUTLASS NVFP4 ~24 is the *optimized* path
  matching AWQ, while stock CUTLASS NVFP4 is slower at ~22).
- **[reported]** **MiniMax-M2.7-AWQ-4bit on 2× Spark** (S-forum-m27-recipe, serapis + miken + co-le):
  `cyankiwi/MiniMax-M2.7-AWQ-4bit`, vLLM via eugr TF5, TP=2. **AWQ is the clear decode winner**:
  tg128 **39.4 tok/s** (peak 40), tg32 41.6 (co-le), vs NVFP4's 25.7 (peak 26). Context scales well:
  tg128 at 65K depth 25.15, at 131K 11.3; tg32 at 100K 21.2. AWQ on 2× is ~1.5× faster decode than
  NVFP4 on 2×. Multiple independent reporters (serapis, miken, co-le) all report ~39–42 tok/s —
  **strong consensus**. **This raises the AWQ-beats-NVFP4 finding to [reported] from multiple
  independent forum sources**, corroborating our first-party measurement.
- **[conjecture]** **MiniMax-M2.7 FP8 (Unsloth) on 4× Spark** (S-forum-m27-recipe, aostang):
  FP8 (Unsloth) on 4 nodes gives **36–37 tok/s** decode (no degradation vs NVFP4, slight *increase*),
  with cache hit 53.6 tok/s @ 2 concurrent. Surprising — FP8 (larger weights) matching/exceeding
  NVFP4 throughput on 4× may be compute-bound at that scale. Single source, unverified.
- **[reported]** **FlashInfer-CUTLASS is now stable enough for NVFP4 recipes** (S-forum-m27-recipe,
  eugr reply): autotuner exceptions fixed, minor FlashInfer optimizations merged. Eugr plans to
  switch all NVFP4 recipes from `VLLM_CUTLASS` to `flashinfer-cutlass`. This is a significant
  backend-status update — see `[[wiki/quantization-on-gb10.md]]`.

### Batch 11 forum ingest (2026-07-13)

- **[conjecture]** **MiniMax-M3-AWQ + EAGLE on 4-node CRS504 switch** (S-forum-4node-crs504,
  CosmicRaisins): M3-AWQ TP=4 with EAGLE drafter, 262K ctx, mns=4, bf16 KV, c=1. Decode 27.7–35.4
  tok/s across context depths (0–64K), prefill 1684–2211 tok/s. This is consistent with existing
  [reported] M3-AWQ TP=4 benchmarks (S-forum-m3-awq-tp4 33 tok/s, S-forum-m3-awq-4x ~30 tok/s)
  — the 100G CRS504 switch doesn't degrade decode vs direct 200G. See
  `[[wiki/benchmarks.md]]` for full table and `[[wiki/multinode-tp-and-networking.md]]` for
  CRS504 findings.
