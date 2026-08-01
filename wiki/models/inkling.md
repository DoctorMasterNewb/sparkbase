# Inkling (Thinking Machines) — NVFP4 on 8× DGX Spark

> **area:** model
> **status:** evolving
> **evidence:** conjecture
> **sources:** S-forum-inkling, S-forum-inkling-nvfp4, S-forum-inkling-small-2x, S-forum-inkling-small-disc
> **updated:** 2026-08-01

Thinking Machines' **Inkling** multimodal MoE family — announced as **Inkling 975B (41B active)** and
**Inkling-Small 276B (12B active)**, 1M-token context, text/image/audio/video. Designed for B300-class
datacenter, but community member `greg190` brought the larger Inkling up on an **8× DGX Spark cluster
(GB10, sm_121a)** with NVFP4 and published the recipe + 12 patches (`blockmos/inkling-sparks-gb10`).

This page records the durable GB10-specific findings from that bring-up. All claims are **[conjecture]**
(single forum source) until independently reproduced — but the thread is unusually technically dense:
real tok/s tables, filed vLLM issues, GPU-coredump-root-caused kernel bugs, and a public patch set.

## The bring-up config (8× Spark, NVFP4, TP=8)

- **[conjecture]** **NVFP4 runs clean on sm_121a with no dtype fallbacks** (S-forum-inkling-nvfp4,
  greg190): the Inkling NVFP4 checkpoint serves with correct output and no silent bf16 fallback —
  "nvfp4 itself is fine." This corroborates the existing finding that NVFP4 is operational on GB10
  via the Marlin weight-only decompress path (`[[wiki/quantization-on-gb10.md]]`).
- **[conjecture]** **8× DGX Spark cluster, GB10 sm_121a, TP=8** (S-forum-inkling-nvfp4): the recipe
  targets 8 nodes (~968 GB combined unified memory). Engine: vLLM (forked, with the patch set below).
- **[conjecture]** **`LAMPORT_RS_SCONV=0` is the escape hatch for Inkling's Lamport collectives on
  RoCE clusters** (S-forum-inkling-nvfp4): Inkling's collectives require **MNNVL (NVLink fabric)** and
  hard-error on RoCE clusters (which is all GB10 has — no NVLink between nodes, only ConnectX-7 RoCE).
  Thinking Machines shipped the escape hatch env var `LAMPORT_RS_SCONV=0` to bypass this. **Why it
  bites on Spark:** GB10 clusters are RoCE-only (no NVLink fabric / no GPUDirect — see
  `[[wiki/platform-gb10.md]]`), so any model using Lamport-style collectives will hard-error without
  this flag. This is a new class of "designed-for-datacenter-fabric" model biting on Spark.
- **[conjecture]** **`scipy` is a new vLLM-main dependency not in older CUDA-13 base images**
  (S-forum-inkling-nvfp4): fix `pip install scipy`. Minor but cost the OP time.
- **[conjecture]** **`--compilation-config` mode pin: `"mode": 0`** (S-forum-inkling-nvfp4): dropping
  `--enforce-eager` without pinning mode=0 sends vLLM into full `torch.compile`, which corrupts
  output on this model. Pin mode 0 in `--compilation-config` instead of relying on `--enforce-eager`.
- **[conjecture]** **`--gpu-memory-utilization 0.70` max on 8× Inkling** (S-forum-inkling-nvfp4): the
  missing 30% is page cache + ray + k8s + CUDA runtime, not waste. **0.78 wedges the node.** This is
  a tighter cap than the typical 0.80–0.90 seen on smaller models — the 8-node orchestration overhead
  (ray, k8s) eats more of the 121 GB pool. Consistent with the proven
  `--gpu-memory-utilization` is NOT a hard cap / UMA sizing guidance
  (`[[wiki/platform-gb10.md]]`).

## Performance: the long-context cliff (the core finding)

- **[conjecture]** **Decode holds up on short prompts then drops hard as context grows**
  (S-forum-inkling-nvfp4, greg190). Reported decode tok/s on 8× Spark NVFP4:

  | context | c1 decode | c8 (total) | c32 (total) |
  |---|---|---|---|
  | short (~100 tok) | 25 (27 w/ MTP k=2) | 80 | 193 |
  | 2048 tokens | 13.5 | 25 | 24 |

  Prefill: **~1,400 tok/s at 2048 ctx** (throughput config up to **2,711 tok/s** — "higher than we
  ever got M3"). The short-prompt numbers are the OP's own bench; the 2048 numbers are via
  `llama-benchy`.

- **[conjecture]** **The cliff is because the sm_121a cute kernels have no paged-KV support**
  (S-forum-inkling-nvfp4): the `tml_fa4` Sm120 attention path has no paged-KV — vLLM's paged cache
  can't feed it. The workaround **re-gathers the whole KV history into contiguous buffers every
  decode step**, which is O(ctx) per token → the engine caps at ~24 tok/s aggregate at real context
  no matter the concurrency. **This is the load-bearing GB10-specific finding:** until paged-KV lands
  in the Sm120/Sm121 cute FA4 path, any model routing through `tml_fa4` on Spark will hit this
  long-context decode cliff. See `[[wiki/attention-and-kv-cache.md]]`.

- **[conjecture]** **MTP is stuck at k=1 (60% draft acceptance)** (S-forum-inkling-nvfp4): the OP
  could not get MTP beyond k=1; draft acceptance ~60%. MTP k=2 adds ~2 tok/s on short context
  (25→27). The multi-depth MTP path is "not there yet on sm_121a" — consistent with the MoE cudagraph
  wall and MTP-needs-cudagraphs findings (`[[wiki/cudagraphs-and-compile.md]]`).

## Cudagraphs: working (after a boundary-bug fix)

- **[conjecture]** **Cudagraphs work on Inkling 8× Spark after root-causing a boundary bug in the
  Sm120 rel-bias attention kernel** (S-forum-inkling-nvfp4): the OP got cudagraphs working (was
  stuck on eager) by root-causing the bug via **GPU coredump** (`CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1`
  + host-mounted `CUDA_COREDUMP_FILE`, then `cuda-gdb`). This is notable because it is a reported
  case of **cudagraphs working on a large MoE on sm_121a** — whereas the proven Wall 1
  (`[[wiki/cudagraphs-and-compile.md]]`) is that large-expert-count MoE cudagraph capture crashes on
  sm_121. Inkling may have a more capture-friendly expert dispatch pattern, or the patched kernel
  path avoids the crash. **Unverified** — single source, no repro.

## Kernel bugs found & filed (the durable artifacts)

These are the most reusable outputs of the bring-up — real sm_121a kernel bugs with one-line fixes,
filed upstream:

- **[conjecture]** **Cute FlashAttention Sm120 kernel doesn't clamp the rel-bias q-row index →
  illegal memory access at some shapes** (S-forum-inkling-nvfp4, filed as **vllm#49049**): caught
  with GPU coredumps, byte-identical fault on 2 ranks = deterministic. One-line fix. This is the bug
  that was blocking cudagraph capture.
- **[conjecture]** **Don't route sm12x to the sheared/`tml_fa4` rel-bias path — the Sm120 kernel
  there discards the relative-position bias** (S-forum-inkling-nvfp4): no bias parameter in the
  Sm80-inherited call → plausible-but-wrong outputs on every layer. The **score-mod
  `vllm_flash_attn/cute` path is the intended sm12x route.** A model with rel-position bias
  silently producing wrong output on sm_121 is a sharp trap.
- **[conjecture]** **Two one-line bugs in the never-before-compiled cute Sm80/Sm120 base**
  (S-forum-inkling-nvfp4): `flash_fwd.py` references `mDynamicCausal` which isn't a kernel parameter
  (→ `NameError` during DSL tracing; safe fix `psc = None`, `dynamic_causal` is SM90-only), and
  `self.is_split_kv` is referenced but never assigned (→ `= False` in init). These suggest the
  Sm120 cute path had never been exercised on this arch before.
- **[conjecture]** **Phantom varlen work tiles → `cudaErrorIllegalAddress`**
  (S-forum-inkling-nvfp4): the `SingleTileVarlenScheduler` launches an upper-bound grid; Sm90 kernels
  check `work_tile.is_valid_tile`, the Sm80/Sm120 base doesn't, so phantom tiles read
  `cu_seqlens[num_batch+1]` out of bounds. **Verified with `compute-sanitizer`** (invalid read
  decoded exactly to the first phantom block). One-line fix: remap phantom tiles to a real tile
  (duplicate compute is benign, `is_valid` stays false so Sm90+ unaffected).
- **[conjecture]** **KV write in `fused_qkvr_prep` races the attention read on a side stream**
  (S-forum-inkling-nvfp4): move to main stream. A real concurrency bug in the FA prep path on sm_121.
- **[conjecture]** **On unified memory, out-of-range indexes usually DON'T crash — they land in
  another allocation and silently corrupt** (S-forum-inkling-nvfp4): "looked like a race for days."
  This is a GB10-specific debugging insight: the absence of discrete VRAM means OOB reads don't fault
  — they silently hit another UMA allocation. Diagnose with `compute-sanitizer`, not crashes. See
  `[[wiki/platform-gb10.md]]` → unified-memory OOM/constraints.
- **[conjecture]** **vLLM V2 model runner traces attention under `FakeTensorMode` — any `.item()` in
  a patch path needs a fake-tensor early-return** (S-forum-inkling-nvfp4): warmup tracing gotcha for
  anyone patching the attention path.

## Debugging technique (reusable)

- **[conjecture]** **GPU coredump workflow on GB10 for kernel bugs**
  (S-forum-inkling-nvfp4): `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` + host-mounted
  `CUDA_COREDUMP_FILE`, then `cuda-gdb` — "names the exact faulting instruction." The OP credits the
  [vLLM CUDA Core Dump blog](https://blog.vllm.ai) workflow. Also: shipping with the coredump
  instrumentation **on** as a stabilizer for a timing-dependent race (small overhead) until the race
  is located — `CUDA_ENABLE_COREDUMP_ON_EXCEPTION=1` perturbs driver timing enough to mask the race.

## Verdict: parked in favor of M3

- **[conjecture]** **Inkling parked; M3 restored as daily driver** (S-forum-inkling-nvfp4): the OP
  reverted to their M3 build after the long-context decode cliff made Inkling impractical. M3 serves
  at **42 tok/s single-user** and "scales with concurrency" — Inkling's ~24 tok/s aggregate ceiling
  at real context (paged-KV workaround) doesn't compete. The public repo (`blockmos/inkling-sparks-gb10`)
  is left for the community to push further. The blocking items are all **kernel maturity on
  sm_121a** (paged KV, long-context, multi-depth MTP), not NVFP4 itself.

## Inkling-Small-NVFP4 on 2× DGX Spark (2026-08-01 forum ingest)

> **evidence:** conjecture (single forum thread, multiple users in same thread)
> **sources:** S-forum-inkling-small-2x, S-forum-inkling-small-disc

The smaller Inkling-Small (276B / 12B active, NVFP4) was released and the community immediately
attempted bring-up on 2× DGX Spark. Key findings:

- **[conjecture]** **NVFP4 fits on 2× Spark but no FP8 KV cache → context capped at ~300K**
  (S-forum-inkling-small-2x, eugr_nv): the official NVFP4 checkpoint
  (`thinkingmachines/Inkling-Small-NVFP4`) fits in 2× GB10 unified memory, but the model does
  not support FP8 KV cache — only BF16 KV. With BF16 KV, 2× Spark (~242 GB combined) can only
  fit ~300K tokens of context, far short of the model's 1M native window. This is a significant
  disadvantage vs DeepSeek-V4-Flash, which uses much less KV memory. Multiple users express
  frustration (sjug, PILCOTHINK). eugr_nv: "dual Sparks don't have enough VRAM to fit >~300K
  tokens with this model as it doesn't support fp8 cache."

- **[conjecture]** **Inkling uses BF16 for global attention — FP8 KV requires FlashAttention
  kernel modification** (S-forum-inkling-small-2x, PILCOTHINK citing vLLM blog): per the vLLM
  blog post (15 Jul 26), "Inkling currently uses BF16 for global attention, so enabling FP8
  will likely require modifying the Flash-attention kernel specifically used by Inkling."
  This means FP8 KV support is not a config toggle — it needs kernel-level work. 0rand notes
  the same pattern in MLX (BF32 for attention → ballooning KV cache).

- **[conjecture]** **spark-vllm-docker recipe available (experimental)** (S-forum-inkling-small-2x,
  eugr_nv / PILCOTHINK): `./run-recipe.sh inkling-small-nvfp4` — uses `vllm-node` container
  with `mods/inkling-sm12-paged-kv` + `mods/drop-caches` patches. Recipe is cluster-only
  (2× Spark TP=2). Tool calling is broken in the current build.

- **[conjecture]** **Tool-calling parser bug — direct streaming emits tool-call markers as
  visible content** (S-forum-inkling-small-2x, ekkis / adrianwild): when
  `--reasoning-parser inkling` and `--tool-call-parser inkling` are both enabled, a tool call
  emitted directly after the model message header (without a preceding thinking block) is
  streamed as visible `<|content_invoke_tool_json|>` text and never reaches the tool parser.
  Non-streaming requests and streams with a preceding thinking block are unaffected. ekkis
  created a patch (`patch_inkling_parser.py`) that keeps the parser in `MESSAGE_HEADER` long
  enough to suppress optional function-name metadata, promotes the reasoning adapter only
  when the direct block marker arrives. adrianwild confirms removing the reasoning parser
  also works as a workaround.

- **[conjecture]** **Tool-eval-bench: 76/100 (4-star "Good")** (S-forum-inkling-small-2x, ekkis):
  first reported tool-eval-bench score for Inkling-Small-NVFP4 on 2× Spark. 94% completion
  rate (5 scenarios excluded due to infrastructure failures — timeouts/5xx on structured
  outputs). 2 safety-critical failures (prompt injection resistance, cross-turn sleeper
  injection). Categories: Parameter Precision 100%, Error Recovery 100%, Localization 100%,
  Structured Reasoning 100%, Code Patterns 100%; weaker: Structured Output 50%, Toolset Scale
  50%, Autonomous Planning 67%. The model is "much more literal in interpreting commands than
  Deepseek v4 Flash" per ekkis. Median turn responsiveness: 2.1s.

- **[conjecture]** **DSV4-Flash uses less KV memory than Inkling-Small at high context**
  (S-forum-inkling-small-disc, thomas.developer1): "DSV4 takes up a LOT less memory when the
  context window starts to fill up. So for just average agentic work DSV4 is still king." This
  is because DSV4-Flash supports NVFP4 KV cache (see `[[wiki/models/minimax.md]]` DSpark),
  while Inkling-Small is stuck at BF16 KV.

- **[conjecture]** **tonyd2wild BF16-KV 262K DSpark variant in progress**
  (S-forum-inkling-small-2x, tonyd615): a community variant
  (`tonyd2wild/Inkling-Small-NVFP4-DSpark-BF16-KV-262K-2x-DGX-Spark`) targeting 262K context
  with BF16 KV + DSpark speculative decoding on 2× Spark. In progress, no benchmarks yet.

- **[conjecture]** **Qwen3.5-122B FP8 as a vision-capable alternative on dual Spark**
  (S-forum-inkling-small-disc, peter.h177): for users needing vision on 2× Spark, Qwen3.5-122B
  FP8 is the current fallback — "I usually try to get around with the Qwen 3.5 122B using FP8
  quant — it can get things done in unity." Notes Qwen3.5-397B has a better visual encoder but
  only lower quant would fit 2× Spark, making it worse than 122B FP8 for vision tasks.

## See also
`[[wiki/attention-and-kv-cache.md]]` (paged-KV, cute FA4) · `[[wiki/cudagraphs-and-compile.md]]`
(MoE cudagraph wall) · `[[wiki/quantization-on-gb10.md]]` (NVFP4 Marlin path) ·
`[[wiki/platform-gb10.md]]` (UMA silent corruption, no GPUDirect) ·
`[[wiki/multinode-tp-and-networking.md]]` (8× cluster, RoCE-only fabric) ·
`[[wiki/roadmap.md]]` (open problems: paged-KV for Sm120 cute FA4, Inkling-Small on 2× Spark)