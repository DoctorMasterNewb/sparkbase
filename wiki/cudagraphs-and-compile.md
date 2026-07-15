# CUDA graphs & torch.compile on GB10

> **area:** cudagraphs
> **status:** open-problem
> **evidence:** proven
> **sources:** S-xnode-cudagraph, S-m3-20tps, S-m3-vision, S-mimo-results, S-sess-jun5, S-pr46372
> **updated:** 2026-07-08

CUDA graphs remove per-token kernel-launch overhead — decisive on GB10 where decode fires thousands
of tiny kernels. But two walls block them for the exact models that need them most (big MoE,
cross-node). Know which wall you're against before you spend a night on it.

## Wall 1 — MoE cudagraph capture crashes on sm_121 (single-node too)

- **[proven]** **Symptom:** enabling cudagraphs (any mode, incl. PIECEWISE) for a large-expert-count
  MoE dies with `CUDA error: illegal memory access` during capture.
- **[proven]** **Root cause:** the MoE dispatch pattern isn't capturable on sm_121 (seen on sm_120 and
  sm_121 independently). **Dense models can use cudagraphs; large MoE cannot.**
- **Workaround:** `--enforce-eager` (or `cudagraph_mode=NONE`). Mandatory for big MoE on GB10.
- **Status:** open / hardware-arch. Cost is real: eager pays full launch overhead every token.

## Wall 2 — cross-node cudagraph capture faults (no GPUDirect)

- **[proven]** **Symptom:** 2-node TP=2, PIECEWISE capture dies with `illegal memory access` at
  `breakable_cudagraph…capture_end()`.
- **[proven]** **Root cause:** the **cross-node host-staged NCCL all-reduce gets captured**. GB10 has no
  GPUDirect (`[[wiki/platform-gb10.md]]`) → NCCL host-bounces → such a collective isn't
  CUDA-graph-capturable; capture finalization faults. Filed: **vllm-project/vllm#46253**
  (S-xnode-cudagraph).

**Why config can't fix it (for custom-quant MoE):**
1. **[proven]** The all-reduce is often **fused** (e.g. `fused_allreduce_gemma_rms_norm`) — not a
   standalone node, so adding `vllm::all_reduce` to `splitting_ops` doesn't exclude it.
2. **[proven]** vLLM **auto-enables `VLLM_USE_BREAKABLE_CUDAGRAPH=1`** for custom-quant models (can't run
   the `VLLM_COMPILE` piecewise path), and `breakable_cudagraph` captures the eager forward, **ignoring
   `splitting_ops`** (a torch.compile concept).
3. **[proven]** Forcing `VLLM_USE_BREAKABLE_CUDAGRAPH=0` → `AssertionError` (model can't use VLLM_COMPILE
   piecewise — which is *why* breakable was auto-enabled). ⟹ all three routes blocked.

- **[proven]** **Prototyped fix (not yet usable):** monkeypatch the cross-node collectives to **eager
  break points** (vLLM's `eager_break_during_capture`, the mechanism attention already uses) → **capture
  succeeds**. But two compounding caveats remain: replay produces garbage unless the collective output +
  captured segments share **static** buffers across the break; and without `CUDA_LAUNCH_BLOCKING=1`
  capture **deadlocks** (`shm_broadcast: No available shared memory broadcast block found in 60 seconds`
  — a lazy-Triton-JIT / NCCL desync between ranks). CLB serializes ranks but cripples throughput.
- **Status:** open / upstream — see PR status below.

**Scope (important, refined 2026-06-29):** **[proven]** this wall is **not universal to cross-node
TP=2.** It hits models on the **breakable_cudagraph path** (custom-quant → auto-enabled
`VLLM_USE_BREAKABLE_CUDAGRAPH`) that capture a **fused/host-staged all-reduce** — i.e. **MiniMax-M3**
(`fused_allreduce_gemma_rms_norm`). Standard piecewise-compile models, and even cross-node MoE like
**MiMo-V2.5** (live, TP=2 across both nodes), capture cudagraphs **cleanly** cross-node. So "cross-node
⇒ eager-only" is too strong — it's M3's fused-norm + breakable path specifically. (Wall 1, MoE-on-sm121,
is the broader eager-forcer.)

**[proven]** **Qwen3.6-35B-A3B** (HauhauCS NVFP4) also captures cleanly — single-node TP=1 with MTP-3,
PIECEWISE mode, 11 graphs in 3s, zero faults (S-sess-jul14). Confirms the standard piecewise path
(Qwen3.6 uses `vllm::qwen_gdn_attention_core` as a split op, not breakable) avoids Wall 2 entirely.

### Upstream status — PR #46372 (`fixes #46253`)
Community dev Pranav-d33 opened **vllm#46372**: add `@eager_break_during_capture` to
`tensor_model_parallel_all_reduce` + copy the result back into the input buffer (so the collective runs
eagerly during capture instead of being captured). As of 2026-06-29 the patch is **+6/-1, all-reduce
only** (narrowed from 3 collectives after our review — the original `copy_` was shape-wrong for
all_gather/reduce_scatter), author requesting re-review; maintainer ZJY0516 had requested changes
("don't disable cudagraph for nccl in all scenarios"); BoyuanFeng asked for a minimal 1-op repro.
- **[proven]** **Does it cover M3?** Maybe: in dev537, `fused_allreduce_gemma_rms_norm` calls the
  decorated `tensor_model_parallel_all_reduce` only on the **fallback** path (the flashinfer one-shot
  fused kernel bypasses it). On cross-node (`--disable-custom-all-reduce`) the fallback is what runs →
  decorator should engage. Worth an empirical M3 run.
- **[proven]** **Likely still blocked on replay.** The fix for correctness is the in-place
  `input_.copy_(result)` — the exact approach our prototype found gives **correct capture but garbage
  replay** (the surrounding captured segments need static buffers across the break). And the no-CLB
  capture deadlock + a single-node transport-capability probe are untouched. So this fixes the *crash*,
  probably not yet the *output*.
- **Staged GB10 test:** a COPY-only patched image (dev537 + the patch, no rebuild) + a minimal 1-op
  all_reduce cudagraph repro (answers BoyuanFeng) + a preflight that refuses to run while the live model
  is up. M3 is the faithful heavy test (weights evicted, 188 GB redownload + harness pass needed).
- **[proven]** **Minimal-repro RESULT (2026-06-29, ran on the pair):** a **bare single cross-node
  `all_reduce` captures AND replays correctly** on GB10 (torch 2.11/cu130, NCCL 2.30.7, host-staged
  RoCE). ⟹ #46253 is **not** "a single op can't be captured here" — it's specific to
  `breakable_cudagraph` capturing M3's **fused** all-reduce. The bare op replays fine precisely because
  there are no captured compute segments around the break.
- **[proven]** **Full-M3 RESULT (2026-06-29, ran on the pair — the faithful test):** patched M3 (PR
  #46372 + the shim's own fuller eager-break: "4 modules + fused-AR fallback") loaded fully (87.6
  GiB/node, 960 s) then **crashed at `Profiling CUDA graph memory: PIECEWISE` with `CUDA error: an
  illegal memory access`** (ProcessGroupNCCL watchdog) — **the exact #46253 signature**. So
  **eager-breaking `tensor_model_parallel_all_reduce` does NOT fix M3 cross-node cudagraph capture.**
  Three M3-on-dev537 bring-up blockers were cleared to get here.
- **[proven]** **Collectives capture test (2026-06-29, the disambiguator) — REFUTES the EP-collective
  hypothesis.** A bare single cross-node **`all_gather`, `reduce_scatter`, `all_to_all`** (the
  EP-dispatch primitives) **AND `all_reduce` ALL capture into a CUDA graph fine** on GB10 (no illegal
  access). So M3's crash is **not** "an EP collective is uncapturable" — the collective ops capture
  cleanly in isolation. ⟹ the residual M3 capture fault is the **sm_121 large-MoE kernel capture (Wall
  1)**, or a collective captured *in-context* (buffer interaction with surrounding ops), **neither
  fixable by a collective eager-break.** Conclusion: PR #46372's approach (eager-break collectives) is
  **the wrong layer for M3** — its real blocker is the MoE-kernel capture, not the cross-node collective.
  (Reconciles with Wall 1: large MoE can't cudagraph on sm_121.)
- **[proven]** **PR import defect found:** #46372's top-level `from
  vllm.compilation.breakable_cudagraph import …` in `communication_op.py` causes a **circular import** on
  dev537 (→ `compilation.monitor` → `vllm.config` mid-init); a lazy import fixes it.

## 2026-07-03 — BOTH #46253 walls cracked (capture + replay work; speed verdict below)

Seven-trial session on the pair (nvfp4-KV M3 + EAGLE3, dev537, stock breakable — shim eager-break
patch DISABLED via `M3_CG_EAGER_BREAK=0`). Root causes found, both one-line-scale, both upstreamable:

1. **[proven]** **The capture crash = `capture_error_mode="global"`.** `breakable_cudagraph._begin_segment`
   drives raw `capture_begin()` with torch's default *global* mode; NCCL requires **thread-local** capture
   mode when collectives are captured — its proxy/progress threads (constantly active on host-staged GB10
   transport) make CUDA calls that invalidate a global-mode capture → the illegal access at
   `capture_end`. Piecewise-path models (MiMo) never see it because `splitting_ops` keeps collectives OUT
   of their graphs. Shim patch: `M3_CG_THREADLOCAL_CAPTURE=1`. With it the crash is GONE.
2. **[proven]** **The no-CLB deadlock = rank phase-desync.** py-spy on the hung pair: rank 1 had FINISHED
   capture and returned to its RPC loop while rank 0 was wedged in `empty_cache` inside `_capture`, waiting
   on a peer collective that will never run. Fix: CPU-group barrier at capture entry + each eager break
   (`M3_CG_CAPTURE_BARRIER=1`, capture-time only, replay untouched).
3. **[proven]** **June's "garbage replay" is FIXED in dev537 stock** — the weak-ref/static-buffer machinery
   in `breakable_cudagraph.add_eager` replays coherently. (June's garbage was the dev197-era machinery + our
   own in-place eager-break wrapper.)

- **[proven]** **Composed result (trial 6): capture + coherent replay cross-node WITHOUT
  `CUDA_LAUNCH_BLOCKING`** — the exact #46253 scenario, solved. Trial matrix: CLB-only = works (8.13
  tok/s, CLB tax); global mode = crash; barrier-only = crash (global mode masks it); thread_local-only =
  deadlock (phase desync); **thread_local + barrier = works**. Caveat: `cudagraph_capture_sizes=[1]` only
  — sizes [1,2,4,8] crash again at the first (bs=8) capture: some batch>1 kernel path is still
  capture-unsafe (untriaged).
- **[proven]** **Speed verdict: no gain — 14.63 tok/s with graphs vs 14.62 eager (code-gen, nvfp4
  config).** Breakable mode sets compile mode NONE, and its graphs wrap only the small per-layer segments
  between ~60 eager attention breaks — launch overhead survives. Same net-zero result tonyd2wild reported
  on their 4× pair. **The real ~2× needs the true `torch.compile` PIECEWISE path, i.e.
  `@support_torch_compile` on the M3 model class** (upstream model work: the arch is hardcoded into the
  breakable auto-enable list in `config/vllm.py` because it isn't compile-traceable). Recipes and shim
  knobs (`M3_CG_*` env vars in the m3-port plugin) preserved locally.

## Consequence: the "20 tok/s" math

**[proven]** For a cross-node MoE you're **eager-bound** (Wall 1 + Wall 2) AND paying ~120 host-bounced
all-reduces/token. The big multipliers are gone:
- **cudagraphs** → walled (both walls).
- **MTP speculative decode** → ~0 gain *without* cudagraphs, and the checkpoint may ship zero MTP
  weights (`num_mtp_modules=0`).
- **Pipeline parallel** (fewer cross-node hops/token) → many vLLM model classes don't implement
  `SupportsPP` (`NotImplementedError`).
- **Inductor compile w/o cudagraph** → ~+25%, risks the lazy-Triton deadlock.

**[proven]** Optimistic eager ceiling ≈ 8–10 tok/s. MiniMax-M3 cross-node lands ~5 tok/s — a long-context
/ multimodal research endpoint, not a daily driver. Fast paths need single-node MoE NVFP4 instead
(`[[wiki/quantization-on-gb10.md]]`).

## Other capture/compile gotchas

- **[proven]** **CUDA-graph memory profiling is on by default** (vLLM ≥ v0.21.0) and shifts effective
  `--gpu-memory-utilization` down (~0.60 behaves like ~0.5875). Bump util slightly to keep KV size, or set
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`.
- **[proven]** **First-shape capture causes a one-time TTFT spike** (e.g. conc-8 first hit ~3 s); benign
  warmup — re-run after warm-up.
- **[proven]** **Cross-node MoE cudagraphs can work** — MiMo-V2.5 NVFP4 runs **TP=2 across both nodes with
  cudagraphs** (live default) and captures cleanly. So the cross-node wall is M3's fused-norm/breakable
  path specifically (see Wall 2 scope), not all cross-node TP=2.

## See also
`[[wiki/platform-gb10.md]]` · `[[wiki/multinode-tp-and-networking.md]]` · `[[wiki/models/minimax.md]]`
