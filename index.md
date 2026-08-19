# sparkbase index

The map of the wiki. Every page, grouped, one line each. Start here. (Contract: [`SCHEMA.md`](SCHEMA.md).)

Every claim on these pages carries an **evidence tag** — `[conjecture]` `[reported]` `[reproduced]`
`[proven]` `[superseded]`. Build on `[proven]`; treat `[conjecture]` as "try it and tell us."

## Foundations
- **[Hardware-parity tenet](wiki/platform-gb10.md#foundational-tenet-hardware-parity-read-before-replicating-any-community-finding)** — DGX Spark is standardized; non-reproduction of a community finding is a software delta on your side, never an immutable hardware difference. Read before replicating any forum result.
- [platform-gb10](wiki/platform-gb10.md) — the hardware: sm_121/12.1a, 121 GB unified, ~270 GB/s (decode is bandwidth-bound), no native FP4/FP8-blockscale, no GPUDirect, OOM=reboot. **Read first.**
- [quantization-on-gb10](wiki/quantization-on-gb10.md) — what runs native (online-dynamic FP8) vs Marlin (FP4, block-scale FP8); ModelOpt-NVFP4; MXFP8/AWQ/AutoRound/GGUF; loader bugs.
- [cudagraphs-and-compile](wiki/cudagraphs-and-compile.md) — the two cudagraph walls (MoE on sm_121, cross-node host-staged NCCL / vllm#46253) and the "20 tok/s" math.
- [multinode-tp-and-networking](wiki/multinode-tp-and-networking.md) — CX7 twins + NCCL_IB_HCA + MTU 9000; no-ray TP; `--disable-custom-all-reduce`; mDNS/sshd ops; why cross-node is slow.
- [attention-and-kv-cache](wiki/attention-and-kv-cache.md) — TRITON_ATTN / DIFFKV / FLASHINFER selection; block-size 128 for MSA; fp8 KV; ViT JIT; gemma-norm ICE.

## Engines & tooling
- [engines](wiki/engines.md) — vLLM vs Atlas vs llama.cpp; Atlas internals (AOT kernels, KV sizing, MTP, loader bugs); durable serving pattern.
- [containers-and-tooling](wiki/containers-and-tooling.md) — known images & what they load; probing tricks; std env; Xet/permission/io_uring gotchas; ComfyUI flags.
- [llama-cpp-rpc](wiki/llama-cpp-rpc.md) — GGUF + 2-node pipeline RPC; `--no-mmap` mandatory; sm_121a build; tensor-split.

## Models
- [mimo-v2.5](wiki/models/mimo-v2.5.md) — 310B Omni MoE NVFP4+MXFP8 DiffKV; mods chain; abliteration-is-damaged diagnostic.
- [minimax](wiki/models/minimax.md) — M2.7 AWQ daily-driver (~24 tok/s); M3 428B MSA+vision cross-node (~5 tok/s, walled).
- [holo-3.1](wiki/models/holo-3.1.md) — computer-use VLM (Qwen3.5 VL MoE); NVFP4 wins; thinking-OFF = 4.2×.
- [gemma-4](wiki/models/gemma-4.md) — 12B unified arch (image support = serveability); FP8 online-dynamic 2× fast path.
- [diffusiongemma](wiki/models/diffusiongemma.md) — 26B-A4B block-diffusion LLM; native in vllm-node; NVFP4 MoE needs VLLM_TEST_FORCE_FP8_MARLIN; bf16 deployed, NVFP4 retired.
- [qwen](wiki/models/qwen.md) — 3.5/3.6/Coder-Next; MoE-A3B NVFP4+MTP ~142 tok/s vs dense ~30; Atlas loader landmines.
- [nemotron-3](wiki/models/nemotron-3.md) — hybrid Mamba-2 MoE; 120B Q8 via llama.cpp RPC; Nano-Omni vision/omni single-node.
- [mistral-small-4](wiki/models/mistral-small-4.md) — 119B NVFP4 MLA; TRITON_MLA resolves head_size=320 on SM121; ~28-33 tok/s [reported]; Eagle/MTP not working; --shm-size 16g kernel crash.
- [step-3.7](wiki/models/step-3.7.md) — retired; kept for the MTP-needs-cudagraphs finding.
- [laguna-s-2.1](wiki/models/laguna-s-2.1.md) — 117.6B MoE NVFP4 single-node; DFlash spec=7; 22.6 tok/s decode, flat across depths; **retired** — output quality below MiMo/DeepSeek, no speed advantage over Qwen3.6.
- [inkling](wiki/models/inkling.md) — Thinking Machines multimodal MoE (975B/41B-active + Small 276B/12B-active); NVFP4 on 8× and 2× Spark, paged-KV cliff, FP8 KV absent (BF16 only), tool-calling parser bug, Lamport-on-RoCE escape hatch, kernel bugs filed.
- [glm-5.2](wiki/models/glm-5.2.md) — Zhipu AI 744B/40B-active MoE (sparse-MLA); 4×–8× Spark recipes, hybrid FP8+NVFP4+MXFP4 quant, MTP quality, reasoning-parser bug, KV kernel constraints.
- [kimi-k3](wiki/models/kimi-k3.md) — Moonshot AI Kimi K3 (~2.8T MoE); REAP-320 MXFP4 on 8× Spark 21-30 tok/s; full model needs 16× GB10; REAP variant loops.
- [muse-glimmer](wiki/models/muse-glimmer.md) — Meta 30B dense with DFlash; llama.cpp 44.6 tok/s Q6_K_XL, vLLM NVFP4 18.65 tok/s; tool-calling BFCL 10-12% (multi-tool fails); vLLM DFlash broken.
- [k-exaone-236b](wiki/models/k-exaone-236b.md) — LG AI Research 237B/23B-active MoE; largest unpruned model on single Spark; mixed-quant GGUF 85.56 GiB, LLLG sliding-window 48 KiB/token KV, full 262K context via ds4 engine.

## Reference
- [benchmarks](wiki/benchmarks.md) — collated decode tok/s + concurrency table; append rows.
- [roadmap](wiki/roadmap.md) — open problems & areas of further development.
- [sources](sources/README.md) — where findings came from (`S-` ids, source-typed).
- [log](log.md) — append-only ingest/change log.

## Forum ingest 2026-08-19 (Batch 79)
- 4 new NVIDIA DGX Spark forum topics found, 2 technically relevant (2 skipped: DeepSeek
  Harness Preview — agent framework link, no GB10 findings; Thermal Performance — DIY
  cooling accessories, no durable technical findings).
- 2 new sources registered (Batch 79). 4 topic IDs added to processed_topics.txt
  (total now 630).
- **Headline finding 1:** GLM-5.2 QuantTrio Int4-Int8Mix on 4× Spark via streamlined
  sparkrun recipe (davedgd/sparkrun-glm52-4x-spark) — uses ciprianveg v18 Docker image,
  supports optional baseten/GLM-5.2-Vision-NVFP4 vision tower, Adaptive MTP. tool-eval-bench
  v2.5.1.dev31: 86/100. llama-benchy: 22.17 tok/s decode (tg32, c1), 556 tok/s prefill.
  AIME25: 90% (30/30, 57 tok/s avg). 1M max context. Corroborates existing [reported]
  20-25 tok/s 4× Spark decode range. [conjecture].
- **Headline finding 2:** K-EXAONE-236B-A23B (LG AI Research, 237B/23B-active MoE) on
  single DGX Spark via ds4 engine — largest reported unpruned model on one GB10. Mixed-
  quant GGUF (IQ2_XXS+Q3_K+Q4_K+Q8 per-tensor): 441.63 GiB BF16 → 85.56 GiB (5.16×).
  Full 262,144-token context at 103.95 GiB resident. LLLG sliding-window schedule =
  48 KiB/token KV (only 12/48 layers hold full context) — key enabler. Decode 10.51
  tok/s @1.4K → 5.42 tok/s @16K. MTP (blk.48) executes but net loss on this HW. 16/16
  OpenAI API validation checks pass. Multi-turn prefix-resume divergence workaround
  (re-tokenization mismatch). NEW model page created. [conjecture].
- Pages touched: models/glm-5.2 (sparkrun 4× recipe section + cross-thread table row
  [conjecture]), models/k-exaone-236b (NEW — full model page [conjecture]), benchmarks
  (2 new [conjecture] rows), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-19 (Batch 78)
- 2 new NVIDIA DGX Spark forum topics found, both technically relevant.
- 2 new sources registered (Batch 78). 2 topic IDs added to processed_topics.txt
  (total now 626).
- **Headline finding 1:** Qwen3.8-27B-MixedInt4-AutoRound for single DGX Spark —
  PILCOTHINK mixed 4-bit quant (sensitive layers FP8/FP16, vision unquantized),
  20.8 GB. MMLU recovery 99.32% (83.49→82.92%, -0.57pp). vLLM recipe: TP=1, fp8 KV,
  MTP nst=3, 1.01M max context, 2.56M-token KV pool (2.54× concurrency). llama-benchy
  decode 21.86 tok/s @ d0 / 17.04 @ d4096 / 17.82 @ d8192, prefill 828-877 tok/s.
  tool-eval-bench 91/100 normal / 92/100 hardmode v2.5.1. SlopOps SAR variant 88/100
  hardmode v2.1.0, MTP nst=3 15.08 tok/s. co-le 35-40 tps on 2× Spark. 0rand DSpark
  28-35 t/s 8-bit single Spark. Confirms proven bandwidth-bound dense 27B regime
  (~17-30 tok/s). 2.56M-token KV pool at 1.01M context is the standout — 20.8 GB
  weights leave most of 121 GB UMA for KV cache. [conjecture].
- **Headline finding 2:** Docker Compose vs Kubernetes for Spark — community consensus:
  spark-vllm-docker + sparkrun sufficient for 2-4 Sparks, k8s overkill. bugsareyummy
  dropped Rancher k8s to recover RAM for vLLM on 121 GB UMA. GB10-specific: k8s RAM
  overhead is a real cost on UMA. [conjecture].
- Pages touched: models/qwen (Qwen3.8-27B-MixedInt4 section [conjecture]), benchmarks
  (5 new [conjecture] rows), containers-and-tooling (Docker vs k8s [conjecture]),
  sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-18 (Batch 77)
- 5 new NVIDIA DGX Spark forum topics found, 4 technically relevant (1 skipped:
  ThinLinc VDI — project announcement, no GB10 inference findings).
- 4 new sources registered (Batch 77). 5 topic IDs added to processed_topics.txt
  (total now 624).
- **Headline finding 1:** HDMI hot-plug A/B test on 2 identical ASUS GX10
  units — controlled experiment isolating display-hotplug from headless
  operation. HDMI connected = 36–37°C; HDMI unplugged (session active) =
  monotonic rise to 45°C / 47.9°C ACPI in 5 min (P8, 3.5 W). Headless control
  unit (no display, GDM greeter) stays cool at 34–36°C — headless alone is not
  sufficient. Corroborates existing [reported] fan-DPMS finding (S-forum-fan-dpms).
  Per-unit (1 of 2 identical) suggests EC/fan hardware variation. [conjecture].
- **Headline finding 2:** USB-C DisplayPort (DFP-1 to DFP-4) not detected after
  boot unless physical replug — new GB10 platform bug on MSI EdgeXpert. HPD/
  sink-detection runs once in narrow early-boot window. 2 confirmers (MSI +
  Lenovo PGX). /sys/class/typec/ empty, software re-probe fails. HDMI-0 works
  at boot. Practical impact: blind during firmware updates. [conjecture].
- **Headline finding 3:** ConnectX-7 promiscuous mode silently fails on GB10 —
  API returns success but hardware does not enter promisc mode. FW 28.45.4028.
  Relevant for DPI/sniffing/proxying on CX-7. [conjecture].
- **Headline finding 4:** MediaLLMProxy — production-deployed OpenAI-compatible
  vision bridge for text-only LLMs. Qwen 3.0 3B VL on llama.cpp on Spark as
  backup vision model. DeepSeek-OCR-2 explored (~3 GB) but weak for non-OCR.
  Extends Pilco-mmbridge pattern with proxy/bridge architecture. [conjecture].
- Pages touched: platform-gb10 (HDMI hot-plug A/B + USB-C DP HPD [conjecture]),
  multinode-tp-and-networking (CX-7 promisc mode [conjecture]), engines
  (MediaLLMProxy vision bridge [conjecture]), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-18 (Batch 76)
- 3 new NVIDIA DGX Spark forum topics found, 0 technically relevant (3 skipped:
  vLLM exit-255 hard-reset — RMA (defective thermostat); GLM-5.3 countdown —
  social/hype; GX10 rescue image with Clonezilla — OS admin).
- No new sources registered. 3 topic IDs added to processed_topics.txt
  (total now 619). No pages touched.

## Forum ingest 2026-08-17 (Batch 75)
- 4 new NVIDIA DGX Spark forum topics found, 2 technically relevant (2 skipped:
  Asus GX10 OS image — buying advice; VoiceChat 11B NIM arm64 roadmap — no
  GB10 findings).
- 2 new sources registered (Batch 75). 4 topic IDs added to processed_topics.txt
  (total now 616).
- **Headline finding 1:** DGX Spark fans do not spin in headless boot mode — temp
  rises to ~60-70°C at idle. Display-hotplug dependent, unit-specific (1 of 2
  identical ASUS GX10 units). Corroborates existing [reported] fan-DPMS finding
  (S-forum-fan-dpms): same mechanism, new manifestation. Not driver-version-
  specific (580.126.09 and 580.173.02 both affected). [conjecture].
- **Headline finding 2:** s2idle suspend fails on DGX Spark GB10 —
  nvidia-suspend.service crashes inside the driver (nv.c:4784 WARNING), PCI PM
  returns -5, suspend never completes. `/sys/module/nvidia/parameters/` does not
  exist. New durable error string. Confirms suspend is not viable on GB10 (use
  full shutdown + smart plug). [conjecture].
- Pages touched: platform-gb10 (headless-boot fan [conjecture], s2idle suspend
  failure nv.c:4784 [conjecture]), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-17 (Batch 74)
- 2 new NVIDIA DGX Spark forum topics found, 1 technically relevant (1 skipped: buyer
  beware Amazon MSI — buying advice).
- 1 new source registered (Batch 74). 2 topic IDs added to processed_topics.txt (total
  now 612).
- **Headline finding:** 2× DGX Spark FE silent hard-locks under sustained DSV4-Flash-0731
  inference — fieldiag PowerStress FAIL (020000600139) on both units at stock, latest
  firmware; 12 lock events, 3× reproduced per unit across fieldiag 1.0.9 + 2.0.4; one
  unit hard-locked mid-PowerStress under MODS-only (no OS). 3rd independent forum thread
  documenting PowerStress thermal failure on GB10 (now on Founders Edition). Two measured
  mitigations: GPU clock cap 2100 MHz (−21% decode / 32K, +2.3× decode @ 262K, zero locks
  after); CPU freq cap 2.4 GHz (free — 92→84 °C, zero perf cost, +16% decode). New durable
  finding: CPU governor is not the lever — only capping active cores' max frequency helps;
  vLLM TP workers busy-poll at 200-350% CPU up to 3.9 GHz. RMA approved ~48h. [conjecture].
- Pages touched: platform-gb10 (FE thermal hard-lock + PowerStress 3rd corroboration +
  GPU clock cap 2100 + CPU freq cap 2.4 GHz + fieldiag 2.0.4 gotchas + RMA notes
  [conjecture]), sources/README, index, log.
- All [conjecture] — single-source forum thread. No evidence promotions.

## Forum ingest 2026-08-16 (Batch 73)
- 4 new NVIDIA DGX Spark forum topics found, all 4 technically relevant.
- 4 new sources registered (Batch 73). 4 topic IDs added to processed_topics.txt (total
  now 610).
- **Headline finding 1:** GB10 Grace CPU energy telemetry full audit (peer-reviewed,
  arXiv:2605.27599, LOCO 2026). DCGM field 156 (cumulative mJ) works for GPU rail.
  No CPU energy interface works — SCMI, I2C, hwmon, powercap all empty. SPBM driver
  (NVDA8800:00) fails on GX10 due to MTKW9000 ACPI memory conflict (0x05170000 overlap).
  spark_hwmon loads cleanly on Acer GN100 (no conflict, 14 power + 4 energy channels).
  NVIDIA staff says "fixed in July Updates" but July EC update does NOT address energy.
  nvidia-smi --query-gpu=energy.consumption returns "Field not valid." Strongly
  corroborates [reported] nvidia-smi 12-27% undercount. [conjecture].
- **Headline finding 2:** Inferact/Muse-Glimmer-30B-NVFP4-W4A4 — 52.55 tok/s on DGX
  Spark with vLLM (Spark Arena Benchmark). NVFP4 W4A4 activation-quantized variant.
  No recipe details provided. [conjecture].
- **Headline finding 3:** ASM2464PD USB4 NVMe enclosure falls back to USB 2.0 on ASUS
  GX10 after every boot — soft-replug script automates fix (cyrozap/usb-to-pcie-re
  reverse-engineered CPU reset). Multiple users confirm since Oct 2025. [conjecture].
- **Headline finding 4:** GLM-5.2 on 4× Spark via ciprianveg gb10-glm-5.2:v18-vision
  stack — 25 tok/s decode, 700+ tok/s prefill at 300K context with DCP2 + decode-aware
  prefill + NVFP4 KV. Corroborates [reported] 20-25 tok/s 4× decode range. [conjecture].
- Pages touched: platform-gb10 (energy telemetry + USB4 soft-replug [conjecture]),
  models/muse-glimmer (NVFP4 W4A4 benchmark [conjecture]), models/glm-5.2 (ciprianveg
  4× data point [conjecture] + cross-thread table row), benchmarks (1 new [conjecture]
  row), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-16 (Batch 72)
- 3 new NVIDIA DGX Spark forum topics found, 2 technically relevant (1 skipped: OpenClaw/
  NemoClaw onboarding Docker pull timeout — tool-specific bug, not GB10-specific).
- 2 new sources registered (Batch 72). 3 topic IDs added to processed_topics.txt (total
  now 606).
- **Headline finding 1:** spark-comfyui — self-healing ComfyUI lifecycle manager for DGX
  Spark. Single script (install/run/update/doctor/status/tune/service/backup/restore/
  reset/recipe) with GB10-specific features: SageAttention sm_121 compile + runtime
  verification, UMA get_free_memory patch (same fix as S-forum-comfyui-optimized), NVFP4
  kernel verification, stuck-clock (power-controller wedge) detection under load,
  TRITON_PTXAS_PATH fix (triton#10331), full containerization, MiniMax-H3 out-of-the-box,
  recipes with sha256 model manifests. Confirmed on ASUS Ascent GX10. [conjecture].
- **Headline finding 2:** DSV4-Flash-0731 b12x build hangs at 92% safetensors load on 2×
  Spark — InstantTensor hybrid draft loader mod is the trigger. Workaround:
  INSTANTTENSOR_DRAFT_LOADER=instanttensor. Upstream bug: fastsafetensors ParallelLoader
  broadcasts on group.WORLD, deadlocking PP-scoped draft loads (vllm-project/vllm bug).
  sparkrun auto-determines NCCL config as alternative. 5 users in thread. [conjecture].
- Pages touched: containers-and-tooling (spark-comfyui [conjecture]), engines
  (DSV4-0731 b12x hang + fastsafetensors PP deadlock + sparkrun alternative
  [conjecture]), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-15 (Batch 71)
- 5 new NVIDIA DGX Spark forum topics found, 3 technically relevant (2 skipped: DIY cooling
  fans — accessory modding; beginner deploy question — resource recommendations).
- 3 new sources registered (Batch 71). 5 topic IDs added to processed_topics.txt (total
  now 603).
- **Headline finding 1:** GB10 spontaneous reboots after July 2026 firmware bundle — GSP
  health check fail (kgspHealthCheck_TU102), NVRM assert flood (gpu_user_shared_data.c:373),
  Xid 120 GSP task exception (supervisor timer interrupt). sbsa_gwdt watchdog action=1
  (DGX OS default) panics after GPU wedges → auto-reboot at ~2h intervals. Root cause:
  fwupd capsules (EC + SBIOS) applied on warm reboots without AC power disconnect. Fix:
  full AC power disconnect → 24+ hours clean. Extends existing power-controller wedge
  pattern to routine firmware updates. Dell Pro Max EC versioning differs from FE.
  [conjecture].
- **Headline finding 2:** Meta Muse Glimmer 30B dense model on DGX Spark — llama.cpp
  UD-Q6_K_XL + DFlash: 44.6 tok/s @ d0, 26.7 @ d8192 (tg128 c1). vLLM NVFP4 + DFlash:
  18.65 tok/s agg (2.42× BF16). vLLM DFlash broken (DFlashMuseGlimmerAssistantModel
  missing); llama.cpp/SGLang DFlash work. Tool-calling BFCL 10-12% — multi-tool
  serialization fails across all runtimes. Controlled A/B: 20/20 single-tool, DFlash
  4.08× speedup. Model sensitive to reasoning truncation. NEW model page created.
  [conjecture].
- **Headline finding 3:** Pilco-mmbridge dedicated thread — detailed vLLM recipes for
  DSV4-Flash + Qwen3.5-9B vision co-hosting on 2× Spark. Key tuning finding:
  `--kv-cache-memory-bytes` explicit allocation is the critical knob for stable
  co-hosting with DSpark spec decode. Initial recipe OOMs → final: 11.9 GB KV for DSV4,
  367 MB for Qwen vision at 0.05 util. Image persistence bug found+fixed. FE-only tested.
  [conjecture].
- Pages touched: platform-gb10 (GSP firmware reboot [conjecture]), models/muse-glimmer
  (NEW — full model page), engines (Pilco-mmbridge detailed recipes + --kv-cache-memory-bytes
  tuning [conjecture]), benchmarks (4 new [conjecture] rows: Muse Glimmer ×4 configs),
  sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-14 (Batch 69)
- 3 new NVIDIA DGX Spark forum topics found, all technically relevant.
- 3 new sources registered (Batch 69). 3 topic IDs added to processed_topics.txt (total
  now 597).
- **Headline finding 1:** Silent idle hard lockup — LPI-3 deep-idle wake failure on
  ASUS GX10. ~97% memory free, zero GPU workload, zero forensic trace (no panic/OOM/
  Xid/hung_task). SoC descending into deepest idle state (LPI-3) at moment of freeze.
  Only happens at idle, never under load. 7+ occurrences. Fourth distinct GB10 freeze
  mechanism: (1) OOM/UVM livelock, (2) thermal shutdown, (3) power-controller wedge,
  (4) idle deep-state wake failure. NVIDIA staff confirms OOM freeze is known but
  idle variant is different. Also: embedding models via transformers show UMA-specific
  memory leak absent on x86. [conjecture].
- **Headline finding 2:** MT7925e WiFi mesh network incompatibility — auth loop (8+
  retries) specific to mesh WiFi equipment, works with simple router. CX-7 DAC cable
  EMI also causes WiFi auth failures (cheap unshielded DAC → fix: outermost port or
  recommended shielded DAC). Both MT7925e and CX-7 on same compact SoC board.
  [conjecture].
- **Headline finding 3:** Sparkup — Ansible provisioning tool with `spbm` firmware
  module for whole-system power telemetry. Corroborates [reported] finding that
  nvidia-smi accounts for only 12-27% of real GB10 power draw. Prometheus + Grafana
  observability stack. [conjecture].
- Pages touched: platform-gb10 (idle LPI-3 lockup [conjecture], WiFi mesh + CX-7 DAC
  EMI [conjecture], Sparkup spbm power telemetry corroboration [conjecture]),
  containers-and-tooling (Sparkup Ansible tool [conjecture]), sources/README, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-14 (Batch 68)
- 4 new NVIDIA DGX Spark forum topics found, all technically relevant.
- 4 new sources registered (Batch 68). 4 topic IDs added to processed_topics.txt (total
  now 594).
- **Headline finding 1:** Smart plug + Auto Boot is the only viable power-management
  solution for multi-node Spark clusters — multiple independent users converge on this
  pattern. 4-node cluster idles 238-260W, 800W+ during inference. Clock cap at 1400 MHz
  saves ~200W for ~5-10% speed loss. Corroborates existing No-WoL and sleep-disabled
  findings. **[reported]** (4 independent users agree).
- **Headline finding 2:** NVIDIA Nemotron-3.5-Lightning-30B-A3B-NVFP4-DFlash on single
  Spark — 78.5 tok/s target, 90.7 tok/s DSpark (+15.6%), 120+ tok/s via sparkrun.
  Hybrid Mamba-2+MoE+Attention, 3B active, ~21 GiB, 1M ctx. Tool-eval 77-80/100
  (vs Qwen3.6-35B 100/100) — throughput-oriented, not agentic-tier. [conjecture].
- **Headline finding 3:** DSV4-Flash-0731 on mainline llama.cpp single Spark — 19.7 tok/s
  single-stream, 52 tok/s @ 4 concurrent, 131K ctx/slot. IQ3_XXS hits reproducible
  throughput dip at c=4 (47.6→26.3). KV q8_0 garbles output (llama.cpp can't represent
  DSV4 native mixed fp8/fp4 KV). coder543 ds4 comparison: DSpark 0% accept in
  continuous-batch path. [conjecture].
- **Headline finding 4:** DSV4 Flash + Qwen3.5-9B vision co-hosting on 2× Spark at 0.75
  util — corroborates memory-starved co-hosting finding. stu.miller uses 3rd Spark for
  vision (offload pattern). Pilco-mmbridge text-to-multimodal bridge + MixedInt4-AutoRound
  vision quant. [conjecture].
- Pages touched: platform-gb10 (smart plug power management [reported], cluster power
  draw data), models/nemotron-3 (Nemotron-3.5-Lightning-30B-A3B [conjecture]), llama-cpp-rpc
  (DSV4-0731 mainline llama.cpp recipe + IQ3_XXS c=4 dip + KV q8_0 garble + ds4 comparison
  [conjecture]), engines (DSV4+Qwen vision co-hosting + Pilco-mmbridge [conjecture]),
  benchmarks (3 new [conjecture] rows), sources/README, index, log.
- Evidence promotion: smart plug power management pattern → **[reported]** (4 independent
  users — CosmicRaisins, jetspark, mashie, peter.h177 — converge on identical solution,
  corroborating existing No-WoL [conjecture] and sleep-disabled [conjecture] findings).
  All other findings [conjecture].

## Forum ingest 2026-08-13 (Batch 67)
- 4 new NVIDIA DGX Spark forum topics found, all technically relevant.
- 4 new sources registered (Batch 67). 4 topic IDs added to processed_topics.txt (total
  now 590).
- **Headline finding 1:** GLM-5.2 unpruned QuantTrio Int4-Int8Mix on 4× Spark — 27 tok/s
  single / 52.5 tok/s @c4, 200K context. Second independent reproduction of tonyd2wild
  recipe (52.5 vs 53.5 c4). Major build gotcha: `VLLM_APPLY_PRESET_PRS` silently merges
  rebased PR branches into pinned vLLM build → fp8_ds_mla KV page-padding crash
  (`shape '[N, 64, 576]' is invalid`). Fix: build PURE ref with `VLLM_APPLY_PRESET_PRS="0"`.
  Agent concurrency monitoring: queue depth > token flow > acceptance rate. [conjecture].
- **Headline finding 2:** tool-eval-bench v2.5.0 scores 5-8 pts lower than v2.0.1 —
  version comparability is broken. Tool author (serapis) confirmed. DragonScale: new
  deterministic agentic-coding benchmark (no LLM judge, builds Flappy Bird game).
  Qwen3.6-35B-A3B scored 96.5, above all cloud models except DSV4-Flash GA (98.75).
  DSV4-Flash-0731 on 2× Spark: 85/100 hardmode v2.5.1. [conjecture].
- **Headline finding 3:** MTP speculation helps more on UMA, not less — but expert-union
  width grows with draft depth on MoE. Low depth (1-3) wins, high depth (6) collapses
  (44.8→44.5→29 tok/s on 122B MoE). Challenges "MTP is net loss on UMA" assumption.
  [conjecture].
- **Headline finding 4:** MiniMax-H3 one-click deploy with Sol-Attention acceleration
  on DGX Spark — Sol-Attn pre-patched for sm_121, ~6 min for 5s 720p video, 12 ComfyUI
  workflows, install wizard. [conjecture].
- Pages touched: models/glm-5.2 (200K 4× recipe + preset-PR drift + concurrency
  monitoring [conjecture]), containers-and-tooling (MiniMax-H3 Sol-Attn [conjecture]),
  engines (tool-eval-bench versioning + DragonScale + DSV4 tool-eval + MTP-on-UMA
  [conjecture]), benchmarks (2 new [conjecture] rows), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-12 (Batch 66)
- 6 new NVIDIA DGX Spark forum topics found (4 technically relevant, 2 skipped: power
  supply ordering — buying advice; crash/reboot — no durable findings, support/RMA).
- 4 new sources registered (Batch 66). 6 topic IDs added to processed_topics.txt (total
  now 586).
- **Headline finding 1:** GB10 fan controller tied to SoC power draw, not thermal
  sensors — fans stop when display blanks (DPMS off) or headless, overheating with no
  fan response at 52-56°C. 3 independent confirmers (DGX Spark FE + 2× ASUS GX10).
  Root cause: fan controller responds to SoC power draw; USB load ≥5W or VNC with
  active app spins fans back up. NVIDIA staff engaged. **[reported]** for core symptom.
  Distinct from EC fan-curve regression (different mechanism). Workarounds: `xset -dpms`,
  VNC with active app, sustained USB load.
- **Headline finding 2:** Driver 595.58.03 / CUDA 13.2 not yet supported on DGX Spark
  — NVIDIA staff confirms; may jump to CUDA 13.3. [conjecture].
- **Headline finding 3:** OpenGauntlet — 31 conversational LLMs benchmarked on single
  DGX Spark (GPT-5.4 judge, TTFT + tok/s at 512/2048/8192 contexts). vLLM cold-start
  376s vs SGLang 151s; vLLM 7/7 arch coverage vs SGLang 4/7; Q4_K_M GGUF TTFT 35s on
  UMA for 31B dense; NVFP4 MoE 37-43 tok/s; sglang confirmed on GB10. 18-row table.
  [conjecture].
- **Headline finding 4:** TensorRT-LLM one-forward-pass readout — extraordinary
  unverified claim (1,014 tokens in 92.3ms on 3× Spark). Self-described "vibe-coded."
  [conjecture] with explicit caveat — do not cite as benchmark.
- Pages touched: platform-gb10 (fan DPMS [reported] + driver 595 [conjecture] + TRT-LLM
  readout [conjecture]), engines (OpenGauntlet section [conjecture]), benchmarks (18 new
  [conjecture] rows), sources/README, index, log.
- Evidence: fan DPMS symptom [conjecture]→[reported] (3 independent confirmers). All
  other findings [conjecture]. No evidence promotions past [reported].

## Forum ingest 2026-08-12 (Batch 65)
- 4 new NVIDIA DGX Spark forum topics found (3 technically relevant, 1 skipped: NIM-vs-Eugr
  question with no replies/data).
- 3 new sources registered (Batch 65). 4 topic IDs added to processed_topics.txt (total now 580).
- **Headline finding 1:** MiniMax-M3 NVFP4 (official `nvidia/MiniMax-M3-NVFP4`) on 4× Spark —
  1M context (1,177,344-token KV pool via 4-bit packed nvfp4 KV), ~31 tok/s decode with EAGLE3,
  native vision + tool-calling. Major bug: NVFP4-Marlin MoE path drops SwiGLU-OAI activation
  params (gemm1_alpha 1.702 / gemm1_beta 1.0 arrive at Marlin kernel as defaults 1.0/0.0) → all
  57 MoE layers compute wrong activation → silent garbage output. 3-file param-threading fix
  documented (config.py + nvfp4.py + modelopt.py). Sibling of #46816/#47552 on NVFP4 quant-config
  chain. [conjecture].
- **Headline finding 2:** Cross-engine single-Spark field notes — identical harness across
  Ollama and vLLM for 4 models. Ollama does not batch (8× agg = single-stream); vLLM 289-313
  tok/s at 8 concurrent. NVFP4 1.1 tok/s on vanilla vLLM (emulation fallback) vs 77.1 tok/s with
  FlashInfer-CUTLASS — 70× from kernel path. MTP nst=2 beats 4. nvidia NVFP4 Gemma-4-26B-A4B
  30.3 tok/s w/o MTP vs Q4_K_M GGUF 49.6 — "obvious A/B makes NVFP4 look slower than it is."
  [conjecture].
- **Headline finding 3:** GLM-5.2 official NVFP4 on 8× Spark TP=8 — 25 tok/s decode, 256K context,
  tool-eval-bench v2.5.1 score 93/100 (highest reported for GLM-5.2 on GB10). Official
  `nvidia/GLM-5.2-NVFP4` via eugr spark-vllm-docker. [conjecture].
- Pages touched: models/minimax (M3 NVFP4 4× 1M recipe + SwiGLU-OAI bug [conjecture]),
  models/glm-5.2 (official NVFP4 8× recipe + tool-eval 93 [conjecture]), models/gemma-4
  (NVFP4 vs Q4_K_M GGUF gap + MTP nst=2>4 [conjecture]), engines (Ollama-vs-vLLM batching +
  NVFP4 missing-kernels 70× + prefill not bottleneck [conjecture]), benchmarks (6 new
  [conjecture] rows), sources/README, index, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-11 (Batch 64)
- 2 new NVIDIA DGX Spark forum topics found, both technically relevant.
- 2 new sources registered (Batch 64). 2 topic IDs added to processed_topics.txt (total now 576).
- **Headline finding 1:** nvidia-conf-xconfig.service recovery on DGX Spark FE — the
  `nvidia-conf-xconfig` package's systemd unit is required by GDM (RequiredBy=
  systemd-logind.service); if the package or `/etc/apt/sources.list.d/spark.sources`
  is lost, apt can't locate it. Recovery: extract spark.sources + GPG key from System
  Recovery image. Corroborates existing apt-upgrade-breaks-driver finding. [conjecture].
- **Headline finding 2:** vLLM `--runner pooling` enables embedding + reranking models
  on single Spark with tiny memory footprint — nomic-embed-text-v1.5 at gpu_memory_utilization
  0.05, bge-reranker-base at 0.015. Co-hostable with main LLM for RAG (unlike vision
  models). BAAI/bge-m3, Nemotron-3-Embed-8B/1B, Qwen 0.6B recommended. Qdrant + Open
  WebUI. Working YAML recipes documented. [conjecture].
- Pages touched: platform-gb10 (nvidia-conf-xconfig recovery [conjecture]), engines
  (embedding+reranking RAG recipes [conjecture]), sources/README, log.
- All [conjecture] — single-source forum threads. No evidence promotions.

## Forum ingest 2026-08-11 (Batch 63)
- 5 new NVIDIA DGX Spark forum topics found (3 technically relevant, 2 skipped: MSI
  EdgeXpert driver mixup, NVIDIA Sync Safari link issue).
- 3 new sources registered (Batch 63). 5 topic IDs added to processed_topics.txt (total now 574).
- **Headline finding 1:** GPU clock energy-efficiency sweep on 2× Spark TP=2 — 17-point
  clock sweep (400-2400 MHz) shows decode flat 47-51 tok/s from 1400-2400 (bandwidth-bound);
  best energy ROI band 1400-1800 MHz (~1350 Wh/1M tokens vs 1688 uncapped = 25% better for
  ~3% speed loss). nvidia-smi accounts for only 12-27% of real GB10 power draw. Prefill
  compute-bound (~14% penalty at 1400). Stock cooling throttles at 2100 MHz (repaste unlocks
  2500+). Strengthens existing [reported] clock-cap finding with energy dimension.
- **Headline finding 2:** LMCache 0.5.3 MP mode deadlocks with aidendle94 DS4F fork —
  vLLM 0.11.x fork IPC surface incompatible with LMCache 0.5.3 (targets vLLM 0.18/0.20+);
  no LMCache version matches both fork's IPC AND DS4F hybrid KV. Status: open [conjecture].
- **Headline finding 3:** KAT-Coder-V2.5-Dev-MTP-int4-AutoRound-SAR — Spark AutoRound int4
  quant with Qwen3.6 MTP headers grafted onto KAT Coder v2.5 Dev. 85+ t/s accepted on Asus
  GB10. Tool-eval 84 vs Ornith 87. Benchmark signals mixed [conjecture].
- Pages touched: platform-gb10 (clock energy sweep [reported]), engines (LMCache IPC
  deadlock [conjecture]), quantization-on-gb10 (KAT Coder AutoRound [conjecture]),
  models/qwen (KAT Coder MTP graft [conjecture]), sources/README, log.
- No evidence promotions (clock-cap finding already [reported]).

## Forum ingest 2026-08-10 (Batch 62)
- 3 new NVIDIA DGX Spark forum topics found (2 technically relevant, 1 skipped: Qwen3.8-27B
  open-weights announcement — no GB10-specific findings).
- 2 new sources registered (Batch 62). 3 topic IDs added to processed_topics.txt (total now 569).
- **Headline finding 1:** CRS804-4DDQ confirmed for 8× Spark clusters — CX-7 port architecture
  clarified (each QSFP56 port backed by 1× PCIe5 x4 ~109Gbps; breakout combines both x4 for
  full 200G; manual 200G port speed required). Highest-density switch for GB10 (1.6T).
- **Headline finding 2:** DSV4-Flash-0731 DSpark on single GB10 via llama.cpp — 31 t/s on code,
  first public dflash-format drafter. KLD quant ladder shows binary trade on 121GB: IQ3_XXS+DSpark
  (31 t/s, KLD 0.24) vs Q3_K_XL (9 t/s, KLD 0.106), no intermediate viable.
- Pages touched: multinode-tp-and-networking (CRS804 8× Spark [conjecture]), llama-cpp-rpc
  (DSV4-Flash-0731 DSpark single-node recipe [conjecture]), benchmarks (1 new [conjecture] row),
  sources/README, log.
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-08-10 (Batch 61)
- 8 new NVIDIA DGX Spark forum topics found (4 technically relevant, 4 skipped: DGX Spark 2
  speculation, NVIDIA Sync Tailscale security advisory on macOS, NVMe AES-256 confirmation, ISO
  download logistics).
- 4 new sources registered (Batch 61). 8 topic IDs added to processed_topics.txt (total now 566).
- **Headline finding 1:** Kernel 6.17.0-1029-nvidia one-way RDMA regression — ib_write_bw
  craters to ~13 Gb/s in one direction (vs ~111 Gb/s healthy) on GB10. Two independent users
  confirm (Claesbas, foogitiff). Fix: rollback to 6.17.0-1026-nvidia. Pins the existing
  kernel-6.17 RoCE regression to a specific build. **[reported]** promotion (2 independent
  sources agree).
- **Headline finding 2:** vllm-snapshot plugin — fast model suspend/restore for vLLM on GB10.
  vLLM sleep level-2 wake re-runs reload_weights (82s safetensors / 30min instanttensor —
  processing wall, not disk I/O). Plugin snapshots weights byte-for-byte, restores via
  cudaMemcpy in ~1.6s / ~9s full swap. Direct consequence of UMA (host RAM = GPU memory).
- **Headline finding 3:** DSV4-Flash-0731 GGUF (Unsloth) — UD-Q8_K_XL 162GB lossless, UD-IQ2_M
  runs on single Spark via llama.cpp with --no-repack flag.
- **Headline finding 4:** DSV4-Flash-0731 DSpark packaged for sparkrun — 58 tps on 2× Spark
  (packaging derivative of existing tonyd2wild recipe, consistent with 55.4 mean / 66.1 peak).
- Pages touched: multinode-tp-and-networking (kernel-1029 RDMA regression [reported]), engines
  (vllm-snapshot plugin [conjecture], DSV4-0731 GGUF + sparkrun [conjecture]), llama-cpp-rpc
  (DSV4-0731 GGUF recipe [conjecture]), benchmarks (1 new [conjecture] row), sources/README,
  index, log.
- Evidence promotion: kernel-6.17.0-1029 RDMA regression [conjecture] → [reported] (2
  independent users confirm identical symptoms + fix).

## Forum ingest 2026-08-08 (Batch 60)
- 2 new NVIDIA DGX Spark forum topics found, both technically relevant.
- 2 new sources registered (Batch 60). 2 topic IDs added to processed_topics.txt (total now 558).
- **Headline finding:** DSV4-Flash-0731-vision — first reported vision-enabled DSV4-Flash-0731
  deployment on 2× DGX Spark. FlyCockpit vLLM plugin (DeepEncoderV2 tower + 40 MB projector).
  Key discovery: vision wrappers that intercept the backbone's forward path silently kill DSpark
  speculative decoding (acceptance 1-15% → 50-64% after wrapper-transparency fix). General pattern
  for any vLLM vision wrapper + spec-decode combo. Throughput 40-50 tps post-fix (~20-30% below
  non-vision DSpark baseline). Also: `chat_template_kwargs: {"thinking": false}` required for
  image requests, tiles=2 token math (257/769/1281), screenshot-specialist vision quality.
- Pages touched: engines (DSV4-Flash-0731-vision — 7 new [conjecture] findings: plugin recipe,
  wrapper-transparency bug, thinking:false for images, tiles=2 token layout, vision quality
  assessment, throughput, webbrain-one 9 GB NVFP4 variant), benchmarks (1 new [conjecture] row),
  sources/README, index, log.
- Topic 379391 (vLLM deep-dive blog posts by swesty): source registered for provenance; no new
  wiki content — the OP links to external blog posts without containing specific durable GB10
  findings (flags, env vars, error strings, tok/s numbers) beyond what's already in the KB.
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-08-08 (Batch 59)
- 2 new NVIDIA DGX Spark forum topics found (1 technically relevant, 1 skipped: system updates question).
- 1 new source registered (Batch 59). 2 topic IDs added to processed_topics.txt (total now 556).
- **Headline finding:** Kimi K3 Coder REAP-320 MXFP4 on 8× GB10 — first reported Kimi K3 variant
  on DGX Spark. Decode 21-30 tok/s (tg1500), peak 35 tok/s, prefill 541-686 tok/s via llama-bench.
  REAP pruned variant "loops a lot" (quality issue); full K3 needs 16× GB10.
- Pages touched: models/kimi-k3 (NEW), benchmarks (1 new [conjecture] row), sources/README, index, log.
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-08-07 (Batch 58)
- 3 new NVIDIA DGX Spark forum topics found (2 technically dense, 1 marginal accessory thread).
- 3 new sources registered (Batch 58). 3 topic IDs added to processed_topics.txt (total now 554).
- **Headline finding:** SparkRing — first custom RDMA collective layer (SIRCL) for GB10, bypassing
  NCCL entirely for inference-critical paths on a 4-node switchless ring. GLM-5.2 at 19-20 tok/s C1
  / 50-63 tok/s C8 aggregate. Fundamental alternative to the NCCL env-var workarounds for switchless
  ring topologies (S-forum-6x-ring-rdma). Also: GLM-5.2 indexer weight bug (57/78 layers run top-k
  on uninitialized memory), VLLM_NVFP4_MLA_PER_TOKEN_SCALE=1, SparkCache DCP4-sharded NVMe KV cache.
- Pages touched: models/glm-5.2 (SparkRing section — SIRCL, MXFP4-Experts-GPTQ + MXFP8-NVFP4-NF3
  hybrid + EXL3 quants, indexer bug, peer ordering bug, CUDA graph lock, SparkCache, Terry01
  reproduction — all [conjecture]), multinode-tp-and-networking (SIRCL custom collective layer —
  first reported NCCL bypass for inference on GB10 [conjecture]), benchmarks (4 new [conjecture]
  rows: GLM-5.2 SparkRing ×3 configs, DSV4-Flash-0731 llama.cpp IQ2_M single-node),
  llama-cpp-rpc (DSV4-Flash-0731 UD-IQ2_M single-node 16.2 tok/s [conjecture]),
  sources/README, index, log.
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-08-07 (Batch 57)
- 7 new NVIDIA DGX Spark forum topics found (5 technically relevant, 2 skipped: recovery help, kernel panic RMA).
- 5 new sources registered (Batch 57). 7 topic IDs added to processed_topics.txt (total now 551).
- **Headline finding:** GB10 may serialize CUDA contexts — `cuInit()` returns
  `CUDA_ERROR_NO_DEVICE` for a second process while another holds a context (Compute Mode=Default,
  20-60s teardown delay, vLLM sleep doesn't release context, Xid 119 interaction). If confirmed,
  single-tenant-per-node is enforced at the driver level, not just by memory.
- Pages touched: platform-gb10 (single-CUDA-context limitation [conjecture], CX-7 27W warning
  confirmed benign by NVIDIA staff [conjecture], MiniMax-H3 thermal freeze + unit-to-unit
  variation [conjecture]), containers-and-tooling (vLLM NGC forward-compat ceiling on 580.173.02
  — caps at CUDA 13.1, no tag supports Qwen3.6 + forward-compats [conjecture]; Unsloth Docker
  recipe — pytorch:25.10-py3, torchao conflict fix [conjecture]), sources/README, index, log.
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-08-06 (Batch 56)
- 5 new NVIDIA DGX Spark forum topics found, all technically relevant.
- 5 new sources registered (Batch 56). 5 topic IDs added to processed_topics.txt (total now 544).
- Pages touched: engines (DSV4-Flash-0731 on ds4 CUDA engine v0.5.4 — single Spark 40 tok/s
  IQ2XXS + DSpark MTP k=2, 131K ctx; 1M ctx fits ~107GB with kv-disk-dir offload; full env
  vars documented — [conjecture]), platform-gb10 (non-DGX OS on Spark — ACPI not DT, Fedora 44
  confirmed on GX10, NVIDIA-maintained kernels needed, Workbench/Field Diagnostics Ubuntu-only
  — [conjecture]; vLLM x86_64 Docker → QEMU emulation on Grace CPU → 3.7 tok/s, CUDA 13
  library pathing, pip overwrites +nv PyTorch — [conjecture]), multinode-tp-and-networking
  (MikroTik CRS812 DDQ 4-node practical setup — disable auto-neg for 200G DAC, static RoCE
  IPs, MTU 9000, AI-generated RouterOS commands unreliable — [conjecture]), benchmarks (3 new
  [conjecture] rows: DSV4-Flash-0731 ds4 CUDA 40 tok/s, Laguna-S-2.1 ModelOpt W4A4 28 tok/s,
  Qwen2.5-Coder-32B QEMU emulation 3.7 tok/s baseline), models/laguna-s-2.1 (ModelOpt NVFP4
  W4A4 variant — 28 tok/s, 88/100 tool calls; model retired, recorded for completeness —
  [conjecture]).
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-08-06 (Batch 55)
- 2 new NVIDIA DGX Spark forum topics found, both technically relevant.
- 2 new sources registered (Batch 55). 2 topic IDs added to processed_topics.txt (total now 539).
- Pages touched: platform-gb10 (CX-7 connection raises idle temp ~10°C, 17 W/node — 3rd
  independent source corroborating CX-7 active thermal penalty → **[reported]** promotion:
  S-forum-cx7-hotplug + S-forum-cx7-dac-power + S-forum-cx7-idle-temp), containers-and-tooling
  (MiniMax-H3 video generation via ComfyUI on single Spark — i2v 174s, t2v 143s, r2v 215s for
  5s/0.2M; ~235s at 768² with easycache+SageAttention KJ nodes; 432s for 10s — [conjecture]),
  benchmarks (5 new [conjecture] MiniMax-H3 video diffusion rows).
- Evidence promotion: CX-7 active thermal/power penalty [conjecture] → [reported].

## Forum ingest 2026-08-05 (Batch 54)
- 6 new NVIDIA DGX Spark forum topics found (4 technically relevant, 2 skipped: Tailscale
  regression, IsaacLab robotics question).
- 4 new sources registered (Batch 54). 6 topic IDs added to processed_topics.txt (total now 537).
- Pages touched: engines (DSV4-Flash-0731-DSpark draft loader weight-mapping bug —
  shared_experts.w1/w3 → gate_up_proj missing, 12 tensors silently dropped, fix +69% tok/s;
  SSE streaming measures steps/s not tok/s; draft quant-config inheritance bug vLLM PR #49133),
  models/qwen (Macaron-V1-Tall 50B Qwen+LoRA on single Spark 25-27 tok/s bf16, MTP +2%
  throughput despite 71.5% acceptance, tool-eval router <base; Qwen3.6-35B-A3B bf16 TP=2
  Ray decode stall to 0.1 tok/s), benchmarks (3 new [conjecture] rows), containers-and-tooling
  (--no-bf16-vae flag for LTX-2.3 audio VAE on spark-comfyui).
- All [conjecture] — single-source forum. No evidence promotions.

## Forum ingest 2026-07-08
- 184 total NVIDIA DGX Spark forum threads processed (20 high-priority + 164 remaining from main + projects forums).
- See `sources/README.md` → Forum sources (Batch 1 + Batch 2) for all registered sources.
- Pages touched: platform-gb10 (power wedge, TMA, thermal, GSP timeout, drivers, display/USB/WiFi issues),
  quantization-on-gb10 (AWQ vs NVFP4, MXFP4, CUTLASS failure, distributed quant),
  multinode-tp-and-networking (CX-7 PCIe, MikroTik switches, 2D parallelism, DDP, SKU mixing),
  engines (Atlas, ds4/DwarfStar, DFlash block-spec), containers-and-tooling (community tools),
  models/mimo-v2.5 (TP=3 virtual-head padding, community recipes), models/minimax (4× recipes,
  llama.cpp RPC, MSA, AWQ/MXFP4/NVFP4-KV variants), benchmarks (21 forum-reported rows),
  llama-cpp-rpc (M3 RPC), roadmap (4 new open problems).

## Forum ingest 2026-07-09
- 160 new NVIDIA DGX Spark forum threads processed (~48 new sources, Batch 3).
- See `sources/README.md` → Batch 3 forum sources for all registered sources.
- Pages touched: platform-gb10 (CX-7 bricking, SDPA corruption, SageAttention, vLLM 26.06 broken,
  OOM hang fix, fwupd mismatch, UMA bandwidth, torchaudio), multinode-tp-and-networking (NCCL 2.30.4,
  SGLang RDMA passthrough, SGLang traps, CUTLASS MoE OOM, 4-node mesh, MTP NEXTN),
  quantization-on-gb10 (KVarN, Spark Auto Round, KV benchmarks, TurboQuant, STREAM LOADING,
  ModelOpt CPU-bound, vLLM regression, MTP math, heterogeneous quant), engines (DDTree, STREAM LOADING,
  SM121 kernel guide, vLLM regression), containers-and-tooling (Nunchaku, ComfyUI, llama.cpp container,
  QAT models, Mistral OOM, torchaudio), models/mimo-v2.5 (DFlash 22→67, v0.24.0 DFlash+NVFP4 KV),
  models/minimax (W4A16-GPTQ corroborated, M2.5 4× SGLang), models/nemotron-3 (Super MTP, Ultra 550B,
  ABI fix), benchmarks (17 new forum-reported rows), roadmap (4 new open problems).

## Forum ingest 2026-07-10
- 10 new forum topics found (8 technically relevant, 2 skipped as social/buying advice).
- 7 new sources registered (Batch 4). 8 topic IDs added to processed_topics.txt (total now 344).
- Pages touched: quantization-on-gb10 (FLUX.2 NVFP4 W4A4 compute, weight-only vs activation-quantized),
  platform-gb10 (UMA mmap double-allocation OOM workaround, TCG OPAL/UEFI corruption, display controller
  pixel clock, ONNX GPU discovery), containers-and-tooling (llama-benchy, cluster dashboard, Sunshine
  RDP, FLUX.2 Images-API server), benchmarks (3 new forum-reported rows: MiniMax-M2.1, GLM-4.7-Flash,
  FLUX.2-dev image gen), models/mimo-v2.5 (topic 375923 already ingested in Batch 3).

## Forum ingest 2026-07-10 (Batch 5)
- 3 new forum topics found, all technically relevant.
- 3 new sources registered (Batch 5). 3 topic IDs added to processed_topics.txt (total now 349).
- Pages touched: models/mimo-v2.5 (SGLang 4× FP8 recipe, MTP OOM, NVFP4 MoE backend gap on SM121a,
  sampling params/Thought Loop mitigation), models/minimax (M2.7 NVFP4/AWQ/FP8 recipes on 2×/4× Spark,
  FlashInfer-CUTLASS beats CUTLASS, AWQ beats NVFP4 corroborated by 3 independent sources → [reported],
  Unsloth FP8 4× 36 tok/s), benchmarks (4 new LLM forum rows + 6 diffusion model image gen rows),
  quantization-on-gb10 (FlashInfer-CUTLASS stability, diffusion weight-only vs activation-quantized
  NVFP4 distinction), containers-and-tooling (DIFFUSERS_ATTN_BACKEND=_native_cudnn speedup,
  diffusion model benchmark table).

## Forum ingest 2026-07-11 (Batch 6)
- 3 new forum topics found, all technically relevant.
- 3 new sources registered (Batch 6). 3 topic IDs added to processed_topics.txt (total now 352).
- Pages touched: platform-gb10 (random shutdowns — thermal paste degradation, CPU hot-spot sensor
  blind spot, PDU fault variant, no WoL, Nsight Systems sudo requirement), multinode-tp-and-networking
  (3-node ring topology NCCL failure, cable mixing MTU mismatch, explicit SSH resolution for >2 nodes).

## Forum ingest 2026-07-12 (Batch 8)
- 1 new forum topic found, technically relevant.
- 1 new source registered (Batch 8). 1 topic ID added to processed_topics.txt (total now 355).
- Pages touched: platform-gb10 (GPU clock wedge follow-up — 5 min wait sufficient [reported],
  power-drain method [conjecture], PSU root-cause hypothesis [conjecture]).

## Forum ingest 2026-07-12 (Batch 9)
- 3 new forum topics found (2 technically relevant, 1 skipped as non-GB10-specific).
- 2 new sources registered (Batch 9). 3 topic IDs added to processed_topics.txt (total now 358).
- Pages touched: platform-gb10 (reboot power-cycle completion failure [conjecture]),
  multinode-tp-and-networking (CX-7 dual setup field report — third-party DAC, TCP ceiling ~16 Gb/s,
  DCGM on GB10, PSI OOM alerting, NetworkManager config, cluster tax metric).

## Forum ingest 2026-07-13 (Batch 10)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 10). 2 topic IDs added to processed_topics.txt (total now 360).
- Pages touched: quantization-on-gb10 (NVIDIA refreshed NVFP4 recipe, Qwen3/Qwen3.6 27B
  TensorRT-LLM errors [conjecture]), containers-and-tooling (nvidia-vfx no aarch64 wheel,
  NVIDIA confirmed no plans, ComfyUI RTX nodes broken [reported]).

## Forum ingest 2026-07-14 (Batch 12)
- 4 new forum topics found (3 technically relevant, 1 skipped as buying advice).
- 3 new sources registered (Batch 12). 4 topic IDs added to processed_topics.txt (total now 368).
- Pages touched: engines (TokenSpeed SM12x-stable engine — prefill +10-14% vs vLLM, decode behind
  70-74%, KV +25%, NCCL 2.30.4 mandatory, build recipe), containers-and-tooling (Spark Studio
  dashboard — live UMA monitor, pre-launch memory guard, agent auto-fix), attention-and-kv-cache
  (DSV4-Flash KV cache ~15 GB/1M tokens/node, CUDA graph memory profiling overhead),
  benchmarks (2 new forum-reported rows: DSV4-Flash TokenSpeed vs vLLM fork).

## Forum ingest 2026-07-14 (Batch 13)
- 1 new forum topic found (technically marginal — non-LLM model recommendation).
- 1 new source registered (Batch 13). 1 topic ID added to processed_topics.txt (total now 369).
- Pages touched: sources/README, log, index. No wiki page edits.
- Finding: ACE-Step v1.5 XL (music generation) runs on single Spark, fits comfortably in VRAM
  with companion 5Hz-LM-4B lyrics model. 3 independent users confirm (danielgbates, joey28,
  aostang) → would be [reported] if it warranted a page. Not ingested to wiki because: outside
  core LLM-inference scope (not vLLM/llama.cpp/sglang), no GB10-specific flags/env vars/errors/
  tok-s numbers/quant formats, and "fits in 121 GB" is trivially true for a single-model workload
  with no quantization. Source registered for provenance; no wiki page created.

## Forum ingest 2026-07-15 (Batch 14)
- 6 new forum topics found (4 technically relevant, 2 skipped: robotics RT kernel, RMA complaint).
- 4 new sources registered (Batch 14). 6 topic IDs added to processed_topics.txt (total now 375).
- Pages touched: models/qwen (Unsloth NVFP4 ~15% slower than nvidia on GB10 [reported] via 3
  independent benchmarks; flashinfer_b12x unavailable on stock vLLM; working Marlin+MTP recipe;
  W4A16 bypass hypothesis; quality parity), quantization-on-gb10 (Unsloth NVFP4 slower on GB10,
  flashinfer_b12x gap, W4A16 bypass), benchmarks (5 new forum-reported rows: nvidia vs Unsloth
  Qwen3.6-35B-A3B NVFP4 comparison), engines (multi-model co-hosting — vision+LLM on 2× Spark
  memory-starved, offload vision to separate machine [reported], multimodal front-end pipeline),
  platform-gb10 (HPC/slurm CPU P/E core topology, CX-7 switch config, Llama 3.2 3B finetune
  8× slower than benchmark — known FAQ).

## Forum ingest 2026-07-15 (Batch 15)
- 4 new forum topics found (3 technically relevant, 1 skipped: buying advice).
- 3 new sources registered (Batch 15). 4 topic IDs added to processed_topics.txt (total now 379).
- Pages touched: platform-gb10 (CX-7 hot-pluggable ports — not visible until cable connected,
  /etc/nvidia/cx7-hotplug-enabled, idle power doubles when active), engines (LLM + ComfyUI
  co-hosting — vLLM KV cache starves UMA, --gpu-memory-utilization 0.7-0.8, llama.cpp better
  for co-hosting, swapoff -a), multinode-tp-and-networking (~23 GB/s cross-node vs ~600 GB/s
  in-box bottleneck, MoE gains flatten past TP=4, FP8 training impossible on sm_121, Megatron
  caveats), models/qwen (Qwen3.5-397B on 8× GB10 31-35 tok/s, architecture comparison),
  benchmarks (1 new row: Qwen3.5-397B FP8 8× GB10).

## Forum ingest 2026-07-13 (Batch 11)
- 4 new forum topics found (2 technically relevant, 2 skipped as non-technical).
- 2 new sources registered (Batch 11). 4 topic IDs added to processed_topics.txt (total now 364).
- Pages touched: engines (easy-vllm code-agent harness, DSV4-Flash GB10 via jasl/vllm SM12x fork,
  torch 2.11+ ABI wall, MXFP4 MoE→MARLIN→UMA OOM, mem_watchdog+earlyoom),
  multinode-tp-and-networking (4-node CRS504 100G switch — 5-10% PP loss, zero decode loss,
  ~13 Gb/s measured traffic, $25 100G cable works), benchmarks (3 new forum rows: DSV4-Flash
  TP=4/TP=2, M3-AWQ+EAGLE TP=4), containers-and-tooling (easy-vllm tool),
  models/minimax (M3-AWQ+EAGLE on CRS504 corroborates existing benchmarks).

## Forum ingest 2026-07-16 (Batch 16)
- 3 new forum topics found (2 technically relevant, 1 skipped as buying advice/warranty/social).
- 2 new sources registered (Batch 16). 3 topic IDs added to processed_topics.txt (total now 382).
- Pages touched: engines (Colibri — pure-C expert-streaming engine for GLM-5.2 744B MoE on
  single Spark, 2.4-3.3 tok/s, O_DIRECT 9.69 GB/s, CACHE_ROUTE experimental routing),
  containers-and-tooling (ComfyUI Docker optimized for DGX Spark — CUDA 13.1 sm_121, SageAttention 2,
  double-VRAM bug fix copy=False + --disable-mmap, --disable-dynamic-vram, cudaMemGetInfo
  under-reports free UMA when co-resident CUDA process — psutil fix),
  platform-gb10 (cudaMemGetInfo under-reports free memory on UMA with co-resident process
  [conjecture]), benchmarks (1 new forum-reported row: GLM-5.2 744B Colibri expert streaming),
  roadmap (Colibri demonstrates expert-streaming approach in practice — bottleneck is attention
  not disk I/O).

## Forum ingest 2026-07-16 (Batch 17)
- 5 new forum topics found (4 technically relevant, 1 skipped: 376589 = buying advice "triple
  stack").
- 4 new sources registered (Batch 17). 5 topic IDs added to processed_topics.txt (total now 387).
- Pages touched: quantization-on-gb10 (NVFP4 meta-analysis — NVFP4 leaves ~half layers bf16 vs
  Int4 all-layers; TRT-LLM NVFP4 slower than GGUF Q4_K_M; bandwidth efficiency 42-48%; NVFP4 now
  operational via community Docker; FlashInfer 0.6.8.1 improvements [reported]),
  models/nemotron-3 (Ollama v0.30.x-v0.31.2 parser regression breaks Nemotron-3-Super on GB10,
  fix: downgrade to 0.24.0 [conjecture]; NVFP4 bandwidth efficiency 42-48% [reported]),
  multinode-tp-and-networking (ib_write_bw falsely reports >64 KiB RDMA WRITE failure on GB10 —
  fabric is fine; NCCL_NET_PLUGIN=none, NCCL_TOPO_FILE correction, RoCE NIC-offloaded counters,
  arp_ignore=1/arp_announce=2 [conjecture]),
  platform-gb10 (Ollama parser regression [conjecture], NVFP4 bandwidth efficiency 42-48%
  [reported]), engines (DSV4-Flash-DSpark-Abliterated source added),
  benchmarks (5 new forum-reported rows: Llama-3.3-70B NVFP4 vs GGUF Q4_K_M, Nemotron-3-Super
  NVFP4 1×/2×, DSV4-Flash-DSpark-Abliterated 50-60 tok/s).

## Forum ingest 2026-07-17 (Batch 18)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 18). 2 topic IDs added to processed_topics.txt (total now 389).
- Pages touched: benchmarks (3 new [conjecture] rows — GLM-5.2-Int4-Int8Mix on 8× GB10 TP8 DCP=1
  ~1,200 t/s prefill / 33–54 t/s decode; TP4+PP2 ~12 t/s MTP collapse; DCP4 decode-starvation
  scheduler), multinode-tp-and-networking (NCCL_BUFFSIZE 16 MB at TP8 [conjecture], TP4+PP2 wrecks
  MTP acceptance → 8% [conjecture], DCP4 decode starvation + decode-aware prefill scheduler
  [conjecture], draft_tensor_parallel_size=1 [conjecture]), quantization-on-gb10 (b12x W4A8 MoE
  backend — INT4 weights + INT8 activations via native FP8 CUTLASS [conjecture]; stale
  topk_indices_buffer in flashinfer SM120 sparse MLA PR#46994 [conjecture]; quantized NextN draft
  token mapping [conjecture]; VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 [conjecture]; Int4-Int8 mix
  quant [conjecture]), models/qwen (Bonsai 27B binary/ternary Qwen3.6-27B — hypothesis: faster
  decode on bandwidth-bound Spark dense, no GB10 benchmarks yet [conjecture]), roadmap (3 new
  open problems: v16+b12x W4A8 isolated contribution, Bonsai sm_121 kernel path, DCP4 scheduler
  on DCP1).

## Forum ingest 2026-07-17 (Batch 19)
- 5 new forum topics found, all technically relevant (USB2 fallback, firmware updates, OTA
  loop, ASUS GX10 firmware, host freeze during TP=2 prefill).
- 5 new sources registered (Batch 19). 5 topic IDs added to processed_topics.txt (total now 394).
- Pages touched: platform-gb10 (USB3 SuperSpeed PHY not registered → USB2 fallback [reported]
  via 7 independent users; MediaTek T-PHY no ACPI binding; new FE firmware EC/UEFI versions;
  DGX Dashboard OTA loop + nvidia-spark-ota-check diagnostic tool; ASUS GX10 v0103 PD firmware
  fixes thermals ~8-10 W lower [reported] + 4× link speed [conjecture]; total host freeze
  during heavy TP=2 prefill = thermal shutdown with zero forensic trace [conjecture]),
  multinode-tp-and-networking (ASUS PD firmware 4× link speed [conjecture]; host freeze
  during prefill = highest combined SoC power scenario [conjecture]),
  sources/README, index, log.

## Forum ingest 2026-07-11 (Batch 7)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 7). 2 topic IDs added to processed_topics.txt (total now 354).
- Pages touched: models/mimo-v2.5 (detailed 2-node renek recipe — driver KV pool diff, NCCL CGA buffer,
  MTP acceptance rates, enforce-eager at 160K, full env/config, 30-33 tok/s; tonyd615 repo 38 tok/s
  non-eager; synthetic vs real-world gap), platform-gb10 (first-boot WiFi SSID not broadcasting,
  QR→product page, monitor+keyboard workaround), attention-and-kv-cache (TRITON_ATTN_DIFFKV
  quantized KV guard), multinode-tp-and-networking (NCCL v2.30u1 CGA buffer), benchmarks (1 new
  forum-reported row: MiMo-V2.5-NVFP4 renek recipe 30-33 tok/s).

## Forum ingest 2026-07-18 (Batch 20)
- 1 new forum topic found, technically relevant (MTP lossless? — quality & prefix-cache bugs).
- 1 new source registered (Batch 20). 1 topic ID added to processed_topics.txt (total now 395).
- Pages touched: engines (MTP measurably affects output quality [conjecture] — tool-call bench ~5
  pts; ~40% speed vs ~2% quality hit on Qwen3.6-27B [conjecture]; vLLM+llama.cpp MTP+prefix-cache
  interaction bugs [conjecture]; DS4F prefix-batch 16384/MTP=4 → 70-75% acceptance [conjecture];
  "theory != deployment" practical-lossiness debate [conjecture]), roadmap (new open problem:
  measure MTP quality impact & prefix-cache interaction on real Spark).

## Forum ingest 2026-07-19 (Batch 22)
- 3 new forum topics found, all technically relevant (EC firmware fan-curve regression, Nemo-RT
  voice agent, LiteLLM multi-model orchestrator).
- 3 new sources registered (Batch 22). 3 topic IDs added to processed_topics.txt (total now 400).
- Pages touched: platform-gb10 (EC firmware 0x0300xxxx breaks fan curve → 96-97°C ACPI zones,
  inaudible fans; EC isolates fan control from OS — fancontrol/pwmconfig/nvidia-settings can't
  override; fwupdmgr downgrade to 0x02004e18 fix; idle 60→32°C, load 35-37°C, 0% throttling,
  120-125W/node @ 95% GPU util; first reported EC firmware *regression* on Spark; fan control is
  EC-isolated, not OS-overridable [conjecture]; relationship to 0x03000508 "improves EC" update
  unresolved), containers-and-tooling (harinezumigel-llm-stack LiteLLM+vLLM orchestrator for
  single-Spark multi-model lifecycle management [conjecture]; thread surfaces sparkstation
  (kshetrajna12/sparkstation); Nemo-RT Community voice agent — VAD+STT+LLM(Qwen3-8B-FP8 via
  vLLM)+TTS on one GPU, OpenAI Realtime API-compatible, ~20 concurrent calls on Spark, native
  FP8 + arm64 build [conjecture]).

## Forum ingest 2026-07-19 (Batch 23)
- 3 new forum topics found, all technically relevant (NVIDIA Sync locale bug, ASUS GX10 thermal
  throttling corroborating the EC fan-curve regression, Inkling 975B/276B MoE announcement).
- 3 new sources registered (Batch 23). 3 topic IDs added to processed_topics.txt (total now 403).
- **Evidence promotion:** the EC firmware fan-curve regression (S-forum-ec-fan-rollback,
  originally [conjecture] in Batch 22) is **promoted to [reported]** — independently corroborated
  on a 3rd OEM SKU (ASUS GX10, S-forum-ec-fan-asus) with the same symptom fingerprint (ACPI zones
  96.6°C, fans N/A, SW/HW thermal slowdown counters). Three OEM SKUs now agree: Gigabyte, MSI FE,
  ASUS GX10.
- Pages touched: platform-gb10 (new Batch 23 section — ASUS GX10 thermal throttling corroborates
  EC fan-curve regression → [reported]; root-cause narrows from EC table to SoC/UEFI interaction
  via byte-identical fan-curve comparison 48%@85°C/54%@93°C/68%@95°C/100%@97°C [conjecture];
  first published GB10 fan-curve bytes [conjecture]; fwupdmgr downgrade unavailable for ASUS GX10
  [conjecture]; dgx-spark-fieldiag 2.0.4-1 ofed-scripts packaging bug [conjecture]; existing
  Batch 22 entry promoted [conjecture]→[reported] with corroboration note),
  multinode-tp-and-networking (NVIDIA Sync / Cluster Assistant fails "Software version" check on
  non-English locale — apt-cache policy parser breaks on localized "Installiert:"/"Installé :"
  labels → false "System Software Update Required"; workaround sudo update-locale
  LC_MESSAGES=en_US.utf8; hotfix pending [conjecture]), roadmap (3 new open problems: Inkling
  975B/276B MoE bring-up not yet characterized, EC fan-curve root-cause isolation, fieldiag
  ofed-scripts dependency gap).

## Forum ingest 2026-07-20 (Batch 24)
- 1 new forum topic found, technically relevant (6× GB10 cluster via MikroTik CRS812, b12x TP=6).
- 1 new source registered (Batch 24). 1 topic ID added to processed_topics.txt (total now 404).
- Pages touched: multinode-tp-and-networking (6× GB10 cluster via CRS812 — b12x backend enables
  non-power-of-2 TP=6 on most models; GLM-5.2 ~30 tok/s single-stream; cluster 800-1180 W peak;
  consistent with sublinear scaling between TP=4 and TP=8 [conjecture]), benchmarks (1 new
  [conjecture] row: GLM-5.2 6× TP=6 ~30 tok/s), roadmap (new open problem: does b12x enable
  arbitrary non-power-of-2 TP on GB10 — virtual-head padding previously required for TP=3).

## Forum ingest 2026-07-21 (Batch 26)
- 4 new forum topics found, all technically relevant.
- 4 new sources registered (Batch 26). 4 topic IDs added to processed_topics.txt (total now 408).
- **Headline finding:** FlashInfer `sparse_mla_sm120` mbarrier livelock on GB10/sm_121 — root-caused
  via cuda-gdb (mbarrier TRYWAIT spin-loop under cold-prefill), validated Triton workaround
  (FLASHMLA_SPARSE + sm12x patch, 560+ clean sessions, no throughput penalty). Major kernel bug
  for any sparse-MLA model on GB10. Flagged for priority hardware verification in roadmap.
- Pages touched: attention-and-kv-cache (FlashInfer livelock + Triton workaround),
  multinode-tp-and-networking (3-node full-mesh guide — CX-7 triangle, cross-connect port0↔port1,
  TP requires power-of-2, 3-node PP ~single-node speed, PP+MTP not supported, LMCache for KV
  node, NCCL mesh merged to main, fastsafetensors freeze, gpu_memory_utilization 0.8 for PP),
  containers-and-tooling (community images lag upstream vLLM 0.25.1/NCCL 2.30.7),
  platform-gb10 (EC firmware 0x00000500→0x00000507 silent failure, fwupdmgr get-results diagnostic),
  benchmarks (2 new [conjecture] rows: Qwen3.5-397B-A17B 3-node PP decode 12–14.4 tok/s + prefill
  912–1242 tok/s), roadmap (2 new open problems: FlashInfer livelock reproduction/fix, 3-node PP
  vs TP=2 overhead measurement).

## Forum ingest 2026-07-21 (Batch 27)
- 2 new forum topics found (1 technically relevant, 1 skipped as non-technical).
- 1 new source registered (Batch 27). 2 topic IDs added to processed_topics.txt (total now 410).
- Pages touched: platform-gb10 (sysfs thermal zone layout under load — zones 0/5 hottest at
  94.6 °C, GPU ~10 °C cooler than CPU; `tegrastats` Jetson Orin Nano binary works on GB10;
  GPU clock capping as thermal mitigation per wildpines.ai blog). All [conjecture].
- Skipped: topic 377428 (AirLLM theoretical parameter-ceiling speculation — no runs, no
  measurements, replies all jokes; adds nothing beyond known 128 GB unified memory / 4 TB SSD
  facts already in platform-gb10 and the Colibri expert-streaming approach in engines).

## Forum ingest 2026-07-22 (Batch 28)
- 10 new forum topics found (4 technically relevant, 5 skipped as social/buying/speculation,
  1 already covered by existing source).
- 4 new sources registered (Batch 28). 10 topic IDs added to processed_topics.txt (total now 420).
- Pages touched: platform-gb10 (UVM page-migration livelock — hard shutdown under sustained load,
  --gpu-memory-utilization 0.85-0.92 fix, platform firmware update, power cap, clock lock;
  GB10B scanout carveout allocation failure in Sway at 6K resolution; RealSense D435 USB
  disconnect fixed by July firmware), containers-and-tooling (sparkDash by MiaAI-Lab — second
  independent multi-Spark monitoring dashboard). All [conjecture].
- Skipped: 377602 (Motif-3-Beta model announcement, no GB10 specifics), 377626 (best model for
  3-node, pointer to existing M3 TP=3 docs), 372722 (buy vs rent, buying advice), 364493
  (Windows 11 ARM installation, not LLM inference), 377396 (Qwen 3.8 launch speculation).
- 376643 (Sparkrun webui by brainchillz) already covered by existing S-forum-sparkdash — same
  repo (brainchillz/sparkdash), just a different forum post. Marked processed, no new source.

## Forum ingest 2026-07-23 (Batch 30)
- 3 new forum topics found, all technically relevant.
- 3 new sources registered (Batch 30). 3 topic IDs added to processed_topics.txt (total now 431).
- **Headline finding:** Mistral Small 4 119B NVFP4 on DGX Spark — 67-post thread with working recipe.
  MLA head_size=320 rejected by all stock backends on SM121; TRITON_MLA (via eugr's spark-vllm-docker)
  resolves it. ~28-33 tok/s decode corroborated by 5 independent forum users → [reported]. Known
  issues: reasoning_effort bug (vLLM 0.17.2rc1), tool-calling leaks (PR #39217), Eagle/MTP not
  working, --shm-size 16g causes kernel crash (must use 4g). vLLM 0.25.1 now publishes native arm64
  images — no custom build needed for base vLLM.
- Pages touched: models/mistral-small-4 (NEW — full model page), models/qwen (Qwen3.6-35B-A3B FP8
  2× recipe 75-80 tok/s [conjecture]), platform-gb10 (CX7 DAC thermal penalty 6°C even after
  software disable — only physical removal works; dgx-spark-mlnx-hotplug udev/ACPI mechanism
  [conjecture]), benchmarks (6 new forum-reported rows: 5 Mistral Small 4 + 1 Qwen3.6 FP8),
  sources/README, index, log.

## Forum ingest 2026-07-22 (Batch 29)
- 4 new forum topics found, all with at least marginal GB10 relevance.
- 4 new sources registered (Batch 29). 4 topic IDs added to processed_topics.txt (total now 428).
- **Headline finding:** 6-node DGX Spark ring topology (S-forum-6x-ring-rdma) — the most
  technically dense multinode thread in weeks. Three major findings: (1) RoCE RC QPs require
  L2 adjacency — routed L3 RDMA fails at the ibv_modify_qp verbs layer for non-adjacent ring
  pairs, explaining why official topologies stop at 3-node full-mesh; (2) NCCL_IB_MERGE_NICS=0
  + NCCL_IB_SUBNET_AWARE_ROUTING=1 (patched NCCL) together fix 6-node ring RDMA — stock NCCL's
  round-robin channel→HCA assignment is not topology-aware; (3) nvidia-peermem refuses to
  insert ("Invalid argument") on GB10 — GPUDirect RDMA unavailable, DOCA GPUNetIO/GDAKI may be
  the intended path. First quantified RDMA-vs-TCP comparison: only ~7% gain (both host-staged).
- Pages touched: multinode-tp-and-networking (7 new [conjecture] findings — RoCE L2-adjacency,
  MERGE_NICS=0+SUBNET_AWARE_ROUTING fix, NCCL channel topology-unawareness, GID asymmetry, TCP
  fallback workaround, nvidia-peermem modprobe failure + GDAKI hypothesis, Hunlx 3-node env
  recipe), platform-gb10 (3 new [conjecture] — UEFI firmware update stepping-stone requirement,
  serial console not supported, sleep/suspend disabled by default), benchmarks (2 new
  [conjecture] rows — Qwen3.6-35B-A3B NVFP4 6-node PP=6 TCP 326 / RDMA 349 tok/s aggregate).
- All [conjecture] — single forum source each. No new wiki pages created.

## Forum ingest 2026-07-23 (Batch 31)
- 2 new forum topics found (1 technically relevant, 1 skipped as model announcement).
- 1 new source registered (Batch 31). 2 topic IDs added to processed_topics.txt (total now 433).
- Pages touched: containers-and-tooling (spark-vllm-docker build flags — `--rebuild-vllm`
  forces local rebuild, `--use-wheels` uses prebuilt wheels, repo builds from `main` with no
  pinned vLLM version — all [conjecture]), models/mistral-small-4 (build flags detail added
  to existing spark-vllm-docker section [conjecture]), sources/README, index, log.
- Skipped: 377762 (Motif-3 Beta model announcement, no GB10 specifics — same model already
  skipped in Batch 28 as topic 377602).

## Forum ingest 2026-07-24 (Batch 32)
- 7 new forum topics found (4 technically relevant, 3 skipped: social, RMA, entitlement).
- 4 new sources registered (Batch 32). 7 topic IDs added to processed_topics.txt (total now 440).
- **Headline finding:** MiniMax-M3 NVFP4 TP=3 on 3× DGX Spark (S-forum-m3-tp3, tonyd615) —
  first working TP=3 recipe via Luke Alonso's chthonic vLLM+b12x virtual sharding commit.
  Three undocumented head-node OOM fixes (safetensors load format, Ray object-store cap,
  Ray memory monitor disable). NCCL LD_PRELOAD shim trap — baked container shim silently
  overrides user-installed NCCL 2.30u1. Cold power-drain fixes stuck ib_write_bw (12.8→111.85
  Gb/s). TP=3 bandwidth fix increases concurrency, not single-stream tok/s (consistent with
  proven latency-bound cross-node decode). EAGLE3 bf16-draft-vs-NVFP4-target dead-ends.
- Pages touched: models/minimax (TP=3 recipe, OOM fixes, NCCL shim, cold power-drain, EAGLE3
  status — all [conjecture]), multinode-tp-and-networking (LD_PRELOAD shim trap, cold
  power-drain bandwidth fix, bandwidth-vs-concurrency, Ray UMA false OOM — all [conjecture]),
  containers-and-tooling (NGC vs community container gap, nightly wheel pipeline, --vllm-ref,
  --name multi-container, VRAM soldered — all [conjecture]), models/laguna-s-2.1 (quality
  corroboration — good for reasoning+tools, fails generative tasks [conjecture]), benchmarks
  (Solar-Open2-250B INT4 on 2× Spark ~15 tok/s [conjecture]), sources/README, index, log.
- Skipped: 377689 (community extinction — social), 377733 (RMA prep), 374727 (entitlement).

## Forum ingest 2026-07-24 (Batch 33)
- 2 new forum topics found (1 technically relevant, 1 skipped as social/meta).
- 1 new source registered (Batch 33). 2 topic IDs added to processed_topics.txt (total now 442).
- **Headline finding:** Qwen3-TTS GGML backend crashes on GB10 — `ggml_cuda_kernel_can_use_pdl`
  `CUDA error: unspecified launch failure` on first inference. Root cause: qwentts-cpp-python
  PyPI wheels built against CUDA 12.8 / sm_120, not sm_121a. Memory ops work, compute kernels
  fail on dispatch. PDL error is a red herring (async CUDA errors are sticky). Fix: force
  `torch` backend (drop `[ggml]` extra, set `--qwen3_tts_backend torch`); CUDA graphs work,
  TTFA 2.65s, steady-state RTF ~1.7. Same sm_120-vs-sm_121a mismatch class as vLLM FP4 CUTLASS
  and Triton ptxas — now in a 3rd ecosystem component (GGML/qwentts.cpp). All [conjecture].
- Pages touched: platform-gb10 (GGML PDL crash root cause + sm_121a targeting gap
  [conjecture]), containers-and-tooling (Qwen3-TTS torch backend workaround + UMA audio
  tensor pinning tip [conjecture]), sources/README, index, log.
- Skipped: 377793 (PSA about using Discourse MCP to follow the forum — social/meta; replies
  reference already-sourced DSV4-Flash-DSpark and Laguna-S-2.1 recipes, no new findings).

## Forum ingest 2026-07-25 (Batch 34)
- 5 new forum topics found (2 technically relevant, 3 skipped: SSH config parser bug,
  macOS SSH tunnel manager, vision model recommendation thread).
- 2 new sources registered (Batch 34). 5 topic IDs added to processed_topics.txt (total now 447).
- Pages touched: containers-and-tooling (stock `vllm/vllm-openai:latest` hangs silently on GB10
  — no SM121 support; LocateAnything-3B bring-up — ARM64 wheel gaps, device_map='auto' UMA
  pitfall, FastAPI server pattern for non-vLLM models), platform-gb10 (device_map='auto' slow
  on 128 GB UMA), models/qwen (stock vLLM hang on Qwen3.6-35B-A3B-NVFP4). All [conjecture].
- Skipped: 378009 (NVIDIA Sync SSH config parser — client-side tooling), 377913 (macOS SSH
  tunnel manager — personal tool), 377759 (vision model recommendations — no durable findings).

## Forum ingest 2026-07-26 (Batch 36)
- 4 new forum topics found (3 technically relevant, 1 skipped as non-technical question with no
  answers/findings).
- 3 new sources registered (Batch 36). 4 topic IDs added to processed_topics.txt (total now 451).
- Pages touched: benchmarks (Solar-Open2-250B NVFP4 W4A4 on 2× Spark — 15.8 tok/s decode, flat
  with depth; FP8 KV = capacity lever not speed lever; full recipe + flags — [conjecture]),
  attention-and-kv-cache (hybrid-linear attention dodges KV-bandwidth wall; FP8 KV capacity vs
  speed distinction — [conjecture]), platform-gb10 (USB-C PD firmware pending update causes
  overheating without load; 30-min power-cycle fix — [conjecture]), engines (GLM-5.2-Vision-NVFP4
  frozen-backbone projector; adaptive MTP dynamic 2–5 draft depth — [conjecture]),
  roadmap (1 new open problem: adaptive MTP feedback-loop overhead on bandwidth-bound decode).
- Skipped: 378102 (CUDA_VISIBLE_DEVICES simulation question — no answers, no findings, buying
  advice context).

## Forum ingest 2026-07-27 (Batch 37)
- 3 new forum topics found (2 technically relevant, 1 skipped as application showcase).
- 2 new sources registered (Batch 37). 3 topic IDs added to processed_topics.txt (total now 462).
- **Headline finding:** Qwen3.5-122B-A10B-int4 is the community consensus single-Spark daily
  driver — 4 independent forum users confirm it as the "king model" → **[reported]**. New tok/s
  numbers: AutoRound int4 ~65 tok/s on 2× Spark (holds linearly past 100K context), FP8 ~35
  tok/s on 1× Spark, sparkrun-recipes patched vLLM v26 build 5 lanes @ 256K ctx 40+ tok/s.
  AutoRound int4 loop tendency flagged; NVFP4 variants may offer better fidelity.
- Pages touched: models/qwen (122B "king model" consensus + sparkrun-recipes + AutoRound loop),
  containers-and-tooling (sparkctl config-driven serving CLI), benchmarks (4 new [conjecture]
  rows: Qwen 122B int4/fp8/hybrid, DSV4-Flash 1× 45-50 tok/s), sources/README, index, log.
- Skipped: 378131 (brewing agent application showcase — Mistral 119B on Spark, no durable
  technical findings).

## Forum ingest 2026-07-27 (Batch 38)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 38). 2 topic IDs added to `sources/processed_topics.txt`
  (total now 464).
- **Headline finding:** GLM-5.2 hybrid FP8+NVFP4+MXFP4 — first reported 3-way mixed-precision
  checkpoint on GB10 (aidendle94). Decode ~20-25 tok/s on 4× Spark (same bandwidth-bound range
  as pure AWQ-INT4 / NVFP4). Tool-eval-bench 86/100 (v2) / 85/100 (v3-GPTQ). Structured Output
  58% root-caused to reasoning-parser bug (thinking off → 100%, +8 pts overall). Word-salad at
  >90k ctx root-caused to hardcoded `repetition_penalty=1.2` (config, not model/hardware). b12x
  sparse-MLA kernel only reads packed fp8 KV pages. New model page created: `wiki/models/glm-5.2.md`.
- Pages touched: models/glm-5.2 (NEW — consolidated all GLM-5.2 findings across 12 sources),
  benchmarks (4 new [conjecture] rows: hybrid v2/v3/llama-benchy-table/official-NVFP4),
  platform-gb10 (ASUS GX10 SoC+TPM firmware — stable, 2-4% noise, slow reboot),
  quantization-on-gb10 (3-way hybrid quant, custom NVFP4 KV cache, repetition_penalty sensitivity),
  engines (reasoning-parser structured-output bug, thinking-off A/B, MTP4 vs MTP5, word-salad root
  cause), attention-and-kv-cache (b12x sparse-MLA KV format constraint — bf16 KV → immediate EOS),
  sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture] (single thread, multiple
  users in same thread using same image → not independent).

## Forum ingest 2026-07-28 (Batch 39)
- 5 new forum topics found (4 technically relevant, 1 skipped: social/buying advice).
- 4 new sources registered (Batch 39). 5 topic IDs added to `sources/processed_topics.txt`
  (total now 469).
- **Headline finding:** Qwen 122B vLLM v26 + fp8 KV + DFlash + int8 lm-head on single Spark
  (styles01) — first working fp8 KV + DFlash on GB10 for hybrid quant models. 3 custom patches
  (inc_hybrid, int8_lmhead_v3, prefix_align). KV 549K→1.37M tokens (2.6×), concurrency 5.24×
  @ 256K, decode 45.98 tok/s, prefill 957 tok/s (+32%). int8 lm-head reclaims ~1.4 GB.
- Pages touched: models/qwen (v26 + fp8 KV + DFlash + int8 lm-head recipe + benchmark table),
  engines (SpeedyColibri — Rust port of Colibri for GLM-5.2, ~1→4 tok/s with fp8),
  containers-and-tooling (whisper.cpp STT Docker — 7 GB10 build gotchas; official llama.cpp
  Docker matches custom builds, --mmap 0 mandatory, power-cycle fixes 40→67 tok/s),
  benchmarks (4 new [conjecture] rows), sources/README, index, log.
- Skipped: 377281 (social/buying advice, 104 posts, no durable findings).
- No evidence promotions past [reported]. All new findings [conjecture].

## Forum ingest 2026-07-29 (Batch 40)
- 5 new forum topics found (3 technically relevant, 2 skipped: boot failure/RMA, power adapter buying advice).
- 3 new sources registered (Batch 40). 5 topic IDs added to `sources/processed_topics.txt`
  (total now 474).
- **Headline finding:** ComfyUI hard-crash root cause on GB10 — GPU power spike (14→85 W
  instantaneous) trips overcurrent protection, distinct from the power-controller wedge. Fix:
  `nvidia-smi -lgc 300,2100` (clock cap to 2100 MHz, ~50 W) + `swapoff -a`. Second user
  confirms clock cap stabilizes. `--highvram` is a trap on UMA; async offload is near-free.
  `CUDA_CACHE_MAXSIZE=4GB` gives 3× rerun speedup.
- **Gemma-4-26B-A4B NVFP4 benchmark:** Unsloth ~17% faster than nvidia on Spark (160 vs 128
  tok/s aggregate @100 concurrent). Spark ~6-7× slower than RTX Blackwell 6000 Pro. Corroborates
  Unsloth-vs-nvidia quant difference pattern (opposite direction from Qwen3.6-35B where Unsloth
  was slower).
- **CUDA MPS on Spark:** first documented MPS setup for multiple vLLM instances on one GB10.
  `EXCLUSIVE_PROCESS` mode + MPS daemon + `--gpu-memory-utilization 0.45` per instance. Latency
  increases, throughput modestly improves; main value is multi-model co-residency.
- Pages touched: platform-gb10 (power spike/overcurrent, clock cap, swapoff mechanism, CUDA_CACHE,
  --highvram UMA trap, ComfyUI no multi-GPU), models/gemma-4 (Unsloth vs nvidia NVFP4 benchmark),
  containers-and-tooling (ComfyUI crash fix recipe, CUDA MPS setup), benchmarks (2 new [conjecture]
  rows: gemma-4-26B unsloth/nvidia NVFP4), sources/README, index, log.
- Skipped: 378157 (won't boot after update — RMA/boot failure, no durable technical findings),
  378245 (power adapter + network connectivity — buying advice, basic setup question).
- No evidence promotions past [reported]. All new findings [conjecture].

## Forum ingest 2026-07-29 (Batch 41)
- 5 new forum topics found, all technically relevant.
- 5 new sources registered (Batch 41). 5 topic IDs added to `sources/processed_topics.txt`
  (total now 479).
- **Headline finding:** Hard power-off under sustained GPU load at ~90W — detailed reproduction
  with stepped FP16 matmul and throttle bit logging. Unit dies before thermal protection engages
  (no throttle flag, GPU only 82°C), persists after full platform firmware update (SOCFW/EC/USBPD
  all current), clock cap 2200 MHz fixes it. CPU 92-97°C while GPU 78-83°C — GPU sensor looks
  normal. DCGM cannot stress GB10 (Skip for all targeted tests). NVIDIA confirms known issue.
  Clock-cap mitigation now corroborated by 4 independent threads → [reported] as standard GB10
  power/thermal mitigation.
- **Unsloth+b12x vs nvidia+Marlin:** Unsloth ~8% faster than nvidia at 100 concurrent on Spark
  (436 vs 404 tok/s agg) — reverses the prior [reported] 15% slower finding. The b12x backend
  (not the quant) appears to be the lever. Working flashinfer_b12x recipe documented
  (CUTE_DSL_ARCH=sm_121a, vllm>=0.25.0, flashinfer>=0.6.13).
- **NVFP4 KV cache:** 1.68× more capacity than FP8 on Spark (2.31M vs 1.37M tokens, Qwen3-4B on
  SGLang). dtype `torch.float4_e2m1fn_x2`. Extends the KV quant ladder: bf16 → fp8 (2×) → NVFP4
  (1.68× over fp8).
- **DSV4-Flash REAP25 PrismaAURA:** third independent ds4 fork for single GB10 — 92/100 tool-eval,
  16.5 tok/s spec decode, measured-KL quant allocation (IQ2+MXFP4+MXFP8 mix via knapsack). Key
  GB10 finding: sub-4-bit formats (IQ2) cannot use tensor cores — dequant overhead negates compute
  advantage. MXFP4 is the only format that escapes to tensor cores natively. marco.palaferri fork
  achieves 854 tok/s prefill via HMMA attention. DSV4-Flash prefill is compute-bound, not
  bandwidth-bound — distinct from decode.
- Pages touched: platform-gb10 (hard power-off at 90W, clock cap [reported], DCGM Skip, GPU
  throttle commands), models/qwen (Unsloth+b12x benchmark, flashinfer_b12x recipe, vLLM 0.25.x
  hang), attention-and-kv-cache (NVFP4 KV cache capacity), quantization-on-gb10 (NVFP4 KV cache,
  sub-4-bit tensor-core wall, W4A8 source-faithful path), engines (DSV4 REAP25 measured-quant,
  marco.palaferri fork, IQ2 tensor-core wall, prefill compute-bound), benchmarks (8 new
  [conjecture] rows), sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture] except the clock-cap
  mitigation which reaches [reported] via 4 independent corroborating sources.

## Forum ingest 2026-07-30 (Batch 42)
- 4 new forum topics found (3 technically relevant, 1 skipped: RMA complaint).
- 3 new sources registered (Batch 42). 4 topic IDs added to `sources/processed_topics.txt`
  (total now 483).
- **Headline finding:** apt upgrade to driver 580.173.02 breaks GPU on OTA2607 —
  "torn" driver/firmware pairing. Ubuntu noble-updates serves 580.173.02 which
  is not paired with OTA2607's GSP/SEC2 firmware (expects 580.159.03). Xid 119,
  GSP_INIT_DONE timeout, nvidia-smi "No devices found". nvidia-spark-ota-check
  reports torn=1. Fix: re-run DGX Dashboard update or downgrade + hold driver.
  580.173.02 works on Sparks with matching firmware — failure is firmware-version-
  dependent, not universal.
- Pages touched: platform-gb10 (driver 580.173.02 torn pairing, USB3→USB2 fallback
  corroborated on Asus GX10 [reported, 8th user/4th OEM SKU], USB SSD intermittent
  20 MB/s drops, Acer Veriton GN100 thermal A/B ~68°C vs 80-82°C other OEMs,
  spark_hwmon power telemetry driver), containers-and-tooling (model storage
  strategies — 4TB NVMe, NFS 10GbE, NVMe-oF 400G, cron offloading, modelctl,
  USB2-at-boot gotcha), benchmarks (Acer thermal A/B table), sources/README,
  index, log.
- Skipped: 378356 (RTL8127 NIC defect — RMA complaint, 1 of 4 identical units
  affected, hardware fault not platform-wide).
- No evidence promotions past [reported]. All new findings [conjecture].

## Forum ingest 2026-07-30 (Batch 43)
- 5 new forum topics found (3 technically relevant, 2 skipped: karaoke app showcase,
  switch buying advice).
- 2 new sources registered (Batch 43). 5 topic IDs added to processed_topics.txt
  (total now 488).
- **Headline finding:** SM121 software support thread (357663, 43 posts) — NVIDIA
  official roadmap response + community fact-check. vLLM --enforce-eager 20-30% perf
  loss, CuTE DSL FP4 restricted to sm_100a (Issue #2800), PyTorch 2.10/Triton 3.6.0/
  FlashInfer 0.5.3+/CUTLASS 4.2.0+ roadmap, SGLang unofficial branch, MoE kernels
  no optimized GB10 configs, tcgen05/DSMEM/TMEM/TMA lacking, CUDA 12.0f vs 12.1a
  distinction, no locked/hidden memory on Spark.
- Pages touched: platform-gb10 (SM121 software support — 10 new [conjecture] findings),
  models/laguna-s-2.1 (DFlash acceptance corroborated by 5 independent users → [reported],
  tool-eval 82-87/100, TP=2 KV cache 2.7M tokens), roadmap (2 new open problems: CuTE
  DSL FP4 sm_100a restriction, vLLM 0.14.0 enforce-eager question), sources/README,
  index, log.
- Skipped: 378524 (AIraoke — app showcase, no GB10 findings), 378255 (switch buying
  advice — CRS504/CRS812 already documented).
- No evidence promotions past [reported].

## Forum ingest 2026-07-31 (Batch 44)
- 3 new forum topics found (1 technically relevant, 2 skipped: social intro, non-technical
  question with no answers).
- 1 new source registered (Batch 44). 3 topic IDs added to `sources/processed_topics.txt`
  (total now 491).
- **Finding:** Xid 31 MMU faults during AMP-enabled YOLOv8s training on GB10 — 5/5 AMP
  runs fault with `ENGINE GRAPHICS GPC2` / `FAULT_PDE ACCESS_TYPE_VIRT_READ`; cuDNN
  `CUDNN_STATUS_EXECUTION_FAILED` at conv2d; FP16 matmul and AMP-disabled training run
  clean. Root cause unresolved (cuDNN/driver/GSP/hardware all open). Non-LLM workload but
  documents a GB10-specific GPU fault signature under AMP conv paths. [conjecture].
- Pages touched: platform-gb10 (Xid 31 MMU fault), sources/README, log.
- Skipped: 378532 (social intro — replies reference already-sourced tools), 377567
  (non-technical question, no answers).
- No evidence promotions past [reported]. No new wiki pages created.

## Forum ingest 2026-08-01 (Batch 46)
- 4 new forum topics found, 3 technically relevant (1 application showcase registered for
  provenance only).
- 4 new sources registered (Batch 46). 4 topic IDs added to processed_topics.txt (total now 505).
- Pages touched: models/inkling (Inkling-Small-NVFP4 on 2× Spark — NVFP4 fits but no FP8 KV
  cache → context capped at ~300K, BF16 KV only; FP8 needs FlashAttention kernel mod per
  vLLM blog; spark-vllm-docker recipe + paged-KV mod; tool-calling parser bug — direct
  streaming emits `<|content_invoke_tool_json|>` as visible content, patched by ekkis;
  tool-eval-bench 76/100; DSV4 uses less KV memory; tonyd2wild BF16-KV 262K DSpark variant
  in progress; Qwen3.5-122B FP8 as vision alternative — all [conjecture]),
  models/qwen (MoE LoRA training — Unsloth LoRA format incompatible with vLLM fused MoE
  expert tensors; NVIDIA AutoModel/NeMo official MoE LoRA recipes for Gemma4-26B-A4B +
  Qwen3.5-35B-A3B produce HF-compatible adapters servable via vLLM --enable-lora — all
  [conjecture]), roadmap (Inkling-Small FP8 KV kernel modification open problem),
  sources/README, index, log.
- Skipped: 377760 (Super Idol Master 3D character asset pipeline — application showcase,
  not LLM inference; AutoRemesher ARM64 Geogram patch is notable but outside core scope;
  registered source for provenance only).
- No evidence promotions past [reported]. All new findings [conjecture]. No new wiki pages
  created.

## Forum ingest 2026-08-02 (Batch 47)
- 2 new forum topics found, both technically relevant.
- 2 new sources registered (Batch 47). 2 topic IDs added to processed_topics.txt (total now 507).
- Pages touched: models/nemotron-3 (Nemotron-3-Super-120B NVFP4 2-node cluster — full vLLM
  recipe with `--mamba_ssm_cache_dtype float32`, model pre-download requirement, fp8 attention
  scaling-factor warnings, 13.67–14.33 tok/s dual-node vs 15 single-node corroborates
  cross-node-is-slower [conjecture]), engines (DeepSeek-V4-Flash-DSpark full YAML recipe via
  eugr spark-vllm-docker — FlashInfer PR 3817 required, `--load-format safetensors` mandatory,
  3-draft-beats-5 tuning A/B: 71.63 vs 48.60 tok/s at c50, 48.35% vs 27.65% acceptance,
  `max_num_batched_tokens=10240` optimal, 16384 doesn't fit at 262K [conjecture]),
  benchmarks (2 new [conjecture] rows: Nemotron-3-Super 2-node, DSV4-Flash-DSpark 3-draft),
  sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture]. No new wiki pages
  created.

## Forum ingest 2026-08-02 (Batch 48)
- 3 new forum topics found (1 technically relevant, 2 skipped: login loop / Thunderbolt dock,
  frozen-won't-boot / RMA).
- 1 new source registered (Batch 48). 3 topic IDs added to processed_topics.txt (total now 510).
- Pages touched: containers-and-tooling (DGX-Spark-Dashboard — third independent community
  monitoring dashboard, dependency-free, ~190 MB image / ~42 MiB RAM on GB10 vs ~600 MiB
  DCGM+Prometheus+Grafana; **NVML over nvidia-smi** for low-overhead GPU monitoring on UMA
  [conjecture]), sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture]. No new wiki pages
  created.

## Forum ingest 2026-08-03 (Batch 49)
- 3 new forum topics found (2 technically relevant, 1 skipped: Chrome ARM64 announcement).
- 2 new sources registered (Batch 49). 3 topic IDs added to processed_topics.txt (total now 513).
- **Headline finding:** Clock-cap 2000 MHz quantitative A/B — largest dataset yet (38-post
  thread, 5+ independent users). LLM decode ≈0% loss (bandwidth-bound), 55-69% power
  reduction, 8-22°C temp drop. cuBLAS SGEMM sweep explains *why*: -23% clock = -9%
  throughput because working set exceeds 24 MB L2 → memory-bandwidth-bound fraction
  doesn't shrink with clock. Diffusion (compute-bound) pays ~12.5%. Systemd persistent
  clock-cap unit documented. Strongly corroborates existing [reported] clock-cap mitigation.
- Pages touched: platform-gb10 (quantitative clock-cap A/B data, cuBLAS sweep table,
  systemd persistent unit, diffusion-vs-LLM compute-bound distinction, prefill ~10%
  penalty — [reported]/[conjecture]), sources/README, index, log.
- Skipped: 378852 (Chrome ARM64 browser announcement — no GB10 technical content).
- No evidence promotions past [reported]. GRM-3.2-Sky source registered for provenance only
  (model evaluation, no GB10-specific flags/configs — identified as Qwen3.5-35B/Ornith
  finetune, tool-eval 86/100, worse than existing Ornith-1.0-35B-int4-AutoRound).

## Forum ingest 2026-08-03 (Batch 50)
- 6 new forum topics found (5 technically relevant, 1 skipped as social/entitlement).
- 5 new sources registered (Batch 50). 6 topic IDs added to processed_topics.txt (total now 519).
- **Headline findings:** (1) partnerdiag PowerStress reproducibly hard-powers-off DGX Spark —
  thermal sensor zone2/zone4 value swap anomaly persists across firmware updates; post-firmware
  box survives with MODS error 082-000-1-020000600139; RMA approved; first published thermal
  sensor anomaly fingerprint on GB10. (2) 4-node QRS812 switch fabric — first published QRS812
  RDMA latency matrix + DSV4-Flash-0731 DSpark TP=4 benchmark (decode ~90 tok/s, prefill ~2500
  tok/s cold, `nvfp4_ds_mla` KV cache dtype). (3) vLLM prefix cache non-deterministic on
  DSV4-Flash-0731 on 2× Spark. (4) Laguna-S-2.1 2× Spark DFlash spec=15 full acceptance curve.
  (5) DGX Dashboard stale firmware metadata (580.159.03 shown as update when 580.173.02 installed).
- Pages touched: platform-gb10 (PowerStress sensor swap anomaly + dashboard stale firmware),
  multinode-tp-and-networking (QRS812 4-node RDMA latency matrix + DSV4-Flash-0731 TP=4),
  engines (prefix cache inconsistency on 0731), benchmarks (2 new rows),
  models/laguna-s-2.1 (DFlash spec=15 acceptance curve), roadmap (2 new open problems:
  prefix cache isolation, thermal sensor swap systemic-vs-unit-specific), sources/README, index, log.
- Skipped: 378500 (50-post "not suitable for professional workloads" — social/entitlement/RMA).
- No evidence promotions past [reported]. All new findings [conjecture] (single-source forum).

## Forum ingest 2026-08-04 (Batch 51)
- 2 new forum topics found, both technically dense and GB10-relevant.
- 2 new sources registered (Batch 51). 2 topic IDs added to processed_topics.txt (total now 521).
- **Headline finding 1:** GLM-5.2 full 753B (unpruned) on 3× Spark via NVFP4+AQLM hybrid
  checkpoint (S-forum-glm52-3x-aqlm, karol.spark) — first reported TP=3 run of the unpruned
  model. 272 GB NVFP4+AQLM checkpoint (~3.1 bits/param). Decode 15.2–16.1 tok/s. Key innovation:
  virtual head padding to 66 (22/rank) instead of 96 (32/rank) — FlashInfer's dispatch table
  tiles heads in groups of 16, so 22 and 32 cost the same attention while 22 saves 31% on GEMMs.
  v3 kernel L1/L2 stream opts (+6.2% normalized decode). v4 MoonViT vision graft. Benchmark
  methodology: compare t/s÷acceptance, never raw t/s.
- **Headline finding 2:** ComfyUI setup & patches for DGX Spark (S-forum-comfyui-triplany,
  Triplany) — UMA memory management fixes, benchmarks across 6 diffusion workflows, comfy-aimdo
  0.3.0 ARM compile fix, LTX 2.3 22B NVFP4 first reported data point.
- Pages touched: models/glm-5.2 (NEW NVFP4+AQLM 3× section + performance table + [reported]
  summary), attention-and-kv-cache (FlashInfer dispatch table head-count tiling), 
  containers-and-tooling (ComfyUI setup & benchmarks), benchmarks (GLM-5.2 3× row + ComfyUI
  diffusion table), roadmap (2 new open problems), sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture] (single-source forum).

## Forum ingest 2026-08-04 (Batch 52)
- 6 new forum topics found (3 technically relevant, 3 skipped: social/speculation, A10G project,
  use-case discussion).
- 3 new sources registered (Batch 52). 6 topic IDs added to processed_topics.txt (total now 527).
- **Headline finding:** DSML tool-call wrapper tag leaks at >60K context on DeepSeek-V4-Flash-0731
  — the `<｜DSML｜tool_calls>` wrapper marker is sometimes skipped by the model at long context,
  causing raw tool-call markup to leak to output. vLLM PR #49117 adds parser recovery but is
  insufficient at 150K; opencode_compat_proxy or LiteLLM hook provides reliable workaround.
  Same tool-call-parser issue class as GLM-5.2 `glm45` reasoning-parser leak.
- Pages touched: engines (DSML leak + PR #49117 + proxy workaround; tool-eval 87/100; 4-config
  benchmark table TP2/TP4/DP4EP/TP2PP2), platform-gb10 (fan control firmware-only — earliest
  forum corroboration of EC fan-curve regression; swap lockup early report; earlyoom -s 80 too
  aggressive for vLLM startup, fix to -s 20), benchmarks (4 new [conjecture] DSV4-Flash-0731 rows),
  sources/README, index, log.
- Skipped: 378958 (Inkling-Small "new king?" — social/speculation), 373658 (Fast Gemma Project —
  A10G, not GB10-specific), 378891 (What problems are you solving? — social/use-case discussion).
- No evidence promotions past [reported]. All new findings [conjecture] (single-source forum).

## Forum ingest 2026-08-05 (Batch 53)
- 4 new forum topics found (2 technically relevant, 2 skipped: model recommendation, switch fan noise).
- 2 new sources registered (Batch 53). 4 topic IDs added to processed_topics.txt (total now 531).
- Pages touched: platform-gb10 (system apt upgrade triggers power-controller wedge — 107→45 tok/s,
  AC power-cycle restores 84 tok/s; new trigger class for existing [reported] wedge [conjecture]),
  models/qwen (QLoRA fine-tuning Qwen3.6-35B-A3B on single Spark — train bf16 / serve NVFP4 +
  --enable-lora hot-attach, NVFP4 has no gradient path; flash-linear-attention 2.52× throughput
  win; batch_size=1 on MoE severely underutilizes GB10 at ~5.3 TFLOP/s; Claude Code session logs
  → SFT data pipeline [conjecture]), benchmarks (1 new [conjecture] row: Qwen3.6-35B-A3B wedge
  107→45→84 tok/s), sources/README, index, log.
- No evidence promotions past [reported]. All new findings [conjecture] (single-source forum).
