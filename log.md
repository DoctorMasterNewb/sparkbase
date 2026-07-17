# Change log

Append-only. One entry per ingest/lint: date, source(s), pages touched, one line of what changed.

## 2026-07-08 — Public seed: sanitize + evidence-tag the initial KB
- Established sparkbase from a private GB10/DGX-Spark knowledge base: added the evidence ladder
  (`conjecture → reported → reproduced → proven → superseded`), the two-stack agent model
  (hardware vs analysis), SCHEMA.md, AGENTS.md + `agents/`, README + CONTRIBUTING.
- Ported all wiki pages: sanitized private setup (hostnames/IPs/service names/personal paths → role
  wording + examples) and tagged every claim on the ladder. First-party bring-ups → `[proven]`;
  external report/forum claims → `[reported]`/`[conjecture]`.
- Rebuilt `sources/README.md` with source types (forum/repo/report/first-party); S-ids kept stable.

## 2026-07-08 — Forum ingest: 20 NVIDIA DGX Spark forum threads

- **Sources:** 20 forum threads from forums.developer.nvidia.com (DGX Spark / GB10 category).
  Registered as `S-forum-*` in `sources/README.md`. All type `forum` → capped at `[conjecture]`
  (single source) or `[reported]` (multiple independent sources agree).
- **Platform:** power-controller wedge corroborated by 4 independent forum threads (clock pinned at
  721/650/611/550 MHz, 10-15 W, no throttle flag, AC power-cycle fix) → raised to `[reported]`.
  Added spark-doctor and spark-gpu-throttle-check community diagnostics. 140 W TDP is CPU+GPU
  combined (GPU typically 35-45 W in vLLM, 85-90 W peak).
- **Quantization:** AWQ 4-bit > NVFP4 decode by ~32% now `[reported]` (eugr PSA generalizes beyond
  MiniMax). MXFP4 online quantization patches (amasawa_seiji) +65% Qwen3.5, +56% gpt-oss. CUTLASS
  FP4 fails on sm_121 (`Error Internal`). Triton ptxas 12.8 lacks sm_121a. Distributed modelopt
  quant pipeline for 100B+ models (6-fix list). modelopt 0.43 missing input_scale keys → garbage.
- **Multinode:** CX-7 PCIe `SlotPowerLimit 0W` bug throttles link to ~13 Gbps (distinct from
  kernel-6.17 regression). 4-node topology with MikroTik CRS504 switch works.
- **Models:** MiMo TP=3 via virtual-head padding (64→96 q, 4→6 KV, zero-masked). MiniMax-M3 4×
  NVFP4 via chthonic vLLM (b12x backend, FULL cudagraph, 524K ctx). M3-AWQ 4× ~30 tok/s. M3
  llama.cpp RPC 2-node ~10.7 tok/s with hybrid tool template.
- **Benchmarks:** 10 forum-reported rows added (DeepSeek-V4-Flash/DSpark, GLM-5.2, MiMo 2×/3×,
  MiniMax-M3 RPC/AWQ, MXFP4 patched Qwen/gpt-oss, Qwen3.5-122B).
- **Roadmap:** 4 new open problems queued for hardware agents (MXFP4 upstreaming, CX-7 PCIe fix,
  distributed quant pipeline, GLM-5.2 expert prune validation).
- **Pages touched:** platform-gb10, quantization-on-gb10, multinode-tp-and-networking,
  models/mimo-v2.5, models/minimax, benchmarks, llama-cpp-rpc, roadmap, index, sources/README.

## 2026-07-08 — Batch 2 forum ingest: remaining 164 threads (main + projects forums)

- **Sources:** 164 additional forum threads processed from both the DGX Spark / GB10 User Forum
  (category 721) and DGX Spark / GB10 Projects forum (category 723). ~50 new sources registered as
  `S-forum-*` in `sources/README.md` (Batch 2 section). Non-technical threads (social, buying
  advice, RMA complaints, entitlement) triaged but not ingested.
- **Platform:** NVIDIA official power spec confirmed (240W total / 140W SoC / 100W rest). TMA not
  on GB10 (consumer Blackwell lacks TMEM). Overheating shutdowns during sustained load affect
  specific units (RMA recommended). GSP_INIT_DONE timeout (Xid 119) after OTA firmware. Driver
  610 + CUDA 13.3 confirmed working. Ubuntu 26.04 clean install guide. Kernel panic recovery
  (initramfs missing + GRUB_TIMEOUT=0). Dual DP-MST, XHCI, MT7925e WiFi, soft lockup issues.
- **Engines:** Atlas engine (Rust+CUDA, 82 tok/s Qwen3-Next-80B, 2.8× vLLM). antirez/ds4
  (DwarfStar 4) custom CUDA-native DS4F engine. DFlash block-spec decode for Qwen122 (81 tok/s
  agent traffic).
- **Quantization:** Qwen3.6-27B NVFP4 MMLU matches FP16 (0.8485 vs 0.8446) — quality preserved.
- **Models:** GLM-5.2 NVFP4 MTP config bug fixed (24 tok/s 128K MTP4). Hy3-295B NVFP4 on 2x Spark
  (21.8 tok/s, enforce-eager wins, nst=1 optimal). MiniMax-M3 variants: AWQ TP=4 33 tok/s, AWQ
  1M nvfp4 KV 25 tok/s, MXFP4 35 tok/s, W4A16-GPTQ vision 33 tok/s. GLM-5.2 1-bit, REAP-less,
  800K ctx variants. Ornith, GigaChat new models noted. Qwen3.5-397B 1M ctx on 2x.
- **Multinode:** MikroTik CRS812/CRS504 switch options for 4× Spark. 2D parallelism (TP×PP) over
  RJ-45 too latency-sensitive (eugr_nv). DDP training NCCL timeout on 2× Spark. FE+Asus mixing OK.
- **Containers:** vLLM 0.23 + Claude Code Docker, btop fork, Model Manager, sparkdash, Tool Eval
  Bench CLI, ThunderKittens 2.0.
- **Benchmarks:** 11 new forum-reported rows added.
- **Pages touched:** platform-gb10, engines, containers-and-tooling, multinode-tp-and-networking,
  benchmarks, sources/README.

## 2026-07-09 — Batch 3 forum ingest: 160 new NVIDIA DGX Spark forum threads

- **Sources:** 160 new forum topics found (not in processed_topics.txt). ~48 new sources registered
  as `S-forum-*` in `sources/README.md` (Batch 3 section). Processed the most technically dense topics
  first (model recipes, benchmarks, quant findings, platform bugs); non-technical threads (social,
  buying advice, HDMI/AV receiver, ChatGPT restriction, power standby) triaged but not ingested.
  152 topic IDs added to `sources/processed_topics.txt` (total now 336).
- **Platform:** CX-7 bricked by unsolicited mlnx-fw-updater auto-firmware flash (novel failure mode).
  Silent SDPA EFFICIENT_ATTENTION corruption in community PyTorch sm_121 builds (NGC wheels unaffected).
  ComfyUI SageAttention silently inactive without python3.12-dev (20× slowdown). nvcr.io/nvidia/vllm:26.06-py3
  image broken (prometheus-fastapi-instrumentator + fastapi 0.136+ incompat). OOM hang fixed by driver
  580.159.03+. DGX Dashboard fwupd/libfwupd version mismatch after OTA 7.5.0. GB10 UMA community bandwidth
  measurements (161 GB/s idle, 90 GB/s under load). torchaudio unavailable on ARM64/CUDA 13.
- **Multinode:** NCCL 2.30.4 critical for 4× Spark (2.28.9 wedges long generation). SGLang container
  RDMA passthrough needs --device=/dev/infiniband (2.5× speedup, 8.2→25 tok/s). SGLang multi-node 3 traps
  (false-positive collective mismatch, EAGLE flags on every node, RDMA). CUTLASS MoE compile OOM fix
  (MAX_JOBS=1). 4-node full mesh without switch (200GBASE-SR4 transceivers). MTP on SGLang NEXTN (+86%
  single-stream Qwen3.5-397B, +154% Gemma-4-31B).
- **Quantization:** KVarN native vLLM KV-cache quantization (3-5× capacity, Qwen 3.6 compat issue).
  Spark Auto Round sensitivity-aware Int4 quant. KV cache benchmarks (q4_0 92% slower @ 64K, uses MORE
  memory than f16; q8_0 only worthwhile). TurboQuant KV cache (155K→413K tokens). STREAM LOADING
  (on-the-fly 4-bit quant). ModelOpt NVFP4 CPU-bound on Spark. vLLM 0.19→0.23 regression (12% slower,
  15% more memory). Dense model MTP bandwidth math. Heterogeneous NVFP4 quant (Spark + RTX 3090).
- **Engines:** DDTree + DFlash (draft-tree, higher acceptance). STREAM LOADING engine mod. Native SM121
  kernel build guide (.so injection, 13→49 tok/s). vLLM version regression documented.
- **Models:** MiMo DFlash 22→67 tok/s (acceptance scales with output structure). MiMo DFlash + NVFP4 KV
  on v0.24.0. Full GLM-4.7 355B NVFP4 on 2× Spark (17.5 tok/s, 4 walls). DeepSeek-V4-Flash 4× Spark
  (49-54 tok/s). Nemotron-3-Super MTP works (1.70×, accept_len 2.7). Nemotron-3-Ultra 550B on 4× Spark
  (42-43 tok/s n8). MiniMax-M3-W4A16-GPTQ corroborated at 36 tok/s (now [reported]). MiniMax-M2.5 4×
  SGLang (124 tok/s n8 agg). Gemma-4-31B + MTP 4× SGLang (153 tok/s n8). Qwen3.5-397B + MTP 4× SGLang
  (40 tok/s n1). Step-3.7-Flash on single Spark via llama.cpp (31 tok/s). GLM-5.2 IQ4_XS 4× (6.28 tok/s).
- **Containers:** Vitoom Nunchaku (Flux.2 2.5× faster, 59% lower VRAM). ComfyUI container for Spark.
  llama.cpp container build guide (LD_LIBRARY_PATH fix). Gemma4 QAT W4A16 models. Mistral-Small-4 NVFP4
  OOM fix (util 0.9 + swap).
- **Benchmarks:** 17 new forum-reported rows added.
- **Roadmap:** 4 new open problems (CX-7 firmware bricking, SDPA corruption, NVMe-oF expert streaming,
  vLLM version regression).
- **Pages touched:** platform-gb10, quantization-on-gb10, multinode-tp-and-networking, engines,
  containers-and-tooling, models/mimo-v2.5, models/minimax, models/nemotron-3, benchmarks, roadmap,
  sources/README, log, index.

## 2026-07-10 — Batch 4 forum ingest: 10 new NVIDIA DGX Spark forum topics

- **Sources:** 10 new forum topics found by fetch_new_topics.py. 8 were technically relevant; 2 skipped
  (374615 = policy/social re ChatGPT restriction, 362764 = buying advice "Value of 2nd Spark?").
  7 new sources registered as `S-forum-*` in `sources/README.md` (Batch 4 section). Topic 375923
  (MiMo DFlash + NVFP4 KV on v0.24.0) was already ingested in Batch 3 as S-forum-mimo-dflash-v024 —
  its topic ID is newly added to processed_topics.txt but no new source was created. 8 topic IDs
  added to `sources/processed_topics.txt` (total now 344).
- **Quantization:** FLUX.2-dev on Spark with torchao NVFP4 W4A4 (activation-quantized) gives ~3×
  speedup — real FP4 compute via Triton kernels, not weight-only. Critical distinction: most
  "NVFP4" FLUX checkpoints floating around are weight-only (matmul upcasts to BF16). modelopt_fp4
  hits diffusers unpack/shape bug on sm_121a. `mslk` missing dependency. CUDA 13 + Blackwell required.
- **Platform:** UMA mmap double-allocation OOM when loading models via HuggingFace transformers —
  mmap pages + CUDA tensors compete for same UMA pool (~134 GB needed for 67 GB model → OOM at 66%).
  Workaround: _EagerSafeOpen monkey-patch (direct-to-CUDA + posix_fadvise page cache eviction) →
  ~72 GB peak. FSDP from_pretrained loads full model on every rank (75 GB/rank). CUDA 13.2 breaks
  adamw_8bit. Unsloth MoE LoRA incompatible with vLLM fused MoE LoRA weight loading. TCG OPAL + UEFI
  admin password corruption after unexpected shutdown — firmware update lockout (no workaround).
  GB10 display controller 165 MHz max pixel clock (4K@60 impossible, 1440p@120Hz max). ONNX Runtime
  GPU device discovery fails on GB10 (safely ignorable sysfs difference).
- **Containers/Tooling:** llama-benchy (context-depth sweep benchmarking for any OpenAI-compatible
  endpoint, by eugr). DGX Spark Cluster Dashboard (web-based multi-node monitoring). Headless Sunshine
  remote desktop setups. FLUX.2-dev as headless OpenAI Images-API server in spark-vllm-docker.
- **Benchmarks:** 3 new forum-reported rows (MiniMax-M2.1-AWQ-4bit ~36 tok/s, GLM-4.7-Flash-AWQ-4bit
  ~41.75 tok/s, FLUX.2-dev image gen ~3× with NVFP4 W4A4).
- **Pages touched:** quantization-on-gb10, platform-gb10, containers-and-tooling, benchmarks,
  sources/README, log, index.

## 2026-07-10 — Batch 5 forum ingest: 3 new NVIDIA DGX Spark forum topics

- **Sources:** 3 new forum topics found by fetch_new_topics.py. All 3 were technically relevant
  (no social/buying/RMA to skip). 3 new sources registered as `S-forum-*` in `sources/README.md`
  (Batch 5 section). 3 topic IDs added to `sources/processed_topics.txt` (total now 349).
- **Models/mimo-v2.5:** SGLang 4× FP8 recipe (mclenithan) — 31.5 tok/s, 256K ctx, tool eval 89/100,
  full multimodal. EAGLE disabled (OOM on unquantized). NCCL_CUMEM_ENABLE=0 critical. NVFP4 MoE
  backend gap on SM121a (Triton can't dequant FP4, Marlin lacks SM121a, flashinfer_dsl untested).
  MTP OOM on 4× unquantized. Sampling params: temp=0.6, top_p=0.95, repetition_penalty=1.2
  (do NOT copy Qwen3 settings — triggers Thought Loop).
- **Models/minimax:** M2.7 NVFP4/AWQ/FP8 recipes on 2×/4× Spark (serapis, ekkis, aostang, miken,
  co-le). FlashInfer-CUTLASS beats CUTLASS (24.12 vs 22.04 tok/s). AWQ-4bit is clear decode winner
  at 39.4 tok/s (peak 40) vs NVFP4 25.7 — 3 independent reporters agree, corroborating first-party
  AWQ-beats-NVFP4 finding. Unsloth FP8 on 4× gives 36-37 tok/s (single source). eugr confirms
  FlashInfer-CUTLASS is now stable enough to switch all NVFP4 recipes.
- **Benchmarks:** 4 new LLM forum-reported rows (MiMo-V2.5 FP8 4× 31.5, M2.7-NVFP4 24.12,
  M2.7-AWQ 39.4, M2.7-FP8 4× 36-37) + 6 diffusion model image gen rows (FLUX.2-klein, Z-Image-Turbo,
  ERNIE-Image-Turbo, SDXL, Krea2-Turbo, Qwen-Image-2512). DIFFUSERS_ATTN_BACKEND=_native_cudnn
  env var speedup documented.
- **Quantization:** FlashInfer-CUTLASS stability update + diffusion NVFP4 weight-only (1.1-1.4×)
  vs activation-quantized W4A4 (~3×) distinction. Weight-only NVFP4 on diffusion ≠ LLM NVFP4 MoE
  path.
- **Containers/Tooling:** DIFFUSERS_ATTN_BACKEND=_native_cudnn, diffusion model benchmarks.
- **Pages touched:** models/mimo-v2.5, models/minimax, benchmarks, quantization-on-gb10,
  containers-and-tooling, sources/README, log, index.

## 2026-07-11 — Batch 6 forum ingest: 3 new NVIDIA DGX Spark forum topics

- **Sources:** 3 new forum topics found by fetch_new_topics.py. All 3 were technically relevant
  (no social/buying/RMA to skip). 3 new sources registered as `S-forum-*` in `sources/README.md`
  (Batch 6 section). 3 topic IDs added to `sources/processed_topics.txt` (total now 352).
- **Platform:** Random shutdowns after long uptime (55+ days) — thermal paste degradation on
  CPU after months of continuous use, OS thermal sensor may report average not hot-spot (one core
  hitting 105°C while sensor reads 95°C). Repaste + case removal fixes (idle 27°C, load 65–73°C).
  PDU fault variant (Spark unable to exceed 35W, zero logs, PDU power-cycle fixes). Same user
  reports GPU power-controller wedge also stopped after repaste — [conjecture] thermal stress
  may be a contributing trigger for the wedge. No WoL on Spark — Auto Boot + hard power cycle
  (IoT relay) is the only automated recovery. Nsight Systems remote profiling requires
  passwordless sudo on the SSH target.
- **Multinode:** 3-node ring topology fails at NCCL init — sparkrun "auto" detects "3 nodes 2
  ports" and defaults to Switch topology instead of Ring. NVIDIA has a dedicated 3-node ring
  guide on build.nvidia.com. Cable mixing (ASUS vs NVIDIA store no-name) causes MTU mismatch
  (1500 vs 9000). Explicit SSH hostname→IP resolution critical for >2 nodes (mgmt-IP wall
  scales with node count).
- **Pages touched:** platform-gb10, multinode-tp-and-networking, sources/README, log, index.

## 2026-07-11 — Batch 7 forum ingest: 2 new NVIDIA DGX Spark forum topics

- **Sources:** 2 new forum topics found by fetch_new_topics.py. Both technically relevant.
  2 new sources registered as `S-forum-*` in `sources/README.md` (Batch 7 section). 2 topic IDs
  added to `sources/processed_topics.txt` (total now 354).
- **MiMo-V2.5:** Detailed 2-node optimization thread (topic 373669, 166 posts). renek posted a
  full vLLM recipe with new GB10-specific findings: driver 595.71.05 gives smaller KV pool than
  595.58.03 (~233K vs ~368K tokens); NCCL v2.30u1 reserves ~7.5 GiB CGA buffer that pushes GB10
  startup check over; TRITON_ATTN_DIFFKV has a defensive guard rejecting quantized KV (patchable);
  MTP=2 acceptance pos-0≈86%, pos-1≈45%, overall≈65%; util 0.89 hard ceiling (0.90 fails);
  enforce-eager required at 160K ctx; VLLM_USE_RAY_V2_EXECUTOR_BACKEND=0 saves ~12 GiB on node 2.
  Performance: 30-33 tok/s single-stream, 57-63 tok/s aggregate@c3. tonyd615 published GitHub
  repo claiming 38 tok/s non-eager (conflicts with renek's enforce-eager requirement — may use
  shorter ctx). renek reports synthetic 39 tok/s but real-world ~33 tok/s ceiling.
- **Platform:** First-boot WiFi onboarding SSID never broadcasts on some units (topic 376293).
  Two separate DGX Sparks (6 months apart) never transmitted the setup SSID. QR code → product
  page, not setup guide. Monitor+keyboard is the workaround. Not universal — second user reports
  WiFi onboarding worked on 4 units. Status: open, no known fix.
- **Pages touched:** models/mimo-v2.5, platform-gb10, attention-and-kv-cache,
  multinode-tp-and-networking, benchmarks, sources/README, log, index.

## 2026-07-12 — Batch 8 forum ingest: 1 new NVIDIA DGX Spark forum topic

- **Sources:** 1 new forum topic found by fetch_new_topics.py. Technically relevant.
  1 new source registered as `S-forum-*` in `sources/README.md` (Batch 8 section). 1 topic ID
  added to `sources/processed_topics.txt` (total now 355).
- **Platform:** GPU clock wedge follow-up (topic 376239, by florin.andrei + 0rand). Confirms 5 min
  power-off wait is sufficient (down from 30 min previously reported) — [reported] (corroborates
  existing ≥60 s guidance). New power-drain method: disconnect power brick, hold power button
  5–10 s to drain capacitors — no wait needed — [conjecture] (single source). Root cause hypothesis:
  the wedge is in PSU power-control circuits stuck in a safety protocol, not the GPU silicon —
  [conjecture] (single source, consistent with proven symptom).
- **Pages touched:** platform-gb10, sources/README, log, index.

## 2026-07-12 — Batch 9 forum ingest: 3 new NVIDIA DGX Spark forum topics

- **Sources:** 3 new forum topics found by fetch_new_topics.py. 2 technically relevant; 1 skipped
  (376447 = generic Ubuntu root account security question, not GB10-specific). 2 new sources
  registered as `S-forum-*` in `sources/README.md` (Batch 9 section). 3 topic IDs added to
  `sources/processed_topics.txt` (total now 358).
- **Platform:** Reboot does not complete — `sudo reboot` shuts down but machine never powers back
  on; requires USB-C cable removal + reinsert (intermittent, full shutdown + power button also
  works). Distinct from GPU power-controller wedge — a power-delivery / soft-reboot completion
  issue. USB-C PD firmware area may be relevant. [conjecture] (single source).
- **Multinode:** Two DGX Sparks over CX-7 direct link field report (griffith.mark). Third-party
  200G QSQP56 DAC works immediately (Amphenol NJAAKK-N911 is certified part). Both CX-7 ports in
  single L2 domain via eSwitch — port choice doesn't matter. Plain TCP ceiling ~16 Gb/s (Grace
  CPU bottleneck, not link; MTU 9000 doesn't help TCP). SSH ~600 MB/s. NetworkManager config
  documented. DCGM works on GB10 (Xid + PCIe replay counters). PSI + swap-out rate better than
  static memory thresholds for OOM alerting (engines reserve 95% of UMA by design). "Cluster tax"
  metric for interconnect cost. mashie corroborates 200G links used but never at 100% load, no
  interface errors.
- **Pages touched:** platform-gb10, multinode-tp-and-networking, sources/README, log, index.

## 2026-07-13 — Batch 10 forum ingest: 2 new NVIDIA DGX Spark forum topics

- **Sources:** 2 new forum topics found by fetch_new_topics.py. Both technically relevant
  (no social/buying/RMA to skip). 2 new sources registered as `S-forum-*` in `sources/README.md`
  (Batch 10 section). 2 topic IDs added to `sources/processed_topics.txt` (total now 360).
- **Quantization:** NVIDIA refreshed the official `build.nvidia.com/spark/nvfp4-quantization`
  recipe (topic 376530). Community user (paul448) posted a companion gist for manual NVFP4
  conversion. Attempted Qwen3-27B and Qwen3.6-27B conversion but hit TensorRT-LLM errors —
  no clean reproducible result yet. [conjecture] (single source). Corroborates existing finding
  that NVFP4 quant on Spark is CPU-bound and can fail silently.
- **Containers/Tooling:** nvidia-vfx (Maxine VFX SDK) has no aarch64 wheel for DGX Spark
  (topic 363267). GB10 is not on the supported GPU list. NVIDIA officially confirmed no plans
  to add VFX support on Spark. ComfyUI RTX upscaler nodes are broken with no fix path. Multiple
  community users requested aarch64 wheels + source access — no response. [reported] (multiple
  independent users + official NVIDIA confirmation). Broader aarch64 wheel gap pattern (cf.
  torchaudio).
- **Pages touched:** quantization-on-gb10, containers-and-tooling, sources/README, log, index.

## 2026-07-13 — Batch 11 forum ingest: 4 new NVIDIA DGX Spark forum topics

- **Sources:** 4 new forum topics found by fetch_new_topics.py. 2 technically relevant (376574
  easy-vllm, 373818 4-node CRS504 cluster results); 2 skipped (376244 = ASUS GX10 OS upgrade
  logistics, 376536 = buying advice / K8s multi-user question). 2 new sources registered as
  `S-forum-*` in `sources/README.md` (Batch 11 section). 4 topic IDs added to
  `sources/processed_topics.txt` (total now 364).
- **Engines:** easy-vllm code-agent harness (sh.ahn) — Claude Code-based meta-harness for
  automating vLLM build/serve/verify/improve on DGX Spark. Key technical findings from its
  DSV4-Flash bring-up: stock vLLM hits a double hard-wall on DSV4-Flash at sm_121 (sparse-MLA
  `major ∈ [9,10]` constraint + MXFP4 MoE → MARLIN repack → UMA OOM → host down). Fix: jasl/vllm
  SM12x fork PR#41834 (SHA c766cbc6) + `--moe-backend humming` + NVML clock telemetry patch.
  torch 2.11+ ABI wall documented (NGC alpha C++ ABI clashes with prebuilt _C — source build
  required). ib_write_bw 208–218 Gb/s on 2× Spark (corroborates proven fabric measurements).
  mem_watchdog + earlyoom host safety stack for UMA OOM prevention. [conjecture] (single source).
- **Multinode:** 4-node CRS504 (100G switch) cluster — 100G link costs only 5–10% prefill,
  zero decode loss vs 200G. Measured inter-node traffic ~13 Gb/s (far below 100G rail).
  $25 Amazon 100G cable works on 2-node direct link. [conjecture] → [reported] (corbett_korbett
  corroborates same-speed observation). CRS504 Noctua fan swap noted (ops). Strengthens existing
  finding that cross-node collectives are CPU-host-bounced, not link-bound.
- **Benchmarks:** 3 new forum-reported rows (DSV4-Flash TP=4 52–53.6 tok/s, DSV4-Flash TP=2
  29.9–36.8 tok/s, M3-AWQ+EAGLE TP=4 27.7–35.4 tok/s). TP=4 DSV4-Flash shows near-linear
  scaling from 2→4 nodes. M3-AWQ+EAGLE on CRS504 consistent with existing [reported] M3-AWQ
  TP=4 benchmarks.
- **Containers/Tooling:** easy-vllm harness registered as a community tool.
- **Pages touched:** engines, multinode-tp-and-networking, benchmarks, containers-and-tooling,
  models/minimax, sources/README, log, index.

## 2026-07-14 — Batch 12 forum ingest: 4 new NVIDIA DGX Spark forum topics

- **Sources:** 4 new forum topics found by fetch_new_topics.py. 3 technically relevant; 1 skipped
  (361947 = buying advice "Is there a Marketplace? I want to get rid of my Dell GB10 1TB"). 3 new
  sources registered as `S-forum-*` in `sources/README.md` (Batch 12 section). 4 topic IDs added to
  `sources/processed_topics.txt` (total now 368).
- **Engines:** TokenSpeed `sm12x-stable` (S-forum-tokenspeed, jasl) — a fifth inference engine
  alongside vLLM/Atlas/llama.cpp/ds4. jasl spent two weeks adding SM12x support; the SM12x path
  lands in `jasl/tokenspeed` (sm12x-stable branch) while the vLLM fork stays maintained. Build:
  torch 2.13, `TOKENSPEED_CUDA_ARCH=121`, FlashInfer CUTLASS MXFP4 MoE, flashinfer-jit-cache to
  skip cold JIT. On the same 2× Spark pair with llama-benchy (MTP2 + fp8 KV + prefix cache):
  cold-context prefill leads vLLM fork by ~10-14% (2057 vs 1866 t/s @ 8K depth), but decode is
  behind ~70-74% (30-33 vs 41-45 tok/s). KV capacity +25% (1.90M vs 1.52M tokens at 131K ctx).
  Tool calling 45/45 clean (zero HTTP 500s), GSM8K 0.96, MTP acceptance higher than vLLM. The
  CUTLASS MoE that wins prefill has weaker small-M decode GEMM; hybrid CUTLASS-prefill + Triton-
  decode path in progress. NCCL 2.30.4 mandatory on multi-node (2.28.9/2.29.7/2.30.7 all wedge) —
  corroborates existing [reported] NCCL 2.30.4 finding. All [conjecture] (single source by jasl).
- **Containers/Tooling:** Spark Studio (S-forum-spark-studio, TheAwakenOne) — MIT-licensed
  open-source inference dashboard for DGX Spark. Launches vLLM/SGLang/llama.cpp/sparkrun recipes
  from web UI. GB10-specific: live unified-memory monitor, pre-launch memory guard (stops models,
  waits for reclaim, refuses launches that won't fit — prevents OOM crash cycle on 128 GB),
  agent auto-fix loop via local Claude Code/Codex CLIs, Optimize Speed (measurement-based ≥10%
  or rollback), multi-node cluster view with no node limit. [conjecture] (single source).
- **Attention/KV cache:** DSV4-Flash KV cache ~15 GB/1M tokens/node on 2× Spark
  (S-forum-dsv4-kvcache, paxren2020) — significantly larger (~5×) than online calculators predict.
  vLLM CUDA graph memory profiling (default since v0.21.0) reserves ~0.6% of usable memory;
  `--gpu-memory-utilization 0.90` ≈ 0.8943 without profiling. Disable with
  `VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS=0`. [conjecture] (single source).
- **Benchmarks:** 2 new forum-reported rows (DSV4-Flash TokenSpeed 30.3-33.3 tok/s, DSV4-Flash
  vLLM jasl fork 41.3-45.3 tok/s — same pair, same config, direct comparison).
- **Pages touched:** engines, containers-and-tooling, attention-and-kv-cache, benchmarks,
  sources/README, log, index.

## 2026-07-14 — Forum ingest: 1 new topic (Batch 13)

- **Source:** 1 new forum thread (S-forum-acestep-music, topic 376653). Type `forum`.
- **Finding:** ACE-Step v1.5 XL music-generation model runs on single DGX Spark, fits comfortably
  in VRAM with companion 5Hz-LM-4B lyrics model. 3 independent users confirm (danielgbates,
  joey28, aostang) → [reported] tier if it warranted a page.
- **Not ingested to wiki:** outside core LLM-inference scope (not served via vLLM/llama.cpp/sglang);
  no GB10-specific flags, env vars, error strings, tok/s numbers, or quant formats; "fits in 121 GB
  unified memory" is trivially true for a single unquantized model. Source registered for provenance
  only. No wiki page created or edited.
- **Pages touched:** sources/README, index, log.

## 2026-07-15 — Forum ingest: Batch 14 — 6 new topics (4 processed, 2 skipped)

- **Sources:** 4 new forum sources registered (Batch 14) in `sources/README.md`.
  6 topic IDs added to `processed_topics.txt` (total now 375).
- **Topics found:** 6 new topics. 4 technically relevant, 2 skipped:
  - 376806 (Franka RT kernel) — robotics, not LLM inference on GB10. Skipped.
  - 376761 (Power on failure after outage) — hardware RMA complaint. Skipped.
- **Pages touched:**
  - models/qwen (Unsloth NVFP4 ~15% slower than nvidia on GB10 — [reported] via 3 independent
    benchmarks; flashinfer_b12x unavailable on stock vLLM; working Marlin MoE+MTP recipe; W4A16
    bypass hypothesis; quality parity; "Paths to 100+ tok/s" updated — Unsloth path struck through
    as tested),
  - quantization-on-gb10 (Unsloth NVFP4 slower on GB10, flashinfer_b12x gap, W4A16 bypass
    hypothesis),
  - benchmarks (5 new forum-reported rows: Qwen3.6-35B-A3B nvidia vs Unsloth NVFP4 comparison),
  - engines (multi-model co-hosting: vision+LLM on 2× Spark is memory-starved, offload vision
    to separate machine [reported], multimodal front-end+text reasoning pipeline pattern),
  - platform-gb10 (HPC/slurm CPU P/E core topology for job binding, CX-7 switch topology config,
    Llama 3.2 3B finetuning 8× slower than benchmark — known FAQ),
  - sources/README, index, log.
- **Key findings:**
  1. Unsloth Qwen3.6 NVFP4 quants are ~15% slower than nvidia NVFP4 on GB10 — 3 independent forum
     benchmarks agree → [reported]. Unsloth's "2.5× faster" is B200-only, does not transfer to sm_121.
  2. flashinfer_b12x kernel unavailable on stock vLLM despite capability detection returning True.
  3. Multi-model co-hosting (vision + LLM) on 2× Spark is memory-starved; offload vision to a
     separate machine is the recommended pattern (multiple users independently arrived at this).
  4. CPU P/E core topology (Cortex-X925 + A725) matters for HPC slurm job binding on Spark.

## 2026-07-15 — Forum ingest: Batch 15 — 4 new topics (3 processed, 1 skipped)

- **Sources:** 3 new forum sources registered (Batch 15) in `sources/README.md`.
  4 topic IDs added to `processed_topics.txt` (total now 379).
- **Topics found:** 4 new topics. 3 technically relevant, 1 skipped:
  - 376822 (Is DGX Spark worth buying for fine-tuning?) — buying advice. Skipped.
- **Pages touched:**
  - platform-gb10 (CX-7 ports are hot-pluggable — not visible in ifconfig/lspci until cable
    connected; /etc/nvidia/cx7-hotplug-enabled controls behavior; idle power draw nearly
    doubles when port active [conjecture]),
  - engines (LLM + ComfyUI co-hosting on 2× Spark — vLLM KV cache starves co-hosted
    workloads; --gpu-memory-utilization 0.7-0.8 enables co-hosting; llama.cpp better than
    vLLM for co-hosting; swapoff -a before loading large models [conjecture]),
  - multinode-tp-and-networking (interconnect is bottleneck for large MoE not memory
    ~23 GB/s cross-node vs ~600 GB/s in-box; MoE gains flatten past TP=4; FP8 training
    impossible on sm_121 (TransformerEngine no backend, no roadmap); Megatron-LM works on
    GB10 with caveats — Megatron Bridge VRAM, FSDP weight-gather, NCCL subnet env vars
    for 3+ nodes [conjecture]),
  - models/qwen (Qwen3.5-397B-A17B on 8× GB10: 31-35 tok/s FP8; architecture comparison
    validated from config.json — 27B dense / 35B-A3B MoE / 397B-A17B MoE parameter math
    [conjecture]),
  - benchmarks (1 new forum-reported row: Qwen3.5-397B-A17B FP8 8× GB10 31-35 tok/s),
  - sources/README, index, log.
- **Key findings:**
  1. CX-7 ports on DGX Spark/ASUS GX10 are hot-pluggable — they don't appear in ifconfig/lspci
     until a cable is connected. Controlled by /etc/nvidia/cx7-hotplug-enabled. Disabling
     hot-plug (or connecting a cable) nearly doubles idle power draw.
  2. vLLM's aggressive KV cache pre-allocation on unified memory makes co-hosting with ComfyUI
     impractical on the same node. Workaround: reduce --gpu-memory-utilization to 0.7-0.8, or
     use llama.cpp (better UMA co-hosting behavior). Practical pattern: LLM on one Spark,
     ComfyUI on the other.
  3. Cross-node interconnect (~23 GB/s) vs in-box (~600 GB/s) is the bottleneck for large MoE
     models, not memory. MoE all-to-all is very sensitive to this 26× gap. Gains flatten past
     TP=4 — the largest reported cluster (8× GB10, Qwen3.5-397B FP8) achieves only 31-35 tok/s.
  4. FP8 training is impossible on sm_121 (TransformerEngine has no backend, no roadmap).
     Megatron-LM works on GB10 for MoE expert parallelism but with VRAM and networking caveats.

## 2026-07-16 — Forum ingest: 3 new topics (Batch 16)

- **Sources:** 3 new forum topics found. 2 technically relevant, 1 skipped (buying advice/warranty/social
  — MSI EdgeXpert reliability thread, topic 371537, contains minor PD firmware 600MHz cap and EC
  firmware corruption mentions but is predominantly buying advice; the PD firmware finding overlaps
  with the already-documented power-controller wedge). Registered 2 new `S-forum-*` sources (Batch 16):
  S-forum-colibri-glm52, S-forum-comfyui-optimized.
- **Pages touched:**
  - engines (Colibri — sixth engine, pure-C expert-streaming for GLM-5.2 744B MoE on single Spark,
    2.4-3.3 tok/s, O_DIRECT 9.69 GB/s, CACHE_ROUTE cache-aware routing, profile breakdown,
    scale-out hypothesis [conjecture]),
  - containers-and-tooling (ComfyUI Docker optimized for DGX Spark — CUDA 13.1/sm_121, SageAttention 2,
    double-VRAM bug fix copy=False + --disable-mmap, --disable-dynamic-vram, CUDA_MODULE_LOADING
    LAZY accidental winner; cudaMemGetInfo under-reports free UMA with co-resident CUDA process —
    psutil.virtual_memory().available fix [conjecture]),
  - platform-gb10 (cudaMemGetInfo under-reports free memory on UMA when another CUDA process is
    resident — generalizes beyond ComfyUI to any multi-process UMA workload [conjecture]),
  - benchmarks (1 new forum-reported row: GLM-5.2 744B Colibri expert streaming 2.39/3.33 tok/s),
  - roadmap (Colibri demonstrates expert-streaming approach in practice — bottleneck is attention
    not disk I/O, suggesting faster storage alone won't fix throughput),
  - sources/README, index, log.
- **Key findings:**
  1. Colibri (JustVugg/colibri) is a new pure-C engine that streams MoE experts from disk, enabling
     a 744B MoE (GLM-5.2) to run on a single 121 GB Spark — first reported engine to do so. 2.4-3.3
     tok/s is very slow but coherent. The profile shows attention dominates (6.16s of 18s), not disk
     I/O — so faster storage alone won't dramatically improve throughput. Experimental CACHE_ROUTE
     (cache-aware expert routing, ~14% substitution) raises expert hit 82→97% and tok/s 2.4→3.3.
  2. `cudaMemGetInfo` (the API behind `torch.cuda.mem_get_info()`) under-reports free unified memory
     when another CUDA process is resident on the same GB10 device. With vLLM holding 34 GB,
     `cudaMemGetInfo` returns ~6 GB free even though 40+ GB is actually available. This causes
     applications (ComfyUI, and potentially others) to needlessly offload to "CPU" — which on UMA
     is the same physical RAM. Fix: use `psutil.virtual_memory().available`. This generalizes to
     any multi-process UMA workload and is a fundamental GB10 platform finding.
  3. ComfyUI on DGX Spark has a double-VRAM bug: `copy=True` in `tensor.to()` with `--disable-mmap`
     duplicates tensor data in the same unified pool. Fix: patch `comfy/utils.py` to `copy=False`.
  4. Topic 371537 (MSI EdgeXpert reliability) skipped — buying advice/warranty/social. Minor technical
     mentions (PD firmware 600MHz cap, EC firmware corruption via DGX dashboard) are single-source
     and overlap with existing power-controller wedge documentation.

## 2026-07-16 — Forum ingest: Batch 17 — 5 new topics (4 processed, 1 skipped)

- **Sources:** 4 new forum sources registered (Batch 17) in `sources/README.md`.
  5 topic IDs added to `processed_topics.txt` (total now 387).
- **Topics found:** 5 new topics. 4 technically relevant, 1 skipped:
  - 376589 (Triple stack) — buying advice/support question about 3-node stacking. Skipped.
- **Pages touched:**
  - quantization-on-gb10 (NVFP4 meta-analysis — NVFP4 leaves ~half layers bf16 unlike Int4 which
    quantizes all layers [reported]; TRT-LLM NVFP4 5 tok/s slower than GGUF Q4_K_M 4.6-4.9 tok/s
    on same Spark [reported]; Nemotron-3-Super NVFP4 bandwidth efficiency 42-48% of theoretical
    ~45 tok/s ceiling [reported]; NVFP4 now operational via community Docker, FlashInfer 0.6.8.1
    merged [reported]; PrismQuant project for full-NVFP4 quantization noted [conjecture]),
  - models/nemotron-3 (Ollama v0.30.x-v0.31.2 parser regression breaks Nemotron-3-Super — SSE
    stream aborts mid-response, no finish_reason; fix: downgrade to Ollama 0.24.0; v0.31.2-rc1
    does NOT fix; full config documented [conjecture]; NVFP4 bandwidth efficiency 42-48%
    [reported]),
  - multinode-tp-and-networking (ib_write_bw falsely reports >64 KiB RDMA WRITE failure on GB10
    — fabric is fine, minimal libibverbs probe passes all sizes to 8 MiB, NCCL all_reduce
    24.0 GB/s busbw zero errors [conjecture]; NCCL_NET_PLUGIN=none required — AWS OFI plugin
    fails on GB10 UMA [conjecture]; NCCL_TOPO_FILE correction — auto-detected PCIe Gen1×1
    [conjecture]; RoCE data NIC-offloaded, use *_phy counters [conjecture]; one interface per
    subnet + arp_ignore=1/arp_announce=2 [conjecture]),
  - platform-gb10 (Ollama parser regression [conjecture], NVFP4 bandwidth efficiency 42-48%
    [reported]),
  - engines (DSV4-Flash-DSpark-Abliterated source added — abliterated/uncensored variant, fork
    of DS4 DSpark recipe, 50-60 tok/s [conjecture]),
  - benchmarks (5 new forum-reported rows: Llama-3.3-70B NVFP4 vs GGUF Q4_K_M, Nemotron-3-Super
    NVFP4 1×/2×, DSV4-Flash-DSpark-Abliterated),
  - sources/README, index, log.
- **Key findings:**
  1. NVFP4 on GB10 leaves ~half of layers in BF16 (quality concerns), unlike Int4/AutoRound which
     quantizes all layers. This structural difference means NVFP4 moves more bytes/token than
     pure Int4, contributing to its decode underperformance. Multiple independent forum sources
     agree → [reported]. Community is working on full-NVFP4 quant (tenari's PrismQuant).
  2. TRT-LLM NVFP4 (NVIDIA's own stack, 5 tok/s) is slower than GGUF Q4_K_M via LM Studio
     (4.6-4.9 tok/s) for the same 70B model on the same Spark — NVIDIA's NVFP4 path is slower than
     non-NVIDIA quant on non-NVIDIA tooling. Multiple reporters → [reported].
  3. NVFP4 achieves only 42-48% of the bandwidth-limited theoretical ceiling on GB10 (measured
     19-22 tok/s vs theoretical ~45 tok/s for Nemotron-3-Super). A well-optimized path should
     reach 60-80%. The gap is software/kernel efficiency, not hardware. Multiple sources →
     [reported].
  4. ib_write_bw falsely reports RDMA WRITE failure above 64 KiB on GB10 — the fabric is healthy.
     A minimal libibverbs probe passes all sizes. NCCL works fine (24.0 GB/s busbw). The failure
     is in the perftest instrument, not the transport. Same defect class seen on CX-5 (2023) and
     CX-7/KVM (2024). Single source → [conjecture].
  5. Ollama v0.30.x-v0.31.2 has a server-side SSE parser regression that breaks Nemotron-3-Super
     on GB10. Fix: downgrade to 0.24.0. v0.31.2-rc1 does NOT fix. Single source → [conjecture].
  6. NCCL_NET_PLUGIN=none is required on GB10 — the bundled AWS OFI plugin fails on unified memory
     regardless of NCCL_IB_DISABLE. Single source → [conjecture].

## 2026-07-17 — Forum ingest: Batch 18 — 2 new topics (both processed)

- **Sources:** 2 new forum sources registered (Batch 18) in `sources/README.md`:
  S-forum-glm52-8x (topic 376831), S-forum-bonsai27b (topic 376879).
  2 topic IDs added to `processed_topics.txt` (total now 389).
- **Topics found:** 2 new topics, both technically relevant:
  - 376831 (GLM-5.2-Int4-Int8Mix on 8× GB10) — highly technical, largest reported DGX Spark cluster.
  - 376879 (Qwen3.6-27B Binary/Ternary Bonsai 27B by Prism-ML) — release announcement + Spark
    relevance hypothesis; no GB10 benchmarks yet.
- **Pages touched:**
  - benchmarks (3 new [conjecture] rows — GLM-5.2-Int4-Int8Mix on 8× GB10 TP8 DCP=1 ~1,200 t/s
    prefill / 33–54 t/s decode; TP4+PP2 ~12 t/s MTP collapse; DCP4 decode-starvation scheduler),
  - multinode-tp-and-networking (NCCL_BUFFSIZE 16 MB at TP8 [conjecture]; TP4+PP2 wrecks MTP
    acceptance → 8% [conjecture]; DCP4 decode starvation + decode-aware prefill scheduler
    [conjecture]; draft_tensor_parallel_size=1 [conjecture]),
  - quantization-on-gb10 (b12x W4A8 MoE backend — INT4 weights + INT8 activations via native
    FP8 CUTLASS [conjecture]; stale topk_indices_buffer in flashinfer SM120 sparse MLA PR#46994
    [conjecture]; quantized NextN draft token mapping [conjecture];
    VLLM_B12X_MLA_SPEC_EXTEND_AS_DECODE=1 [conjecture]; Int4-Int8 mix quant [conjecture]),
  - models/qwen (Bonsai 27B binary/ternary Qwen3.6-27B — hypothesis: faster decode on
    bandwidth-bound Spark dense, no GB10 benchmarks yet [conjecture]),
  - roadmap (3 new open problems: v16+b12x W4A8 isolated contribution, Bonsai sm_121 kernel
    path, DCP4 decode-aware scheduler on DCP1),
  - sources/README, index, log.
- **Key findings:**
  1. GLM-5.2-Int4-Int8Mix on 8× GB10 is the largest reported DGX Spark cluster run — ~1,200 t/s
     prefill (v16-unified branch is the biggest prefill lever), 33–54 t/s avg decode (33–39 prose,
     40–55 code, peak 54.5–58). Stack: vLLM v16-unified fork (local-inference-lab/vllm 5dffea8) +
     b12x W4A8 MoE (lukealonso/b12x 97b3d64) + CosmicRaisins DCP1 patches + NCCL 2.30.4. All from
     one thread (ciprianveg + penguinchang) → [conjecture], not [reported] (same thread, not
     independent).
  2. b12x W4A8 MoE is a new quant regime on GB10: INT4 weights (Marlin decompress) + INT8
     activations (native FP8 CUTLASS). Extends the "fewest bytes" principle to activations and
     may explain the decode jump from ~28–49 (older MTP k=4 image) to 33–54 t/s. Not yet
     isolated from the 8× scale or v16 branch contributions.
  3. TP4+PP2 raises prefill to ~1,800 t/s but collapses MTP acceptance to ~8% → decode drops to
     ~12 t/s. Pipeline parallelism is too latency-sensitive for MTP draft acceptance on Spark —
     corroborates existing PP-over-ethernet finding (S-forum-2d-parallel).
  4. DCP4 on TP8 causes decode starvation (decode → 0.0–0.2 tok/s during prefill). A decode-aware
     prefill scheduler (ENABLE_DECODE_AWARE_PREFILL=1) caps stalls to ~1.6s but decode still
     drops to ~2.74 tok/s under pressure. DCP1 is ~30% faster prefill / ~60% faster gen but DCP4
     enables 320K×10 context (3.2M KV tokens).
  5. Stale topk_indices_buffer in flashinfer SM120 sparse MLA (PR #46994) is a subtle bug class
     that silently drops MTP acceptance from ~85% to ~30% with no error — needs two patches
     (flashinfer + b12x_mla_sparse.py). GB10/sm_121-specific sparse-MLA kernel bug.
  6. Bonsai 27B (Prism-ML) — 1-bit/ternary Qwen3.6-27B, ~94% quality, much smaller footprint.
     Hypothesis: faster decode on bandwidth-bound Spark dense. No GB10 benchmarks; sm_121 kernel
     path for 1-bit/ternary unverified (Marlin doesn't support it natively). Queued for hardware
     agent verification.
