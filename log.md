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

## 2026-07-17 — Forum ingest: Batch 19 — 5 new topics (all processed)

- **Sources:** 5 new forum sources registered (Batch 19) in `sources/README.md`:
  S-forum-usb2-fallback (362015), S-forum-fw-july2026 (376890), S-forum-ota-loop (376981),
  S-forum-asus-fw0103 (364160), S-forum-host-freeze-tp2 (376882).
  5 topic IDs added to `processed_topics.txt` (total now 394).
- **Topics found:** 5 new topics, all technically relevant (no social/buying/RMA to skip):
  - 362015 (USB2 fallback) — USB3 SuperSpeed PHY not registered, all USB at 480 Mbps.
  - 376890 (New firmware) — FE Spark EC + UEFI SoC firmware update.
  - 376981 (July 2026 update issue) — DGX Dashboard OTA stuck in loop.
  - 364160 (ASUS GX10 firmware) — BIOS v0103 PD capsule fixes thermals + link speed.
  - 376882 (Host freeze TP=2 prefill) — Total host death during heavy multi-node prefill.
- **Pages touched:**
  - platform-gb10 (USB3 SuperSpeed PHY not registered → USB2 fallback [reported] via 7
    independent users; MediaTek T-PHY `phy-mtk-tphy` has no ACPI binding and is not loaded;
    debugfs portsc RxDetect on all controllers; not universal — some FE Sparks work fine;
    new FE firmware EC 0x03000302→0x03000508, UEFI SoC 0x0200980f→0x02009b0b [conjecture];
    DGX Dashboard OTA stuck in loop — manual `apt upgrade` workaround, nvidia-spark-ota-check
    diagnostic tool with torn-score [conjecture]; ASUS GX10 v0103 PD/0x507 capsule fixes
    thermals ~8-10 W lower [reported] via 2 independent users, 4× link speed [conjecture];
    total host freeze during heavy TP=2 prefill = thermal shutdown with zero forensic trace
    across kdump/watchdogs/netconsole/NCCL Flight Recorder [conjecture]),
  - multinode-tp-and-networking (ASUS PD firmware 4× link speed [conjecture] — may relate
    to CX-7 SlotPowerLimit 0W throttle; host freeze during heavy prefill = highest combined
    SoC power draw scenario [conjecture]),
  - sources/README, index, log.
- **Key findings:**
  1. USB3 SuperSpeed PHY not registered on some FE DGX Sparks — all USB falls back to 480 Mbps
     USB 2.0. 7 independent users report the issue. Root cause indicator: MediaTek T-PHY
     (`phy-mtk-tphy`) has no ACPI binding and is not loaded. Debugfs confirms all xHCI
     SuperSpeed ports stuck in RxDetect. Not universal (elsaco's FE Spark works fine). No
     firmware fix confirmed. → [reported] (multiple independent users, same symptom).
  2. New FE Spark firmware available: EC 0x03000302→0x03000508, UEFI SoC 0x0200980f→0x02009b0b.
     May address the USB2 fallback and power-controller wedge (both EC/firmware-level).
     LVFS publication lagged the dashboard announcement. → [conjecture] (impact unconfirmed).
  3. DGX Dashboard OTA can get stuck in a persistent update loop — manual `apt upgrade` or
     `apt full-upgrade` is the workaround. The `nvidia-spark-ota-check` tool
     (`/opt/nvidia/spark-ota-check/check_ota_status.py`) exposes `torn-score` (0 = fully
     applied) and per-component version comparison. Related to existing fwupd mismatch
     finding. → [conjecture] (single source, but consistent with known fwupd issues).
  4. ASUS Ascent GX10 BIOS/Firmware v0103 — the PD/0x507 (USB-C PD 5.7) capsule update
     lowers thermals by ~8-10 W (2 independent users via UPS measurement, ComfyUI peak
     75-80°C→65-70°C) → [reported]. The 4× inter-Spark link speed improvement is
     single-source → [conjecture]. The PD capsule may influence CX-7 power delivery or
     PCIe slot power advertisement — potentially related to the existing CX-7
     SlotPowerLimit 0W throttle finding. The GUI update failed on both machines; manual
     `capsule_update.sh usbpd_5.7.cap` worked. July 2026 OTA also triggers the loop issue
     on Asus (Asus pipeline lags NVIDIA availability).
  5. Total host freeze (not process hang) during heavy multi-node TP=2 vLLM prefill on 2×
     Spark — Step-3.7-Flash-NVFP4 via spark-vllm-docker. Zero forensic trace across kdump,
     hung_task_panic, softlockup_panic, bidirectional netconsole, and NCCL Flight Recorder.
     Diagnosed as thermal shutdown (field diagnostic failed → RMA). Heavy non-cached prefill
     maximally stresses GPU + CPU-side host-staged NCCL simultaneously — the highest combined
     SoC power scenario. Consistent with existing thermal sensor blind-spot finding. →
     [conjecture] (single source, but diagnosis confirmed by NVIDIA field diagnostic).

## 2026-07-18 — Forum ingest: Batch 20 — 1 new topic (processed)

- **Sources:** 1 new forum source registered (Batch 20) in `sources/README.md`:
  S-forum-mtp-lossless (377030). 1 topic ID added to `processed_topics.txt`
  (total now 395).
- **Topics found:** 1 new topic, technically relevant (no social/buying/RMA to skip):
  - 377030 (MTP lossless?) — quality debate: MTP measurably affects output quality; vLLM +
    llama.cpp MTP+prefix-cache interaction bugs; DS4F prefix-batch tuning.
- **Pages touched:**
  - engines (new ingest section — MTP quality impact [conjecture]: up to ~5 pts on tool-call
    bench, temperature tuning does not eliminate the gap; ~40% speed vs ~2% quality hit on
    Qwen3.6-27B [conjecture]; vLLM + llama.cpp both have MTP+prefix-cache interaction bugs
    causing visible degradation that disappears when prefix caching is off [conjecture];
    DS4F prefix-batch 16384 / MTP=4 → 70-75% acceptance (80% coding, 70% llama-benchy),
    prefix-batch eats KV cache on UMA [conjecture]; "theory != deployment" practical-lossiness
    debate — strict verification would kill acceptance rates, real deployments cut corners
    [conjecture], countered by the math-lossless argument [conjecture]),
  - roadmap (new open problem: measure MTP quality impact & prefix-cache interaction on real
    Spark — run MTP-on vs off with prefix caching ON/OFF on a known model, isolate whether
    the prefix-cache interaction is the sole cause),
  - sources/README, index, log.
- **Key findings:**
  1. MTP measurably affects output quality, not just throughput — up to ~5 points on tool-call
     bench, not explained by noise (JasonW). A second user (Azampatti) reports "almost
     identical" capability-suite scores with/without MTP, ~40% speed vs ~2% quality hit on
     Qwen3.6-27B — so the delta is workload-dependent (capability benchmarks small,
     tool-call evals larger). → [conjecture] (single thread; two users but same thread, not
     independent).
  2. vLLM and llama.cpp both have MTP + prefix-caching interaction bugs (mangosq/Yen): visible
     degradation only when both are enabled together; without prefix caching, no visible
     degradation. Practical mitigation: disable prefix caching with MTP, or leave MTP off
     for agentic workflows. This is an engine bug, not a theoretical MTP property — affects
     both engines Spark users run. → [conjecture] (single source for the bug claim).
  3. DS4F MTP tuning (0rand): prefix-batch 16384 with MTP=4 → 70-75% stable prediction quality
     (80% coding, 70% llama-benchy). Tuning is model-dependent (attention type, heads,
     cache size, num prediction tokens). Prefix-batch size "greatly eats into KV cache"
     — a real tradeoff on GB10's 121 GB unified memory. → [conjecture].
  4. Unresolved in-thread debate on whether practical MTP is "lossy by design" (Nerhun:
     strict verification kills acceptance, deployments cut corners) vs "mathematically
     lossless if implemented correctly" (A3refaat, JasonW: causality is enforced, 0%
     acceptance costs throughput not quality). The observed quality deltas suggest at
     least some serving stacks do not enforce strict verification. → [conjecture].
- **Evidence cap:** All findings capped at [conjecture] — single forum thread, no independent
  corroboration, no hardware verification available. Quality-impact claim is load-bearing
  and explicitly flagged for hardware-agent measurement in roadmap.

## 2026-07-18 — Forum ingest: Batch 21 — 2 new topics (both processed)

- **Sources:** 2 new forum sources registered (Batch 21) in `sources/README.md`:
  S-forum-machineid (377208), S-forum-nm-phantom (377220). 2 topic IDs added to
  `processed_topics.txt` (total now 397). Also folded the previously-registered but
  unplaced S-forum-cve (374930) into platform-gb10.md alongside the machine-id finding.
- **Topics found:** 2 new topics, both technically GB10-relevant (no social/buying/RMA
  to skip):
  - 377208 (MSI EdgeXpert DGX Spark having identical Machine IDs) — CVE-2026-24218;
    MSI + ASUS GX10 both ship cloned images with byte-identical `/etc/machine-id` and
    SSH host keys; one-liner fix; MSI patched May 2026.
  - 377220 (Connection failed / Activation of network connection failed on DGX Spark
    — root cause and clean fix) — NetworkManager phantom DHCP profiles auto-created for
    the ConnectX QSFP ports retry every ~45 s when carrier present but no DHCP server;
    full diagnosis + nmcli autoconnect fix.
- **Pages touched:**
  - platform-gb10 (new "Batch 21" ingest section — OEM images ship with identical
    `/etc/machine-id` + identical SSH host keys → CVE-2026-24218 [reported]; affects
    MSI EdgeXpert + ASUS GX10; two independent OEMs/users; one-liner fix
    `systemd-machine-id-setup` + `ssh-keygen -A`; SSH impersonation risk is real when
    Sparks are direct-cabled over CX-7; not a GB10 hardware defect, an OEM imaging
    defect; also placed S-forum-cve on this page),
  - multinode-tp-and-networking (new "Batch 21" ingest section — NetworkManager phantom
    DHCP profile retry loop on ConnectX QSFP ports [conjecture]; GB10-specific because
    the multiple CX-7 fabric interfaces trigger it; `ip-config-unavailable` = has link,
    no lease; `nmcli connection.autoconnect no` fix; timing gotcha on in-flight cycle;
    clustering playbook doesn't conflict),
  - sources/README, log.
- **Key findings:**
  1. OEM DGX Spark images (MSI EdgeXpert, ASUS GX10) ship with byte-identical
     `/etc/machine-id` — and therefore identical SSH host keys — because the factory
     DGX OS image is cloned without re-running `systemd-machine-id-setup`. This is
     CVE-2026-24218 (NVIDIA Security Bulletin: DGX Spark - May 2026). Two independent
     OEMs/users (ohaibuzzle on MSI, JW2026 on ASUS) → [reported]. Real Spark-specific
     bite: SSH host-key collision enables silent on-path impersonation when Sparks are
     direct-cabled over the CX-7 fabric; DUID-based DHCPv6 also collides. MSI reportedly
     patched in May 2026 but some units still ship with the original image. One-liner
     fix provided. Note: OEM imaging defect, not a GB10 hardware defect.
  2. NetworkManager on out-of-box DGX OS auto-creates DHCP profiles for every ConnectX
     QSFP interface; any port with carrier but no DHCP server (typical Spark-to-Spark
     direct cable) loops activate→ip-config→fail→retry every ~45 s, firing a desktop
     popup and flooding the journal. The multiple CX-7 fabric interfaces are exactly
     what triggers it (GB10-specific). `ip-config-unavailable` distinguishes "no cable"
     from "cable, no lease". Clean fix: `nmcli connection.autoconnect no` on the looping
     profiles; verify with `sleep 120; journalctl ... | grep -i fail`. Doesn't conflict
     with the clustering playbook. NVIDIA staff had already confirmed it's a
     NetworkManager message, not a connectivity error; this adds the full diagnosis
     chain. → [conjecture] (single source for the fix; corroborated by the existing
     [conjecture] NetworkManager fabric config finding S-forum-cx7-dual-setup).
- **Evidence cap:** Machine-id/CVE finding capped at [reported] (two independent OEMs/
  users + the official NVIDIA Security Bulletin). NetworkManager phantom-profile
  finding capped at [conjecture] (single source; no hardware verification available).

## 2026-07-19 — Forum ingest: Batch 22 — 3 new topics (all processed)

- **Sources:** 3 new forum sources registered (Batch 22) in `sources/README.md`:
  S-forum-ec-fan-rollback (377069), S-forum-nemo-rt (376248),
  S-forum-litellm-orchestrator (376407). 3 topic IDs added to
  `processed_topics.txt` (total now 400).
- **Topics found:** 3 new topics, all technically GB10-relevant (no social/buying/RMA
  to skip):
  - 377069 (EC firmware rollback fixes fan curve) — GB10 firmware-level thermal finding.
  - 376248 (Nemo-RT voice agent) — vLLM-based GB10 tool, native-FP8 + unified-memory
    rationale, ~20 concurrent calls on Spark.
  - 376407 (LiteLLM multi-model orchestrator) — single-Spark model lifecycle tool;
    thread also surfaces sparkstation.
- **Pages touched:**
  - platform-gb10 (new "Batch 22" ingest section — EC firmware 0x0300xxxx breaks fan
    curve → 96-97°C ACPI zones, inaudible fans; EC isolates fan control from OS
    (fancontrol/pwmconfig/nvidia-settings can't override); fwupdmgr downgrade to
    0x02004e18 fix; idle 60→32°C, load 35-37°C, 0% throttling, 120-125W/node @ 95%
    GPU util; avoid blanket fwupdmgr update afterward; first reported EC firmware
    *regression* on Spark; relationship to 0x03000508 "improves EC" update unresolved
    [conjecture]; fan control is EC-isolated, not OS-overridable [conjecture]),
  - containers-and-tooling (new "Batch 22" ingest section — harinezumigel-llm-stack
    LiteLLM+vLLM orchestrator for single-Spark multi-model lifecycle management
    [conjecture]; thread surfaces sparkstation (kshetrajna12/sparkstation) and
    reinforces existing single-tenant-per-node constraint; Nemo-RT Community voice
    agent — VAD+STT+LLM(Qwen3-8B-FP8 via vLLM)+TTS on one GPU, OpenAI Realtime
    API-compatible, ~20 concurrent calls on Spark, native FP8 + arm64 build
    [conjecture]),
  - sources/README, log.
- **Key findings:**
  1. EC firmware 0x0300xxxx breaks the fan curve on DGX Spark — ACPI zones hit 96-97°C,
     fans inaudible, case too hot to touch. The EC isolates fan control from the OS, so
     fancontrol/pwmconfig/nvidia-settings cannot override it. Fix: `fwupdmgr downgrade`
     the EC to 0x02004e18 (full procedure documented). After rollback: idle 60→32°C,
     load 35-37°C under vLLM (~120-125W/node @ 95% GPU util), 0% thermal throttling.
     Warning: don't run a blanket `fwupdmgr update` afterward (re-pushes broken 0x3).
     This is the first reported EC firmware *regression* on Spark and the first finding
     that fan control is EC-isolated (not OS-overridable). Single source (one cluster
     report spanning multiple nodes) → [conjecture]. Tied to the existing EC firmware
     lineage — the newer 0x03000508 branch reportedly *improves* EC stability, so the
     relationship between the broken fan-curve branch and the "improves EC" update is
     unresolved; users should test 0x03000508 before rolling back.
  2. harinezumigel-llm-stack is a thin LiteLLM+vLLM Docker orchestrator for managing
     multiple local models on a single Spark (inference/guard/coding/RAG) where memory
     precludes co-residency. Defines models in config.yaml + .env, reuses containers,
     exposes a single OpenAI-compatible endpoint. Thread also surfaces sparkstation
     (kshetrajna12/sparkstation) — a unified gateway for vLLM/SGLang/TensorRT-LLM.
     Reinforces the [proven] single-tenant-per-node constraint: multi-model on one Spark
     is lifecycle-management (start/stop/swap), not co-residency. → [conjecture].
  3. Nemo-RT Community is a real-time bilingual ES/EN voice agent (VAD+STT+LLM+TTS)
     co-located on one GPU, OpenAI Realtime API-compatible. On DGX Spark: ~20 concurrent
     calls, sub-second TTFA. GB10-relevant: native FP8 for the Qwen3-8B-FP8 LLM stage
     (via vLLM), 128 GB unified memory as the concurrency enabler, arm64 build = no
     cross-compile. Reference perf (RTX 4090): full stack ~21.5 GB, LLM 52 tok/s
     single-stream. Single source → [conjecture]. Marginal to core LLM-inference scope
     but registered because it exercises the native-FP8 + vLLM + unified-memory path.
- **Evidence cap:** All three findings capped at [conjecture] — single forum source each,
  no independent corroboration, no hardware verification available.

---

## 2026-07-19 — Scheduled forum ingest (Batch 23)

- **Date:** 2026-07-19
- **Source count:** 3 new forum topics processed (3 new sources registered: S-forum-sync-locale,
  S-forum-ec-fan-asus, S-forum-inkling).
- **Topic IDs processed:** 377079, 377044, 377238 (total processed_topics.txt now 403).
- **Pages touched:**
  - **platform-gb10** (new Batch 23 section — ASUS GX10 thermal throttling after EC 0x02000005 /
    UEFI 0x03000006 update; ACPI zones 96.6°C, GPU 85-90°C, SW thermal slowdown ~23.7s, HW ~4.7s,
    fans N/A, clocks 2385→2190 MHz, `tviol=1` continuous; corroborates S-forum-ec-fan-rollback on
    a 3rd OEM SKU → **EC fan-curve regression promoted [conjecture]→[reported]** across Gigabyte +
    MSI FE + ASUS GX10; root-cause narrows: EC 0x02000004 vs 0x02000005 fan-curve table
    byte-identical (48%@85°C, 54%@93°C, 68%@95°C, 100%@97°C) → regression likely SoC/UEFI
    interaction, not curve-table edit [conjecture]; first published GB10 fan-curve bytes
    [conjecture]; fwupdmgr downgrade unavailable for ASUS GX10 (no LVFS capsule exposed)
    [conjecture]; dgx-spark-fieldiag 2.0.4-1 packaging bug — ofed-scripts dependency has no
    installation candidate, blocks latest field diagnostics [conjecture]; existing Batch 22
    entry + sub-bullet promoted [conjecture]→[reported] with corroboration note; NVIDIA escalated
    internally, case 260716-000029),
  - **multinode-tp-and-networking** (NVIDIA Sync / Cluster Assistant fails "Software version"
    check on non-English locale — root cause: `apt-cache policy dgx-spark-ota-update-meta` parser
    looks for "Installed:" but localized output says "Installiert:" (de_DE.utf8) / "Installé :"
    (fr) → false "System Software Update Required" on a fully-up-to-date node; workaround
    `sudo update-locale LC_MESSAGES=en_US.utf8` (no reboot); suggested upstream fix: `LC_ALL=C`
    prefix or `dpkg-query -W -f='${Version}'`; hotfix reportedly pending from NVIDIA [conjecture];
    blocks cluster pairing — the prerequisite for all multi-node TP work — on non-English OEM
    images),
  - **roadmap** (3 new open problems: Inkling 975B/276B MoE bring-up not yet characterized —
    announcement only, 8× Spark cluster underway, no recipe/benchmarks; EC fan-curve root-cause
    isolation — EC table vs. SoC/UEFI interaction needs firmware-level isolation, would resolve
    the 0x0300xxxx attribution and tell ASUS GX10 owners if a SoC/UEFI-only rollback is viable;
    dgx-spark-fieldiag 2.0.4-1 ofed-scripts dependency gap — blocks latest field diagnostics),
  - **sources/README, index, log.**
- **Key findings:**
  1. **EC firmware fan-curve regression promoted [conjecture]→[reported].** The ASUS GX10 report
     (S-forum-ec-fan-asus) independently corroborates the Gigabyte/MSI FE finding
     (S-forum-ec-fan-rollback) with the same symptom fingerprint: ACPI zones 96-97°C, fans N/A,
     SW/HW thermal slowdown counters active under sustained inference. Three OEM SKUs now agree.
     This is the first [reported]-tier platform finding promoted by cross-SKU corroboration in
     sparkbase. NVIDIA has escalated internally (Neill, support case 260716-000029).
  2. **Root-cause narrows from EC table to SoC/UEFI interaction.** A static byte comparison of
     ASUS EC capsules 0x02000004 vs 0x02000005 shows the 7-step fan curve is byte-identical
     (targets: 48% @ 85°C, 54% @ 93°C, 68% @ 95°C, 100% @ 97°C). Since the curve table didn't
     change, the regression trigger is upstream of the curve bytes — a SoC/UEFI interaction, an
     earlier EC version, or an SKU-specific difference. This refines the original "0x0300xxxx
     broke the fan profile" attribution. Also: the `fwupdmgr downgrade` workaround is NOT
     available to ASUS GX10 owners (LVFS exposes no older capsule) — a workaround gap for one
     OEM SKU that the original finding didn't cover.
  3. **NVIDIA Sync locale bug blocks cluster pairing on non-English OEM images.** The Cluster
     Assistant's "Verifying Devices" step runs `apt-cache policy dgx-spark-ota-update-meta` over
     SSH and parses the human-readable output for an `Installed:` line — which is localized to
     `Installiert:` / `Installé :` on non-English locales, causing a false "System Software
     Update Required" error on fully-up-to-date nodes. Workaround: `sudo update-locale
     LC_MESSAGES=en_US.utf8`. This bites on Spark because OEM Sparks ship in many locales and
     cluster pairing is the gateway to all multi-node TP work. A hotfix is reportedly pending.
  4. **Inkling 975B / Inkling-Small 276B MoE announced — 8× Spark cluster bring-up underway.**
     New multimodal MoE family (975B/41B-active + 276B/12B-active, 1M context, text/image/audio/
     video). Registered as a roadmap open problem only — no GB10-specific config, quant recipe, or
     benchmark has been reported yet. Open questions: NVFP4 fit on 2× Spark, 1M context vs.
     unified-memory ceiling, MoE cudagraph wall, engine selection. Promote to a model page once a
     real bring-up with flags + tok/s lands.
- **Evidence cap:** All new findings capped at [conjecture] (single forum source each) except the
  EC fan-curve regression, which is promoted to [reported] via three independent OEM-SKU sources
  (Gigabyte, MSI FE, ASUS GX10) exhibiting the same symptom fingerprint — no hardware
  verification available, so [reported] is the ceiling per the analysis-agent stack.

## 2026-07-20 — Forum ingest: 1 NVIDIA DGX Spark forum topic (Batch 24)

- **Source:** 1 new forum thread (topic 376585, "6x GB10 Cluster w/ MikroTik CRS812 = 768GB RAM",
  category: projects). Registered as `S-forum-6x-cluster` in `sources/README.md`. Type `forum`
  → capped at `[conjecture]` (single source).
- **Pages touched:** multinode-tp-and-networking (new Batch 24 section — 6× GB10 cluster via
  MikroTik CRS812; b12x backend reportedly enables non-power-of-2 TP=6 on most models, a first
  for GB10 clusters where stock vLLM assumes powers-of-2; GLM-5.2 ~30 tok/s single-stream;
  cluster peak 800-1180 W; consistent with sublinear scaling between TP=4 ~22-24 and TP=8
  33-54 tok/s; no YAML/docker shared, claim unverifiable from post alone), benchmarks (1 new
  [conjecture] row: GLM-5.2 6× TP=6 ~30 tok/s), roadmap (new open problem: does b12x enable
  arbitrary non-power-of-2 TP on GB10 — 3-node previously required virtual-head padding).
- **Evidence cap:** [conjecture] — single forum source, no config/flags/NCCL verification shared,
  no benchmarking methodology described. Not promoted past [conjecture].

## 2026-07-20 — Forum ingest: 4 NVIDIA DGX Spark forum topics (Batch 25)

- **Sources:** 4 new forum threads registered in `sources/README.md` (Batch 25):
  `S-forum-inkling-nvfp4` (377306), `S-forum-kimi-k3-ceiling` (377091),
  `S-forum-intern-s2` (377342), `S-forum-pmu-amu` (377280). All type `forum` → capped at
  `[conjecture]` (single source each). No second independent source corroborated any finding,
  so nothing promoted past `[conjecture]`.
- **Pages touched:**
  1. **NEW PAGE: `wiki/models/inkling.md`** — Thinking Machines Inkling (975B/41B-active MoE)
     NVFP4 bring-up on 8× DGX Spark by greg190. The technically densest topic of the batch: real
     tok/s tables (25 tok/s c1 short → 13.5 tok/s @ 2048 ctx decode cliff; prefill 1,400–2,711
     tok/s), public recipe + 12 patches (`blockmos/inkling-sparks-gb10`), filed vllm#49049.
     Key GB10-specific findings: (a) **`tml_fa4` Sm120/Sm121 cute FA4 path has no paged-KV** →
     workaround re-gathers whole KV history per decode step (O(ctx)/token) → ~24 tok/s aggregate
     ceiling at real context — the load-bearing blocker for rel-bias/FA4-arch models on sm_121a;
     (b) **NVFP4 runs clean** (no dtype fallbacks); (c) **`LAMPORT_RS_SCONV=0`** escape hatch —
     Inkling's Lamport collectives require MNNVL/NVLink fabric, hard-error on RoCE (all GB10 has);
     (d) cudagraphs working after root-causing a Sm120 rel-bias boundary bug via GPU coredump;
     (e) MTP stuck at k=1 (60% draft acceptance); (f) five real sm_121a kernel bugs with one-line
     fixes (rel-bias q-row index clamp, phantom varlen tiles, fused_qkvr_prep stream race,
     mDynamicCausal NameError, is_split_kv unassigned); (g) UMA silent-corruption insight
     (out-of-range indexes don't crash, they land in another allocation); (h) `--gpu-memory-
     utilization 0.70` max (0.78 wedges), `--compilation-config` mode pin. OP parked Inkling in
     favor of M3 (42 tok/s single-user). Added to `index.md`.
  2. **attention-and-kv-cache** — new section: `tml_fa4` Sm120 path has no paged-KV (the
     load-bearing attention finding from the Inkling bring-up, with the O(ctx) regather
     workaround and the ~24 tok/s aggregate ceiling). Also notes the Sm80-inherited rel-bias
     path discards the relative-position bias → wrong outputs; score-mod
     `vllm_flash_attn/cute` is the intended sm12x route.
  3. **platform-gb10** — new Batch 25 section: (a) ARM PMU/AMU counters on GB10 — correct PMU
     event differs from ARMv8; Cortex-A725 capped to 1 GHz, X925 up to ~1375 MHz; community
     kernel module (CyrIng) implements correct event selection; corroborates P/E-core asymmetry;
     (b) UMA silent-corruption insight from Inkling bring-up (OOB GPU reads don't fault on UMA,
     they hit another allocation → diagnose with compute-sanitizer) — sharpens the existing
     proven "no discrete VRAM to fault on" finding.
  4. **multinode-tp-and-networking** — new Batch 25 section: (a) switch-less 5-node full mesh
     via MST sub-port splitting (break 4×50G → 2×50G per QSFP port → 6 RoCE interfaces for 5
     nodes, ~$800 optical cost vs MikroTik, first reported technique to push switch-less mesh
     past 4 nodes on GB10); (b) practical GB10 cluster ceiling math (~115 GB usable/node → 16
     nodes for 2.8T @ 4-bit, ~$100k, 2000-3200W; ~4 nodes for Opus-class; viable 200B-class
     alternatives list); (c) Inkling's Lamport collectives require MNNVL/NVLink → hard-error on
     RoCE, `LAMPORT_RS_SCONV=0` escape hatch — a new class of "designed-for-datacenter-fabric"
     model biting on GB10's RoCE-only interconnect.
  5. **benchmarks** — 5 new [conjecture] rows: Inkling NVFP4 8× TP=8 decode at short/2048 ctx
     (c1/c8/c32), prefill (1,400–2,711 tok/s); summary noting the long-context decode cliff and
     M3 comparison.
  6. **roadmap** — refined the existing Inkling 975B/276B open problem (975B now characterized on
     the model page; 276B on 2× Spark remains open); added 3 new open problems: (i) paged-KV
     support for `tml_fa4` Sm120/Sm121 cute FA4 path (the load-bearing blocker for rel-bias/FA4
     models on GB10); (ii) Intern-S2-Preview-397B on 4× Spark (no quant small enough for 2× yet,
     announcement only); (iii) MST sub-port splitting for switch-less 5-node mesh (verify on
     real hardware).
  7. **index.md** — added `wiki/models/inkling.md` to the Models section.
- **Topic 377091 (Kimi K3 ceiling):** mostly social/opinion/buying-advice (out of scope per
  AGENTS.md), but extracted the 3 durable technical nuggets above (MST 5-node mesh technique,
  cluster-sizing math, viable-model-class list). The "models will keep getting smaller" vs
  "frontier is growing" debate was explicitly excluded as out-of-scope opinion.
- **Topic 377342 (Intern-S2-Preview-397B):** model announcement with no GB10-specific
  config/recipe/quant — registered source and roadmap open problem only. No model page created
  (no bring-up data to distill).
- **Topic 377280 (PMU/AMU counters):** small but durable platform finding (ARM PMU event
  selection, A725/X925 clock facts) — placed on platform-gb10.
- **Evidence cap:** All new findings capped at `[conjecture]` (single forum source each). The
  Inkling bring-up is unusually technically dense (public repo, filed issue, concrete tok/s,
  GPU-coredump-root-caused bugs) but remains a single source — no independent corroboration, no
  hardware verification available. Per the analysis-agent stack, `[conjecture]` is the ceiling.
  The tml_fa4 paged-KV finding and the UMA silent-corruption insight are flagged as high-value
  verification targets for hardware agents in `roadmap.md`.

## 2026-07-21 — Forum ingest: 4 new topics (Batch 26)

- **Sources:** 4 new forum threads from forums.developer.nvidia.com. Registered as `S-forum-*`
  in `sources/README.md` (Batch 26). All type `forum` → capped at `[conjecture]`.
- **Pages touched:** attention-and-kv-cache (FlashInfer sparse_mla_sm120 mbarrier livelock on
  GB10 — root cause via cuda-gdb, Triton workaround validated 560+ sessions),
  multinode-tp-and-networking (3-node full-mesh guide, TP power-of-2 requirement, PP+MTP
  incompatible, LMCache for KV-cache node, fastsafetensors freeze, gpu_memory_utilization 0.8
  for PP, NCCL mesh merged to main, Qwen3.5-397B-A17B 3-node PP benchmarks),
  containers-and-tooling (community Spark Docker images lag upstream vLLM 0.25.1/NCCL 2.30.7),
  platform-gb10 (EC firmware 0x00000500→0x00000507 silent update failure, fwupdmgr get-results
  diagnostic), benchmarks (2 new [conjecture] rows: Qwen3.5-397B-A17B 3-node PP decode + prefill),
  roadmap (2 new open problems: FlashInfer livelock upstream fix/reproduction, 3-node PP vs
  TP=2 overhead measurement).
- **Topic 377417 (vLLM 0.25.1 + NCCL 2.30.7):** community Docker images lag upstream. Durable
  container ecosystem data point — users needing latest PP/mesh features must build or find
  non-standard images.
- **Topic 377334 (FlashInfer sparse-MLA livelock):** **the highest-value finding this batch.**
  Exceptionally well-evidenced single-source report: cuda-gdb device-side receipt showing
  mbarrier TRYWAIT spin-loop, journaled 30,000+ engine steps/rank, 8/8 reproduction at ≥60K
  cold-prefill, validated Triton workaround with 560+ clean sessions at no throughput cost.
  Placed on attention-and-kv-cache as a major GB10 kernel bug. Tagged [conjecture] (single
  source, no hardware verification) but flagged in roadmap as high-priority verification target.
- **Topic 365296 (3-node mesh):** comprehensive 3-node guide by eugr+dbsci with benchmarks by
  chunkai721 and field-report issues by jameslacroix. Multiple durable findings: TP requires
  power-of-2 (attention head divisibility), 3-node PP ~single-node speed, PP+MTP not supported,
  NCCL mesh merged to main, fastsafetensors loader freeze, gpu_memory_utilization 0.8 for PP
  stability, LMCache for dedicated KV-cache node (untested).
- **Topic 363464 (update loop):** EC firmware 0x00000500→0x00000507 fails silently —
  `fwupdmgr get-results` diagnostic shows the failure state. Overlaps with existing OTA loop
  findings; the new durable bit is the fwupdmgr diagnostic and specific firmware version range.
- **Evidence cap:** All new findings capped at `[conjecture]` (single forum source each). The
  FlashInfer livelock report is the most evidence-rich single source ingested to date (cuda-gdb
  device-side receipt, journaled step counts, multiple capture bundles, public evidence pack) —
  flagged for priority hardware verification. Per the analysis-agent stack, `[conjecture]` is
  the ceiling regardless of evidence quality within a single source.

## 2026-07-21 — Scheduled forum ingest: 2 new topics

- **Sources:** 2 new forum topics found by `scripts/fetch_new_topics.py`. 1 technically
  relevant (377375 — thermal zones under load), 1 skipped as non-technical (377428 —
  AirLLM theoretical speculation, replies all jokes, no new GB10 data beyond known
  128 GB unified memory / 4 TB SSD facts already in platform-gb10).
- **Sources registered:** 1 new source (S-forum-temps-normal, Batch 27). Both topic IDs
  added to `processed_topics.txt` (total now 410).
- **Pages touched:** platform-gb10 (3 new [conjecture] findings: sysfs thermal zone
  layout under load — zones 0/5 hottest at 94.6 °C, GPU ~10 °C cooler than CPU;
  `tegrastats` Jetson Orin Nano binary works on GB10 but adds no sensor mapping;
  GPU clock capping as thermal mitigation per wildpines.ai blog).
- **Findings:** The zone-0/5-hottest pattern corroborates the EC-fan-regression
  fingerprint (S-forum-ec-fan-asus: zones 0/5 → 96.6 °C) → the pattern is now [reported]
  across 3+ threads, though exact numbers from this thread stay [conjecture] (single
  source). The `tegrastats` portability from Jetson Orin Nano → GB10 is a new tooling
  note. The clock-capping mitigation is a new conjecture that a hardware agent could
  verify (tok/s-vs-°C tradeoff).
- **Evidence cap:** All new findings capped at `[conjecture]` (single forum source).

## 2026-07-22 — Scheduled forum ingest: 10 new topics (Batch 28)

- **Date:** 2026-07-22
- **Source count:** 10 new forum topics found by `scripts/fetch_new_topics.py`. 4 technically
  relevant, 5 skipped (social/buying/speculation/OS install), 1 already covered by existing
  source (376643 = same repo as S-forum-sparkdash, different forum post).
- **Sources registered:** 4 new `S-forum-*` sources (Batch 28): S-forum-uvm-livelock (377478),
  S-forum-sway-scanout (370458), S-forum-sparkdash-mia (377550), S-forum-realsense-d435 (351088).
  10 topic IDs added to `processed_topics.txt` (total now 420).
- **Pages touched:**
  - **platform-gb10** (new Batch 28 section — 3 findings: UVM page-migration livelock causing
    hard shutdown under sustained load, the "128 GB unified-memory cliff" — weights + KV cache +
    CUDA workspace share one pool, over-commitment causes hard-lock with no OOM-killer/no log;
    fix: `--gpu-memory-utilization` 0.85-0.92, don't co-load large models, leave ~10-15 GB free,
    platform firmware (BIOS/BMC) update, `nvidia-smi -pm 1` + `-pl` power cap, `-lgc` clock lock;
    PSU overheating variant; GB10B scanout carveout allocation failure (`memmgrAllocScanoutCarveout-
    RegionResources_GB10B`) in Sway compositor at 6K resolution — fails with <4 GB/122 GB used
    because UMA pool fragmentation prevents contiguous carveout; RealSense D435 USB disconnect
    on Dell GB10 — fixed by July 2026 firmware update),
  - **containers-and-tooling** (new Batch 28 section — sparkDash by MiaAI-Lab: second independent
    multi-Spark monitoring dashboard with LLM tok/s, SSH power controls, WoL, worker-node flag),
  - **sources/README**, **index**, **log**.
- **Key findings:**
  1. **UVM page-migration livelock is a distinct failure mode from OOM.** When weights + KV cache
     + CUDA workspace exceed the 128 GB UMA pool, the GB10 doesn't cleanly OOM — it hard-locks
     with no warning, no log, and the OOM-killer never fires. This is consistent with the existing
     [proven] "unified-memory OOM = hard reboot" finding but adds a specific mechanism (UVM
     page-migration livelock) and a practical mitigation (`--gpu-memory-utilization` 0.85-0.92,
     don't co-load, leave 10-15 GB free). The fingerprint (model-agnostic, dies only under work,
     worse with co-loaded models) is a useful diagnostic. → [conjecture] (single source, but
     mechanism is consistent with proven findings).
  2. **GB10B has a scanout carveout allocation path that can fail with abundant free memory.**
     `memmgrAllocScanoutCarveoutRegionResources_GB10B` allocates physically contiguous carveout
     from the UMA pool for display scanout buffers. At 6K resolution, multiple ~121 MB buffers
     need several hundred MB contiguous — UMA fragmentation at boot can prevent this even with
     <4 GB used. This is GB10B-specific (no equivalent on discrete GPUs where VRAM is separate
     from system RAM). → [conjecture] (single source, well-analyzed by parallelArchitect).
  3. **RealSense D435 USB disconnect on Dell GB10 is fixed by July 2026 firmware.** NVIDIA staff
     confirmed. Adds to the pattern of USB subsystem fragility on GB10 (USB2 fallback, XHCI HC
     died) that firmware updates address. → [conjecture].
  4. **sparkDash (MiaAI-Lab) is a second independent multi-Spark monitoring dashboard.** Distinct
     from the earlier sparkdash by brainchillz. Reinforces the pattern of community-built dashboards
     for multi-Spark ops. → [conjecture].
- **Evidence cap:** All new findings capped at `[conjecture]` (single forum source each). The UVM
  livelock finding is consistent with the [proven] "unified-memory OOM = hard reboot" finding but
  the specific mechanism (page-migration livelock vs. plain OOM) and the mitigation thresholds
  (0.85-0.92, 10-15 GB free) are single-source and not independently corroborated.

## 2026-07-22 — Scheduled forum ingest: Batch 29 — 4 new topics processed

- **Sources:** 4 new forum threads from forums.developer.nvidia.com (DGX Spark / GB10 category).
  Registered as `S-forum-6x-ring-rdma`, `S-forum-uefi-fw-fail`, `S-forum-serial-console`,
  `S-forum-sleep-disabled` in `sources/README.md` (Batch 29). All type `forum` → capped at
  `[conjecture]` (single source each). 4 topic IDs added to `processed_topics.txt` (total now 428).
- **Pages touched:** multinode-tp-and-networking (7 new [conjecture] findings from 6-node ring
  topology thread — RoCE L2-adjacency requirement, NCCL_IB_MERGE_NICS=0 + SUBNET_AWARE_ROUTING
  fix, NCCL channel→HCA round-robin topology-unawareness, GID table asymmetry, TCP fallback
  as stable workaround, nvidia-peermem modprobe failure + GDAKI/GPUNetIO hypothesis, Hunlx's
  3-node env recipe), platform-gb10 (3 new [conjecture] findings — UEFI firmware update
  stepping-stone requirement + dmidecode -t 45 diagnostic, serial console not supported,
  sleep/suspend disabled by default), benchmarks (2 new [conjecture] rows — Qwen3.6-35B-A3B
  NVFP4 6-node PP=6 TCP vs RDMA).
- **Headline finding:** 6-node DGX Spark ring topology thread (S-forum-6x-ring-rdma) is the
  most technically dense multinode thread in weeks. Three major findings:
  1. **RoCE RC QPs require L2 adjacency** — routed (L3) RDMA fails at the ibv_modify_qp verbs
     layer for non-adjacent ring node pairs. This is a fundamental RoCE protocol constraint,
     not NCCL-specific, and explains why official topologies stop at 3-node full-mesh.
  2. **NCCL_IB_MERGE_NICS=0 + NCCL_IB_SUBNET_AWARE_ROUTING=1 (patched NCCL) together fix
     6-node ring RDMA** — stock NCCL's round-robin channel→HCA assignment is not topology-
     aware and silently routes channels onto ports cabled to a different neighbor. Both
     flags are required together (merge stops virtual 400G bonding, subnet-aware routing
     picks the correct physical port per peer via GID/subnet lookup).
  3. **GPUDirect RDMA unavailable — nvidia-peermem refuses to insert** — `modprobe
     nvidia-peermem` fails with "Invalid argument" on kernel 6.17.0-1021-nvidia, zero
     dmesg diagnostic. NCCL logs "GPU Direct RDMA Disabled" for all HCAs. GIN_IB_GDAKI
     plugin suggests DOCA GPUNetIO/GDAKI may be the intended Grace-Blackwell GPU-NIC path.
     This directly explains why RDMA vs TCP is only ~7% faster (both host-staged without
     GPUDirect). First quantified RDMA-vs-TCP comparison on GB10.
- **Evidence cap:** All new findings capped at `[conjecture]` (single forum source each).
  The nvidia-peermem modprobe failure corroborates the existing [proven] "No GPUDirect RDMA"
  finding on platform-gb10.md but adds the specific failure mode and GDAKI hypothesis.
- **Skipped:** None — all 4 topics had at least marginal GB10 relevance. Topics 369350
  (serial console) and 377582 (sleep/suspend) are thin but definitive NVIDIA-staff-answered
  platform facts; registered for provenance and added as minor [conjecture] entries.

## 2026-07-22 — Laguna-S-2.1-NVFP4 first-party benchmark

- **Sources:** S-laguna-v251-bench (first-party), S-forum-laguna-dflash (forum)
- **Pages touched:** wiki/models/laguna-s-2.1.md (created), wiki/benchmarks.md (Laguna row added
  to main table), index.md (Laguna page listed), sources/README.md (two new sources registered)
- **What changed:** Created model page for Laguna-S-2.1-NVFP4 with full working config, boot
  timeline, and llama-benchy depth sweep results. Laguna-S-2.1 is a 117.6B MoE (8.5B active,
  256 experts, 48 SWA+global layers) running single-node TP=1 on vLLM 0.25.1 + FlashInfer nightly.
  Decode 22.6 tok/s (peak 32.7) with DFlash spec=7 — flat across 0–16K depth (SWA architecture
  prevents attention degradation). Prefill 3.2K–3.9K tok/s. DFlash acceptance low on prose
  (mean 22.6 vs peak 32.7), consistent with forum reports. Cold start ~15 min. PIECEWISE
  cudagraph only (DFlash + FlashInfer limitation). Registered as S-laguna-v251-bench [proven]
  and S-forum-laguna-dflash [reported].

## 2026-07-22 — Laguna-S-2.1-NVFP4 retired

- **Sources:** S-laguna-v251-bench (first-party)
- **Pages touched:** wiki/models/laguna-s-2.1.md (status → retired, quality assessment added),
  index.md (retirement noted)
- **What changed:** User subjective testing found Laguna-S-2.1 output quality on par with
  Qwen3.6-35B-A3B for prose and chart work — accurate long-context chart processing but lower
  writing quality than MiMo-V2.5 or DeepSeek-V4-Flash. At 22.6 tok/s with 69.3 GiB footprint it
  offers no advantage over Qwen3.6 TP=2 (67 tok/s, 23.4 GiB). Weights deleted (136 GB reclaimed),
  Docker image removed, recipe artifacts removed, coordinator registration deleted, sibling
  Conflicts= cleaned. No swapper recipe retained. Model page kept for reference with [proven]
  benchmark data and retirement rationale.

## 2026-07-23 — Forum ingest: Batch 30 — 3 new topics

- **Sources:** 3 new forum topics found by fetch_new_topics.py. All 3 technically relevant.
  3 new sources registered as `S-forum-*` in `sources/README.md` (Batch 30 section). 3 topic IDs
  added to `sources/processed_topics.txt` (total now 431).
- **Topics:**
  - 363863 (Mistral Small 4 119B NVFP4 on DGX Spark) — 67-post thread, highly technical. First
    confirmed working config for this model on GB10. Central finding: MLA head_size=320 rejected
    by all stock backends on SM121; TRITON_MLA (via eugr's spark-vllm-docker) resolves it.
  - 373995 (80 t/s with Qwen3.6-35B-A3B-FP8) — 2-post thread with a working 2× Spark TP=2 recipe
    via spark-vllm-docker run-recipe.sh, 75-80 tok/s output, detailed TTFT/prefill numbers.
  - 366858 (How to disable CX7 equivalent way to removing DAC?) — 2-post thread on CX7 DAC
    thermal/power penalty; software disable insufficient, only physical DAC removal brings
    temps down; dgx-spark-mlnx-hotplug package mechanism documented.
- **New wiki page:** `wiki/models/mistral-small-4.md` — full model page with MLA head_size=320
  wall, working recipe, benchmarks (5 independent reporters → [reported]), known issues
  (reasoning_effort bug, tool-calling PR #39217, Eagle/MTP not working, --shm-size 16g kernel
  crash), community Docker images, quality assessment.
- **Pages touched:** wiki/models/mistral-small-4.md (NEW), wiki/models/qwen.md (Qwen3.6-35B-A3B
  FP8 2× recipe [conjecture]), wiki/platform-gb10.md (CX7 DAC thermal penalty [conjecture],
  dgx-spark-mlnx-hotplug udev/ACPI mechanism [conjecture]), wiki/benchmarks.md (6 new
  forum-reported rows), sources/README.md, index.md, log.md.
- **Key findings:**
  1. TRITON_MLA resolves the MLA head_size=320 wall on SM121 for Mistral Small 4 — no
     VLLM_MLA_DISABLE=1 needed. [reported] via 5 independent forum users.
  2. Mistral Small 4 119B NVFP4 runs at ~28-33 tok/s single-stream on GB10, fits on a single
     node (~60 GB). [reported]
  3. --shm-size 16g causes a kernel crash on GB10 (independent of gpu-memory-utilization);
     must use 4g with max-num-batched-tokens 4096. [reported]
  4. vLLM 0.25.1 publishes native linux/arm64 images — no custom Avarok/eugr build needed for
     base vLLM on Spark anymore. [reported]
  5. Eagle/MTP speculative decoding does not work for Mistral Small 4 on GB10 as of vLLM 0.21.
     [conjecture]
  6. CX7 DAC cable causes ~6°C higher temps even after software mlx5_core unbind + PCI remove —
     only physical DAC ejection brings temps down. [conjecture]
  7. Qwen3.6-35B-A3B-FP8 on 2× Spark TP=2: 75-80 tok/s output, cold TTFT 0.68s (5K) / 8.49s
     (81K), prefix cache kicks in hard on 2nd runs. [conjecture]

## 2026-07-23 — Forum ingest: Batch 31 — 2 new topics

- **Sources:** 2 new forum topics found by fetch_new_topics.py. 1 technically relevant, 1
  skipped (model announcement, no GB10 specifics). 1 new source registered as `S-forum-*`
  in `sources/README.md` (Batch 31 section). 2 topic IDs added to
  `sources/processed_topics.txt` (total now 433).
- **Topics:**
  - 376722 (Spark-vllm-docker: Force rebuild) — 3-post Q&A thread about spark-vllm-docker
    build flags. Durable findings: `--rebuild-vllm` forces local rebuild (vs pulling
    pre-built image); `--use-wheels` uses prebuilt wheels instead of compiling from source;
    repo always builds from `main` (no pinned vLLM version). Single source → [conjecture].
  - 377762 (New Motif-3 Beta Release) — model announcement, no GB10-specific content (no
    flags, env vars, tok/s on Spark, quant formats). Skipped. Same model already skipped
    in Batch 28 (topic 377602).
- **Pages touched:** wiki/containers-and-tooling.md (spark-vllm-docker build flags
  [conjecture], sources + updated date), wiki/models/mistral-small-4.md (build flags
  detail added to existing spark-vllm-docker section [conjecture], sources), sources/README.md,
  index.md, log.md.
- **Key findings:**
  1. `--rebuild-vllm` flag forces local vLLM image rebuild in eugr's spark-vllm-docker
     instead of pulling pre-built. [conjecture]
  2. `--use-wheels` flag uses prebuilt wheels instead of compiling vLLM from source.
     [conjecture]
  3. spark-vllm-docker repo always builds from `main` — no pinned vLLM version tag, so
     images track vLLM HEAD at build time. [conjecture]

## 2026-07-24 — Forum ingest: Batch 32 — 7 new topics (4 processed, 3 skipped)

- **Sources:** 4 new forum sources registered (Batch 32) in `sources/README.md`:
  S-forum-m3-tp3, S-forum-vllm-containers, S-forum-laguna-quality, S-forum-solar-open2.
  7 topic IDs added to `processed_topics.txt` (total now 440).
- **Topics found:** 7 new topics. 4 technically relevant, 3 skipped:
  - 377689 (community extinction) — social. Skipped.
  - 377733 (Prep for RMA?) — RMA complaint. Skipped.
  - 374727 (Permanent entitlement) — entitlement. Skipped.
- **Pages touched:**
  - models/minimax (M3 NVFP4 TP=3 on 3× Spark — chthonic vLLM+b12x virtual sharding,
    3 head-node OOM fixes, NCCL LD_PRELOAD shim trap, cold power-drain bandwidth fix,
    EAGLE3 bf16-vs-NVFP4 dead-end, 200K ctx over RoCE — all [conjecture]),
  - multinode-tp-and-networking (baked LD_PRELOAD beats LD_LIBRARY_PATH, cold power-drain
    fixes stuck ib_write_bw 12.8→111.85 Gb/s, TP=3 bandwidth→concurrency not tok/s,
    Ray object store + memory monitor false OOM on UMA — all [conjecture]),
  - containers-and-tooling (NGC lags 2 versions, nightly wheel regression pipeline,
    --vllm-ref, --name multi-container, VRAM soldered — all [conjecture]),
  - models/laguna-s-2.1 (quality corroboration — good for reasoning+tools, fails
    generative tasks, ~20-30 tps [conjecture]),
  - benchmarks (Solar-Open2-250B INT4 on 2× Spark ~15 tok/s, pp2048 ~2227 tok/s
    [conjecture]),
  - sources/README, index, log.
- **Key findings:**
  1. MiniMax-M3 NVFP4 TP=3 works on 3× DGX Spark via Luke Alonso's chthonic vLLM+b12x
     virtual sharding commit (fb63c9a) — 64 attn / 4 KV heads made divisible by 3
     automatically. First reported TP=3 recipe for M3. [conjecture]
  2. Three undocumented head-node OOM fixes for Ray on UMA: --load-format safetensors
     (no GDS), --object-store-memory 1073741824 (Ray reserves 30% for unused plasma),
     RAY_memory_monitor_refresh_ms=0 (96% RAM is normal on UMA, not a leak). [conjecture]
  3. Baked LD_PRELOAD NCCL shim in community Docker silently overrides user-installed
     NCCL — beats both symlink swap and LD_LIBRARY_PATH. Always check LD_PRELOAD in
     container env. [conjecture]
  4. Cold power-drain (unplug bricks ~90s) fixes stuck ib_write_bw 12.8→111.85 Gb/s —
     warm reboot does NOT work. Same class as GPU clock wedge and CX-7 SlotPowerLimit
     throttle. Power-cycle first before debugging NCCL config. [conjecture]
  5. Going from 12→100 Gb/s link speed on TP=3 did not change single-stream tok/s —
     concurrency increased instead. Consistent with proven latency-bound cross-node
     decode. [conjecture]
  6. Solar-Open2-250B (250B-A15B MoE) INT4 on 2× Spark: ~15 tok/s decode, ~2227 tok/s
     prefill, flat across depths to 32K. New Korean government-backed model. [conjecture]

## 2026-07-24 — Forum ingest: 2 new topics (Batch 33)

- **Sources:** 2 new forum topics found. 1 technically relevant (Qwen3-TTS GGML crash +
  torch backend fix), 1 skipped (PSA about Discourse MCP — social/meta, replies reference
  already-sourced DSV4-Flash and Laguna-S recipes).
- 1 new source registered (Batch 33): S-forum-qwen3tts-ggml. 2 topic IDs added to
  processed_topics.txt (total now 442).
- **Pages touched:** platform-gb10 (GGML CUDA PDL crash on GB10 — kernels built against
  CUDA 12.8 / sm_120 produce invalid kernels on dispatch; PDL capability check is a red
  herring due to async CUDA errors; consistent with existing sm_121a targeting gap
  [conjecture]), containers-and-tooling (Qwen3-TTS torch backend workaround — drop [ggml]
  extra, force --qwen3_tts_backend torch, CUDA graphs work, TTFA 2.65s, steady-state
  RTF ~1.7; UMA audio tensor pinning tip [conjecture]), sources/README, index, log.
- **Key finding:** Qwen3-TTS GGML backend crashes on GB10 because qwentts-cpp-python
  wheels target CUDA 12.8 / sm_120, not sm_121a. Memory ops work, compute kernels fail
  on dispatch. The PDL (Programmatic Dependent Launch) error is a downstream symptom,
  not the root cause. Fix: force the torch backend (CUDA-graph-accelerated PyTorch, no
  GGML). This is the same CUDA 12.8-vs-13.0 / sm_120-vs-sm_121a architecture mismatch
  class already documented for vLLM FP4 CUTLASS and Triton ptxas, now confirmed in a
  third ecosystem component (GGML/qwentts.cpp). All [conjecture] — single detailed
  source + one corroborating reply. No new wiki pages created. No index changes needed.

## Forum ingest 2026-07-25 (Batch 34)

- 5 new forum topics found (2 technically relevant, 3 skipped: SSH config parser bug, macOS
  SSH tunnel manager, vision model recommendation thread).
- 2 new sources registered (Batch 34). 5 topic IDs added to processed_topics.txt (total now 447).
- **Headline finding:** stock `vllm/vllm-openai:latest` hangs silently during model load on GB10
  — reaches backend selection but never "Application startup complete"; root cause is no SM121
  support in stock image. This corroborates the well-documented stock-vLLM-on-sm_121 gap and
  reinforces that users must use GB10-tuned builds (spark-vllm-docker --tf5, CUDA 13/SM121 wheels).
  All [conjecture].
- Pages touched: containers-and-tooling (stock vLLM hang + LocateAnything-3B bring-up — ARM64
  wheel gaps for decord/deepspeed/bitsandbytes/liger_kernel, device_map='auto' UMA pitfall,
  MoonViT HF auth hang, FastAPI server pattern for non-vLLM models), platform-gb10
  (device_map='auto' slow on 128 GB UMA — UMA-specific HuggingFace pitfall), models/qwen
  (stock vLLM hang on Qwen3.6-35B-A3B-NVFP4 — image not flags is the failure), sources/README,
  index, log.
- Skipped: 378009 (NVIDIA Sync SSH config parser bug — client-side tooling, not GB10 inference),
  377913 (macOS menu-bar SSH tunnel manager — personal macOS tool, not GB10-specific),
  377759 (vision model recommendations — no durable GB10 flags/env/errors/numbers/quant formats).

## 2026-07-25 — Forum ingest: 5 new topics (Batch 35)

- **Sources:** 5 new forum topics found, all technically relevant. 5 new sources registered
  (Batch 35). 5 topic IDs added to processed_topics.txt (total now 452).
- **Sources registered:** S-forum-gridbook (PrismaQuant GridBook codebook quant plugin),
  S-forum-nfs-modelshare (NFS HF cache + docker save|ssh pattern), S-forum-mikrotik-cr804-042
  (CRS804 + FS breakout cable — NCCL stuck at 0.5 GB/s, cold power-drain fix), S-forum-ling3-flash
  (Ant Ling-3.0-Flash 124B-A5B announcement — weights pending), S-forum-woolyai (WoolyAI
  closed-source multi-agent inference stack — vendor benchmarks, no repro recipe).
- **Pages touched:** quantization-on-gb10 (GridBook — 41 codebook formats 1.781-6 bit, native
  FP8/NVFP4-grid dequant via tensor-core table lookup, ~10% decode / 30% prefill overhead;
  Qwen3.6-27B 5.5-bit KL 0.0049; Hy3-295B-A21B 2.9-bit single-Spark; MTP-head quant optimizer;
  GGUF IQ on vLLM found lacking; MXFP8 abandoned; REAP pruning ineffective on modern MoE — all
  [conjecture]), multinode-tp-and-networking (CRS804-4DDQ + FS QDD-400G-2QPC02 breakout — link
  healthy but NCCL/ib_write_bw stuck at ~0.5 GB/s, DCQCN throttling, cold power-drain fix;
  MikroTik auto-negotiate may need explicit bandwidth for ~20-24 GB/s on 4× clusters; NFS-share
  HF cache pattern with CX-7 IPs + fstab automount + docker save|ssh; sparkrun native NFS cache
  support — all [conjecture]), engines (WoolyAI multi-agent stack — closed-source, multi-model
  swap scheduler, no repro recipe, community-skeptic; PrismaQuant GridBook vLLM plugin — not a
  standalone engine, quant plugin extending PrismaQuant — all [conjecture]), benchmarks (3 new
  forum-reported rows for WoolyAI DSV4-Flash/Gemma-4-26B/Nemotron-3-Nano-Omni at C1/C4;
  Ant Ling-3.0-Flash 124B-A5B announced/upcoming — [conjecture]), roadmap (2 new open problems:
  verify GridBook native-dequant performance + quality claims on real GB10; re-ingest Ling-3.0-Flash
  when weights drop and benchmark NVFP4/AutoRound INT4 on single Spark), sources/README, index, log.
- **Headline findings:**
  1. **PrismaQuant GridBook** (tenari/RobTand) is the most technically significant single-Spark
     quant development this batch: a vLLM plugin exposing 41 codebook quant formats (1.781-6 bit)
     with dictionary entries constrained to the FP8/NVFP4 grid so dequant runs at full tensor-core
     speed via table lookup. Two HF checkpoints released (Qwen3.6-27B 5.5-bit, Hy3-295B-A21B
     2.9-bit). Claims ~10% decode / 30% prefill overhead and KL 0.0049 for the Qwen3.6-27B. If
     verified on real GB10, this is the most promising path to single-Spark 300B-class MoE serving
     — sub-NVFP4 codebook rates without sacrificing native tensor-core dequant. All [conjecture].
  2. **CRS804-4DDQ first-use failure** is the same cold-power-drain class as the proven ib_write_bw
     and GPU clock wedge fixes: link negotiates and ping/TCP/NCCL work, but RDMA throughput is
     stuck at ~0.5 GB/s (DCQCN throttling, packet_seq_err climbing) until a full AC power-drain
     forces the CX-7 firmware to apply correct settings. Warm reboot does not work. The
     mtk-hotplug-handler.sh script is a partial alternative for remote-only users.
  3. **Ant Ling-3.0-Flash 124B-A5B** (Ant Group/Alibaba) is a strong upcoming single-Spark
     contender: 124B total / 5B active MoE with hybrid-linear KDA:MLA attention (5:1), 256K native
     context. NVFP4 (≈70 GB) or AutoRound INT4 (≈62 GB) should fit on a single 121 GB Spark. Weights
     expected "after Aug 3." Queued for re-ingest.
  4. **WoolyAI** is a closed-source multi-model agentic inference stack with a resident-model-swap
     scheduler — a genuine use-case shape vLLM/SGLang don't natively serve. But the benchmarks are
     vendor-reported with no repro recipe, no launch command, no source code; community-skeptic.
     C1 per-request decode (14-23 tok/s) is slower than llama.cpp/DSpark; the C4 aggregate numbers
     (56-91 tok/s) are batch-amortized and don't exceed vLLM/SGLang at equivalent concurrency.
- All [conjecture] — single-source forum reports, no hardware verification available. No new wiki
  pages created. No index changes needed (all findings folded into existing pages).

## 2026-07-26 — Forum ingest: 4 new NVIDIA DGX Spark forum topics (Batch 36)

- **Sources:** 4 new forum topics from forums.developer.nvidia.com. 3 technically relevant, 1
  skipped (non-technical question with no answers). 3 new sources registered as `S-forum-*` in
  `sources/README.md` (Batch 36): S-forum-glm52-vision, S-forum-solar-open2-nvfp4,
  S-forum-typec-thermal. All type `forum` → capped at `[conjecture]` (single source each).
- **Pages touched:** benchmarks (Solar-Open2-250B NVFP4 W4A4 on 2× Spark — 15.8 tok/s decode c1,
  flat with depth to 32K; FP8 KV speed-neutral but doubles pool 10.17× concurrency; full recipe +
  flags: VLLM_NVFP4_GEMM_BACKEND=flashinfer-cutlass, VLLM_USE_FLASHINFER_MOE_FP4=1, util 0.90,
  vLLM v0.25.1 from source sm121, UpstageAI fork v0.22.0-solar-open2 — [conjecture]),
  attention-and-kv-cache (two durable findings: (1) hybrid linear attention (36/48 KDA layers)
  dodges KV-bandwidth wall — decode flat with depth, generalizes sparse/hybrid finding to 4th
  arch class; (2) FP8 KV is capacity lever not speed lever on hybrid-linear models, contrasting
  proven speed-lever finding on full-attention models — [conjecture]), platform-gb10 (USB-C PD
  firmware pending update causes overheating without load after July 23 OTA; 30-min full
  power-cycle forces pending update to apply; related to existing reboot-doesn't-complete
  USB-C PD finding — [conjecture]), engines (GLM-5.2-Vision-NVFP4 — frozen-backbone 49.5M-param
  projector maps MoonViT 1152→GLM 6144-dim; adaptive MTP dynamically switches 2–5 drafted tokens
  based on p2–p4 acceptance — new MTP regime on Spark, all existing recipes use fixed
  num_speculative_tokens — [conjecture]), roadmap (1 new open problem: adaptive MTP feedback-loop
  overhead on bandwidth-bound GB10 decode), sources/README, index, log.
- **Headline findings:**
  1. **Solar Open2 NVFP4 — linear attention makes decode ~free with depth on Spark.** The
     load-bearing result: 15.4 tok/s at 32K depth vs 15.8 at depth 0 (−2.5%), while every
     full-attention model on the same pair decays hard. This generalizes the proven
     sparse/hybrid-attention finding (Nemotron-3 Mamba-2, Holo hybrid, MSA sparse) to a 4th
     architecture class (KDA linear attention). The FP8-KV-capacity-vs-speed distinction is a
     new GB10 rule of thumb: FP8 KV for speed on full-attention, FP8 KV for capacity on
     hybrid-linear.
  2. **GLM-5.2-Vision + adaptive MTP — two firsts on Spark.** First vision-enabled GLM-5.2
     (frozen-backbone projector, no weight changes), and first adaptive (dynamic-depth) MTP
     recipe — all existing MTP on Spark uses fixed draft depth. The adaptive approach is
     theoretically sound but unbenchmarked on GB10; feedback-loop overhead on the bandwidth-
     bound decode path is the open question.
  3. **USB-C PD firmware thermal — a 3rd USB-C PD platform issue.** Pending PD firmware update
     not applying during OTA causes sustained heat at idle; 30-min power-cycle forces the
     update. This joins the existing reboot-doesn't-complete and ASUS GX10 PD capsule findings
     as evidence that the USB-C power-delivery subsystem is a recurring source of platform
     instability on Spark.
- All [conjecture] — single-source forum reports, no hardware verification available. No new wiki
  pages created. No index changes needed (all findings folded into existing pages).

## 2026-07-26 — Scheduled forum ingest: 3 new topics, 0 ingested

- **Sources:** 3 new forum topics scanned (378130, 378110, 378031).
- **Pages touched:** none — all 3 topics were non-technical for GB10 inference purposes.
- **Disposition:**
  - 378130 (Asus GX10 first boot): general onboarding/setup advice — no durable GB10 findings.
  - 378110 (Google Chrome now available for arm64): generic ARM64 software availability — not
    GB10-specific.
  - 378031 (MSI Edge Expert Nvme devices not found): hardware RMA troubleshooting (NVMe not
    detected in BIOS) — OEM-specific hardware failure, no inference-relevant content.
- All 3 topic IDs appended to `sources/processed_topics.txt` to prevent re-scanning. No sources
  registered, no wiki edits, no index changes.

## 2026-07-27 — Scheduled forum ingest: 3 new topics, 2 ingested

- **Sources:** 3 new forum topics found (378066, 378131, 376858). 2 new sources registered
  (Batch 37). 3 topic IDs added to `sources/processed_topics.txt` (total now 462).
- **Pages touched:** models/qwen (Qwen3.5-122B "king model" daily-driver consensus — 4
  independent users confirm as best single-Spark model [reported]; AutoRound int4 ~65 tok/s
  on 2× Spark, FP8 ~35 tok/s on 1×; sparkrun-recipes patched vLLM v26 5 lanes @ 256K 40+ tok/s;
  AutoRound loop tendency; DSV4-Flash single-Spark 45-50 tok/s; Laguna corroborates retirement),
  containers-and-tooling (sparkctl — config-driven multi-provider model serving CLI, YAML
  configs, load balancing for clusters, contextual data plane), benchmarks (4 new
  [conjecture] rows: Qwen 122B AutoRound int4 2× ~65 tok/s, Qwen 122B fp8 1× ~35 tok/s,
  Qwen 122B hybrid vLLM v26 40+ tok/s 5-lane, DSV4-Flash 1× 45-50 tok/s / 240 @ c16),
  sources/README, index, log.
- **Disposition:**
  - 378066 (Best Daily Single Spark Driver — King Model): INGESTED — 4 independent users
    confirm Qwen3.5-122B-int4 as community consensus single-Spark daily driver → [reported].
    Multiple tok/s numbers, recipe pointer (sparkrun-recipes), quant quality observation
    (AutoRound loops), DSV4-Flash alternative. Technically dense, GB10-specific.
  - 376858 (Built a tool for managing model serving on Sparks): INGESTED — sparkctl tool
    registered as S-forum-sparkctl. Config-driven orchestration, multi-provider, load
    balancing. Reinforces existing community-tool pattern. Single source → [conjecture].
  - 378131 (Local model running real-world brewing agents): SKIPPED — application showcase
    (home brewery with Mistral 119B on Spark, WhatsApp interface). No durable technical
    GB10 findings (no flags, env vars, errors, quant details, tok/s numbers). The only
    GB10-relevant detail (Mistral 119B running on Spark as daily driver) is noted in the
    378066 thread analysis as corroboration that Mistral 119B is in community use.
- **Evidence promotions:** Qwen3.5-122B-A10B-int4 as single-Spark daily driver → [reported]
  (4 independent users agree). No claims promoted past [reported] — no hardware verification.
- No new wiki pages created. No index changes needed (all findings folded into existing pages).
