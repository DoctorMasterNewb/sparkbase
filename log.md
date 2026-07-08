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
