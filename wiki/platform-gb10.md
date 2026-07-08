# Platform: GB10 / DGX Spark

> **area:** platform
> **status:** stable
> **evidence:** proven
> **sources:** S-xnode-cudagraph, S-m3-vision, S-nemotron-rpc, S-networking, S-spark-powercap, S-dgxspark-report, S-forum-clock721, S-forum-power-crash, S-forum-15w-loop, S-forum-60w-cap, S-forum-power-spec, S-forum-tma, S-forum-thermal, S-forum-cooling-cage, S-forum-gsp-timeout, S-forum-driver610, S-forum-headless-boot
> **updated:** 2026-07-08

The hardware facts every model bring-up assumes. Read this first.

## Foundational tenet: hardware parity (read before replicating any community finding)

**[proven]** The DGX Spark is a mass-produced, standardized machine. A given pair is bit-for-bit the
same GB10 / sm_121 / 121 GB-unified / CX7-fabric hardware every other DGX Spark developer has. There is
**no immutable factor** that separates one dev environment from another. When a forum/community recipe
reports a result (e.g. "36 tok/s") and it doesn't reproduce, the **null hypothesis is a software
difference on your side** — never "my box is special."

Software differences that DO cause non-reproduction (all fixable, none immutable):
- **Your own deviations** — the #1 culprit. Mutating the recipe (util, ctx, cudagraph mode, dropping a
  mod) then blaming the hardware. Reproduce the config **exactly** first; change one variable at a time
  only after the faithful run is characterized.
- **Environment you control** — desktop-vs-headless (costs ~10 GB unified → forces lower util; fix:
  `systemctl isolate multi-user.target`), leftover containers/page-cache eating the startup budget, a
  wedged power controller (`nvidia-smi` clock check), mDNS/mgmt-IP vs fabric-IP.
- **Build/version skew** — a mod patching a slightly different code version; a from-source build vs
  tested wheels; a silently-swallowed prewarm/patch failure. Verify the artifact, don't assume.
- **A genuinely hard-but-shared problem** — e.g. the cross-node cudagraph JIT desync
  (`[[wiki/cudagraphs-and-compile.md]]`). Hard ≠ hardware-specific; it's hit and solved in software
  (a prewarm). If someone's works and yours doesn't, find the software delta.

**The failure mode this tenet exists to prevent:** attributing a failure to a phantom immutable
"our-hardware-is-different" factor, which *stops the investigation prematurely*. The correct move is
always: run the exact recipe/build, isolate one variable at a time, and treat any gap as a software
delta to be found — because on identical hardware, it always is.

## The box

- **[proven]** **GPU:** NVIDIA GB10 (Grace-Blackwell), **aarch64/arm64**, compute capability **12.1
  (sm_121)**. Build/serve with **`TORCH_CUDA_ARCH_LIST=12.1a`** and, for native builds, NVCC gencode
  `arch=compute_121,code=sm_121` (llama.cpp: `-DCMAKE_CUDA_ARCHITECTURES=121a-real`).
- **[reported]** **SoC (per vendor/research report, S-dgxspark-report):** GB10 Grace-Blackwell superchip
  (`MediaTek AHJ11488B`), TSMC 3 nm. CPU = **20 ARM cores** (10× Cortex-X925 up to 3.9 GHz + 10×
  Cortex-A725), 8 MB L3 — the orchestration/preprocessing engine (a busy CPU steals the shared SoC power
  budget from the GPU, see the power-wedge issue below). CPU↔GPU share the LPDDR5X pool over
  **NVLink-C2C (~5× PCIe5 bandwidth)**. **140 W SoC TDP** from an external **240 W USB-C PSU** (leftmost
  port). Blackwell: 5th-gen Tensor / 4th-gen RT cores, ~1 PFLOP sparse FP4 (~1000 TOPS) — but decode is
  bandwidth-bound, so that compute rarely binds.
- **[proven]** **Memory:** **~121 GB unified** per node (CPU+GPU share it). No discrete VRAM. Two Sparks
  ≈ 242 GB, which is the whole reason multi-node exists here (188 GB models don't fit one node).
- **[proven]** **Memory bandwidth ≈ 270 GB/s.** This is the decode ceiling — single-stream decode is
  **bandwidth-bound**, not compute-bound. Empirics: bf16 dense decode of 22.3 GB weights = 7.6 tok/s
  (~theoretical 12 at 270 GB/s); halving weight bytes (bf16→FP8) gave a near-exact **2.08×** speedup.
  Rule of thumb: **fewer weight bytes/token ⇒ faster decode** — this is why low-bit quant wins even when
  the kernel is a slow weight-only decompress (see `[[wiki/quantization-on-gb10.md]]`).
- **[proven]** **No native low-precision compute.** GB10 has **no native FP4 compute and no native FP8
  *block-scale*** ("Your GPU does not have native support for FP4/FP8 computation"). Only *per-tensor /
  dynamic* FP8 hits a native CUTLASS path. Everything else (FP4, block-scaled FP8) runs as **Marlin
  weight-only decompress** — correct, memory-cheap, but compute-bound at high batch. Details on the
  quant page.
- **[proven]** **No GPUDirect RDMA.** Cross-node NCCL collectives are **host-staged** (bounce through
  host buffers + CPU-side progress) over the ConnectX-7 fabric. `ib_write_bw` is healthy (~111 Gb/s /
  ~13 GB/s) but collectives are not zero-copy. This single fact drives the cudagraph wall and the
  cross-node throughput hit — see `[[wiki/cudagraphs-and-compile.md]]` and
  `[[wiki/multinode-tp-and-networking.md]]`.

## Operating constraints

- **[proven]** **Single-tenant per node.** Unified memory + everything binding one serving port ⇒
  exactly one model server per node at a time. Stop the current unit before launching another. A
  swapper/supervisor should enforce this (`Conflicts=` on the systemd unit + a swap step).
- **[proven]** **Unified-memory OOM = hard reboot.** There's no graceful OOM-kill; over-committing memory
  wedges the box (no discrete VRAM to fault on). **Size conservatively** — set
  `--gpu-memory-utilization` with headroom, account for the OS + server process on the head, and for
  cross-node watch the asymmetry (head also runs OS + launcher). For llama.cpp, `--no-mmap` is mandatory
  to avoid the page-cache filling unified memory (see `[[wiki/llama-cpp-rpc.md]]`).
- **[proven]** **`--gpu-memory-utilization` is NOT a hard cap here.** On a discrete GPU, util × VRAM
  bounds allocation. On GB10 the KV-pool probe reads **free *system* RAM** and sees the whole 121 GB, so
  an engine can allocate well past `util × 121` (observed: Atlas at util 0.40 ≈ 48 GB "budget" still took
  ~68 GB). Real caps are engine-specific (Atlas `--high-speed-swap*`, or a docker `--memory` cgroup
  limit, which the free-RAM probe *does* respect) — see `[[wiki/engines.md]]`.
- **[proven]** **`nvidia-smi` reports memory as N/A.** Unified memory isn't a discrete pool — use
  **`free -h`** to see used/free. DGX OS release 7.5.0 (arm64); ConnectX hot-plug + telemetry are
  dgx-spark packages.
- **[proven]** **Standard container invocation:** `--gpus all --ipc=host --network host` + cache mounts
  (`~/.cache/huggingface`, `~/.cache/vllm`, `~/.cache/flashinfer`). Standard env:
  `TORCH_CUDA_ARCH_LIST=12.1a`, `VLLM_SKIP_P2P_CHECK=1`, `FLASHINFER_JIT_LOG_LEVEL=ERROR`
  (`HF_HUB_OFFLINE=1` when weights are cached).

## GPU power-controller wedge (the "14 W cap") — known firmware bug

**Symptom.** **[proven]** Under real load a GPU shows **~96% utilization but a stuck low power draw
(~12–14 W) and a clock frozen at one exact value** (observed **611 MHz**, max is 3003) with **zero
active throttle reasons** in `nvidia-smi -q -d CLOCK,PERFORMANCE` (not thermal, not SW power cap, not
app-clock). On a TP=2 pair this is brutal: the two GPUs run in lockstep, so the wedged node drags the
*whole cluster* to its clock — every model is slow, **consistently**, regardless of engine or quant.
(Found chasing why a model's single-stream was ~35 tok/s vs a forum's ~56: the **head** was wedged at
611 MHz/12 W while the healthy worker ran 2431 MHz/35 W and its clock *varied* normally.)

**Tell it apart from CPU/power arbitration.** A legitimately power-limited GPU shows a throttle flag and
a *fluctuating* clock. A wedged controller is **pinned to one exact value with no throttle flag** — that
pinned-exact reading is the fingerprint. (GB10 is a shared-SoC superchip, so heavy head-side CPU —
desktop GUI, serving stack — is a real but *secondary* drag; it is not the 14 W cap.)

**Root cause.** **[reported]** The GPU's power-management controller gets stuck in a low-power state —
triggered by a crash during memory load, a sleep/wake, an app crash, or a spontaneous firmware glitch.
(Corroborated by an external writeup, S-spark-powercap, plus first-party observation.)

**Workaround — full AC power-cycle of the affected node (a reboot does NOT fix it):**
1. Graceful stop + full **shutdown** (not reboot).
2. **Unplug the PSU from the wall ≥ 60 s** — a soft reboot leaves residual power and the controller
   stays wedged. This step is the whole fix.
3. Plug back in, power on. Only the wedged node needs it (diagnose per-node with `nvidia-smi` under load).

**Verify:** under load the node should draw real watts (tens of W) and clock toward ~2400 MHz, and
single-stream tok/s should jump.

**Status:** `open` (firmware-level; recurs). Diagnostic when stuck: `sudo nvidia-bug-report.sh`. Check
this **first** when cluster tok/s is mysteriously low across every model.

### Forum corroboration (2026-07-08 ingest)

The power-controller wedge is now **[reported]** (multiple independent forum sources agree, all
symptoms match the first-party finding above):

- **[reported]** GPU clock pinned at **721 MHz** / ~10 W / 96% util / 55°C / no throttle flag —
  `nvidia-smi -lgc 3003` has no effect, `Supported Clocks: N/A` (S-forum-clock721). Fix: unplug
  USB-C + AC power, wait, re-plug. Same fingerprint: pinned-exact clock, zero throttle reason.
- **[reported]** After an OOM/crash, GPU draws **5–9 W** and is restored **only** by unplugging the
  power brick (reboot does NOT fix) (S-forum-power-crash). The Asus Ascent variant shows the same
  behavior; healthy state draws ~70 W at ~2400 MHz under load.
- **[reported]** GPU trapped at **650 MHz / 15 W** with an artificial **50°C T.Limit** (chip at 44°C)
  and SW Power Capping counter accumulating µs (S-forum-15w-loop). Fix: unplug from wall ~1–2 min.
  Community diagnostic: `spark-doctor` CLI (joeynyc/spark-doctor) and `spark-gpu-throttle-check`
  (hoesing/spark-gpu-throttle-check) — both detect the wedge by sampling clocks under load.
- **[reported]** On the Asus GX10, GPU draws max ~60 W even with CPU idle, SW Power Capping counter
  active (S-forum-60w-cap). Forum consensus: the 140 W TDP is the **combined CPU+GPU** envelope; GPU
  typically sees 35–45 W during vLLM, 85–90 W peak in burn tests — the 60 W cap is normal platform
  behavior, not the wedge (distinct from the pinned-clock fingerprint).
**[conjecture]** A `spark-doctor` / `spark-gpu-throttle-check` script should be run on every new
bring-up to rule out the wedge before benchmarking (multiple forum users discovered the wedge
only after unexplained slow tok/s).

### Batch 2 forum ingest (2026-07-08)

- **[reported]** **NVIDIA official power spec** (S-forum-power-spec, MackenzieNVIDIA): peak total
  system power = **240 W**; GB10 SoC TDP (GPU+CPU) = **140 W**; remaining 100 W = ConnectX-7 + SSD +
  USB-C provisions. `nvidia-smi` wattage measures **GPU power only** (not total SoC).
- **[reported]** **TMA (Tensor Memory Accelerator) is NOT exposed on GB10** (S-forum-tma, s0ne):
  consumer-grade Blackwell (GB10, RTX 5090, RTX 6000 Pro) lacks TMEM — TMA is datacenter-only
  (B100/B200/GB200). Kernel developers targeting TMA on GB10 will find no `cp.async.bulk.tensor`
  instructions in SASS.
- **[reported]** **Overheating shutdowns during sustained GPU loads** (ComfyUI video gen ~10 min)
  affect specific units — NVIDIA could not reproduce across FE/OEM SKUs, recommends RMA for
  affected units (S-forum-thermal). Community cooling solutions: 3D-printed ducted cooling cage
  with Noctua 120mm fan brings idle GPU to ~40°C (S-forum-cooling-cage).
- **[conjecture]** **GSP_INIT_DONE timeout (Xid 119)** + SEC2 secure-boot timeout
  (`RmInitAdapter failed (0x62:0x65:2028)`) after OTA firmware update — GPU fails to initialize,
  `nvidia-smi` returns "No devices were found" (S-forum-gsp-timeout). PCIe enumerates, driver
  loads, but GSP firmware hangs at init. May require RMA.
- **[conjecture]** **Driver 610.43.02 + CUDA 13.3** works on Spark (82 W under vLLM, 66°C, 95%
  util) — requires disabling SecureBoot or enrolling MOK for signed driver (S-forum-driver610).
  Ubuntu 26.04 clean install with drivers 610 + CUDA 13.3 + ZFS also confirmed working
  (S-forum-ubuntu2604); CX7 power fix (15 W consumption) needed post-install.
- **[conjecture]** **Auto-power-on for headless** (S-forum-headless-boot): BIOS setting exists for
  "start on power" — set by default on DGX Spark FE; OEM variants may need manual BIOS config.
- **[conjecture]** **Kernel panic after dashboard update** (S-forum-kernel-panic): initramfs
  missing after kernel update DKMS failure; `GRUB_TIMEOUT=0` in headless mode blocks recovery.
  Fix: boot recovery media, `dpkg --configure -a`, rebuild initramfs, or set `GRUB_TIMEOUT=5`.
- **[conjecture]** **Dual DP-MST scanout fails** on driver 580.159.03 — single stream works, two
  simultaneous DP-MST streams fail (S-forum-dp-mst).
- **[conjecture]** **XHCI "HC Died"** with RealSense D435i 30fps depth+RGB streaming (S-forum-xhci)
  — USB subsystem stability issue.
- **[conjecture]** **MT7925e WiFi** cannot connect to any network after OOBE on some DGX OS builds
  — "Failed to set PTK to the driver" / "key addition failed" (S-forum-wifi-mt7925).
- **[conjecture]** **Soft lockup** in `nvidia_modeset` DisplayPort path during Xorg logout on
  kernel 6.17.0-1018-nvidia (S-forum-soft-lockup-dp).

## Reference cluster

Multi-node findings here assume **2× DGX Spark (GB10)** by **role**:

| Role | Runs |
|---|---|
| `<head>` | TP=2 head rank, single-node head models |
| `<worker>` | TP=2 worker rank, vision/omni single-node models |

Direct-cabled ConnectX-7 200G RoCE between them, one serving port per node. Substitute your own
hostnames/IPs/served-name aliases; the knowledge here is role- and hardware-relative, not tied to any
specific box. See `[[wiki/multinode-tp-and-networking.md]]` for the fabric setup.

## See also
`[[wiki/quantization-on-gb10.md]]` · `[[wiki/cudagraphs-and-compile.md]]` ·
`[[wiki/multinode-tp-and-networking.md]]` · `[[wiki/containers-and-tooling.md]]`
