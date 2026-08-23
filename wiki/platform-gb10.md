# Platform: GB10 / DGX Spark

> **area:** platform
> **status:** stable
> **evidence:** mixed
> **sources:** S-forum-update-loop, S-forum-temps-normal, S-forum-uvm-livelock, S-forum-sway-scanout, S-forum-realsense-d435, S-forum-6x-ring-rdma, S-forum-uefi-fw-fail, S-forum-serial-console, S-forum-sleep-disabled, S-forum-cx7-dac-power, S-forum-qwen3tts-ggml, S-forum-locateanything, S-forum-typec-thermal, S-forum-asus-fw-jul25, S-forum-comfyui-crash, S-forum-power-90w, S-forum-gpu-throttle-cmd, S-forum-driver580-173, S-forum-model-storage, S-forum-acer-thermal, S-forum-sm121-support, S-forum-170hx-spark, S-forum-xid31-yolo, S-forum-um-kernel-init, S-forum-cx7-pcie-power, S-forum-cooler-temps, S-forum-powerstress, S-forum-dashboard-fw-stale, S-forum-fan-firmware, S-forum-earlyoom-config, S-forum-cx7-idle-temp, S-forum-nondgx-os, S-forum-vllm-qemu, S-forum-cuda-single-ctx, S-forum-cx7-27w-benign, S-forum-thermal-freeze, S-forum-clock-energy-sweep, S-forum-xconfig-recovery, S-forum-fan-dpms, S-forum-driver595, S-forum-trtllm-readout, S-forum-power-mgmt, S-forum-wifi-mesh, S-forum-idle-lockup, S-forum-sparkup, S-forum-gsp-reboot-jul2026, S-forum-energy-telemetry, S-forum-asm2464pd-replug, S-forum-fe-thermal-rma, S-forum-fan-headless-boot, S-forum-suspend-fail, S-forum-hdmi-hotplug-ab, S-forum-usbc-dp-hpd, S-forum-gx10-fw-recovery, S-forum-uefi-capsule-password, S-forum-75w-crash, S-forum-fieldiag-signedby, S-forum-triton-sm121a, S-sm121-nvfp4, S-forum-513mhz-wedge, S-gb10-profile
> **updated:** 2026-08-23

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
- **[superseded]** ~~**No native low-precision compute.** GB10 has no native FP4 compute and no native
  FP8 *block-scale* ("Your GPU does not have native support for FP4/FP8 computation").~~ **Overturned
  2026-08-22 — and it should never have been `[proven]`.** The quoted string is a **vLLM kernel-dispatch
  log line**, not a hardware statement; it was promoted to a hardware fact and then to a page tenet.
  (superseded-by: the two claims below.) (S-sm121-nvfp4)
- **[proven]** **GB10 HAS native block-scaled FP4 and FP8 tensor-core compute.** sm_121 is in the
  **`compute_120f` family** (CUDA C Programming Guide Table 28: `compute_120f` ⇒ CC 12.0 *and* 12.1),
  and PTX ISA 8.8 Table 64 promotes `mma` with `.e2m1` + `.block_scale` + `.scale_vec::4X` — **NVFP4** —
  into that family's feature set. So the silicon has
  `mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::[2X|4X]` (**2× the Ada FP8 tensor-core rate,
  4× with an FP32 accumulator**, per CUTLASS), `kind::mxf4.block_scale`, and
  `kind::mxf8f6f4.block_scale` (**block-scaled FP8**). First-party confirmation on real hardware: the
  shipped `vllm-node-v0260` `_C` extension carries 1542 `SM120_*` symbols including
  `Sm120TmaWarpSpecializedBlockScaled…`, `_qutlass_C.abi3.so` carries the SM120 blockscaled NVFP4 set,
  and dense NVFP4 already auto-selects CUTLASS on this box with **no marlin env at all** (see the
  Qwen3.8-27B entry in `[[wiki/quantization-on-gb10.md]]`). (S-sm121-nvfp4)
- **[proven]** **The Marlin fallbacks are kernel coverage, not silicon.** In the same build,
  `_moe_C_stable_libtorch.abi3.so` contains only `sm100` `mxf4nvf4` — the **MoE grouped FP4 GEMM is
  not compiled for SM120/121**, which is exactly why *dense* FP4 runs native here while *MoE* FP4 falls
  to Marlin. Upstream CUTLASS ships the missing kernel
  (`examples/79d_blackwell_geforce_nvfp4_grouped_gemm`). What native FP4 does and does not buy — it
  moves prefill and the concurrency plateau, **not** single-stream decode, which stays bandwidth-bound —
  is on `[[wiki/quantization-on-gb10.md]]`. (S-sm121-nvfp4)
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
- **[conjecture]** **Silent idle hard lockup — LPI-3 deep-idle wake failure (distinct from
  OOM/thermal/wedge freezes)** (S-forum-idle-lockup, luis.poveda9321): a DGX Spark (ASUS GX10,
  DGX OS 7.5.0, driver 580.173.02, kernel 6.17.0-1029-nvidia) reproducibly hard-locks at
  **idle** — ~97% memory free (3.5 GB / 123 GB used), zero GPU workload, zero GPU processes,
  load average ~0.1, GPU at P8 / ~4W / 41°C. The same "zero forensic trace" signature as
  other GB10 freezes (no panic, no OOM-killer, no hung_task, no soft-lockup, no RCU-stall,
  no Xid, no NVRM error — kernel log ends mid-write on a routine `nvidia_ctl_close`), but
  **the trigger is the opposite of OOM**: the SoC is descending into its deepest idle state
  (LPI-3) with PCIe ASPM at default when it locks. An out-of-band vitals logger (fsync'd
  every 3s, outside journald) confirms the idle state to within ~15s of each freeze.
  **Only happens at idle, never under load.** 7+ occurrences across Aug 6–12 on one unit;
  a second user (icoicqico123) reports the same idle-lockup pattern with embedding models
  via the `transformers` library — the starting VRAM is much larger on Spark than x86
  (even for the same script), and a UMA-specific memory leak is absent on x86 CUDA. NVIDIA
  staff (aniculescu) confirms the OOM freeze is a known issue being worked on, but the
  idle/non-OOM variant is a **different failure mode** — the driver's memory-usage-based
  process killer can't engage when there's no memory pressure. `hung_task_panic`,
  `softlockup_panic`, `panic_on_rcu_stall` armed + kdump active — if the next freeze
  leaves any kernel code running, it should capture a panic + core dump. **Status:**
  `open` — no known workaround; suspected to be a CPU/PCIe idle power-state transition
  (LPI-3 wake) failing to complete. This is the **fourth distinct GB10 freeze mechanism**
  documented: (1) OOM/UVM livelock (memory pressure), (2) thermal shutdown (temp), (3)
  power-controller wedge (firmware), (4) idle deep-state wake failure (this finding).
  Relevant for always-on deployments where the Spark sits idle between inference bursts.
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
- **[conjecture]** **`cudaMemGetInfo` under-reports free memory on UMA when another CUDA process is
  resident** (S-forum-comfyui-optimized, Haidij): `cudaMemGetInfo` (the API behind
  `torch.cuda.mem_get_info()`) reports only memory **not currently allocated by any CUDA process**
  on the device — not the true free unified pool. When vLLM holds 34 GB, `cudaMemGetInfo` returns
  ~6 GB free even though 40+ GB of host unified memory is actually available. This causes
  applications using `cudaMemGetInfo` for memory decisions (e.g. ComfyUI's model offload logic)
  to needlessly offload to "CPU" — which on UMA is the *same physical RAM*. Fix: use
  `psutil.virtual_memory().available` instead (semantically the same pool on GB10). This
  generalizes to any multi-process UMA workload. See `[[wiki/containers-and-tooling.md]]`.

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

### Batch 86 forum ingest (2026-08-23)

- **[reported]** GPU clock pinned at **513 MHz** / ~13 W under load / no throttle flag — AC
  power-cycle fixes (S-forum-513mhz-wedge, christian.pappert). Same fingerprint as all prior
  reports (pinned-exact clock, zero throttle reason, ~3× performance loss). 6th independent
  forum source corroborating the power-controller wedge. Also on ASUS GX10 after update
  (rad777, 507–598 MHz on one of two units).
- **[reported]** **`dmesg` "Detected insufficient power on the PCIe slot (27W)" is from the
  Mellanox CX-7 NIC driver (`mlx5_core`), NOT the GPU.** The GB10 GPU connects via NVLink C2C,
  not PCIe. This message is **safe to ignore** for GPU clock issues (elsaco, S-forum-513mhz-wedge).
  Diagnostic clarification: don't confuse the CX-7 PCIe power warning with the GPU wedge.
- **[conjecture]** Enhanced `spark-gpu-throttle-check` fork adds NVML direct telemetry, throttle
  reason decoding, clock ramp-up timing, stability scoring, baseline comparison
  (`parallelArchitect/spark-gpu-throttle-check`, S-forum-513mhz-wedge).

### Batch 8 forum ingest (2026-07-12)

- **[reported]** **5 min power-off wait is sufficient** (S-forum-clock-5min, florin.andrei): the
  original thread's comments mentioned a 30 min wait; a follow-up confirms a **5 min** wait
  (power off, disconnect power brick both sides, wait 5 min, reconnect, boot) cleared the wedge.
  This corroborates the existing ≥60 s guidance and suggests the residual drain time needed is
  shorter than initially reported.
- **[conjecture]** **Power-drain method — no wait needed** (S-forum-clock-5min, 0rand): an
  alternative to waiting: disconnect the power brick from the AC socket (not from the unit),
  then **press and hold the power button 5–10 s** to drain the capacitors, then reconnect. The
  user attributes the root cause to the **PSU power-control circuits** getting stuck in a safety
  protocol — not the GPU or the unit itself. Single source; plausible mechanism consistent with
  the proven symptom, but the capacitor-drain technique is unverified by other sources.
  - **[conjecture]** Root cause hypothesis: the wedge is in the **PSU's power-control logic**,
    not the GPU silicon (S-forum-clock-5min, 0rand). This is a single-source forum hypothesis;
    the proven observation is that a full AC power-cycle clears it and a reboot does not.

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
- **[conjecture]** **MT7925e WiFi mesh network incompatibility — auth loop on mesh WiFi
  equipment** (S-forum-wifi-mesh, simply_today): on 2× new DGX Sparks, the MT7925e WiFi
  requires 8+ connection attempts to authenticate against mesh WiFi networks (both 2.4G
  and 5G), but connects immediately to a simple non-mesh router. Factory reset, system
  recovery reimage, and firmware updates did not fix the issue. The auth loop is
  **specific to the mesh WiFi equipment**, not the Spark hardware itself — but it
  affects both units identically. A second user (hoesing) reports that a **cheap/unshielded
  CX-7 DAC cable** in the ConnectX ports causes similar WiFi authentication failures (EMI
  interference), with two fixes: move the DAC to the outermost CX-7 port, or use the
  NVIDIA-recommended shielded DAC cable. This is GB10-specific: the MT7925e WiFi chip and
  CX-7 NIC are on the same compact SoC board, so high-speed 200G serdes traffic can
  interfere with WiFi RF if the DAC shielding is inadequate. The OP's CX-7 ports were
  empty during testing, so the mesh incompatibility is a distinct issue from the DAC EMI
  problem — but both point to MT7925e WiFi fragility on the Spark platform. See the
  existing MT7925e OOBE failure (S-forum-wifi-mt7925) and CX-7 DAC thermal/power findings
  (S-forum-cx7-dac-power).
- **[conjecture]** **Soft lockup** in `nvidia_modeset` DisplayPort path during Xorg logout on
  kernel 6.17.0-1018-nvidia (S-forum-soft-lockup-dp).

### Batch 3 forum ingest (2026-07-09)

- **[conjecture]** **ConnectX-7 bricked by unsolicited mlnx-fw-updater firmware flash** (S-forum-cx7-bricked):
  During a routine `apt install`, `mlx-fw-updater` auto-triggered a CX-7 firmware update (28.45.4028 →
  28.47.1088) without user consent, despite raising its own BME/DMA prerequisite warning. Both CX-7
  interfaces bricked — stuck in `pre-init / static_config_not_done`, error -110. System boots but CX-7
  non-functional. **Recommendation:** pin/disable the mlnx-fw-updater autoupdater to prevent unsolicited
  firmware flashes. Recovery may require warranty/RMA. (ASUS GX10, PSID NVD0000000087.)
- **[conjecture]** **Silent SDPA EFFICIENT_ATTENTION corruption on custom PyTorch sm_121 builds**
  (S-forum-sdpa-corruption): a popular community-built PyTorch base image (built from source for sm_121)
  ships with a numerically broken `EFFICIENT_ATTENTION` backend — output norms 1.5×–27× off from CPU
  reference, no NaN/Inf, silently corrupted. `MATH` and `FLASH` backends are correct on the same hardware.
  **Root cause is in the image build's gencode handling** (`NVCC_GENCODE=-gencode=arch=compute_121,code=sm_121`
  with no family fallback), NOT in PyTorch source (byte-identical source tree across versions).
  **NVIDIA's NGC PyTorch wheels are NOT affected** (`nvcr.io/nvidia/pytorch:25.12-py3` and `:26.03-py3`
  both produce correct EFFICIENT output). Lesson: prefer NGC wheels over community builds for sm_121.
- **[conjecture]** **ComfyUI SageAttention silently inactive** (S-forum-sage-attn): `--use-sage-attention`
  on DGX Spark may silently fall back to PyTorch attention if `python3.12-dev` is not installed —
  SageAttention's Triton JIT shim compile fails, and the failure is *graceful* (everything renders, just
  20× slower). SDXL 1024²: ~140s (broken) → 6–8s (fixed). Fix: `sudo apt install python3.12-dev`.
  DGX OS doesn't ship the Python dev headers by default.
- **[conjecture]** **nvcr.io/nvidia/vllm:26.06-py3 image broken** (S-forum-vllm-2606-broken): every
  OpenAI API request (including `/health`) returns HTTP 500 — `'_IncludedRouter' object has no
  attribute 'path'`. Root cause: `prometheus-fastapi-instrumentator` incompatible with `fastapi >= 0.137`.
  The 26.02-py3 image works. Regression in the 26.06 image's middleware.
- **[reported]** **OOM hang fixed by driver 580.159.03+** (S-forum-device-hang): earlier drivers let
  unified-memory OOM hang the entire device; driver 580.159.03+ kills the offending process instead.
  Update Spark to latest DGX OS + driver ≥580.159.03 to avoid hard hangs under memory pressure.
- **[conjecture]** **DGX Dashboard Updates page hangs after OTA 7.5.0** (S-forum-fwupd-mismatch):
  `fwupd` daemon and `libfwupd` library left at mismatched versions after OTA → `fwupd.service` fails
  → `fwupdmgr` hangs indefinitely → Dashboard API calls hang. Fix: align fwupd/libfwupd package versions.
- **[conjecture]** **GB10 UMA bandwidth community measurements** (S-forum-gb10-baseline):
  community probes (`uma_bw`) report **161 GB/s idle, 90 GB/s under load** (driver 580.142, CUDA 13.0)
  — lower than the ~270 GB/s theoretical. The probes also report **CPU read 7.6 GB/s, CPU write 63 GB/s**
  for the shared LPDDR5X pool. Peak BW not reported by the platform (memory clock N/A on HW_COHERENT_UMA).
- **[conjecture]** **torchaudio unavailable on ARM64 / CUDA 13** (S-forum-qwen-tts-arm64): no
  ABI-compatible `torchaudio` wheel exists for DGX Spark's CUDA 13 / SM 12.1 / aarch64 — blocks
  Qwen3-TTS and other audio models. `torchaudio` is deprecated and not included in NVIDIA PyTorch
  containers. Workaround: use PyTorch from pytorch.org instead of NGC containers.

### Batch 4 forum ingest (2026-07-10)

- **[conjecture]** **UMA mmap double-allocation causes OOM when loading models via HuggingFace
  transformers** (S-forum-qwen35-lora-uma, danielkreuzhofer): when loading safetensors via `mmap`,
  the memory-mapped pages (~67 GB) and the materialized CUDA tensors (~67 GB) compete for the same
  physical UMA pool — on Spark they don't live in separate pools like discrete GPUs. A 67 GB bf16
  model needs ~134 GB total (mmap + CUDA) → OOM kill at 119/134 ≈ 66%, exactly where the process
  dies (66% of 1,026 weight tensors = 677). `device_map="sequential"`, `offload_state_dict=True`,
  `SAFETENSORS_FAST_GPU=1`, and `load_in_4bit=True` (BnB quantization buffers OOM earlier at 4%)
  do NOT fix the fundamental mmap double-allocation on UMA. Workaround: monkey-patch `safe_open`
  with an _EagerSafeOpen wrapper that (1) loads tensors direct-to-CUDA, (2) eagerly loads+closes
  each shard instead of keeping all 14 file handles open, (3) evicts the page cache after each shard
  via `posix_fadvise(POSIX_FADV_DONTNEED)`. Peak memory drops to ~72 GB (one shard's transient
  mmap + accumulated CUDA tensors), leaving ~47 GB headroom for LoRA adapters, optimizer states,
  activations. `load_in_16bit=True` + `device_map={“”: torch.cuda.current_device()}` +
  `attn_implementation="sdpa"` needed for Unsloth FastModel on some setups.
  - **[conjecture]** **FSDP `from_pretrained` loads full model on every rank** (jesse75, same
    thread): full-weight continued pre-training with FSDP across 3 nodes — `from_pretrained` loads
    the full model on every rank before FSDP gets to it, eating 75 GB of the 128.5 GB UMA pool
    before training starts. Expert-level sharding works (11.55B params/rank confirmed) but the
    initial load wall remains. See related thread
    https://forums.developer.nvidia.com/t/363945.
  - **[conjecture]** CUDA 13.2 (nvcr.io/nvidia/pytorch:26.03-py3) breaks `adamw_8bit` optimizer
    → switch to `adamw_torch`. `TORCH_CUDA_ARCH_LIST=12.1` (not 12.0) for GB10 (sm_121).
  - **[conjecture]** Unsloth LoRA trained on Qwen3.5 MoE may fail to load in vLLM — vLLM's fused
    MoE LoRA expects per-expert tensors, Unsloth produces fused expert tensors. No simple patch;
    requires rewriting weight loading logic.
- **[conjecture]** **TCG OPAL password + UEFI admin password corrupted after unexpected shutdown**
  (S-forum-opal-uefi, cvella): a DGX Spark that shut down unexpectedly had its TCG OPAL
  self-encryption password and UEFI administrator password corrupted simultaneously. PSID reset
  recovered the drive, but UEFI admin password became blank and firmware capsule updates are
  locked out — cannot disable secure boot or authorize capsule updates post-reimage. Appears to
  be a UEFI corruption from the unexpected shutdown. Second identical Spark unaffected. **Status:**
  `open` — no known workaround for the firmware update lockout.
- **[conjecture]** **Capsule Update blocked by UEFI Administrator Password never set — no reset
  method, RMA required** (S-forum-uefi-capsule-password, burnsy56): after a standard
  `fwupdmgr upgrade` (EC 0x03000302→0x03000508, UEFI SoC 0x0200980f→0x02009b0b — the same
  firmware versions as S-forum-fw-july2026), the reboot stops at a blue "Capsule Update /
  Enter Admin Password" screen. The user never configured a UEFI Administrator Password.
  After 3 cycles of 3 failed password attempts each (including empty submissions), the
  system proceeded to normal login — but the capsule update status is unknown. NVIDIA
  Customer Care escalated via case #260816-000170; NVIDIA staff (aniculescu) confirms
  there is currently **no method to reset or clear the admin password** and recommends
  RMA. This is a distinct failure mode from S-forum-opal-uefi (which involved an unexpected
  shutdown corrupting an existing password): here, no password was ever set, yet the
  capsule update demands one. The firmware update packages involved (dgx-spark-ota-update-meta
  26.03.1→26.04.1, dgx-dashboard 0.23.3→0.29.1, nvidia-spark-ota-check 1.0.16-1 new) are
  standard DGX OS OTA updates. **Status:** `open` — users hitting this should contact NVIDIA
  support; do not attempt to bypass the password prompt. Corroborates the broader pattern
  of UEFI/firmware update fragility on DGX Spark (S-forum-opal-uefi, S-forum-gx10-fw-recovery,
  S-forum-uefi-fw-fail).
- **[conjecture]** **GB10 internal display controller has a 165 MHz max pixel clock** (S-forum-sunshine-rdp,
  LsDmTandAI): this limits headless remote desktop streaming via Sunshine — 4K@60 is impossible,
  1440p@120Hz is the best achievable. Relevant for users extending Spark beyond SSH/CLI to native
  desktop via Sunshine+Moonlight. Community repos: eelbaz/dgx-spark-headless-sunshine,
  seanGSISG/dgx-spark-sunshine-setup.
- **[conjecture]** **ONNX Runtime GPU device discovery fails on GB10** (S-forum-wan2gp-onnx,
  kdb8756): `onnxruntime` reports `device_discovery.cc:89 ReadFileContents Failed to open file:
  "/sys/class/drm/card0/device/vendor"` — GB10's sysfs layout differs from discrete GPUs. This is
  due to the newness of GB10 and is **safely ignorable** — ONNX functions normally despite the
  warning. PyTorch 2.9.1+cu129 confirmed working with Wan2GP on GB10.

### Batch 6 forum ingest (2026-07-11)

- **[reported]** **Random shutdowns after long uptime — thermal paste degradation** (S-forum-thermal-shutdown):
  multiple users report Sparks powering off randomly after weeks/months of continuous operation, with
  **no OOM or thermal logs** to explain the shutdown. One user (arctic.gus) found thermal paste
  **dried out** on both units after months of 24/7 use; CPU temp sensor was hitting 95°C regularly.
  Repasting + removing the outer shell (adding USB fans) brought idle to ~27°C, load to 65–73°C, and
  **eliminated all shutdowns** for 6+ weeks. The OS thermal sensor likely reports an **average of all
  cores, not the hot-spot** — one core corner may exceed 105°C while the sensor reads lower, inducing
  a silent thermal shutdown. Heatsink contact gaps/air bubbles observed on disassembly. **Second
  user** (Zatz): same symptom traced to a **PDU fault** — one Spark unable to exceed ~35 W before
  tripping/shutting down with zero logs; fix was unplugging PDU from wall + Spark for ~30 s.
  **Third user** (robin.s): environmental heat was a factor; stacking Sparks worsened it.
  - **[conjecture]** Same user (arctic.gus) reports the **GPU power-controller wedge** (see above)
    also stopped recurring after repaste + case removal — suggesting the wedge may have a **thermal
    root cause** in some cases, not purely firmware. This is a single-source observation; the
    proven firmware-level root cause stands, but thermal stress may be a contributing trigger.
- **[conjecture]** **No Wake-on-LAN support on DGX Spark** (S-forum-thermal-shutdown, peter.h177):
  the Spark does not support WoL. The only automated recovery for an unresponsive/shut-down unit is
  the **Auto Boot BIOS setting** (on by default on FE) combined with a **hard power cycle** (e.g.
  IoT relay on the power brick, driven by a monitoring RPi/ESP32 via GPIO). A watchdog script
  (pinging mDNS every ~5 min, triggering relay on consecutive failures) recovers both thermal
  shutdowns and OOM wedges. See also `[[wiki/platform-gb10.md]]` → auto-power-on for headless.
- **[conjecture]** **Nsight Systems remote profiling requires sudo access** (S-forum-nsight-remote,
  mt42): Nsight Systems GUI on a remote host (e.g. MacBook) connecting via SSH to a Spark requires
  the SSH user to have **passwordless sudo** — if `sudo` prompts for a password, Nsight's remote
  target init fails with "No root access: Superuser (sudo) access is required." Workaround: enable
  `NOPASSWD` in sudoers for the profiling user, or SSH as root (not recommended). DGX Spark's
  default sudo config prompts for a password.

### Batch 7 forum ingest (2026-07-11)

- **[conjecture]** **First-boot WiFi onboarding SSID never broadcasts on some units**
  (S-forum-onboarding, griffith.mark): on at least two separate DGX Sparks (one purchased six
  months apart), the advertised WiFi setup network (SSID + password on the included card) was
  **never transmitted** — no setup SSID visible from a Mac, regardless of power-cycles. The QR
  code on the card resolves to the **DGX Spark product page** (not a setup guide), offering no
  troubleshooting for the failed wireless path. **Workaround:** connect a monitor + keyboard —
  the first-boot wizard launches immediately and proceeds normally (user creation, networking,
  updates). A second user (amurnane123) reports WiFi onboarding worked flawlessly on 4 units
  (headless, power + ethernet only) — so the missing-SSID behavior is **not universal**, may be
  unit-specific or batch-specific. Status: `open` — no known fix for the missing SSID; NVIDIA
  has not commented.

### Batch 9 forum ingest (2026-07-12)

- **[conjecture]** **Reboot does not complete — requires USB-C cable removal**
  (S-forum-reboot-powercycle, jp176): `sudo reboot` shuts the machine down but it
  never powers back on — no power light, disappears from network. The only recovery
  is to physically remove the USB-C power cable, wait ~10 s, reinsert, and press the
  power button. A full `shutdown` followed by a power-button start also works, and a
  subsequent reboot may succeed normally — the behavior is **intermittent**. The unit
  is fully up to date including firmware; the latest update mentioned USB-C PD
  stability fixes. This is distinct from the GPU power-controller wedge (no clock
  pinning or low-power state observed) — it's a **power-delivery / soft-reboot
  completion** issue. Single source; may be related to the USB-C PD firmware area
  that NVIDIA is already patching. Status: `open`.

### Batch 14 forum ingest (2026-07-15)

- **[conjecture]** **HPC/slurm on DGX Spark — CPU P/E core topology matters for job binding**
  (S-forum-hpc-slurm, pavuknm): `numactl` reports one socket with Cortex-X925 (performance) and another
  with Cortex-A725 (efficiency) — 10 P-cores + 10 E-cores. Efficiency cores can bottleneck performance
  cores in MPI jobs; `--cpu-bind=map_cpu` in slurm can pin to specific cores once mapping is known.
  Suggested approach: two slurm partitions (all-core vs P-core-only). On conventional x86/PCIe systems
  only a few CPUs are needed for full GPU utilization, but NVLink-C2C may change this dynamic. No MPI
  library currently optimizes for ARM P/E core asymmetry. NVIDIA Deepops all-in-one slurm setup
  (login+ondemand+compute+slurm-master on one node) works on a single Spark after ansible playbook
  patching; enroot/pyxis containers are the most efficient way to test GenAI (TensorRT-LLM, vLLM).
- **[conjecture]** **CX-7 in switch topology needs special configuration** (S-forum-hpc-slurm,
  pavuknm): using ConnectX-7 with an external switch (vs direct-cable) requires additional
  configuration beyond the standard direct-cable setup. Referenced ServeTheHome article on GB10
  ConnectX-7 200GbE networking differences. RoCE (not real InfiniBand) — confirmed by bugsareyummy
  and dbsci. Relevant for HPC users planning >2-node clusters with switches.
- **[conjecture]** **Llama 3.2 3B full fine-tuning 8× slower than benchmark** (S-forum-llama32-finetune,
  arijitmukh007): measured 0.59 steps/s vs expected ~5 steps/s (benchmark claims ~80k tok/s peak
  fine-tuning). NVIDIA redirect to DGX Spark Performance FAQ + benchmarking guide — the gap is a
  known issue with a documented FAQ answer, not a new finding. Training throughput on GB10 is limited
  by the same bandwidth/compute constraints that bound inference; benchmark numbers assume specific
  configurations (batch size, sequence length, data pipeline) that may not match user setups.

### Batch 15 forum ingest (2026-07-15)

- **[conjecture]** **CX-7 ports are hot-pluggable — not visible until a cable is connected**
  (S-forum-cx7-hotplug, elsaco): on the ASUS GX10 (and DGX Spark), the ConnectX-7 interfaces
  may not appear in `ifconfig` or `lspci` when no cable is plugged in. This is by design: the
  CX-7 ports are hot-pluggable, controlled by the file `/etc/nvidia/cx7-hotplug-enabled`.
  The relevant dmesg marker is `mlx5_core 0000:01:00.1: Port module event: module 1, Cable
  unplugged`. To disable hot-plug behavior (force interfaces always-on): `sudo rm -f
  /etc/nvidia/cx7-hotplug-enabled` and reboot; restore the file to re-enable. This explains
  why users see "missing" CX-7 interfaces — the port is powered down waiting for a cable.
  **[conjecture]** mashie reports that disabling hot-plug (or plugging in a cable) causes
  idle power draw to nearly double — the CX-7 port draws significant power when active
  even with no traffic. Relevant to the 100W "rest" budget (CX-7 + SSD + USB, see power
  spec above).

### Batch 17 forum ingest (2026-07-16)

- **[conjecture]** **Ollama v0.30.x–v0.31.2 SSE parser regression on Nemotron-3-Super**
  (S-forum-nemotron-ollama, frank.stockmans): since Ollama adopted llama.cpp as backend (v0.30),
  a server-side SSE parser regression causes the stream to abort mid-response → no `finish_reason`
  on the client. Not config/model/temp/ctx related. Temp 0.3 only reduced rate; stock model failed
  identically. **Fix:** downgrade to Ollama 0.24.0 (last known-good). Verified 20/20 multi-tool
  requests clean, zero parse errors. v0.31.2-rc1 does NOT fix. Config: GB10, Ubuntu 24.04.4,
  kernel 6.17.0-1021-nvidia, CUDA 13.0, driver 580.159.03; `nemotron-3-super-512k` (Q4_K_M ~87 GB,
  524288 ctx). Related to existing llama.cpp sm_121 build + GGUF incompatibility fix
  (S-forum-nemotron-sm121). See `[[wiki/models/nemotron-3.md]]`.
- **[reported]** **NVFP4 on GB10 achieves only 42–48% of bandwidth-limited theoretical ceiling**
  (S-forum-nvfp4-broken, DropTheBeat): quantified across multiple forum measurements. For
  Nemotron-3-Super (12B active @ 0.5 bytes = ~6 GB/token), theoretical ceiling at 273 GB/s = ~45
  tok/s. Measured 19–22 tok/s = 42–48%. A well-optimized NVFP4 path should reach 60–80% (routine on
  GB10 in other configurations). The gap is software/kernel efficiency, not hardware bandwidth.
  See `[[wiki/quantization-on-gb10.md]]` → NVFP4 meta-analysis.

### Batch 19 forum ingest (2026-07-17)

- **[reported]** **USB3 SuperSpeed PHY not registered — all USB falls back to 480 Mbps USB 2.0**
  (S-forum-usb2-fallback, rstovall, elsaco, paulsc.liu, rob-engassist, al9999, pontostroy):
  on some FE DGX Sparks (kernel 6.17.0-1008-nvidia, EC FW 0x02004e12, SoC FW 0x02009418),
  all xHCI SuperSpeed ports are stuck in RxDetect — no USB3 device is ever detected.
  Debugfs confirms: `portsc = 0x000002a0 Powered Not-connected Disabled Link:RxDetect
  PortSpeed:0` on every controller (NVDA8000:00 through :04). Root cause indicator: no USB
  PHY provider registered in the kernel — `devm_usb_get_phy_by_phlite` finds no PHY;
  **MediaTek T-PHY (`phy-mtk-tphy`) has no ACPI binding and is not loaded**. The `uas` module
  is loaded but `usb-storage` claims devices because they enumerate at USB 2.0 speed (UAS
  requires SuperSpeed). Multiple independent users report the same "always falls back to
  480 Mbps" symptom, including devices rated for 20 Gbps. Not universal — elsaco's FE Spark
  enumerates USB3.2 Gen2x2 (20000M/x2) at full speed. The USB-C hot-plug workaround (unplug
  + replug after boot) works for some users but not all. Some enclosure chips (ASM2464) are
  reported as especially problematic. DGX Spark USB ports are USB4 (40 Gbps) capable.
  Status: `open` — no firmware fix confirmed yet. 7 users in the thread report the issue.
- **[conjecture]** **New FE Spark firmware available: EC 0x03000302→0x03000508, UEFI SoC
  0x0200980f→0x02009b0b** (S-forum-fw-july2026, elsaco): the EC update "improves the
  performance and stability of the Embedded Controller"; the UEFI/SoC update "improves the
  performance and stability of the SoC Firmware including UEFI and GPU". May not appear via
  `fwupdmgr` immediately — LVFS publication lagged the dashboard announcement. One user
  confirmed updating all 8 units successfully. Relevant for the USB2 fallback and
  power-controller wedge (both EC/firmware-level). Tracked as [conjecture] pending
  confirmation of which issues the firmware addresses.
- **[conjecture]** **DGX Dashboard OTA stuck in update loop — manual `apt upgrade` workaround**
  (S-forum-ota-loop, andybchen131, elsaco): the DGX Dashboard "System Update" can get stuck
  in a persistent loop — clicking update and auto-rebooting multiple times does not complete.
  Workaround: run `sudo apt update && sudo apt upgrade` (or `sudo apt full-upgrade` for held
  packages) from a terminal instead. The dashboard continues showing updates as long as
  packages are on hold. The diagnostic tool `nvidia-spark-ota-check` (`/opt/nvidia/spark-ota-check/check_ota_status.py`)
  exposes: `is-ota-available`, `torn-score` (0 = fully applied), `installed-versions`,
  `ota-versions`, `summary`. nv-docker-options package may be missing after the update.
  Related to the existing fwupd/libfwupd mismatch finding (S-forum-fwupd-mismatch).
- **[reported]** **ASUS Ascent GX10 BIOS/Firmware v0103 — PD firmware capsule fixes CX-7 link
  speed, lowers thermals, ~8-10 W less** (S-forum-asus-fw0103, brian322, trithemius):
  GX10 BIOS/Firmware v0103 includes SOC/0x305, EC/0x204, **PD/0x507** (USB-C PD 5.7),
  TPM/7.2.4.1. The PD capsule update failed via the System Upgrade GUI on both machines
  (manual shell fix: `./capsule_update.sh usbpd_5.7.cap`). After the update: **inter-Spark
  connection speed reportedly 4× faster**, MiniMax M2.5 tok/s up to 25-30 range, machines
  running cooler (case temperature noticeably lower). trithemius confirms: Z-Image in ComfyUI
  went from 75-80°C peak to 65-70°C max; **~8-10 W lower** power consumption measured via UPS.
  Two independent users agree on the thermal improvement → [reported]. The 4× link speed
  claim is single-source [conjecture]. **Caveat**: July 2026 system update on Asus also
  triggers the same OTA loop issue (btvd, robert287) — the Asus OTA pipeline lags NVIDIA's
  availability, causing a mismatch. The `nvidia-spark-ota-check` tool (above) helps diagnose.
- **[conjecture]** **Total host freeze (not process hang) during heavy multi-node TP=2 prefill
  on 2× Spark — thermal shutdown** (S-forum-host-freeze-tp2, heathen0711, jrsphd): serving
  Step-3.7-Flash-NVFP4 via spark-vllm-docker (TP=2, Ray), heavy non-cached prefill (long/
  resumed-chat prompts) caused one node to totally freeze — no ping, no SSH, no display —
  5 times across 2 days. Normal chat at 70% of 256K context did NOT trigger it. An initial
  UMA OOM bug (percentage-based `gpu_memory_utilization` with no floor) was fixed but the
  freezes continued. Every freeze left **zero forensic trace**: no OOM-killer, no kernel
  panic, no NVRM/Xid GPU fault, no hung-task/softlockup warning, kdump never produced a
  vmcore despite being enabled. Added `hung_task_panic`, `softlockup_panic`, bidirectional
  netconsole, and NCCL Flight Recorder — still nothing. The simultaneous wedge of interrupts/
  scheduler/NIC points to hardware/firmware-level lockup. Config: driver 580.159.03,
  kernel 6.17.0-1026-nvidia, CUDA 13.0. **Diagnosed as thermal shutdown** — the user ran the
  DGX Spark field diagnostic and it failed, prompting RMA. Units were in an air-conditioned
  room with 120mm 4k rpm fans; thermal can still occur on affected units (see existing
  [reported] thermal paste degradation / sensor blind-spot finding, S-forum-thermal-shutdown).
  The "zero forensic trace" pattern is a GB10-specific signature: a total hardware-level
  freeze leaves nothing for software watchdogs to capture. This is consistent with the
  existing finding that OS thermal sensors may report average not hot-spot temperature.

### Batch 23 forum ingest (2026-07-19)

- **[reported]** **EC firmware fan-curve regression on DGX Spark — ASUS GX10 corroborates the
  Gigabyte/MSI finding; root cause narrows from EC table to SoC/UEFI interaction**
  (S-forum-ec-fan-asus, giunta.francesco; corroborates S-forum-ec-fan-rollback):
  **Symptom:** on an ASUS GX10 (DGX Spark OEM SKU) after EC 0x02000005 + SoC/UEFI 0x03000006 +
  UEFI-device 0x00000507 firmware updates, sustained inference drives ACPI `thermal_zone0` and
  `thermal_zone5` to **96.6°C** (GPU 85–90°C), `nvidia-smi dmon` shows continuous `tviol=1`,
  SW Thermal Slowdown accumulates ~23.7 s and HW Thermal Slowdown ~4.7 s over a ~4.5-min run,
  GPU clocks drop from ~2385–2411 MHz to ~2190–2379 MHz, fans stay **N/A** in `nvidia-smi` and
  Linux exposes no controllable PWM device. Stopping the workload returns zones 0/5 to ~63.8°C.
  This is the **same symptom fingerprint** as S-forum-ec-fan-rollback (Gigabyte + MSI FE) —
  three independent OEM SKUs now report the regression → the core finding **promotes from
  [conjecture] to [reported]**. NVIDIA has escalated internally (Neill, case 260716-000029).
  **Root-cause nuance (new):** the OP performed a static byte comparison of the official ASUS
  EC capsules 0x02000004 vs 0x02000005 and found the recovered 7-step fan curve is
  **byte-identical** between the two versions — targets: **48% @ 85°C, 54% @ 93°C, 68% @ 95°C,
  100% @ 97°C**. Since the EC fan-curve table did not change, the regression likely originates
  from a **SoC/UEFI interaction** (or an earlier EC version / SKU-specific difference), not a
  curve-table edit. This refines the original S-forum-ec-fan-rollback root-cause attribution
  ("the `0x0300xxxx` EC firmware version broke the fan profile") — the table is unchanged, so
  the trigger is upstream of the curve bytes.
  **Workaround gap:** unlike the Gigabyte/MSI FE path, **`fwupdmgr get-releases` offers no
  downgrade candidate for the ASUS GX10** (device minimum versions are EC 0x01000000, SoC/UEFI
  0x02000000, but LVFS does not expose an older capsule). The `fwupdmgr downgrade` fix
  documented under S-forum-ec-fan-rollback is therefore **not available to ASUS GX10 owners**.
  **[conjecture]** Recovery for ASUS GX10 requires either an NVIDIA-patched firmware or LVFS
  exposing an older capsule. Status: `open` — NVIDIA engineering reviewing; no patched
  fan-curve/SoC firmware confirmed for any SKU.
- **[conjecture]** **Fan curve table values for ASUS GX10 EC** (S-forum-ec-fan-asus): the
  recovered 7-step autonomous fan curve targets are **48% @ 85°C, 54% @ 93°C, 68% @ 95°C,
  100% @ 97°C**. These are the first published fan-curve bytes for a GB10 OEM SKU. The 100%
  target is reached only at 97°C — consistent with the observed 96–97°C ACPI-zone plateau
  before throttling bites. Single source (static capsule analysis by one user) → [conjecture];
  a hardware agent capturing EC telemetry on a throttling unit could confirm the curve is
  actually being followed vs. ignored.
- **[conjecture]** **dgx-spark-fieldiag 2.0.4-1 packaging bug — `ofed-scripts` dependency has
  no installation candidate** (S-forum-ec-fan-asus): the latest Field Diagnostics package
  visible in the official CUDA APT repo (2.0.4-1) cannot be installed because it depends on
  `ofed-scripts`, which has no installation candidate in the configured official repositories.
  The older 1.0.9-1 installs fine. This blocks running the latest field diagnostics on
  affected units — a tooling gap that impedes triaging the thermal regression (and likely
  other hardware faults). Single source → [conjecture]. NVIDIA has been asked to either
  publish `ofed-scripts` or drop the dependency.

### Batch 22 forum ingest (2026-07-19)

- **[reported]** **EC firmware 0x0300xxxx breaks the fan curve on DGX Spark — roll back to
  0x02004e18 to restore aggressive cooling** (S-forum-ec-fan-rollback, veelacleave;
  corroborated by S-forum-ec-fan-asus on ASUS GX10 → promoted [conjecture]→[reported]):
  **Symptom:** after a recent EC firmware update, DGX Sparks (Gigabyte + MSI FE variants
  observed) run extremely hot — case temperature difficult to touch, ACPI zones hitting
  96–97°C, fans virtually inaudible under load. The **Embedded Controller (EC) isolates fan
  control from the OS**, so `fancontrol`, `pwmconfig`, and `nvidia-settings` cannot override
  the broken fan curve. **Root cause:** the `0x0300xxxx` EC firmware version broke the fan
  profile. **Workaround — roll back the EC firmware via `fwupdmgr`:**
  1. `sudo fwupdmgr get-devices` → find the **Embedded Controller** entry, copy its Device ID
     (long hex string, e.g. `de4d7b5fa8e558b2…`).
  2. `sudo fwupdmgr downgrade <DEVICE_ID>` → select the `0x02004e18` release (or latest
     `0x02…` available). For scripted/cluster use: `echo "1" | sudo fwupdmgr downgrade
     <DEVICE_ID> -y`.
  3. `sudo reboot` — the staged capsule is caught at boot, EC is re-flashed, system boots back.
  **Results (reported by OP across a multi-node cluster):** idle 60°C → ~32°C; under sustained
  vLLM inference (~120–125 W/node at 95% GPU util) package/GPU temps sit 35–37°C; **0% thermal
  throttling**, PROCHOT trips ceased. **Warning:** after rolling back, **do not run a blanket
  `sudo fwupdmgr update`** — it will push the broken `0x3` firmware back. JW2026 suggests
  hex-editing the staged version to `0x3` so `fwupdmgr` won't re-offer it until a fixed `0x4+`
  ships. **Caveat:** originally single source (one cluster report, though it spans multiple
  nodes); now corroborated by S-forum-ec-fan-asus on the ASUS GX10 SKU → promoted to
  [reported]. NVIDIA has escalated internally. Tied to the existing EC
  firmware lineage (cf. S-forum-fw-july2026: EC 0x03000302→0x03000508 is the *newer* branch
  that reportedly *improves* EC stability — the relationship between the broken `0x0300xxxx`
  fan-curve branch and the `0x03000508` "improves EC" update is unresolved; users on the
  broken branch should test the newer `0x03000508` before rolling back, or roll back if it
  doesn't help). Status: `open` — no NVIDIA-patched fan-curve firmware confirmed.
  - **[reported]** This is the first reported case of an EC firmware *regression* (vs. an
    improvement) on Spark, and the first finding that **fan control is EC-isolated and not
    OS-overridable** — `fancontrol`/`pwmconfig`/`nvidia-settings` are dead ends for fan tuning
    on GB10. Any "fix the fan curve" work must target the EC firmware (or, per
    S-forum-ec-fan-asus, the SoC/UEFI layer that drives it), not the OS. Promoted from
    [conjecture] to [reported] after independent corroboration on the ASUS GX10 SKU
    (S-forum-ec-fan-asus) — three OEM SKUs (Gigabyte, MSI FE, ASUS GX10) now exhibit the
    same regression fingerprint.

### Batch 21 forum ingest (2026-07-18)

- **[reported]** **OEM DGX Spark images ship with identical `/etc/machine-id` (and identical
  SSH host keys) — CVE-2026-24218, affects MSI EdgeXpert and ASUS GX10**
  (S-forum-machineid, ohaibuzzle, emptysands, JW2026): two MSI EdgeXpert DGX Sparks
  fresh from setup had byte-identical `/etc/machine-id`
  (`295f5139615f4bbaa29921a29574c7a3` on both), and therefore identical SSH host keys
  (machine-id seeds `ssh-keygen -A`'s key derivation). Googling that machine-id surfaced
  other users' journal logs — the cloned image wasn't sanitized at the factory. Root
  cause: OEMs clone a single DGX OS image across units without re-running
  `systemd-machine-id-setup`. emptysands links it to **CVE-2026-24218** (NVIDIA Security
  Bulletin: DGX Spark - May 2026; registered separately as S-forum-cve). JW2026 confirms
  the ASUS GX10 has the same issue — both OEMs clone. MSI reportedly patched in May 2026;
  some units still ship with the original DGX OS image as of 2026-07. Two independent
  OEMs/users → [reported]. **Why it bites on Spark:** SSH host-key collision enables
  silent on-path host impersonation (a real risk when Sparks are direct-cabled over the
  CX-7 fabric, see `[[wiki/multinode-tp-and-networking.md]]`); DUID-based stateful
  DHCPv6 address generation also collides; journald `user-<uid>.journal` paths collide
  if logs are ever aggregated. **One-liner fix:**
  `sudo rm -f /etc/machine-id /var/lib/dbus/machine-id && sudo systemd-machine-id-setup
  && sudo rm -f /etc/ssh/ssh_host_* && sudo ssh-keygen -A && sudo reboot`.
  Recommended for any freshly-unboxed OEM Spark before first network exposure. A
  postinst-style "if machine_id in (known-bad list): reset()" guard has been suggested
  but not shipped. Note: this is *not* a GB10 hardware defect — it's an OEM imaging
  defect that happens to affect the DGX Spark OEM SKUs (MSI EdgeXpert, ASUS GX10).

### Batch 29 forum ingest (2026-07-22)

- **[conjecture]** **UEFI firmware update fails when installed version can't bridge to current —
  stepping-stone firmware needed** (S-forum-uefi-fw-fail, dmaynor, lewdenlw): a DGX Spark's UEFI
  firmware update repeatedly fails during the capsule-on-disk boot flow (blue error box after a
  few minutes). `fwupdmgr get-history` shows the UEFI Device Firmware update `Update State: Success`
  for the SoC update but a separate UEFI device (PD firmware) reports a bad version `0x00000001`
  — lower than the minimum `0x00000400`. The root cause per community diagnosis (lewdenlw): the
  installed firmware version is too old to bridge directly to the target version; the capsule
  update path requires a **stepping-stone intermediate firmware** (e.g. manually download and
  flash version 0x0304 first, then update to current). A clean OS reimage from an updated image
  also avoids the issue. NVIDIA staff (aniculescu) diagnosed via `fwupdmgr get-devices` +
  `dmidecode -t 45` — the latter reveals the *actual* firmware versions vs what `fwupdmgr`
  reports: `dmidecode -t 45 | egrep -A4 "EC|PD|UEFI|FLASH"` shows separate entries for FLASH,
  UEFI, EC Firmware, and PD Firmware (PD0 FW1/FW2: 5.7, PD1 FW1/FW2: 0.0 — secondary PD
  controller unpopulated). The DGX Spark Field Diagnostic passes despite the firmware update
  failure (all tests OK: GpuStress, C2CStress, CpuStress, PowerStress, ThermalStress, FioSSD,
  MemStress). Single source → [conjecture]. **Why it bites on Spark:** this is a firmware
  update path gap that can leave a Spark stuck on an old UEFI/EC version — relevant to the
  existing EC firmware fan-curve regression (S-forum-ec-fan-rollback) and the USB2 fallback
  (S-forum-usb2-fallback), both of which need firmware updates to fix. Related to the existing
  [conjecture] fwupd/libfwupd mismatch (S-forum-fwupd-mismatch) and OTA loop
  (S-forum-ota-loop, S-forum-update-loop). The durable diagnostic is `dmidecode -t 45` to
  see actual firmware versions when `fwupdmgr` reports inconsistent state.

- **[conjecture]** **DGX Spark serial console not supported — removed from Porting Guide**
  (S-forum-serial-console, ragge, aniculescu/NVIDIA): the DGX Spark Porting Guide previously
  listed "Serial console support for flashing and remote management" under Remote Management,
  but NVIDIA staff confirmed this is **not supported and has been removed from the guide**.
  No serial console access is available from outside the machine (no network serial, no
  physical serial port exposed). Single source (NVIDIA staff confirmation) → [conjecture]
  for the platform capability claim; the removal from the guide is a documented fact. Relevant
  for headless/remote management planning — the only remote management paths are SSH over
  network and the DGX Dashboard.

- **[conjecture]** **Sleep/suspend is disabled by default on DGX OS — overrideable**
  (S-forum-sleep-disabled, allanmac, aniculescu/NVIDIA): sleep/suspend mode
  (`AllowSuspend`) is **disabled by default** on DGX OS installations. NVIDIA staff
  (aniculescu) confirmed this is by design and is **overrideable** if desired. The OP noted
  their MSI install had it disabled (and intended to re-enable). Single source (NVIDIA
  staff confirmation) → [conjecture] for the default-config claim. Relevant for 24/7
  inference deployments — the default disabled-suspend is correct for serving workloads,
  but users who intentionally want suspend (e.g. desktop use) need to know it's a deliberate
  default, not a bug.

- **[reported]** **Smart plug + Auto Boot is the only viable power-management solution for
  multi-node Spark clusters** (S-forum-power-mgmt, CosmicRaisins, jetspark, mashie,
  peter.h177): no sleep/suspend mechanism reliably preserves inference state, and
  stopping all LLM services + ConnectX-7 to save power, then restarting, takes the same
  time as a cold start. Multiple independent users converged on the same pattern: **full
  shutdown → smart plug cuts AC → smart plug restores AC on demand → Auto Boot BIOS
  setting powers the node back on**. This corroborates the existing "No WoL" finding
  (S-forum-thermal-shutdown) and the sleep-disabled-by-default finding above. Quantified
  power draw: a 4-node cluster idles at **238 W** (no switch) or **260 W** (with CRS504),
  and hits **800 W+** during inference. Clock-capping to 1400 MHz saves ~200 W across
  4 nodes for ~5-10% decode speed loss (prefill suffers more), consistent with the
  [reported] clock energy-efficiency sweep (S-forum-clock-energy-sweep).

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

### Batch 25 forum ingest (2026-07-20)

- **[conjecture]** **ARM PMU/AMU counters on GB10 Spark — correct PMU event differs from ARMv8;
  A725 and X925 max clock frequencies** (S-forum-pmu-amu, CyrIng): when building a kernel module
  to read PMC (Performance Monitor Counters) on the GB10's ARM SoC, the **correct PMU event for the
  Spark differs from the one for ARMv8** — the standard ARMv8 PMU event codes do not directly map.
  A community kernel module (CyrIng project, master branch) now implements the correct event
  selection. Two frequency facts surfaced: the **Cortex-A725 (E-core) is capped to 1 GHz**, while
  the **Cortex-X925 (P-core) goes up to ~1375 MHz**. This creates a scaling anomaly when normalizing
  PMC reads to max core frequency: X925 scales to ~5 GHz in the normalized reading while A725 stays
  at factory 2.8 GHz. Relevant to anyone doing low-level CPU performance profiling on GB10 (e.g.
  profiling the host-staged NCCL collectives, or the CPU-side preprocessing power budget). This
  corroborates the existing `[reported]` SoC topology (10× X925 + 10× A725) and the P/E-core
  asymmetry finding (S-forum-hpc-slurm). Single source → [conjecture].
- **[conjecture]** **On unified memory, out-of-range GPU indexes usually DON'T crash — they land in
  another allocation and silently corrupt** (S-forum-inkling-nvfp4, greg190): "looked like a race
  for days." The absence of discrete VRAM means OOB GPU reads don't fault — they silently hit
  another UMA allocation, producing corruption that masquerades as a concurrency bug. Diagnose with
  `compute-sanitizer` (which decoded the invalid read to the exact phantom block), not by waiting
  for crashes. This is a GB10-specific debugging insight that generalizes to any UMA kernel
  debugging — corroborates and sharpens the existing `[proven]` "unified-memory OOM = hard reboot /
  no discrete VRAM to fault on" finding above. See also `[[wiki/models/inkling.md]]` → kernel bugs.

### Batch 26 forum ingest (2026-07-21)

- **[conjecture]** **EC firmware update 0x00000500→0x00000507 fails silently — DGX Dashboard
  offers updates indefinitely** (S-forum-update-loop, podstawek): the DGX Dashboard repeatedly
  offers firmware updates even after multiple reboot cycles and manual `apt dist-upgrade`. The
  root cause is a **silent EC firmware update failure**: `sudo fwupdmgr get-results` shows the
  UEFI Device Firmware update (0x00000500→0x00000507) in `Update State: Failed` with error
  `failed to run update on reboot: expected 0x00000507 and got 0x00000500` — the capsule update
  didn't apply. `apt dist-upgrade` reports nothing to do (0 upgraded, 0 newly installed), so the
  package-level update is complete but the firmware capsule never installs. **Workaround**:
  full power-cycle — shutdown, unplug USB-C power adapter from the Spark, unplug from wall,
  press and hold power button for 10 seconds, wait 5 minutes, reconnect and power on. May need
  2–3 cycles. This is related to the existing `[conjecture]` OTA loop findings
  (S-forum-ota-loop, S-forum-fw-july2026) and the power-cycle workarounds documented for the GPU
  clock wedge (S-forum-clock-5min). The new durable bit: `fwupdmgr get-results` as a diagnostic
  to check if a firmware update actually failed, and the specific EC firmware version range
  (0x00000500→0x00000507). Single source → [conjecture].

### Batch 28 forum ingest (2026-07-22)

- **[conjecture]** **UVM page-migration livelock causes hard shutdown under sustained load — the
  "128 GB unified-memory cliff"** (S-forum-uvm-livelock, stuart.trusty): when weights + KV cache +
  CUDA workspace together creep past the 128 GB UMA pool, the GB10 does **not** get a clean OOM —
  it enters a **UVM page-migration livelock** that hard-locks the entire box with no warning, no
  log, and the OOM-killer never fires. Loading two big models (e.g. Qwen 3.5 35B + 27B) or the 122B
  alone leaves almost no headroom; a busy run tips it over. **Fingerprint:** model-agnostic, dies
  only when doing real work (never just loading), worse with multiple models co-loaded. **Fix
  (reported working by the forum user):** (1) cap `--gpu-memory-utilization` to **0.85–0.92**
  (not 0.94+), don't co-load multiple large models, leave **~10–15 GB free**; (2) update platform
  firmware (BIOS/BMC, not just OS — "everything's updated" almost always means OS+driver, the
  Spark's system firmware is separate); (3) `sudo nvidia-smi -pm 1` (persistence mode) +
  `sudo nvidia-smi -pl <watts>` (cap a few watts under max); (4) `sudo nvidia-smi -lgc <min>,<max>`
  (lock clocks to kill power transients). The memory cap (#1) was the big one. Single source
  (one detailed forum reply) → [conjecture], but the mechanism is consistent with the [proven]
  "unified-memory OOM = hard reboot" finding above. Also: a second user (oddjobsandservices)
  reports the same abrupt-shutdown symptom caused by **PSU overheating** — the PSU was laying
  flat on a carpeted floor, underside extremely hot; standing it on its edge fixed it. The machine
  has great cooling but the PSU does not. NVIDIA staff (aniculescu) recommends running DGX Spark
  Field Diag to rule out hardware faults.
- **[conjecture]** **GB10B scanout carveout allocation failure with Sway compositor at high
  resolution** (S-forum-sway-scanout, dlludllu, parallelArchitect): `NV_ERR_NO_MEMORY` (error
  0x51) from `memmgrAllocScanoutCarveoutRegionResources_GB10B` — a GB10B-specific display
  allocation path, not normal CUDA or system memory allocation. At 6144×3456@60Hz / 32bpp, each
  scanout buffer is ~121 MB; multiple buffers need several hundred MB of **physically contiguous**
  carveout from the UMA pool. The UMA pool can be fragmented enough at boot that this fails **even
  with <4 GB used out of 122 GB** (nvidia-smi confirms only lightweight display compositor
  processes: Sway 104 MB, Alacritty 83–184 MB each, Firefox ~375 MB — well under 2 GB total GPU
  memory). `WLR_SCENE_DISABLE_DIRECT_SCANOUT=1` does not fix it. Driver 580.142, CUDA 13.0,
  kernel 6.17.0-1018-nvidia. Suggested isolation: drop to 3840×2160 and check whether errors stop
  (tests whether 6K resolution is the trigger). Single source → [conjecture]. GB10-specific
  because the scanout carveout path (`memmgrAllocScanoutCarveoutRegionResources_GB10B`) is
  unique to the GB10B display controller and doesn't exist on discrete GPUs.
- **[conjecture]** **RealSense D435 USB disconnect on Dell GB10 — fixed by July 2026 firmware**
  (S-forum-realsense-d435, qobi): RealSense D435 (idVendor=8086, idProduct=0b07) connected to Dell
  GB10 running librealsense2 v2.56.5/v2.57.4 (RSUSB build) sometimes disconnects; unplug/replug
  does not fix, `rmmod`/`modprobe` of uvcvideo and related modules does not fix, only a reboot
  restores the device. dmesg shows repeated "Found UVC 1.50 device" entries. NVIDIA staff
  (aniculescu) confirmed the issue should be fixed with the July 2026 firmware update. This is
  related to the existing [reported] USB2 fallback finding (S-forum-usb2-fallback) and the
  XHCI "HC Died" finding (S-forum-xhci) — all point to USB subsystem fragility on GB10 that
  firmware updates address. Single source → [conjecture].

### Batch 27 forum ingest (2026-07-21)

- **[conjecture]** **sysfs thermal zone layout under load — zones 0/5 are the hot spots**
  (S-forum-temps-normal, DannyTup): on a Founders Edition Spark under GPU benchmark
  load (25 °C ambient), the 7 `/sys/class/thermal/thermal_zone*` entries (all generic
  `acpitz` type) read: **zones 0 & 5 ≈ 94.6 °C**, zones 1–4 ≈ 68–69 °C, zone 6 ≈ 71.6 °C.
  Temps fluctuated but spent most time between 90–95 °C. The GPU itself runs **~10 °C
  cooler than the CPU** under the same workload (sjug, corroborating the zone 0/5 = CPU
  hypothesis). Notably, CPU usage was ~0 % — the load was all GPU, yet the GPU was the
  "coolest" part. No thermal shutdown occurred overnight at these temps. This is
  consistent with the EC-fan-regression thermal zone fingerprint (S-forum-ec-fan-asus:
  zones 0/5 → 96.6 °C), but here the user had a desk fan blowing across the unit and the
  internal fan was ramping (just not enough). Single source for the exact numbers →
  [conjecture]; the zone-0/5-are-hottest pattern is now [reported] across 3+ threads.
- **[conjecture]** **`tegrastats` utility works on DGX Spark** (S-forum-temps-normal,
  elsaco): the `tegrastats` binary copied from a **Jetson Orin Nano** runs on GB10 and
  reports RAM, SWAP, per-core CPU freq/util, and 7 `acpitz` temperature readings.
  Sample idle output: `RAM 1431/124610MB (lfb 91x4MB) SWAP 0/16384MB` + 20 CPU cores
  at 0 % util, `acpitz@34.8C` … `acpitz@33.8C`. It does **not** reveal which physical
  sensor each `acpitz` zone maps to — all zones are generic `acpitz` type, so the
  zone-to-sensor mapping remains undocumented by NVIDIA. Tooling note: `tegrastats`
  adds nothing over `sysfs` for temps, but its RAM/SWAP/CPU-freq summary is useful as
  a one-line snapshot. Single source → [conjecture].
- **[conjecture]** **GPU clock capping as a thermal mitigation** (S-forum-temps-normal,
  digirho): an external blog (wildpines.ai, "Your DGX Spark Is Cooking Itself")
  recommends **capping the GPU clock** to reduce temperatures, claiming only a small
  performance loss. This is a different mitigation path from the EC-firmware rollback
  (S-forum-ec-fan-rollback) or the 3D-printed cooling cage (S-forum-cooling-cage) — it
  trades compute headroom for thermal headroom at the OS/driver level. No specific
  clock value or measured temp delta cited in the thread. Single source referencing
  a blog → [conjecture]; a hardware agent could measure the tok/s-vs-°C tradeoff.

### Batch 30 forum ingest (2026-07-23)

- **[conjecture]** **CX7 DAC thermal penalty — 6°C higher even after software disable**
  (S-forum-cx7-dac-power, meanaverage): with a QSFP DAC cable plugged in, Spark temperatures
  run ~6°C higher and power usage increases. This persists **even after** unbinding the mlx5_core
  driver (`echo "$dev" | sudo tee /sys/bus/pci/drivers/mlx5_core/unbind`) and removing the PCI
  devices (`echo 1 | sudo tee /sys/bus/pci/devices/$dev/remove`) for all four CX7 BDFs
  (0000:01:00.0, 0000:01:00.1, 0002:01:00.0, 0002:01:00.1). **Only physical DAC ejection** brings
  temperatures down. This means the CX7 PHY/serdes draws power whenever a cable is physically
  inserted, independent of driver state — software unbind is insufficient. Relevant for users running
  long hot jobs who don't need 200GbE (and have 10GbE redundant paths). **Now [reported] — see
  Batch 55 below** (3 independent sources agree: S-forum-cx7-hotplug, S-forum-cx7-dac-power,
  S-forum-cx7-idle-temp).
- **[conjecture]** **dgx-spark-mlnx-hotplug package manages CX7 via udev + ACPI hotplug driver**
  (S-forum-cx7-dac-power, raphael.amorim): the `dgx-spark-mlnx-hotplug` package installs udev rules
  (`/lib/udev/rules.d/90-mtk-hotplug.rules`) and a handler script
  (`/opt/nvidia/dgx-spark-mlnx-hotplug/mtk-hotplug-handler.sh`) that manage CX7 hotplug events via
  the `MTKP0001` ACPI platform driver (`cx7-pcie-hotplug`). The udev rules trigger on `ACTION=="add"`
  and `HOTPLUG_STATE=="plugin"` events. This is the software mechanism behind the CX7 hot-pluggable
  behavior documented in Batch 15 (S-forum-cx7-hotplug). Single source → [conjecture].

### Batch 33 forum ingest (2026-07-24)

- **[conjecture]** **GGML CUDA PDL crash on GB10 — kernels built against CUDA 12.8 / sm_120
  produce invalid kernels on dispatch** (S-forum-qwen3tts-ggml, swann.schilling): when running
  Qwen3-TTS with the GGML backend (`faster-qwen3-tts[ggml]` → `qwentts-cpp-python` from PyPI),
  model loading succeeds but the very first inference dies with:
  `CUDA error: unspecified launch failure` in `ggml_cuda_kernel_can_use_pdl` at
  `ggml/src/ggml-cuda/common.cuh:1602` (`cudaFuncGetAttributes(&attr, kernel)`). The PDL
  (Programmatic Dependent Launch) capability check is **not** the actual failing kernel — CUDA
  errors are asynchronous and sticky, so the next CUDA API call surfaces the error regardless of
  what it does. The real issue is a **CUDA architecture mismatch specific to GB10**: the default
  `qwentts-cpp-python` wheel is built against **CUDA 12.8** targeting generic **sm_120**, not
  GB10's **sm_121a**. Builds against 12.8 / sm_120 can load and copy weights (memory ops work)
  but produce **invalid kernels the instant they're dispatched**. This is consistent with the
  existing [reported] finding that Triton's bundled ptxas 12.8 lacks sm_121a
  (S-forum-nvfp4-ray) and that GB10 wants CUDA 13.0 with explicit sm_121/sm_121a targeting.
  A second user (Drew_the_AI_Guy) corroborates: the GGML kernel was likely compiled for an SM
  version the GB10 driver doesn't expose, or PDL metadata is incompatible with Blackwell.
  Separately, NVIDIA's forum has an open thread noting PDL behaves oddly on GB10 specifically.
  **Fix:** force the `torch` backend instead of `ggml` (CUDA-graph-accelerated PyTorch, no GGML
  path). See `[[wiki/containers-and-tooling.md]]` for the Qwen3-TTS workaround details. Single
  detailed source + one corroborating reply → [conjecture] (would be [reported] with another
  independent confirmation). GB10-specific because it's the sm_121a targeting gap, not a generic
  GGML bug.

### Batch 34 forum ingest (2026-07-25)

- **[conjecture]** **`device_map='auto'` is slow on 128 GB unified memory** (S-forum-locateanything,
  swann.schilling): on the GB10's 128 GB unified pool, HuggingFace's `device_map='auto'` runs a
  metadata analysis pass that can appear frozen for many minutes — the user observed the
  `LocateAnythingWorker` using `.to(device)` directly loads from cache in under a second instead.
  This is related to the existing `[conjecture]` UMA mmap double-allocation finding
  (S-forum-qwen35-lora-uma) — both are UMA-specific pitfalls in HuggingFace's device-mapping
  logic, where code designed for multi-GPU discrete layouts doesn't account for the unified pool.
  Single source → [conjecture]. See `[[wiki/containers-and-tooling.md]]` for the full
  LocateAnything bring-up.

### Batch 36 forum ingest (2026-07-26)

- **[conjecture]** **DGX Spark overheating without load after firmware update — pending USB-C PD
  firmware not installed** (S-forum-typec-thermal, unicornxoxo2): after a July 23 system update,
  a DGX Spark became extremely hot to the touch even when powered off (still hot after 20 min
  with power on but no workload). Running vLLM (Qwen3.6-35B-A3B) caused a spontaneous reboot after
  ~1h. On the next boot, the system displayed "an important update has been installed" for **USB-C
  PD (type-C power) firmware** — a pending update that had not installed previously. After a
  30-min full power-cycle (complete PSU disconnect), the unit returned to normal temperatures and
  stable operation. **Root cause appears to be a pending USB-C PD firmware update that failed to
  apply during the normal OTA cycle**, causing a power-delivery issue that manifests as sustained
  heat even at idle. This is distinct from the GPU power-controller wedge (no clock pinning
  observed) and the EC fan-curve regression (different firmware subsystem). It is most closely
  related to the existing [conjecture] reboot-doesn't-complete finding (S-forum-reboot-powercycle),
  which also implicates USB-C PD firmware — both point to the USB-C power-delivery subsystem as a
  recurring source of platform instability. **Workaround:** full AC power-cycle (unplug PSU ≥30
  min in this case) forces the pending PD firmware update to apply on next boot.
  - **[conjecture]** `nvidia-smi -lgc 0,2000` (clock cap to 2000 MHz) was suggested as a
    thermal-mitigation workaround by paulsc.liu; the OP attributed the issue to the firmware
    update, not GPU clocks, and the 30-min power-cycle resolved it. The clock-cap approach was
    not needed in this case but remains a general thermal workaround. Single source → [conjecture].
  - NVIDIA staff (Neill) requested version numbers for both the July 23 update and the type-C
    firmware update — not yet provided. Status: `open` — version tracking pending.

### Batch 40 forum ingest (2026-07-29)

- **[conjecture]** **GPU power spike trips overcurrent protection — distinct from the
  power-controller wedge** (S-forum-comfyui-crash, jas.burton): during ComfyUI LTX Video
  generation, the GB10 jumps from idle (~14 W) to full load (~85 W) instantly when denoising
  kicks in. This **6× power transient** appears to trip overcurrent/power-delivery protection
  in the compact Spark chassis, causing an **immediate hard shutdown** — no OOM, no thermal
  flag, no CUDA error, no log entry in `dmesg`/`journalctl`/ComfyUI. Peak temp was only 78 °C
  (not thermal). Only 49/119 GB RAM used (not OOM). `journalctl` shows "corrupted or uncleanly
  shut down" after reboot. **This is a different failure mode from the power-controller wedge**
  (which pins the clock at a low value with no throttle flag) — here the GPU was running
  normally at ~85 W then instantly died from the transient. Config: driver 580.126.09,
  CUDA 13.0, PyTorch 2.10.0+cu130, kernel 6.14.0-1015-nvidia.
  - **[conjecture]** **Fix — clock cap to 2100 MHz** (`sudo nvidia-smi -lgc 300,2100`):
    limits max GPU clock to 2100 MHz (down from default 2418 / boost 3003), keeping power
    draw at ~50 W instead of 85 W. 1800 MHz was "rock solid," 2100 MHz is the "sweet spot."
    Note: `nvidia-smi -pl` (power limit) shows N/A on GB10, so **clock capping is the only
    way to control power**. A second user (frozenace88) confirms the clock cap stabilized
    their system: 79 °C, 69.77 W, 2086 MHz, 96% util — stable. A third user (knitvoger1)
    reports `nvidia-smi -lgc 300,2100` succeeds but `nvidia-smi --query-gpu=clocks.applications.graphics`
    still shows 2418 MHz — the lock may not take effect on all units/firmware versions.
  - **[conjecture]** **Fix — disable swap** (`sudo swapoff -a` + `vm.swappiness=10`): on
    unified memory, swap is actively harmful — when the system approaches memory limits, the
    OS pages to swap → saturates the system bus → display times out → total lockup (instead
    of a clean OOM kill). This corroborates the existing swapoff guidance
    (S-forum-llm-comfyui, S-forum-uvm-livelock) with a specific diagnosis of the mechanism
    (bus saturation → display timeout → lockup). See also `[[wiki/engines.md]]`.
  - **[conjecture]** **`CUDA_CACHE_MAXSIZE=4294967296` (4 GB)** — expanding the PTX→SASS
    kernel compilation cache from the default ~256 MB to 4 GB gives a **3× speedup on
    reruns** (kernel compilation amortized). Relevant for any CUDA workload with JIT
    compilation (ComfyUI, vLLM with Triton kernels, etc.).
  - **[conjecture]** **`--highvram` is a trap on unified memory** (S-forum-comfyui-crash,
    jas.burton): the ComfyUI flag sounds like "use all the VRAM!" but on Spark it forces
    every model to stay pinned on GPU simultaneously. With LTX Video (15 GB) + VAE (2.3 GB)
    + ReActor + RIFE, that pushes past 80 GB and OOMs — especially with an LLM co-loaded.
    Working flags: `--listen 0.0.0.0 --bf16-unet --bf16-vae --bf16-text-enc
    --use-sage-attention` (no `--highvram`, no `--disable-async-offload`, no `--gpu-only`).
    Let ComfyUI's async weight offloader do its job — on UMA the "offload" is basically a
    pointer update, nearly free. This corroborates the existing ComfyUI UMA findings
    (S-forum-comfyui-optimized, S-forum-comfyui-container). See
    `[[wiki/containers-and-tooling.md]]`.
  - **[conjecture]** **ComfyUI has no real multi-GPU/multi-node support** (gpieceoffice):
    the `worksplit-multigpu` branch on ComfyUI GitHub loads the model on each GPU and
    computes in parallel (not model-parallel split). It was abandoned ~end of 2025.
    Multi-node model loading is not feasible in ComfyUI.

### Batch 41 forum ingest (2026-07-29)

- **[conjecture]** **Hard power-off under sustained GPU load at ~90W — persists after full platform
  firmware update** (S-forum-power-90w, pacardenaz): a DGX Spark (FE, BIOS 5.36_0ACUM018,
  SOCFW 2.155.11, EC 3.5.8, USBPD 0.5.22, OTA2607, driver 580.159.03) hard-powers-off whenever
  sustained FP16 matmul load pushes GPU power above ~90W — fully reproducible with a stepped
  workload (4096→20480 matrix size). Key diagnostic findings:
  - **Dies before thermal protection engages.** At free clocks no thermal reason bit is ever
    asserted — it goes from throttle 0x0 straight to power-off, dying at GPU 82°C (cooler than the
    83°C step it just completed). CPU/SoC reaches 92–97°C while GPU reads 78–83°C. The GPU sensor
    never looks abnormal.
  - **Clock cap fixes it.** `nvidia-smi -lgc 300,2200` limits GPU to 2200 MHz; the unit completes
    all steps plus one beyond the crash point. At the capped clock, SwThermalSlowdown (0x20) does
    get asserted and the unit survives by throttling from 92W down to ~82W. Same peak power,
    opposite outcome — the free-clock ramp appears too fast for protection to react.
  - **No orderly shutdown.** `journalctl` shows zero shutdown markers — the log simply stops
    mid-operation. No kernel panic, no pstore (`/sys/fs/pstore` empty), no rasdaemon errors, no
    GPU ECC errors. This is the same "zero forensic trace" signature as the TP=2 host freeze
    (S-forum-host-freeze-tp2) and the UVM livelock shutdown (S-forum-uvm-livelock).
  - **Firmware update did not fix it** — SOCFW 2.152.15→2.155.11, EC 3.3.2→3.5.8, USB-C PD
    applied 0x00000516. After the update, the unit dies *sooner* (step 8192 at 88.82W vs step
    16384 at 91.81W before).
  - **DCGM cannot stress GB10.** `dcgmi diag -r 3` reports Skip for targeted_power,
    targeted_stress, memory_bandwidth, memory, pcie, and diagnostic on GB10 — only the
    software/deployment group runs. `nvidia-smi` reports power.limit, power.max_limit, and all
    temperature thresholds as N/A.
  - **Memory is not the constraint** — died with 36.7 GB used (of 124.6 GB) in one run and 99.5 GB
    in another. Pure matmul workload, no KV cache or UVM livelock involved.
  - NVIDIA staff (aniculescu) confirmed this is a **known issue** and recommends lowering GPU
    clock max + running DGX Spark Fieldiag. This corroborates the existing power-controller wedge
    and thermal shutdown findings — the clock-cap workaround is now corroborated by 3+ independent
    sources (S-forum-comfyui-crash, S-forum-gpu-throttle-cmd, this thread). Status: `open`.
  - **[reported]** **`nvidia-smi -lgc <min>,<max>` clock cap is the standard GB10 power/thermal
    mitigation** — now corroborated by 4 independent forum threads (S-forum-comfyui-crash
    2100 MHz, S-forum-gpu-throttle-cmd 2000 MHz, S-forum-power-90w 2200 MHz, S-forum-temps-normal
    referencing wildpines.ai blog). The pattern: cap GPU clocks to 2000–2200 MHz to keep power
    under ~60–70W, avoiding overcurrent trips and thermal shutdowns with minimal performance loss
    (full-speed GPU self-throttles to ~2150 MHz under load anyway). On GB10, `nvidia-smi -pl`
    (power limit) reports N/A, so **clock capping is the only power-control mechanism available**.

- **[conjecture]** **GPU clock cap commands reference** (S-forum-gpu-throttle-cmd, elsaco,
  azampatti): `sudo nvidia-smi -lgc 0,2000` limits GPU to 2000 MHz. At full speed under
  context-prefill or heavier models, power draws ~80W; at 2000 MHz it's ~60W or less. The GPU
  self-throttles to ~2150 MHz under sustained load with basically unaffected speed. Reboots/
  firmware updates reset the clock lock, requiring re-application. This thread is a reference
  for the clock-cap mitigation corroborated across multiple findings.

### Batch 49 forum ingest (2026-08-03)

- **[reported]** **Clock-cap 2000 MHz: near-zero LLM decode loss, 55% power reduction, 8-22°C
  temp drop — quantified across 5+ independent users** (S-forum-cooler-temps, 38-post thread,
  2618 views). This is the largest quantitative A/B dataset for the clock-cap mitigation and
  strongly corroborates the existing **[reported]** finding above. The thread demonstrates
  *why* clock capping works so well on GB10: LLM single-stream decode is bandwidth-bound
  (proven), so SM clock has almost no leverage on throughput. Key measurements:
  - **[reported]** **LLM decode ≈0% loss at 2000 MHz across multiple models** —
    azampatti: Qwen3.6-35B-A3B-NVFP4 ~113 tok/s at both stock 2400-2470 MHz (43 W) and 2000 MHz
    (25 W); Qwen3.5-122B-A10B-hybrid same performance at 2000 MHz, better when heat-soaked.
    whpthomas: 12h quantization at 1982 MHz / 43 W → 68°C GPU / 78°C CPU, 799 s/it; at 2456 MHz
    / 74 W → 82°C / 93°C, 804 s/it — **0.6% performance loss** for 42% power reduction.
    KojiChou: 3× Asus Ascent GX10 A/B — MiniMax-M2.7-NVFP4 TP=2 decode 24.8→~24.5 tok/s
    (within run-to-run noise of 23.7-25.1 tok/s), power 42→19 W (-55%), temp 72→66°C.
  - **[conjecture]** **Diffusion is compute-bound — ~12.5% speed loss at 2000 MHz** (ijontichy):
    Z-Image-Turbo image gen 7.25 s/image (stock, 84°C peak) → 8.17 s/image (2000 MHz, 62°C
    peak). KojiChou: 30-step image gen 305→313 s (+2.6%), image-to-video 1286→1382 s (+7.5%),
    temp 87→74°C, power 68.5→39 W (-43%). Diffusion has large latent token sets processed
    per forward pass → compute-bound, not bandwidth-bound → clock reduction costs real speed.
  - **[conjecture]** **cuBLAS SGEMM sweep — -23% clock = -9% throughput, bandwidth-bound
    explanation** (g6.67300): swept `nvidia-smi -lgc` from 1800-2320 MHz with sustained
    4096×4096×4096 SGEMM (TF32), GPU pre-cooled to ≤45°C so zero throttling at any point:

    | locked clock (target) | measured avg | avg TFLOP/s | vs. natural boost |
    |---|---|---|---|
    | 1800 MHz | 1794 MHz | 36.41 | -9.0% throughput, -22.7% clock |
    | 2000 MHz | 1995 MHz | 38.21 | -4.5% throughput, -14.0% clock |
    | 2100 MHz | 2087 MHz | 39.50 | -1.3% throughput, -10.1% clock |
    | 2200 MHz | 2179 MHz | 39.66 | -0.9% throughput, -6.1% clock |
    | 2260 MHz | 2237.5 MHz | 40.24 | +0.6% throughput, -3.6% clock |
    | natural boost | 2320.6 MHz | 40.01 | baseline |

    Clocking down 23% (2321→1794 MHz) costs only 9% throughput — less than half the linear
    expectation. Root cause: GB10's LPDDR5X bandwidth (273 GB/s) is low for a GPU, and its
    24 MB L2 is well under the ~192 MB working set of 3× 4096² FP32 matrices, so a chunk of
    every SGEMM is memory-bandwidth-bound and doesn't shrink with clock. Confirmed: at
    1024×1024 (fits in L2), throughput tracks clock almost linearly. The "clock doesn't
    matter" effect is specific to workloads whose working set exceeds L2 — most non-trivial
    GEMM/attention shapes at this size. Open-source benchmark:
    `nvcc -O3 simple_gpu_bench.cu -o simple_gpu_bench -lcublas -lnvidia-ml`.
  - **[conjecture]** **Prefill ~10% penalty at 2000 MHz** (paxren2020): DeepSeek-V4-Flash
    pp1000 @ d100000 — ~10% prefill slowdown at locked clock, peak temp 85→70°C, power
    85→48 W. Decode (tg128) unaffected. Consistent with prefill being more compute-bound
    than decode.
  - **[conjecture]** **Systemd unit for persistent clock cap across reboots** (card.ps):
    `nvidia-smi -lgc` does not survive reboot; `-pm 1` (persistence mode) also resets.
    Create `/etc/systemd/system/nvidia-power-limit.service`:
    ```ini
    [Unit]
    Description=Set NVIDIA GPU Clock Limit
    After=nvidia-persistenced.service
    Wants=nvidia-persistenced.service
    [Service]
    Type=oneshot
    ExecStart=/usr/bin/bash -c 'sleep 5 && nvidia-smi -pm 1 && nvidia-smi -lgc 0,2000'
    RemainAfterExit=yes
    [Install]
    WantedBy=multi-user.target
    ```
    Then `sudo systemctl daemon-reload && sudo systemctl enable nvidia-power-limit.service`.
  - **[conjecture]** **2000 MHz chosen because it never triggers thermal throttling**
    (azampatti): at 2200 MHz the GPU still occasionally throttles down to ~1900 MHz; at
    2000 MHz it never throttles. 2000 MHz is the sweet spot for maximal power/thermal
    reduction with minimal performance impact on bandwidth-bound (LLM decode) workloads.
    `-pl` (power limit) is N/A on GB10, so clock capping is the only available mechanism.
  - **Summary:** the clock-cap mitigation is now corroborated by 5+ independent forum
    threads with quantitative data → strengthens the existing **[reported]** finding. The
    mechanism is now well-explained: GB10's bandwidth-bound decode is insensitive to SM
    clock, so capping to 2000 MHz trades ~0-5% LLM performance for 43-69% power reduction
    and 8-22°C lower temperatures. Compute-bound workloads (diffusion, prefill) pay
    ~10-12% but still benefit thermally.

### Batch 38 forum ingest (2026-07-27)

- **[conjecture]** **New ASUS GX10 SoC + TPM firmware update — stable, no significant performance
  change** (S-forum-asus-fw-jul25, robert287): ASUS GX10 received new SoC and TPM firmware updates
  in late July 2026. Both applied on first try; nodes stable at 96% GPU load pulling 70W+. Reboot
  was notably slow ("FOREVER," tens of minutes per one user). Benchmark delta: 2-4% across model
  benchmarks — within noise / attributable to fresh clean state, not a real performance boost. The
  update references "performance enhancements" but none were measurable. A concurrent minor NVIDIA
  driver update may contribute to stability. This is an ASUS GX10 data point (vs the FE Spark
  firmware updates tracked in S-forum-fw-july2026 / S-forum-asus-fw0103) — corroborates that both
  OEMs are actively shipping firmware in the July 2026 window. No version numbers captured. Status:
  `open` — no regression observed, no measurable improvement.

### Batch 42 forum ingest (2026-07-30)

- **[conjecture]** **apt upgrade to driver 580.173.02 breaks GPU on OTA2607 —
  "torn" driver/firmware pairing** (S-forum-driver580-173, chenette): on 2 × DGX Spark
  (GB10) running DGX OS OTA2607 (DGX_SWBUILD_VERSION=7.2.3, DGX_OTA_VERSION=7.3.1,
  kernel 6.17.0-1026-nvidia), a routine `apt upgrade` pulled
  `nvidia-driver-580-open` from **580.159.03** → **580.173.02** (Ubuntu
  noble-updates/restricted + noble-security/restricted). After reboot, GPU fails to
  initialize with the same Xid 119 / SEC2 secure-boot timeout fingerprint as
  S-forum-gsp-timeout:
  ```
  NVRM: Xid 119, Timeout after 6s waiting for GSP_INIT_DONE (function 4097)
  NVRM: ksec2PrepareBootCommands_GB20B: SEC2 secure boot partition timed out.
  NVRM: RmInitAdapter failed! (0x62:0x65:2028)
  $ nvidia-smi → "No devices were found"
  ```
  `nvidia-spark-ota-check` reports the OTA as **"torn"** (152/153 checks pass; the
  **only** failing component is the driver — OTA2607 expects 580.159.03, but
  580.173.02 was installed). SOCFW, EC, and all system packages are correctly at
  OTA2607. Both units failed identically. Root cause: Ubuntu's restricted pocket
  serves a driver newer than — and not paired with — the GPU secure-boot (GSP/SEC2)
  firmware shipped in OTA2607. The `nv-update-disable` mechanism did **not** prevent
  the upgrade. **Workaround:** downgrade driver to 580.159.03 (exact archived debs
  from Launchpad librarian) + `apt-mark hold`. **Resolution:** re-running the DGX
  Dashboard update on 2026-07-28 fixed it — the dashboard applied the remaining
  OTA components that pair with 580.173.02. **Key GB10 insight:** a plain `apt
  upgrade` on a Spark can install an unpaired driver that the on-box firmware
  rejects, bricking GPU init — always use the DGX Dashboard or pin driver packages.
  - **[conjecture]** **580.173.02 works on some Sparks** (amurnane123, same thread):
    4 DGX Sparks running driver 580.173.02 with current firmware — all GPUs work
    normally (nvidia-smi shows GB10, 46°C, 11W idle, vLLM worker running). This
    means the failure is **firmware-version-dependent**, not a universal
    incompatibility — 580.173.02 is fine *if* the platform firmware is at the
    matching version. The regression only bites when the driver outpaces the
    firmware (OTA2607 firmware + 580.173.02 driver = torn pairing).
  - **[conjecture]** **Reinstall + Secure Boot disable as alternative fix**
    (padrian, same thread): on 4 × Sparks (2 FE + 2 Gigabyte), 2 Gigabyte units
    hit the same issue. Fix: `sudo apt install --reinstall nvidia-driver-580-open
    nvidia-utils-580 nvidia-compute-utils-580 nvidia-settings` + disable Secure
    Boot. The FE units were unaffected. This suggests Secure Boot's driver signing
    verification may also play a role.
  - Status: `fixed` — re-running DGX Dashboard update resolves the torn pairing.
    Related to the existing OTA loop findings (S-forum-ota-loop,
    S-forum-update-loop) and the GSP_INIT_DONE timeout (S-forum-gsp-timeout) —
    same Xid 119 class, but here the root cause is a driver/firmware version
    mismatch rather than a firmware update alone.

- **[conjecture]** **USB3→USB2 fallback at boot corroborated on Asus GX10 —
  external drive connected at boot sticks at USB2** (S-forum-model-storage,
  gaborm): every DGX OEM and the FE have an issue initializing the high-speed
  USB layer if an external drive is already connected at boot/start — the
  connection gets stuck at USB2 speed. **Fix:** unmount, disconnect, reconnect,
  mount. This corroborates the existing **[reported]** USB3 SuperSpeed PHY
  fallback finding (S-forum-usb2-fallback, 7 independent users) with an 8th
  user and extends it to the Asus GX10 OEM SKU — now confirmed across FE,
  Gigabyte, MSI, and ASUS variants. The symptom is the same: USB3 device
  enumerates only at USB 2.0 (480 Mbps) speed when connected at boot.
  - **[conjecture]** **USB SSD speed drops to 20 MB/s intermittently** (starkrun,
    same thread): a USB SSD connected to a Spark normally transfers at ~775 MB/s
    but randomly drops to 20 MB/s and stays stuck — no errors in `dmesg`, reboot
    does not fix, it "just started working fine the next day." Cause unknown
    (thermal throttling of the enclosure controller hypothesized by x1917x).
    Single source → [conjecture]. Relevant to anyone using USB SSD for model
    storage on Spark.

- **[conjecture]** **Acer Veriton GN100 thermal A/B test — both units ~68°C
  under sustained load, no throttling** (S-forum-acer-thermal, jjustice): two
  Acer Veriton GN100 (DGX Spark OEM) units running Qwen3.5-122B-A10B INT4
  AutoRound + DFlash via vLLM (`aeon-vllm-ultimate`), 1 hour continuous
  `llama-benchy` load (pp2048/tg512, concurrency 3, 300 runs). Results: both
  units settled at **68-70°C under load** (one brief 82°C spike on unit A,
  recovered), 96% GPU util, ~25 tok/s per request, **zero thermal throttling**,
  zero errors. CPU usage stayed low (6.3% / 5.3% avg). The idle temperature gap
  (42°C vs 43°C) did not persist under load. Both landed within the range
  reported by StorageReview's OEM cooling comparison: **Acer peaks ~68°C** vs
  **80-82°C for other OEM builds** where thermal throttling begins. After the
  test, both units idled at ~40°C. Config: mini-rack with space, no extra fans.
  This is the first published Acer Veriton GN100 thermal data point →
  [conjecture]. Corroborates the existing thermal findings (S-forum-temps-normal
  zones 0/5 at 94.6°C, S-forum-thermal-shutdown) — the Acer chassis appears to
  run cooler than FE/Gigabyte/MSI under the same workload.
  - **[conjecture]** **spark_hwmon driver for full system power telemetry**
    (azampatti, same thread): `antheas/spark_hwmon` — a Linux hwmon driver for
    the DGX Spark (GB10 SoC) that exposes full system power telemetry via
    standard `sensors` / sysfs interfaces. Referenced as a tool for more
    detailed thermal monitoring. Single source → [conjecture].

### Batch 43 forum ingest (2026-07-30)

- **[reported]** **SM121 software support status — NVIDIA official response + community
  fact-check** (S-forum-sm121-support, 43-post thread): NVIDIA staff (johnny_nv) posted an
  official roadmap response; community (baristankut) fact-checked it line-by-line. Key
  durable findings:
  - **[conjecture]** **vLLM `--enforce-eager` required in certain versions for correctness —
    20-30% performance loss** (baristankut, confirmed by johnny_nv). vLLM 0.14.0 (expected
    shortly) improves Blackwell compatibility and reduces reliance on eager execution.
    Corroborates existing `[[wiki/cudagraphs-and-compile.md]]` finding that MoE cudagraph
    capture fails on sm_121.
  - **[conjecture]** **CuTE DSL FP4 restricted to sm_100a only** — CUTLASS Issue #2800 open.
    C++ API works on sm_121, but Python DSL still restricts FP4 to sm_100a. CUTLASS compatible
    with DGX Spark from v4.2.0; latest v4.3.5; v4.4.x adds better CuTE DSL (johnny_nv).
  - **[conjecture]** **PyTorch 2.10** (scheduled Jan 21, 2026) includes FBGEMM and CUTLASS
    matmul integrations for sm12x. CUDA kernels compiled at major arch family level (sm12x),
    not per-SKU. Only Tensor Core–specific kernels need conditional compilation (already
    handled). (johnny_nv)
  - **[conjecture]** **Triton 3.6.0** (RC, tied to PyTorch release pipeline) contains SM121
    fixes. Available via PyTorch test/nightly index. Latest stable: Triton 3.5.1. (johnny_nv,
    baristankut)
  - **[conjecture]** **FlashInfer 0.5.3+** supports sm12x with distributed wheels for DGX Spark.
    Latest: 0.6.1. (johnny_nv)
  - **[conjecture]** **SGLang runs via unofficial community branch** (lmsysorg/sglang:spark),
    not mainline. GitHub Issue #11658 open with temporary workarounds (Triton PTXAS errors,
    FP8 CUTLASS dispatch failures). Official wheels available as alternative. (baristankut,
    johnny_nv)
  - **[conjecture]** **MoE kernels: no optimized configs for NVIDIA_GB10** — runtime warning
    confirms this. Active development. (baristankut)
  - **[conjecture]** **CUDA 12.0f vs 12.1a distinction**: 12.0f is the correct baseline for
    GeForce Blackwell general support; 12.1a only needed for chip-family-specific features.
    Using a specific build variant to unlock optional features is an architectural capability
    distinction, not a workaround. (johnny_nv)
  - **[conjecture]** **tcgen05, DSMEM, TMEM, TMA/multicast support lacking on sm_121** —
    referenced from dgx-spark-playbooks (closed Dec 19, 2025). Hardware appears present but
    not supported by sm_121 PTXAS. Corroborates existing **[reported]** TMA finding (S-forum-tma).
    (alexander.korolev.germany, baristankut)
  - **[conjecture]** **No locked/hidden memory on DGX Spark** (S-forum-170hx-spark,
    FlossingEnthusiast): unlike CMP 170HX crypto cards (which have unlockable hidden HBM),
    the Spark's 121 GB unified memory is the full available pool — no extra "locked" memory.
    This refutes speculation about hidden VRAM on Spark.

### Batch 44 forum ingest (2026-07-31)

- **[conjecture]** **Xid 31 MMU faults during AMP-enabled training on GB10** (S-forum-xid31-yolo,
  dall9): Repeated Xid 31 (MMU fault) kernel events on DGX Spark during AMP-enabled YOLOv8s
  training. All 5 AMP-enabled runs (batch 16/960, batch 12/960, batch 8/640) produced Xid 31
  with `ENGINE GRAPHICS GPC2` and `FAULT_PDE ACCESS_TYPE_VIRT_READ`, from different Python
  PIDs. With `CUDA_LAUNCH_BLOCKING=1`, the synchronous error was
  `RuntimeError: cuDNN error: CUDNN_STATUS_EXECUTION_FAILED at torch.nn.functional.conv2d`.
  A standalone FP16 `torch.matmul` loop ran 3,000 s / 208,000 iterations without any Xid;
  one AMP-disabled training run went 3.5 h / 7 epochs without Xid (limited evidence — raw
  stdout not retained). Environment: DGX OS 7.5.0, Ubuntu 24.04.4, kernel 6.17.0-1026-nvidia,
  host driver 580.159.03, NGC PyTorch 26.06-py3 (PyTorch 2.13.0a0, Ultralytics 8.4.110),
  CUDA forward-compat (CUDA 13.3 user driver 610.43.02 over kernel 580.159.03). Telemetry
  ≤71 °C / 44 W over the captured window (does not exclude short transients). The GPC2
  consistency across all events is noted as a reason not to rule out hardware/firmware.
  NGC PyTorch 24.08 comparison was invalid (reported GB10 as unsupported, then hit a
  NumPy/OpenCV ABI error before GPU work). **Root cause unresolved** — alternatives include
  application-level illegal access, cuDNN/driver/GSP/address-translation interaction, or
  localized hardware/firmware. This is a **non-LLM training workload** but documents a
  GB10-specific GPU fault signature (Xid 31 / GPC2 / FAULT_PDE) under AMP conv paths that
  may be relevant to any AMP-enabled compute on Spark.

### Batch 45 forum ingest (2026-07-31)

- **[conjecture]** **Unified-memory kernel-init allocations consume ~50 GB before weight loading
  starts on GB10** (S-forum-um-kernel-init, rp_37716): on `nvcr.io/nvidia/vllm:26.07-py3` (also
  reproduced on `vllm/vllm-openai:v0.26.0-aarch64`), vLLM kernel/backend initialization
  (FlashInfer FP8 scaled MM, DeepGEMM PDL, FlashInfer-CUTLASS NVFP4, TRITON_ATTN) consumes
  **~50 GB of unified memory** *before* `weight_utils.py` begins reading the checkpoint. The
  drop lands precisely at kernel/backend selection time (T+25s: 120 GB → 68 GB → 45 GB over
  ~15s). This is NOT a measurement bug — `psutil.virtual_memory().available` accurately reads
  real system memory. **Downstream effect:** for any checkpoint exceeding ~90% of the
  post-init available RAM, vLLM's auto-prefetch optimization disables itself ("Auto-prefetch
  is disabled… checkpoint size exceeds 90% of available RAM"), forcing the slow non-prefetch
  disk-read path — several extra minutes on a 75 GB model. This is a GB10/Grace-Blackwell
  UMA-specific finding: on discrete GPUs, kernel init allocations come from a separate VRAM
  pool and don't shrink the system RAM available for weight prefetch. The auto-prefetch
  check runs before kernel init settles, measuring available RAM as if that allocation doesn't
  happen. Single source → [conjecture], but the mechanism is consistent with the proven UMA
  constraints documented on this page.
- **[conjecture]** **CX-7 PCIe "insufficient power on the PCIe slot (27W)" on all 4 ports**
  (S-forum-cx7-pcie-power, ammarabbaxi13): `dmesg` shows `mlx5_pcie_event: Detected
  insufficient power on the PCIe slot (27W)` on all 4 CX-7 interfaces during 2-node
  deployment. `iperf3` shows 19.3 Gbits/sec with 6405 retries (unreliable), but
  `ib_write_bw` reports 111.60 Gb/sec (healthy — consistent with proven fabric measurements).
  The 27W PCIe slot power warning is the same class as the documented `SlotPowerLimit 0W`
  bug (S-forum-cx7-13gbps) but at 27W instead of 0W. Unplugging both machines for 1 min
  did NOT fix it in this case. Model loading (Qwen3.5-122B-FP8 TP=2 Ray) fails after weight
  load with `gloo Connection closed by peer` — the worker node crashes. Fix was found via
  the NCCL all-reduce deadlock thread. Concurrent requests stall 20-30s before generation
  starts. Single source → [conjecture].

### Batch 50 forum ingest (2026-08-03)

- **[conjecture]** **partnerdiag PowerStress reproducibly hard-powers-off the box — thermal sensor
  swap anomaly persists across firmware updates** (S-forum-powerstress, digiegg): A DGX Spark
  running always-on LLM inference experienced repeated hard power-offs during scheduled GPU
  workload windows (4 times in 5 days, zero forensic trace — no vmcore, empty pstore, no Xid,
  journal stops mid-line). NVIDIA's `partnerdiag` field diagnostic reproduces the failure
  **every time** on the PowerStress test: GpuStress, C2CStress, CpuStress1, CpuStress2 all PASS;
  PowerStress never returns — the machine powers off mid-test (~3m20s in).
  - **External 1 Hz thermal sampler caught the event**: zone0/zone5 hold a flat 88.7°C plateau for
    ~6 min, then jump +9.8°C in 4 seconds (88.0→97.8°C) while zone4 simultaneously collapses
    85.6→~70°C. The power loss follows 2 s later. A 9°C rise in 2 s is not thermal mass — reads
    as a sensor handoff/miscalibration or a real unmanaged hotspot nothing reacts to.
  - **Zone2/zone4 sensor value swap anomaly**: in both pre- and post-firmware runs, zone2 and
    zone4 exchange values over ~3 s (zone2 drops 81.4→65.6°C while zone4 rises 68.0→81.4°C)
    while zone0/zone5 sit flat at ~97.6°C. This anomaly **survived both EC and SoC firmware
    capsule updates**, suggesting a sensor-side issue rather than a control-loop bug.
  - **Firmware update stops the hard power-off but not the thermal fault**: after updating
    EC 0x03000302→0x03000508 and SoC FW 0x0200980f→0x02009b0b (via fwupdmgr capsule-on-disk),
    PowerStress now completes: `FAILED [8:11s]`, error code **082-000-1-020000600139**
    ("Acceptable temperature limits exceeded or the thermal sensor is broken or miscalibrated").
    The box stays up instead of dying, but reaches the same ~97.8°C peak. This is a **new
    variant** of the thermal-shutdown class — the firmware update converts a hard power-off
    into a graceful FAIL with error code, but does not fix the underlying thermal/sensor fault.
  - **RMA approved** by NVIDIA (Neill) after collecting field diagnostic logs. First documented
    case of partnerdiag PowerStress yielding a clean MODS error code on a unit that previously
    hard-powered-off.
  - **fieldiag install requires Secure Boot disabled** (mashie): Secure Boot prevents loading
    drivers and possibly the dgx-spark-fieldiag package installation itself.
  - **Corroborates**: the hard-power-off-with-zero-forensic-trace pattern
    (S-forum-thermal-shutdown, S-forum-host-freeze-tp2, S-forum-power-90w, S-forum-uvm-livelock),
    the fieldiag ofed-scripts dependency gap (S-forum-ec-fan-asus — same `ofed-scripts` missing
    dependency hit by DannyTup in this thread), and the 97-98°C ACPI zone thermal threshold
    (S-forum-ec-fan-rollback, S-forum-ec-fan-asus, S-forum-temps-normal).
  - **New durable finding**: the zone2/zone4 sensor value swap is a **sensor mapping/calibration
    problem**, not a thermal-mass phenomenon — it persists across firmware updates and is
    consistent across runs. First published thermal sensor anomaly fingerprint on GB10. Tagged
    [conjecture] (single source, single unit — may be unit-specific hardware fault). RMA approved,
    so the unit may be replaced rather than the issue fixed in firmware.

- **[conjecture]** **DGX Dashboard stale firmware metadata — shows nvidia-firmware-580-580.159.03
  as available update when 580.173.02 is already installed** (S-forum-dashboard-fw-stale,
  kafej666, sggin1, elsaco): The DGX Dashboard Updates page presents `nvidia-firmware-580-
  580.159.03` as a pending update even on systems already running driver 580.173.02. The
  `nvidia-firmware-580` apt package (from `noble-updates/restricted arm64`) contains the
  kernel-module firmware blobs `nvidia/580.173.02/gsp_tu10x.bin` and `gsp_ga10x.bin` — the
  installed driver loads the correct 580.173.02 firmware, but the dashboard's OTA metadata is
  stale (references the older .159 version). The GPU is healthy (`nvidia-smi` reports 580.173.02,
  no issues). This is the same class of OTA metadata staleness as S-forum-ota-loop and
  S-forum-fwupd-mismatch. Workaround: ignore the stale Update button while the GPU is functioning
  correctly; verify with `uname -r; nvidia-smi --query-gpu=driver_version --format=csv,noheader`.
  3 users in the same thread confirm the pattern → [conjecture] (root cause unstated).

### Batch 52 forum ingest (2026-08-04)

- **[conjecture]** **Fan control is entirely firmware — no PWM, no BMC, no OS override**
  (S-forum-fan-firmware, nvidia3815, Mach_AI, eugr): the 62-post thread confirms what later
  sources (S-forum-ec-fan-rollback, S-forum-ec-fan-asus) established via firmware regression —
  the EC isolates fan control from the OS. `fancontrol`/`pwmconfig`/`nvidia-settings` cannot
  override the fan curve. Fans ramp only at high-80s/90s °C ACPI zone threshold. Under normal
  load, GPU reaches ~84°C, CPU late-80s. Multiple users report the chassis is hot to touch
  but functions correctly. This early thread (Oct 2025) is the first forum documentation of
  the firmware-only fan control constraint — corroborates the later `[reported]` finding
  that EC firmware changes can break the fan curve (S-forum-ec-fan-rollback → S-forum-ec-fan-asus).
  All [conjecture] in this thread (no controlled measurements).
- **[conjecture]** **Swap exhaustion → total system lockup (early corroboration)**
  (S-forum-fan-firmware, RazielAU, eugr): when vLLM + llama.cpp co-loaded models exhaust
  unified memory, the system pages to swap → crawls to near-complete stop → SSH commands
  take minutes to process → X11 session dead. `killall -KILL` eventually works but takes
  ~5 minutes. Disabling swap (`swapoff -a`) forces OOM-kill instead (process crashes or
  system resets, but doesn't lock up). This is the earliest forum report of the swap-lockup
  mechanism, corroborating the later `[conjecture]` bus-saturation diagnosis
  (S-forum-comfyui-crash) and the `[proven]` "unified-memory OOM = hard reboot" finding.
- **[conjecture]** **earlyoom -s 80 too aggressive for vLLM startup on Spark** (S-forum-earlyoom-config,
  helge): sparkrun activates earlyoom with `EARLYOOM_ARGS=-s 80` (trigger when 80% of RAM
  used), but vLLM model loading has a transient memory peak that temporarily exceeds 80%
  before settling — even `--gpu-memory-utilization 0.75` doesn't help because the peak is
  during weight-loading initialization, not steady-state. Fix: `sudo sed -i
  '/^EARLYOOM_ARGS=/ s/-s 80/-s 20/' /etc/default/earlyoom && sudo systemctl restart earlyoom`
  — trigger only when <20% of swap is left. Note: sparkrun overwrites this on cluster
  config, so the fix must be re-applied after sparkrun cluster setup. This is a practical
  operational finding for anyone using sparkrun's earlyoom safety stack on Spark — the
  default threshold is calibrated for generic servers, not for the 128 GB UMA memory
  pattern where transient spikes during model loading are normal. Single source → [conjecture].

### Batch 53 forum ingest (2026-08-05)

- **[conjecture]** **System apt upgrade can trigger the power-controller wedge** (S-forum-jul31-wedge,
  unicornxoxo2): a July 31 `apt upgrade` (standard Ubuntu packages — tar, gawk, krb5,
  gstreamer, remmina, libgphoto2, etc.; no NVIDIA packages in the log) caused Qwen3.6-35B-A3B
  NVFP4 decode on vLLM v0.25.0 to drop from **107 tok/s → 45 tok/s** (TPOT 8.55 ms → 21.16 ms,
  ITL 28.98 ms → 79.15 ms) — a ~2.4× regression with no thermal issue (unit "slightly warm").
  The wedge was cleared by a full AC power-cycle (unplug from wall socket, wait a few
  minutes, reconnect), restoring **84 tok/s** (TPOT 11.07 ms, ITL 27.64 ms). Notably the
  post-fix MTP acceptance dropped from 79.81% to 50.02% — the user attributes this to a
  different model state after the power-cycle, not the wedge itself. This corroborates the
  existing [reported] power-controller wedge pattern (pinned low clock, no throttle flag,
  AC power-cycle fix) and adds a new trigger: a routine OS-level `apt upgrade` can
  precipitate it, not just NVIDIA firmware/driver updates. Single source → [conjecture].
  Note: the July 23 update also caused idle overheating ("roasting like hell in stale")
  on the same unit — see S-forum-typec-thermal for the USB-C PD firmware pending-update
  pattern.

### Batch 57 forum ingest (2026-08-07)

- **[conjecture]** **GB10 may effectively serialize CUDA contexts — `cuInit()` returns
  `CUDA_ERROR_NO_DEVICE` (rc=100) for a second process while another holds a context**
  (S-forum-cuda-single-ctx, tom450): on DGX Spark (GB10, sm_121, 128 GB unified, driver
  580.159.03, CUDA 13, Ubuntu 24.04 aarch64), a second CUDA process calling `cuInit(0)`
  receives `CUDA_ERROR_NO_DEVICE` (rc=100) as long as another process holds a live CUDA
  context — **even though `nvidia-smi -q` reports Compute Mode: Default** (which normally
  allows concurrent contexts). After the context-holding process exits, `cuInit()` still
  returns 100 for **20–60+ seconds** before succeeding. Reproduced host↔container (Docker
  `--gpus all`) and host↔host. `nvidia-smi` works fine throughout (device listed, usual GB10
  `[N/A]` memory metrics). vLLM's sleep mode (`/sleep?level=1`) frees ~36 GB but the sleeping
  process keeps its context → second process still gets `NO_DEVICE`; only a full process exit
  releases the device. No MIG, no MPS, `nvidia-uvm` loaded. The reporter links this to the Xid
  119 (GSP RPC timeout) pattern (S-forum-gsp-timeout, S-forum-driver580-173): if a new process
  starts its allocation burst during the previous context's teardown window, it can
  reproducibly push the GPU into Xid 119 (`_memdescAllocInternal` `NV_ERR_NO_MEMORY` and
  `GSP_RM_ALLOC` timeouts in dmesg). Serializing GPU users and gating on `cuInit()` success
  before starting the next process eliminated the crashes. **Status:** `open` — unknown
  whether this is an intended GB10 limitation or a driver bug. This is a GB10-specific
  concurrency constraint: if confirmed, it means the "single-tenant per node" rule
  (documented [proven] above) is enforced at the **driver/context level**, not just by
  memory pressure. It also means vLLM sleep mode does NOT enable a second CUDA process to
  coexist — only full process exit frees the device. Single source → [conjecture].
  Related to the existing `[conjecture]` `cudaMemGetInfo` under-reporting finding
  (S-forum-comfyui-optimized) — both are UMA multi-process constraints unique to GB10.

- **[conjecture]** **ConnectX-7 27W "insufficient power" boot warning on all 4 ports is
  benign — confirmed by NVIDIA staff** (S-forum-cx7-27w-benign, james587, aniculescu):
  on a new replacement DGX Spark (OTA 7.5.0, driver 580.173.02, kernel 6.17.0-1029-nvidia),
  all four ConnectX-7 PCIe functions report `mlx5_core: insufficient power … 27W` at boot.
  NVIDIA staff (aniculescu) confirmed: **"The 27 W Power messages are benign and do not
  indicate an actual fault with your system."** This corroborates the existing
  `[conjecture]` CX-7 PCIe power warning finding (S-forum-cx7-pcie-power, which saw the
  same 27W warning on all 4 ports) — now confirmed as expected platform behavior, not a
  fault. Firmware inventory on the replacement unit: EC 0x03000508, UEFI 0x02009b0b, CX7
  firmware 28.45.4028, Samsung NVMe NXHB202Q, platform bundle 5.36_0ACUM018. Three signed
  capsule updates had been applied during factory provisioning. `fwupdmgr` reports no
  additional updates available. Docker 29.2.1 + containerd 2.2.1 installed and validated
  on ARM64. Single source → [conjecture] (NVIDIA staff confirmation, but only one thread).

- **[conjecture]** **DGX Spark hard-freeze under sustained MiniMax-H3 inference — PowerStress
  thermal failure, unit-to-unit thermal variation** (S-forum-thermal-freeze,
  tannerhaggerman, zc142365, sggin1): a DGX Spark (DGX OS 7.5.0, kernel 6.17.0-1029-nvidia,
  driver 580.173.02, EC 3.5.8, current firmware) reproducibly hard-freezes under sustained
  GPU inference (MiniMax-H3 864×480, 5-second inference). **No OOM, no Xid, no kernel
  panic, no application exception** — the same "zero forensic trace" signature documented
  across S-forum-thermal-shutdown, S-forum-host-freeze-tp2, S-forum-power-90w,
  S-forum-uvm-livelock. Thermal data: idle GPU 47°C, ACPI 49.8°C at ~4W; under load
  (96% GPU util, ~70-83W): GPU reached 84°C, hottest ACPI/SoC zone reached **93.1°C**
  within ~2 minutes, then hard-freeze. `partnerdiag` PowerStress fails with
  **MODS-020000610139** ("acceptable temperature limits exceeded or thermal sensor
  broken/miscalibrated") — same error class as S-forum-powerstress (082-000-1-020000600139).
  GpuStress passes. 240W adapter verified, display + airflow verified. A second user
  (zc142365) resolved similar symptoms by **downgrading firmware and locking max clock to
  2000 MHz** — corroborating the existing **[reported]** clock-cap mitigation. A third user
  (sggin1) running the same MiniMax-H3 workload on their Spark reports much cooler
  temperatures: GPU 58°C, ACPI 62°C at only **15.17W** under load (101s render) — a
  dramatic unit-to-unit thermal variation (84°C/83W vs 58°C/15W for the same model).
  This large variation is consistent with the existing thermal-paste-degradation and
  sensor-blind-spot findings (S-forum-thermal-shutdown). The OP's unit appears to have
  a thermal fault (PowerStress failure + MODS error code → RMA candidate). Single thread
  with 3 users → [conjecture] for the freeze mechanism; the clock-cap workaround is
  already [reported] from prior batches. MiniMax-H3 GPU memory: ~24 GB (diffusion) +
  15 GB (TE) + 5 GB (VAE) ≈ 44 GB.

### Batch 55 forum ingest (2026-08-06)

- **[reported]** **CX-7 connection raises idle temperature ~10°C (42→52°C) with no load**
  (S-forum-cx7-idle-temp, elvisnwh + mashie): connecting two Sparks via CX-7 raises idle
  temperature ~10°C (from 42°C to 52°C) even with nothing loaded, fresh from boot, minimal
  traffic. mashie explains: the CX-7 chip is powered off when no cable is connected; when
  active it adds **~17 W of heat per node**. This is now the **3rd independent source**
  corroborating the CX-7 active thermal/power penalty: S-forum-cx7-hotplug (idle power
  nearly doubles when cable connected), S-forum-cx7-dac-power (6°C higher with DAC even
  after software unbind), and now S-forum-cx7-idle-temp (10°C higher, 17 W/node quantified).
  Three independent forum threads agree → **promoted to [reported]**. The ~17 W figure is
  consistent with the ~100 W "rest" budget allocation noted above (CX-7 + SSD + USB).

### Batch 56 forum ingest (2026-08-06)

- **[conjecture]** **Non-DGX OS possible on Spark — ACPI (not DT), NVIDIA-maintained kernels
  needed** (S-forum-nondgx-os, NVES + hiroshiya + elsaco): NVIDIA staff confirms DGX OS is
  the officially supported, optimized, and validated environment; users may install other
  OS but support will be limited and NVIDIA may ask for a reimage to factory for hardware
  triage. Key technical findings for running alternative distros:
  - **ACPI, not Device Tree**: the DGX Spark uses ACPI, so any newer Linux distro will
    work (unlike Jetson-style DT-based systems). This is a structural platform fact.
  - **NVIDIA-maintained kernels needed**: you must compile kernels yourself or rely on
    community builds (e.g. `graham33/nixos-dgx-spark` for NixOS). Distro package maintainers
    are unlikely to provide compatible kernels.
  - **Fedora 44 confirmed working** on GX10 (hiroshiya): kernel `7.0.12-nv-1016.16`,
    driver 595.84, CUDA 13.2, 121.63 GB RAM, btrfs. `nvidia-smi` reports GB10 at 43°C,
    3W idle, CUDA 13.2.
  - **Toolchain caution**: some distros ship newer toolchains that may make the CUDA stack
    unhappy — most distros provide older toolchains for compatibility.
  - **Software limitations**: NVIDIA Workbench and Field Diagnostics suite are built
    specifically for Ubuntu 24.04; they won't run on other distros. elsaco: "if you enjoy
    tinkering, any distro will do; if you look for stability and convenience, DGX OS is best."
  - **Red Hat semi-official support**: a community guide for building a custom RHEL kernel
    for DGX Spark exists (referenced in the thread).
  Single source for most details (NVIDIA staff + 2 community users) → [conjecture]. The
  ACPI-not-DT fact is a durable platform finding. Relevant to users who need a specific
  distro for compliance/infrastructure reasons but want to use Spark hardware.

- **[conjecture]** **vLLM x86_64 Docker images trigger QEMU emulation on Grace CPU → 3.7 tok/s**
  (S-forum-vllm-qemu, rithinsundar87): standard Docker Hub `vllm/vllm-openai` images default
  to **x86_64**, which triggers QEMU emulation on the Grace (ARM64) CPU. This "starves the
  GPU" due to translation overhead — Qwen2.5-Coder-32B-Instruct via vLLM measured only
  **3.7 tok/s** (vs expected 20-40+ on native ARM64). Three issues identified:
  1. **Instruction set mismatch**: x86_64 image → QEMU → massive CPU overhead before GPU
     even sees the workload.
  2. **CUDA 13 library pathing**: on `nvcr.io/nvidia/pytorch:25.01-py3` base, `pip install
     vllm` leads to `ImportError: libcudart.so.13 not found` — the runtime is installed in
     a nested Python directory, not in `/usr/lib` or `/usr/local/cuda/lib64`.
  3. **pip overwrites NVIDIA-optimized PyTorch**: bare `pip install vllm` uninstalls the
     NVIDIA-optimized `+nv` PyTorch wheel, replacing it with a generic build lacking
     Blackwell (SM 10.0) math kernels. The poster incorrectly refers to sm_121 as "SM 10.0."
  Single post, no replies → [conjecture]. **Why it bites on Spark:** this is the most
  common trap for new Spark users — pulling the default vLLM Docker image gives QEMU
  emulation, not native execution. The fix is to use ARM64-native images (e.g.
  `spark-vllm-docker`, NGC ARM64 tags, or community-built ARM64 images). Corroborates the
  existing `[conjecture]` stock vLLM hang finding (S-forum-vllm-stock-hang) — both are
  "wrong image on ARM64" failure modes. The 3.7 tok/s figure is a useful baseline for
  "how slow QEMU emulation is" vs native ARM64.

### Batch 63 forum ingest (2026-08-11)

- **[reported]** **GPU clock energy-efficiency sweep — 1400-1800 MHz is the sweet spot for
  bandwidth-bound LLM decode on GB10, ~25% better Wh/1M tokens than uncapped for ~3% speed
  loss** (S-forum-clock-energy-sweep, peter.h177 + jetspark + arctic.gus + co-le): the most
  quantitative clock-vs-power-vs-energy dataset for GB10 LLM inference to date. A 17-point
  clock sweep (400–2400 MHz, both nodes capped in lockstep via `nvidia-smi -lgc`) on 2× DGX
  Spark TP=2 running DSV4-Flash-0731 FP8 + DSpark + prefix caching (14-16h/day workload):
  - **Decode is bandwidth-bound — flat across the top of the range:** 47.4-51.3 tok/s from
    1400-2400 MHz. Raising the clock just makes the SMs "wait faster." This strongly
    corroborates the existing **[reported]** clock-cap finding (Batch 41/49) with a new
    dimension: energy efficiency (Wh/1M tokens), not just power/thermal.

    | Cap (MHz) | Decode tok/s | Wall power (2 nodes) | Wh / 1M tokens |
    |---|---|---|---|
    | 2400 (uncapped) | 51.34 | 330 W | 1,688 |
    | 2200 | 51.43 | 274 W | 1,480 |
    | 1900 | 50.74 | 252 W | 1,381 |
    | 1700 | 49.80 | 242 W | 1,350 |
    | 1400 | 47.74 | 234 W | 1,362 |
    | 800 | 34.90 | 211 W | 1,679 |

  - **Best energy ROI band: 1400-1800 MHz** — ~25% better Wh/1M tokens than uncapped for
    ~3% decode speed loss. Below ~1000 MHz the curve turns back up: generation itself costs
    ~190 W regardless of clock, so you stretch that bill over more hours. Uncapped is not
    even the fastest setting (2200 MHz at 51.43 tok/s slightly beats 2400 at 51.34).
  - **`nvidia-smi` accounts for only 12-27% of real GB10 power draw** — wall-socket meter
    is the only reliable way to measure total system power. This is a durable GB10 finding:
    `nvidia-smi` power readings are not useful for energy-efficiency calculations on this
    platform (GPU power is a small fraction of total SoC + system draw).
  - **Prefill is compute-bound — ~14% penalty at 1400 MHz vs uncapped.** In agentic loops
    with cold prefills, clock capping costs real prefill throughput. But with prefix caching,
    only the first turn pays full prefill, so decode dominates in practice.
  - **`-lgc` does not survive reboot** and GB10 snaps to discrete clock steps (ask for 1200,
    get 1098). Always read back the actual clock. Re-apply the cap after every reboot or
    use a systemd service. (jetspark posted a systemd unit — `jetspark-gpu-clock-cap.service`
    — corroborating the existing systemd pattern from Batch 49, S-forum-cooler-temps.)
  - **arctic.gus: stock cooling throttles at 2100 MHz** — with cases off + repasted, the
    GPU boosts to 2500+ MHz and prefill throughput climbs further. Stock cooling cannot
    avoid thermal throttling above ~2100 MHz, which is why extra MHz yields no improvement
    past that point on stock units. Consistent with the existing thermal findings
    (S-forum-thermal-shutdown, S-forum-acer-thermal).
  - **co-le corroborates** with DSV4-Flash-0731 + DSpark on 2× Asus Ascent GX10: 2000 MHz
    is the "proper pick" (very close to peak), runs at 2150 MHz for increased prefill +
    decode speed. This is a 3rd independent user confirming the clock-cap sweet spot.
  - Multiple independent users (peter.h177, jetspark, arctic.gus, co-le) contributing
    quantitative data → strengthens the existing **[reported]** clock-cap finding. The
    energy-efficiency dimension (Wh/1M tokens) is new and durable. **No evidence
    promotion** — the [reported] tier is already established; this adds corroborating
    data and the energy metric.

### Batch 64 forum ingest (2026-08-11)

- **[conjecture]** **nvidia-conf-xconfig.service — package recovery on DGX Spark after
  broken apt state** (S-forum-xconfig-recovery, hpcm + elsaco + Neill/NVIDIA): the
  `nvidia-conf-xconfig.service` systemd unit is part of the `nvidia-conf-xconfig`
  package on DGX OS. Its service definition:
  ```ini
  [Unit]
  Description=NVIDIA Xconfig service
  Before=graphical.target gdm.service

  [Service]
  Type=oneshot
  ExecStart=/usr/sbin/nvidia-conf-xconfig

  [Install]
  RequiredBy=systemd-logind.service
  ```
  If the service file (or the package) is missing, **GDM cannot start** — the system
  boots to a black screen with `Unit nvidia-conf-xconfig.service not found`. The package
  is installable via `apt install --reinstall nvidia-conf-xconfig` **only if the Spark
  APT repository is intact**. The repo config lives at `/etc/apt/sources.list.d/spark.sources`
  with a corresponding GPG key — both can be lost if a user manually deletes packages
  (e.g. during recovery from an unsupported kernel/driver upgrade). If the repo file is
  gone, `apt` reports `Unable to locate package nvidia-conf-xconfig`. **Recovery path:**
  extract `spark.sources` and the GPG key from the DGX Spark System Recovery image, then
  `apt install --reinstall nvidia-conf-xconfig`. A dummy service (`ExecStart=/bin/true`)
  restores the GUI but may have unexpected side effects — proper package reinstall is
  preferred. NVIDIA staff (Neill) recommends the DGX Dashboard as the update path and
  the System Recovery image as the fallback for badly damaged package state. This
  corroborates the existing finding that **plain `apt upgrade` on a Spark can break the
  kernel+driver pairing** (S-forum-driver580-173): the user upgraded to an unsupported
  kernel, then driver 595 via `ubuntu-drivers devices`, which cascaded into package
  removals. Single forum thread → [conjecture].

### Batch 66 forum ingest (2026-08-12)

- **[reported]** **Fans stop when display blanks (DPMS off) or headless — GB10 fan controller
  tied to SoC power draw, not thermal sensors** (S-forum-fan-dpms, dmayer1 + x1917x + sjug;
  NVIDIA staff Neill engaged): a distinct overheating mechanism from the EC fan-curve
  regression (S-forum-ec-fan-rollback). **Symptom:** when the display powers off (DPMS
  blank) or the unit runs headless with no display detected, the fans stop entirely —
  no RPM reported via hwmon. The chassis becomes "too hot to touch" while `nvidia-smi`
  shows the GPU idle at P8, 3W, 0% util. ACPI thermal zones climb to 52-56°C with no
  fan response. Wiggling the mouse / waking the display spins the fans back up within
  seconds. Reproducible on demand via `xset dpms force off`. Field diagnostics pass
  (cooling hardware is healthy) — the fans only fail to engage based on display/power
  state, pointing to firmware/EC logic.

  **Root cause (community-discovered):** the fan controller responds to **SoC power
  draw**, not thermal sensor readings. x1917x demonstrated this systematically with a
  smart outlet: both a "hot" (fans off) and "cold" (fans on) unit draw the same ~25W
  at idle, but adding any external load that raises SoC power above ~29-30W spins the
  fans:
  - USB load ≥5W (e.g. charging a phone at 5V/1A) → fans start, unit cools to ~38°C
  - VNC session with active rendering app (System Monitor/DGX Dashboard) → +8W → fans on
  - Connected monitor + keyboard + USB hub → +13W → fans on
  - Idle VNC session (static screen, no rendering) → insufficient power draw → fans stay off
  - Threshold: fans slow below ~+4-4.5W of additional load

  **This is a different mechanism from the EC fan-curve regression** (S-forum-ec-fan-
  rollback / S-forum-ec-fan-asus): the EC fan-curve issue is a *broken temperature-to-
  fan-speed mapping* (fans spin but too slow at high temps). This DPMS/power-draw issue
  is the fan controller *not engaging at all* when SoC power draw is low, regardless of
  temperature. Both produce overheating, but the root causes are distinct.

  **Driver version correlation:** x1917x's "cold" unit (fans always work) runs driver
  580.159.03; the "hot" unit (fans stop) runs 580.173.02 — suggesting the driver/firmware
  update may be a trigger. sjug reports the issue appeared after a fresh upgrade from
  OOBE, and downgrading EC firmware helps if CX7 NIC is connected for the update.

  **Workarounds (NVIDIA staff-confirmed):**
  1. `xset -dpms` — disable display sleep if a monitor is connected
  2. VNC with an active rendering app (idle VNC is not sufficient)
  3. Any sustained USB load on the device (≥5W, e.g. phone charging)
  4. Do not leave the unit fully idle and headless without one of the above

  **Status:** `open` — NVIDIA engineering investigating. dmayer1's unit additionally
  failed CX7Stress field diagnostic → RMA; the CX7 failure may block the EC firmware
  downgrade workaround path. Multiple independent users (dmayer1 on DGX Spark FE,
  x1917x on 2× ASUS GX10, sjug) confirm the core symptom → [reported]. The power-draw
  threshold finding is from a single systematic investigator (x1917x) → [conjecture]
  for the specific threshold value.

- **[conjecture]** **Driver 595.58.03 / CUDA 13.2 not yet supported on DGX Spark**
  (S-forum-driver595, chrm + aniculescu + _cjg): NVIDIA staff (aniculescu) confirms
  driver 595 is not yet supported on DGX Spark. The 595.58.03 certified Linux-aarch64
  release's supported-devices list does not include GB10. CUDA 13.2 "had some issues,"
  and the community speculates the Spark may jump directly to CUDA 13.3 (possibly
  August). This is consistent with the existing [conjecture] finding that driver
  610.43.02 + CUDA 13.3 works on Spark (S-forum-driver610) — the supported path may
  skip 595/13.2 entirely. Single thread, multiple users → [conjecture] (NVIDIA staff
  comment is authoritative but informal, not an official roadmap statement).

- **[conjecture]** **TensorRT-LLM one-forward-pass readout engine — extraordinary
  unverified claim** (S-forum-trtllm-readout, lcoleman0422): a forum post claims a
  TensorRT-LLM branch (`nemoclaw/v9.16-rc14`) that replaces autoregressive decode
  with a single base-model forward pass + non-autoregressive readout decoder, measured
  on 3× DGX Spark with `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`:
  - 1,014 tokens in 92.3 ms (live request); 8,192-token envelope in 511 ms
  - sm_121, CUDA 13.0, TensorRT 10.14.1.48
  - `NemotronHForCausalLM`: 52 layers, hybrid Mamba+MLP+attention, hidden 2,688, vocab 131,072
  - Chunked vocabulary projection (8 chunks × 16,384 entries), functional sinusoidal
    positions (no learned table), in-graph EOS scan, autoregressive fallback hard-disabled

  **Evidence assessment:** the poster self-describes as a "forward-deployed engineer"
  with "no formal experience in this field" and acknowledges early work was "vibe-coded."
  The claim of generating 1,014 tokens in 92.3 ms (~10,990 tok/s) is ~100-1000× beyond
  any known GB10 inference method and would require a fundamentally new architecture
  (non-autoregressive readout from a single hidden state). No independent verification,
  no reproducibility details sufficient to replicate, no peer review. The repo
  (`github.com/lcoleman0422/TensorRT-LLM`) is linked but the approach has not been
  validated by any other source. **Tagged [conjecture] — this is an extraordinary claim
  that requires extraordinary evidence.** A hardware agent with 3× Spark could attempt
  reproduction, but the approach itself (non-autoregressive readout from frozen hidden
  state) is not a known working paradigm for general text generation. Queue for
  observation only; do not cite as a performance benchmark.

### Batch 69 forum ingest (2026-08-14)

- **[conjecture]** **Sparkup Ansible playbook — `spbm` firmware module for whole-system power
  telemetry** (S-forum-sparkup, vladtemian): an open-source Ansible playbook that provisions
  a DGX Spark from a fresh DGX OS install to a working training box (Docker + NVIDIA runtime,
  users + GitHub SSH keys, signed kernel, firmware staged, ufw). The notable GB10-specific
  component is the **`spbm` firmware module** (OFF by default) that exposes **whole-system
  power** (wall draw) into Prometheus — because `nvidia-smi` sees only the GPU rail, which is
  about half the wall draw. This corroborates the existing **[reported]** finding that
  `nvidia-smi` accounts for only 12-27% of real GB10 power draw (S-forum-clock-energy-sweep).
  Observability stack: node + GPU exporters, Prometheus with 1-year retention, Grafana on
  `http://spark.local` with a provisioned dashboard. GitHub: `vtemian/sparkup`. Single source
  → [conjecture]. Relevant for cluster operators wanting wall-power monitoring without
  external smart plugs (cf. S-forum-power-mgmt smart-plug pattern).

### Batch 71 forum ingest (2026-08-15)

- **[conjecture]** **GB10 spontaneous reboots after July 2026 firmware bundle — GSP health
  check fail + Xid 120 + NVRM assert flood; root cause is warm-rebooted fwupd capsules, fix is
  full AC power disconnect** (S-forum-gsp-reboot-jul2026, nathanviveiros + dgxspark + elsaco):
  a Dell Pro Max with GB10 (kernel 6.17.0-1029-nvidia, driver 580.173.02, CUDA 13.0, headless)
  applied the July 2026 system update bundle: EC 0x01000800 → 0x02000b00, SBIOS 5.36_2.1.0 →
  5.36_4.0.0, device firmware 0x507 → 0x516. The capsules applied on warm reboots only. Within
  ~4 hours, the machine spontaneously rebooted 3 times at near-exact 2-hour intervals (14:07,
  16:07, 18:16). Crash signatures:

  1. **GSP health check failure**: `NVRM: kgspHealthCheck_TU102` + `_issueRpcAndWait:
     rpcRecvPoll failed with status 0x00000062 for fn 76 sequence 5742` + GSP "critical error
     120" — the same `0x62` status seen in existing GSP init timeout reports
     (S-forum-gsp-timeout).
  2. **NVRM assert flood**: hundreds of repeated
     `nvAssertFailedNoLog: Assertion failed: (status == NV_OK) || (status ==
     NV_ERR_GPU_IN_FULLCHIP_RESET) @ gpu_user_shared_data.c:373` over ~12 seconds, then reboot.
  3. **Silent wedge**: third crash had NO NVRM/GSP errors at all — journal simply stops. The
     GSP errors in crashes 1-2 are likely a symptom, not root cause.
  4. **Xid 120 mid-uptime** (spotted by elsaco): `Xid 120, GSP task exception: supervisor timer
     interrupt (cause:0x8000000000000005) @ pc:0x1a232ec, partition:4#0, task:3` — GSP throwing
     exceptions during normal operation well before each crash. Per the Xid catalog, XID=120
     indicates an error in GSP core code or a timeout waiting for GSP to respond to an RPC.

  **Watchdog mechanism:** `sbsa_gwdt` is loaded and armed with `action=1` (panic on WS0
  interrupt). This is the DGX OS default — `/etc/modprobe.d/sbsa_gwdt.conf` contains
  `options sbsa_gwdt action=1`, owned by the `nvidia-sbsa-gwdt-options` package. The reboots
  are the watchdog firing after the GPU driver wedges, not a power or EC issue. The older
  sbsa_gwdt blacklist issue from prior reboot threads does not apply.

  **Root cause + fix:** `fwupd-refresh.service` ran seconds before two of the three crashes,
  making it the initial suspect (masked as a test). However, the real root cause was that the
  fwupd capsules (EC + SBIOS + device firmware) were applied on **warm reboots only** — a
  full AC power disconnect was never done. Per dgxspark's suggestion, a clean shutdown +
  wall power disconnect for a few minutes at ~19:00 local resulted in **24+ hours clean**:
  zero reboots, zero Xid/NVRM/GSP events. `fwupd-refresh.timer` was subsequently unmasked and
  fired cleanly with no GPU events — exonerated. It was only ever poking firmware that had
  been applied without a full power cycle.

  **Takeaway:** after fwupd applies EC/SBIOS capsules on a GB10 (Dell or FE), **do a full AC
  power disconnect, not just a reboot.** Warm-rebooted firmware left the GSP throwing task
  exceptions and health-check failures until the cold cycle. This is consistent with the
  existing [proven] power-controller wedge finding that a soft reboot does not clear firmware-
  level state — the same AC-disconnect fix applies. This finding extends the pattern: the
  issue isn't limited to post-crash recovery; it can also manifest after routine firmware
  updates applied without a cold power cycle. The Xid 120 / GSP task exception is a new
  symptom variant (existing reports had Xid 119 GSP_INIT_DONE timeout, S-forum-gsp-timeout).
  Note: Dell Pro Max EC versioning (0x02000b00) differs from FE Spark (0x3000508) — the FE
  comparison doesn't apply to Dell units. Single thread → [conjecture]. Flagged for hardware
  verification: any GB10 unit receiving the July 2026 bundle should cold-cycle to confirm.

### Batch 73 forum ingest (2026-08-16)

- **[conjecture]** **GB10 Grace CPU energy telemetry — full audit: GPU rail measurable via
  DCGM, CPU rail is a blind spot; SPBM driver fails on GX10 due to MTKW9000 ACPI memory
  conflict; spark_hwmon works on Acer GN100; July EC update does NOT fix the gap**
  (S-forum-energy-telemetry, deepak.panigrahy03 + aniculescu): a systematic audit of all
  seven known energy interfaces on GB10 (DGX Spark / ASUS GX10, not GH200), now
  peer-reviewed and accepted at LOCO 2026 workshop (arXiv:2605.27599). Findings:

  **What works (GPU energy):**
  - **DCGM field 156** (`total_energy_consumption`) — cumulative millijoule counter,
    delta-read method validated. Covers GPU compute rail only. Install:
    `sudo apt install -y datacenter-gpu-manager && sudo systemctl start nvidia-dcgm`,
    then `dcgmi dmon -e 155,156 -c 3`. Counter increments ~3,471 mJ between 1-second
    samples at 3.5W idle — consistent with field 155 instantaneous watts.
  - **DCGM field 155/157** — instantaneous watts, working.
  - **ARMv8 PMUv3** — 70+ performance counter events per cluster (cycles, IPC, L1/L2/L3
    cache, branch prediction) after `perf_event_paranoid = -1`. Zero energy events.

  **What does NOT work (CPU/system energy):**
  - `nvidia-smi --query-gpu=energy.consumption` → "Field not valid."
  - SCMI bus active (scmi-clocks, scmi-regulator, scmi-mpam loaded) but **no powercap
    or sensor protocol** present.
  - All six I2C buses empty — no INA monitors on the board.
  - No `hwmon` `energy_uj` or `power_input` files anywhere.
  - **DCGM fields 1130/1132/1133** (CPU/SysIO/Module power) — recognized by DCGM but
    return `0.000` on this firmware.
  - `/sys/class/powercap/` is **empty** on the most current retail DGX Spark firmware
    (BIOS 5.36_0ACUM027, Jun 2026 + July EC update 0x03000508).
  - `stress-ng` 8-core CPU load (15s) → **zero change** in DCGM field 156. Grace CPU
    rail confirmed excluded empirically.

  **SPBM driver (NVDA8800:00) — the core conflict:**
  - NVDA8800:00 is present at `_SB_.MTEL`. SPBM driver v0.3.0 installed via DKMS.
  - **Fails on GX10/DGX Spark at boot**: `platform NVDA8800:00: failed to claim resource
    0: [mem 0x05170000-0x051cffff]` → `acpi NVDA8800:00: platform device creation
    failed: -16 (EBUSY)`.
  - **Root cause**: MTKW9000:00 (MediaTek wireless peripheral) claims overlapping memory
    `[0x05160000-0x051affff]` before SPBM initializes. The SSPM shared memory SPBM needs
    falls within MTKW9000's claimed range. This is a **firmware ACPI resource allocation
    conflict**, not a driver bug.
  - **Board-specific**: on the **Acer Veriton GN100** (same GB10 SoC, kernel
    6.17.0-1021-nvidia), `spark_hwmon` (antheas/spark_hwmon) loads cleanly with no
    resource conflict: `spbm NVDA8800:00: resolved 45/45 register offsets from _DSM` →
    14 power + 4 energy + 8 temp channels readable. Acer's firmware allocates resources
    without overlap. The fix on GX10 is a **firmware ACPI table change**, not hardware.

  **GN100 full energy chain (spark_hwmon working):**
  - `sys_total: 27,686 mW`, `dc_input: 29,125 mW`, `soc_pkg: 17,069 mW`,
    `cpu_gpu: 5,773 mW`, `cpu_p: 454 mW` (P-cores), `cpu_e: 19 mW` (E-cores),
    `gpu: 4,853 mW`, `dla: 1 mW`.
  - Cumulative energy accumulators in µJ: `pkg`, `cpu_p`, `cpu_e`, `gpu` all incrementing.
  - DCGM field 156 rate (~4,437 mJ/s) matches SPBM gpu accumulator (~5,354 mW) with
    ~992 mW difference — consistent with GPU memory and NVLink-C2C overhead that DCGM
    doesn't count. The two interfaces are complementary.

  **NVIDIA staff response + firmware status:**
  - aniculescu (NVIDIA): "This should be fixed in our July Updates." User checked: the
    July update is an **EC firmware update only** (0x03000302 → 0x03000508, LVFS Release
    ID 143461), unrelated to energy attribution. `fwupdmgr get-updates` returns nothing
    further — the retail DGX Spark is fully up to date.
  - Full platform device scan after the July EC update: NVDA8800:00 still has no driver
    bound, `/sys/class/powercap/` remains empty.
  - The process-level GPU energy attribution gap persists on the most current firmware
    available on any GB10 unit as of August 14, 2026.

  **Why this matters for agentic AI on Spark:**
  - Agentic workloads are dominated by CPU-bound orchestration, not inference compute.
    Measurements show OOI (orchestration overhead index) reaching 4.33×–7.63× over
    linear baselines. Without CPU energy counters, orchestration overhead is
    unmeasurable on GB10 — precisely the component that dominates cost.

  **Implication for sparkbase**: this finding strongly corroborates the existing
  **[reported]** finding that `nvidia-smi` accounts for only 12-27% of real GB10 power
  draw (S-forum-clock-energy-sweep). The DCGM field 156 method is the practical GPU-side
  energy measurement tool today. The CPU rail gap means any energy-efficiency
  calculation on GB10 that relies on `nvidia-smi` or DCGM alone is missing the CPU
  component — which dominates agentic workloads. Single thread (one researcher, one
  paper) → [conjecture], though the audit is thorough and peer-reviewed. Flagged for
  hardware verification: a hardware agent could run `dcgmi dmon -e 155,156` and verify
  the DCGM field 156 method on their own unit.

- **[conjecture]** **ASM2464PD USB4 NVMe enclosure falls back to USB 2.0 (480 Mbps) on
  ASUS GX10 after every boot — soft-replug script automates the fix**
  (S-forum-asm2464pd-replug, JW2026 + paulsc.liu + gaborm + vedcsolution): ASM2464PD-based
  USB4 NVMe enclosures consistently fall back to USB 2.0 speed (480 Mbps) on the ASUS
  Ascent GX10 after every power-up. Manual cable replug restores full 20 Gbps (20000M/x2).
  A community script (`jsconsultancy/asm2464pd-soft-replug`) automates the reset after
  boot by sending a software-triggered CPU reset to the ASM2464PD controller (based on
  reverse engineering from `cyrozap/usb-to-pcie-re`: the `e8 50 + 13×00` register write
  sequence). Users modify VID/PID, mount point, and expected speed in the script. Multiple
  users confirm the issue has existed since October 2025. This is related to but distinct
  from the existing USB2 fallback finding (S-forum-usb2-fallback) — that finding covers
  the MediaTek T-PHY ACPI binding gap affecting all USB3 SuperSpeed; this finding is
  specific to ASM2464PD USB4 enclosures and has a software workaround. Single thread,
  multiple users agreeing on the symptom → [conjecture] (the workaround is a single-source
  script). Relevant for Spark users relying on external USB4 NVMe for model storage.

### Batch 74 forum ingest (2026-08-17)

- **[conjecture]** **2× DGX Spark FE silent hard-locks under sustained DSV4-Flash-0731
  inference — fieldiag PowerStress FAIL on both units, RMA approved ~48h**
  (S-forum-fe-thermal-rma, tniccum): Two Founders Edition units hard-locked silently
  under sustained long-context inference (DeepSeek-V4-Flash-0731, vLLM TP=2 over
  dual-rail RoCE, 262K-token prefill). **12 lock events across the pair over 3 days.**
  No panic, no OOM, no NVRM error, no shutdown record; peer logs `mlx5 Port: 1 Link
  DOWN`. Physical power cycle required — a 120 s systemd hardware watchdog failed to
  recover 3 of 4 events. At the moment of death: `CLOCK THROTTLED 2197/3003 MHz
  [HW_THERMAL_SLOWDOWN]`, GPU 88→90 °C, CPU zones 97→98 °C, ACPI zones 92–98 °C
  against a single critical trip at 104 °C. Serving a single ordinary request drives
  the hotter unit's zones to 97.4 °C. Fans remain inaudible throughout (EC-internal
  fan control, no OS surface — corroborates S-forum-fan-firmware). One unit was
  dust-degraded (failed GpuStress before intake cleaning, passed after — brush the
  front grille); the PowerStress failure survives cleaning on both units.
  - **fieldiag PowerStress FAIL — 3rd independent report, now on Founders Edition:**
    `partnerdiag --field --run_on_error`, stock clocks, Secure Boot disabled. Results
    identical across 3 runs per unit (fieldiag 1.0.9 on 2026-08-10, 2.0.4 on 08-11/12):
    GpuStress, C2C, CpuStress1/2, ThermalStress, FioSSD, MemStress, CX7Stress all OK;
    **PowerStress FAIL @ ~8:10** on both units. Error **020000600139** ("Acceptable
    temperature limits exceeded or the thermal sensor is broken or miscalibrated") —
    same error class as S-forum-powerstress (082-000-1-020000600139) and
    S-forum-thermal-freeze (MODS-020000610139). During the very first run, unit A
    **hard-locked mid-PowerStress with no OS loaded** (MODS diagnostic driver only),
    eliminating all software explanations. This is the **3rd independent forum thread**
    documenting PowerStress thermal failure on GB10 (after S-forum-powerstress,
    S-forum-thermal-freeze) — strengthens the pattern to [reported]-level consensus
    on the symptom, but the analysis-agent ceiling caps the promotion here at
    [conjecture] for this single-source thread's specific claims. FE config: DGX OS
    7.2.3, OTA 7.5.0, kernel 6.17.0-1026-nvidia, driver 580.173.02, EC 0x03000508;
    EC rollback to 0x02004e18 tested — no effect on FE (OEM fan-curve issue does not
    apply to Founders Edition).
  - **Mitigation 1 — GPU clock cap (`nvidia-smi -lgc 300,2100`) stabilizes the cluster:**
    2100 MHz sits just under the observed throttle floor of ~2197 MHz. After capping,
    HW Thermal Slowdown drops to 0 µs cumulative on both nodes, and ~12 consecutive
    262K-context runs against DSV4-Flash complete with zero locks (previously 3/3
    killed a node). Measured cost: **−21% decode / −7% prefill at 32K depth**; at
    262K depth decode actually improved **2.3×** because a stable clock beats one
    oscillating in and out of thermal slowdown. 2200 MHz also survives but re-engages
    mild throttling on the hotter unit. The lock does not survive reboots — persist
    via a systemd oneshot. Corroborates the existing **[reported]** clock-cap
    mitigation pattern (S-forum-clock-energy-sweep, S-forum-cooler-temps,
    S-forum-thermal-freeze). Note this is a *higher* cap than the 2000 MHz commonly
    recommended — the OP measured 2100 as the clean value for their units.
  - **Mitigation 2 — CPU frequency cap (`scaling_max_freq` → 2.4 GHz) is free:**
    the hottest sensors are CPU-cluster zones, 8–10 °C above the GPU. vLLM TP workers
    busy-poll at 200–350% CPU, spinning cores at up to 3.9 GHz doing nothing useful.
    Capping all 20 cores to 2.4 GHz (stock ships the performance governor, all cores
    unpinned): sustained zones **92→84 °C** (hot unit) / 85→80 °C, **identical wall
    time — zero performance cost**. Decode at unchanged GPU cap actually improved
    **16%** (thermal jitter had been costing throughput). This is a **new durable
    GB10-specific finding**: the CPU governor is not the lever — only capping the
    active cores' max frequency helps, and it is free thermal headroom for any
    multi-node vLLM deployment where TP workers busy-poll. Single source → [conjecture].
    Flagged for hardware verification: a hardware agent could A/B the CPU freq cap
    on their own cluster and measure the thermal/throughput delta.
  - **fieldiag 2.0.4 install gotchas** (corroborates S-forum-ec-fan-asus,
    S-forum-powerstress on the ofed-scripts gap):
    - hard-depends on `ofed-scripts` (add the DOCA repo, then `apt install
      dgx-spark-fieldiag kernel-mft-dkms`);
    - in-place upgrade from 1.0.9 leaves the old launcher behind — `partnerdiag`
      aborts with "More than one fieldiag packages found" (remove the old
      `onediagfield.r9.257.3`);
    - CX7Stress leaves the ConnectX links DOWN when it finishes — `ip link set
      <if> up && netplan apply` to restore the fabric.
  - **RMA process notes**: fieldiag is the RMA qualification tool (per its user guide).
    Run it at stock config before filing and attach `summary.json`, `run.log`, and
    per-test logs. The portal accepts .zip/.txt/.pdf (not .md/.gz/.tgz — even when
    support asks for .tgz, wrap it in a .zip) and rejects attachments above ~10 MB.
    Exclude the ~46 MB GpuStress video stimulus files (.vp9/.h264) from the log dir.
    Standard RMA is ship-first. Both units RMA'd within ~48 h of filing
    (Case #260812-000102).

### Batch 75 forum ingest (2026-08-17)

- **[conjecture]** **Fans do not spin in headless boot — temperature rises to ~70°C;
  display-hotplug dependent, unit-specific** (S-forum-fan-headless-boot,
  jasonzhou_spk + aniculescu + josephbreda + solodu1116): On a DGX Spark FE (DGX OS
  7.4.0, driver 580.126.09, kernel 6.17.0-1008-nvidia, BIOS 5.36_0ACUM018), booting
  headless (no HDMI monitor connected) causes the fans to remain inactive even as
  the chassis temperature rises to 60–70°C at idle (GPU P8, 3W, 0% util). Connecting
  an HDMI monitor and rebooting restores normal fan behavior, stabilizing temps at
  35–40°C. `gnome-remote-desktop-daemon` appears in the process list when a monitor
  is connected. NVIDIA staff (aniculescu) could not reproduce and asked whether the
  display manager was still active in headless mode; josephbreda reports running
  dual Sparks fully headless 24/7 with no such behavior. A third user (solodu1116)
  reproduced a closely related issue on **1 of 2 identical ASUS Ascent GX10 units**
  (both same kernel 6.17.0-1029-nvidia, driver 580.173.02, DGX OTA 7.5.0, same BIOS/
  EC/UEFI/PD firmware): the affected unit idled at GPU 55–58°C / ACPI 58.5–61.7°C
  (P8, 3.8–4.0 W) with an audibly slower fan; the control unit (fully headless at
  GDM greeter, never logged in) stayed at GPU 34–35°C. On the affected unit, HDMI +
  local X11 login reduced temps to 36–40°C; setting GNOME blank-screen timeout to
  Never cooled further to 36–37°C; physically unplugging HDMI while the session
  stayed active caused a monotonic rise to GPU 45°C / ACPI 47.9°C over 5 minutes
  (P8, 3.48–3.71 W). This is **display-hotplug dependent, not purely headless**,
  and only manifests on some units. **This corroborates the existing [reported]
  finding that GB10 fan control is tied to display/SoC-power-draw state, not
  thermal sensors** (S-forum-fan-dpms): the headless-boot symptom is a specific
  manifestation of the same fan-controller-doesn't-engage pattern. The
  unit-specific nature (1 of 2 identical units) is new data — suggests a hardware
  or EC firmware variation, not a universal platform bug. The older driver
  (580.126.09) on the OP's unit and newer driver (580.173.02) on solodu1116's
  affected unit both exhibit the issue → not driver-version-specific. No
  resolution provided in thread; NVIDIA staff engaged but unable to reproduce.
  Single thread (3 users, 1 reproducer + 1 non-reproducer + 1 partial reproducer)
  → [conjecture] for this specific headless-boot manifestation; corroborates
  existing [reported] fan-DPMS finding at the mechanism level. Flagged for
  hardware verification: a hardware agent could test whether `systemctl status
  display-manager` is active in headless mode and whether a dummy HDMI EDID
  emulator prevents the issue.

- **[conjecture]** **s2idle suspend fails on DGX Spark GB10 — nvidia-suspend.service
  crashes inside the driver (nv.c:4784), PCI PM returns -5** (S-forum-suspend-fail,
  tsetjpc): Running `sudo systemctl suspend` on a DGX Spark FE (kernel
  6.17.0-1029-nvidia, driver 580.173.02 open kernel module for aarch64, BIOS
  5.36_0ACUM018, s2idle mode confirmed via dmesg) starts the suspend sequence but
  never actually suspends. `nvidia-suspend.service` fails with exit code 1;
  `/usr/bin/nvidia-sleep.sh` line 45 (`echo "$1" > /proc/driver/nvidia/suspend`)
  hits an I/O error. dmesg shows a kernel **WARNING at `nv_set_system_power_state`
  (nv.c:4784)** inside the nvidia driver, then the generic PCI PM layer fails with
  **error -5** (`pci_pm_suspend(): nv_pmops_suspend [nvidia] returns -5`), and the
  system immediately resumes a fraction of a second after entering. Diagnostics:
  all three nvidia power-management services (`nvidia-suspend`, `nvidia-resume`,
  `nvidia-hibernate`) are enabled; `/sys/module/nvidia/parameters/` **does not
  exist** (no `PreserveVideoMemoryAllocations` / `TemporaryFilePath` sysfs
  entries), even though `modinfo nvidia` confirms the module declares
  `NVreg_PreserveVideoMemoryAllocations`; `/etc/modprobe.d/nvidia-power-
  management.conf` does not exist. This is consistent with the existing
  **[conjecture]** finding that sleep/suspend is **disabled by default on DGX OS**
  (S-forum-sleep-disabled, allanmac + aniculescu) — this user attempted to override
  the default and discovered the driver itself cannot complete the suspend path.
  The `nv.c:4784` WARNING is a new durable error string for the KB. The missing
  `/sys/module/nvidia/parameters/` path suggests the open kernel module may not
  fully instantiate the power-management sysfs surface on GB10/aarch64, or that
  the `NVreg_PreserveVideoMemoryAllocations` parameter needs to be explicitly set
  at module load time. Single post, no replies, no resolution → [conjecture].
  Relevant for users who attempt to enable suspend on a Spark (against the default
  disabled configuration): the driver-level suspend path is broken, not just
  disabled by convention. Corroborates the platform-level guidance that **suspend
  is not a viable power-management mechanism on GB10** — use full shutdown + smart
  plug instead (S-forum-power-mgmt [reported]).

### Batch 77 forum ingest (2026-08-18)

- **[conjecture]** **HDMI hot-plug A/B test confirms display-state → fan/thermal
  link — per-unit, not purely headless** (S-forum-hdmi-hotplug-ab, solodu1116 +
  x1917x + ajvazan): A controlled experiment on **2 identical ASUS Ascent GX10
  units** (both kernel 6.17.0-1029-nvidia, driver 580.173.02, DGX OTA 7.5.0,
  same BIOS/EC/UEFI/PD firmware) isolates the display-hotplug variable from
  headless operation. **Affected unit:** after cold start, GPU rose from 38°C
  to 48°C in 10 min, then cycled at 55–58°C GPU / 58.5–61.7°C ACPI (P8, 0%
  util, 3.8–4.0 W) with audibly slower fan. **Control unit** (fully headless
  at GDM greeter, never logged in, no DRM display): stayed at 34–36°C GPU,
  4.6–4.8 W — **headless alone does not reproduce the issue**. **Clean A/B on
  affected unit:** HDMI connected + local X11 session + GNOME blank timeout
  Never → cooled to 36–37°C GPU / 39.8°C ACPI. Physically unplugging HDMI
  (session stays active, IdleHint=no) → monotonic rise to **45°C GPU / 47.9°C
  ACPI in 5 min** (P8, 3.48–3.71 W, 4–6% util). The physical EDID disappeared,
  leaving a zero-byte `Unknown-1` virtual connector. **This links physical
  display hot-plug state to thermal/fan behavior independently of compute
  load.** `nvidia-smi` reports fan speed N/A; no Linux fan input or PWM nodes
  on either unit → actual RPM cannot be compared. x1917x provides 4
  workarounds (USB ≥5 W load, monitor+keyboard+mouse, VNC with active app,
  CX7 cable to another Spark). ajvazan suggests HDMI dummy plug. **Corroborates
  existing [reported] fan-DPMS finding** (S-forum-fan-dpms): same mechanism
  (fan controller tied to display/SoC-power-draw state, not thermal sensors).
  New data: (1) headless alone is not sufficient — the affected unit had a
  display *connected* then *disconnected*, while the control unit never had
  one; (2) the per-unit nature (1 of 2 identical units) suggests an EC firmware
  or fan hardware variation. Single thread → [conjecture] for this specific
  A/B; corroborates [reported] fan-DPMS at the mechanism level.

- **[conjecture]** **USB-C DisplayPort ports (DFP-1 to DFP-4) not detected
  after boot unless monitor cable physically replugged** (S-forum-usbc-dp-hpd,
  riccardo1981 + helge + Mkei88): On MSI EdgeXpert GB10 (DGX OS, kernel
  6.17.0-1029-nvidia, driver 580.173.02), none of the 4 USB-C DisplayPort
  outputs are detected after cold boot even when a monitor is connected via
  USB-C→VGA adapter and powered on before/during boot. Only HDMI-0 works at
  boot. Physical unplug+replug of USB-C immediately triggers detection (EDID
  read confirmed via `journalctl`). **Software re-probe fails:** `xrandr
  --output USB-C-0 --off` then `--auto`, and `nvidia-settings -q dpys` polling
  over several minutes — all report disconnected. `/sys/class/typec/` is empty
  (no typec-class controller exposed); `/sys/bus/thunderbolt/devices/` is empty
  (not USB4/TBT tunneling). **Root cause hypothesis:** HPD/sink-detection issue
  at the PD controller or mux level — the detection window closes early in
  boot sequencing. If the monitor isn't already powered and negotiated by the
  time the GB10 boot sequence passes, the query is never repeated. Second
  confirmer (helge) on **Lenovo Thinkstation PGX** with native USB-C (no
  adapter) — same behavior, including monitors that draw power from the USB-C
  port. Third user (Mkei88) with ASUS + MSI using direct USB-C-to-USB-C to
  portable monitor reports no issue — suggests adapter negotiation latency may
  be a factor on some setups. Firmware fully up to date (`fwupdmgr get-updates`
  shows latest EC + UEFI). **Practical impact:** during firmware updates
  (multiple reboots, some taking minutes), users are "completely blind" without
  HDMI attached — strongly recommend HDMI monitor during updates. Single
  thread (3 users, 2 confirmers + 1 non-reproducer) → [conjecture]. New
  durable GB10 platform bug: USB-C DP HPD only runs once in a narrow early-boot
  window. No software workaround — requires physical replug.

### Batch 80 forum ingest (2026-08-20)

- **[conjecture]** **ASUS GX10 firmware recovery from bricked state —
  interrupted firmware flash bricks the unit; manual capsule recovery without
  RMA** (S-forum-gx10-fw-recovery, aquaponicCowboy + Neill): Two ASUS Ascent
  GX10 units froze on the ASUS splash screen after `apt update && apt upgrade`.
  Root cause: the firmware flash takes several minutes and cycles through
  multiple resets (SoC → BIOS → EC); it **looks hung partway through**, and
  power-cycling during the flash is what bricks it. This is the lethal
  combination — the firmware update itself isn't fatal, but an interrupted
  reboot mid-flash is. NVIDIA staff (Neill) confirms the multi-reset flash
  behavior is expected, especially via USB-C→HDMI (multiple blank-screen
  periods). **Key takeaway: never power-cycle during a firmware update — wait
  patiently through all blank screens.**

  **Recovery procedure (no data loss, no RMA):**
  1. **Power-drain reset** (try first): unplug, hold power button ~60s,
     reconnect, power on. Revived 1 of 2 units.
  2. If drain reset fails: the official GX10 recovery/rescue image
     **black-screens** on both installer and rescue options (can reach GRUB but
     selecting either → black screen). A **stock Ubuntu 24.04 arm64 live USB**
     boots fine — use "Try Ubuntu" for a working terminal. (Must be aarch64 —
     GX10 is Grace-Blackwell ARM.)
  3. From the live session: confirm `efivars` accessible
     (`ls /sys/firmware/efi/efivars`), mount the internal EFI System Partition
     (FAT32 on NVMe, e.g. `nvme0n1p1`), stage the ASUS `.cap` firmware capsule
     from a USB stick onto the ESP, arm `OsIndications` for capsule-on-disk,
     reboot — the capsule flashes on next boot.
  4. ASUS `.cap` capsules use **OS-driven capsule-on-disk**, not an in-BIOS
     flash menu (ASUS support incorrectly directed to a non-existent BIOS
     flash action).

  Corroborates existing firmware-update findings: the power-controller wedge
  pattern (S-forum-clock721, S-forum-power-crash, S-forum-gsp-reboot-jul2026)
  where AC power disconnect is the fix; the fwupd capsule-on-disk flow
  (S-forum-update-loop, S-forum-uefi-fw-fail); and the general guidance that
  firmware updates require patience and full AC power cycles. New durable
  data: (1) the official recovery image may black-screen — a stock arm64
  Ubuntu live USB is a viable alternative; (2) manual capsule staging onto the
  EFI partition from a live session is a working no-RMA recovery path for ASUS
  GX10. Single thread (2 users: OP + NVIDIA staff) → [conjecture].

### Batch 82 forum ingest (2026-08-21)

- **[conjecture]** **DGX Spark hardware crashes at ~75W, reboots repeatedly —
  clock cap 2100 MHz fixes** (S-forum-75w-crash, cory.farr): The system
  hard-crashes at around 75W power draw and reboots. Fix: `sudo nvidia-smi -lgc
  300,2100` caps the GPU clock at 2100 MHz, keeping power below the crash
  threshold. Release with `sudo nvidia-smi -rgc`. This corroborates the existing
  **[reported]** clock-cap mitigation (S-forum-comfyui-crash 2100 MHz,
  S-forum-gpu-throttle-cmd 2000 MHz, S-forum-power-90w 2200 MHz) with a new
  independent user reporting the same symptom and same fix. The ~75W crash
  threshold is consistent with the power-controller overcurrent protection
  pattern (S-forum-comfyui-crash: 85W transient trips overcurrent). Single
  source → [conjecture], but strengthens the existing [reported] finding with
  another corroboration. Notable: OP requests a "better built-in setup" —
  no persistent clock-cap mechanism is available in DGX OS by default (users
  must use a systemd unit, see S-forum-clock-energy-sweep).

- **[conjecture]** **Field Diagnostics install fails with Signed-By apt
  conflict — documentation error in Field Diagnostics guide** (S-forum-
  fieldiag-signedby, asdf.think365 + Neill): When installing the NVIDIA DGX
  Spark Field Diagnostics tool, the guide incorrectly instructs adding the
  CUDA apt repository — but DGX Spark OS already ships with that repo
  pre-configured under a different keyring name (`cuda_debian_prod.gpg` vs
  `cuda-archive-keyring.gpg`). This produces:
  ```
  E: Conflicting values set for option Signed-By regarding source
  .../compute/cuda/repos/ubuntu2404/sbsa/:
  /usr/share/keyrings/cuda_debian_prod.gpg != /usr/share/keyrings/cuda-archive-keyring.gpg
  E: The list of sources could not be read.
  ```
  **Fix (confirmed by NVIDIA staff Neill):** remove the duplicate entry:
  `sudo rm /etc/apt/sources.list.d/cuda-sbsa-ubuntu2404.list && sudo apt update`.
  This is a documentation/tooling gap, not a hardware bug — but it blocks
  running fieldiag on affected units, which impedes hardware triage. Adds to
  the existing fieldiag install gotchas catalog (S-forum-ec-fan-asus
  ofed-scripts dep gap, S-forum-fe-thermal-rma fieldiag 2.0.4 install issues,
  S-forum-powerstress secure-boot requirement). The thread also reports idle
  overheating with inaudible fans and system freezes within 15 min of boot
  while downloading models in LM Studio (<10% CPU/RAM) — consistent with the
  existing [reported] fan-DPMS / overheating patterns (S-forum-fan-dpms,
  S-forum-fan-headless-boot) but no new diagnostic findings beyond what is
  already documented. Single source (2-post thread, NVIDIA staff confirmed
  the fix) → [conjecture].

### Batch 85 forum ingest (2026-08-22)

- **[conjecture]** **Triton compilation crashes on sm_121a when building
  vLLM/PyTorch from source — ptxas "not implemented" errors** (S-forum-
  triton-sm121a, saskia.hold): Building vLLM + Triton + PyTorch from source
  on DGX Spark (non-Docker) produces Triton `CompilationError` in
  `qwen_gdn_linear_attn.py` at `make_ir`/`ast_to_ttir` — `ptxas` reports
  `sm_121a not implemented`. The from-source path yields 150–600 tok/s
  prefill on Qwen3.6-35B-A3B-NVFP4, vs 5000–6000 prefill + 90–110+ decode
  from sparkrun validated recipes (jomark). The performance gap is
  attributable to missing sm_121a kernel dispatch in the from-source build
  (falling back to unoptimized paths), not hardware. Community advice: use
  sparkrun or eugr's spark-vllm-docker for a validated sm_121a baseline
  (davedgd); Podman suggested as Docker alternative for organizations with
  Docker licensing constraints (wga472). The OP also requests CUDA 13.2 +
  native sm_121a toolchain support from NVIDIA. This corroborates the
  existing [reported] finding that sm_121 software support is severely
  lacking (S-forum-sm121-support: 43-post thread on SM121 support gaps).
  Single source → [conjecture]. See also
  `[[wiki/quantization-on-gb10.md]]` → kernel coverage gaps.


## [proven] Machine balance — what is worth optimising on GB10 (2026-08-23, S-gb10-profile)

**GB10's ridge point is ~916 FLOP per byte loaded** (~250 TFLOP/s dense FP4 over 273 GB/s). A weight
loaded once does 2 FLOPs per token in the batch, so **arithmetic intensity = 4 x (tokens per expert)**
for a 4-bit MoE:

| regime (256e / 8 active) | tokens/expert | arithmetic intensity | ceiling on FP4 utilisation |
|---|---|---|---|
| decode c1 | 0.03 | 0.1 | **~0%** |
| decode c64 | 2 | 8 | **0.9%** |
| prefill pp2048 | 64 | 256 | 28% |
| prefill chunk 16384 | 512 | 2048 | 100% |

**Saturating FP4 needs ~229 tokens per expert = ~7,300 concurrent tokens.** "The tensor cores are
95% idle during decode" is therefore **arithmetic, not waste** — at batch 1 you re-read every weight
to do two FLOPs with it, on any hardware. Kernel effort aimed at that is wasted.

**[proven] The only decode levers on this box** are (a) move fewer bytes/token — quantise, **and
verify it happened** (`[[wiki/quantization-on-gb10.md]]`, adoption); (b) raise tokens-per-forward
(concurrency, speculative decode); (c) remove per-step overhead (cudagraphs). Making the GEMM faster
is not among them. Prefill differs — a large chunked prefill can reach the ridge. (S-gb10-profile)
