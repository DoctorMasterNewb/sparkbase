# Platform: GB10 / DGX Spark

> **area:** platform
> **status:** stable
> **evidence:** proven
> **sources:** S-forum-update-loop, S-forum-temps-normal, S-forum-uvm-livelock, S-forum-sway-scanout, S-forum-realsense-d435, S-forum-6x-ring-rdma, S-forum-uefi-fw-fail, S-forum-serial-console, S-forum-sleep-disabled, S-forum-cx7-dac-power, S-forum-qwen3tts-ggml, S-forum-locateanything, S-forum-typec-thermal, S-forum-asus-fw-jul25, S-forum-comfyui-crash, S-forum-power-90w, S-forum-gpu-throttle-cmd, S-forum-driver580-173, S-forum-model-storage, S-forum-acer-thermal, S-forum-sm121-support, S-forum-170hx-spark
> **updated:** 2026-07-30

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
  long hot jobs who don't need 200GbE (and have 10GbE redundant paths). Single source → [conjecture].
  Corroborates existing [conjecture] that CX7 idle power nearly doubles when a cable is connected
  (S-forum-cx7-hotplug, mashie).
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
