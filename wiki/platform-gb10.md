# Platform: GB10 / DGX Spark

> **area:** platform
> **status:** stable
> **evidence:** proven
> **sources:** S-xnode-cudagraph, S-m3-vision, S-nemotron-rpc, S-networking, S-spark-powercap, S-dgxspark-report, S-forum-clock721, S-forum-power-crash, S-forum-15w-loop, S-forum-60w-cap, S-forum-power-spec, S-forum-tma, S-forum-thermal, S-forum-cooling-cage, S-forum-gsp-timeout, S-forum-driver610, S-forum-headless-boot, S-forum-cx7-bricked, S-forum-sdpa-corruption, S-forum-sage-attn, S-forum-vllm-2606-broken, S-forum-device-hang, S-forum-fwupd-mismatch, S-forum-gb10-baseline, S-forum-qwen-tts-arm64, S-forum-qwen35-lora-uma, S-forum-opal-uefi, S-forum-sunshine-rdp, S-forum-wan2gp-onnx, S-forum-thermal-shutdown, S-forum-nsight-remote, S-forum-onboarding, S-forum-clock-5min, S-forum-reboot-powercycle, S-forum-cx7-dual-setup, S-forum-hpc-slurm, S-forum-llama32-finetune, S-forum-cx7-hotplug, S-forum-comfyui-optimized, S-forum-nemotron-ollama, S-forum-nvfp4-broken, S-forum-usb2-fallback, S-forum-fw-july2026, S-forum-ota-loop, S-forum-asus-fw0103, S-forum-host-freeze-tp2, S-forum-machineid, S-forum-cve
> **updated:** 2026-07-18

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
