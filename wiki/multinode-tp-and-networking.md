# Multi-node TP & networking

> **area:** multinode
> **status:** stable
> **evidence:** proven
> **sources:** S-networking, S-mimo-results, S-m3-vision, S-xnode-cudagraph, S-sess-jun11, S-nemotron-rpc, S-pr46372, S-dgxspark-report, S-forum-cx7-13gbps, S-forum-mikrotik, S-forum-ddp-timeout, S-forum-2d-parallel, S-forum-sglang-traps, S-forum-glm47-rdma, S-forum-4node-mesh, S-forum-roce-397b-mtp, S-forum-ds4f-4x-vllm, S-forum-m25-sglang-4x, S-forum-3node-nccl, S-forum-mimo-2x-opt, S-forum-cx7-dual-setup, S-forum-4node-crs504, S-forum-qwen397-arch, S-forum-ibwrite-false, S-forum-glm52-8x, S-forum-asus-fw0103, S-forum-host-freeze-tp2, S-forum-nm-phantom, S-forum-sync-locale, S-forum-6x-cluster, S-forum-kimi-k3-ceiling, S-forum-inkling-nvfp4, S-forum-3node-mesh, S-forum-6x-ring-rdma, S-forum-m3-tp3
> **updated:** 2026-07-24

Two Sparks (242 GB combined) run models a single 121 GB node can't. The fabric works, but **no
GPUDirect** makes cross-node collectives host-staged — fine for latency-bound decode, costly for
throughput, and the source of the cudagraph wall (`[[wiki/cudagraphs-and-compile.md]]`).

## The fabric (ConnectX-7)

- **[proven]** Each Spark's QSFP port is **two PCIe5 x4 links** = two "twin" RoCE interfaces (e.g.
  `rocep1s0f1` + `roceP2p1s0f1`), each ~100G. To get full ~200G NCCL bandwidth you must use **both
  twins**: `export NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1`.
- **[proven]** Assign an IP to only **one** Ethernet twin per link (RoCE is what matters), **MTU 9000**
  (jumbo), and **never put both twins on the same subnet** (confuses autodiscovery/routing). The
  reference pair uses the direct-cabled **fabric subnet at MTU 9000** as the "stable fabric" for all
  inter-node traffic; the separate management LAN (`enP7s7`, MTU 1500, WiFi-adjacent) is **never** used
  for collectives.
- **[proven]** Verify the link before blaming software: `ib_write_bw <peer> -d rocep1s0f1 --report_gbits
  -q 4 -R` → expect ~111 Gb/s peak; `ib_write_lat` → ~1.5 µs. Healthy `ib_write_bw` but slow collectives
  = the no-GPUDirect host-bounce, not a broken cable.
- **[proven]** Connecting both ports gives no extra bandwidth. Daisy-chaining 3 Sparks sustains only
  100G/pair; >2 nodes well wants a QSFP switch and **powers-of-2 node counts** (2/4/8) for vLLM.

## Kernel / firmware fabric caveats (external best-practices, cross-checked 2026-07-01)

From an external DGX-Spark optimization report (S-dgxspark-report), verified against our cluster:

- **[reported]** **Kernel 6.17 RoCE regression.** Some 6.17 builds cap RoCE to **13–16 Gb/s** (link +
  latency look normal; only throughput collapses); the documented fix is downgrading to the **6.11**
  kernel. **[proven]** **Our `6.17.0-1026-nvidia` (DGX-OS) build is NOT affected** — TP=2 tok/s match
  published benchmarks, so it's evidently patched. If cross-node throughput mysteriously craters, **check
  the kernel first**.
- **[reported]** **ConnectX-7 firmware cap.** Old OEM CX-7 firmware can mis-report PCIe as **Gen1×1** and
  cap the bus to ~3 GB/s; update via `sudo fwupdmgr enable-remote lvfs-testing` → ~24 GB/s, then disable
  the remote. **[proven]** Ours is `28.45.4028` (recent) and healthy.
- **[proven]** **`NCCL_NET_GDR_LEVEL`** — external guides set `=0` (GPUDirect on the unified SoC memory
  can hard-lock the box). We use `=LOC`, functionally equivalent cross-node (there's no GPUDirect anyway).
- **[proven]** **Do NOT copy `NCCL_IB_GID_INDEX=3` from external guides.** Correct for a single-NIC box,
  but on our dual-NIC pair it gives NIC1 a *link-local* GID → QP hangs (see the per-NIC GID fix below).
  Use empty/auto.

**[proven]** External best-practices we already run (report ≈ our setup): dual-rail NIC binding, Ray V2 /
no-Ray `mp`, forcing TP comms onto the fabric, DSpark spec-decode, `flashinfer-cutlass` NVFP4 GEMM, MTP
where it helps, MTU 9000. Net: no unapplied win in that report; the biggest lever it *omits* is the 14 W
power wedge (`[[wiki/platform-gb10.md]]`).

## NCCL build / env

- **[proven]** Build NCCL for the arch: `make … NVCC_GENCODE="-gencode=arch=compute_121,code=sm_121"`
  (v2.28.x).
- Test across nodes: set `NCCL_SOCKET_IFNAME` / `UCX_NET_DEVICES` to the fabric Ethernet iface,
  `NCCL_IB_HCA` to both RoCE twins, `NCCL_IB_DISABLE=0`, then `all_gather_perf -b 16G -e 16G`.
- **[proven]** vLLM cross-node env seen in working recipes: `NCCL_CUMEM_ENABLE=0`, `NCCL_NVLS_ENABLE=0`,
  `VLLM_SKIP_P2P_CHECK=1`.
- **[proven]** **`NCCL_MAX_NCHANNELS=2` is REQUIRED on this pair** (launch tooling sets it; recipes pin
  it). The default (many) channels **hangs RDMA QP setup** → `init_process_group` / first collective never
  returns, silently. Also pin `NCCL_PROTO=LL` and `NCCL_NET_GDR_LEVEL=LOC` (the proven decode-latency
  settings). Hard-won: a hand-rolled 2-node NCCL job that omits `NCCL_MAX_NCHANNELS=2` hangs at init with
  zero output — looks like a rendezvous/firewall problem but is the channel count. (S-pr46372 debugging.)

## Distributed executor: no-ray wins

- **[proven]** **Ray V2 hangs cross-node on this pair.** Placement group is created and both GPUs
  reserved, but `RayWorkerProc` actors wedge in cross-node init before weight load (also: InstantTensor
  loader broadcasts cross-node and hangs). Confirmed across MiMo and MiniMax bring-ups.
- **[proven]** **Use native multi-node (mp / vllm-distributed):** `--nnodes 2 --node-rank {0,1}
  --master-addr <head-ip>:25000` (example fabric IP — substitute yours), default loader (NOT
  instanttensor). NCCL world_size=2, PYNCCL all-reduce, no hang. Recipes carry a `-noray` suffix for this.
- **[proven]** **Exception:** a 1M-ctx + NVFP4-KV MiMo recipe runs Ray successfully — Ray isn't *always*
  broken, but **no-ray is the safe default**; reach for Ray only with a proven recipe.
- **[proven]** **`--disable-custom-all-reduce`** — the custom all-reduce path is unreliable on sm_121
  (single-node/NVLink-only design); with TP=2 every layer all-reduces so one bad reduce corrupts
  everything. Plain NCCL/PYNCCL works.

## Cross-node bring-up: the mgmt-IP wall (root-caused 2026-06-30)

> **[proven]** **Symptom → Root cause → Fix.** The single thing that breaks cross-node launch on a
> dual-homed cluster: components advertise the node's **default-route IP = the firewalled mgmt LAN**. The
> fabric subnet is direct-cabled & open; the mgmt LAN drops inter-node random ports, so any peer that
> dials a mgmt IP hangs forever in **SYN-SENT**. Diagnose with
> `ss -tn state syn-sent | grep <mgmt-subnet>` on the worker — a stuck mgmt connection is the tell.

Four distinct bugs, in the order they surface (fix all; the last is the real wall) — the file/script
names below are from a cross-node launcher's orchestration layer:

1. **[proven]** **Head dist-init IP = mgmt.** `scripts/ip_detect.sh` does `ip route get 8.8.8.8` → the
   mgmt IP, injected as `--dist-init-addr` → rendezvous times out (`DistStoreError: 1/2 clients joined`).
   Patch `ip_detect.sh` to `ip route get <worker-fabric-ip>` (the fabric peer) first.
2. **[proven]** **`NCCL_SOCKET_IFNAME` lists mgmt first** (`enP7s7,enp1s0f1np1,…`). Pin to `enp1s0f1np1`
   (+ GLOO/TP).
3. **[proven]** **Per-NIC GID: never force a global `NCCL_IB_GID_INDEX`.** The RoCE-v2 IPv4 GID is at
   **idx 3 on NIC0** (the head fabric IP) but **idx 5 on NIC1** (`roceP2p1s0f1`, the second-rail IP); a
   forced global `=3` gives NIC1 a *link-local* GID → its QP never connects → hang. Set
   `NCCL_IB_GID_INDEX=""` (empty → NCCL auto-detects per-NIC; verified working). Patch `ib_detect.sh` to
   emit empty, or override in recipe.
4. **[proven]** **`NODE_IP`/`HOST_IP` = mgmt — THE wall.** `infiniband.py` sets `NODE_IP=DETECTED_MGMT_IP`,
   and SGLang's `get_ip()` (network.py) falls back to the 8.8.8.8 route-trick → mgmt. This IP is advertised
   in the cross-node **ShmBroadcast/GroupCoordinator rendezvous**, so the peer dials the mgmt IP and hangs.
   **Fix:** patch `infiniband.py` to set `NODE_IP`/`HOST_IP`/`SGLANG_HOST_IP` to the fabric IP
   (`DETECTED_IB_IPS.split(",")[0]`). (`SGLANG_LOCAL_IP_NIC=enp1s0f1np1` also works but needs the
   `netifaces` pip pkg, absent in the image — use HOST_IP.) Belt-and-suspenders: prepend the recipe
   `command` with `echo "<head-ip> <head-hostname> …" >> /etc/hosts` (avahi resolves `.local` → mgmt).

**[proven]** With all four, cross-node NCCL comm-init **completes** and the model loads. **Revert these
patches on launcher upgrade** — re-apply from this list. Transport note: NCCL auto-picks **Socket** over IB
on this fabric (no GPUDirect, so both host-stage anyway); forcing either is unnecessary once the IPs are
fabric.

**[proven]** **Same wall bites vLLM cross-node mp (not just SGLang).** Confirmed 2026-06-30 on a
DeepSeek-V4-Flash docker-compose bring-up: vLLM logs `mq_connect_ip=<mgmt-ip>` and the worker hangs in
SYN-SENT to it — NCCL is fine (fabric), but vLLM's **EngineCore↔Worker message queue** uses `get_ip()` →
default-route mgmt. Fix: set **`VLLM_HOST_IP=<this node's fabric IP>`** (and `HOST_IP`) **per node** —
head=`<head-ip>`, worker=`<worker-ip>` → `mq_connect_ip` becomes the fabric IP, no hang. (mimo's vllm-ray
avoids this via Ray's own addressing; raw `mp` cross-node needs the env.) This is the generic lesson: **on
a dual-homed cluster, force every engine's self-IP to the fabric.**

**[proven]** **But serving still fails — the forward deadlocks (the real M3 ceiling).** Init works, weights
load (NVFP4), Uvicorn comes up, `/v1/models` answers — but the **first forward pass deadlocks**: TP=2 the
worker GPU spins 96% on the per-layer all-reduce while the head sits 0% then both go idle; PP=2 both GPUs
stay flat 0%. cudagraph **capture** also hangs (bs=16, 0% GPU). Same host-staged-collective wall that
blocks vLLM — **SGLang does not escape it on our 2-node RoCE cluster**; the forum's cudagraph/tok-s numbers
don't reproduce here. See `[[wiki/models/minimax.md]]` + `[[wiki/cudagraphs-and-compile.md]]`.

**[proven]** Useful nugget: **PP=2 splits KV across nodes** → `max_total_num_tokens=127k` (vs TP=2's
3106), and needs `mem-fraction-static≈0.90` (REAP25 loads ~90 GB/node — near BF16 size, *not* the forum's
43.5 GB).

## Launch tooling

- **[proven]** A cross-node launcher (a uv tool with a recipe schema) is the production path: it
  ssh-drives the worker from the head, sets `NCCL_IB_HCA` to both twins + autodiscovers interfaces, and
  wraps `docker run` — e.g. `run <recipe> --hosts <head-ip>,<worker-ip> --ensure --no-follow` (example
  fabric IPs — substitute yours). The model-swapper's TP=2 units wrap exactly this.
- **llama.cpp RPC** is a different multi-node model (pipeline, not TP) — see `[[wiki/llama-cpp-rpc.md]]`.

## The cost (why cross-node is slow)

**[proven]** Per token a TP=2 MoE does ~120 cross-node all-reduces over **host-bounced** NCCL (no
GPUDirect, ~2.8 GB/s effective for the collective pattern), and is **eager-forced** (cudagraph wall).
That's structural: MiniMax-M3 ~5 tok/s, Nemotron-120B Q8 over llama.cpp RPC ~10.5 tok/s. If a model fits
on one node, **serve it single-node** — cross-node is for models that don't fit, not for going faster.

## Networking ops gotchas (DGX Spark)

- **[proven]** **avahi/mDNS conflict storms:** the DGX avahi conf uses `deny-interfaces=docker0,br-*,veth*`
  but **avahi has no wildcard support** → it binds every docker veth/bridge, hears its own announcements,
  and the device name climbs suffixes (`…-2 … -528`) until `<host>.local` stops resolving. Fix:
  `allow-interfaces=enP7s7`, `use-ipv6=no`, `publish-aaaa-on-ipv4=no`, restart avahi both nodes. The Spark
  avahi conf is **OTA-overwritable** (`nvidia-spark-avahi-conf` package) — keep a backup.
- **[proven]** **Stock Spark sshd is key-only** (`PasswordAuthentication no`). Tools needing a first-time
  password login (e.g. NVIDIA Sync) need a LAN-scoped `Match Address … PasswordAuthentication yes` block.
- **[conjecture]** **NVIDIA Sync / Cluster Assistant fails the "Software version" check on non-English
  locales** (S-forum-sync-locale): during the "Verifying Devices" step of pairing two DGX Sparks, the
  Cluster Assistant reports a false *"System Software Update Required — update to April 2026 or later"*
  even when the node is fully up to date (e.g. OTA2607 / July 2026, `nvidia-spark-ota-check` = 100%
  match). **Root cause:** NVIDIA Sync checks the software level by running
  `apt-cache policy dgx-spark-ota-update-meta` over SSH and parsing the human-readable output for an
  `Installed:` line. On a non-English locale (e.g. `LANG=de_DE.utf8`), `apt` localizes the label to
  `Installiert:` (French: `Installé :`, etc.) → the parser finds no version → generic "update required"
  error. The error message is misleading: the actual failure is "could not determine version," not
  "version too old." **Workaround** (keeps all other locale settings, no reboot):
  `sudo update-locale LC_MESSAGES=en_US.utf8` — new SSH sessions pick it up, then retry in the
  Cluster Assistant. **Suggested upstream fix:** prefix `LC_ALL=C` to the `apt-cache` invocation, or
  use machine-readable output: `LC_ALL=C apt-cache policy …` / `dpkg-query -W -f='${Version}' …`.
  A hotfix is reportedly releasing soon (NVIDIA aniculescu). Environment: 2× DGX Spark, DGX OS 7.5.0,
  OTA2607, kernel 6.17.0-1026-nvidia, NVIDIA Sync on macOS, CX-7 dual 200 GbE direct connection.
  Single source → [conjecture]. **Why it bites on Spark:** blocks cluster pairing (the prerequisite
  for all multi-node TP work on this wiki) on any non-English OEM image — and OEM Sparks ship in many
  locales. Related to the sshd password-login gotcha above (both are NVIDIA Sync pairing friction).
- **[proven]** **Don't advertise AAAA/IPv6** if UFW only allows the IPv4 LAN — IPv6 SYNs to :22 get dropped
  and SSH dead-ends.

### Batch 15 forum ingest (2026-07-15)

- **[conjecture]** **Interconnect is the bottleneck for large MoE, not memory**
  (S-forum-qwen397-arch, raphael.amorim): cross-node bandwidth is ~23 GB/s vs ~600 GB/s
  in-box — a ~26× gap. MoE all-to-all communication is very sensitive to this asymmetry,
  making large MoE models (e.g. Qwen3.5-397B-A17B) communication-bound on multi-Spark
  clusters. This corroborates the proven finding that cross-node collectives are
  host-staged and decode-bandwidth-limited (see "The cost" section above). The ~23 GB/s
  figure is consistent with the proven ~2.8 GB/s effective collective throughput
  (different measurement contexts — raw link vs collective pattern).
- **[conjecture]** **MoE gains flatten past TP=4 on GB10 clusters** (S-forum-qwen397-arch,
  raphael.amorim): the largest cluster reported in the forums (8× GB10) runs
  Qwen3.5-397B-FP8 inference at 31–35 tok/s, and MoE scaling gains flatten past TP=4.
  Expert parallelism helps up to 4 nodes, but beyond that the all-to-all overhead
  dominates. Relevant for planning >4-node clusters.
- **[conjecture]** **FP8 training does not exist on sm_121** (S-forum-qwen397-arch,
  raphael.amorim): TransformerEngine has no FP8 backend for sm_121, and NVIDIA has
  confirmed no roadmap for it. This is consistent with the existing `[proven]` finding
  that GB10 has no native FP4/block-scale-FP8 compute. Training on GB10 is limited to
  BF16/FP16. See also `[[wiki/quantization-on-gb10.md]]`.
- **[conjecture]** **Megatron-LM works on GB10 with caveats** (S-forum-qwen397-arch,
  raphael.amorim): Megatron/NeMo does work on GB10 for MoE at scale (expert parallelism),
  but: (1) Megatron Bridge consumes excessive VRAM on GB10 (use `vlm_step`, reduce
  `num_workers`); (2) FSDP/DeepSpeed won't survive the weight-gather traffic on 200GbE
  for 200B+ models; (3) training on 3+ nodes without a switch requires NCCL subnet-aware
  env vars (mandatory past 2 nodes). The "low MFU" report on GB10 Megatron was a config
  error, not an sm_121 limitation.

### Batch 18 forum ingest (2026-07-17)

- **[conjecture]** **`NCCL_BUFFSIZE=16777216` (16 MB) improves long-context decode allreduce on TP8**
  (S-forum-glm52-8x, ciprianveg): the default 8 MB NCCL buffer starts bottlenecking the allreduce
  on long-context decode at TP8 scale. Raising to 16 MB adds ~10%+ gen speed at high context on top
  of other DCP1 tweaks. This is a GB10-specific collective-tuning finding at 8× scale — consistent
  with the existing `[reported]` NCCL 2.30.4 mandatory finding (S-forum-ds4f-4x-vllm, S-forum-tokenspeed)
  and `[conjecture]` NCCL_CUMEM_ENABLE=0 / NCCL_MAX_NCHANNELS=2 / NCCL_BUFFSIZE=8388608 (MiMo TP=2
  recipe, S-forum-mimo-2x-opt). At TP8, the larger buffer pays off more; at TP2, 8 MB is fine.
- **[conjecture]** **TP4+PP2 raises prefill (~1,800 vs ~1,200 t/s) but wrecks MTP (acceptance → ~8%)**
  (S-forum-glm52-8x, ciprianveg): GLM-5.2 Int4-Int8 on 8× GB10 — the pipeline split collapses MTP
  acceptance to ~8%, dragging decode to ~12 t/s (vs 33–54 at TP8+PP1). Production stays on TP8+PP1.
  Corroborates the existing `[conjecture]` PP-over-ethernet is too latency-sensitive finding
  (S-forum-2d-parallel) — here the latency sensitivity hits MTP draft acceptance specifically, not
  just throughput.
- **[conjecture]** **DCP4 on TP8 causes decode starvation during concurrent prefill** (S-forum-glm52-8x,
  penguinchang): with DCP4 (distributed KV cache, 4 ranks) on TP8 GLM-5.2, concurrent long-prefill
  requests starve ongoing decode to ~0.0–0.2 tok/s until prefill completes — distributed KV cache
  prefill saturates the interconnect. A custom "decode-aware prefill" scheduler patch
  (ENABLE_DECODE_AWARE_PREFILL=1, DECODE_PREFILL_TOKEN_BUDGET=1024, IDLE_PREFILL_TOKEN_BUDGET=16384,
  MAX_LONG_PREFILLS_PER_STEP=1) caps decode stall to ~1.6 s and keeps all 4 prefill requests
  completing, but decode still drops to ~2.74 tok/s under pressure. Validated on 8× DGX Spark
  (1 continuous decode + 4 × ~8K prefills: idle PP 831.6 tok/s, pressure PP 735.8, pressure decode
  2.74, max stall 1.64 s). DCP1 avoids this entirely (~30% faster prefill, ~60% faster gen per
  the OP), but DCP4 enables 320K×10 context (3.2M KV tokens). This is a GB10-specific distributed-KV
  scheduling issue — the host-staged cross-node collectives amplify the prefill/decode contention.
- **[conjecture]** **`draft_tensor_parallel_size=1` avoids TP8 collectives on every MTP draft step**
  (S-forum-glm52-8x, ciprianveg): keeping the MTP drafter unsharded (draft_tp=1) is a ~10% gen-speed
  lever at TP8 — paying TP8 collectives on every draft step would dominate the spec-decode overhead.
  Consistent with the MTP-needs-cudagraphs / cross-node-amortization findings on
  `[[wiki/cudagraphs-and-compile.md]]`.

### Batch 19 forum ingest (2026-07-17)

- **[conjecture]** **ASUS GX10 PD firmware capsule (v0103, PD/0x507) reportedly 4× faster
  inter-Spark link** (S-forum-asus-fw0103, brian322): after manually applying
  `capsule_update.sh usbpd_5.7.cap` (the PD capsule failed via GUI on both machines),
  the inter-Spark connection speed was reported as 4× faster and MiniMax M2.5 tok/s
  improved to 25-30 range. The USB-C PD firmware may influence CX-7 power delivery or
  PCIe slot power advertisement — potentially related to the existing `[conjecture]`
  CX-7 `SlotPowerLimit 0W` throttle finding (S-forum-cx7-13gbps). Single source for the
  4× claim; thermal improvement corroborated by trithemius (see platform-gb10). If
  confirmed, this would be a significant firmware-level fix for CX-7 throughput on the
  ASUS GX10 variant. Status: `open` — needs hardware verification.
- **[conjecture]** **Total host freeze during heavy TP=2 prefill — thermal shutdown, not
  software** (S-forum-host-freeze-tp2, heathen0711): serving Step-3.7-Flash-NVFP4 via
  spark-vllm-docker (TP=2, Ray) on 2× Spark, heavy non-cached prefill caused total host
  death (no ping/SSH/display) with zero forensic trace. The "zero trace" pattern is
  GB10-specific: kdump, hung_task_panic, softlockup_panic, netconsole, and NCCL Flight
  Recorder all captured nothing — suggesting a hardware/firmware-level lockup below the
  OS's ability to log. Diagnosed as thermal shutdown via the NVIDIA field diagnostic
  (failed → RMA). This is relevant to multi-node because heavy prefill (long/resumed-chat
  prompts with low cache hit rate) maximally stresses both the GPU and the CPU-side
  host-staged NCCL collectives simultaneously — the highest combined SoC power draw
  scenario. See `[[wiki/platform-gb10.md]]` → thermal shutdown for the full finding.

### Batch 24 forum ingest (2026-07-20)

- **[conjecture]** **6× GB10 cluster via MikroTik CRS812 — b12x backend enables non-power-of-2
  TP=6** (S-forum-6x-cluster, mclenithan): a 6-node DGX Spark cluster (768 GB combined unified
  memory) networked via a **MikroTik CRS812** switch (2× 200G + 1× 400G port, with breakout on
  the 400G port) — the same CRS812 option documented for 4-node in S-forum-mikrotik, here pushed
  to 6 nodes. The **b12x backend** (lukealonso/b12x, the same unified SM120 sparse-MLA + PCIe
  DCP collectives stack used in the 8× GLM-5.2 run, S-forum-glm52-8x) reportedly enables
  **TP=6 on most models** — notably, vLLM's stock distributed executor assumes powers-of-2
  node counts (2/4/8) for tensor parallel; b12x appears to relax this constraint on GB10.
  **GLM-5.2 runs at ~30 tok/s single-stream** on this 6-node cluster (consistent with the
  33–54 tok/s range reported on 8× GB10 at TP=8, S-forum-glm52-8x — fewer nodes, slightly lower
  throughput, same order of magnitude). Cluster peak power draw: **800–1180 W** (~133–197 W/node,
  consistent with the 140–240 W per-node envelope documented in S-forum-power-spec and
  S-forum-driver610). Replies ask about virtual-head padding (per S-forum-mimo-3x technique)
  and whether all 6 nodes actively compute vs some only hold weights — unanswered in the thread.
  Single source → `[conjecture]`. **Why it bites on Spark:** non-power-of-2 TP has been an open
  question on GB10 (3-node required virtual-head padding, S-forum-3node-nccl); if b12x genuinely
  enables arbitrary TP, it changes cluster sizing economics (6× CRS812 vs 8× needing CRS804).
  Status: `open` — no YAML/docker shared, no ib_write_bw or NCCL verification, power claim
  unverified. See also `[[wiki/benchmarks.md]]` for the GLM-5.2 6× tok/s data point.

### Batch 25 forum ingest (2026-07-20)

- **[conjecture]** **Switch-less 5-node full mesh via MST sub-port splitting — break 4×50G → 2×50G
  per QSFP port** (S-forum-kimi-k3-ceiling, mashie): to build a 5-node cluster without a switch, the
  trick is to split each physical QSFP port that currently runs 4×50G into **sub-ports of 2×50G
  using MST (Multi-Stream Transport)**. After splitting one port, the node has **6 RoCE interfaces
  instead of the usual 4** (which is what a 4-node mesh needs); for 5 nodes, split the other port
  too. This trades bandwidth (half per sub-port) for optimal latency and no switch. Cabling: for
  4 nodes → 2 regular QSFP56 DAC cables + 4× ~$150 transceivers + ~$200 optical splitter cables
  ≈ **~$800 optical**. Going 4→5 nodes adds another 6 transceivers + ~$300 cabling. A MikroTik
  switch + DAC cables is cheaper than a transceiver-based full mesh; a **DAC-based full mesh would be
  cheaper than a MikroTik for 5 nodes**. The OP is driven by latency (no noisy switch in the office).
  Commercial 3-headed DACs may make this much cheaper if available. **Why it bites on Spark:** the
  CX-7 dual-port constraint (2 QSFP ports/node) limits a direct-cable mesh to ~4 nodes without
  sub-port tricks; MST splitting is the first reported technique to push switch-less mesh to 5 nodes
  on GB10. Single source, unverified (OP is "currently working on" it) → [conjecture]. Related to
  the existing 4-node full-mesh finding (S-forum-4node-mesh) and CRS812/CRS804 switch options
  (S-forum-mikrotik).
- **[conjecture]** **Practical GB10 cluster ceiling ≈ 4 nodes for Opus-class; 16 nodes for 2–3T
  frontier models** (S-forum-kimi-k3-ceiling, CosmicRaisins): cluster-sizing math for the new
  multi-trillion-parameter model class (Kimi K3 2.8T, Minimax M3 Pro ~2.5T): at **~115 GB usable
  per node**, a 2.8T model at 4-bit needs **~16 nodes** (~$100k, 2000–3200W under inference). With
  ~50B active params, a 16-node cluster "should" push ~35–50 tok/s decode — but that setup is "out
  of prosumer reach." The OP's thesis: the **practical ceiling of a sane GB10 cluster is ~4 nodes**
  (Opus-class), not the frontier — unless going to 2-bit/1-bit quants. A list of viable 200B-class
  alternatives that fit 4× Spark was surfaced: **Step-3.7-Flash, Command-A-Plus, Inkling-Small,
  Laguna-M.1, Qwen3.5-397B-A17B, Hy3**. This is opinion + math, not a measured result — tagged
  [conjecture]. The durable part is the sizing math (~115 GB usable/node × 4-bit → 16 nodes for
  2.8T) and the viable-model-class list. The "models will keep getting smaller" vs "frontier is
  growing" debate is out of scope for this KB.
- **[conjecture]** **Inkling's Lamport collectives require MNNVL (NVLink fabric) — hard-error on
  RoCE; escape hatch `LAMPORT_RS_SCONV=0`** (S-forum-inkling-nvfp4, greg190): Thinking Machines'
  Inkling uses Lamport-style collectives that require an **NVLink fabric (MNNVL)** and hard-error on
  RoCE clusters — which is all GB10 has (ConnectX-7 RoCE, no NVLink between nodes, no GPUDirect —
  see `[[wiki/platform-gb10.md]]`). TML shipped the escape hatch env var `LAMPORT_RS_SCONV=0` to
  bypass this. **Why it bites on Spark:** this is a new class of "designed-for-datacenter-NVLink-
  fabric" model that hard-errors on GB10's RoCE-only interconnect. Any future model adopting
  Lamport/MNNVL collectives will need this flag (or an equivalent) on Spark. The 8× Spark Inkling
  bring-up used this to get the cluster running. See `[[wiki/models/inkling.md]]`.

### Batch 26 forum ingest (2026-07-21)

- **[conjecture]** **3-node full-mesh networking guide for spark-vllm-docker and sparkrun**
  (S-forum-3node-mesh, eugr + dbsci): a 3-node DGX Spark cluster can be built as a **full mesh
  without a switch** — each Spark has 2 QSFP ports, and 3 cables connect them in a triangle
  (Spark1.Port0→Spark2.Port1, Spark2.Port0→Spark3.Port1, Spark3.Port0→Spark1.Port0). **Must
  cross-connect port0↔port1** (not port0↔port0) or the mesh may not work properly. Each port has
  two logical partitions (4 RoCE interfaces per machine); each link can burst up to full 200 Gbps
  when the other is underutilized. This enables **pipeline-parallel** inference workloads across 3
  nodes. The 3-node mesh NCCL functionality has been **merged into NCCL main** (no special branch
  needed anymore — earlier builds required a `dgxspark-3node-ring` branch or TF5 container).
  Community guides: spark-vllm-docker + sparkrun both support 3-node mesh config. Single source
  (authoritative community contributor) → [conjecture].
- **[conjecture]** **TP requires power-of-2 node counts (attention head divisibility)**
  (S-forum-3node-mesh, eugr): tensor parallelism requires the model's attention head count to be
  divisible by TP value. Since models typically have 64 or 128 heads, TP must be 2/4/8/etc.
  **Uneven tensor splits (e.g. TP=3) may be possible but no implementations exist yet.** With 3
  Sparks, you are limited to **pipeline parallel or data parallel**, or TP=2 on 2 nodes + a 3rd
  node for embedding/reranking/small-fast model. This corroborates the existing `[proven]` finding
  that powers-of-2 node counts are recommended for vLLM.
- **[conjecture]** **3-node pipeline-parallel is slower than 2-node TP=2, roughly equivalent to
  single-node speed** (S-forum-3node-mesh, eugr): TP is only supported on 2/4/8/2^n nodes — using
  all 3 nodes in pipeline-parallel adds overhead making it "roughly equivalent to a single Spark."
  If the model fits on one Spark, 3 nodes can run data-parallel to increase concurrent request
  capacity. You can still run 2-node workflows on a 3-node cluster at full 2-node speed.
- **[conjecture]** **LMCache for dedicated KV-cache-only node on 3× Spark** (S-forum-3node-mesh,
  Phaserblast, eugr): with 3 Sparks, one possible architecture is 2 nodes running TP=2 for the
  model + a 3rd node serving as a dedicated KV cache host via **LMCache** (`vllm-project/vllm`).
  Not yet tested in practice — eugr notes it "should be possible" but performance impact is
  unknown. Relevant for long-context workloads where KV cache exceeds available memory after model
  weights (e.g. Qwen3.5-397B-A17B on 2× Spark leaves few GB for KV).
- **[conjecture]** **vLLM pipeline-parallel + MTP not supported** (S-forum-3node-mesh,
  jameslacroix): `--speculative-config` with MTP (`num_speculative_tokens: 2`) throws
  `NotImplementedError: Pipeline parallelism is not supported on this vLLM version`. MTP is
  incompatible with PP — relevant for 3-node PP deployments. If MTP is needed, use TP (2/4/8 nodes)
  instead.
- **[conjecture]** **fastsafetensors loader freezes at 29% on Xet-downloaded models — use
  instanttensor** (S-forum-3node-mesh, jameslacroix): on 3-node PP deployments, the fastsafetensors
  loader hangs at 29% when loading Xet-downloaded model files. Workaround: switch to the
  instanttensor loader. This corroborates the existing `[proven]` finding about HF Xet hangs.
- **[conjecture]** **`gpu_memory_utilization: 0.85` causes silent worker death on 3-node PP — use
  0.8** (S-forum-3node-mesh, jameslacroix): at 0.85 util, workers silently die during inference
  (SIGTERM, exit code 1) right after NCCL P2P communicator creation between pipeline stages. Gloo
  metadata transport shows "Connection closed by peer" as a symptom. Reducing to 0.8 is stable.
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` also helps prevent memory fragmentation
  crashes. Relevant for multi-node PP where per-node memory headroom is tighter.
- **[conjecture]** **Qwen3.5-397B-A17B-int4-AutoRound on 3-node PP benchmarks** (S-forum-3node-mesh,
  chunkai721): llama-benchy v0.3.5 results on 3-node PP cluster:
  - `tg32` (token gen, 32 tokens): ~12–14.4 tok/s across context depths 0–32768 (peak 15.3 tok/s)
  - `pp2048` (prompt processing, 2048 tokens): ~1070–1242 tok/s, scaling with context depth
    (912 tok/s @ depth 0, 1242 tok/s @ depth 8192, 1070 tok/s @ depth 32768)
  - TTFT ranges 2.2–7.6 s depending on depth/batch
  - These numbers confirm 3-node PP decode is ~single-node speed (consistent with eugr's claim
    above) and prefill is competitive. Single source → [conjecture].

### Batch 29 forum ingest (2026-07-22)

- **[conjecture]** **RoCE RC queue pairs require L2 adjacency — routed (L3) RDMA fails on
  non-adjacent node pairs in a switchless ring** (S-forum-6x-ring-rdma, alpaslan.erdag):
  in a 6-node DGX Spark ring topology (each node has 2 ConnectX-7 cards = 4 RoCE ports,
  connected to exactly 2 physical neighbors via dual-rail 200G links, L3 /31 per link +
  OSPF/FRR routing), `ib_write_bw` works perfectly between direct neighbors but **fails
  for any non-adjacent pair** requiring IP-routed (multi-hop) RDMA. The failure is at the
  raw verbs layer, not NCCL-specific: `ibv_modify_qp ... Connection timed out, curr state
  INIT, next state RTR` — the RC queue pair handshake cannot traverse L3 hops. Root cause:
  **RoCE's RC QP setup requires L2 adjacency** (real or bridged); IP forwarding re-routes
  the packet rather than transparently bridging the Ethernet frame. This explains why
  officially-documented topologies stop at 3-node full-mesh (every node is a direct L2
  neighbor) and require a switch beyond that. Mashie confirms: "RoCE needs directly
  connected interfaces from a layer 2 point of view. If you do bridging through the
  intermediate nodes it should work." Single source for the isolation; consistent with
  RoCEv2 protocol semantics. **Why it bites on Spark:** the CX-7 dual-port constraint
  (2 QSPP ports/node) limits direct-cable L2 mesh to ~3–4 nodes; ring topologies that
  try to scale beyond this with L3 routing hit this verbs-layer wall at every non-adjacent
  pair. See also the existing 4-node full-mesh (S-forum-4node-mesh) and 5-node MST
  sub-port splitting (S-forum-kimi-k3-ceiling) techniques that stay at L2.

- **[conjecture]** **NCCL_IB_MERGE_NICS=0 + NCCL_IB_SUBNET_AWARE_ROUTING=1 (patched NCCL)
  together fix 6-node ring RDMA** (S-forum-6x-ring-rdma, alpaslan.erdag): two env vars,
  **both required together**, enable correct NCCL RDMA across a 6-node switchless ring:
  - `NCCL_IB_MERGE_NICS=0` — stops NCCL from bonding 2 physical ports per CX-7 card into
    one virtual 400Gbps device (confirmed in logs: "Skipping makeVDevice"). With merge
    left ON, the merged virtual device tries to open QPs on both underlying physical ports
    for a single logical peer; since each port goes to a *different* ring neighbor (dual-rail
    to different peers, not both rails to the same peer), the second QP always times out.
  - `NCCL_IB_SUBNET_AWARE_ROUTING=1` — (requires a patched NCCL) selects the correct
    physical port per peer via GID/subnet lookup. Without this, NCCL's round-robin
    channel→device assignment (channel0→dev0, channel1→dev1, …) picks ports cabled to
    a different neighbor entirely — silently routing channels onto hardware not wired
    to the peer. Neither `NCCL_ALGO=Ring`, `NCCL_SKIP_TREE_CONNECT=1`, nor
    `NCCL_IB_MERGE_NICS=0` alone is topology-aware at the NIC-selection level.
  With both set: `ncclCommInitRank` completes cleanly, "Connected all rings", no
  `ibv_modify_qp` timeouts, full 6-node PP=6 pipeline loads and serves. Single source →
  [conjecture]. The `NCCL_IB_SUBNET_AWARE_ROUTING` flag requires a patched NCCL (not stock
  2.28.9) — may be in newer NCCL main. Related to the existing [conjecture]
  NCCL_IB_MERGE_NICS finding from the 3-node mesh (S-forum-3node-mesh, Hunlx's env
  recipe uses both flags).

- **[conjecture]** **NCCL channel→HCA round-robin assignment is not topology-aware —
  fails in switchless multi-port ring** (S-forum-6x-ring-rdma, alpaslan.erdag): each DGX
  Spark has 4 physical RoCE ports but only 2 physical neighbors in a ring. NCCL assigns
  its 16 logical channels to HCAs by round-robin (channel0→dev0, channel1→dev1, …,
  channel4→dev0, …), assuming any port can reach any peer (true in a switched fabric, false
  in a switchless ring). In a ring, 2 of 4 ports are cabled to a *different* neighbor —
  so a subset of channels get silently routed onto hardware not wired to the peer.
  `NCCL_ALGO=Ring`, `NCCL_SKIP_TREE_CONNECT=1`, and `NCCL_IB_MERGE_NICS=0` do not change
  this assignment. The only fix is `NCCL_IB_SUBNET_AWARE_ROUTING=1` (patched NCCL, see
  above). **Why it bites on Spark:** this is a switchless-topology-specific failure mode
  — switched fabrics don't hit it because all ports can reach all peers. Relevant for
  any >3-node switchless GB10 deployment using all 4 RoCE ports.

- **[conjecture]** **GID table asymmetry between two ConnectX-7 cards — disable IPv6 on
  ring interfaces + force consistent NCCL_IB_GID_INDEX** (S-forum-6x-ring-rdma,
  alpaslan.erdag): the two CX-7 cards on a Spark have asymmetric GID tables — extra
  privacy-extension IPv6 addresses shift the IPv4-mapped RoCEv2 GID index between cards.
  Fix: disable IPv6 on all ring interfaces and force a consistent `NCCL_IB_GID_INDEX`
  across all nodes. This corroborates the existing [proven] per-NIC GID finding
  (different GID indices on NIC0 vs NIC1) from the 2-node bring-up, generalized to
  the multi-card ring case. Hunlx's 3-node recipe uses `NCCL_IB_GID_INDEX=3` with a
  per-node `NCCL_IB_HCA` mapping (see S-forum-3node-mesh Batch 26).

- **[conjecture]** **NCCL_IB_DISABLE=1 (TCP/Socket fallback) is a stable workaround for
  6-node PP — ~326 tok/s aggregate, ~7% slower than RDMA** (S-forum-6x-ring-rdma,
  alpaslan.erdag): when RDMA setup proved too complex (before the MERGE_NICS=0 +
  SUBNET_AWARE_ROUTING fix), `NCCL_IB_DISABLE=1` forces NCCL onto TCP/Socket transport.
  Combined with a stable per-node identity address (dummy0 interface, to work around
  Gloo/NCCL special-casing of `lo`), Ray + vLLM init cleanly across all 6 nodes and
  PP=6 inference works end-to-end. Qwen3.6-35B-A3B-NVFP4 on 6-node PP=6: **~21 tok/s
  per request** (20 concurrent, 326 tok/s aggregate TCP, 349 tok/s aggregate RDMA).
  The ~7% gain from RDMA vs TCP is surprisingly small — see the GPUDirect finding below.
  Single source for the throughput numbers → [conjecture]. See also
  [[wiki/benchmarks.md]] for the data point.

- **[conjecture]** **GPUDirect RDMA unavailable on GB10 — `nvidia-peermem` module
  refuses to insert with "Invalid argument"** (S-forum-6x-ring-rdma, alpaslan.erdag):
  NCCL logs `GPU Direct RDMA Disabled for HCA 0/1/2/3` for every RoCE interface on every
  run, regardless of settings. Attempting `sudo modprobe nvidia-peermem` fails with
  `modprobe: ERROR: could not insert 'nvidia_peermem': Invalid argument` — zero dmesg
  output, `ib_core` loaded, `nvidia-peermem.ko` matches the running kernel's vermagic
  (6.17.0-1021-nvidia) exactly, so it's not a version mismatch. The module just refuses
  to insert with no diagnostic. NCCL also loads a GIN plugin: `GIN/Plugin: Assigned
  plugin GIN_IB_GDAKI type 3` — suggesting **DOCA GPUNetIO / GDAKI** (GPU-initiated async,
  leveraging NVLink-C2C coherent CPU-GPU memory) may be the intended GPU-NIC data path
  on Grace-Blackwell, not the classical PCIe P2P `nvidia_peermem` path. This corroborates
  the existing [proven] "No GPUDirect RDMA" finding on the platform page — the new bit is
  the specific `modprobe` failure mode and the GDAKI/GPUNetIO hypothesis for the
  *intended* path. **Why the RDMA vs TCP gain is only ~7%:** without GPUDirect, both
  RDMA and TCP transport are host-staged (GPU↔CPU↔NIC, not GPU↔NIC directly) — the CPU
  bounce is the bottleneck either way, so switching from TCP to RDMA saves only the
  TCP protocol overhead, not the host-staging cost. This is the first quantified
  RDMA-vs-TCP comparison on GB10 and directly explains the proven "cross-node is slow"
  finding. Single source → [conjecture]. See [[wiki/platform-gb10.md]] → No GPUDirect RDMA.

- **[conjecture]** **Hunlx's 3-node switchless env recipe — per-node NCCL_IB_HCA mapping
  + subnet-aware routing** (S-forum-6x-ring-rdma, Hunlx): a working 3-node switchless
  mesh env config from the same thread. Key elements: `NCCL_IB_MERGE_NICS=0`,
  `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_IB_GID_INDEX=3` (forces RoCEv2 IPv4),
  `NCCL_P2P=1` (prevents cross-node P2P ring init deadlocks), `NCCL_CROSS_NIC=1`,
  `NCCL_IB_RETRY_CNT=7`, `NCCL_IB_TIMEOUT=22`, `VLLM_SKIP_CUSTOM_ALLREDUCE=1`, plus a
  per-node `NCCL_IB_HCA` mapping via a `case "$VLLM_HOST_IP"` switch (inverted interface
  order on one node to match physical cabling). Uses `UCX_NET_DEVICES` and
  `GLOO_SOCKET_IFNAME` on the management interface. This corroborates the existing
  [conjecture] 3-node mesh findings (S-forum-3node-mesh) and adds the specific env var
  recipe. Single source → [conjecture].

### Batch 32 forum ingest (2026-07-24)

- **[conjecture]** **Baked `LD_PRELOAD` NCCL shim beats `LD_LIBRARY_PATH` and symlinks**
  (S-forum-m3-tp3, tonyd615): a vLLM container had a baked `LD_PRELOAD` pointing at an older NCCL
  "local-inference" 2.30.4 shim. The baked `LD_PRELOAD` silently overrode both a symlink swap and
  an `LD_LIBRARY_PATH` prepend — the NCCL banner read 2.30.4 even with 2.30u1 installed. The old
  shim lacked the subnet-aware override needed for 3-node switchless mesh, causing
  `ibv_modify_qp err 110`. **Fix:** force `LD_PRELOAD` to the 2.30u1 lib and unset the shim env
  vars. **Why it bites on Spark:** community Docker images may bake NCCL shims that silently
  override user-installed NCCL builds — always check `LD_PRELOAD` in the container env, not just
  `LD_LIBRARY_PATH`. Related to the existing [proven] NCCL 2.30.4 mandatory finding
  (S-forum-ds4f-4x-vllm) and [conjecture] vLLM 0.25.1/NCCL 2.30.7 image lag (S-forum-vllm025-nccl).

- **[conjecture]** **Cold power-drain fixes stuck `ib_write_bw` on healthy CX-7 hardware**
  (S-forum-m3-tp3, tonyd615, mashie): `ib_write_bw` stuck at ~12.8 Gb/s on healthy Gen5 x4 / 200G
  hardware (did not scale with queue pairs: q=4 and q=16 both landed at 12.8). A **full cold
  power-drain** (power off, unplug bricks ~90s, power back on) cleared it to **111.85 Gb/s** —
  matching eugr's documented number. A warm reboot did **not** fix it. This is the same class of
  cold-power-cycle fix documented for the GPU clock wedge (S-forum-clock721, S-forum-clock-5min)
  and CX-7 `SlotPowerLimit 0W` throttle (S-forum-cx7-13gbps) — the CX-7 NIC or its PCIe link
  gets into a throttled state that only a full power-cycle clears. **Why it bites on Spark:**
  before debugging NCCL/RoCE config, verify raw `ib_write_bw` — if it's stuck at ~13 Gb/s, no
  amount of NCCL env tuning will help; power-cycle first.

- **[conjecture]** **TP=3 bandwidth fix increases concurrency, not single-stream tok/s**
  (S-forum-m3-tp3, tonyd615): after fixing the RoCE link from 12→100 Gb/s, single-stream tok/s
  did not change — **concurrency increased instead**. This is consistent with the proven finding
  that cross-node TP is latency-bound (host-staged all-reduces) for single-stream decode, not
  bandwidth-bound. More link bandwidth helps when multiple concurrent requests saturate the
  collective pipeline, not for a single request's ~120 sequential all-reduces/token.

- **[conjecture]** **Ray object store + memory monitor cause false OOM on unified memory**
  (S-forum-m3-tp3, tonyd615): two Ray-specific OOM traps on GB10 unified memory: (1) Ray reserves
  ~30% of RAM (~36 GB/node) for a plasma object store that vLLM TP never uses — on the head, this
  + 84 GB shard + KV overcommits the 121 GB box → `NVRM: Out of memory` during weight load. Fix:
  `--object-store-memory 1073741824` (cap to 1 GB, frees ~35 GB/node). (2) After warmup, the head
  sits at ~96% RAM (normal on UMA) — Ray's 95% memory monitor false-kills rank-0
  (`NODE_OUT_OF_MEMORY`) with no real OOM. Fix: `RAY_memory_monitor_refresh_ms=0`. **Why it bites
  on Spark:** Ray's memory assumptions (discrete CPU RAM vs GPU VRAM) don't hold on unified memory
  — high utilization is normal, not a leak. Related to the existing [proven] Ray V2 hangs finding
  and the [conjecture] UVM livelock finding (S-forum-uvm-livelock).

## See also
`[[wiki/platform-gb10.md]]` · `[[wiki/cudagraphs-and-compile.md]]` · `[[wiki/llama-cpp-rpc.md]]` · `[[wiki/engines.md]]`

## Forum ingest: ConnectX-7 PCIe power throttling (2026-07-08)

- **[conjecture]** **CX-7 link capped at ~13 Gbps** (expected ~92+ Gbps per interface, ~190 Gbps
  combined) — caused by `mlx5_core: Detected insufficient power on the PCIe slot (27W)` in dmesg
  (S-forum-cx7-13gbps). Root cause: `lspci` reports `SlotPowerLimit 0W` → the driver throttles,
  thinking there's insufficient power. Link negotiates at 200 Gbps / 32 GT/s correctly, but throughput
  is capped regardless of MTU, ring buffers, or qdisc settings. **[reported]** Multiple forum users
  hit this; the fix is not yet confirmed — one user reported resolving it after a firmware/driver
  update, but the `SlotPowerLimit 0W` reading suggests a BIOS/firmware PCIe slot-power advertisement
  bug. This is distinct from the kernel-6.17 RoCE regression (also 13–16 Gbps, but that's a kernel
  throughput cap, not a PCIe power event).
- **[conjecture]** **4-node topology** (S-forum-cx7-13gbps context): a 100G MikroTik CRS504 switch
  works for 4× GB10 clusters; daisy-chaining sustains 100G/pair (vs 200G direct). Powers-of-2 node
  counts recommended for vLLM; 3-node TP=3 requires virtual-head padding (see
  `[[wiki/models/mimo-v2.5.md]]`).

### Batch 2 forum ingest (2026-07-08)

- **[reported]** **MikroTik switch options for 4× Spark** (S-forum-mikrotik): CRS804-4DDQ (4×
  QSFP56-DD, 400G) is the ideal switch but frequently sold out. CRS812 (2×200G + 1×400G port) works
  with a breakout cable on the 400G port for the other 2 Sparks. CRS504 (100G only) also works but
  at reduced bandwidth. Community confirmed all three functional for TP=4.
- **[conjecture]** **2D parallelism (TP×PP) on 4× Spark** (S-forum-2d-parallel, eugr_nv): PP over
  RJ-45 ethernet (vs TP over CX-7) is "too latency-sensitive" — potential performance gains undone
  by ethernet latency. PP requires much less bandwidth but is latency-sensitive; stick to TP over
  the CX-7 fabric.
- **[conjecture]** **NCCL ALLREDUCE timeout during DDP training** (S-forum-ddp-timeout): rank desync
  during distributed training on 2× Spark (Ray cluster). `ProcessGroupNCCL` watchdog timeout after
  1800s — rank 1 finished collective #1155873 but didn't join #1155874. Training (not inference) is
  less battle-tested on this fabric than serving.
- **[reported]** **Mixing FE and Asus Ascent in a 2-node cluster** (S-forum-mix-skus): no issues
  reported — CX-7 firmware compatible across OEM variants. The atypical dual PCIe5 x4 link setup is
  identical across all GB10 SKUs.

### Batch 3 forum ingest (2026-07-09)

- **[reported]** **NCCL 2.30.4 is critical for 4× Spark vLLM** (S-forum-ds4f-4x-vllm): the CUDA 13.0
  base image ships NCCL 2.28.9, which **hard-wedges every long generation** on 4× Spark TP=4. Upgrading
  to `libnccl2=2.30.4-1+cuda13.2` fixes it. This was the single fix that unlocked 49–54 tok/s
  DeepSeek-V4-Flash on 4× Spark. **Corroborated** by S-forum-nemotron-super-mtp (also uses NCCL 2.30.4).
- **[reported]** **SGLang container RDMA passthrough** (S-forum-glm47-rdma): SGLang in Docker does NOT
  inherit `/dev/infiniband` automatically (unlike vLLM venv). Without `--device=/dev/infiniband
  --cap-add=IPC_LOCK --ulimit memlock=-1` + `NCCL_IB_HCA=rocep1s0f0 NCCL_IB_DISABLE=0`, SGLang silently
  falls back to socket transport — **2.5× slower** (GLM-4.7-FP8: 8.2 → 25.1 tok/s after RDMA enable).
  Verify: `grep -c "via NET/IB" <logs>` > 0 and `grep -c "via NET/Socket" <logs>` == 0.
  **[reported]** Same finding in S-forum-roce-397b-mtp: RoCE vs TCP socket on 4× Spark SGLang =
  NCCL bus bandwidth 2.12 → 9.78 GB/s (4.6×), tok/s 34.6 → 65.4 (1.88×). Three things required for
  RoCE with SR-IOV VFs in Kubernetes pods: (1) VF interfaces must have host-side IPv4 addresses
  configured (else RoCE GID table only has link-local entries → `ibv_modify_qp` fails), (2) pods must
  run privileged (host-device CNI moves the iface but NOT `/dev/infiniband/*`), (3) NetworkAttachmentDefinitions
  must exist for all VF indices in use.
- **[conjecture]** **SGLang multi-node traps** (S-forum-sglang-traps): three debugging traps on 4× Spark:
  (1) `TORCH_DISTRIBUTED_DEBUG=DETAIL` produces **false-positive** collective mismatch errors — SGLang's
  head process spawns sidecars that do local broadcasts, inflating PyTorch's global SequenceNumber
  tracker; use `NCCL_DEBUG=TRACE` instead for real cross-rank tracing. (2) EAGLE speculative decoding
  flags must be on **every node**, not just rank 0 — workers hang silently if flags are missing. (3) See
  the RDMA passthrough issue above.
- **[conjecture]** **CUTLASS MoE compile OOM on 4× Spark** (S-forum-m25-sglang-4x): `flashinfer_cutlass`
  MoE kernel JIT compilation exhausts host RAM — `nvcc` compiling CUTLASS grouped-GEMM templates in
  parallel eats ~5–8 GB per kernel, and default ninja parallelism on Spark's 20-core SoC fans out enough
  concurrent jobs to OOM the unified memory pool. Fix: `-e MAX_JOBS=1 -e NVCC_THREADS=1
  -e OMP_NUM_THREADS=4` — adds ~10–15 min to first launch (kernel cache empty), subsequent launches reuse
  cached kernels. Also drop `--mem-fraction-static` from 0.9 to 0.8 for compilation headroom.
- **[conjecture]** **4-node full mesh without a switch** (S-forum-4node-mesh): 200GBASE-SR4 transceivers
  + MPO-12 to LC-LC breakout cables + LC-LC duplex couplers can create a 4-node full mesh using only one
  QSFP port per node (ring of 100G links) + two regular QSFP56 DAC cables for the diagonals. ~5W/node
  added heat (less than CX-7 idle power savings). Untested in practice but theoretically sound for
  latency-sensitive TP workloads without a switch.
- **[reported]** **MTP on SGLang: `--speculative-algorithm NEXTN`** (S-forum-roce-397b-mtp,
  S-forum-gemma4-mtp-4x): SGLang's built-in NEXTN speculative decoding uses the model's own MTP head
  (no separate draft model needed). Qwen3.5-397B + MTP on 4× Spark: **40 tok/s @ n1** (+86% over
  baseline 21.5), 110.9 tok/s @ n8. Gemma-4-31B + MTP on 4× Spark: 26.68 tok/s @ n1 (+154%), 153 tok/s
  @ n8 (+80%). For Qwen3.5 with hybrid attention, also needs `--mamba-scheduler-strategy` settings.
  `--speculative-num-steps 3 --speculative-num-draft-tokens 4` is the winning config.

### Batch 6 forum ingest (2026-07-11)

- **[conjecture]** **3-node ring topology challenges** (S-forum-3node-nccl, nvidia4468): expanding
  from 2→3 Sparks fails at NCCL init, even with all standard prerequisites (passwordless sudo, SSH
  keys, firewall disabled, identical firmware/CUDA). sparkrun's "auto" NCCL setup detects **"3 nodes
  2 ports"** and wants to configure a **Switch topology instead of a Ring** — odd node counts don't
  cleanly map to the dual-port CX-7 ring model (port0→port1 per node). NVIDIA's build.nvidia.com has
  a dedicated "Connect Three DGX Spark in a Ring Topology" guide — follow it for 3-node setups
  rather than extrapolating from the 2-node recipe.
- **[conjecture]** **Cable mixing causes MTU mismatch** (S-forum-3node-nccl): mixing ASUS QSFP
  cables with the NVIDIA store's "$99 recommended" no-name cable (type unspecified — QSFP56 or
  QSFP112 unclear) produces inconsistent MTU negotiation — the no-name cable's port comes up at
  **1500 MTU** while ASUS cables negotiate at 9000 MTU. Manual netplan MTU 9000 override did not
  fix the NCCL failure. Use identical cables from the same vendor for all links in a cluster.
- **[conjecture]** **Explicit SSH hostname→IP resolution needed for >2 nodes** (S-forum-3node-nccl,
  amurnane123): on a 4-node cluster, SSH must be explicitly configured so that hostnames always
  resolve to the correct fabric IPs (not mDNS or mgmt LAN IPs). The mgmt-IP wall
  (see `[[wiki/multinode-tp-and-networking.md]]` → Cross-node bring-up) scales with node count —
  more nodes = more chances for a component to advertise a mgmt IP. Hardcode `/etc/hosts` entries
  or use explicit `HostName` in SSH config for every node pair.

### Batch 7 forum ingest (2026-07-11)

- **[conjecture]** **NCCL v2.30u1 CGA buffer pushes GB10 startup check over** (S-forum-mimo-2x-opt,
  renek): newer NCCL (v2.30u1) reserves a ~7.5 GiB CGA buffer that can push the GB10 startup memory
  check over the limit. Fix: use a GB10-targeted NCCL build + `NCCL_CUMEM_ENABLE=0`. Consistent
  with the existing `NCCL_CUMEM_ENABLE=0` requirement proven on this pair. Also pin
  `NCCL_NTHREADS=8`, `NCCL_NSOCKS_PERTHREAD=2`, `NCCL_BUFFSIZE=8388608` for the MiMo TP=2 recipe.

### Batch 9 forum ingest (2026-07-12)

- **[conjecture]** **Third-party 200G QSFP56 passive DAC works immediately** (S-forum-cx7-dual-setup,
  griffith.mark): a standard 200G QSFP56 passive DAC (sourced from Memory Express, Canada) linked
  at 200 Gb/s immediately with no configuration — the NVIDIA-branded "DGX Spark Stacking DAC Cable"
  is not required. The NVIDIA-certified part appears to be **Amphenol NJAAKK-N911** (QSFP112,
  400 mm), though NVIDIA's Spark Stacking guide publishes no part number. NVIDIA's guide says to
  use the same port on both units (both left or both right, viewed from rear).
- **[conjecture]** **Both CX-7 ports sit in a single L2 domain via NIC eSwitch** (S-forum-cx7-dual-setup,
  griffith.mark): on both machines, both CX-7 ports sit in a single L2 domain via the NIC's eSwitch,
  so exact port-to-port choice doesn't matter in practice for a direct-cabled pair. This is a
  useful clarification — the guide's "same port" advice is not a hard requirement.
- **[conjecture]** **Plain TCP (iperf3) ceiling ~16 Gb/s — Grace CPU is the bottleneck, not the
  link** (S-forum-cx7-dual-setup, griffith.mark): iperf3 with 8 streams over the CX-7 link yields
  ~16 Gb/s — the Grace CPU's TCP stack is the bottleneck, not the 200G link. Jumbo frames (MTU 9000)
  did **not** change this. The 200G bandwidth is for RDMA/RoCE (NCCL, multi-node engines), not TCP.
  SSH file transfers: ~600 MB/s (a 51 GB checkpoint ships in ~85 s). This corroborates the proven
  finding that cross-node collectives are host-staged — the CPU is in the data path for any
  non-RDMA traffic.
- **[conjecture]** **NetworkManager config for CX-7 direct link** (S-forum-cx7-dual-setup,
  griffith.mark): persistent fabric interface config via NetworkManager:
  `nmcli con add type ethernet ifname enp1s0f0np0 con-name cx7-cluster ipv4.method manual
  ipv4.addresses 10.77.0.1/24 ipv6.method disabled connection.autoconnect yes`
  (.2 on the other node), plus `/etc/hosts` aliases. MTU 9000 validated with
  `ping -M do -s 8972` to rule out silent blackhole. Interface name `enp1s0f0np0` differs from
  the `rocep1s0f1`/`roceP2p1s0f1` naming in the reference cluster — naming may vary by driver
  version or OEM.
- **[conjecture]** **DCGM works on GB10 including Xid error and PCIe replay counters**
  (S-forum-cx7-dual-setup, griffith.mark): DCGM monitoring works fine on GB10, including
  Xid error counters and PCIe replay counters. Per-link fabric counters (throughput, retransmits
  on CX-7 interfaces) are scraped over the fabric itself so monitoring survives LAN/WiFi flaps.
  This corroborates that DCGM is functional on the GB10 platform.
- **[conjecture]** **PSI (pressure stall) + swap-out rate is better OOM alerting than static memory
  thresholds for inference nodes** (S-forum-cx7-dual-setup, griffith.mark): static "low free memory"
  thresholds are wrong for inference nodes — an engine that reserves 95% of unified memory up front
  leaves ~9 GB free by design and is healthy. Moving OOM alerting to **PSI (pressure stall
  information) + swap-out rate** catches real contention instead of alerting on a working
  reservation. A "cluster tax" metric (multi-node throughput ÷ sum of same nodes run independently)
  gives one number for what the interconnect costs.
- **[conjecture]** **200G links are used during model startup but never at 100% load**
  (S-forum-cx7-dual-setup, mashie): monitoring the 200G links during DeepSeek-V4-Flash startup
  shows they are used, though never at 100% load. No errors clocked on any interfaces. This
  corroborates the proven finding that cross-node collectives are host-staged (the link is
  healthy but the CPU bounce caps effective collective throughput well below link rate).

### Batch 11 forum ingest (2026-07-13)

- **[conjecture]** **100G link (CRS504 switch) costs only 5–10% PP, zero decode loss vs 200G**
  (S-forum-4node-crs504, CosmicRaisins): on a 4-node CRS504 cluster, forcing a 2-node direct
  link down to a single 100G rail showed no change in decode and only 5–10% prefill loss vs
  the full 200G link. Measured inter-node traffic during TP=4 inference was only ~13 Gb/s —
  far below even a single 100G rail. A $25 Amazon 100G cable works with zero issues on a
  2-node direct link. **[reported]** corbett_korbett corroborates: 100G cable gives "pretty
  much the same speed" as 200G. This strengthens the existing finding that cross-node
  collectives are CPU-host-bounced (not link-bound) — a CRS504 switch is a viable cheaper
  alternative to the CRS804 for 4-node clusters. See also S-forum-mikrotik (CRS504 option).
- **[conjecture]** **CRS504 Noctua fan swap** (S-forum-4node-crs504, CosmicRaisins): swapping
  stock CRS504 fans for Noctua 40mm significantly reduces noise, though the switch forces
  ~5000 RPM minimum (slight whine remains). Non-critical ops finding.

### Batch 17 forum ingest (2026-07-16)

- **[conjecture]** **ib_write_bw falsely reports >64 KiB RDMA WRITE failure on GB10 — fabric is
  fine** (S-forum-ibwrite-false, noc19): `ib_write_bw` (RC) deterministically fails above exactly
  65,536 bytes with local protection error (`0x3b 0x0 0x9d`) on the responder — every node pair,
  every RDMA device, 72/72 cells. Boundary is exactly 16 pages. Survives kernel downgrade, IOMMU
  passthrough, fresh reimage; invariant to MR flags, ODP, and relaxed ordering. A minimal
  libibverbs probe doing the identical RC WRITE with responder-side content verification **passes
  at every size to 8 MiB**, every pair, every device — 72/72 clean, byte-exact. NCCL
  all_reduce_perf full sweep: zero validation errors, 24.0 GB/s busbw. The failure is specific to
  the `perftest` instrument, not the transport. Prior reports of the same defect class: thread
  243518 (CX-5, 2023) and 282142 (CX-7/KVM, 2024). Cluster: 3-node Dell Pro Max FCM1253, DGX OS
  7.5.0, CX-7 FW 28.45.4028, dual-rail RoCEv2 via 2× MikroTik CRS804.
  - **[conjecture]** **NCCL_NET_PLUGIN=none required on GB10** (S-forum-ibwrite-false): the
    bundled AWS OFI plugin fails on GB10 unified memory regardless of `NCCL_IB_DISABLE`.
    Set `NCCL_NET_PLUGIN=none` to use NCCL's native IB transport.
  - **[conjecture]** **NCCL_TOPO_FILE correction needed — auto-detected PCIe Gen1×1**
    (S-forum-ibwrite-false): NCCL's detected topology read the GPU PCIe link as Gen1×1 and
    cost-modeled itself to ~0.9 GB/s. A corrected `NCCL_TOPO_FILE` restored expected performance.
    This corroborates the existing `[reported]` CX-7 firmware cap finding (old FW mis-reports
    PCIe as Gen1×1 — S-dgxspark-report).
  - **[conjecture]** **RoCE data is NIC-offloaded — netdev soft counters and tcpdump see nothing**
    (S-forum-ibwrite-false): use `*_phy` counters for port totals and `*_vport_unicast_bytes`
    for per-MAC attribution. The two MACs of one cage share the phy counter.
  - **[conjecture]** **One interface per subnet per node + arp_ignore=1/arp_announce=2**
    (S-forum-ibwrite-false): four subnets for the four fabric netdevs, plus
    `arp_ignore=1`/`arp_announce=2` scoped to the fabric interfaces prevents cross-interface ARP
    collapsing both MACs onto one PCIe x4 root.

### Batch 21 forum ingest (2026-07-18)

- **[conjecture]** **NetworkManager "Connection failed / Activation of network connection
  failed" popup = phantom DHCP profiles looping on ConnectX QSFP ports**
  (S-forum-nm-phantom, YolandaHuang): out-of-box DGX OS on GB10 exposes several Ethernet
  interfaces (`enp1s0f0np0`, `enp1s0f1np1`, `enP2p1s0f0np0`, `enP2p1s0f1np1`). The primary
  port is managed by a netplan-generated profile, but **NetworkManager also auto-creates
  default DHCP profiles (`Wired connection 1..5`) for the remaining ports**. Any port that
  has carrier but no DHCP server behind it (typical when a cable goes directly to another
  Spark, or to a switch segment with no DHCP service) enters a retry loop: activate →
  ip-config → fail (`ip-config-unavailable`) → wait ~45 s → repeat, firing a desktop
  notification each cycle and flooding the NetworkManager journal (which buries real
  network problems later). Does not affect working connections (primary Ethernet, Wi-Fi,
  SSH). **GB10-specific angle:** the multiple CX-7 QSFP fabric interfaces are exactly what
  triggers the phantom-profile storm — a single-port box wouldn't see it. **Diagnosis:**
  `journalctl -u NetworkManager --since "30 min ago" --no-pager | grep -iE "fail|timed out"
  | tail -20` — `ip-config-unavailable` means the port HAS link but got no DHCP lease
  (distinguishes "nothing plugged in" from "plugged in but nobody is serving addresses").
  Cross-check with `nmcli connection show` / `nmcli device status`: looping profiles have
  no DEVICE bound (`--`), while working connections are untouched. **Clean fix:** disable
  autoconnect on the offending profiles only (reversible, no reboot, no service restart):
  `sudo nmcli connection modify "Wired connection 1" connection.autoconnect no` (substitute
  the profile names your journal reported). **Timing gotcha:** `autoconnect no` does not
  cancel a retry cycle already in flight — expect one more failure round, then verify with
  `sleep 120; journalctl -u NetworkManager --since "2 min ago" --no-pager | grep -i fail`
  (empty output = fixed). `nmcli device disconnect <iface>` may error
  `Error: This device is not active` because the device sits in disconnected state between
  attempts — harmless. **To use those ports later:** the phantom-profile loop and "using
  the QSFP ports for clustering" don't conflict — don't rely on DHCP there; follow the
  official Spark clustering playbook (netplan-based, on both systems), or give the profile
  a static address and re-enable: `sudo nmcli connection modify "Wired connection 1"
  ipv4.method manual ipv4.addresses 10.0.0.1/24 connection.autoconnect yes && sudo nmcli
  connection up "Wired connection 1"`. NVIDIA staff had already confirmed (threads 357235,
  352948) this is a NetworkManager message, not a real connectivity error — this finding
  adds the full diagnosis chain and the exact commands. Related to the existing
  NetworkManager fabric config finding (S-forum-cx7-dual-setup).
