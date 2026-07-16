# Multi-node TP & networking

> **area:** multinode
> **status:** stable
> **evidence:** proven
> **sources:** S-networking, S-mimo-results, S-m3-vision, S-xnode-cudagraph, S-sess-jun11, S-nemotron-rpc, S-pr46372, S-dgxspark-report, S-forum-cx7-13gbps, S-forum-mikrotik, S-forum-ddp-timeout, S-forum-2d-parallel, S-forum-sglang-traps, S-forum-glm47-rdma, S-forum-4node-mesh, S-forum-roce-397b-mtp, S-forum-ds4f-4x-vllm, S-forum-m25-sglang-4x, S-forum-3node-nccl, S-forum-mimo-2x-opt, S-forum-cx7-dual-setup, S-forum-4node-crs504, S-forum-qwen397-arch, S-forum-ibwrite-false
> **updated:** 2026-07-16

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
