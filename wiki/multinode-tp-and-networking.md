# Multi-node TP & networking

> **area:** multinode
> **status:** stable
> **evidence:** proven
> **sources:** S-networking, S-mimo-results, S-m3-vision, S-xnode-cudagraph, S-sess-jun11, S-nemotron-rpc, S-pr46372, S-dgxspark-report, S-forum-cx7-13gbps, S-forum-mikrotik, S-forum-ddp-timeout, S-forum-2d-parallel, S-forum-sglang-traps, S-forum-glm47-rdma, S-forum-4node-mesh, S-forum-roce-397b-mtp, S-forum-ds4f-4x-vllm, S-forum-m25-sglang-4x, S-forum-3node-nccl
> **updated:** 2026-07-11

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
