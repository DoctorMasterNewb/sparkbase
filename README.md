# 🧠 sparkbase

A **community knowledge base** for serving LLMs on **NVIDIA GB10 / DGX Spark**.

Every model brought up on a Spark teaches someone something the hard way — a cudagraph that crashes
only cross-node, a quant kernel vLLM won't dispatch, an NCCL transport with no GPUDirect, a chat
template that silently emits garbage. sparkbase is where those findings live so the next person (or
their coding agent) doesn't re-derive them. It's a
[Karpathy-style LLM wiki](https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f):
distilled markdown pages an LLM **accumulates** into over time, instead of re-reading raw logs on
every question.

## What makes it trustworthy: the evidence ladder

DGX Spark knowledge is scattered across forum posts and unproven repos, and a lot of it is wrong or
version-skewed. So **every claim in sparkbase wears its evidence on its sleeve**:

| Tag | Means |
|---|---|
| `[conjecture]` | one source says so — a forum post, a repo README, a slide. Unverified. |
| `[reported]` | multiple independent sources agree, but no one here has run it. |
| `[reproduced]` | run once on a real DGX Spark and it held. |
| `[proven]` | verified on real GB10 hardware, with the run cited. The load-bearing tier. |
| `[superseded]` | was believed, later overturned — kept with a pointer to what replaced it. |

You always know whether a line is a rumor or a measurement. The full rules are in [`SCHEMA.md`](SCHEMA.md).

## Layout

| Path | What |
|---|---|
| [`index.md`](index.md) | the map — every page, grouped, one line each. **Start here.** |
| `wiki/` | the knowledge: platform, multinode, cudagraphs, quantization, attention, per-model pages, benchmarks, roadmap |
| `sources/` | registry of where findings came from (forum / repo / report / first-party), with `S-` ids |
| [`SCHEMA.md`](SCHEMA.md) | the contract: page format, evidence ladder, how to ingest/query/lint |
| [`AGENTS.md`](AGENTS.md) + `agents/` | guidance for the two kinds of agent that maintain this KB |
| `log.md` | append-only record of what was ingested/changed when |

## How to use it

- **Bringing up a model on a Spark?** Skim [`wiki/platform-gb10.md`](wiki/platform-gb10.md) + the
  relevant quant/attention pages, then the model page if one exists. Watch the evidence tags — build on
  `[proven]`, treat `[conjecture]` as "try it and tell us."
- **Have a Spark and want to help?** You can turn conjecture into proof. See
  [`agents/stack-hardware.md`](agents/stack-hardware.md) and [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **No hardware but want to help?** Ingest and organize the community's scattered findings into
  well-tagged conjecture. See [`agents/stack-analysis.md`](agents/stack-analysis.md).
- **Want the open problems?** [`wiki/roadmap.md`](wiki/roadmap.md).

## The two-stack model

sparkbase is maintained by agents (and humans) in two roles, split by **who can touch the metal**:
agents *with* a real Spark verify and promote claims to `[reproduced]`/`[proven]`; agents *without*
ingest, triage, and organize conjecture but never overstate it. Point any capable coding agent at this
repo, tell it to read `AGENTS.md`, and it can review, validate, and extend the KB within its stack's rules.

MIT licensed. Contributions welcome — see [`CONTRIBUTING.md`](CONTRIBUTING.md).
