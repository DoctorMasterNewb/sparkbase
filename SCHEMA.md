# sparkbase — schema & operating manual

sparkbase is a **community knowledge base** for serving LLMs on **NVIDIA GB10 / DGX Spark**
hardware. It follows Andrej Karpathy's LLM-wiki pattern
(https://gist.github.com/karpathy/442a6bf555914893e9891c11519de94f): knowledge is *distilled and
accumulated* into durable markdown pages, not re-derived from raw logs every time.

The thing that makes sparkbase trustworthy is that **every claim wears its evidence on its sleeve.**
A rumor from a forum thread and a number measured on real silicon look different on the page, so a
reader (human or agent) always knows how much weight a line can bear. This file is the contract for
how that's done. Read it before ingesting or editing.

## The three layers

1. **Sources (raw, cited)** — `sources/`. Where a finding came from: a forum post, a GitHub repo, a
   vendor report, or a first-party experiment on real hardware. We don't paraphrase sources into
   truth; we cite them and record *what kind* of source each is. `sources/README.md` is the registry
   (each source gets a short `S-` id).
2. **Wiki (distilled)** — `wiki/`. Atomic, deduplicated pages of reusable knowledge. This is the
   product. A fact lives on exactly one page; other pages link to it. Every claim carries an
   **evidence tag** (below).
3. **Schema (this file)** — how the wiki is built and kept honest.

Plus two ledgers at the root:
- `index.md` — the map: every wiki page with a one-line description, grouped by area. Keep current.
- `log.md` — append-only. One entry per ingest/lint: date, source(s), pages touched, one line of what changed.

## The evidence ladder

This is the heart of sparkbase. Every non-trivial claim is tagged with where it sits on this ladder.
Tags are **machine-readable brackets** at the start of the claim: `[conjecture]`, `[reported]`,
`[reproduced]`, `[proven]`, `[superseded]`.

| Tag | Bar it clears | Who can assign it |
|---|---|---|
| `[conjecture]` | A single source asserts it — one forum post, one repo's README, one vendor slide, or a plausible hypothesis. **Unverified.** | anyone |
| `[reported]` | **Multiple independent** sources agree, but no one in sparkbase has run it. Community consensus, still not first-hand. | anyone |
| `[reproduced]` | Someone ran it on *a* DGX Spark and it held — but the run isn't fully characterized or independently repeated. | an agent/human with hardware |
| `[proven]` | Verified on real GB10 hardware, with the run/number cited (a first-party `S-` source). The load-bearing tier. | an agent/human with hardware |
| `[superseded]` | Was believed; later overturned. Kept for the record with a pointer to what replaced it. | anyone, with a successor link |

Rules:
- **Only hardware-backed work reaches `[reproduced]`/`[proven]`.** An agent without a Spark can ingest,
  triage, and raise a claim to `[reported]`, but must never promote past it. See `agents/`.
- **A claim is tagged at its highest *honestly earned* tier**, and cites the source(s) that put it
  there. Promoting a claim means adding the new source and bumping the tag in the same edit.
- **Conflicting evidence** doesn't silently overwrite. If a `[proven]` first-party result contradicts
  a `[reported]` forum consensus, the proven one wins the claim and the forum claim becomes
  `[superseded]` with a one-line note on the disagreement. Newer/validated wins; record the supersession.
- **Numbers are proven or they're conjecture.** A tok/s figure without a first-party run behind it is
  `[conjecture]`/`[reported]` at best, and must name whose number it is ("the forum reports ~56 tok/s").

Optional per-claim detail when it helps: a `(confidence: reproduced 2×, head+worker)` aside, or a
`(superseded-by: <link>)`. Don't bloat every line — add detail only where the stakes warrant it.

## Page format

Every `wiki/*.md` page starts with a metadata header, then content:

```markdown
# <Title>

> **area:** platform | multinode | cudagraphs | quantization | attention | llama.cpp | containers | model | benchmarks | roadmap
> **status:** stable | evolving | open-problem
> **evidence:** proven | mixed | reported | conjecture   ← the page's dominant / most-cautious tier
> **sources:** S-xxx, S-yyy
> **updated:** YYYY-MM-DD

<body>
```

Body conventions:
- Lead with the takeaway (the rule/finding), then the why, then specifics (exact flags, env vars,
  error strings, numbers, model ids, image tags). Concrete > prose.
- **Tag claims inline.** `- **[proven]** Decode is bandwidth-bound (~270 GB/s ceiling). (S-...)` /
  `- **[conjecture]** A forum post claims MXFP4 dispatches natively; untested. (S-forum-...)`.
  Purely definitional or structural lines (headers, "see also") need no tag.
- **Issues** use this shape: **Symptom → Root cause → Workaround → Status** (`fixed` / `open` /
  `upstream:<link>`), and carry an evidence tag on the finding.
- Cross-link with `[[wiki/path/page.md]]` (or normal markdown links). Link liberally.
- **Keep it GB10-specific and vendor-neutral.** Generic vLLM/llama.cpp knowledge belongs upstream —
  only what *bites on Spark* lives here. No private hostnames, IPs, internal service names, or
  personal filesystem paths (see "Sanitization" below).

## Source types

Every row in `sources/README.md` records a **type**, because the type caps the evidence tier a source
can justify on its own:

| type | e.g. | on its own justifies up to |
|---|---|---|
| `forum` | a DGX Spark forum / Reddit / Discord post | `[conjecture]` (→ `[reported]` if several agree) |
| `repo` | a GitHub repo, README, or recipe, unverified | `[conjecture]` (→ `[reported]` if several agree) |
| `report` | a vendor spec sheet, blog, or research writeup | `[conjecture]`/`[reported]` |
| `first-party` | an experiment/benchmark run on a real DGX Spark, with the run recorded | `[reproduced]` / `[proven]` |

A source registry row: `| S-id | type | one-line description | URL or "first-party: <what/when>" | date |`.
Public sources cite a URL. First-party sources cite the experiment (what was run, on what config,
when) — **not** a private filesystem path.

## Operations

### Ingest (add knowledge from a new source)
See `agents/ingest.md` for the full procedure. In short:
1. Register the source in `sources/README.md` (assign an `S-` id; record type + URL/experiment + date).
2. Extract only **durable, GB10-relevant** findings.
3. For each finding: find the page it belongs on (`index.md` is the map). Merge into the right claim —
   don't append blindly. Tag it at the tier its evidence earns (a forum source ⇒ `[conjecture]`).
4. Cite the source id in the page's `sources:`, update `evidence:` if the dominant tier changed, bump `updated:`.
5. Append one `log.md` entry.

### Query (answer from the KB)
- Read `index.md`, jump to the relevant page(s), answer from distilled pages — not raw sources.
- **Carry the evidence tier into the answer.** "It's proven that…" vs "a forum post conjectures…".
- If the KB can't answer, say so (optionally ingest the missing knowledge so next time it can).

### Promote / demote (keep tiers honest)
- New corroborating source ⇒ raise the tag (e.g. a second independent forum report: `[conjecture]`→`[reported]`).
- A first-party run confirming a `[reported]` claim ⇒ `[proven]`, cite the run. Only hardware agents do this.
- A first-party run *contradicting* a claim ⇒ the claim becomes `[superseded]`, the proven result takes its place.

### Lint (keep it clean — run periodically)
- Dedupe: the same fact on two pages → keep the better one, link the other.
- Reconcile contradictions per the ladder (newer/validated wins; note the supersession).
- Fix dead cross-links; ensure every page is in `index.md`; ensure `status:` and `evidence:` are honest.
- Flag stale `[conjecture]`/`[reported]` claims that a hardware agent could cheaply verify (feed `roadmap.md`).
- Split pages that sprawl; merge stubs.

## Sanitization (this is a public repo)

sparkbase is community-shared. Keep it free of anyone's private setup:
- **No** internal hostnames, private IPs presented as *the* cluster, internal service/app names, or
  personal filesystem paths (`~/services/...`, `~/.claude/...`).
- Refer to nodes by **role** (`<head>`, `<worker>`) and use **example** fabric IPs framed as examples.
- First-party findings cite *that they were measured on a real DGX Spark* and the config — not where
  the files live on the author's machine.

## Cluster context (what pages may assume)

The reference platform is a **DGX Spark (GB10)** node — compute capability **12.1 (sm_121,
`TORCH_CUDA_ARCH_LIST=12.1a`)**, **~121 GB unified memory** per node, **~270 GB/s** memory bandwidth,
**no native FP4/block-scale-FP8 compute**, **no GPUDirect**. Multi-node findings assume **2× DGX Spark**
direct-cabled over ConnectX-7 (RoCE) — head + worker roles, serving on a single port. Any *specific*
numbers beyond these are first-party and tagged accordingly. See `[[wiki/platform-gb10.md]]`.

## Scope

In: GB10/DGX-Spark inference — vLLM, llama.cpp, sglang; quant kernels; multinode TP; networking;
cudagraphs/compile; per-model bring-up gotchas; benchmarks; open problems. Out: anything not specific
to running models on this hardware.
