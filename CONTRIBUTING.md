# Contributing to sparkbase

sparkbase is a knowledge base, not a codebase — you contribute *findings*, and the currency is
**honest evidence**. A contribution is good when it's durable, GB10-specific, on the right page, and
tagged at the tier its evidence actually earns. Read [`SCHEMA.md`](SCHEMA.md) once; it's the whole
contract.

## The two stacks

Which rules you follow depends on one thing — **do you have a real DGX Spark to run on?**

- **Hardware contributors** can verify claims and promote them to `[reproduced]`/`[proven]`, add
  first-party benchmarks, and let measurement overrule the forums. Guidance: [`agents/stack-hardware.md`](agents/stack-hardware.md).
- **Analysis contributors** (no hardware) ingest forum posts, GitHub repos, and reports into
  well-organized, well-tagged conjecture, flag contradictions, and queue experiments — but never
  promote past `[reported]`. Guidance: [`agents/stack-analysis.md`](agents/stack-analysis.md).

This works for humans and for coding agents alike: point an agent at the repo and tell it to read
[`AGENTS.md`](AGENTS.md).

## The rules that keep it trustworthy

1. **Never overstate evidence.** Tag every claim on the ladder (`conjecture → reported → reproduced →
   proven → superseded`). When unsure, tag *lower*. A plausible forum number is `[conjecture]`, not proof.
2. **Only hardware-backed work reaches `[reproduced]`/`[proven]`**, and it must cite the run/config.
3. **Cite your source.** Register it in `sources/README.md` with its `type` (forum / repo / report /
   first-party) and a URL or a first-party experiment description.
4. **Merge, don't append.** Strengthen the existing claim before adding a new bullet. One fact, one place.
5. **Keep it GB10-specific and vendor-neutral.** Only what *bites on Spark*. No private hostnames, IPs,
   internal service names, or personal filesystem paths — refer to nodes by role (`<head>`/`<worker>`).
6. **Log it.** One line in `log.md`: date, source(s), pages touched, what changed.

## How to open a change

Standard GitHub flow: fork, edit the relevant `wiki/*.md` page(s) + `sources/README.md` + `log.md`,
open a PR. In the PR description, say what evidence tier your claims are and why. Hardware-verified PRs
should include the config and the measured result so a reviewer can see the `[proven]` is earned.
