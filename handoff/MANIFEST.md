# optimization-playground handoff manifest

This manifest captures the project state and restore policy for moving
`optimization-playground` work to a new device.

## Repositories

Primary repo:

- `ai-blaise/optimization-playground`

Common adjacent repos used by this project:

- `ai-blaise/Megatron-LM`
- `ai-blaise/infrastructure`
- `ai-blaise/autoinfer`
- `ai-blaise/dynamo-prod-k8s`
- `deepseek-ai/TileKernels`
- `NVIDIA/cutlass`
- `yao-jz/intra-kernel-profiler`
- `aturker1/cutest`

`unpack-state.sh` clones only the primary repo by default. Set
`OP_HANDOFF_CLONE_ADJACENT=1` to clone the adjacent repos listed above.

## Current customization surface

The project state has included work on:

- Gated Attention and GatedNorm model-path wiring.
- IndexCache and NVFP4 IndexCache Indexer.
- NVFP4 IndexCache+HISA Indexer, Compression Ratio = 4:1.
- 2.5-bit dense TurboQuant MLA KV.
- 2-bit dense HIGGS MLA KV.
- SMC-SD serving/speculative decode integration.
- HiSparse, LayerSplit, TokenSpeed, FlashSampling, NCCLX, and Warp Decode
  experiments.
- Dynamo production deployment scripts and ai-blaise infrastructure helpers.
- Megatron `upstream-dev` and NVFP4 indexer forward/backward integration.

Always verify the current GitHub state before assuming any item above is still
the latest accepted implementation.

## Target model context

Frequently used target models:

- `BlaiseAI/DeepSeek-V3.2-REAP-345B-NVFP4-W4A4KV4-IndexerK8-FP8-GatedNorm-G1`
- `BlaiseAI/DeepSeek-V3.2-REAP-345B-SpinQuant-ActKV-NVFP4`

HF config deployment should support the custom stack where applicable,
including IndexCache, NVFP4 IndexCache+HISA, dense KV quantization, Gated
Attention, and GatedNorm. SMC-SD remains a serving/deployment policy unless a
future explicit HF serving-config surface is added.

## Shared agentmemory state

Canonical local file:

```text
~/.agentmemory/standalone.json
```

The memory store is shared with at least one other agent. That agent may be
executing the same process. Therefore:

- Do not create a new memory instance as part of handoff.
- Do not wipe or replace the destination memory file if it exists.
- Merge source memories into the destination append-only.
- Preserve conflicting values under conflict keys.
- Persist the memory structure on the new device exactly as a shared JSON store
  unless the user explicitly changes the memory backend.

Required memory actions for agents:

- Read relevant memories before each implementation phase.
- Write start, decision, validation, failure, and final-state entries.
- Use unique keys with project and UTC/local timestamp context.
- Record negative results as well as successful candidates.

## Setup/auth expectations

Auth is intentionally not bundled. On a new device, verify:

- GitHub CLI auth with access to `ai-blaise` repos.
- Google Cloud SDK auth for project `blaise-478114`.
- Hugging Face auth with access to the private `BlaiseAI` model repos.
- SSH access to any active VM only when keys are provided by the user.

The setup scripts can install CLIs, but browser/token auth remains interactive.

## Execution rules

Project-level rules to preserve across devices:

1. Prioritize compatibility with upstream repos and keep changes minimal.
2. Follow Google C++/Python style guidance and the Rust style guide where
   relevant.
3. End each implementation phase with accuracy/effectiveness verification,
   slop cleanup, docs, commit, and push.
4. Prefer uploaded materials, official docs, papers, reference repos, and
   existing building blocks over from-scratch implementations.
5. Use `ai-blaise/infrastructure` as setup reference where relevant.
6. For custom GPU kernels, use CuTe where appropriate, use IKP for targeted
   analysis, and consult the Blackwell/CUTLASS/CuTe reference materials.
7. Download what is needed for implementation and verification.
8. Execute autonomously when access exists.
9. Loop: implement, test and verify, iterate until objectives are met, write
   docs, clean up, commit, push.
10. Use shared agentmemory extensively. This is the top-priority continuity
    rule.

## Restore success criteria

After `unpack-state.sh`:

- `~/.agentmemory/standalone.json` exists and contains merged source memories.
- Existing destination memories, if any, are still present.
- A backup of the pre-merge destination exists when a destination file existed.
- The primary repo exists at the destination path.
- `git remote -v`, `git status --short`, and `git log -1 --oneline` work.
- The agent has read this manifest and written a restore-complete memory entry.
