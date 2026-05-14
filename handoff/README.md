# optimization-playground handoff

Self-contained handoff tooling for moving in-flight `optimization-playground`
work to another device while preserving the shared agentmemory state used by
multiple agents.

This is adapted from `SpencerGarnets/ion-w5-handoff`, but the memory model is
different: agentmemory is shared with another agent that may be executing the
same process. The restore path is therefore append-only and merge-based. It
never deletes an existing memory store and it never creates a fresh memory
instance unless no store exists.

## Contents

| File | Purpose |
| --- | --- |
| `pack-state.sh` | Create a timestamped handoff tarball from this machine. |
| `unpack-state.sh` | Restore on a new device, merging shared agentmemory. |
| `MANIFEST.md` | Project state, expected repos, and operating rules. |
| `setup/setup-all.sh` | Run all setup helpers in order. |
| `setup/setup-prereqs.sh` | Install or verify shell prerequisites. |
| `setup/setup-github.sh` | Install/verify GitHub CLI auth and git identity. |
| `setup/setup-gcloud.sh` | Install/verify Google Cloud SDK auth. |
| `setup/setup-huggingface.sh` | Install/verify Hugging Face Hub auth. |

## One-shot flow on the source device

```bash
cd /path/to/optimization-playground
bash handoff/pack-state.sh
```

The script writes `optimization-playground-handoff-YYYYMMDD-HHMMSS.tar.gz` in
the current directory. Commit and push code before moving machines whenever
possible; GitHub remains the source of truth for code.

By default the pack script skips live Git command capture so a local submodule
or filesystem stall cannot block the handoff. To include Git remotes, HEAD,
recent log, status, and worktree metadata, run:

```bash
OP_HANDOFF_CAPTURE_GIT=1 bash handoff/pack-state.sh
```

## One-shot flow on the new device

```bash
git clone https://github.com/ai-blaise/optimization-playground.git
cd optimization-playground
bash handoff/setup/setup-all.sh
bash handoff/unpack-state.sh /path/to/optimization-playground-handoff-*.tar.gz
```

The setup scripts are idempotent. Auth steps may require browser/token
interaction. The restore script can run unattended after the bundle is present.

## Agent prompt for a fresh session

```text
Read handoff/README.md and handoff/MANIFEST.md fully.

Use the existing shared agentmemory. Do not create a fresh instance unless no
agentmemory exists on this device. If restoring from a handoff bundle, run:

  bash handoff/unpack-state.sh /path/to/optimization-playground-handoff-*.tar.gz

Then verify:

  python3 - <<'PY'
  import json, pathlib
  p = pathlib.Path.home() / ".agentmemory" / "standalone.json"
  data = json.loads(p.read_text())
  print(p, len(data))
  PY

Before doing any repo work, read recent memories matching:
optimization-playground, NVFP4, HISA, IndexCache, HIGGS, GatedNorm,
Gated Attention, SMC-SD, TokenSpeed, FlashSampling, NCCLX, LayerSplit.

Continue the project loop: implement, test and verify, iterate until successful,
write docs, clean up slop, commit, and push upstream.
```

## Shared agentmemory contract

The canonical local memory file is:

```text
~/.agentmemory/standalone.json
```

Rules:

- Treat the file as shared state. Another agent may be reading or writing it.
- Use timestamped, descriptive keys for every write.
- Never delete or truncate the store during handoff.
- Restore by merging source keys into the destination store.
- If a source key conflicts with a different destination value, preserve both:
  the destination key is kept and the source value is written under a
  timestamped conflict key.
- Read recent project memories before starting work and after any context
  transition.

`unpack-state.sh` uses an advisory lock at
`~/.agentmemory/standalone.json.lock` for its merge. Agents that write directly
to the JSON file should use the same lock when possible.

## What is not bundled

The bundle intentionally does not include:

- GitHub, Google Cloud, or Hugging Face tokens.
- SSH private keys.
- Model weights or Hugging Face cache contents.
- Docker image layers.
- VM-local build trees.

The bundle records enough metadata to reconstruct those pieces using the setup
scripts and the project repos.
