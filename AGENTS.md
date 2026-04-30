# Optimization Playground Agent Rules

- Treat `~/.agentmemory/standalone.json` as a required workflow dependency:
  read it before every nontrivial optimization iteration, and write a concise
  durable entry after every result, diagnosis, launch, correction, or decision
  before continuing.
- For the active REAP optimization lane, work only on the combined
  IndexCache + dense TurboQuant + SMC-SD implementation. Do not optimize HISA
  unless a later user instruction explicitly reopens it.
- Always benchmark the full IndexCache + dense TurboQuant + SMC-SD combo. The
  baseline is the current upstream combo implementation, not vanilla MTP.
- Run this lane on `a4-us-001-rl9`.
- Optimize end-to-end throughput as the primary metric, with less than 1% TTFT
  regression on every matrix cell: 8192/1024, 16384/1024, 16384/4096,
  32768/4096, and 65536/4096.
- Keep IndexCache close to the reference integration. Focus iteration on dense
  TurboQuant and SMC-SD unless measurements prove IndexCache is the blocker.
- Dense TurboQuant 2.5-bit must beat the upstream NVFP4 KV cache comparator
  (`73e93bebd6cbd2dbec8ea6dd1a78529bbc58080b`) in both memory footprint and
  end-to-end throughput for this lane. Use NVFP4 as a kernel and systems
  reference, not as the target stack. For REAP NSA/IndexCache, native NVFP4
  throughput comparison is blocked until an NSA FP4 KV pool/backend exists; do
  not count a non-NSA or BF16 fallback run as an apples-to-apples comparator.
- Use systematic, targeted analysis: identify the hotspot, record the evidence,
  make the smallest defensible change, test it through autoinfer, and record the
  result before starting the next change.
- For kernel work, collect IKP profiler artifacts before promotion or upstream
  push. Use NV-BFP for B200/NVIDIA block-floating-point quantization and decode
  kernel investigations when relevant.
- Preserve prompt privacy: do not store prompts, model outputs, token IDs, or
  request payloads in artifacts.
- Write clear docs before every upstream push.
