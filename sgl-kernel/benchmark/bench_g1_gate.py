import argparse
import json

import torch

from sgl_kernel import g1_gate_forward

try:
    from sgl_kernel import g1_gate_forward_fused
except ImportError:  # pragma: no cover - older sgl-kernel builds
    g1_gate_forward_fused = None


def measure(fn, warmup: int, iters: int) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> None:
    parser = argparse.ArgumentParser(description='Benchmark the BF16 G1 gate kernels.')
    parser.add_argument('--hidden-size', type=int, default=7168)
    parser.add_argument('--tokens', type=str, default='1,16,64,256,1024')
    parser.add_argument('--warmup', type=int, default=50)
    parser.add_argument('--iters', type=int, default=200)
    parser.add_argument('--json', action='store_true')
    args = parser.parse_args()

    torch.cuda.set_device(0)
    torch.manual_seed(20260517)
    rows = []
    for tokens in [int(x) for x in args.tokens.split(',') if x]:
        shape = (tokens, args.hidden_size)
        linear_out = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        attn_out = torch.randn(shape, device='cuda', dtype=torch.bfloat16)
        output = torch.empty_like(attn_out)
        gate = torch.empty_like(attn_out)

        def full_alloc():
            return g1_gate_forward(linear_out, attn_out)

        def full_out():
            return g1_gate_forward(linear_out, attn_out, output=output, gate=gate)

        row = {
            'tokens': tokens,
            'elements': tokens * args.hidden_size,
            'full_alloc_ms': measure(full_alloc, args.warmup, args.iters),
            'full_out_ms': measure(full_out, args.warmup, args.iters),
        }
        expected = (attn_out.float() * torch.sigmoid(linear_out.float())).to(torch.bfloat16)
        actual, _ = g1_gate_forward(linear_out, attn_out)
        row['full_maxdiff'] = float((actual.float() - expected.float()).abs().max().item())

        if g1_gate_forward_fused is not None:
            fused_output = torch.empty_like(attn_out)

            def fused_alloc():
                return g1_gate_forward_fused(linear_out, attn_out)

            def fused_out():
                return g1_gate_forward_fused(linear_out, attn_out, output=fused_output)

            row['fused_alloc_ms'] = measure(fused_alloc, args.warmup, args.iters)
            row['fused_out_ms'] = measure(fused_out, args.warmup, args.iters)
            fused_actual = g1_gate_forward_fused(linear_out, attn_out)
            row['fused_maxdiff'] = float(
                (fused_actual.float() - expected.float()).abs().max().item()
            )
            row['alloc_speedup_pct'] = (
                (row['full_alloc_ms'] - row['fused_alloc_ms'])
                / row['full_alloc_ms']
                * 100.0
            )
            row['out_speedup_pct'] = (
                (row['full_out_ms'] - row['fused_out_ms'])
                / row['full_out_ms']
                * 100.0
            )
        rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    headers = sorted({key for row in rows for key in row})
    print(','.join(headers))
    for row in rows:
        print(','.join(str(row.get(key, '')) for key in headers))


if __name__ == '__main__':
    main()
