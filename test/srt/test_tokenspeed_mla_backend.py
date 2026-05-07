import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]


def read_source(path: str) -> str:
    return (ROOT / path).read_text()


def test_tokenspeed_mla_backend_is_wired_as_opt_in_backend():
    server_args = read_source("python/sglang/srt/server_args.py")
    registry = read_source("python/sglang/srt/layers/attention/attention_registry.py")
    handler = read_source(
        "python/sglang/srt/models/deepseek_common/attention_backend_handler.py"
    )
    deepseek_utils = read_source("python/sglang/srt/models/deepseek_common/utils.py")

    assert '"tokenspeed_mla"' in server_args
    assert '@register_attention_backend("tokenspeed_mla")' in registry
    assert 'AttentionBackendRegistry.register("tokenspeed_mla"' in handler
    assert '"tokenspeed_mla"' in deepseek_utils


def test_tokenspeed_mla_backend_keeps_custom_stack_out_of_vanilla_path():
    backend = read_source("python/sglang/srt/layers/attention/tokenspeed_mla_backend.py")

    assert "tokenspeed_mla_decode" in backend
    assert "tokenspeed_mla_prefill" in backend
    assert "output_scale=" in backend
    assert "enable_turboquant" not in backend
    assert "nsa_indexcache" not in backend
    assert "smc" not in backend.lower()


def test_tokenspeed_mla_backend_documents_runtime_constraints():
    backend = read_source("python/sglang/srt/layers/attention/tokenspeed_mla_backend.py")
    docs = read_source("docs/advanced_features/attention_backend.md")

    assert "SM100" in backend
    assert "page_size 32 or 64" in backend
    assert "tokenspeed-mla" in docs


def test_tokenspeed_mla_routes_deepseek_nsa_through_custom_stack():
    server_args = read_source("python/sglang/srt/server_args.py")
    nsa_backend = read_source("python/sglang/srt/layers/attention/nsa_backend.py")

    assert '"tokenspeed_mla",' in server_args
    assert "Routing TokenSpeed MLA through the NSA backend" in server_args
    assert 'self.attention_backend = "nsa"' in server_args
    assert 'self.nsa_decode_backend = "tokenspeed_mla"' in server_args
    assert "TokenSpeed MLA is a dense MLA backend" not in server_args
    assert "does not implement the DeepSeek DSA/NSA" not in server_args

    assert '_NSA_IMPL_T: TypeAlias = Literal[' in nsa_backend
    assert '"tokenspeed_mla"' in nsa_backend
    assert "def _pack_tokenspeed_selected_kv(" in nsa_backend
    assert "def _forward_tokenspeed_mla_selected(" in nsa_backend
    assert "get_turboquant_selected_kv_buffer" in nsa_backend


def test_tokenspeed_mla_nsa_validation_preserves_custom_stack_flags():
    server_args = read_source("python/sglang/srt/server_args.py")
    server_args_docs = read_source("docs/advanced_features/server_arguments.md")

    assert '"bfloat16": {"flashmla_sparse", "tokenspeed_mla"}' in server_args
    assert '"tilelang",\n                    "tokenspeed_mla",' in server_args
    assert "TokenSpeed MLA NSA selected-page integration currently" in server_args
    assert "`trtllm`, `tokenspeed_mla`" in server_args_docs
    assert "--enable-turboquant-dense-kv-cache" in server_args
    assert "--nsa-prefill-cp-kv-storage-mode=layersplit" in server_args
    assert "--nsa-indexer-mode" in server_args
    assert "--enable-hisparse" in server_args


def test_tokenspeed_mla_ab_script_keeps_dense_mla_comparison_separate():
    script = read_source("scripts/playground/run-tokenspeed-mla-ab.sh")

    assert "CANDIDATE_BACKEND=${CANDIDATE_BACKEND:-tokenspeed_mla}" in script
    assert "--attention-backend" in script
    assert "--enable-hisparse" not in script
    assert "--nsa-indexer-mode" not in script
    assert "--enable-turboquant-dense-kv-cache" not in script
    assert "--speculative-algorithm" not in script


def test_tokenspeed_mla_ab_script_dry_run_builds_baseline_and_candidate():
    script = ROOT / "scripts/playground/run-tokenspeed-mla-ab.sh"
    subprocess.run(["bash", "-n", str(script)], check=True)
    result = subprocess.run(
        ["bash", str(script)],
        check=True,
        env={
            **os.environ,
            "DRY_RUN": "1",
            "MATRIX": "8:1",
            "MODEL_PATH": "dummy",
            "LOAD_FORMAT": "dummy",
            "TP_SIZE": "1",
            "DP_SIZE": "1",
        },
        text=True,
        capture_output=True,
    )

    assert "--attention-backend trtllm_mla" in result.stdout
    assert "--attention-backend tokenspeed_mla" in result.stdout
    assert "--random-input-len 8" in result.stdout
    assert "--random-output-len 1" in result.stdout
    assert "--enable-hisparse" not in result.stdout


def _require_tokenspeed_kernel_env():
    if os.environ.get("SGLANG_TEST_TOKENSPEED_KERNELS") != "1":
        pytest.skip("set SGLANG_TEST_TOKENSPEED_KERNELS=1 to run TokenSpeed kernels")

    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("TokenSpeed MLA correctness requires CUDA")
    if torch.cuda.get_device_capability()[0] != 10:
        pytest.skip("TokenSpeed MLA correctness requires SM100")
    tokenspeed_mla = pytest.importorskip("tokenspeed_mla")
    return torch, tokenspeed_mla


def _reference_decode(torch, query, kv_cache, block_tables, seq_lens, scale, kv_lora_rank):
    page_size = kv_cache.shape[1]
    outputs = []
    for batch_idx, seq_len in enumerate(seq_lens.tolist()):
        pages = []
        remaining = seq_len
        for page in block_tables[batch_idx].tolist():
            if page < 0 or remaining <= 0:
                break
            take = min(page_size, remaining)
            pages.append(kv_cache[page, :take])
            remaining -= take

        kv = torch.cat(pages, dim=0).float()
        q = query[batch_idx].float()
        scores = torch.einsum("qhd,kd->qhk", q, kv) * scale
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qhk,kd->qhd", probs, kv[:, :kv_lora_rank]))

    return torch.stack(outputs, dim=0)


def test_tokenspeed_mla_decode_matches_torch_reference():
    torch, tokenspeed_mla = _require_tokenspeed_kernel_env()
    torch.manual_seed(0)

    device = torch.device("cuda")
    batch_size = 2
    q_len = 1
    num_heads = 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    page_size = 32
    seq_lens = torch.tensor([37, 19], dtype=torch.int32, device=device)
    block_tables = torch.tensor([[0, 1], [2, -1]], dtype=torch.int32, device=device)
    query = torch.randn(
        batch_size,
        q_len,
        num_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    kv_cache = torch.randn(3, page_size, head_dim, dtype=torch.bfloat16, device=device)
    workspace = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device=device)
    scale = head_dim**-0.5

    output = tokenspeed_mla.tokenspeed_mla_decode(
        query=query,
        kv_cache=kv_cache,
        workspace_buffer=workspace,
        kv_lora_rank=kv_lora_rank,
        qk_rope_head_dim=qk_rope_head_dim,
        block_tables=block_tables,
        seq_lens=seq_lens,
        max_seq_len=int(seq_lens.max().item()),
        softmax_scale=scale,
        enable_pdl=False,
    )
    expected = _reference_decode(
        torch, query, kv_cache, block_tables, seq_lens, scale, kv_lora_rank
    )

    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=3e-2)


def _reference_prefill(torch, query, key, value, seq_lens, scale):
    outputs = []
    offset = 0
    for seq_len in seq_lens.tolist():
        q = query[offset : offset + seq_len].float()
        k = key[offset : offset + seq_len, 0].float()
        v = value[offset : offset + seq_len, 0].float()
        scores = torch.einsum("qhd,kd->qhk", q, k) * scale
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=query.device).tril()
        scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("qhk,kd->qhd", probs, v))
        offset += seq_len
    return torch.cat(outputs, dim=0)


def test_tokenspeed_mla_prefill_matches_torch_reference():
    torch, tokenspeed_mla = _require_tokenspeed_kernel_env()
    torch.manual_seed(1)

    device = torch.device("cuda")
    seq_lens = torch.tensor([3, 5], dtype=torch.int32, device=device)
    cum_seq_lens = torch.tensor([0, 3, 8], dtype=torch.int32, device=device)
    num_heads = 128
    kv_lora_rank = 512
    qk_rope_head_dim = 64
    head_dim = kv_lora_rank + qk_rope_head_dim
    query = torch.randn(8, num_heads, head_dim, dtype=torch.bfloat16, device=device)
    key = torch.randn(8, 1, head_dim, dtype=torch.bfloat16, device=device)
    value = torch.randn(8, 1, kv_lora_rank, dtype=torch.bfloat16, device=device)
    scale = head_dim**-0.5

    output = tokenspeed_mla.tokenspeed_mla_prefill(
        query=query,
        key=key,
        value=value,
        seq_lens=seq_lens,
        cum_seq_lens=cum_seq_lens,
        max_seq_len=int(seq_lens.max().item()),
        batch_size=seq_lens.numel(),
        softmax_scale=scale,
        is_causal=True,
        return_lse=False,
        enable_pdl=False,
    )
    expected = _reference_prefill(torch, query, key, value, seq_lens, scale)

    torch.testing.assert_close(output.float(), expected, rtol=2e-2, atol=3e-2)
