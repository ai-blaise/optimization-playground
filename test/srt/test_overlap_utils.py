from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
overlap_utils = pytest.importorskip("sglang.srt.managers.overlap_utils")
FutureIndices = overlap_utils.FutureIndices
FutureMap = overlap_utils.FutureMap


class _NoSpecAlgo:
    def is_none(self):
        return True

    def is_smc(self):
        return False

    def is_some(self):
        return False


def test_future_map_accepts_wrapped_future_indices():
    future_map = FutureMap.__new__(FutureMap)
    future_map.device = torch.device("cpu")
    future_map.spec_algo = _NoSpecAlgo()
    future_map.output_tokens_buf = torch.zeros((5,), dtype=torch.int64)
    future_map.new_seq_lens_buf = torch.zeros((5,), dtype=torch.int64)
    future_map.publish_ready = None
    future_map.fwd_prepare_d2h_stream = None

    indices = FutureIndices(indices=torch.tensor([1, 3], dtype=torch.int64))

    future_map.publish(indices, torch.tensor([7, 9], dtype=torch.int32))
    future_map.stash(indices, torch.tensor([101, 103], dtype=torch.int32))

    assert future_map.new_seq_lens_buf.tolist() == [0, 7, 0, 9, 0]
    assert future_map.output_tokens_buf.tolist() == [0, 101, 0, 103, 0]

    batch = SimpleNamespace(
        input_ids=None,
        req_pool_indices_cpu=indices.indices,
        spec_info=SimpleNamespace(future_indices=indices),
    )
    future_map.resolve_seq_lens_cpu(batch)
    future_map.set_input_ids_sentinel(batch, indices)

    assert batch.seq_lens.tolist() == [7, 9]
    assert batch.seq_lens_cpu.tolist() == [7, 9]
    assert batch.seq_lens_sum == 16
    assert batch.input_ids.tolist() == [-1, -3]
