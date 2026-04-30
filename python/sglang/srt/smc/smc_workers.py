"""SMC Worker: A speculative-decoding-like algorithm with two independent models.

Draft model performs gamma+1 autoregressive decode steps.
Score model performs one extend forward pass on the drafted tokens.
Computes logprob difference between the two models per request.
No rejection - all drafted tokens are accepted.
"""

from __future__ import annotations

import dataclasses
import logging
import time
from typing import Optional

import torch

from sglang.srt.environ import envs
from sglang.srt.layers.dp_attention import (
    disable_dp_size,
    get_attention_dp_rank,
    get_attention_tp_group,
)
from sglang.srt.layers.quantization.fp8_utils import fp8_gemm_runner_backend_context
from sglang.srt.layers.moe.utils import (
    speculative_moe_a2a_backend_context,
    speculative_moe_backend_context,
)
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ModelWorkerBatch
from sglang.srt.managers.tp_worker import TpModelWorker
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode, ForwardMode
from sglang.srt.server_args import ServerArgs
from sglang.srt.smc.smc_debug_utils import append_smc_probe_record, smc_probe_enabled
from sglang.srt.smc.smc_info import SMCDraftInput
from sglang.srt.speculative.base_spec_worker import BaseSpecWorker
from sglang.srt.speculative.spec_utils import draft_tp_context
from sglang.srt.utils import empty_context

logger = logging.getLogger(__name__)


class SMCWorker(BaseSpecWorker):
    """SMC worker composing two independent TpModelWorkers (draft + score)."""

    def __init__(
        self,
        server_args: ServerArgs,
        gpu_id: int,
        tp_rank: int,
        dp_rank: Optional[int],
        moe_ep_rank: int,
        attn_cp_rank: int,
        moe_dp_rank: int,
        nccl_port: int,
        target_worker: TpModelWorker,
    ):
        self.server_args = server_args
        self.gpu_id = gpu_id
        self.tp_rank = tp_rank
        self.device = server_args.device
        self._target_worker = target_worker  # score model

        self.gamma = server_args.speculative_num_steps
        self.speculative_num_draft_tokens = self.gamma + 1
        self.smc_draft_temperature = server_args.smc_draft_temperature
        self.smc_target_temperature = max(
            float(server_args.smc_target_temperature), 1e-5
        )

        # Share req_to_token_pool, separate KV caches
        self.req_to_token_pool, self.token_to_kv_pool_allocator = (
            target_worker.get_memory_pool()
        )

        # Set class-level constant for KV allocation
        SMCDraftInput.ALLOC_LEN_PER_DECODE = self.speculative_num_draft_tokens

        # Override context length of draft model to match score model
        server_args.context_length = target_worker.model_runner.model_config.context_len

        # Do not capture cuda graph during TpModelWorker init -
        # we capture manually after the draft model is fully set up
        backup_disable_cuda_graph = server_args.disable_cuda_graph
        server_args.disable_cuda_graph = True
        self.draft_dp_context = (
            disable_dp_size if server_args.enable_dp_attention else empty_context
        )

        draft_init_tp_context = (
            draft_tp_context(get_attention_tp_group())
            if server_args.enable_dp_attention
            else empty_context()
        )

        # Create draft TpModelWorker - fully independent, no shared lm_head/embed.
        # Under DP attention, the target TP group is split into DP-local
        # attention-TP groups. The standalone GLM draft must be initialized under
        # that attention-TP group so its layer shards and KV rows match the local
        # draft batch rather than the target's global TP group.
        # The DeepSeek target keeps the process-level FP8 backend selected by
        # --fp8-gemm-backend. The GLM FP8 draft is initialized under a separate
        # backend override because FlashInfer TRT-LLM groupwise GEMM can assert
        # on GLM's projection shapes during long SMC decode on B200.
        draft_fp8_backend = envs.SGLANG_SMC_DRAFT_FP8_GEMM_BACKEND.get()
        with (
            self.draft_dp_context(),
            draft_init_tp_context,
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
            fp8_gemm_runner_backend_context(draft_fp8_backend),
        ):
            self._draft_worker = TpModelWorker(
                server_args=server_args,
                gpu_id=gpu_id,
                tp_rank=tp_rank,
                pp_rank=0,
                dp_rank=dp_rank,
                moe_ep_rank=moe_ep_rank,
                attn_cp_rank=attn_cp_rank,
                moe_dp_rank=moe_dp_rank,
                nccl_port=nccl_port,
                is_draft_worker=True,
                req_to_token_pool=self.req_to_token_pool,
                token_to_kv_pool_allocator=self.token_to_kv_pool_allocator,
                memory_pool_config=target_worker.model_runner.memory_pool_config,
            )

        self.draft_runner = self._draft_worker.model_runner
        self.score_runner = self._target_worker.model_runner
        self._pending_draft_prefill_batches: list[
            tuple[set[str], ModelWorkerBatch]
        ] = []

        # Restore cuda graph flag and capture graphs for the draft model
        server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_runner.server_args.disable_cuda_graph = backup_disable_cuda_graph
        self.draft_tp_context = (
            draft_tp_context if server_args.enable_dp_attention else empty_context
        )
        with (
            self.draft_dp_context(),
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            # Create multi-step draft attention backend (FA3, triton, etc.)
            # This pre-computes per-step attention metadata in one call,
            # avoiding per-step replay_prepare overhead.
            from sglang.srt.speculative.draft_utils import DraftBackendFactory

            # MultiStepBackend creates (num_steps - 1) sub-backends.
            # SMC needs gamma+1 forwards, so pass gamma+2 to get gamma+1 backends.
            factory = DraftBackendFactory(
                server_args,
                self.draft_runner,
                topk=1,
                speculative_num_steps=self.gamma + 2,
            )
            self.draft_attn_backend = factory.create_decode_backend()

            if not backup_disable_cuda_graph:
                try:
                    self.draft_runner.init_device_graphs()
                except Exception:
                    logger.exception(
                        "SMC draft CUDA graph capture failed; continuing with eager "
                        "draft execution while keeping target graphs enabled."
                    )
                    self.draft_runner.graph_runner = None

    @property
    def target_worker(self):
        return self._target_worker

    @property
    def draft_worker(self):
        return self._draft_worker

    @property
    def model_config(self):
        return self._target_worker.model_config

    @property
    def model_runner(self):
        return self._target_worker.model_runner

    def clear_cache_pool(self):
        pass

    @staticmethod
    def _copy_kv_pool_locs(pool, dst_locs: torch.Tensor, src_locs: torch.Tensor) -> None:
        if dst_locs.numel() == 0:
            return

        dst_locs = dst_locs.to(dtype=torch.int64)
        src_locs = src_locs.to(dtype=torch.int64)

        with torch.no_grad():
            if hasattr(pool, "k_buffer") and hasattr(pool, "v_buffer"):
                for k_buf, v_buf in zip(pool.k_buffer, pool.v_buffer):
                    k_buf[dst_locs] = k_buf[src_locs]
                    v_buf[dst_locs] = v_buf[src_locs]

            if hasattr(pool, "kv_buffer"):
                for kv_buf in pool.kv_buffer:
                    kv_buf[dst_locs] = kv_buf[src_locs]
                if hasattr(pool, "_tq_dirty"):
                    pool._tq_dirty = [True] * len(pool._tq_dirty)

            if hasattr(pool, "index_k_with_scale_buffer"):
                page_size = int(pool.page_size)
                page_pairs = torch.stack(
                    (dst_locs // page_size, src_locs // page_size), dim=1
                )
                page_pairs = torch.unique(page_pairs, dim=0)
                dst_pages = page_pairs[:, 0]
                src_pages = page_pairs[:, 1]
                for index_buf in pool.index_k_with_scale_buffer:
                    index_buf[dst_pages] = index_buf[src_pages]

    def copy_smc_kv_cache(self, dst_locs: torch.Tensor, src_locs: torch.Tensor) -> None:
        self._copy_kv_pool_locs(
            self._target_worker.model_runner.token_to_kv_pool,
            dst_locs,
            src_locs,
        )
        self._copy_kv_pool_locs(
            self._draft_worker.model_runner.token_to_kv_pool,
            dst_locs,
            src_locs,
        )

    def materialize_smc_parent_draft_prefix(self, req) -> None:
        """Replay deferred draft-prefix prefill before SMC particle creation."""
        rid = getattr(req, "rid", None)
        if rid is None or not self._pending_draft_prefill_batches:
            return

        to_run: list[tuple[set[str], ModelWorkerBatch]] = []
        remaining: list[tuple[set[str], ModelWorkerBatch]] = []
        for rids, draft_batch in self._pending_draft_prefill_batches:
            if rid in rids:
                to_run.append((rids, draft_batch))
            else:
                remaining.append((rids, draft_batch))
        self._pending_draft_prefill_batches = remaining

        if not to_run:
            return

        start_ns = self._probe_mark(
            "extend.deferred_draft_prefill",
            "start",
            sync=True,
            rid=rid,
            chunks=len(to_run),
        )
        with (
            self.draft_dp_context(),
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            for _, draft_batch in to_run:
                self._draft_worker.forward_batch_generation(draft_batch)
        end_ns = self._probe_mark(
            "extend.deferred_draft_prefill",
            "end",
            sync=True,
            rid=rid,
            chunks=len(to_run),
        )
        self._probe(
            "extend.deferred_draft_prefill",
            "duration",
            rid=rid,
            chunks=len(to_run),
            duration_ms=(end_ns - start_ns) / 1_000_000,
        )

    def _probe(self, phase: str, event: str, **fields) -> None:
        if not smc_probe_enabled:
            return
        try:
            dp_rank = get_attention_dp_rank()
        except AssertionError:
            dp_rank = None
        append_smc_probe_record(
            {
                "phase": phase,
                "event": event,
                "tp_rank": self.tp_rank,
                "dp_rank": dp_rank,
                **fields,
            }
        )

    def _probe_mark(self, phase: str, event: str, sync: bool = False, **fields) -> int:
        if smc_probe_enabled and sync:
            torch.cuda.synchronize()
        ts = time.perf_counter_ns()
        self._probe(phase, event, **fields)
        return ts

    @staticmethod
    def _clone_tensor(tensor: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if tensor is None:
            return None
        return tensor.detach().clone()

    def _clone_deferred_draft_batch(
        self, draft_batch: ModelWorkerBatch
    ) -> ModelWorkerBatch:
        token_ids_logprobs = draft_batch.token_ids_logprobs
        if token_ids_logprobs is not None:
            token_ids_logprobs = [
                list(x) if x is not None else None for x in token_ids_logprobs
            ]
        return dataclasses.replace(
            draft_batch,
            input_ids=self._clone_tensor(draft_batch.input_ids),
            req_pool_indices=self._clone_tensor(draft_batch.req_pool_indices),
            seq_lens=self._clone_tensor(draft_batch.seq_lens),
            out_cache_loc=self._clone_tensor(draft_batch.out_cache_loc),
            seq_lens_cpu=self._clone_tensor(draft_batch.seq_lens_cpu),
            top_logprobs_nums=list(draft_batch.top_logprobs_nums)
            if draft_batch.top_logprobs_nums is not None
            else None,
            token_ids_logprobs=token_ids_logprobs,
            extend_seq_lens=list(draft_batch.extend_seq_lens)
            if draft_batch.extend_seq_lens is not None
            else None,
            extend_prefix_lens=list(draft_batch.extend_prefix_lens)
            if draft_batch.extend_prefix_lens is not None
            else None,
            extend_logprob_start_lens=list(draft_batch.extend_logprob_start_lens)
            if draft_batch.extend_logprob_start_lens is not None
            else None,
            extend_input_logprob_token_ids=self._clone_tensor(
                draft_batch.extend_input_logprob_token_ids
            ),
            input_embeds=self._clone_tensor(draft_batch.input_embeds),
            replace_embeds=self._clone_tensor(draft_batch.replace_embeds),
            replace_positions=self._clone_tensor(draft_batch.replace_positions),
            orig_seq_lens=self._clone_tensor(draft_batch.orig_seq_lens),
            lora_ids=list(draft_batch.lora_ids)
            if draft_batch.lora_ids is not None
            else None,
            reqs=list(draft_batch.reqs) if draft_batch.reqs is not None else None,
        )

    def _queue_deferred_draft_prefill(self, draft_batch: ModelWorkerBatch) -> None:
        rids = {
            req.rid
            for req in (draft_batch.reqs or [])
            if req is not None and getattr(req, "rid", None) is not None
        }
        if not rids:
            return
        self._pending_draft_prefill_batches.append(
            (rids, self._clone_deferred_draft_batch(draft_batch))
        )
        self._probe(
            "extend.deferred_draft_prefill",
            "queued",
            rids=sorted(rids),
            pending_batches=len(self._pending_draft_prefill_batches),
            input_tokens=int(draft_batch.input_ids.numel())
            if draft_batch.input_ids is not None
            else None,
            out_cache_locs=int(draft_batch.out_cache_loc.numel())
            if draft_batch.out_cache_loc is not None
            else None,
        )

    @staticmethod
    def _should_defer_parent_draft_prefill(batch: ModelWorkerBatch) -> bool:
        return any(
            req is not None and getattr(req, "smc_particle_idx", None) is None
            for req in (batch.reqs or [])
        )

    # ------------------------------------------------------------------ #
    #  Main entry point - called by the scheduler
    # ------------------------------------------------------------------ #

    def forward_batch_generation(self, batch):
        # Non-overlap scheduler passes ScheduleBatch; convert to ModelWorkerBatch.
        from sglang.srt.managers.schedule_batch import ScheduleBatch

        if isinstance(batch, ScheduleBatch):
            batch = batch.get_model_worker_batch()

        if batch.forward_mode.is_extend() or batch.is_extend_in_batch:
            return self._forward_extend(batch)
        else:
            return self._forward_decode(batch)

    # ------------------------------------------------------------------ #
    #  EXTEND (prefill) - prefill both models, return score result
    # ------------------------------------------------------------------ #

    def _forward_extend(self, batch: ModelWorkerBatch):
        extend_start_ns = self._probe_mark(
            "extend",
            "start",
            batch_size=len(batch.seq_lens),
            forward_mode=str(batch.forward_mode),
        )
        # Capture the real DP-local extend size before the score model builds a
        # ForwardBatch. ForwardBatch.prepare_mlp_sync_batch mutates the shared
        # global_num_tokens list in-place for padding, but the draft prefill must
        # write exactly the unpadded local rows.
        local_draft_tokens = self._local_extend_num_tokens(batch)
        self._probe(
            "extend",
            "local_tokens",
            local_draft_tokens=local_draft_tokens,
            input_ids=int(batch.input_ids.numel()) if batch.input_ids is not None else None,
            out_cache_loc=int(batch.out_cache_loc.numel())
            if batch.out_cache_loc is not None
            else None,
            global_num_tokens=batch.global_num_tokens,
        )

        # 1. Score model prefill
        score_start_ns = self._probe_mark("extend.score_prefill", "start", sync=True)
        score_result = self._target_worker.forward_batch_generation(batch)
        score_end_ns = self._probe_mark("extend.score_prefill", "end", sync=True)
        self._probe(
            "extend.score_prefill",
            "duration",
            duration_ms=(score_end_ns - score_start_ns) / 1_000_000,
        )

        if len(batch.seq_lens) == 0 or local_draft_tokens == 0:
            score_result.next_token_ids = torch.empty(
                0, dtype=torch.int64, device=self.device
            )
            score_result.next_draft_input = SMCDraftInput.create_idle_input(
                self.device
            )
            score_result.accept_lens = torch.empty(
                0, dtype=torch.int32, device=self.device
            )
            self._probe(
                "extend",
                "end_idle",
                duration_ms=(time.perf_counter_ns() - extend_start_ns) / 1_000_000,
            )
            return score_result

        # 2. Draft model prefill - for streaming parents, defer external draft
        # prompt KV materialization until after the target first token is sent.
        draft_batch = self._make_clean_batch(
            batch, input_token_count=local_draft_tokens
        )
        bs = len(draft_batch.seq_lens)
        if self._should_defer_parent_draft_prefill(batch):
            self._queue_deferred_draft_prefill(draft_batch)
            score_result.next_draft_input = SMCDraftInput(
                verified_id=score_result.next_token_ids,
                new_seq_lens=draft_batch.seq_lens,
                num_tokens_per_req=self.speculative_num_draft_tokens,
            )
            score_result.accept_lens = torch.zeros(
                bs, dtype=torch.int32, device=self.device
            )
            self._probe(
                "extend",
                "end_deferred_draft_prefill",
                batch_size=bs,
                duration_ms=(time.perf_counter_ns() - extend_start_ns) / 1_000_000,
            )
            return score_result

        with (
            self.draft_dp_context(),
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            draft_start_ns = self._probe_mark(
                "extend.draft_prefill",
                "start",
                sync=True,
                batch_size=bs,
                local_draft_tokens=local_draft_tokens,
            )
            draft_result = self._draft_worker.forward_batch_generation(draft_batch)
            draft_end_ns = self._probe_mark("extend.draft_prefill", "end", sync=True)
            self._probe(
                "extend.draft_prefill",
                "duration",
                duration_ms=(draft_end_ns - draft_start_ns) / 1_000_000,
            )

        # 3. Use draft model's sampled token as verified_id (draft drives generation)
        score_result.next_token_ids = draft_result.next_token_ids

        # x0 is sampled but its KV is NOT written during prefill.
        # Keep seq_lens at prompt_len (not +1) so the first decode cycle
        # starts at the correct position and writes x0's KV.
        # batch.seq_lens is a schedule-stream tensor - safe without verify_done.
        score_result.next_draft_input = SMCDraftInput(
            verified_id=draft_result.next_token_ids,
            new_seq_lens=draft_batch.seq_lens,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )
        score_result.accept_lens = torch.zeros(bs, dtype=torch.int32, device=self.device)
        self._probe(
            "extend",
            "end",
            batch_size=bs,
            duration_ms=(time.perf_counter_ns() - extend_start_ns) / 1_000_000,
        )
        return score_result

    # ------------------------------------------------------------------ #
    #  DECODE - draft AR -> score extend -> logprob diff
    # ------------------------------------------------------------------ #

    def _forward_decode(self, batch: ModelWorkerBatch):
        if batch.forward_mode.is_idle():
            return self._forward_idle(batch)

        decode_start_ns = self._probe_mark(
            "decode",
            "start",
            forward_mode=str(batch.forward_mode),
            batch_size=len(batch.req_pool_indices)
            if batch.req_pool_indices is not None
            else None,
        )

        # record_stream on tensors created on the previous forward stream
        # that we'll read on this forward stream. seq_lens is now advanced
        # on the schedule stream (in prepare_for_decode), so it's protected
        # by forward_stream.wait_stream(schedule_stream) - no record_stream needed.
        current_stream = torch.get_device_module(self.device).current_stream()
        if batch.req_pool_indices is not None:
            batch.req_pool_indices.record_stream(current_stream)

        draft_input: SMCDraftInput = batch.spec_info
        if draft_input.verified_id is not None:
            draft_input.verified_id.record_stream(current_stream)

        bs = len(draft_input._orig_seq_lens)
        if bs == 0:
            return self._forward_idle(batch)
        gamma = self.gamma
        self._probe(
            "decode",
            "state",
            batch_size=bs,
            gamma=gamma,
            orig_seq_lens_sum=int(draft_input._orig_seq_lens_sum),
            num_tokens_per_req=draft_input.num_tokens_per_req,
        )

        with (
            self.draft_dp_context(),
            self.draft_tp_context(self.draft_runner.tp_group),
            speculative_moe_backend_context(),
            speculative_moe_a2a_backend_context(),
        ):
            # ---- 1. Prepare draft: assign cache locs, create ForwardBatch ----
            prepare_start_ns = self._probe_mark("decode.prepare_draft", "start", sync=True)
            draft_fb, can_cuda_graph, cache_locs, all_positions, all_seq_lens = (
                draft_input.prepare_for_draft(
                    self.req_to_token_pool,
                    batch,
                    getattr(self.draft_runner, "graph_runner", None),
                    self.draft_runner,
                    gamma,
                )
            )
            prepare_end_ns = self._probe_mark("decode.prepare_draft", "end", sync=True)
            self._probe(
                "decode.prepare_draft",
                "duration",
                can_cuda_graph=bool(can_cuda_graph),
                duration_ms=(prepare_end_ns - prepare_start_ns) / 1_000_000,
            )
            # The reference SMC path uses can_cuda_graph as the single decision
            # for graph replay vs. eager multi-step draft decode. Newer SGLang
            # ModelRunner.forward probes graph_runner again, so carry the SMC
            # decision on the ForwardBatch to keep local DP draft batches eager
            # when they intentionally lack target-global MLP token metadata.
            draft_fb.disable_graph_runner = not can_cuda_graph

            # ---- 2. Draft AR: gamma+1 decode steps ----
            # Initialize multi-step attention metadata ONCE for all steps.
            # Each sub-backend uses its speculative_step_id to compute the
            # correct per-step cache_seqlens and page_table.
            use_multistep = (
                self.draft_attn_backend is not None and not can_cuda_graph
            )
            if use_multistep and not draft_fb.forward_mode.is_idle():
                meta_start_ns = self._probe_mark(
                    "decode.draft_metadata", "start", sync=True
                )
                # Set spec_info and base prefix seq_lens for multi-step metadata init.
                # Each sub-backend adds its speculative_step_id to get per-step values.
                draft_fb.spec_info = draft_input
                draft_fb.seq_lens = draft_input._orig_seq_lens
                draft_fb.seq_lens_sum = draft_input._orig_seq_lens_sum
                draft_fb.seq_lens_cpu = draft_input._orig_seq_lens_cpu
                self.draft_attn_backend.init_forward_metadata(draft_fb)
                meta_end_ns = self._probe_mark("decode.draft_metadata", "end", sync=True)
                self._probe(
                    "decode.draft_metadata",
                    "duration",
                    duration_ms=(meta_end_ns - meta_start_ns) / 1_000_000,
                )

            x0 = draft_input.verified_id  # (bs,)
            all_tokens = [x0]
            draft_logprobs = []
            current_ids = x0

            for step in range(gamma + 1):
                step_start_ns = self._probe_mark(
                    "decode.draft_step",
                    "start",
                    sync=True,
                    step=step,
                    use_multistep=bool(use_multistep),
                    can_cuda_graph=bool(can_cuda_graph),
                )
                # Update only the fields that change per step - no GPU->CPU sync
                draft_fb.input_ids = current_ids
                draft_fb.positions = all_positions[:, step].contiguous()
                draft_fb.out_cache_loc = cache_locs[:, step].contiguous()

                if use_multistep:
                    # Swap to this step's pre-initialized attention backend
                    draft_fb.attn_backend = self.draft_attn_backend.attn_backends[step]
                    draft_out = self.draft_runner.forward(
                        draft_fb, skip_attn_backend_init=True
                    )
                else:
                    # Fallback: per-step forward with full metadata init
                    draft_fb.seq_lens = all_seq_lens[:, step].contiguous()
                    draft_fb.seq_lens_sum = (
                        draft_input._orig_seq_lens_sum + bs * (step + 1)
                    )
                    draft_fb.seq_lens_cpu = (
                        draft_input._orig_seq_lens_cpu + (step + 1)
                    )
                    draft_out = self.draft_runner.forward(draft_fb)
                logits = draft_out.logits_output.next_token_logits  # (bs, vocab)

                # Sample next token
                scaled_logits = logits / self.smc_draft_temperature
                log_probs = torch.log_softmax(scaled_logits, dim=-1)
                if self.smc_draft_temperature > 0:
                    next_token = torch.multinomial(
                        log_probs.exp(), num_samples=1
                    ).squeeze(-1)
                else:
                    next_token = torch.argmax(logits, dim=-1)

                # Collect logprob (only first gamma steps)
                if step < gamma:
                    token_logprob = log_probs.gather(
                        1, next_token.unsqueeze(1)
                    ).squeeze(1)
                    draft_logprobs.append(token_logprob)

                all_tokens.append(next_token)
                current_ids = next_token
                step_end_ns = self._probe_mark(
                    "decode.draft_step", "end", sync=True, step=step
                )
                self._probe(
                    "decode.draft_step",
                    "duration",
                    step=step,
                    duration_ms=(step_end_ns - step_start_ns) / 1_000_000,
                )

            # draft_logprobs: gamma entries - logprob(x1),...,logprob(x(gamma))
            draft_logprobs_stacked = torch.stack(draft_logprobs, dim=1)  # (bs, gamma)

        # ---- 3. Score verify: [x0, ..., x(gamma)] via TARGET_VERIFY ----
        verify_prepare_start_ns = self._probe_mark(
            "decode.prepare_verify", "start", sync=True
        )
        verify_forward_batch, can_run_cuda_graph = draft_input.prepare_for_verify(
            self.req_to_token_pool,
            batch,
            self._target_worker,
            all_tokens,
            cache_locs,
            gamma,
        )
        verify_prepare_end_ns = self._probe_mark(
            "decode.prepare_verify", "end", sync=True
        )
        self._probe(
            "decode.prepare_verify",
            "duration",
            can_run_cuda_graph=bool(can_run_cuda_graph),
            duration_ms=(verify_prepare_end_ns - verify_prepare_start_ns) / 1_000_000,
        )

        verify_start_ns = self._probe_mark("decode.target_verify", "start", sync=True)
        score_result = self._target_worker.forward_batch_generation(
            model_worker_batch=None,
            forward_batch=verify_forward_batch,
            is_verify=True,
            skip_attn_backend_init=not can_run_cuda_graph,
        )
        verify_end_ns = self._probe_mark("decode.target_verify", "end", sync=True)
        self._probe(
            "decode.target_verify",
            "duration",
            duration_ms=(verify_end_ns - verify_start_ns) / 1_000_000,
        )

        # ---- 4. Extract score logprobs from logits (first gamma only) ----
        post_start_ns = self._probe_mark("decode.postprocess", "start", sync=True)
        # TARGET_VERIFY returns raw logits; compute logprobs directly
        score_logits = score_result.logits_output.next_token_logits  # (bs*(gamma+1), vocab)
        expected_rows = bs * (gamma + 1)
        assert score_logits.shape[0] == expected_rows, (
            f"TARGET_VERIFY logits truncated: got {score_logits.shape[0]} rows, "
            f"expected {expected_rows} (bs={bs}, gamma+1={gamma+1}, "
            f"cuda_graph={can_run_cuda_graph})"
        )
        # NOTE(CCC): Check whether we need a logprob for now.
        score_log_probs = torch.log_softmax(score_logits, dim=-1)
        score_log_probs = score_log_probs.reshape(bs, gamma + 1, -1)
        # Gather logprobs for tokens [x1, ..., x_gamma] at positions [0, ..., gamma-1]
        target_tokens = torch.stack(all_tokens[1 : gamma + 1], dim=1)  # (bs, gamma)
        score_logprobs_stacked = score_log_probs[:, :gamma, :].gather(
            2, target_tokens.unsqueeze(2)
        ).squeeze(2)  # (bs, gamma)

        # ---- 5. Compute logprob diff ----
        logprob_diff = (score_logprobs_stacked - draft_logprobs_stacked).sum(dim=1)

        # ---- 6. Sample bonus token from tempered target logits ----
        bonus_logits = score_logits.reshape(bs, gamma + 1, -1)[:, -1, :]
        bonus_log_probs = torch.log_softmax(
            bonus_logits / self.smc_target_temperature, dim=-1
        )
        bonus = torch.multinomial(bonus_log_probs.exp(), num_samples=1).squeeze(-1)

        # ---- 7. Build output ----
        # Output: [x1, ..., x_r, bonus] - bonus replaces overdraft x_{r+1}
        output_token_ids = torch.stack(
            all_tokens[1 : gamma + 1] + [bonus], dim=1
        )  # (bs, gamma+1)
        next_token_ids = output_token_ids.reshape(-1)
        accept_lens = torch.full((bs,), gamma + 1, dtype=torch.int32, device=self.device)
        next_verified_id = bonus

        # record_stream on output tensors: created on forward stream,
        # read by scheduler via copy_to_cpu on schedule stream.
        next_token_ids.record_stream(current_stream)
        accept_lens.record_stream(current_stream)
        next_verified_id.record_stream(current_stream)
        logprob_diff.record_stream(current_stream)
        post_end_ns = self._probe_mark("decode.postprocess", "end", sync=True)
        self._probe(
            "decode.postprocess",
            "duration",
            duration_ms=(post_end_ns - post_start_ns) / 1_000_000,
        )

        # Use schedule-stream new_seq_lens (stashed in prepare_for_decode).
        # No forward-stream tensor crosses to the scheduler -> no verify_done needed.
        next_draft_input = SMCDraftInput(
            verified_id=next_verified_id,
            new_seq_lens=draft_input._new_seq_lens,
            logprob_diff=logprob_diff,
            num_tokens_per_req=self.speculative_num_draft_tokens,
        )

        self._probe(
            "decode",
            "end",
            batch_size=bs,
            duration_ms=(time.perf_counter_ns() - decode_start_ns) / 1_000_000,
        )
        return GenerationBatchResult(
            logits_output=score_result.logits_output,
            next_token_ids=next_token_ids,
            accept_lens=accept_lens,
            next_draft_input=next_draft_input,
            logprob_diff=logprob_diff,
            can_run_cuda_graph=can_run_cuda_graph,
        )

    def _forward_idle(self, batch: ModelWorkerBatch):
        return GenerationBatchResult(
            logits_output=LogitsProcessorOutput(next_token_logits=None),
            next_token_ids=torch.empty(0, dtype=torch.int64, device=self.device),
            accept_lens=torch.empty(0, dtype=torch.int32, device=self.device),
            next_draft_input=SMCDraftInput.create_idle_input(self.device),
        )

    @staticmethod
    def _local_extend_num_tokens(batch: ModelWorkerBatch) -> int:
        if batch.forward_mode.is_idle():
            return 0

        if batch.global_num_tokens is not None:
            global_num_tokens = batch.global_num_tokens
            dp_rank = 0
            if len(global_num_tokens) > 1:
                try:
                    dp_rank = get_attention_dp_rank()
                except AssertionError:
                    dp_rank = 0
            if 0 <= dp_rank < len(global_num_tokens):
                return int(global_num_tokens[dp_rank])

        if batch.extend_num_tokens is not None:
            return int(batch.extend_num_tokens)

        if batch.input_ids is None or batch.forward_mode.is_idle():
            if batch.input_embeds is None or batch.forward_mode.is_idle():
                return 0
            return int(batch.input_embeds.shape[0])
        return int(batch.input_ids.numel())

    @staticmethod
    def _normalize_extend_lens(
        batch: ModelWorkerBatch, local_num_tokens: int
    ) -> tuple[list[int], list[int], list[int]]:
        if batch.extend_seq_lens is not None:
            extend_seq_lens = [int(x) for x in batch.extend_seq_lens]
        else:
            extend_seq_lens = [local_num_tokens]

        batch_size = len(batch.seq_lens)
        if len(extend_seq_lens) != batch_size:
            extend_seq_lens = extend_seq_lens[:batch_size]
            extend_seq_lens.extend([1] * (batch_size - len(extend_seq_lens)))

        if sum(extend_seq_lens) != local_num_tokens and batch_size > 0:
            # Preserve one generated-logit row per local request, then assign
            # any remaining prompt tokens according to the original request order.
            if local_num_tokens >= batch_size:
                remaining = local_num_tokens - batch_size
                normalized = [1] * batch_size
                for i, old_len in enumerate(extend_seq_lens):
                    if remaining <= 0:
                        break
                    take = min(max(old_len - 1, 0), remaining)
                    normalized[i] += take
                    remaining -= take
                if remaining > 0:
                    normalized[-1] += remaining
                extend_seq_lens = normalized
            else:
                extend_seq_lens = [1] * local_num_tokens

        if batch.extend_prefix_lens is not None:
            extend_prefix_lens = [int(x) for x in batch.extend_prefix_lens]
        else:
            extend_prefix_lens = batch.seq_lens_cpu.tolist()
        extend_prefix_lens = extend_prefix_lens[: len(extend_seq_lens)]

        if batch.extend_logprob_start_lens is not None:
            extend_logprob_start_lens = [
                min(int(x), extend_seq_lens[i])
                for i, x in enumerate(batch.extend_logprob_start_lens[: len(extend_seq_lens)])
            ]
        else:
            extend_logprob_start_lens = [0] * len(extend_seq_lens)

        return extend_seq_lens, extend_prefix_lens, extend_logprob_start_lens

    def _make_clean_batch(
        self, batch: ModelWorkerBatch, input_token_count: Optional[int] = None
    ) -> ModelWorkerBatch:
        """Create a copy of the batch with no spec_info (for draft model)."""
        if input_token_count is None:
            input_token_count = self._local_extend_num_tokens(batch)
        extend_seq_lens, extend_prefix_lens, extend_logprob_start_lens = (
            self._normalize_extend_lens(batch, input_token_count)
        )
        local_batch_size = len(extend_seq_lens)
        input_ids = batch.input_ids
        if (
            input_ids is not None
            and input_token_count
            and input_ids.numel() != input_token_count
        ):
            input_ids = input_ids[:input_token_count].contiguous()
            logger.warning(
                "SMC draft prefill trimmed target-padded input_ids from %s to %s",
                batch.input_ids.numel(),
                input_token_count,
            )

        input_embeds = batch.input_embeds
        if (
            input_embeds is not None
            and input_token_count
            and input_embeds.shape[0] != input_token_count
        ):
            input_embeds = input_embeds[:input_token_count].contiguous()

        out_cache_loc = batch.out_cache_loc
        if (
            out_cache_loc is not None
            and input_token_count
            and out_cache_loc.numel() != input_token_count
        ):
            # DP attention may carry target-side padded cache locations into the
            # ModelWorkerBatch. The independent draft model writes only local
            # input tokens, so keep the draft KV store shape local.
            out_cache_loc = out_cache_loc[:input_token_count].contiguous()
            logger.warning(
                "SMC draft prefill trimmed target-padded out_cache_loc from %s to %s",
                batch.out_cache_loc.numel(),
                input_token_count,
            )

        seq_lens = batch.seq_lens
        if seq_lens is not None and seq_lens.shape[0] != local_batch_size:
            seq_lens = seq_lens[:local_batch_size].contiguous()
        seq_lens_cpu = batch.seq_lens_cpu
        if seq_lens_cpu is not None and seq_lens_cpu.shape[0] != local_batch_size:
            seq_lens_cpu = seq_lens_cpu[:local_batch_size].contiguous()
        req_pool_indices = batch.req_pool_indices
        if (
            req_pool_indices is not None
            and req_pool_indices.shape[0] != local_batch_size
        ):
            req_pool_indices = req_pool_indices[:local_batch_size].contiguous()
        reqs = batch.reqs
        if reqs is not None and len(reqs) != local_batch_size:
            reqs = reqs[:local_batch_size]
        lora_ids = batch.lora_ids
        if lora_ids is not None and len(lora_ids) != local_batch_size:
            lora_ids = lora_ids[:local_batch_size]

        return dataclasses.replace(
            batch,
            forward_mode=ForwardMode.IDLE
            if batch.forward_mode.is_idle()
            else ForwardMode.EXTEND,
            spec_info=None,
            capture_hidden_mode=CaptureHiddenMode.NULL,
            input_ids=input_ids,
            input_embeds=input_embeds,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_sum=int(seq_lens_cpu.sum().item())
            if seq_lens_cpu is not None
            else batch.seq_lens_sum,
            out_cache_loc=out_cache_loc,
            extend_num_tokens=input_token_count,
            extend_seq_lens=extend_seq_lens,
            extend_prefix_lens=extend_prefix_lens,
            extend_logprob_start_lens=extend_logprob_start_lens,
            global_num_tokens=None,
            global_num_tokens_for_logprob=None,
            is_extend_in_batch=False,
            all_extend_in_batch=False,
            can_run_dp_cuda_graph=False,
            global_forward_mode=None,
            lora_ids=lora_ids,
            reqs=reqs,
        )
