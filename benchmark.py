import argparse
import hashlib
import inspect
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
from typing import Literal
from types import SimpleNamespace

import distributed as dist
import numpy as np
import torch
from loguru import logger
from model import (
    DFlashDraftModel,
    extract_context_feature,
    load_and_process_dataset,
    sample,
)
from rich import print
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache


@contextmanager
def nvtx_range(name: str):
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


_CURRENT_PROFILE_PHASE: str | None = None
_PACKED_VERIFY_CAPABILITY_CACHE: dict[int, bool] = {}
_PACKED_VERIFY_FALLBACK_WARNED_TARGETS: set[int] = set()
_PACKED_VERIFY_PATH_STATS = {
    "packed": {"tokens": 0, "time": 0.0},
    "fallback": {"tokens": 0, "time": 0.0},
}
STAGE_C_PACKED_METADATA_REQUIREMENT = (
    "stage_c_full_speculative packed verify path requires target.forward to accept "
    "slot_mapping/context_lens/block_tables kwargs (or **kwargs). "
    "If unsupported, benchmark.py auto-falls back to stage_b_target_only compatibility path."
)


@contextmanager
def profile_phase(name: str | None):
    global _CURRENT_PROFILE_PHASE
    prev = _CURRENT_PROFILE_PHASE
    _CURRENT_PROFILE_PHASE = name
    try:
        yield
    finally:
        _CURRENT_PROFILE_PHASE = prev


class ModuleRangeProfiler:
    def __init__(self, root: torch.nn.Module):
        self._handles = []
        for module_name, module in root.named_modules():
            kind = self._classify_kind(module_name, module)
            if kind is None:
                continue
            self._handles.append(
                module.register_forward_pre_hook(self._make_pre_hook(kind), with_kwargs=True)
            )
            self._handles.append(module.register_forward_hook(self._post_hook, with_kwargs=True))

    @staticmethod
    def _classify_kind(module_name: str, module: torch.nn.Module) -> str | None:
        lowered = f"{module_name}.{module.__class__.__name__}".lower()
        if any(
            token in lowered
            for token in [
                "q_proj",
                "k_proj",
                "v_proj",
                "qkv",
                "wqkv",
                "query_key_value",
                "c_attn",
            ]
        ):
            return "qkv"
        if any(token in lowered for token in ["self_attn", "attention", ".attn", "attn"]):
            return "attn"
        if any(token in lowered for token in ["mlp", "ffn", "feed_forward"]):
            return "ffn"
        return None

    @staticmethod
    def _make_pre_hook(kind: str):
        def _pre_hook(module, args, kwargs):
            if not torch.cuda.is_available():
                module._dflash_nvtx_push = False
                return
            if _CURRENT_PROFILE_PHASE is None:
                module._dflash_nvtx_push = False
                return
            torch.cuda.nvtx.range_push(f"{_CURRENT_PROFILE_PHASE}.{kind}")
            module._dflash_nvtx_push = True

        return _pre_hook

    @staticmethod
    def _post_hook(module, args, kwargs, output):
        if not torch.cuda.is_available():
            return output
        pushed = getattr(module, "_dflash_nvtx_push", False)
        if pushed:
            torch.cuda.nvtx.range_pop()
            module._dflash_nvtx_push = False
        return output


def cuda_time() -> float:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


@dataclass
class BlockMapping:
    logical_start: int
    logical_end: int
    physical_block_id: int
    content_hash: str


class BlockManager:
    def __init__(self, block_size: int):
        if block_size <= 0:
            raise ValueError("block_size must be positive")
        self.block_size = block_size
        self._next_block_id = 0
        self._free_block_ids: list[int] = []
        self._block_refcount: dict[int, int] = {}
        self._prefix_hash_to_block: dict[str, int] = {}
        self._seq_registry: dict[int, "Sequence"] = {}

    def register_sequence(self, sequence: "Sequence") -> None:
        self._seq_registry[sequence.seq_id] = sequence

    @staticmethod
    def hash_tokens(token_ids: list[int], logical_start: int) -> str:
        payload = f"{logical_start}|" + " ".join(str(token) for token in token_ids)
        return hashlib.sha1(payload.encode("utf-8")).hexdigest()

    def _allocate_physical_block(self) -> int:
        if self._free_block_ids:
            block_id = self._free_block_ids.pop()
        else:
            block_id = self._next_block_id
            self._next_block_id += 1
        self._block_refcount.setdefault(block_id, 0)
        return block_id

    def retain(self, block_id: int) -> None:
        if block_id not in self._block_refcount:
            self._block_refcount[block_id] = 0
        self._block_refcount[block_id] += 1

    def release(self, block_id: int) -> None:
        if block_id not in self._block_refcount:
            raise ValueError(f"attempting to release unknown block_id={block_id}")
        self._block_refcount[block_id] -= 1
        if self._block_refcount[block_id] < 0:
            raise ValueError(f"block_id={block_id} refcount became negative")
        if self._block_refcount[block_id] == 0:
            self._free_block_ids.append(block_id)

    def acquire_block(
        self,
        token_ids: list[int],
        logical_start: int,
        allow_prefix_reuse: bool = True,
    ) -> tuple[int, str, bool]:
        content_hash = self.hash_tokens(token_ids, logical_start)
        cache_hit = allow_prefix_reuse and content_hash in self._prefix_hash_to_block
        if cache_hit:
            block_id = self._prefix_hash_to_block[content_hash]
        else:
            block_id = self._allocate_physical_block()
            self._prefix_hash_to_block[content_hash] = block_id
        self.retain(block_id)
        return block_id, content_hash, cache_hit

    def rollback(self, seq_id: int, rollback_to: int) -> None:
        sequence = self._seq_registry.get(seq_id)
        if sequence is None:
            raise ValueError(f"unknown seq_id={seq_id}")
        if rollback_to > sequence.num_cached_tokens:
            raise ValueError("rollback target cannot exceed current num_cached_tokens")
        while sequence.block_table and sequence.block_table[-1].logical_start >= rollback_to:
            removed = sequence.block_table.pop()
            self.release(removed.physical_block_id)
        sequence.num_cached_tokens = rollback_to
        sync_sequence_blocks(sequence, self)

    def rollback_sequence(self, sequence: "Sequence", new_num_cached_tokens: int) -> None:
        self.register_sequence(sequence)
        self.rollback(sequence.seq_id, new_num_cached_tokens)

    def check_consistency(self, sequence: "Sequence") -> None:
        mapped_tokens = sum(mapping.logical_end - mapping.logical_start for mapping in sequence.block_table)
        if mapped_tokens != sequence.num_cached_tokens:
            raise ValueError(
                f"sequence {sequence.seq_id} cache mismatch: num_cached_tokens={sequence.num_cached_tokens}, mapped_tokens={mapped_tokens}"
            )


def sync_sequence_blocks(sequence: "Sequence", block_manager: BlockManager) -> None:
    target_num_tokens = sequence.num_cached_tokens
    block_size = block_manager.block_size

    expected_mappings: list[tuple[int, int, list[int]]] = []
    for block_start in range(0, target_num_tokens, block_size):
        block_end = min(block_start + block_size, target_num_tokens)
        expected_mappings.append((block_start, block_end, sequence.token_ids[block_start:block_end]))

    longest_prefix = 0
    for mapping, expected in zip(sequence.block_table, expected_mappings):
        expected_start, expected_end, expected_tokens = expected
        if mapping.logical_start != expected_start or mapping.logical_end != expected_end:
            break
        if mapping.content_hash != block_manager.hash_tokens(expected_tokens, expected_start):
            break
        longest_prefix += 1

    for stale_mapping in sequence.block_table[longest_prefix:]:
        block_manager.release(stale_mapping.physical_block_id)
    sequence.block_table = sequence.block_table[:longest_prefix]

    for logical_start, logical_end, block_tokens in expected_mappings[longest_prefix:]:
        physical_block_id, content_hash, _ = block_manager.acquire_block(
            block_tokens,
            logical_start=logical_start,
            allow_prefix_reuse=True,
        )
        sequence.block_table.append(
            BlockMapping(
                logical_start=logical_start,
                logical_end=logical_end,
                physical_block_id=physical_block_id,
                content_hash=content_hash,
            )
        )

    block_manager.check_consistency(sequence)


@dataclass
class Sequence:
    seq_id: int
    token_ids: list[int]
    num_cached_tokens: int
    block_table: list[BlockMapping]
    pre_verify: bool
    num_acc_tokens: int
    finished: bool
    pending_kv_append: list[int]


class Scheduler:
    def __init__(self, sequences: list[Sequence], block_manager: BlockManager | None = None):
        self.waiting: list[Sequence] = list(sequences)
        self.running: list[Sequence] = []
        self.finished: list[Sequence] = []
        self._seq_registry: dict[int, Sequence] = {seq.seq_id: seq for seq in sequences}
        self.block_manager = block_manager
        self._draft_transactions: dict[int, dict[str, int | list[int]]] = {}

    def has_pending(self) -> bool:
        return len(self.waiting) > 0 or len(self.running) > 0

    def schedule_next_batch(
        self,
        max_batch_tokens: int | None = None,
        max_batch_size: int | None = None,
        return_grouped: bool = False,
    ) -> list[Sequence] | list[list[Sequence]]:
        while self.waiting:
            self.running.append(self.waiting.pop(0))

        if not self.running:
            return []

        self.running.sort(key=lambda seq: (seq.num_cached_tokens, seq.seq_id))
        selected_groups: dict[int, list[Sequence]] = {}
        selected_tokens = 0
        for seq in self.running:
            if seq.finished:
                continue
            if max_batch_size is not None and selected_tokens >= max_batch_size:
                break
            if max_batch_tokens is not None and selected_tokens + 1 > max_batch_tokens:
                break
            selected_groups.setdefault(seq.num_cached_tokens, []).append(seq)
            selected_tokens += 1

        if not selected_groups:
            return []
        grouped_batches = [selected_groups[pos] for pos in sorted(selected_groups)]
        if return_grouped:
            return grouped_batches
        return grouped_batches[0]

    def mark_finished(self, seq: Sequence) -> None:
        seq.finished = True
        self.running = [running_seq for running_seq in self.running if running_seq.seq_id != seq.seq_id]
        self.finished.append(seq)

    def append_draft_tokens(self, seq_id: int, draft_tokens: list[int]) -> None:
        if self.block_manager is None:
            raise ValueError("block_manager is required for draft transactions")
        seq = self._seq_registry[seq_id]
        draft_start = seq.num_cached_tokens
        seq.token_ids = seq.token_ids[:draft_start] + draft_tokens
        seq.num_cached_tokens = draft_start + len(draft_tokens)
        sync_sequence_blocks(seq, self.block_manager)
        occupied_blocks = [
            mapping.physical_block_id
            for mapping in seq.block_table
            if mapping.logical_start >= draft_start
        ]
        self._draft_transactions[seq_id] = {
            "start": draft_start,
            "num_tokens": len(draft_tokens),
            "num_blocks": len(occupied_blocks),
        }

    def consume_draft_transaction(self, seq_id: int) -> dict[str, int | list[int]]:
        txn = self._draft_transactions.pop(seq_id, None)
        if txn is None:
            raise ValueError(f"no draft transaction for seq_id={seq_id}")
        return txn

    def rollback(self, seq_id: int, rollback_to: int) -> None:
        if self.block_manager is None:
            raise ValueError("block_manager is required for rollback")
        seq = self._seq_registry.get(seq_id)
        if seq is None:
            raise ValueError(f"unknown seq_id={seq_id}")
        seq.token_ids = seq.token_ids[:rollback_to]
        self.block_manager.rollback(seq_id, rollback_to)


@dataclass
class BatchDecodeState:
    output_ids: torch.Tensor
    input_lengths: torch.Tensor
    start_pos: torch.Tensor
    finished_mask: torch.Tensor
    active_indices: torch.Tensor
    acceptance_lengths_per_row: list[list[int]]
    max_length: int
    position_ids: torch.Tensor
    past_key_values_target: DynamicCache
    past_key_values_draft: DynamicCache
    target_hidden: torch.Tensor | None
    decode_start: float
    draft_prefill: bool
    active_batch_size_trace: list[int]
    grouped_batch_count_trace: list[int]
    grouped_batch_sizes_trace: list[list[int]]


BatchedDecodeMode = Literal[
    "legacy",
    "stage_a_prefill_only",
    "stage_b_target_only",
    "stage_c_full_speculative",
]


def gather_active_rows(cache: DynamicCache, active_indices: torch.Tensor) -> DynamicCache:
    gathered_cache = DynamicCache()
    key_cache = getattr(cache, "key_cache", [])
    value_cache = getattr(cache, "value_cache", [])
    gathered_cache.key_cache = [key.index_select(0, active_indices) for key in key_cache]
    gathered_cache.value_cache = [value.index_select(0, active_indices) for value in value_cache]
    if hasattr(cache, "_seen_tokens"):
        gathered_cache._seen_tokens = cache._seen_tokens
    return gathered_cache


def scatter_back_rows(
    full_cache: DynamicCache,
    active_cache: DynamicCache,
    active_indices: torch.Tensor,
) -> None:
    full_key_cache = getattr(full_cache, "key_cache", [])
    active_key_cache = getattr(active_cache, "key_cache", [])
    full_value_cache = getattr(full_cache, "value_cache", [])
    active_value_cache = getattr(active_cache, "value_cache", [])

    if len(full_key_cache) != len(active_key_cache):
        raise ValueError("cache layer count mismatch while scattering active rows")
    for layer_idx, (full_key, active_key) in enumerate(zip(full_key_cache, active_key_cache)):
        if full_key.shape[0] <= int(active_indices.max().item()):
            raise ValueError(f"invalid active index for key cache layer {layer_idx}")
        full_key.index_copy_(0, active_indices, active_key)
    for full_value, active_value in zip(full_value_cache, active_value_cache):
        full_value.index_copy_(0, active_indices, active_value)


def validate_cache_batch_size(cache: DynamicCache, expected_batch: int, cache_name: str) -> None:
    for layer_idx, key in enumerate(getattr(cache, "key_cache", [])):
        if key.shape[0] != expected_batch:
            raise ValueError(
                f"{cache_name} layer {layer_idx} batch={key.shape[0]}, expected active batch={expected_batch}"
            )
    for layer_idx, value in enumerate(getattr(cache, "value_cache", [])):
        if value.shape[0] != expected_batch:
            raise ValueError(
                f"{cache_name} layer {layer_idx} value batch={value.shape[0]}, expected active batch={expected_batch}"
            )


def block_table_context_len(sequence: Sequence) -> int:
    if not sequence.block_table:
        return 0
    return sequence.block_table[-1].logical_end


def validate_sequence_runtime_consistency(
    sequence: Sequence,
    forward_visible_context_len: int,
    block_manager: BlockManager,
) -> None:
    block_manager.check_consistency(sequence)
    table_context_len = block_table_context_len(sequence)
    if sequence.num_cached_tokens != table_context_len:
        raise ValueError(
            f"sequence {sequence.seq_id} logical/table mismatch: "
            f"num_cached_tokens={sequence.num_cached_tokens}, table_context_len={table_context_len}"
        )
    expected_forward_visible_context_len = sequence.num_cached_tokens + (1 if sequence.pre_verify else 0)
    if expected_forward_visible_context_len != forward_visible_context_len:
        raise ValueError(
            f"sequence {sequence.seq_id} logical/forward mismatch: "
            f"expected_forward_visible_context_len={expected_forward_visible_context_len}, "
            f"forward_visible_context_len={forward_visible_context_len}, pre_verify={sequence.pre_verify}"
        )
    if sequence.pending_kv_append:
        pending_len = len(sequence.pending_kv_append)
        if pending_len > sequence.num_cached_tokens:
            raise ValueError(
                f"sequence {sequence.seq_id} pending kv mismatch: pending_len={pending_len}, "
                f"num_cached_tokens={sequence.num_cached_tokens}"
            )
        pending_start = sequence.num_cached_tokens - pending_len
        pending_suffix = sequence.token_ids[pending_start : sequence.num_cached_tokens]
        if pending_suffix != sequence.pending_kv_append:
            raise ValueError(
                f"sequence {sequence.seq_id} pending kv suffix mismatch: "
                f"pending={sequence.pending_kv_append}, suffix={pending_suffix}"
            )


def commit_pending_kv(
    sequence: Sequence,
    block_manager: BlockManager,
    target_cache_view: DynamicCache | None = None,
    rollback_to: int | None = None,
) -> None:
    del target_cache_view
    if not sequence.pending_kv_append:
        validate_sequence_runtime_consistency(
            sequence=sequence,
            forward_visible_context_len=len(sequence.token_ids),
            block_manager=block_manager,
        )
        return

    pending_len = len(sequence.pending_kv_append)
    if sequence.num_cached_tokens < pending_len:
        raise ValueError(
            f"sequence {sequence.seq_id} commit underflow: "
            f"num_cached_tokens={sequence.num_cached_tokens}, pending_len={pending_len}"
        )

    pending_start = sequence.num_cached_tokens - pending_len
    if sequence.token_ids[pending_start : sequence.num_cached_tokens] != sequence.pending_kv_append:
        raise ValueError(
            f"sequence {sequence.seq_id} pending kv does not match token suffix before commit"
        )

    try:
        sync_sequence_blocks(sequence, block_manager)
        validate_sequence_runtime_consistency(
            sequence=sequence,
            forward_visible_context_len=len(sequence.token_ids),
            block_manager=block_manager,
        )
    except Exception:
        if rollback_to is not None:
            block_manager.rollback(sequence.seq_id, rollback_to)
            sequence.token_ids = sequence.token_ids[: sequence.num_cached_tokens]
            sequence.pending_kv_append.clear()
        raise

    sequence.pending_kv_append.clear()


def collate_prompts(
    input_ids_list: list[torch.Tensor],
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if not input_ids_list:
        raise ValueError("input_ids_list must not be empty")

    batch_size = len(input_ids_list)
    input_lengths = torch.tensor(
        [tensor.shape[-1] for tensor in input_ids_list],
        device=input_ids_list[0].device,
        dtype=torch.long,
    )
    max_length = int(input_lengths.max().item())
    input_ids_padded = torch.full(
        (batch_size, max_length),
        pad_token_id,
        dtype=torch.long,
        device=input_ids_list[0].device,
    )
    attention_mask = torch.zeros(
        (batch_size, max_length),
        dtype=torch.long,
        device=input_ids_list[0].device,
    )

    for row, input_ids in enumerate(input_ids_list):
        length = input_ids.shape[-1]
        input_ids_padded[row, :length] = input_ids[0]
        attention_mask[row, :length] = 1

    return input_ids_padded, attention_mask, input_lengths


def build_batch_cache_metadata(
    scheduled_batch: list[Sequence],
    active_start_pos: torch.Tensor,
) -> dict[str, list[list[int]] | list[int]]:
    block_tables = [[mapping.physical_block_id for mapping in seq.block_table] for seq in scheduled_batch]
    context_lens = [int(seq.num_cached_tokens) for seq in scheduled_batch]
    slot_mapping = [int(pos.item()) for pos in active_start_pos]
    return {
        "block_tables": block_tables,
        "context_lens": context_lens,
        "slot_mapping": slot_mapping,
    }


def prepare_packed_verify_inputs(
    sequences: list[Sequence],
    gamma: int,
) -> dict[str, list[int] | list[list[int]] | list[tuple[int, int]]]:
    if gamma <= 0:
        raise ValueError("gamma must be positive")

    verify_lens = [1 if seq.pre_verify else gamma for seq in sequences]
    packed_ranges: list[tuple[int, int]] = []
    slot_mapping: list[int] = []
    cursor = 0
    for seq_idx, verify_len in enumerate(verify_lens):
        packed_ranges.append((cursor, cursor + verify_len))
        slot_mapping.extend([seq_idx] * verify_len)
        cursor += verify_len

    context_lens = [int(seq.num_cached_tokens) for seq in sequences]
    block_tables = [[mapping.physical_block_id for mapping in seq.block_table] for seq in sequences]

    return {
        "verify_lens": verify_lens,
        "packed_ranges": packed_ranges,
        "slot_mapping": slot_mapping,
        "context_lens": context_lens,
        "block_tables": block_tables,
    }


def _supports_packed_verify_kwargs(target: AutoModelForCausalLM) -> bool:
    target_id = id(target)
    if target_id in _PACKED_VERIFY_CAPABILITY_CACHE:
        return _PACKED_VERIFY_CAPABILITY_CACHE[target_id]

    probes = [getattr(target, "forward", None), getattr(target, "__call__", None)]
    required_kwargs = {"slot_mapping", "context_lens", "block_tables"}
    for probe in probes:
        if probe is None:
            continue
        try:
            signature = inspect.signature(probe)
        except (TypeError, ValueError):
            continue
        has_var_kwargs = any(
            p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values()
        )
        supports = has_var_kwargs or required_kwargs.issubset(signature.parameters)
        _PACKED_VERIFY_CAPABILITY_CACHE[target_id] = supports
        return supports

    _PACKED_VERIFY_CAPABILITY_CACHE[target_id] = False
    return False


def _is_packed_kwargs_typeerror(exc: TypeError) -> bool:
    message = str(exc)
    return any(token in message for token in ["slot_mapping", "context_lens", "block_tables"])


def build_verify_batch_inputs(
    block_output_ids: torch.Tensor,
    block_position_ids: torch.Tensor,
    verify_lens: list[int],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if block_output_ids.ndim != 2 or block_position_ids.ndim != 2:
        raise ValueError("verify inputs must be 2D tensors")
    if block_output_ids.shape != block_position_ids.shape:
        raise ValueError("verify input_ids and position_ids must share shape")

    batch_size, block_size = block_output_ids.shape
    if len(verify_lens) != batch_size:
        raise ValueError("verify_lens and verify input batch size mismatch")

    verify_mask = torch.zeros_like(block_output_ids, dtype=torch.long)
    for row, verify_len in enumerate(verify_lens):
        if verify_len <= 0 or verify_len > block_size:
            raise ValueError(f"verify_len={verify_len} out of range [1, {block_size}]")
        verify_mask[row, :verify_len] = 1

    return block_output_ids, block_position_ids, verify_mask


def infer_cache_batch_size(cache: DynamicCache) -> int | None:
    key_cache = getattr(cache, "key_cache", [])
    if key_cache:
        return int(key_cache[0].shape[0])
    value_cache = getattr(cache, "value_cache", [])
    if value_cache:
        return int(value_cache[0].shape[0])
    return None


@torch.inference_mode()
def init_batch_state(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    temperature: float,
) -> tuple[BatchDecodeState, float]:
    assert input_ids.ndim == 2, "input_ids must be [B, S]"
    batch_size, num_input_tokens = input_ids.shape
    assert batch_size >= 1, "batch size must be >= 1"

    max_length = num_input_tokens + max_new_tokens
    output_ids = torch.full(
        (batch_size, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    assert output_ids.ndim == 2, "output_ids must be 2D"
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    assert position_ids.ndim == 2 and position_ids.shape[0] == 1
    prefill_position_ids = attention_mask.long().cumsum(dim=-1) - 1
    prefill_position_ids = prefill_position_ids.masked_fill(attention_mask == 0, 0)

    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    prefill_start = cuda_time()
    with nvtx_range("target.prefill.forward"):
        output = target(
            input_ids,
            attention_mask=attention_mask,
            position_ids=prefill_position_ids,
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True if block_size > 1 else False,
        )

    assert output.logits.ndim == 3, "target logits must be [B, T, V]"
    for row in range(batch_size):
        row_input_length = int(input_lengths[row].item())
        output_ids[row, :row_input_length] = input_ids[row, :row_input_length]

    last_token_indices = (input_lengths - 1).to(torch.long)
    batch_indices = torch.arange(batch_size, device=model.device)
    next_token_logits = output.logits[batch_indices, last_token_indices].unsqueeze(1)
    next_tokens = sample(next_token_logits, temperature)
    output_ids[:, num_input_tokens : num_input_tokens + 1] = next_tokens

    target_hidden = None
    if block_size > 1:
        prefill_target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        assert prefill_target_hidden.ndim == 3 and prefill_target_hidden.shape[0] == batch_size
        target_hidden = torch.zeros(
            (batch_size, output_ids.shape[1], prefill_target_hidden.shape[-1]),
            dtype=prefill_target_hidden.dtype,
            device=prefill_target_hidden.device,
        )
        target_hidden[:, : prefill_target_hidden.shape[1], :] = prefill_target_hidden

    state = BatchDecodeState(
        output_ids=output_ids,
        input_lengths=input_lengths,
        start_pos=torch.full((batch_size,), num_input_tokens, device=model.device, dtype=torch.long),
        finished_mask=torch.zeros(batch_size, dtype=torch.bool, device=model.device),
        active_indices=torch.arange(batch_size, device=model.device, dtype=torch.long),
        acceptance_lengths_per_row=[[] for _ in range(batch_size)],
        max_length=max_length,
        position_ids=position_ids,
        past_key_values_target=past_key_values_target,
        past_key_values_draft=past_key_values_draft,
        target_hidden=target_hidden,
        decode_start=cuda_time(),
        draft_prefill=True,
        active_batch_size_trace=[],
        grouped_batch_count_trace=[],
        grouped_batch_sizes_trace=[],
    )
    return state, state.decode_start - prefill_start


@torch.inference_mode()
def draft_propose_step(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    state: BatchDecodeState,
    scheduled_batch: list[Sequence],
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert state.output_ids.ndim == 2
    assert state.active_indices.ndim == 1

    active_indices = state.active_indices
    active_batch = int(active_indices.numel())
    assert active_batch > 0, "active batch must be non-empty"
    active_start_pos = state.start_pos.index_select(0, active_indices)
    assert torch.unique(active_start_pos).numel() == 1, "active rows must share start_pos"
    active_start_pos_scalar = int(active_start_pos[0].item())
    assert active_start_pos_scalar < state.max_length
    _ = build_batch_cache_metadata(scheduled_batch=scheduled_batch, active_start_pos=active_start_pos)

    block_output_ids = state.output_ids.index_select(0, active_indices)[:, active_start_pos_scalar : active_start_pos_scalar + block_size].clone()
    block_position_ids = state.position_ids[:, active_start_pos_scalar : active_start_pos_scalar + block_size].expand(active_batch, -1)
    assert block_output_ids.ndim == 2 and block_output_ids.shape[0] == active_batch

    active_draft_cache = gather_active_rows(state.past_key_values_draft, active_indices)
    validate_cache_batch_size(active_draft_cache, active_batch, "past_key_values_draft")

    if block_size > 1:
        assert state.target_hidden is not None and state.target_hidden.ndim == 3
        active_target_hidden = state.target_hidden.index_select(0, active_indices)
        with profile_phase("draft"):
            with nvtx_range("draft.forward"):
                noise_embedding = target.model.embed_tokens(block_output_ids)
                assert noise_embedding.ndim == 3 and noise_embedding.shape[:2] == block_output_ids.shape
                draft_hidden = model(
                    target_hidden=active_target_hidden,
                    noise_embedding=noise_embedding,
                    position_ids=state.position_ids[
                        :, active_start_pos_scalar : active_start_pos_scalar + block_size
                    ].expand(active_batch, -1),
                    past_key_values=active_draft_cache,
                    use_cache=True,
                    is_causal=False,
                )
        with nvtx_range("draft.output_head"):
            draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
        assert draft_logits.ndim == 3 and draft_logits.shape[0] == block_output_ids.shape[0]
        block_output_ids[:, 1:] = sample(draft_logits)
        if state.draft_prefill:
            state.draft_prefill = False
            state.decode_start = cuda_time()

    return block_output_ids, block_position_ids, active_indices


@torch.inference_mode()
def target_verify_step(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    state: BatchDecodeState,
    scheduled_batch: list[Sequence],
    active_indices: torch.Tensor,
    block_output_ids: torch.Tensor,
    block_position_ids: torch.Tensor,
    block_size: int,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, DynamicCache]:
    assert block_output_ids.ndim == 2 and block_position_ids.ndim == 2
    active_batch = int(active_indices.numel())
    assert block_output_ids.shape[0] == active_batch
    assert block_output_ids.shape[1] == block_size
    assert block_position_ids.shape[0] == active_batch and block_position_ids.shape[1] == block_size
    active_start_pos = state.start_pos.index_select(0, active_indices)
    _ = build_batch_cache_metadata(scheduled_batch=scheduled_batch, active_start_pos=active_start_pos)

    active_target_cache = gather_active_rows(state.past_key_values_target, active_indices)

    packed_meta = prepare_packed_verify_inputs(scheduled_batch, gamma=block_size)
    verify_lens = packed_meta["verify_lens"]
    assert isinstance(verify_lens, list)
    if len(verify_lens) != active_batch:
        raise ValueError("verify_lens and active batch size mismatch")

    slot_mapping = packed_meta["slot_mapping"]
    context_lens = packed_meta["context_lens"]
    block_tables = packed_meta["block_tables"]
    assert isinstance(slot_mapping, list)
    assert isinstance(context_lens, list)
    assert isinstance(block_tables, list)

    verify_input_ids, verify_position_ids, verify_mask = build_verify_batch_inputs(
        block_output_ids=block_output_ids,
        block_position_ids=block_position_ids,
        verify_lens=verify_lens,
    )

    cache_batch_size = infer_cache_batch_size(active_target_cache)
    input_batch_size = int(verify_input_ids.shape[0])
    if cache_batch_size is not None and input_batch_size != cache_batch_size:
        raise ValueError(
            f"target verify batch mismatch: input batch={input_batch_size}, cache batch={cache_batch_size}"
        )
    validate_cache_batch_size(active_target_cache, input_batch_size, "past_key_values_target")

    verify_kwargs = {
        "attention_mask": verify_mask,
        "position_ids": verify_position_ids,
        "past_key_values": active_target_cache,
        "use_cache": True,
        "output_hidden_states": True if block_size > 1 else False,
    }
    packed_verify_supported = _supports_packed_verify_kwargs(target)
    if not packed_verify_supported and id(target) not in _PACKED_VERIFY_FALLBACK_WARNED_TARGETS:
        logger.warning(
            "target verify packed metadata is unavailable; using dense compatibility verify path"
        )
        _PACKED_VERIFY_FALLBACK_WARNED_TARGETS.add(id(target))
    if packed_verify_supported:
        verify_kwargs.update(
            {
                "slot_mapping": slot_mapping,
                "context_lens": context_lens,
                "block_tables": block_tables,
            }
        )

    verify_path = "packed" if packed_verify_supported else "fallback"
    verify_start = cuda_time()
    with profile_phase("target.verify"):
        with nvtx_range("target.verify.forward"):
            try:
                output = target(verify_input_ids, **verify_kwargs)
            except TypeError as exc:
                if not packed_verify_supported and not _is_packed_kwargs_typeerror(exc):
                    raise
                if packed_verify_supported and not _is_packed_kwargs_typeerror(exc):
                    raise
                _PACKED_VERIFY_CAPABILITY_CACHE[id(target)] = False
                verify_kwargs.pop("slot_mapping", None)
                verify_kwargs.pop("context_lens", None)
                verify_kwargs.pop("block_tables", None)
                if id(target) not in _PACKED_VERIFY_FALLBACK_WARNED_TARGETS:
                    logger.warning(
                        "target verify does not support packed metadata kwargs (slot_mapping/context_lens/block_tables); falling back to dense compatibility verify path"
                    )
                    _PACKED_VERIFY_FALLBACK_WARNED_TARGETS.add(id(target))
                verify_path = "fallback"
                output = target(verify_input_ids, **verify_kwargs)
    verify_elapsed = cuda_time() - verify_start
    verify_token_count = int(verify_mask.sum().item())
    _PACKED_VERIFY_PATH_STATS[verify_path]["tokens"] += verify_token_count
    _PACKED_VERIFY_PATH_STATS[verify_path]["time"] += float(verify_elapsed)

    target_logits = output.logits.clone()
    target_logits = target_logits.masked_fill(verify_mask.unsqueeze(-1) == 0, -torch.inf)

    posterior = sample(target_logits, temperature)
    assert posterior.ndim == 2 and posterior.shape[0] == block_output_ids.shape[0]

    max_cmp = block_size - 1
    if max_cmp == 0:
        acceptance_len_vec = torch.zeros((active_batch,), dtype=torch.long, device=block_output_ids.device)
    else:
        compare_positions = torch.arange(max_cmp, device=block_output_ids.device).unsqueeze(0)
        valid_compare = compare_positions < (torch.tensor(verify_lens, device=block_output_ids.device).unsqueeze(1) - 1)
        mismatches = (block_output_ids[:, 1 : 1 + max_cmp] != posterior[:, :max_cmp]) & valid_compare
        first_mismatch = torch.where(
            mismatches.any(dim=1),
            mismatches.float().argmax(dim=1),
            torch.full((active_batch,), -1, dtype=torch.long, device=block_output_ids.device),
        )
        acceptance_len_vec = torch.where(first_mismatch >= 0, first_mismatch, valid_compare.sum(dim=1))
    assert acceptance_len_vec.ndim == 1 and acceptance_len_vec.shape[0] == block_output_ids.shape[0]

    target_hidden = None
    if block_size > 1:
        verify_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        target_hidden = verify_hidden * verify_mask.unsqueeze(-1).to(dtype=verify_hidden.dtype)
        assert target_hidden.ndim == 3 and target_hidden.shape[0] == block_output_ids.shape[0]

    return posterior, acceptance_len_vec, target_hidden, active_target_cache


def apply_acceptance_step(
    state: BatchDecodeState,
    active_indices: torch.Tensor,
    block_output_ids: torch.Tensor,
    posterior: torch.Tensor,
    acceptance_len_vec: torch.Tensor,
    target_hidden: torch.Tensor | None,
    active_target_cache: DynamicCache,
    block_size: int,
) -> None:
    assert block_output_ids.ndim == 2 and posterior.ndim == 2
    active_batch = int(active_indices.numel())
    assert block_output_ids.shape[0] == posterior.shape[0] == active_batch
    assert acceptance_len_vec.ndim == 1 and acceptance_len_vec.shape[0] == active_batch
    if torch.any((acceptance_len_vec < 0) | (acceptance_len_vec > block_size - 1)):
        raise ValueError(f"acceptance_len must be in [0, {block_size - 1}] for each active row")

    active_start_pos = state.start_pos.index_select(0, active_indices)

    for local_row, row in enumerate(active_indices.tolist()):
        acceptance_len = int(acceptance_len_vec[local_row].item())
        row_start_pos = int(active_start_pos[local_row].item())
        row_end_pos = row_start_pos + acceptance_len + 1
        state.output_ids[row, row_start_pos:row_end_pos] = block_output_ids[local_row, : acceptance_len + 1]
        state.output_ids[row, row_end_pos] = posterior[local_row, acceptance_len]
        state.acceptance_lengths_per_row[row].append(acceptance_len + 1)

    state.start_pos[active_indices] = active_start_pos + acceptance_len_vec + 1
    state.finished_mask[active_indices] |= state.start_pos.index_select(0, active_indices) >= state.max_length

    with nvtx_range("kv_update.target_cache"):
        validate_cache_batch_size(active_target_cache, active_batch, "past_key_values_target")
        scatter_back_rows(state.past_key_values_target, active_target_cache, active_indices)
    if block_size > 1 and target_hidden is not None:
        if state.target_hidden is None:
            raise ValueError("target_hidden must be initialized when block_size > 1")
        for local_row, row in enumerate(active_indices.tolist()):
            acceptance_len = int(acceptance_len_vec[local_row].item())
            row_start_pos = int(active_start_pos[local_row].item())
            row_end_pos = row_start_pos + acceptance_len + 1
            target_hidden_slice = target_hidden[local_row, : acceptance_len + 1, :]
            state_hidden_slice = state.target_hidden[row, row_start_pos:row_end_pos, :]
            assert target_hidden_slice.shape == state_hidden_slice.shape
            state.target_hidden[row, row_start_pos:row_end_pos, :] = target_hidden_slice



def finalize_outputs(
    state: BatchDecodeState,
    mask_token_id: int,
    stop_token_ids: list[int] | None,
    time_to_first_token: float,
    seq_ids: list[int] | None = None,
) -> list[SimpleNamespace]:
    assert state.output_ids.ndim == 2
    responses: list[SimpleNamespace] = []
    stop_tokens = None
    if stop_token_ids is not None:
        stop_tokens = torch.tensor(stop_token_ids, device=state.output_ids.device)

    ordered_seq_ids = seq_ids if seq_ids is not None else list(range(state.output_ids.shape[0]))

    for row in ordered_seq_ids:
        row_full_output = state.output_ids[row]
        row_output = row_full_output[row_full_output != mask_token_id]
        num_input_tokens = int(state.input_lengths[row].item())

        if stop_tokens is not None and row_output.numel() > num_input_tokens:
            stop_token_indices = torch.isin(row_output[num_input_tokens:], stop_tokens).nonzero(
                as_tuple=True
            )[0]
            if stop_token_indices.numel() > 0:
                row_output = row_output[: num_input_tokens + int(stop_token_indices[0].item()) + 1]

        row_output = row_output.unsqueeze(0)

        num_output_tokens = row_output.shape[1] - num_input_tokens
        total_decode_time = cuda_time() - state.decode_start
        time_per_output_token = total_decode_time / max(num_output_tokens, 1)
        responses.append(
            SimpleNamespace(
                output_ids=row_output,
                num_input_tokens=num_input_tokens,
                num_output_tokens=num_output_tokens,
                time_to_first_token=time_to_first_token,
                time_per_output_token=time_per_output_token,
                acceptance_lengths=state.acceptance_lengths_per_row[row],
                active_batch_size_trace=state.active_batch_size_trace.copy(),
                grouped_batch_count_trace=state.grouped_batch_count_trace.copy(),
                grouped_batch_sizes_trace=[sizes.copy() for sizes in state.grouped_batch_sizes_trace],
            )
        )

    return responses


@torch.inference_mode()
def dflash_generate(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> list[SimpleNamespace]:
    assert input_ids.ndim == 2, "input_ids must be [B, S]"
    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    input_lengths = torch.full(
        (input_ids.shape[0],), input_ids.shape[1], device=input_ids.device, dtype=torch.long
    )
    return dflash_generate_batch_staged(
        model=model,
        target=target,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        mask_token_id=mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
    )


def dflash_generate_batch_staged(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
    enable_multi_start_pos_grouping: bool = True,
) -> list[SimpleNamespace]:
    block_manager = BlockManager(block_size=block_size)
    state, time_to_first_token = init_batch_state(
        model=model,
        target=target,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        mask_token_id=mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
    )

    sequences = [
        Sequence(
            seq_id=row,
            token_ids=state.output_ids[row, : int(input_lengths[row].item()) + 1].tolist(),
            num_cached_tokens=int(state.start_pos[row].item()),
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        )
        for row in range(state.output_ids.shape[0])
    ]
    for sequence in sequences:
        block_manager.register_sequence(sequence)
        sync_sequence_blocks(sequence, block_manager)
    scheduler = Scheduler(sequences, block_manager=block_manager)
    stop_tokens = (
        torch.tensor(stop_token_ids, device=state.output_ids.device) if stop_token_ids is not None else None
    )

    while scheduler.has_pending():
        for sequence in sequences:
            validate_sequence_runtime_consistency(
                sequence=sequence,
                forward_visible_context_len=len(sequence.token_ids),
                block_manager=block_manager,
            )
        if enable_multi_start_pos_grouping:
            scheduled_groups = scheduler.schedule_next_batch(
                max_batch_size=state.output_ids.shape[0],
                return_grouped=True,
            )
        else:
            scheduled_batch = scheduler.schedule_next_batch(max_batch_size=state.output_ids.shape[0])
            scheduled_groups = [scheduled_batch] if scheduled_batch else []
        if not scheduled_groups:
            break

        state.grouped_batch_count_trace.append(len(scheduled_groups))
        state.grouped_batch_sizes_trace.append([len(group) for group in scheduled_groups])

        for scheduled_batch in scheduled_groups:
            state.active_indices = torch.tensor(
                [seq.seq_id for seq in scheduled_batch],
                dtype=torch.long,
                device=state.output_ids.device,
            )
            state.active_batch_size_trace.append(int(state.active_indices.numel()))

            block_output_ids, block_position_ids, active_indices = draft_propose_step(
                model=model,
                target=target,
                state=state,
                scheduled_batch=scheduled_batch,
                block_size=block_size,
            )

            for local_row, seq in enumerate(scheduled_batch):
                scheduler.append_draft_tokens(seq.seq_id, block_output_ids[local_row].tolist())

            posterior, acceptance_len_vec, target_hidden, active_target_cache = target_verify_step(
                model=model,
                target=target,
                state=state,
                scheduled_batch=scheduled_batch,
                active_indices=active_indices,
                block_output_ids=block_output_ids,
                block_position_ids=block_position_ids,
                block_size=block_size,
                temperature=temperature,
            )
            apply_acceptance_step(
                state=state,
                active_indices=active_indices,
                block_output_ids=block_output_ids,
                posterior=posterior,
                acceptance_len_vec=acceptance_len_vec,
                target_hidden=target_hidden,
                active_target_cache=active_target_cache,
                block_size=block_size,
            )

            for local_row, row in enumerate(active_indices.tolist()):
                seq = sequences[row]
                seq.pre_verify = False
                accepted_len = int(acceptance_len_vec[local_row].item()) + 1
                seq.num_acc_tokens += accepted_len

                txn = scheduler.consume_draft_transaction(seq.seq_id)
                draft_start = int(txn["start"])
                accepted_tokens = block_output_ids[local_row, :accepted_len].tolist()
                verify_token = int(posterior[local_row, accepted_len - 1].item())
                seq.pending_kv_append.extend(accepted_tokens + [verify_token])

                scheduler.rollback(seq.seq_id, draft_start + accepted_len)
                seq.token_ids = seq.token_ids[: seq.num_cached_tokens] + [verify_token]
                seq.num_cached_tokens += 1
                commit_pending_kv(
                    sequence=seq,
                    block_manager=block_manager,
                    target_cache_view=state.past_key_values_target,
                    rollback_to=draft_start,
                )

                row_input_length = int(state.input_lengths[row].item())
                reached_max_new_tokens = (seq.num_cached_tokens - row_input_length) >= max_new_tokens
                no_available_position = seq.num_cached_tokens >= state.output_ids.shape[1] - 1
                hit_stop_token = False
                if stop_tokens is not None:
                    row_tokens = state.output_ids[row, row_input_length : seq.num_cached_tokens + 1]
                    hit_stop_token = row_tokens.numel() > 0 and bool(torch.isin(row_tokens, stop_tokens).any())

                seq.finished = reached_max_new_tokens or no_available_position or hit_stop_token
                state.finished_mask[row] = seq.finished
                if seq.finished:
                    scheduler.mark_finished(seq)

        for sequence in sequences:
            validate_sequence_runtime_consistency(
                sequence=sequence,
                forward_visible_context_len=len(sequence.token_ids),
                block_manager=block_manager,
            )

    return finalize_outputs(
        state=state,
        mask_token_id=mask_token_id,
        stop_token_ids=stop_token_ids,
        time_to_first_token=time_to_first_token,
        seq_ids=[seq.seq_id for seq in sorted(scheduler.finished, key=lambda item: item.seq_id)],
    )


def dflash_generate_batch_stage_a_prefill_only(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
) -> list[SimpleNamespace]:
    state, _ = init_batch_state(
        model=model,
        target=target,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        mask_token_id=mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
    )

    responses: list[SimpleNamespace] = []
    for row in range(input_ids.shape[0]):
        row_input = input_ids[row : row + 1, : int(input_lengths[row].item())]
        row_result = dflash_generate(
            model=model,
            target=target,
            input_ids=row_input,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )[0]
        if temperature == 0.0:
            expected_prefix = state.output_ids[row, : int(input_lengths[row].item()) + 1]
            if not torch.equal(row_result.output_ids[0, : expected_prefix.numel()], expected_prefix):
                raise ValueError("batched prefill output misaligned with per-row decode input")
        responses.append(row_result)

    return responses


def dflash_generate_batch_stage_b_target_only(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
) -> list[SimpleNamespace]:
    state, time_to_first_token = init_batch_state(
        model=model,
        target=target,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        mask_token_id=mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        temperature=temperature,
    )

    while True:
        reached_max_new_tokens = (state.start_pos - state.input_lengths) >= max_new_tokens
        no_available_position = state.start_pos >= state.output_ids.shape[1] - 1

        hit_stop_token = torch.zeros_like(state.finished_mask)
        if stop_token_ids is not None:
            stop_tokens = torch.tensor(stop_token_ids, device=state.output_ids.device)
            for row in range(state.output_ids.shape[0]):
                row_start = int(state.input_lengths[row].item())
                row_end = int(state.start_pos[row].item()) + 1
                row_tokens = state.output_ids[row, row_start:row_end]
                if row_tokens.numel() > 0 and torch.isin(row_tokens, stop_tokens).any():
                    hit_stop_token[row] = True

        state.finished_mask = hit_stop_token | reached_max_new_tokens | no_available_position
        state.active_indices = (~state.finished_mask).nonzero(as_tuple=True)[0]
        if state.active_indices.numel() == 0:
            break
        state.active_batch_size_trace.append(int(state.active_indices.numel()))

        active_indices = state.active_indices
        active_batch = int(active_indices.numel())
        active_start_pos = state.start_pos.index_select(0, active_indices)
        if torch.unique(active_start_pos).numel() != 1:
            raise ValueError("active rows must share start_pos in stage_b_target_only")
        active_start_pos_scalar = int(active_start_pos[0].item())

        block_output_ids = state.output_ids.index_select(0, active_indices)[
            :, active_start_pos_scalar : active_start_pos_scalar + 1
        ]
        block_position_ids = state.position_ids[:, active_start_pos_scalar : active_start_pos_scalar + 1].expand(
            active_batch, -1
        )

        active_target_cache = gather_active_rows(state.past_key_values_target, active_indices)
        validate_cache_batch_size(active_target_cache, active_batch, "past_key_values_target")
        with profile_phase("target.verify"):
            with nvtx_range("target.verify.forward"):
                output = target(
                    block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=active_target_cache,
                    use_cache=True,
                    output_hidden_states=False,
                )
        next_token = sample(output.logits[:, -1:, :], temperature).squeeze(1)
        state.output_ids[active_indices, active_start_pos_scalar + 1] = next_token
        state.start_pos[active_indices] = active_start_pos + 1

        with nvtx_range("kv_update.target_cache"):
            validate_cache_batch_size(active_target_cache, active_batch, "past_key_values_target")
            scatter_back_rows(state.past_key_values_target, active_target_cache, active_indices)

    return finalize_outputs(
        state=state,
        mask_token_id=mask_token_id,
        stop_token_ids=stop_token_ids,
        time_to_first_token=time_to_first_token,
    )


def dflash_generate_batch_with_mode(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int] | None,
    temperature: float,
    batched_decode_mode: BatchedDecodeMode,
    enable_multi_start_pos_grouping: bool,
) -> list[SimpleNamespace]:
    if batched_decode_mode == "legacy":
        responses: list[SimpleNamespace] = []
        for row in range(input_ids.shape[0]):
            row_input = input_ids[row : row + 1, : int(input_lengths[row].item())]
            responses.extend(
                dflash_generate(
                    model=model,
                    target=target,
                    input_ids=row_input,
                    mask_token_id=mask_token_id,
                    max_new_tokens=max_new_tokens,
                    block_size=block_size,
                    stop_token_ids=stop_token_ids,
                    temperature=temperature,
                )
            )
        return responses
    if batched_decode_mode == "stage_a_prefill_only":
        return dflash_generate_batch_stage_a_prefill_only(
            model=model,
            target=target,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
    if batched_decode_mode == "stage_b_target_only":
        return dflash_generate_batch_stage_b_target_only(
            model=model,
            target=target,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
    if batched_decode_mode == "stage_c_full_speculative":
        if _supports_packed_verify_kwargs(target):
            return dflash_generate_batch_staged(
                model=model,
                target=target,
                input_ids=input_ids,
                attention_mask=attention_mask,
                input_lengths=input_lengths,
                mask_token_id=mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
                enable_multi_start_pos_grouping=enable_multi_start_pos_grouping,
            )
        logger.warning(
            "batched-decode-mode=stage_c_full_speculative requires target verify packed metadata kwargs; auto-fallback to stage_b_target_only compatibility path"
        )
        return dflash_generate_batch_stage_b_target_only(
            model=model,
            target=target,
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_lengths=input_lengths,
            mask_token_id=mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=block_size,
            stop_token_ids=stop_token_ids,
            temperature=temperature,
        )
    raise ValueError(f"unknown batched decode mode: {batched_decode_mode}")


def dflash_generate_batch(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int] | None,
    temperature: float = 0.0,
    batched_decode_mode: BatchedDecodeMode = "stage_c_full_speculative",
    enable_multi_start_pos_grouping: bool = True,
) -> list[SimpleNamespace]:
    return dflash_generate_batch_with_mode(
        model=model,
        target=target,
        input_ids=input_ids,
        attention_mask=attention_mask,
        input_lengths=input_lengths,
        mask_token_id=mask_token_id,
        max_new_tokens=max_new_tokens,
        block_size=block_size,
        stop_token_ids=stop_token_ids,
        temperature=temperature,
        batched_decode_mode=batched_decode_mode,
        enable_multi_start_pos_grouping=enable_multi_start_pos_grouping,
    )


def summarize_latency_percentiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan")}
    arr = np.asarray(values, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p90": float(np.percentile(arr, 90)),
        "p99": float(np.percentile(arr, 99)),
    }


def aggregate_active_batch_trace(responses: list[SimpleNamespace]) -> list[int]:
    if not responses:
        return []
    max_steps = max(len(getattr(r, "active_batch_size_trace", [])) for r in responses)
    if max_steps == 0:
        return []
    trace = []
    for step in range(max_steps):
        values = [
            r.active_batch_size_trace[step]
            for r in responses
            if step < len(getattr(r, "active_batch_size_trace", []))
        ]
        trace.append(int(round(float(np.mean(values)))))
    return trace


def aggregate_grouped_batch_count_trace(responses: list[SimpleNamespace]) -> list[int]:
    if not responses:
        return []
    max_steps = max(len(getattr(r, "grouped_batch_count_trace", [])) for r in responses)
    if max_steps == 0:
        return []
    trace = []
    for step in range(max_steps):
        values = [
            r.grouped_batch_count_trace[step]
            for r in responses
            if step < len(getattr(r, "grouped_batch_count_trace", []))
        ]
        trace.append(int(round(float(np.mean(values)))))
    return trace


def summarize_group_size_distribution(
    responses: list[SimpleNamespace],
) -> dict[int, float]:
    group_sizes = list(
        chain.from_iterable(
            chain.from_iterable(getattr(r, "grouped_batch_sizes_trace", []))
            for r in responses
        )
    )
    if not group_sizes:
        return {}
    total = len(group_sizes)
    return {
        size: count / total
        for size, count in sorted(
            ((size, group_sizes.count(size)) for size in sorted(set(group_sizes))),
            key=lambda item: item[0],
        )
    }


def compute_throughput_tokens_per_second(responses: list[SimpleNamespace]) -> float:
    total_tokens = sum(r.num_output_tokens for r in responses)
    total_time = sum(r.time_per_output_token * max(r.num_output_tokens, 1) for r in responses)
    if total_time <= 0:
        return float("nan")
    return float(total_tokens / total_time)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name-or-path", type=str, required=True)
    parser.add_argument("--draft-name-or-path", type=str, required=True)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--throughput-min-speedup", type=float, default=1.2)
    parser.add_argument(
        "--disable-multi-start-pos-grouping",
        action="store_true",
        help="fallback to legacy single start_pos group per decode round",
    )
    parser.add_argument(
        "--batched-decode-mode",
        type=str,
        default="stage_c_full_speculative",
        choices=[
            "legacy",
            "stage_a_prefill_only",
            "stage_b_target_only",
            "stage_c_full_speculative",
        ],
        help=f"staged rollout mode for batched decode; use legacy for fast rollback. {STAGE_C_PACKED_METADATA_REQUIREMENT}",
    )
    args = parser.parse_args()

    logger.info("{}", STAGE_C_PACKED_METADATA_REQUIREMENT)
    _PACKED_VERIFY_PATH_STATS["packed"]["tokens"] = 0
    _PACKED_VERIFY_PATH_STATS["packed"]["time"] = 0.0
    _PACKED_VERIFY_PATH_STATS["fallback"]["tokens"] = 0
    _PACKED_VERIFY_PATH_STATS["fallback"]["time"] = 0.0

    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init()
    torch.cuda.set_device(dist.local_rank())
    device = torch.device(f"cuda:{dist.local_rank()}")

    def has_flash_attn():
        try:
            import flash_attn  # noqa: F401

            return True
        except ImportError:
            logger.warning(
                "flash_attn is not installed. Falling back to torch.sdpa. The speedup will be lower."
            )
            return False

    installed_flash_attn = has_flash_attn()

    target = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()

    draft_model = DFlashDraftModel.from_pretrained(
        args.draft_name_or_path,
        attn_implementation="flash_attention_2" if installed_flash_attn else "sdpa",
        dtype=torch.bfloat16,
    ).to(device).eval()
    packed_verify_supported = _supports_packed_verify_kwargs(target)
    print(f"packed_verify_supported={packed_verify_supported}")

    # Required profiling ranges:
    # - draft.qkv / draft.attn / draft.ffn
    # - target.verify.qkv / target.verify.attn / target.verify.ffn
    ModuleRangeProfiler(target)
    ModuleRangeProfiler(draft_model)

    block_size = args.block_size if args.block_size is not None else draft_model.block_size

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is None:
            raise ValueError("Tokenizer does not provide a pad_token_id or eos_token_id for fallback")
        tokenizer.pad_token = tokenizer.eos_token
        logger.warning(
            "Tokenizer has no pad token. Falling back to eos token id {} as pad token.",
            tokenizer.pad_token_id,
        )
    dataset = load_and_process_dataset(args.dataset)

    if args.max_samples is not None and len(dataset) > args.max_samples:
        dataset = dataset.shuffle(seed=0).select(range(args.max_samples))

    responses = []
    indices = list(range(dist.rank(), len(dataset), dist.size()))
    for chunk_start in tqdm(
        range(0, len(indices), args.batch_size), disable=not dist.is_main()
    ):
        batch_indices = indices[chunk_start : chunk_start + args.batch_size]
        batch_instances = [dataset[idx] for idx in batch_indices]
        row_state = {
            row_id: {
                "turn_index": 0,
                "messages": [],
                "done": False,
                "instance": instance,
            }
            for row_id, instance in enumerate(batch_instances)
        }
        active_rows = list(row_state.keys())

        while active_rows:
            prompt_rows = []
            input_ids_batch = []
            for row_id in active_rows:
                state = row_state[row_id]
                turn_index = state["turn_index"]
                instance = state["instance"]

                if turn_index >= len(instance["turns"]):
                    state["done"] = True
                    continue

                user_content = instance["turns"][turn_index]
                state["messages"].append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    state["messages"],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids_batch.append(
                    tokenizer.encode(input_text, return_tensors="pt").to(target.device)
                )
                prompt_rows.append(row_id)

            active_rows = [row_id for row_id in active_rows if not row_state[row_id]["done"]]

            if not input_ids_batch:
                continue

            input_ids_padded, attention_mask, input_lengths = collate_prompts(
                input_ids_list=input_ids_batch,
                pad_token_id=tokenizer.pad_token_id,
            )

            batch_response = {}
            for bs in [1, block_size]:
                with nvtx_range(f"generate.block_size_{bs}"):
                    batch_response[bs] = dflash_generate_batch(
                        model=draft_model,
                        target=target,
                        input_ids=input_ids_padded,
                        attention_mask=attention_mask,
                        input_lengths=input_lengths,
                        mask_token_id=draft_model.mask_token_id,
                        max_new_tokens=args.max_new_tokens,
                        block_size=bs,
                        stop_token_ids=[tokenizer.eos_token_id],
                        temperature=args.temperature,
                        batched_decode_mode=args.batched_decode_mode,
                        enable_multi_start_pos_grouping=not args.disable_multi_start_pos_grouping,
                    )

            responses_by_row = {
                row_id: {
                    1: baseline_resp,
                    block_size: spec_resp,
                }
                for row_id, baseline_resp, spec_resp in zip(
                    prompt_rows,
                    batch_response[1],
                    batch_response[block_size],
                    strict=True,
                )
            }

            for row_id, response in responses_by_row.items():
                state = row_state[row_id]
                spec_response = response[block_size]
                generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                state["messages"].append({"role": "assistant", "content": output_text})
                state["turn_index"] += 1
                if state["turn_index"] >= len(state["instance"]["turns"]):
                    state["done"] = True
                responses.append(response)

            active_rows = [row_id for row_id in active_rows if not row_state[row_id]["done"]]

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    baseline_responses = [r[1] for r in responses]
    speculative_responses = [r[block_size] for r in responses]

    t1 = np.mean([r.time_per_output_token for r in baseline_responses])
    tb = np.mean([r.time_per_output_token for r in speculative_responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    throughput_baseline = compute_throughput_tokens_per_second(baseline_responses)
    throughput_spec = compute_throughput_tokens_per_second(speculative_responses)
    throughput_speedup = throughput_spec / throughput_baseline
    print(f"Throughput tokens/s baseline={throughput_baseline:.2f}, speculative={throughput_spec:.2f}, speedup={throughput_speedup:.2f}")
    packed_tokens = _PACKED_VERIFY_PATH_STATS["packed"]["tokens"]
    packed_time = _PACKED_VERIFY_PATH_STATS["packed"]["time"]
    fallback_tokens = _PACKED_VERIFY_PATH_STATS["fallback"]["tokens"]
    fallback_time = _PACKED_VERIFY_PATH_STATS["fallback"]["time"]
    packed_throughput = float(packed_tokens / packed_time) if packed_time > 0 else float("nan")
    fallback_throughput = (
        float(fallback_tokens / fallback_time) if fallback_time > 0 else float("nan")
    )
    print(f"packed_verify_supported={packed_verify_supported}")
    print(
        "Verify path throughput tokens/s "
        f"packed={packed_throughput:.2f} (tokens={packed_tokens}), "
        f"fallback={fallback_throughput:.2f} (tokens={fallback_tokens})"
    )
    if block_size >= 4 and throughput_speedup < args.throughput_min_speedup:
        raise RuntimeError(
            f"Performance gate failed for block_size={block_size}: throughput speedup {throughput_speedup:.2f} < {args.throughput_min_speedup:.2f}"
        )

    acceptance_means = [np.mean(r.acceptance_lengths) for r in speculative_responses if r.acceptance_lengths]
    tau = float(np.mean(acceptance_means)) if acceptance_means else float("nan")
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r.acceptance_lengths for r in speculative_responses]))
    histogram = [
        acceptance_lengths.count(b) / len(acceptance_lengths)
        for b in range(block_size + 1)
    ] if acceptance_lengths else []
    if histogram:
        print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")

    active_batch_trace = aggregate_active_batch_trace(speculative_responses)
    print(f"Active batch size trace (mean by decode step): {active_batch_trace}")

    grouped_batch_count_trace = aggregate_grouped_batch_count_trace(speculative_responses)
    print(f"Grouped batches per decode round (mean by decode step): {grouped_batch_count_trace}")

    group_size_distribution = summarize_group_size_distribution(speculative_responses)
    if group_size_distribution:
        print(
            "Group size distribution: "
            + str({k: f"{v * 100:.1f}%" for k, v in group_size_distribution.items()})
        )

    ttft_percentiles = summarize_latency_percentiles([r.time_to_first_token for r in speculative_responses])
    tpot_percentiles = summarize_latency_percentiles([r.time_per_output_token for r in speculative_responses])
    print(
        "TTFT percentiles (s): "
        f"p50={ttft_percentiles['p50']:.4f}, p90={ttft_percentiles['p90']:.4f}, p99={ttft_percentiles['p99']:.4f}"
    )
    print(
        "TPOT percentiles (s/token): "
        f"p50={tpot_percentiles['p50']:.6f}, p90={tpot_percentiles['p90']:.6f}, p99={tpot_percentiles['p99']:.6f}"
    )


if __name__ == "__main__":
    main()
