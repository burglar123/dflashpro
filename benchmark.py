import argparse
import random
import time
from contextlib import contextmanager
from dataclasses import dataclass
from itertools import chain
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
    torch.cuda.synchronize()
    return time.perf_counter()


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


def gather_active_rows(cache: DynamicCache, active_indices: torch.Tensor) -> DynamicCache:
    gathered_cache = DynamicCache()
    gathered_cache.key_cache = [key.index_select(0, active_indices) for key in cache.key_cache]
    gathered_cache.value_cache = [value.index_select(0, active_indices) for value in cache.value_cache]
    if hasattr(cache, "_seen_tokens"):
        gathered_cache._seen_tokens = cache._seen_tokens
    return gathered_cache


def scatter_back_rows(
    full_cache: DynamicCache,
    active_cache: DynamicCache,
    active_indices: torch.Tensor,
) -> None:
    if len(full_cache.key_cache) != len(active_cache.key_cache):
        raise ValueError("cache layer count mismatch while scattering active rows")
    for layer_idx, (full_key, active_key) in enumerate(zip(full_cache.key_cache, active_cache.key_cache)):
        if full_key.shape[0] <= int(active_indices.max().item()):
            raise ValueError(f"invalid active index for key cache layer {layer_idx}")
        full_key.index_copy_(0, active_indices, active_key)
    for full_value, active_value in zip(full_cache.value_cache, active_cache.value_cache):
        full_value.index_copy_(0, active_indices, active_value)


def validate_cache_batch_size(cache: DynamicCache, expected_batch: int, cache_name: str) -> None:
    for layer_idx, key in enumerate(cache.key_cache):
        if key.shape[0] != expected_batch:
            raise ValueError(
                f"{cache_name} layer {layer_idx} batch={key.shape[0]}, expected active batch={expected_batch}"
            )
    for layer_idx, value in enumerate(cache.value_cache):
        if value.shape[0] != expected_batch:
            raise ValueError(
                f"{cache_name} layer {layer_idx} value batch={value.shape[0]}, expected active batch={expected_batch}"
            )


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
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
        assert target_hidden.ndim == 3 and target_hidden.shape[0] == batch_size

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
    )
    return state, state.decode_start - prefill_start


@torch.inference_mode()
def draft_propose_step(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    state: BatchDecodeState,
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
                        :, active_draft_cache.get_seq_length() : active_start_pos_scalar + block_size
                    ].expand(active_batch, -1),
                    past_key_values=active_draft_cache,
                    use_cache=True,
                    is_causal=False,
                )
        with nvtx_range("draft.output_head"):
            draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
        assert draft_logits.ndim == 3 and draft_logits.shape[0] == block_output_ids.shape[0]
        with nvtx_range("kv_update.draft_cache"):
            active_draft_cache.crop(active_start_pos_scalar)
            validate_cache_batch_size(active_draft_cache, active_batch, "past_key_values_draft")
            scatter_back_rows(state.past_key_values_draft, active_draft_cache, active_indices)
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

    active_target_cache = gather_active_rows(state.past_key_values_target, active_indices)
    validate_cache_batch_size(active_target_cache, active_batch, "past_key_values_target")

    with profile_phase("target.verify"):
        with nvtx_range("target.verify.forward"):
            output = target(
                block_output_ids,
                position_ids=block_position_ids,
                past_key_values=active_target_cache,
                use_cache=True,
                output_hidden_states=True if block_size > 1 else False,
            )

    posterior = sample(output.logits, temperature)
    assert posterior.ndim == 2 and posterior.shape[0] == block_output_ids.shape[0]
    acceptance_len_vec = (block_output_ids[:, 1:] == posterior[:, :-1]).cumprod(dim=1).sum(dim=1)
    assert acceptance_len_vec.ndim == 1 and acceptance_len_vec.shape[0] == block_output_ids.shape[0]

    target_hidden = None
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)
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
        new_start_pos_scalar = int(state.start_pos[active_indices][0].item())
        active_target_cache.crop(new_start_pos_scalar)
        validate_cache_batch_size(active_target_cache, active_batch, "past_key_values_target")
        scatter_back_rows(state.past_key_values_target, active_target_cache, active_indices)
    if block_size > 1 and target_hidden is not None:
        if state.target_hidden is None:
            raise ValueError("target_hidden must be initialized when block_size > 1")
        for local_row, row in enumerate(active_indices.tolist()):
            acceptance_len = int(acceptance_len_vec[local_row].item())
            state.target_hidden[row] = target_hidden[local_row, : acceptance_len + 1, :]



def finalize_outputs(
    state: BatchDecodeState,
    mask_token_id: int,
    stop_token_ids: list[int] | None,
    time_to_first_token: float,
) -> list[SimpleNamespace]:
    assert state.output_ids.ndim == 2
    responses: list[SimpleNamespace] = []
    stop_tokens = None
    if stop_token_ids is not None:
        stop_tokens = torch.tensor(stop_token_ids, device=state.output_ids.device)

    for row in range(state.output_ids.shape[0]):
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
                if row_tokens.numel() == 0:
                    continue
                if torch.isin(row_tokens, stop_tokens).any():
                    hit_stop_token[row] = True

        state.finished_mask = hit_stop_token | reached_max_new_tokens | no_available_position

        state.active_indices = (~state.finished_mask).nonzero(as_tuple=True)[0]
        if state.active_indices.numel() == 0:
            break

        block_output_ids, block_position_ids, active_indices = draft_propose_step(
            model=model,
            target=target,
            state=state,
            block_size=block_size,
        )
        posterior, acceptance_len_vec, target_hidden, active_target_cache = target_verify_step(
            model=model,
            target=target,
            state=state,
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

    return finalize_outputs(
        state=state,
        mask_token_id=mask_token_id,
        stop_token_ids=stop_token_ids,
        time_to_first_token=time_to_first_token,
    )


def dflash_generate_batch(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    input_lengths: torch.Tensor,
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> list[SimpleNamespace]:
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
    args = parser.parse_args()

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
        batch_messages = [[] for _ in batch_instances]
        max_turns = max(len(instance["turns"]) for instance in batch_instances)

        for turn_index in range(max_turns):
            active_rows = []
            input_ids_batch = []
            for row, instance in enumerate(batch_instances):
                if turn_index >= len(instance["turns"]):
                    continue
                user_content = instance["turns"][turn_index]
                batch_messages[row].append({"role": "user", "content": user_content})
                input_text = tokenizer.apply_chat_template(
                    batch_messages[row],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                input_ids_batch.append(
                    tokenizer.encode(input_text, return_tensors="pt").to(target.device)
                )
                active_rows.append(row)

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
                    )

            for local_i, row in enumerate(active_rows):
                response = {
                    1: batch_response[1][local_i],
                    block_size: batch_response[block_size][local_i],
                }
                spec_response = response[block_size]
                generated_ids = spec_response.output_ids[0, spec_response.num_input_tokens :]
                output_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
                batch_messages[row].append({"role": "assistant", "content": output_text})
                responses.append(response)

    if dist.size() > 1:
        responses = dist.gather(responses, dst=0)
        if not dist.is_main():
            return
        responses = list(chain(*responses))

    t1 = np.mean([r[1].time_per_output_token for r in responses])
    tb = np.mean([r[block_size].time_per_output_token for r in responses])
    print(f"Decoding speedup: {t1 / tb:.2f}")

    tau = np.mean([np.mean(r[block_size].acceptance_lengths) for r in responses])
    print(f"Average Acceptance length: {tau:.2f}")

    acceptance_lengths = list(chain(*[r[block_size].acceptance_lengths for r in responses]))
    histogram = [
        acceptance_lengths.count(b) / len(acceptance_lengths)
        for b in range(block_size + 1)
    ]
    print(f"Acceptance length histogram: {[f'{x * 100:.1f}%' for x in histogram]}")


if __name__ == "__main__":
    main()
