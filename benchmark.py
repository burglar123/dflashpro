import argparse
import random
import time
from contextlib import contextmanager
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
) -> SimpleNamespace:
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids = torch.full(
        (1, max_length + block_size),
        mask_token_id,
        dtype=torch.long,
        device=model.device,
    )
    position_ids = torch.arange(output_ids.shape[1], device=model.device).unsqueeze(0)
    past_key_values_target = DynamicCache()
    past_key_values_draft = DynamicCache()

    # Prefill stage
    prefill_start = cuda_time()
    with nvtx_range("target.prefill.forward"):
        output = target(
            input_ids,
            position_ids=position_ids[:, :num_input_tokens],
            past_key_values=past_key_values_target,
            use_cache=True,
            logits_to_keep=1,
            output_hidden_states=True if block_size > 1 else False,
        )

    output_ids[:, :num_input_tokens] = input_ids
    output_ids[:, num_input_tokens : num_input_tokens + 1] = sample(output.logits, temperature)
    if block_size > 1:
        target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)

    time_to_first_token = cuda_time() - prefill_start

    # Decode stage
    decode_start = cuda_time()
    start = input_ids.shape[1]
    acceptance_lengths = []
    draft_prefill = True

    while start < max_length:
        block_output_ids = output_ids[:, start : start + block_size].clone()
        block_position_ids = position_ids[:, start : start + block_size]
        if block_size > 1:
            with profile_phase("draft"):
                with nvtx_range("draft.forward"):
                    noise_embedding = target.model.embed_tokens(block_output_ids)
                    draft_hidden = model(
                        target_hidden=target_hidden,
                        noise_embedding=noise_embedding,
                        position_ids=position_ids[
                            :, past_key_values_draft.get_seq_length() : start + block_size
                        ],
                        past_key_values=past_key_values_draft,
                        use_cache=True,
                        is_causal=False,
                    )
            with nvtx_range("draft.output_head"):
                draft_logits = target.lm_head(draft_hidden[:, -block_size + 1 :, :])
            with nvtx_range("kv_update.draft_cache"):
                past_key_values_draft.crop(start)
            block_output_ids[:, 1:] = sample(draft_logits)
            if draft_prefill:
                draft_prefill = False
                decode_start = cuda_time()

        with profile_phase("target.verify"):
            with nvtx_range("target.verify.forward"):
                output = target(
                    block_output_ids,
                    position_ids=block_position_ids,
                    past_key_values=past_key_values_target,
                    use_cache=True,
                    output_hidden_states=True if block_size > 1 else False,
                )

        posterior = sample(output.logits, temperature)
        acceptance_length = (
            (block_output_ids[:, 1:] == posterior[:, :-1])
            .cumprod(dim=1)
            .sum(dim=1)[0]
            .item()
        )
        output_ids[:, start : start + acceptance_length + 1] = block_output_ids[
            :, : acceptance_length + 1
        ]
        output_ids[:, start + acceptance_length + 1] = posterior[:, acceptance_length]

        acceptance_lengths.append(acceptance_length + 1)
        start += acceptance_length + 1
        with nvtx_range("kv_update.target_cache"):
            past_key_values_target.crop(start)
        if block_size > 1:
            target_hidden = extract_context_feature(output.hidden_states, model.target_layer_ids)[
                :, : acceptance_length + 1, :
            ]

        if stop_token_ids is not None and any(
            stop_token_id in output_ids[:, num_input_tokens:]
            for stop_token_id in stop_token_ids
        ):
            break

    output_ids = output_ids[:, :max_length]
    output_ids = output_ids[:, output_ids[0] != mask_token_id]
    if stop_token_ids is not None:
        stop_token_ids = torch.tensor(stop_token_ids, device=output_ids.device)
        stop_token_indices = torch.isin(
            output_ids[0][num_input_tokens:], stop_token_ids
        ).nonzero(as_tuple=True)[0]
        if stop_token_indices.numel() > 0:
            output_ids = output_ids[:, : num_input_tokens + stop_token_indices[0] + 1]

    num_output_tokens = output_ids.shape[1] - num_input_tokens
    total_decode_time = cuda_time() - decode_start
    time_per_output_token = total_decode_time / num_output_tokens

    return SimpleNamespace(
        output_ids=output_ids,
        num_input_tokens=num_input_tokens,
        num_output_tokens=num_output_tokens,
        time_to_first_token=time_to_first_token,
        time_per_output_token=time_per_output_token,
        acceptance_lengths=acceptance_lengths,
    )


def dflash_generate_batch(
    model: DFlashDraftModel,
    target: AutoModelForCausalLM,
    input_ids_batch: list[torch.Tensor],
    mask_token_id: int,
    max_new_tokens: int,
    block_size: int,
    stop_token_ids: list[int],
    temperature: float = 0.0,
) -> list[SimpleNamespace]:
    responses = []
    for input_ids in input_ids_batch:
        responses.append(
            dflash_generate(
                model=model,
                target=target,
                input_ids=input_ids,
                mask_token_id=mask_token_id,
                max_new_tokens=max_new_tokens,
                block_size=block_size,
                stop_token_ids=stop_token_ids,
                temperature=temperature,
            )
        )
    return responses


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

            batch_response = {}
            for bs in [1, block_size]:
                with nvtx_range(f"generate.block_size_{bs}"):
                    batch_response[bs] = dflash_generate_batch(
                        model=draft_model,
                        target=target,
                        input_ids_batch=input_ids_batch,
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
