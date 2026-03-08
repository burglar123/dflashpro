from types import SimpleNamespace

import pytest
import torch
from transformers import DynamicCache

from benchmark import (
    BlockManager,
    Scheduler,
    Sequence,
    dflash_generate,
    dflash_generate_batch,
    gather_active_rows,
    init_batch_state,
    prepare_packed_verify_inputs,
    scatter_back_rows,
    sync_sequence_blocks,
    target_verify_step,
)


class DummyDraftModel:
    def __init__(self, device: torch.device):
        self.device = device
        self.mask_token_id = -1
        self.target_layer_ids = [0]

    def __call__(
        self,
        target_hidden: torch.Tensor,
        noise_embedding: torch.Tensor,
        position_ids: torch.Tensor,
        past_key_values=None,
        use_cache=True,
        is_causal=False,
    ) -> torch.Tensor:
        return noise_embedding


class DummyTargetModel:
    def __init__(self, vocab_size: int = 256):
        self.vocab_size = vocab_size
        self.device = torch.device("cpu")
        self.model = SimpleNamespace(embed_tokens=lambda token_ids: torch.zeros((*token_ids.shape, 8)))

    def lm_head(self, hidden_states: torch.Tensor) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        return torch.zeros((bsz, seq_len, self.vocab_size), dtype=torch.float32)

    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=None,
        output_hidden_states=False,
        **kwargs,
    ):
        bsz, seq_len = input_ids.shape
        logits = torch.full((bsz, seq_len, self.vocab_size), -1e9, dtype=torch.float32)
        next_ids = (input_ids + 1) % self.vocab_size
        logits.scatter_(2, next_ids.unsqueeze(-1), 0.0)

        hidden = torch.zeros((bsz, seq_len, 8), dtype=torch.float32)
        hidden_states = [hidden, hidden] if output_hidden_states else None

        if past_key_values is not None:
            if not hasattr(past_key_values, "key_cache"):
                past_key_values.key_cache = []
                past_key_values.value_cache = []
            if not past_key_values.key_cache:
                past_key_values.key_cache = [torch.zeros((bsz, 1, seq_len, 4), dtype=torch.float32)]
                past_key_values.value_cache = [torch.zeros((bsz, 1, seq_len, 4), dtype=torch.float32)]

        return SimpleNamespace(logits=logits, hidden_states=hidden_states)




class DummyTargetModelNoPackedKwargs(DummyTargetModel):
    def __call__(
        self,
        input_ids: torch.Tensor,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        use_cache=True,
        logits_to_keep=None,
        output_hidden_states=False,
    ):
        return super().__call__(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            logits_to_keep=logits_to_keep,
            output_hidden_states=output_hidden_states,
        )
def run_single(model, target, prompt_ids, max_new_tokens=8):
    with torch.inference_mode():
        out = dflash_generate(
            model=model,
            target=target,
            input_ids=prompt_ids,
            mask_token_id=model.mask_token_id,
            max_new_tokens=max_new_tokens,
            block_size=1,
            stop_token_ids=[255],
            temperature=0.0,
        )[0]
    return out.output_ids[0]


def test_batch1_matches_single_path_temperature_zero():
    torch.manual_seed(0)
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    prompt = torch.tensor([[10, 20, 30]], dtype=torch.long)

    solo_ids = run_single(model, target, prompt)

    with torch.inference_mode():
        batch_out = dflash_generate_batch(
            model=model,
            target=target,
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            input_lengths=torch.tensor([prompt.shape[1]], dtype=torch.long),
            mask_token_id=model.mask_token_id,
            max_new_tokens=8,
            block_size=1,
            stop_token_ids=[255],
            temperature=0.0,
        )[0]

    assert torch.equal(solo_ids, batch_out.output_ids[0])


def test_batch2_and_batch4_no_cross_contamination():
    torch.manual_seed(0)
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    prompts = torch.tensor(
        [
            [1, 2, 3],
            [10, 11, 12],
            [42, 43, 44],
            [100, 101, 102],
        ],
        dtype=torch.long,
    )

    for batch_size in (2, 4):
        batch_prompts = prompts[:batch_size]
        with torch.inference_mode():
            batch_responses = dflash_generate_batch(
                model=model,
                target=target,
                input_ids=batch_prompts,
                attention_mask=torch.ones_like(batch_prompts),
                input_lengths=torch.full((batch_size,), batch_prompts.shape[1], dtype=torch.long),
                mask_token_id=model.mask_token_id,
                max_new_tokens=6,
                block_size=1,
                stop_token_ids=[255],
                temperature=0.0,
            )

        for row in range(batch_size):
            row_tokens = batch_responses[row].output_ids[0]
            assert (row_tokens >= 0).all(), "generated row must not contain mask token"

            solo = run_single(model, target, batch_prompts[row : row + 1], max_new_tokens=6)
            assert torch.equal(row_tokens, solo), f"row {row} changed under batch={batch_size}"


def test_stage_modes_are_rollback_safe():
    torch.manual_seed(0)
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)
    prompt = torch.tensor(
        [
            [3, 4, 5],
            [30, 31, 32],
        ],
        dtype=torch.long,
    )

    for mode in [
        "legacy",
        "stage_a_prefill_only",
        "stage_b_target_only",
        "stage_c_full_speculative",
    ]:
        with torch.inference_mode():
            responses = dflash_generate_batch(
                model=model,
                target=target,
                input_ids=prompt,
                attention_mask=torch.ones_like(prompt),
                input_lengths=torch.full((prompt.shape[0],), prompt.shape[1], dtype=torch.long),
                mask_token_id=model.mask_token_id,
                max_new_tokens=5,
                block_size=1,
                stop_token_ids=[255],
                temperature=0.0,
                batched_decode_mode=mode,
            )
        assert len(responses) == prompt.shape[0]
        for row in range(prompt.shape[0]):
            row_tokens = responses[row].output_ids[0]
            assert torch.equal(row_tokens[: prompt.shape[1]], prompt[row])
            assert (row_tokens >= 0).all()


def test_cache_alignment_after_active_row_compaction_and_scatter():
    full_cache = DynamicCache()
    full_cache.key_cache = [
        torch.arange(5 * 2 * 4 * 3, dtype=torch.float32).reshape(5, 2, 4, 3)
    ]
    full_cache.value_cache = [
        torch.arange(5 * 2 * 4 * 3, dtype=torch.float32).reshape(5, 2, 4, 3) + 10_000
    ]

    active_indices = torch.tensor([1, 3], dtype=torch.long)
    active_cache = gather_active_rows(full_cache, active_indices)

    # Simulate per-token updates in compacted active cache.
    active_cache.key_cache[0][:, :, 2:, :] = -7
    active_cache.value_cache[0][:, :, 1:3, :] = -9

    original_inactive_key = full_cache.key_cache[0][0].clone()
    original_inactive_value = full_cache.value_cache[0][4].clone()

    scatter_back_rows(full_cache, active_cache, active_indices)

    assert torch.equal(full_cache.key_cache[0][0], original_inactive_key)
    assert torch.equal(full_cache.value_cache[0][4], original_inactive_value)

    assert torch.equal(full_cache.key_cache[0][1, :, 2:, :], torch.full((2, 2, 3), -7.0))
    assert torch.equal(full_cache.key_cache[0][3, :, 2:, :], torch.full((2, 2, 3), -7.0))
    assert torch.equal(full_cache.value_cache[0][1, :, 1:3, :], torch.full((2, 2, 3), -9.0))
    assert torch.equal(full_cache.value_cache[0][3, :, 1:3, :], torch.full((2, 2, 3), -9.0))


def test_block_manager_prefix_cache_reuses_blocks_and_tracks_refcount():
    manager = BlockManager(block_size=2)
    seq_a = Sequence(
        seq_id=0,
        token_ids=[1, 2, 3, 4, 5],
        num_cached_tokens=4,
        block_table=[],
        pre_verify=True,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )
    seq_b = Sequence(
        seq_id=1,
        token_ids=[1, 2, 3, 9, 10],
        num_cached_tokens=4,
        block_table=[],
        pre_verify=True,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )

    sync_sequence_blocks(seq_a, manager)
    sync_sequence_blocks(seq_b, manager)

    assert seq_a.block_table[0].physical_block_id == seq_b.block_table[0].physical_block_id
    assert seq_a.block_table[1].physical_block_id != seq_b.block_table[1].physical_block_id


def test_block_manager_rollback_and_consistency():
    manager = BlockManager(block_size=3)
    seq = Sequence(
        seq_id=0,
        token_ids=[7, 8, 9, 10, 11, 12],
        num_cached_tokens=6,
        block_table=[],
        pre_verify=False,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )

    sync_sequence_blocks(seq, manager)
    seq.num_cached_tokens = 4
    manager.rollback_sequence(seq, seq.num_cached_tokens)
    sync_sequence_blocks(seq, manager)

    assert len(seq.block_table) == 2
    assert seq.block_table[1].logical_start == 3
    assert seq.block_table[1].logical_end == 4


def test_prepare_packed_verify_inputs_respects_pre_verify_lengths():
    seqs = [
        Sequence(
            seq_id=0,
            token_ids=[1, 2, 3],
            num_cached_tokens=2,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[4, 5, 6, 7],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]
    packed = prepare_packed_verify_inputs(seqs, gamma=4)
    assert packed["verify_lens"] == [1, 4]
    assert packed["packed_ranges"] == [(0, 1), (1, 5)]
    assert packed["slot_mapping"] == [0, 1, 1, 1, 1]


def test_target_verify_step_handles_ragged_verify_lengths_without_misalignment():
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    state, _ = init_batch_state(
        model=model,
        target=target,
        input_ids=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        input_lengths=torch.tensor([3, 3], dtype=torch.long),
        mask_token_id=model.mask_token_id,
        max_new_tokens=4,
        block_size=3,
        temperature=0.0,
    )

    scheduled_batch = [
        Sequence(
            seq_id=0,
            token_ids=[10, 11, 12, 13],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[20, 21, 22, 23],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]

    active_indices = torch.tensor([0, 1], dtype=torch.long)
    block_output_ids = torch.tensor([[13, 99, 99], [23, 24, 25]], dtype=torch.long)
    block_position_ids = torch.tensor([[3, 4, 5], [3, 4, 5]], dtype=torch.long)

    posterior, acceptance_len_vec, _, _ = target_verify_step(
        model=model,
        target=target,
        state=state,
        scheduled_batch=scheduled_batch,
        active_indices=active_indices,
        block_output_ids=block_output_ids,
        block_position_ids=block_position_ids,
        block_size=3,
        temperature=0.0,
    )

    assert acceptance_len_vec.tolist() == [0, 2]
    assert posterior[0, 0].item() == 14
    assert posterior[1, 0].item() == 24



def test_target_verify_step_falls_back_when_target_does_not_support_packed_metadata_kwargs():
    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModelNoPackedKwargs(vocab_size=256)

    state, _ = init_batch_state(
        model=model,
        target=target,
        input_ids=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        input_lengths=torch.tensor([3, 3], dtype=torch.long),
        mask_token_id=model.mask_token_id,
        max_new_tokens=4,
        block_size=3,
        temperature=0.0,
    )

    scheduled_batch = [
        Sequence(
            seq_id=0,
            token_ids=[10, 11, 12, 13],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[20, 21, 22, 23],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]

    posterior, acceptance_len_vec, _, _ = target_verify_step(
        model=model,
        target=target,
        state=state,
        scheduled_batch=scheduled_batch,
        active_indices=torch.tensor([0, 1], dtype=torch.long),
        block_output_ids=torch.tensor([[13, 99, 99], [23, 24, 25]], dtype=torch.long),
        block_position_ids=torch.tensor([[3, 4, 5], [3, 4, 5]], dtype=torch.long),
        block_size=3,
        temperature=0.0,
    )

    assert acceptance_len_vec.tolist() == [0, 2]
    assert posterior.shape == torch.Size([2, 3])


def test_stage_c_mode_auto_falls_back_to_stage_b_when_packed_metadata_is_unsupported(monkeypatch):
    import benchmark as benchmark_module

    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModelNoPackedKwargs(vocab_size=256)
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    called = {"stage_b": False}

    def fake_stage_b(**kwargs):
        called["stage_b"] = True
        return [SimpleNamespace(output_ids=prompt, num_input_tokens=3, num_output_tokens=0, time_to_first_token=0.0, time_per_output_token=0.0, acceptance_lengths=[], active_batch_size_trace=[])]

    monkeypatch.setattr(benchmark_module, "dflash_generate_batch_stage_b_target_only", fake_stage_b)

    benchmark_module.dflash_generate_batch_with_mode(
        model=model,
        target=target,
        input_ids=prompt,
        attention_mask=torch.ones_like(prompt),
        input_lengths=torch.tensor([3], dtype=torch.long),
        mask_token_id=model.mask_token_id,
        max_new_tokens=2,
        block_size=2,
        stop_token_ids=[255],
        temperature=0.0,
        batched_decode_mode="stage_c_full_speculative",
    )

    assert called["stage_b"] is True

def _build_sequence(seq_id: int, token_ids: list[int], num_cached_tokens: int) -> Sequence:
    return Sequence(
        seq_id=seq_id,
        token_ids=list(token_ids),
        num_cached_tokens=num_cached_tokens,
        block_table=[],
        pre_verify=False,
        num_acc_tokens=0,
        finished=False,
        pending_kv_append=[],
    )


def test_scheduler_and_block_manager_rollback_are_atomic():
    manager = BlockManager(block_size=2)
    seq = _build_sequence(seq_id=0, token_ids=[10, 11], num_cached_tokens=2)
    manager.register_sequence(seq)
    sync_sequence_blocks(seq, manager)

    scheduler = Scheduler([seq], block_manager=manager)
    scheduler.schedule_next_batch()
    scheduler.append_draft_tokens(seq_id=0, draft_tokens=[12, 13, 14])

    scheduler.rollback(seq_id=0, rollback_to=3)

    assert seq.num_cached_tokens == 3
    assert seq.token_ids == [10, 11, 12]
    assert all(mapping.logical_start < 3 for mapping in seq.block_table)
    manager.check_consistency(seq)


def test_transactional_accept_reject_paths_cover_all_reject_mid_reject_all_accept():
    scenarios = [
        ("all_reject", 0, [20, 21, 99]),
        ("mid_reject", 2, [20, 21, 30, 31, 99]),
        ("all_accept", 4, [20, 21, 30, 31, 32, 33, 99]),
    ]

    for _, accepted_draft_len, expected_tokens in scenarios:
        manager = BlockManager(block_size=2)
        seq = _build_sequence(seq_id=0, token_ids=[20, 21], num_cached_tokens=2)
        manager.register_sequence(seq)
        sync_sequence_blocks(seq, manager)
        scheduler = Scheduler([seq], block_manager=manager)
        scheduler.schedule_next_batch()

        draft_tokens = [30, 31, 32, 33]
        scheduler.append_draft_tokens(seq_id=0, draft_tokens=draft_tokens)
        txn = scheduler.consume_draft_transaction(seq_id=0)
        draft_start = int(txn["start"])

        scheduler.rollback(seq_id=0, rollback_to=draft_start + accepted_draft_len)
        seq.token_ids = seq.token_ids[: seq.num_cached_tokens] + [99]
        seq.num_cached_tokens += 1
        sync_sequence_blocks(seq, manager)

        assert seq.token_ids == expected_tokens
        assert seq.num_cached_tokens == len(expected_tokens)
        manager.check_consistency(seq)


def test_eos_hit_removes_sequence_from_running_and_releases_recyclable_blocks():
    manager = BlockManager(block_size=2)
    seq0 = _build_sequence(seq_id=0, token_ids=[1, 2], num_cached_tokens=2)
    seq1 = _build_sequence(seq_id=1, token_ids=[3, 4], num_cached_tokens=2)
    for seq in (seq0, seq1):
        manager.register_sequence(seq)
        sync_sequence_blocks(seq, manager)

    scheduler = Scheduler([seq0, seq1], block_manager=manager)
    scheduler.schedule_next_batch()

    scheduler.append_draft_tokens(seq_id=0, draft_tokens=[5, 6, 7, 8])
    free_before = len(manager._free_block_ids)
    scheduler.rollback(seq_id=0, rollback_to=2)
    free_after = len(manager._free_block_ids)

    scheduler.mark_finished(seq0)

    assert seq0.finished is True
    assert seq0 not in scheduler.running
    assert seq0 in scheduler.finished
    assert free_after >= free_before + 2


@pytest.mark.parametrize(
    "block_size,acceptance_len",
    [
        (2, 0),
        (2, 1),
        (4, 0),
        (4, 2),
        (4, 3),
    ],
)
def test_stage_c_acceptance_updates_target_hidden_slice_e2e(monkeypatch, block_size: int, acceptance_len: int):
    import benchmark as benchmark_module

    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    original_target_verify_step = benchmark_module.target_verify_step
    original_apply_acceptance_step = benchmark_module.apply_acceptance_step
    apply_calls = []

    def fake_target_verify_step(
        model,
        target,
        state,
        scheduled_batch,
        active_indices,
        block_output_ids,
        block_position_ids,
        block_size,
        temperature,
    ):
        active_batch = int(active_indices.numel())
        posterior = torch.full((active_batch, block_size), 77, dtype=torch.long, device=state.output_ids.device)
        acceptance_len_vec = torch.full(
            (active_batch,),
            acceptance_len,
            dtype=torch.long,
            device=state.output_ids.device,
        )
        hidden_dim = int(state.target_hidden.shape[-1])
        target_hidden = torch.arange(
            active_batch * block_size * hidden_dim,
            dtype=state.target_hidden.dtype,
            device=state.target_hidden.device,
        ).reshape(active_batch, block_size, hidden_dim)
        active_target_cache = benchmark_module.gather_active_rows(state.past_key_values_target, active_indices)
        return posterior, acceptance_len_vec, target_hidden, active_target_cache

    def wrapped_apply_acceptance_step(
        state,
        active_indices,
        block_output_ids,
        posterior,
        acceptance_len_vec,
        target_hidden,
        active_target_cache,
        block_size,
    ):
        assert target_hidden is not None
        row = int(active_indices[0].item())
        row_start_pos = int(state.start_pos[row].item())
        local_acceptance_len = int(acceptance_len_vec[0].item())
        row_end_pos = row_start_pos + local_acceptance_len + 1
        before_row = state.target_hidden[row].clone()

        original_apply_acceptance_step(
            state=state,
            active_indices=active_indices,
            block_output_ids=block_output_ids,
            posterior=posterior,
            acceptance_len_vec=acceptance_len_vec,
            target_hidden=target_hidden,
            active_target_cache=active_target_cache,
            block_size=block_size,
        )

        after_row = state.target_hidden[row]
        assert torch.equal(after_row[:row_start_pos], before_row[:row_start_pos])
        assert torch.equal(after_row[row_end_pos:], before_row[row_end_pos:])
        assert torch.equal(
            after_row[row_start_pos:row_end_pos],
            target_hidden[0, : local_acceptance_len + 1, :],
        )
        apply_calls.append(local_acceptance_len)

    monkeypatch.setattr(benchmark_module, "target_verify_step", fake_target_verify_step)
    monkeypatch.setattr(benchmark_module, "apply_acceptance_step", wrapped_apply_acceptance_step)

    with torch.inference_mode():
        responses = benchmark_module.dflash_generate_batch(
            model=model,
            target=target,
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            input_lengths=torch.tensor([prompt.shape[1]], dtype=torch.long),
            mask_token_id=model.mask_token_id,
            max_new_tokens=1,
            block_size=block_size,
            stop_token_ids=None,
            temperature=0.0,
            batched_decode_mode="stage_c_full_speculative",
        )

    assert len(responses) == 1
    assert apply_calls == [acceptance_len]

    monkeypatch.setattr(benchmark_module, "target_verify_step", original_target_verify_step)
    monkeypatch.setattr(benchmark_module, "apply_acceptance_step", original_apply_acceptance_step)


def test_target_verify_step_uses_packed_verify_lengths_for_pre_verify_rows():
    class RecordingTargetModel(DummyTargetModel):
        def __init__(self, vocab_size: int = 256):
            super().__init__(vocab_size=vocab_size)
            self.last_input_shape = None
            self.last_attention_mask = None
            self.last_slot_mapping = None
            self.last_context_lens = None
            self.last_block_tables = None

        def __call__(self, input_ids: torch.Tensor, *args, **kwargs):
            self.last_input_shape = tuple(input_ids.shape)
            self.last_attention_mask = kwargs.get("attention_mask")
            self.last_slot_mapping = kwargs.get("slot_mapping")
            self.last_context_lens = kwargs.get("context_lens")
            self.last_block_tables = kwargs.get("block_tables")
            return super().__call__(input_ids, *args, **kwargs)

    model = DummyDraftModel(device=torch.device("cpu"))
    target = RecordingTargetModel(vocab_size=256)

    state, _ = init_batch_state(
        model=model,
        target=target,
        input_ids=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        input_lengths=torch.tensor([3, 3], dtype=torch.long),
        mask_token_id=model.mask_token_id,
        max_new_tokens=4,
        block_size=4,
        temperature=0.0,
    )

    scheduled_batch = [
        Sequence(
            seq_id=0,
            token_ids=[10, 11, 12, 13],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[20, 21, 22, 23],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]

    active_indices = torch.tensor([0, 1], dtype=torch.long)
    block_output_ids = torch.tensor([[13, 99, 99, 99], [23, 24, 25, 26]], dtype=torch.long)
    block_position_ids = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 6]], dtype=torch.long)

    target_verify_step(
        model=model,
        target=target,
        state=state,
        scheduled_batch=scheduled_batch,
        active_indices=active_indices,
        block_output_ids=block_output_ids,
        block_position_ids=block_position_ids,
        block_size=4,
        temperature=0.0,
    )

    assert target.last_input_shape == (2, 4)
    assert target.last_attention_mask is not None
    assert target.last_attention_mask.tolist() == [[1, 0, 0, 0], [1, 1, 1, 1]]
    assert target.last_slot_mapping == [0, 1, 1, 1, 1]
    assert target.last_context_lens == [3, 3]
    assert target.last_block_tables == [[], []]


def test_target_verify_step_validates_cache_batch_alignment_for_sequence_packed_inputs(monkeypatch):
    import benchmark as benchmark_module

    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)

    state, _ = init_batch_state(
        model=model,
        target=target,
        input_ids=torch.tensor([[10, 11, 12], [20, 21, 22]], dtype=torch.long),
        attention_mask=torch.ones((2, 3), dtype=torch.long),
        input_lengths=torch.tensor([3, 3], dtype=torch.long),
        mask_token_id=model.mask_token_id,
        max_new_tokens=4,
        block_size=4,
        temperature=0.0,
    )

    scheduled_batch = [
        Sequence(
            seq_id=0,
            token_ids=[10, 11, 12, 13],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=True,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
        Sequence(
            seq_id=1,
            token_ids=[20, 21, 22, 23],
            num_cached_tokens=3,
            block_table=[],
            pre_verify=False,
            num_acc_tokens=0,
            finished=False,
            pending_kv_append=[],
        ),
    ]

    wrong_cache = DynamicCache()
    wrong_cache.key_cache = [torch.zeros((1, 1, 3, 4), dtype=torch.float32)]
    wrong_cache.value_cache = [torch.zeros((1, 1, 3, 4), dtype=torch.float32)]

    monkeypatch.setattr(benchmark_module, "gather_active_rows", lambda cache, active_indices: wrong_cache)

    active_indices = torch.tensor([0, 1], dtype=torch.long)
    block_output_ids = torch.tensor([[13, 99, 99, 99], [23, 24, 25, 26]], dtype=torch.long)
    block_position_ids = torch.tensor([[3, 4, 5, 6], [3, 4, 5, 6]], dtype=torch.long)

    with pytest.raises(ValueError, match="target verify batch mismatch"):
        benchmark_module.target_verify_step(
            model=model,
            target=target,
            state=state,
            scheduled_batch=scheduled_batch,
            active_indices=active_indices,
            block_output_ids=block_output_ids,
            block_position_ids=block_position_ids,
            block_size=4,
            temperature=0.0,
        )


@pytest.mark.parametrize(
    "scenario,acceptance_len,verify_token,stop_tokens,expect_stop",
    [
        ("all_reject", 0, 150, [255], False),
        ("mid_reject", 1, 151, [255], False),
        ("all_accept", 2, 152, [255], False),
        ("all_accept_with_eos", 2, 199, [199], True),
    ],
)
def test_stage_c_rollback_paths_are_stable_with_acceptance_and_eos(
    monkeypatch,
    scenario: str,
    acceptance_len: int,
    verify_token: int,
    stop_tokens: list[int],
    expect_stop: bool,
):
    import benchmark as benchmark_module

    model = DummyDraftModel(device=torch.device("cpu"))
    target = DummyTargetModel(vocab_size=256)
    prompt = torch.tensor([[10, 11, 12]], dtype=torch.long)

    original_target_verify_step = benchmark_module.target_verify_step

    def scripted_target_verify_step(
        model,
        target,
        state,
        scheduled_batch,
        active_indices,
        block_output_ids,
        block_position_ids,
        block_size,
        temperature,
    ):
        active_batch = int(active_indices.numel())
        posterior = torch.full(
            (active_batch, block_size),
            verify_token,
            dtype=torch.long,
            device=state.output_ids.device,
        )
        acceptance_len_vec = torch.full(
            (active_batch,),
            acceptance_len,
            dtype=torch.long,
            device=state.output_ids.device,
        )
        active_target_cache = benchmark_module.gather_active_rows(state.past_key_values_target, active_indices)
        hidden_dim = int(state.target_hidden.shape[-1])
        target_hidden = torch.zeros(
            (active_batch, block_size, hidden_dim),
            dtype=state.target_hidden.dtype,
            device=state.target_hidden.device,
        )
        return posterior, acceptance_len_vec, target_hidden, active_target_cache

    monkeypatch.setattr(benchmark_module, "target_verify_step", scripted_target_verify_step)

    with torch.inference_mode():
        first = benchmark_module.dflash_generate_batch(
            model=model,
            target=target,
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            input_lengths=torch.tensor([prompt.shape[1]], dtype=torch.long),
            mask_token_id=model.mask_token_id,
            max_new_tokens=1,
            block_size=3,
            stop_token_ids=stop_tokens,
            temperature=0.0,
            batched_decode_mode="stage_c_full_speculative",
        )[0].output_ids[0]
        second = benchmark_module.dflash_generate_batch(
            model=model,
            target=target,
            input_ids=prompt,
            attention_mask=torch.ones_like(prompt),
            input_lengths=torch.tensor([prompt.shape[1]], dtype=torch.long),
            mask_token_id=model.mask_token_id,
            max_new_tokens=1,
            block_size=3,
            stop_token_ids=stop_tokens,
            temperature=0.0,
            batched_decode_mode="stage_c_full_speculative",
        )[0].output_ids[0]

    assert torch.equal(first, second), f"scenario {scenario} output drifted after rollback"
    if expect_stop:
        assert verify_token in first.tolist()
