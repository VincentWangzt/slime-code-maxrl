from types import SimpleNamespace

import pytest
import torch

from slime.backends.megatron_utils import loss as loss_module
from slime.backends.megatron_utils import model as model_module
from slime.backends.megatron_utils.model_provider import LinearForLastLayer
from slime.ray.rollout import _tensorize_rollout_data_for_training
from slime.rollout import regression_rollout
from slime.utils.regression import (
    inverse_regression_prediction,
    merge_indexed_regression_predictions,
    transform_regression_target,
    uses_scalar_head,
)
from slime.utils.types import Sample


@pytest.mark.unit
@pytest.mark.parametrize(
    ("value", "transform", "expected"),
    [
        (-2.5, "identity", -2.5),
        (0.0, "log10p", 0.0),
        (9.0, "log10p", 1.0),
        (99.0, "log10p", 2.0),
    ],
)
def test_regression_target_transforms(value, transform, expected):
    assert transform_regression_target(value, transform) == pytest.approx(expected)


@pytest.mark.unit
@pytest.mark.parametrize("value", [None, "not-a-number", float("nan"), float("inf")])
def test_regression_target_transform_rejects_invalid_values(value):
    with pytest.raises(ValueError):
        transform_regression_target(value, "identity")


@pytest.mark.unit
def test_log10p_rejects_negative_target_but_inverse_keeps_negative_prediction():
    with pytest.raises(ValueError, match="non-negative"):
        transform_regression_target(-1, "log10p")
    assert inverse_regression_prediction(-1.0, "log10p") == pytest.approx(-0.9)
    with pytest.raises(ValueError, match="overflowed"):
        inverse_regression_prediction(1e308, "log10p")


@pytest.mark.unit
def test_terminal_extractor_uses_each_packed_sample_tail_and_ignores_padding():
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1)

    predictions = loss_module.extract_last_token_predictions(logits, [2, 3])

    assert predictions.tolist() == [1.0, 4.0]


@pytest.mark.unit
def test_regression_loss_reduces_one_scalar_per_sample_and_only_tail_has_gradient(monkeypatch):
    monkeypatch.setattr(loss_module.mpu, "get_context_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        loss_module.mpu,
        "get_data_parallel_world_size",
        lambda with_context_parallel: 1,
    )
    logits = torch.arange(8, dtype=torch.float32).reshape(1, 8, 1).requires_grad_()
    batch = {
        "total_lengths": [2, 3],
        "response_lengths": [1, 1],
        "loss_masks": [torch.ones(1), torch.ones(1)],
        "rollout_mask_sums": torch.ones(2),
        "regression_targets": [torch.tensor([3.0]), torch.tensor([6.0])],
    }
    args = SimpleNamespace(
        loss_type="regression_loss",
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        allgather_cp=False,
    )

    scaled_loss, _, log = loss_module.loss_function(
        args,
        batch,
        num_microbatches=2,
        step_global_batch_size=4,
        logits=logits,
    )
    scaled_loss.backward()

    # Errors are (1 - 3)^2 and (4 - 6)^2. Megatron's two-microbatch
    # scaling returns this microbatch's global-mean contribution.
    assert scaled_loss.item() == pytest.approx(4.0)
    assert log["keys"] == ["loss", "mse"]
    nonzero_gradient_positions = torch.nonzero(logits.grad.reshape(-1), as_tuple=False).reshape(-1).tolist()
    assert nonzero_gradient_positions == [1, 4]


@pytest.mark.unit
def test_scalar_head_is_biased_and_uses_existing_initialization():
    config = SimpleNamespace(sequence_parallel=False, init_method_std=0.01)
    head = LinearForLastLayer(2560, 1, config=config)

    assert tuple(head.weight.shape) == (1, 2560)
    assert tuple(head.bias.shape) == (1,)
    assert torch.count_nonzero(head.bias).item() == 0
    logits, bias = head(torch.zeros(2, 2560))
    assert logits.dtype == torch.float32
    assert tuple(logits.shape) == (2, 1)
    assert bias is None


def _checkpoint_reinit_result(monkeypatch, tmp_path, *, role, loss_type, checkpoint_shapes):
    checkpoint_path = tmp_path / "iter_0000001"
    checkpoint_path.mkdir()
    (checkpoint_path / ".metadata").write_text("metadata", encoding="utf-8")
    monkeypatch.setattr(model_module, "get_load_checkpoint_path_by_args", lambda args: checkpoint_path)
    monkeypatch.setattr(model_module, "unwrap_model", lambda model: model)

    import megatron.core.dist_checkpointing.serialization as serialization

    metadata = {
        name: SimpleNamespace(global_shape=shape)
        for name, shape in checkpoint_shapes.items()
    }
    monkeypatch.setattr(serialization, "load_tensors_metadata", lambda path: metadata)
    runtime_model = [SimpleNamespace(output_layer=torch.nn.Linear(4, 1, bias=True))]
    args = SimpleNamespace(load=str(tmp_path), loss_type=loss_type)
    return model_module._scalar_output_layer_needs_reinit(args, runtime_model, role)


@pytest.mark.unit
def test_base_lm_head_is_reinitialized_for_regression_actor(monkeypatch, tmp_path):
    assert _checkpoint_reinit_result(
        monkeypatch,
        tmp_path,
        role="actor",
        loss_type="regression_loss",
        checkpoint_shapes={"model.output_layer.weight": (10, 4)},
    )


@pytest.mark.unit
def test_scalar_checkpoint_preserves_trained_head_on_resume(monkeypatch, tmp_path):
    assert not _checkpoint_reinit_result(
        monkeypatch,
        tmp_path,
        role="actor",
        loss_type="regression_loss",
        checkpoint_shapes={
            "model.output_layer.weight": (1, 4),
            "model.output_layer.bias": (1,),
        },
    )


@pytest.mark.unit
def test_existing_critic_scalar_head_behavior_is_unchanged(monkeypatch, tmp_path):
    assert uses_scalar_head(SimpleNamespace(loss_type="policy_loss"), "critic")
    assert not uses_scalar_head(SimpleNamespace(loss_type="policy_loss"), "actor")
    assert _checkpoint_reinit_result(
        monkeypatch,
        tmp_path,
        role="critic",
        loss_type="policy_loss",
        checkpoint_shapes={"model.output_layer.weight": (10, 4)},
    )


@pytest.mark.unit
def test_regression_rollout_keeps_label_and_marks_one_terminal_selection(monkeypatch):
    tokenizer = SimpleNamespace(encode=lambda prompt, add_special_tokens: [11, 12, 13])
    monkeypatch.setattr(regression_rollout, "_TOKENIZER", tokenizer)
    sample = Sample(index=7, prompt="rendered assistant prefix", label=9.0)

    regression_rollout.prepare_regression_sample(SimpleNamespace(hf_checkpoint="unused"), sample)

    assert sample.tokens == [11, 12, 13]
    assert sample.label == 9.0
    assert sample.response_length == 1
    assert sample.loss_mask == [1]
    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.unit
def test_regression_targets_survive_cpu_tensorization():
    rollout_data = {"regression_targets": [1.25, 2.5]}

    _tensorize_rollout_data_for_training(rollout_data)

    assert [target.dtype for target in rollout_data["regression_targets"]] == [torch.float32, torch.float32]
    assert [target.tolist() for target in rollout_data["regression_targets"]] == [[1.25], [2.5]]


@pytest.mark.unit
def test_indexed_prediction_merge_validates_coverage_duplicates_and_finiteness():
    assert merge_indexed_regression_predictions([0, 1], [(1, 2.0), (0, 1.0)]) == {0: 1.0, 1: 2.0}
    with pytest.raises(ValueError, match="Duplicate"):
        merge_indexed_regression_predictions([0], [(0, 1.0), (0, 2.0)])
    with pytest.raises(ValueError, match="coverage"):
        merge_indexed_regression_predictions([0, 1], [(0, 1.0)])
    with pytest.raises(ValueError, match="non-finite"):
        merge_indexed_regression_predictions([0], [(0, float("nan"))])
