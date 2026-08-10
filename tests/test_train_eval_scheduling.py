from types import SimpleNamespace

import pytest

import train as train_module
import train_async as train_async_module


class _RemoteMethod:
    def __init__(self, callback):
        self._callback = callback

    def remote(self, *args, **kwargs):
        return self._callback(*args, **kwargs)


class _RolloutManager:
    def __init__(self, events):
        self.eval = _RemoteMethod(lambda rollout_id: events.append(("eval", rollout_id)))
        self.prepare_regression_eval = _RemoteMethod(
            lambda rollout_id: (events.append(("prepare_regression_eval", rollout_id)), "eval-data")[1]
        )
        self.finish_regression_eval = _RemoteMethod(
            lambda rollout_id, predictions: events.append(("finish_regression_eval", rollout_id, predictions))
        )
        self.generate = _RemoteMethod(lambda rollout_id: (events.append(("generate", rollout_id)), rollout_id))
        self.dispose = _RemoteMethod(lambda: events.append(("dispose", None)))


class _ActorModel:
    def __init__(self, events):
        self._events = events

    def update_weights(self):
        self._events.append(("update_weights", None))

    def async_train(self, rollout_id, rollout_data_ref, external_data=None):
        del rollout_data_ref, external_data
        self._events.append(("train", rollout_id))

    def clear_memory(self):
        self._events.append(("clear_memory", None))

    def predict_regression(self, rollout_data_ref):
        assert rollout_data_ref == "eval-data"
        self._events.append(("predict_regression", None))
        return [(0, 1.5)]


def _run_train(
    monkeypatch,
    train_module_under_test,
    *,
    start_rollout_id,
    num_rollout,
    skip_eval_before_train=False,
    loss_type="policy_loss",
):
    events = []
    rollout_manager = _RolloutManager(events)
    actor_model = _ActorModel(events)
    args = SimpleNamespace(
        release_train=False,
        offload_rollout=False,
        check_weight_update_equal=False,
        num_rollout=num_rollout,
        eval_interval=20,
        skip_eval_before_train=skip_eval_before_train,
        start_rollout_id=start_rollout_id,
        offload_train=False,
        use_critic=False,
        num_critic_only_steps=0,
        save_interval=None,
        rollout_global_dataset=False,
        colocate=False,
        update_weights_interval=1000,
        loss_type=loss_type,
    )

    monkeypatch.setattr(train_module_under_test, "configure_logger", lambda: None)
    monkeypatch.setattr(train_module_under_test, "init_tracking", lambda args: None)
    monkeypatch.setattr(train_module_under_test, "finish_tracking", lambda args: None)
    monkeypatch.setattr(train_module_under_test, "create_placement_groups", lambda args: {"rollout": object()})
    monkeypatch.setattr(
        train_module_under_test,
        "create_rollout_manager",
        lambda args, placement_group: (rollout_manager, None),
    )
    monkeypatch.setattr(
        train_module_under_test,
        "create_training_models",
        lambda args, placement_groups, manager: (actor_model, None),
    )
    monkeypatch.setattr(train_module_under_test.ray, "get", lambda value: value)

    train_module_under_test.train(args)
    return events


@pytest.mark.unit
@pytest.mark.parametrize("train_module_under_test", [train_module, train_async_module])
@pytest.mark.parametrize(
    ("start_rollout_id", "expected_eval_rollout_id"),
    [(0, 0), (1, 0), (100, 99)],
)
def test_eval_before_train_uses_current_checkpoint_step(
    monkeypatch,
    train_module_under_test,
    start_rollout_id,
    expected_eval_rollout_id,
):
    events = _run_train(
        monkeypatch,
        train_module_under_test,
        start_rollout_id=start_rollout_id,
        num_rollout=start_rollout_id + 1,
    )

    assert ("eval", expected_eval_rollout_id) in events
    assert events.index(("eval", expected_eval_rollout_id)) < events.index(("generate", start_rollout_id))


@pytest.mark.unit
@pytest.mark.parametrize("train_module_under_test", [train_module, train_async_module])
def test_eval_only_uses_loaded_checkpoint_step(monkeypatch, train_module_under_test):
    events = _run_train(monkeypatch, train_module_under_test, start_rollout_id=100, num_rollout=0)

    assert ("eval", 99) in events
    assert not any(event == "generate" for event, _ in events)
    assert not any(event == "train" for event, _ in events)


@pytest.mark.unit
@pytest.mark.parametrize("train_module_under_test", [train_module, train_async_module])
def test_skip_eval_before_train_applies_to_resumed_training(monkeypatch, train_module_under_test):
    events = _run_train(
        monkeypatch,
        train_module_under_test,
        start_rollout_id=100,
        num_rollout=101,
        skip_eval_before_train=True,
    )

    assert not any(event == "eval" for event, _ in events)


@pytest.mark.unit
@pytest.mark.parametrize("train_module_under_test", [train_module, train_async_module])
def test_regression_startup_eval_dispatches_through_megatron(monkeypatch, train_module_under_test):
    events = _run_train(
        monkeypatch,
        train_module_under_test,
        start_rollout_id=0,
        num_rollout=1,
        loss_type="regression_loss",
    )

    assert ("prepare_regression_eval", 0) in events
    assert ("predict_regression", None) in events
    assert ("finish_regression_eval", 0, [(0, 1.5)]) in events
    assert not any(event[0] == "eval" for event in events)
    assert events.index(("finish_regression_eval", 0, [(0, 1.5)])) < events.index(("generate", 0))


@pytest.mark.unit
@pytest.mark.parametrize("train_module_under_test", [train_module, train_async_module])
def test_regression_periodic_eval_dispatches_through_megatron(monkeypatch, train_module_under_test):
    events = _run_train(
        monkeypatch,
        train_module_under_test,
        start_rollout_id=19,
        num_rollout=20,
        skip_eval_before_train=True,
        loss_type="regression_loss",
    )

    assert ("prepare_regression_eval", 19) in events
    assert ("finish_regression_eval", 19, [(0, 1.5)]) in events
    assert events.index(("train", 19)) < events.index(("prepare_regression_eval", 19))
