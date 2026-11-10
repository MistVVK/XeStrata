"""`strata update --yes` while a model serves: the refusal says "the update", not "an engine update" (only Control may be pending; ARCFIX E5, owner-approved 11-09)."""
import pytest

from strata_control import engine_update, updater


def test_the_refusal_names_the_update_not_an_engine_update(monkeypatch):
    model = engine_update.RunningModel("launch-qwen", "qwen3.8-flash-next-iq3_s", "ready", 18434, "launch")
    monkeypatch.setattr(engine_update, "running_models", lambda serving=None: [model])
    gate = engine_update.ModelGate(ask=None)
    with pytest.raises(updater.UpdateError) as caught:
        gate.consent([{"profile": "launch-qwen", "state": "ready"}], "It would update the Control 0.1.59 -> 0.1.59.1.")
    text = str(caught.value)
    assert "is running, and the update needs it stopped, so nothing was changed." in text
    assert "engine update" not in text
    assert caught.value.exit_code == 7
