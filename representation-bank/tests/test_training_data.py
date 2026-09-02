import json

import pytest

from repbank.tinker_ops import load_training_records, scheduled_learning_rate


def test_cosine_schedule_warms_up_and_decays() -> None:
    rates = [scheduled_learning_rate(i, 100, 1e-4, "cosine", 0.1, 0.05)
             for i in range(100)]
    assert rates[0] == 1e-5
    assert rates[9] == 1e-4
    assert rates[-1] < rates[10]
    assert rates[-1] > 5e-6


def test_load_local_jsonl(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = [
        {"role": "true", "messages": [{"role": "user", "content": "Q"},
                                         {"role": "assistant", "content": "A"}]},
        {"role": "hal", "messages": [{"role": "user", "content": "Q"},
                                        {"role": "assistant", "content": "B"}]},
    ]
    path.write_text("\n".join(json.dumps(row) for row in rows))
    assert load_training_records(str(path)) == rows


def test_load_hf_dataset(monkeypatch):
    rows = [{"conversation": [{"role": "user", "content": "Q"},
                               {"role": "assistant", "content": "A"}]}]
    calls = {}

    def fake_load_dataset(name, config, **kwargs):
        calls.update(name=name, config=config, **kwargs)
        return rows

    import datasets
    monkeypatch.setattr(datasets, "load_dataset", fake_load_dataset)
    loaded = load_training_records(
        "org/data", split="validation", dataset_config="clean",
        revision="abc123", messages_column="conversation",
    )
    assert loaded == [{"messages": rows[0]["conversation"]}]
    assert calls == {"name": "org/data", "config": "clean", "split": "validation",
                     "revision": "abc123"}


def test_rejects_non_chat_rows(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text('{"text": "not chat"}\n')
    with pytest.raises(ValueError, match="messages"):
        load_training_records(str(path))
