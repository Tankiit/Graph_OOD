import json

import pytest

from repbank.generation_set import (
    FrozenGenerationSet,
    find_answer_start,
    freeze_tinker_generations,
    g1_report,
)


class TinyTokenizer:
    backend_tokenizer = None

    def encode(self, text):
        return [ord(char) for char in text]

    def get_vocab(self):
        return {"a": 0}


def test_freeze_checksum_and_g1(tmp_path):
    rows = [
        {"pair_id": "q1", "sample_id": 0, "role": "true", "prompt": "Q:",
         "generation": " A", "label_protocol": "exact"},
        {"pair_id": "q1", "sample_id": 1, "role": "hal", "prompt": "Q:",
         "generation": " B", "label_protocol": "exact"},
    ]
    frozen = freeze_tinker_generations(
        rows, TinyTokenizer(), tokenizer_id="tiny", source_model="tiny",
        decode_config={"samples": 2}, base_seed=7,
    )
    path = tmp_path / "generation_set.json"
    frozen.write(path)
    loaded = FrozenGenerationSet.read(path)
    assert loaded.records[0].answer_start == 2
    assert loaded.records[0].seed == 7
    assert loaded.records[1].seed == 8
    assert g1_report(loaded)["pair_yield"] == 1.0


def test_checksum_detects_mutation(tmp_path):
    frozen = freeze_tinker_generations(
        [{"pair_id": "q", "role": "true", "prompt": "Q", "generation": "A",
          "label_protocol": "exact"}],
        TinyTokenizer(), tokenizer_id="tiny", source_model="tiny",
        decode_config={"samples": 1}, base_seed=0,
    )
    path = tmp_path / "generation_set.json"
    frozen.write(path)
    payload = json.loads(path.read_text())
    payload["records"][0]["answer_text"] = "changed"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="checksum mismatch"):
        FrozenGenerationSet.read(path)


def test_boundary_merge_moves_answer_start_left():
    assert find_answer_start([1, 2, 3], [1, 2, 9, 4]) == 2
