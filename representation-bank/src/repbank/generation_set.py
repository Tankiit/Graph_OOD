from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

SCHEMA_VERSION = 1


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()


def sha256_value(value: Any) -> str:
    return hashlib.sha256(canonical_bytes(value)).hexdigest()


class FrozenRecord(BaseModel):
    question_id: str
    pair_id: str
    role: Literal[0, 1]  # 0 correct, 1 wrong
    prompt_text: str
    answer_text: str
    token_ids: list[int]
    answer_start: int = Field(ge=1)
    label: int
    label_protocol: str
    source_model: str
    decode_config: dict[str, Any]
    seed: int

    @model_validator(mode="after")
    def validate_span(self) -> FrozenRecord:
        if self.answer_start >= len(self.token_ids):
            raise ValueError("answer_start must point inside token_ids")
        if self.label != 1 - self.role:
            raise ValueError("label must be 1 for correct role=0 and 0 for wrong role=1")
        return self


class FrozenGenerationSet(BaseModel):
    schema_version: int = SCHEMA_VERSION
    tokenizer_id: str
    tokenizer_sha256: str
    records: list[FrozenRecord]
    checksum_sha256: str = ""

    def payload(self) -> dict[str, Any]:
        return self.model_dump(exclude={"checksum_sha256"})

    def computed_checksum(self) -> str:
        return sha256_value(self.payload())

    def seal(self) -> FrozenGenerationSet:
        return self.model_copy(update={"checksum_sha256": self.computed_checksum()})

    def verify(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported generation-set schema {self.schema_version}")
        actual = self.computed_checksum()
        if self.checksum_sha256 != actual:
            raise ValueError(f"generation-set checksum mismatch: expected {self.checksum_sha256}, got {actual}")

    def write(self, path: str | Path, *, force: bool = False) -> Path:
        target = Path(path)
        if target.exists() and not force:
            raise FileExistsError(f"refusing to overwrite frozen generation set: {target}")
        sealed = self.seal()
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".partial")
        temporary.write_bytes(canonical_bytes(sealed.model_dump()) + b"\n")
        temporary.replace(target)
        return target

    @classmethod
    def read(cls, path: str | Path) -> FrozenGenerationSet:
        result = cls.model_validate_json(Path(path).read_text())
        result.verify()
        return result


def tokenizer_digest(tokenizer: Any) -> str:
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is not None:
        return hashlib.sha256(backend.to_str().encode()).hexdigest()
    return sha256_value(tokenizer.get_vocab())


def find_answer_start(prompt_ids: list[int], full_ids: list[int]) -> int:
    common = 0
    for prompt_token, full_token in zip(prompt_ids, full_ids, strict=False):
        if prompt_token != full_token:
            break
        common += 1
    if common == len(full_ids):
        raise ValueError("answer produced no tokens")
    # A boundary merge makes the final prompt token answer-dependent. Treat it
    # as answer span so no answer likelihood term is silently dropped.
    return common


def freeze_tinker_generations(
    rows: list[dict[str, Any]], tokenizer: Any, *, tokenizer_id: str,
    source_model: str, decode_config: dict[str, Any], base_seed: int,
) -> FrozenGenerationSet:
    records: list[FrozenRecord] = []
    question_order: dict[str, int] = {}
    for index, row in enumerate(rows):
        if row.get("role") not in {"true", "hal"}:
            continue
        prompt = row.get("prompt_text", row.get("prompt"))
        answer = row.get("answer_text", row.get("generation"))
        if not prompt or not answer:
            raise ValueError(f"row {index} lacks prompt/answer text")
        full_ids = tokenizer.encode(prompt + answer)
        prompt_ids = tokenizer.encode(prompt)
        answer_start = find_answer_start(prompt_ids, full_ids)
        role = 0 if row["role"] == "true" else 1
        question_id = str(row.get("question_id", row.get("pair_id", index)))
        question_order.setdefault(question_id, len(question_order))
        sample_id = int(row.get("sample_id", 0))
        records.append(FrozenRecord(
            question_id=question_id,
            pair_id=str(row.get("pair_id", question_id)),
            role=role,
            prompt_text=prompt,
            answer_text=answer,
            token_ids=full_ids,
            answer_start=answer_start,
            label=1 - role,
            label_protocol=str(row["label_protocol"]),
            source_model=source_model,
            decode_config=decode_config,
            seed=base_seed + question_order[question_id] * int(decode_config.get("samples", 1))
            + sample_id,
        ))
    if not records:
        raise ValueError("no labeled records to freeze")
    return FrozenGenerationSet(
        tokenizer_id=tokenizer_id,
        tokenizer_sha256=tokenizer_digest(tokenizer),
        records=records,
    ).seal()


def g1_report(generation_set: FrozenGenerationSet) -> dict[str, Any]:
    roles: dict[str, set[int]] = defaultdict(set)
    for record in generation_set.records:
        roles[record.pair_id].add(record.role)
    paired = sum(values == {0, 1} for values in roles.values())
    counts = Counter(record.role for record in generation_set.records)
    return {
        "records": len(generation_set.records),
        "questions": len(roles),
        "paired_questions": paired,
        "pair_yield": paired / len(roles) if roles else 0.0,
        "correct_records": counts[0],
        "wrong_records": counts[1],
        "checksum_sha256": generation_set.checksum_sha256,
    }


def write_adapter_view(generation_set: FrozenGenerationSet, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".partial")
    with temporary.open("w") as handle:
        for record in generation_set.records:
            row = {
                "pair_id": record.pair_id,
                "role": "true" if record.role == 0 else "hal",
                "messages": [
                    {"role": "user", "content": record.prompt_text},
                    {"role": "assistant", "content": record.answer_text},
                ],
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)
    return target


def merge_frozen_sets(parts: list[FrozenGenerationSet]) -> FrozenGenerationSet:
    """Create a new sealed version without modifying any constituent set."""
    if not parts:
        raise ValueError("at least one frozen generation set is required")
    family = {(part.tokenizer_id, part.tokenizer_sha256) for part in parts}
    if len(family) != 1:
        raise ValueError("cannot merge generation sets from different tokenizer families")
    records = [record for part in parts for record in part.records]
    identities = [(record.pair_id, record.seed) for record in records]
    if len(identities) != len(set(identities)):
        raise ValueError("merged generation sets contain duplicate pair_id/seed records")
    first = parts[0]
    return FrozenGenerationSet(
        tokenizer_id=first.tokenizer_id,
        tokenizer_sha256=first.tokenizer_sha256,
        records=records,
    ).seal()
