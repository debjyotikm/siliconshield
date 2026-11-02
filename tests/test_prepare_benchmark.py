"""Reviewed benchmark publication checks using synthetic rows only."""

import hashlib
import json

import pytest

from tools import prepare_benchmark as preparation


def encode(rows):
    return (
        "\n".join(json.dumps(row, sort_keys=True) for row in rows) + "\n"
    ).encode()


@pytest.fixture
def population(tmp_path, monkeypatch):
    rows = [
        {
            "id": f"{pair}-{label}",
            "state": "Synthetic RTL assessment input",
            "questions": {},
            "expected": {"security": label},
            "benchmark_area": "fifo",
            "benchmark_source": "synthetic",
            "pair_id": pair,
        }
        for pair in ("remove", "keep")
        for label in preparation.LABELS
    ]
    source = tmp_path / "input.jsonl"
    original = encode(rows)
    source.write_bytes(original)
    monkeypatch.setattr(
        preparation, "SOURCE_SHA256", hashlib.sha256(original).hexdigest()
    )
    monkeypatch.setattr(
        preparation,
        "REVIEWED_SHA256",
        hashlib.sha256(encode(rows[3:])).hexdigest(),
    )
    monkeypatch.setattr(preparation, "SOURCE_ROWS", 6)
    monkeypatch.setattr(preparation, "REVIEWED_ROWS", 3)
    monkeypatch.setattr(preparation, "EXCLUDED_PAIRS", ("remove",))
    return source, rows, original


def test_exact_filter_and_receipt_preserve_source(population, tmp_path):
    source, rows, original = population
    output = tmp_path / "reviewed"
    receipt = preparation.prepare(str(source), output)
    assert (output / "benchmark-reviewed.jsonl").read_bytes() == encode(
        rows[3:]
    )
    assert json.loads((output / "provenance.json").read_text()) == receipt
    assert receipt["removed_rows"] == 3
    assert receipt["excluded_pairs"] == ["remove"]
    assert source.read_bytes() == original
    assert not list(tmp_path.glob(".reviewed-benchmark-*"))


@pytest.mark.parametrize("failure", ["source", "triplets", "result"])
def test_refusal_does_not_publish(population, tmp_path, monkeypatch, failure):
    source, _, original = population
    if failure == "source":
        source.write_bytes(b" " + original)
    elif failure == "triplets":
        monkeypatch.setattr(preparation, "EXCLUDED_PAIRS", ("absent",))
    else:
        monkeypatch.setattr(preparation, "REVIEWED_SHA256", "incorrect")
    output = tmp_path / "reviewed"
    with pytest.raises(ValueError):
        preparation.prepare(str(source), output)
    assert not output.exists()


def test_existing_output_is_preserved(population, tmp_path):
    source, _, _ = population
    output = tmp_path / "reviewed"
    preparation.prepare(str(source), output)
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        preparation.prepare(str(source), output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
