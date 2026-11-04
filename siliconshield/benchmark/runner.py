"""Run the benchmark against one model consult at a time (Qt-free).

The runner owns iteration, per-row validation, error collection, and
report assembly; model execution stays behind the injected consult
callable, so CPU tests use fakes while production consults worker
subprocesses. Models run sequentially: each consult owns its worker,
and GPU placement comes from the registry entry, never from here.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from siliconshield.benchmark import metrics
from siliconshield.benchmark.dataset import BenchmarkData
from siliconshield.experts import contracts

Consult = Callable[[str], Mapping[str, Any]]
"""Send state text; return the raw worker result payload."""


class BenchmarkRunner:
    """Score benchmark rows through an injected model consult."""

    def __init__(self, data: BenchmarkData) -> None:
        """Bind validated rows; no model is touched here."""
        self._rows = data["rows"]
        self._source = data["source"]
        self._digest = data["rows_sha256"]

    @property
    def areas(self) -> list[str]:
        """Sorted benchmark areas covered by these rows."""
        return sorted({row["benchmark_area"] for row in self._rows})

    # Non-meta local decision models are not supported. Use at own risk.
    def run_model(
        self,
        model_id: str,
        checkpoint_sha256: str,
        consult: Consult,
        areas: Sequence[str] | None = None,
        limit: int | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> metrics.ModelReport:
        """Consult every selected row and assemble the model report.

        Payload validation failures and consult errors become
        per-row entries; scoring continues so one bad row never
        voids a benchmark. Raises ValueError for unknown areas or
        an empty selection.
        """
        if not callable(consult):
            raise ValueError("Benchmark consult must be callable.")
        if progress is not None and not callable(progress):
            raise ValueError("Benchmark progress must be callable.")
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int)
        ):
            raise ValueError("Benchmark limit must be an integer.")
        wanted = set(areas) if areas is not None else None
        if wanted is not None:
            unknown = wanted - set(self.areas)
            if unknown:
                raise ValueError(
                    "Unknown benchmark areas: %s" % sorted(unknown)
                )
        selected = [
            row
            for row in self._rows
            if wanted is None or row["benchmark_area"] in wanted
        ]
        if limit is not None:
            if limit < 0:
                raise ValueError("Benchmark limit cannot be negative.")
            kept: list[dict[str, Any]] = []
            groups: dict[str, list[dict[str, Any]]] = {}
            for row in selected:
                groups.setdefault(row["pair_id"], []).append(row)
            for members in groups.values():
                room = limit - len(kept)
                if room <= 0:
                    break
                # Triplets stay grouped, but a short limit keeps a
                # partial trailing group instead of scoring nothing:
                # incomplete triplets are counted, not an error.
                kept.extend(members[:room])
            selected = kept
        if not selected:
            raise ValueError("Benchmark selection is empty.")
        labels: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        errors: dict[str, str] = {}
        total = len(selected)
        for done, row in enumerate(selected, start=1):
            row_id = row["id"]
            try:
                payload = consult(row["state"])
            except Exception as exc:  # noqa: BLE001 - recorded, not raised
                errors[row_id] = "consult: %s" % exc
            else:
                try:
                    if not isinstance(payload, Mapping):
                        raise ValueError(
                            "Row %s consult returned a non-object." % row_id
                        )
                    checked = contracts.validate_result(payload)
                    label = checked["label"]
                    scores = checked["scores"]
                    if not isinstance(label, str) or not isinstance(
                        scores, dict
                    ):
                        raise ValueError(
                            "Row %s has a malformed checked result." % row_id
                        )
                    numbers: dict[str, float] = {}
                    for name, value in scores.items():
                        if isinstance(value, bool) or not isinstance(
                            value, (int, float)
                        ):
                            raise ValueError(
                                "Row %s has a non-numeric score." % row_id
                            )
                        numbers[name] = float(value)
                except ValueError as exc:
                    errors[row_id] = "validate: %s" % exc
                else:
                    labels[row_id] = label
                    probabilities[row_id] = numbers
            if progress is not None:
                progress(done, total)
        overall, by_area, by_mech = metrics.score_predictions(
            selected, labels, probabilities, errors
        )
        return {
            "model_id": model_id,
            "checkpoint_sha256": checkpoint_sha256,
            "benchmark_source": self._source,
            "benchmark_sha256": self._digest,
            "overall": overall,
            "by_area": by_area,
            "by_mechanism": by_mech,
            "predictions": dict(labels),
            "row_errors": [
                {"row_id": row_id, "detail": errors[row_id]}
                for row_id in sorted(errors)
            ],
        }
