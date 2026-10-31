"""Tests for versioned question contracts."""

import json
import unittest

from siliconshield.app.laya_worker import QUESTION as WORKER_QUESTION
from siliconshield.training.laya import question_contracts as contracts


class QuestionContractsTest(unittest.TestCase):
    def test_legacy_matches_worker_bytes(self):
        worker = WORKER_QUESTION["security"]
        built = contracts.question_for(contracts.LEGACY_CONTRACT)
        self.assertEqual(
            built["security"]["instructions"], worker["instructions"]
        )
        self.assertEqual(built["security"]["criteria"], worker["criteria"])

    def test_registry_round_trip(self):
        for version in contracts.known_contracts():
            questions = contracts.question_for(version)
            self.assertEqual(contracts.match_version(questions), version)
            self.assertEqual(len(contracts.digest_for(version)), 64)

    def test_unknown_version_refused(self):
        with self.assertRaises(ValueError):
            contracts.question_for("no-such-contract")
        with self.assertRaises(ValueError):
            contracts.digest_for("no-such-contract")

    def test_foreign_questions_refused(self):
        with self.assertRaises(ValueError):
            contracts.match_version(
                {
                    "security": {
                        "instructions": "something else",
                        "criteria": list(contracts.LABELS),
                    }
                }
            )
        with self.assertRaises(ValueError):
            contracts.match_version({})

    def test_validate_rows(self):
        rows = [
            {
                "id": "r%d" % i,
                "questions": contracts.question_for(contracts.NEW_CONTRACT),
            }
            for i in range(3)
        ]
        self.assertEqual(
            contracts.validate_rows(rows, contracts.NEW_CONTRACT),
            contracts.NEW_CONTRACT,
        )
        mixed = rows + [
            {
                "id": "old",
                "questions": contracts.question_for(contracts.LEGACY_CONTRACT),
            }
        ]
        with self.assertRaises(ValueError):
            contracts.validate_rows(mixed, contracts.NEW_CONTRACT)

    def test_export_record_round_trip(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            record = contracts.write_export_record(tmp, contracts.NEW_CONTRACT)
            self.assertEqual(record["version"], contracts.NEW_CONTRACT)
            read = contracts.read_export_record(tmp)
            self.assertEqual(read["version"], contracts.NEW_CONTRACT)
            self.assertFalse(read["legacy_default"])

    def test_legacy_default_for_unrecorded_export(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            read = contracts.read_export_record(tmp)
            self.assertEqual(read["version"], contracts.LEGACY_CONTRACT)
            self.assertTrue(read["legacy_default"])

    def test_tampered_record_refused(self):
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            contracts.write_export_record(tmp, contracts.NEW_CONTRACT)
            path = Path(tmp) / "question_contract.json"
            record = json.loads(path.read_text())
            record["instructions"] = "tampered"
            path.write_text(json.dumps(record))
            with self.assertRaises(ValueError):
                contracts.read_export_record(tmp)


if __name__ == "__main__":
    unittest.main()
