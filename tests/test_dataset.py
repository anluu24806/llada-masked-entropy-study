"""CPU-only parser and immutable custom-dataset tests; no network fixtures."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from llada_study import dataset as dataset_module

from llada_study.dataset import (
    load_gold,
    load_public,
    normalize_numeric,
    parse_answer,
    prepare_dataset,
)


class AnswerParserTests(unittest.TestCase):
    def assertParsed(self, text, family, identity):
        parsed = parse_answer(text, family)
        self.assertEqual(parsed["status"], "PARSED")
        self.assertEqual(parsed["identity"], identity)

    def assertUnparsed(self, text, family):
        parsed = parse_answer(text, family)
        self.assertEqual(parsed["status"], "U")
        self.assertIsNone(parsed["identity"])

    def test_numeric_canonicalization_is_exact(self):
        examples = {
            "00042.000": "42",
            "+12.3400": "12.34",
            "-00012.3400": "-12.34",
            "-0.000": "0",
            "1,234,567.8900": "1234567.89",
            "-12,345.600": "-12345.6",
            "123456789012345678901234567890.1234500":
                "123456789012345678901234567890.12345",
        }
        for value, expected in examples.items():
            with self.subTest(value=value):
                self.assertEqual(normalize_numeric(value), expected)

    def test_numeric_rejects_floats_booleans_and_unsupported_notation(self):
        for value in (1.5, True, False, None, "1e3", "NaN", "Infinity", "1,23"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                normalize_numeric(value)
        self.assertEqual(normalize_numeric(1234), "1234")

    def test_gsm_numeric_answer(self):
        for value, identity in (("+1,234.500", "1234.5"), ("-0002.500", "-2.5"),
                                ("-0", "0"), ("42", "42")):
            with self.subTest(value=value):
                self.assertParsed("Reasoning may precede the result.\nAnswer: " + value,
                                  "GSM8K", identity)

    def test_gsm_rejects_non_decimal_and_bad_comma_grouping(self):
        for value in ("1e3", "1E+3", "NaN", "Infinity", "1/2", "$42", "42%",
                      "1,23", "12,34,567", "1234,567", "1,,234", "1,234,",
                      "1 234", "--2", "forty-two", "42 apples"):
            with self.subTest(value=value):
                self.assertUnparsed("Answer: " + value, "GSM8K")

    def test_answer_must_be_one_complete_final_nonempty_line(self):
        for text in ("", "42", "answer: 42", "The Answer: 42", "Answer:",
                     "Answer: 42\nThis is the answer.", "Answer: 42\nAnswer: 42",
                     "Answer: 41\nReasoning\nAnswer: 42", "Answer: 42. Done.",
                     "Answer: 42\n```", "```\nAnswer: 42\n```"):
            with self.subTest(text=text):
                self.assertUnparsed(text, "GSM8K")
        self.assertParsed("Reasoning\nAnswer: 42\n\n", "GSM8K", "42")

    def test_arithmetic_requires_integer(self):
        for value, identity in (("+00042", "42"), ("-003", "-3"), ("-0", "0")):
            with self.subTest(value=value):
                self.assertParsed("Answer: " + value, "ARITHMETIC", identity)
        for value in ("42.0", "1,000", "1e2", "1/2", "2 3", "--2"):
            with self.subTest(value=value):
                self.assertUnparsed("Answer: " + value, "ARITHMETIC")

    def test_evidence_tracking_requires_one_alphabetic_word(self):
        self.assertParsed("Answer: ALIce", "EVIDENCE_TRACKING", "alice")
        for value in ("Alice Bob", "Alice1", "Alice.", "Alice-Bob", "123", "_alice"):
            with self.subTest(value=value):
                self.assertUnparsed("Answer: " + value, "EVIDENCE_TRACKING")

    def test_symbolic_transform_canonicalizes_signed_integer_sequence(self):
        self.assertParsed("Answer: +01, -002, 0", "SYMBOLIC_TRANSFORM", "1,-2,0")
        self.assertParsed("Answer: -0", "SYMBOLIC_TRANSFORM", "0")
        for value in ("1,,2", "1,2,", ",1,2", "1.0,2", "1 2", "[1,2]", "1e2,3"):
            with self.subTest(value=value):
                self.assertUnparsed("Answer: " + value, "SYMBOLIC_TRANSFORM")


class CustomDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.public_path = self.root / "questions.jsonl"
        self.gold_path = self.root / "answers.jsonl"
        self.run = self.root / "run"
        self.rows = [
            {"uid": "1" * 64, "family": "GSM8K", "prompt": "What is the total?",
             "metadata": {"source": "test", "index": 0, "tags": ["decimal"]}},
            {"uid": "2" * 64, "family": "ARITHMETIC", "prompt": "Subtract the values."},
            {"uid": "3" * 64, "family": "EVIDENCE_TRACKING", "prompt": "Who has the object?"},
            {"uid": "4" * 64, "family": "SYMBOLIC_TRANSFORM", "prompt": "Transform the list."},
        ]
        self.gold_rows = [
            {"uid": "1" * 64, "gold": "+1,234.500"},
            {"uid": "2" * 64, "gold": "-003"},
            {"uid": "3" * 64, "gold": "ALIce"},
            {"uid": "4" * 64, "gold": "+01, -002, 0"},
        ]
        self.expected_gold = {"1" * 64: "1234.5", "2" * 64: "-3",
                              "3" * 64: "alice", "4" * 64: "1,-2,0"}
        self.write_inputs()

    def write_jsonl(self, path, rows):
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
                        encoding="utf8")

    def write_inputs(self, rows=None, gold_rows=None):
        self.write_jsonl(self.public_path, self.rows if rows is None else rows)
        self.write_jsonl(self.gold_path, self.gold_rows if gold_rows is None else gold_rows)

    def prepare(self):
        return prepare_dataset(str(self.public_path), self.run, {}, gold_path=self.gold_path)

    def snapshot_run(self):
        return {path.relative_to(self.run).as_posix(): path.read_bytes()
                for path in self.run.rglob("*") if path.is_file()}

    def assertRejected(self):
        with self.assertRaises((ValueError, RuntimeError)):
            self.prepare()
        self.assertFalse((self.run / "data" / "DATASET_SEAL.json").exists())

    def test_round_trip_preserves_public_metadata_and_canonicalizes_gold(self):
        seal = self.prepare()
        self.assertIsInstance(seal, dict)
        self.assertEqual(load_public(self.run), self.rows)
        self.assertEqual(load_gold(self.run), self.expected_gold)
        self.assertEqual(json.loads((self.run / "data" / "DATASET_SEAL.json").read_text("utf8")), seal)
        public_parts = sorted((self.run / "data").glob("public_*.jsonl"))
        gold_parts = sorted((self.run / "data").glob("protected_gold_*.jsonl"))
        self.assertTrue(public_parts)
        self.assertTrue(gold_parts)
        self.assertEqual(seal["uid_count"], len(self.rows))
        public_names = [path.relative_to(self.run).as_posix() for path in public_parts]
        gold_names = [path.relative_to(self.run).as_posix() for path in gold_parts]
        self.assertEqual(seal["public_files"], public_names)
        self.assertEqual(seal["gold_files"], gold_names)
        self.assertEqual(set(seal["files"]), set(public_names + gold_names))
        for path in public_parts + gold_parts:
            with self.subTest(path=path.name):
                info = seal["files"][path.relative_to(self.run).as_posix()]
                self.assertEqual(info["sha256"], hashlib.sha256(path.read_bytes()).hexdigest())
                self.assertEqual(info["bytes"], path.stat().st_size)
        for path in public_parts:
            for line in path.read_text("utf8").splitlines():
                row = json.loads(line)
                self.assertNotIn("gold", row)
                self.assertNotIn("answer", row)

    def test_identical_resume_returns_original_seal_without_rewriting(self):
        first = self.prepare()
        before = self.snapshot_run()
        mtimes = {path: path.stat().st_mtime_ns for path in self.run.rglob("*") if path.is_file()}
        self.assertEqual(self.prepare(), first)
        self.assertEqual(self.snapshot_run(), before)
        self.assertEqual({path: path.stat().st_mtime_ns for path in mtimes}, mtimes)

    def test_changed_public_input_fails_without_overwriting_sealed_run(self):
        self.prepare()
        before = self.snapshot_run()
        changed = copy.deepcopy(self.rows)
        changed[0]["prompt"] = "A changed question"
        self.write_inputs(rows=changed)
        with self.assertRaises((ValueError, RuntimeError)):
            self.prepare()
        self.assertEqual(self.snapshot_run(), before)

    def test_changed_gold_input_fails_without_overwriting_sealed_run(self):
        self.prepare()
        before = self.snapshot_run()
        changed = copy.deepcopy(self.gold_rows)
        changed[0]["gold"] = "99"
        self.write_inputs(gold_rows=changed)
        with self.assertRaises((ValueError, RuntimeError)):
            self.prepare()
        self.assertEqual(self.snapshot_run(), before)

    def test_source_byte_changes_fail_even_if_json_rows_are_identical(self):
        self.prepare()
        before = self.snapshot_run()
        self.public_path.write_bytes(self.public_path.read_bytes() + b"\n")
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertEqual(self.snapshot_run(), before)

    def test_corrupt_public_output_fails_without_repairing_or_overwriting(self):
        self.prepare()
        output = next((self.run / "data").glob("public_*.jsonl"))
        output.write_bytes(output.read_bytes() + b"\n")
        corrupted = self.snapshot_run()
        with self.assertRaises((ValueError, RuntimeError)):
            self.prepare()
        self.assertEqual(self.snapshot_run(), corrupted)

    def test_corrupt_gold_output_fails_without_repairing_or_overwriting(self):
        self.prepare()
        output = next((self.run / "data").glob("protected_gold_*.jsonl"))
        output.write_bytes(output.read_bytes() + b"\n")
        corrupted = self.snapshot_run()
        with self.assertRaises((ValueError, RuntimeError)):
            self.prepare()
        self.assertEqual(self.snapshot_run(), corrupted)

    def test_loaders_reject_corrupt_sealed_files(self):
        self.prepare()
        public_part = next((self.run / "data").glob("public_*.jsonl"))
        public_part.write_bytes(public_part.read_bytes() + b"\n")
        with self.assertRaises(RuntimeError):
            load_public(self.run)
        with self.assertRaises(RuntimeError):
            load_gold(self.run)

    def test_missing_sealed_output_is_not_recreated_on_resume(self):
        self.prepare()
        gold_part = next((self.run / "data").glob("protected_gold_*.jsonl"))
        gold_part.unlink()
        before = self.snapshot_run()
        with self.assertRaises(RuntimeError):
            self.prepare()
        self.assertEqual(self.snapshot_run(), before)
        self.assertFalse(gold_part.exists())

    def test_duplicate_public_uid_is_rejected(self):
        self.write_inputs(rows=self.rows + [copy.deepcopy(self.rows[0])])
        self.assertRejected()

    def test_duplicate_custom_question_with_distinct_uids_is_rejected(self):
        duplicate = copy.deepcopy(self.rows[0])
        duplicate["uid"] = "5" * 64
        duplicate["metadata"] = {"source": "different source"}
        self.write_inputs(rows=self.rows + [duplicate],
                          gold_rows=self.gold_rows + [{"uid": "5" * 64, "gold": "1234.5"}])
        self.assertRejected()

    def test_same_prompt_in_different_custom_families_is_allowed(self):
        changed = copy.deepcopy(self.rows)
        changed[1]["prompt"] = changed[0]["prompt"]
        self.write_inputs(rows=changed)
        self.prepare()
        self.assertEqual(load_public(self.run), changed)

    def test_duplicate_gold_uid_is_rejected(self):
        self.write_inputs(gold_rows=self.gold_rows + [copy.deepcopy(self.gold_rows[0])])
        self.assertRejected()

    def test_missing_gold_uid_is_rejected(self):
        self.write_inputs(gold_rows=self.gold_rows[:-1])
        self.assertRejected()

    def test_extra_gold_uid_is_rejected(self):
        self.write_inputs(gold_rows=self.gold_rows + [{"uid": "5" * 64, "gold": "1"}])
        self.assertRejected()

    def test_unknown_family_is_rejected(self):
        changed = copy.deepcopy(self.rows)
        changed[0]["family"] = "UNKNOWN_FAMILY"
        self.write_inputs(rows=changed)
        self.assertRejected()

    def test_uid_must_be_lowercase_64_character_hex(self):
        for uid in ("a" * 63, "a" * 65, "A" * 64, "g" * 64, "", 123):
            with self.subTest(uid=uid):
                changed = copy.deepcopy(self.rows)
                changed[0]["uid"] = uid
                self.write_inputs(rows=changed)
                self.assertRejected()

    def test_public_answer_and_gold_fields_are_forbidden(self):
        for field in ("answer", "gold"):
            with self.subTest(field=field):
                changed = copy.deepcopy(self.rows)
                changed[0][field] = "1234.5"
                self.write_inputs(rows=changed)
                self.assertRejected()

    def test_invalid_gold_is_rejected_by_family(self):
        for index, invalid in ((0, "1e3"), (1, "3.5"), (2, "Alice Bob"), (3, "1,,2")):
            with self.subTest(family=self.rows[index]["family"]):
                changed = copy.deepcopy(self.gold_rows)
                changed[index]["gold"] = invalid
                self.write_inputs(gold_rows=changed)
                self.assertRejected()


class ExistingDiscoveryDeduplicationTests(unittest.TestCase):
    def setUp(self):
        self.rows = [
            {"uid": "9" * 64, "family": "ARITHMETIC", "prompt": "Compute 2 + 2.",
             "metadata": {"source": "later alias"}},
            {"uid": "8" * 64, "family": "EVIDENCE_TRACKING", "prompt": "Track parcel.",
             "metadata": {"source": "another alias"}},
            {"uid": "5" * 64, "family": "ARITHMETIC", "prompt": "Compute 2 + 3."},
            {"uid": "1" * 64, "family": "ARITHMETIC", "prompt": "Compute 2 + 2.",
             "metadata": {"source": "representative"}},
            {"uid": "2" * 64, "family": "EVIDENCE_TRACKING", "prompt": "Track parcel.",
             "metadata": {"source": "earlier representative"}},
            {"uid": "7" * 64, "family": "ARITHMETIC", "prompt": "Compute 2 + 2.",
             "metadata": {"source": "third alias"}},
        ]
        self.gold = [{"uid": row["uid"], "gold": answer}
                     for row, answer in zip(self.rows, ("4", "alice", "5", "4", "alice", "4"))]
        self.expected_public = [self.rows[3], self.rows[4], self.rows[2]]
        self.expected_gold = [{"uid": "1" * 64, "gold": "4"},
                              {"uid": "2" * 64, "gold": "alice"},
                              {"uid": "5" * 64, "gold": "5"}]

    def test_aliases_counts_and_representatives_are_exact(self):
        original_public, original_gold = copy.deepcopy(self.rows), copy.deepcopy(self.gold)
        public, gold, metadata = dataset_module._deduplicate_existing(self.rows, self.gold)
        self.assertEqual(public, self.expected_public)
        self.assertEqual(gold, self.expected_gold)
        self.assertEqual(metadata, {
            "rule": "exact_family_prompt_lexicographic_uid_v1",
            "source_row_count": 6,
            "distinct_question_count": 3,
            "excess_source_rows": 3,
            "duplicate_group_count": 2,
            "alias_uids": {"1" * 64: ["1" * 64, "7" * 64, "9" * 64],
                           "2" * 64: ["2" * 64, "8" * 64],
                           "5" * 64: ["5" * 64]},
        })
        self.assertEqual(list(metadata["alias_uids"]), ["1" * 64, "2" * 64, "5" * 64])
        self.assertEqual(self.rows, original_public)
        self.assertEqual(self.gold, original_gold)

    def test_public_and_gold_input_order_cannot_change_result(self):
        expected = dataset_module._deduplicate_existing(self.rows, self.gold)
        orders = (list(reversed(self.rows)), self.rows[2:] + self.rows[:2],
                  sorted(self.rows, key=lambda row: row["uid"]))
        for index, rows in enumerate(orders):
            with self.subTest(order=index):
                self.assertEqual(dataset_module._deduplicate_existing(rows, list(reversed(self.gold))),
                                 expected)

    def test_conflicting_canonical_gold_in_duplicate_group_is_rejected(self):
        changed = copy.deepcopy(self.gold)
        changed[0]["gold"] = "999"
        with self.assertRaises(ValueError):
            dataset_module._deduplicate_existing(self.rows, changed)

    def test_family_is_part_of_question_identity(self):
        rows = [{"uid": "2" * 64, "family": "ARITHMETIC", "prompt": "The same prompt"},
                {"uid": "1" * 64, "family": "GSM8K", "prompt": "The same prompt"}]
        gold = [{"uid": "2" * 64, "gold": "4"}, {"uid": "1" * 64, "gold": "4"}]
        public, unique_gold, metadata = dataset_module._deduplicate_existing(rows, gold)
        self.assertEqual(public, list(reversed(rows)))
        self.assertEqual(unique_gold, list(reversed(gold)))
        self.assertEqual(metadata["distinct_question_count"], 2)
        self.assertEqual(metadata["duplicate_group_count"], 0)
        self.assertEqual(metadata["excess_source_rows"], 0)

    def test_prompt_comparison_is_exact(self):
        rows = [{"uid": str(index) * 64, "family": "ARITHMETIC", "prompt": prompt}
                for index, prompt in enumerate(("Compute 2 + 2.", "Compute 2 + 2. ",
                                                "compute 2 + 2."), start=1)]
        gold = [{"uid": row["uid"], "gold": "4"} for row in rows]
        public, unique_gold, metadata = dataset_module._deduplicate_existing(rows, gold)
        self.assertEqual(public, rows)
        self.assertEqual(unique_gold, gold)
        self.assertEqual(metadata["distinct_question_count"], 3)
        self.assertEqual(metadata["duplicate_group_count"], 0)

    def test_existing_bundle_is_deduplicated_after_gold_normalization_without_source_edits(self):
        with tempfile.TemporaryDirectory() as temp:
            project = Path(temp)
            source = project / "data" / "existing_discovery"
            source.mkdir(parents=True)
            source_gold = copy.deepcopy(self.gold)
            source_gold[0]["gold"] = "+004"
            source_gold[1]["gold"] = "ALICE"
            for name, rows in (("public.jsonl", self.rows), ("gold.jsonl", source_gold)):
                (source / name).write_text(
                    "".join(json.dumps(row, allow_nan=False) + "\n" for row in rows), encoding="utf8")
            (source / "PROVENANCE.json").write_text('{"source":"temporary test bundle"}\n', encoding="utf8")
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            run = project / "run"
            with patch.object(dataset_module, "PROJECT", project):
                seal = prepare_dataset("existing_discovery", run, {})
                self.assertEqual(load_public(run), self.expected_public)
                self.assertEqual(load_gold(run), {row["uid"]: row["gold"] for row in self.expected_gold})
                self.assertEqual(seal["uid_count"], 3)
                self.assertEqual(seal["source_row_count"], 6)
                self.assertEqual(seal["distinct_question_count"], 3)
                self.assertEqual(seal["provenance"]["deduplication"],
                                 dataset_module._deduplicate_existing(self.rows, self.gold)[2])
                self.assertEqual(prepare_dataset("existing_discovery", run, {}), seal)
            self.assertEqual({path.name: path.read_bytes() for path in source.iterdir()}, before)

    def test_duplicate_gsm_questions_are_rejected_without_downloading(self):
        rows = [{"uid": "1" * 64, "family": "GSM8K", "prompt": "The same question"},
                {"uid": "2" * 64, "family": "GSM8K", "prompt": "The same question"}]
        gold = [{"uid": row["uid"], "gold": "4"} for row in rows]
        with tempfile.TemporaryDirectory() as temp:
            run = Path(temp) / "run"
            with patch.object(dataset_module, "_gsm8k_rows",
                              return_value=(rows, gold, {"kind": "gsm8k"})):
                with self.assertRaises(ValueError):
                    prepare_dataset("gsm8k", run, {})
            self.assertFalse((run / "data" / "DATASET_SEAL.json").exists())


if __name__ == "__main__":
    unittest.main()
