"""Unit tests for merge_datasets.py — validate_record and load_jsonl integration."""

from __future__ import annotations

import io
import json
import sys
import tempfile
from pathlib import Path

import pytest

# Import the functions under test
from merge_datasets import validate_record, load_jsonl, write_jsonl


# ---------------------------------------------------------------------------
# validate_record — unit tests
# ---------------------------------------------------------------------------

class TestValidateRecord:
    """Tests for validate_record(record, source_path, line_number)."""

    def _path(self) -> Path:
        return Path("test_file.jsonl")

    def test_valid_record_returns_true(self, capsys):
        record = {"instruction": "Do something", "input": "some input", "output": "some output"}
        assert validate_record(record, self._path(), 1) is True
        captured = capsys.readouterr()
        assert captured.err == ""

    def test_missing_instruction_returns_false(self, capsys):
        record = {"input": "x", "output": "y"}
        assert validate_record(record, self._path(), 5) is False
        captured = capsys.readouterr()
        assert "missing key 'instruction'" in captured.err
        assert "line 5" in captured.err

    def test_missing_input_returns_false(self, capsys):
        record = {"instruction": "x", "output": "y"}
        assert validate_record(record, self._path(), 10) is False
        captured = capsys.readouterr()
        assert "missing key 'input'" in captured.err
        assert "line 10" in captured.err

    def test_missing_output_returns_false(self, capsys):
        record = {"instruction": "x", "input": "y"}
        assert validate_record(record, self._path(), 3) is False
        captured = capsys.readouterr()
        assert "missing key 'output'" in captured.err

    def test_empty_instruction_returns_false(self, capsys):
        record = {"instruction": "", "input": "x", "output": "y"}
        assert validate_record(record, self._path(), 2) is False
        captured = capsys.readouterr()
        assert "empty value for key 'instruction'" in captured.err

    def test_whitespace_only_instruction_returns_false(self, capsys):
        record = {"instruction": "   \t\n", "input": "x", "output": "y"}
        assert validate_record(record, self._path(), 7) is False
        captured = capsys.readouterr()
        assert "empty value for key 'instruction'" in captured.err

    def test_empty_output_returns_false(self, capsys):
        record = {"instruction": "x", "input": "y", "output": ""}
        assert validate_record(record, self._path(), 1) is False
        captured = capsys.readouterr()
        assert "empty value for key 'output'" in captured.err

    def test_whitespace_only_output_returns_false(self, capsys):
        record = {"instruction": "x", "input": "y", "output": "  "}
        assert validate_record(record, self._path(), 1) is False
        captured = capsys.readouterr()
        assert "empty value for key 'output'" in captured.err

    def test_warning_includes_source_path(self, capsys):
        path = Path("some/dataset.jsonl")
        record = {"instruction": "", "input": "x", "output": "y"}
        validate_record(record, path, 1)
        captured = capsys.readouterr()
        assert str(path) in captured.err

    def test_warning_includes_line_number(self, capsys):
        record = {"input": "x", "output": "y"}  # missing instruction
        validate_record(record, self._path(), 42)
        captured = capsys.readouterr()
        assert "42" in captured.err

    def test_extra_keys_are_ignored(self, capsys):
        record = {
            "instruction": "Do something",
            "input": "some input",
            "output": "some output",
            "extra_field": "ignored",
        }
        assert validate_record(record, self._path(), 1) is True
        assert capsys.readouterr().err == ""

    def test_empty_dict_returns_false(self, capsys):
        assert validate_record({}, self._path(), 1) is False
        assert capsys.readouterr().err != ""


# ---------------------------------------------------------------------------
# load_jsonl — integration with validate_record
# ---------------------------------------------------------------------------

class TestLoadJsonlValidation:
    """Tests that load_jsonl skips invalid records via validate_record."""

    def _write_jsonl(self, path: Path, records: list) -> None:
        with path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    def test_valid_records_are_loaded(self, tmp_path):
        records = [
            {"instruction": "A", "input": "B", "output": "C"},
            {"instruction": "D", "input": "E", "output": "F"},
        ]
        p = tmp_path / "data.jsonl"
        self._write_jsonl(p, records)
        result = load_jsonl(p)
        assert len(result) == 2

    def test_record_missing_key_is_skipped(self, tmp_path, capsys):
        records = [
            {"instruction": "A", "input": "B", "output": "C"},  # valid
            {"instruction": "A", "input": "B"},                  # missing output
        ]
        p = tmp_path / "data.jsonl"
        self._write_jsonl(p, records)
        result = load_jsonl(p)
        assert len(result) == 1
        assert result[0]["output"] == "C"
        assert "missing key 'output'" in capsys.readouterr().err

    def test_record_with_empty_value_is_skipped(self, tmp_path, capsys):
        records = [
            {"instruction": "A", "input": "B", "output": "C"},  # valid
            {"instruction": "", "input": "B", "output": "C"},   # empty instruction
        ]
        p = tmp_path / "data.jsonl"
        self._write_jsonl(p, records)
        result = load_jsonl(p)
        assert len(result) == 1
        assert result[0]["instruction"] == "A"

    def test_record_with_whitespace_value_is_skipped(self, tmp_path, capsys):
        records = [
            {"instruction": "A", "input": "B", "output": "C"},
            {"instruction": "A", "input": "   ", "output": "C"},  # whitespace input
        ]
        p = tmp_path / "data.jsonl"
        self._write_jsonl(p, records)
        result = load_jsonl(p)
        assert len(result) == 1

    def test_all_invalid_records_returns_empty_list(self, tmp_path):
        records = [
            {"instruction": "", "input": "B", "output": "C"},
            {"input": "B", "output": "C"},
        ]
        p = tmp_path / "data.jsonl"
        self._write_jsonl(p, records)
        result = load_jsonl(p)
        assert result == []

    def test_invalid_json_line_is_skipped(self, tmp_path, capsys):
        p = tmp_path / "data.jsonl"
        with p.open("w") as f:
            f.write('{"instruction": "A", "input": "B", "output": "C"}\n')
            f.write("not valid json\n")
        result = load_jsonl(p)
        assert len(result) == 1

    def test_file_not_found_exits(self, tmp_path):
        with pytest.raises(SystemExit) as exc_info:
            load_jsonl(tmp_path / "nonexistent.jsonl")
        assert exc_info.value.code == 1


# ---------------------------------------------------------------------------
# main() — zero valid records edge case
# ---------------------------------------------------------------------------

class TestMainZeroValidRecords:
    """Tests the zero-valid-records warning and empty output file behaviour."""

    def test_zero_valid_records_writes_empty_files_and_exits_0(self, tmp_path, capsys):
        """When both input files have no valid records, main() writes empty outputs and exits 0."""
        youtube = tmp_path / "youtube.jsonl"
        instagram = tmp_path / "instagram.jsonl"
        train_out = tmp_path / "train.jsonl"
        val_out = tmp_path / "val.jsonl"

        # Write files with only invalid records
        youtube.write_text('{"instruction": "", "input": "x", "output": "y"}\n', encoding="utf-8")
        instagram.write_text('{"input": "x", "output": "y"}\n', encoding="utf-8")

        # Patch sys.argv and run main
        import merge_datasets
        original_argv = sys.argv[:]
        sys.argv = [
            "merge_datasets.py",
            "--youtube", str(youtube),
            "--instagram", str(instagram),
            "--train-output", str(train_out),
            "--val-output", str(val_out),
        ]
        try:
            with pytest.raises(SystemExit) as exc_info:
                merge_datasets.main()
            assert exc_info.value.code == 0
        finally:
            sys.argv = original_argv

        # Both output files should exist and be empty
        assert train_out.exists()
        assert val_out.exists()
        assert train_out.read_text(encoding="utf-8") == ""
        assert val_out.read_text(encoding="utf-8") == ""

        # Warning should have been printed to stderr
        captured = capsys.readouterr()
        assert "No valid records found after filtering" in captured.err
