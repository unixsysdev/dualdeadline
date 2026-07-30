from pathlib import Path

from specstream.io import atomic_json


def test_atomic_json(tmp_path: Path) -> None:
    output = tmp_path / "nested" / "value.json"
    atomic_json(output, {"b": 2, "a": 1})
    assert output.read_text() == '{\n  "a": 1,\n  "b": 2\n}\n'

