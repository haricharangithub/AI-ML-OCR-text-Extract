from src.text_extraction import find_target_line, normalize_text


def test_normalize_text():
    assert normalize_text("  hello   world  ") == "hello world"


def test_find_target_line_prefers_confidence():
    lines = [
        {"text": "foo 1234", "conf": 10},
        {"text": "163233702292313922_1_lWV", "conf": 50},
        {"text": "163233702292313922_1_lWV", "conf": 40},
    ]
    target = find_target_line(lines)
    assert target is not None
    assert target["text"] == "163233702292313922_1_lWV"
    assert target["conf"] == 50

