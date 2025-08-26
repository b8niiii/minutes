import json
from pipeline import make_chunks, merge_partials


def test_make_chunks_time_slicing():
    stitched = [
        {"speaker": "S1", "start": 0, "end": 50, "text": "a"},
        {"speaker": "S2", "start": 50, "end": 220, "text": "b"},
        {"speaker": "S1", "start": 220, "end": 400, "text": "c"},
    ]
    chunks = make_chunks(stitched, chunk_seconds=180)
    assert len(chunks) == 3
    assert chunks[0]["start"] == 0
    assert chunks[1]["start"] == 180
    assert chunks[2]["start"] == 360
    assert chunks[0]["entries"][0]["speaker"] == "S1"
    assert chunks[1]["entries"][0]["speaker"] == "S2"
    assert chunks[2]["entries"][0]["speaker"] == "S1"


def test_merge_partials_dedup():
    partials = [
        {
            "chunk_window": {"start": "00:00:00", "end": "00:03:00"},
            "topic_title": "Intro",
            "key_points": ["A"],
            "decisions": [{"text": "Approve budget", "rationale": "Need growth"}],
            "action_items": [
                {"owner": "Alice", "task": "Prepare report", "due": "TBD"}
            ],
            "risks_or_blockers": [],
            "open_questions": [],
            "mentions": [],
        },
        {
            "chunk_window": {"start": "00:03:00", "end": "00:06:00"},
            "topic_title": "Budget",
            "key_points": ["B"],
            "decisions": [{"text": "approve budget", "rationale": "Need growth"}],
            "action_items": [
                {
                    "owner": "Alice",
                    "task": "Prepare report",
                    "due": "2024-01-01",
                }
            ],
            "risks_or_blockers": ["Risk"],
            "open_questions": ["Q"],
            "mentions": [],
        },
    ]
    merged = merge_partials(partials)
    assert len(merged["decisions"]) == 1
    assert len(merged["action_items"]) == 1
    assert merged["action_items"][0]["due"] == "2024-01-01"
    assert "Risk" in merged["risks"]
    assert "Q" in merged["open_questions"]
