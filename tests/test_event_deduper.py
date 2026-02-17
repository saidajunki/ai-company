from __future__ import annotations

from event_deduper import EventDeduper


def test_deduper_dedups_same_key():
    d = EventDeduper(ttl_seconds=900)
    assert d.should_process("C1:1.0") is True
    assert d.should_process("C1:1.0") is False
    assert d.should_process("C1:2.0") is True
    assert d.should_process("C2:1.0") is True


def test_deduper_allows_empty_key():
    d = EventDeduper(ttl_seconds=900)
    assert d.should_process("") is True


def test_deduper_disabled_when_ttl_zero():
    d = EventDeduper(ttl_seconds=0)
    assert d.should_process("k") is True
    assert d.should_process("k") is True

