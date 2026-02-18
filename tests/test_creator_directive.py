from __future__ import annotations

from creator_directive import parse_creator_directive


def test_parse_explicit_cancel_command():
    directive = parse_creator_directive("中止 deadbeef: いったん止める")
    assert directive is not None
    assert directive.kind == "cancel"
    assert directive.task_id == "deadbeef"


def test_non_directive_sentence_with_stop_keyword_returns_none():
    directive = parse_creator_directive(
        "全く触れられていない事業を掘り起こした場合、継続か停止かの判断はどう進めますか？",
    )
    assert directive is None


def test_free_form_directive_with_task_id_is_detected():
    directive = parse_creator_directive(
        "この案件は一旦保留でお願いします task_id: cafefeed",
    )
    assert directive is not None
    assert directive.kind == "pause"
    assert directive.task_id == "cafefeed"


def test_long_multi_line_instruction_is_not_treated_as_directive():
    text = (
        "Webサイト運営の方針です。\\n"
        "停止時はCreatorへ報告してください。\\n"
        "最新AI技術を継続的に扱ってください。"
    )
    directive = parse_creator_directive(text)
    assert directive is None
