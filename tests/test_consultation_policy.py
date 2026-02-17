from __future__ import annotations

from datetime import datetime, timezone

from consultation_policy import assess_creator_consultation, should_escalate_task_failure
from models import ConstitutionModel, TaskEntry


def test_assess_creator_consultation_minor_by_default():
    a = assess_creator_consultation("公開先はGitHubで良いですか？")
    assert a.is_major is False


def test_assess_creator_consultation_cost_is_major():
    a = assess_creator_consultation("広告出稿に$100使っていい？")
    assert a.is_major is True
    assert a.reason in ("cost", "contract")


def test_assess_creator_consultation_direction_is_major():
    a = assess_creator_consultation("会社の方向性をピボットしたい")
    assert a.is_major is True
    assert a.reason == "direction"


def test_assess_creator_consultation_risk_is_major():
    a = assess_creator_consultation("著作権的に問題ないですか？")
    assert a.is_major is True
    assert a.reason == "risk"


def test_assess_creator_consultation_matches_constitution_scope_phrase():
    const = ConstitutionModel()
    scope_phrase = const.creator_intervention.scope[0]
    a = assess_creator_consultation(f"質問: {scope_phrase} を進めて良い？", constitution=const)
    assert a.is_major is True
    assert a.reason == "creator_scope"


def test_should_escalate_task_failure_creator_source():
    now = datetime.now(timezone.utc)
    task = TaskEntry(
        task_id="t1",
        description="minor",
        priority=5,
        source="creator",
        status="failed",
        created_at=now,
        updated_at=now,
        error="err",
        retry_count=3,
        max_retries=3,
    )
    assert should_escalate_task_failure(task) is True


def test_should_escalate_task_failure_priority_high():
    now = datetime.now(timezone.utc)
    task = TaskEntry(
        task_id="t1",
        description="minor",
        priority=1,
        source="autonomous",
        status="failed",
        created_at=now,
        updated_at=now,
        error="err",
        retry_count=3,
        max_retries=3,
    )
    assert should_escalate_task_failure(task) is True


def test_should_escalate_task_failure_minor_autonomous_low_priority_false():
    now = datetime.now(timezone.utc)
    task = TaskEntry(
        task_id="t1",
        description="minor",
        priority=5,
        source="autonomous",
        status="failed",
        created_at=now,
        updated_at=now,
        error="err",
        retry_count=3,
        max_retries=3,
    )
    assert should_escalate_task_failure(task) is False

