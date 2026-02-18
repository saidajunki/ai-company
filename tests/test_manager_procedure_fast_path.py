from __future__ import annotations

from manager import Manager, init_company_directory


def test_manager_recalls_procedure_after_unrelated_messages(tmp_path):
    company_id = "manager-procedure"
    init_company_directory(tmp_path, company_id)
    mgr = Manager(tmp_path, company_id)
    mgr.startup()

    sent: list[str] = []

    def _capture(text: str, **_kwargs):
        sent.append(text)

    mgr._slack_send = _capture  # type: ignore[method-assign]

    save_text = """
手順名: VPS再起動後リカバリ
```bash
cd /opt/apps/ai-company
git pull origin main
.venv/bin/pip install --force-reinstall .
systemctl restart ai-company
    ```
""".strip()
    mgr.process_message(save_text, user_id="U-test")
    assert len(mgr.procedure_store.list_active()) == 1

    sent.clear()
    mgr.process_message("さっきのVPS再起動後リカバリ手順をもう一度教えて", user_id="U-test")
    assert sent
    assert "source_of_truth:" in sent[-1]
    assert "git pull origin main" in sent[-1]

    sent.clear()
    mgr.process_message("関係ない雑談です。今日は眠いです。", user_id="U-test")
    assert any("LLMクライアントが設定されていません" in m for m in sent)

    sent.clear()
    mgr.process_message("VPS再起動後リカバリ手順を再掲して", user_id="U-test")
    assert sent
    assert "systemctl restart ai-company" in sent[-1]
