from __future__ import annotations

from manager import init_company_directory
from procedure_store import ProcedureStore


def test_ingest_and_recall_verbatim_block(tmp_path):
    company_id = "procedure-test"
    init_company_directory(tmp_path, company_id)
    store = ProcedureStore(tmp_path, company_id)

    text = """
手順名: VPS再起動後リカバリ
```bash
cd /opt/apps/ai-company
git pull origin main
.venv/bin/pip install --force-reinstall .
systemctl restart ai-company
systemctl status ai-company --no-pager -n 20
```
""".strip()

    result = store.ingest_text(text, source="creator_message")
    assert len(result.created) == 1
    doc = result.created[0]
    assert doc.name == "VPS再起動後リカバリ"
    assert doc.steps[0] == "cd /opt/apps/ai-company"
    assert doc.steps[-1] == "systemctl status ai-company --no-pager -n 20"

    recalled = store.find_best_for_request("VPS再起動後リカバリ手順をもう一度教えて")
    assert recalled is not None
    reply = store.render_reply(recalled)
    assert "```bash" in reply
    assert "systemctl restart ai-company" in reply
    assert "source_of_truth:" in reply


def test_shared_procedure_updates_shared_index(tmp_path):
    company_id = "procedure-shared"
    init_company_directory(tmp_path, company_id)
    store = ProcedureStore(tmp_path, company_id)

    text = """
手順名: 社内共有デプロイ手順
これは社内共有ドキュメントとして保存してください。
1. cd /opt/apps/ai-company
2. git pull origin main
3. systemctl restart ai-company
""".strip()
    result = store.ingest_text(text, source="creator_message")
    assert len(result.created) == 1
    assert result.created[0].visibility == "shared"

    shared_index = tmp_path / "companies" / company_id / "knowledge" / "shared" / "INDEX.md"
    content = shared_index.read_text(encoding="utf-8")
    assert "社内共有デプロイ手順" in content


def test_forget_hint_does_not_store(tmp_path):
    company_id = "procedure-forget"
    init_company_directory(tmp_path, company_id)
    store = ProcedureStore(tmp_path, company_id)

    text = """
手順名: 一時的テスト手順
これは重要ではないので忘れてください。
1. echo test
2. echo done
""".strip()

    result = store.ingest_text(text, source="creator_message")
    assert len(result.created) == 0
    assert store.list_all() == []
