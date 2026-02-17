"""Context Builder — LLM呼び出し用のシステムプロンプトを構築する.

Provides:
- build_system_prompt: コンテキスト情報からシステムプロンプトを構築

Requirements: 2.1, 2.2, 2.3, 2.4, 2.5, 7.3
"""

from __future__ import annotations

from dataclasses import dataclass, field

from models import (
    ConstitutionModel,
    ConversationEntry,
    CommitmentEntry,
    CreatorReview,
    DecisionLogEntry,
    InitiativeEntry,
    ResearchNote,
    StrategyDirection,
    TaskEntry,
)


_RESULT_TRUNCATE_LEN = 200


@dataclass
class TaskHistoryContext:
    """タスク履歴コンテキスト."""

    completed: list[TaskEntry] = field(default_factory=list)
    failed: list[TaskEntry] = field(default_factory=list)
    running: list[TaskEntry] = field(default_factory=list)
    paused: list[TaskEntry] = field(default_factory=list)
    canceled: list[TaskEntry] = field(default_factory=list)


def build_system_prompt(
    constitution: ConstitutionModel | None,
    wip: list[str],
    recent_decisions: list[DecisionLogEntry],
    budget_spent: float,
    budget_limit: float,
    conversation_history: list[ConversationEntry] | None = None,
    vision_text: str | None = None,
    curated_memory_text: str | None = None,
    daily_memory_text: str | None = None,
    creator_reviews: list[CreatorReview] | None = None,
    research_notes: list[ResearchNote] | None = None,
    task_history: TaskHistoryContext | None = None,
    active_initiatives: list[InitiativeEntry] | None = None,
    strategy_direction: StrategyDirection | None = None,
    model_catalog_text: str | None = None,
    rolling_summary: str | None = None,
    recalled_memories: list[str] | None = None,
    slack_thread_context: str | None = None,
    open_commitments: list[CommitmentEntry] | None = None,
) -> str:
    """コンテキスト情報からシステムプロンプトを構築する."""
    sections: list[str] = [
        "あなたはAI会社の社長AIです。以下のコンテキストに基づいて行動してください。",
    ]

    # --- 会社憲法 ---
    sections.append(_build_constitution_section(constitution))

    # --- ビジョン・事業方針 ---
    sections.append(_build_vision_section(vision_text))

    # --- キュレートメモリ（絶対に忘れない） ---
    sections.append(_build_curated_memory_section(curated_memory_text))

    # --- 評価（Creatorスコア） ---
    sections.append(_build_creator_score_section(constitution, creator_reviews))

    # --- 現在のWIP ---
    sections.append(_build_wip_section(wip))

    # --- 約束/TODO（Commitments） ---
    sections.append(_build_commitments_section(open_commitments))

    # --- 直近の意思決定 ---
    sections.append(_build_decisions_section(recent_decisions))

    # --- 予算状況 ---
    sections.append(_build_budget_section(budget_spent, budget_limit))

    # --- イニシアチブ ---
    sections.append(_build_initiative_section(active_initiatives))

    # --- 戦略方針 ---
    sections.append(_build_strategy_section(strategy_direction))

    # --- リサーチノート ---
    sections.append(_build_research_section(research_notes))

    # --- 永続メモリ（要約） ---
    sections.append(_build_rolling_summary_section(rolling_summary))

    # --- 長期記憶（リコール） ---
    sections.append(_build_memory_recall_section(recalled_memories))

    # --- タスク履歴 ---
    sections.append(_build_task_history_section(task_history))

    # --- Slackスレッド（コンテキスト） ---
    sections.append(_build_slack_thread_section(slack_thread_context))

    # --- Dailyメモ（本日） ---
    sections.append(_build_daily_memory_section(daily_memory_text))

    # --- 会話履歴 ---
    sections.append(_build_conversation_section(conversation_history))

    # --- 利用可能なモデル ---
    model_section = _build_model_catalog_section(model_catalog_text)
    if model_section:
        sections.append(model_section)

    # --- VPSインフラ操作ガイダンス ---
    sections.append(_build_infra_guidance_section())

    # --- 応答フォーマット ---
    sections.append(_build_format_section())

    return "\n\n".join(sections)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _build_constitution_section(constitution: ConstitutionModel | None) -> str:
    lines = ["## 会社憲法"]
    if constitution is None:
        lines.append("憲法未設定")
    else:
        lines.append(f"- 目的: {constitution.purpose}")
        lines.append(
            f"- 予算上限: ${constitution.budget.limit_usd}/{constitution.budget.window_minutes}分"
        )
        lines.append(f"- WIP制限: {constitution.work_principles.wip_limit}件")
    return "\n".join(lines)


def _build_vision_section(vision_text: str | None) -> str:
    lines = ["## ビジョン・事業方針"]
    if not vision_text:
        lines.append("ビジョン未設定")
    else:
        lines.append(vision_text)
    return "\n".join(lines)


def _build_wip_section(wip: list[str]) -> str:
    lines = ["## 現在のWIP"]
    if not wip:
        lines.append("現在進行中のタスクなし")
    else:
        for item in wip:
            lines.append(f"- {item}")
    return "\n".join(lines)


def _build_decisions_section(recent_decisions: list[DecisionLogEntry]) -> str:
    lines = ["## 直近の意思決定"]
    if not recent_decisions:
        lines.append("直近の意思決定なし")
    else:
        for entry in recent_decisions[:5]:
            lines.append(f"- {entry.date}: {entry.decision}（理由: {entry.why}）")
    return "\n".join(lines)


def _build_budget_section(budget_spent: float, budget_limit: float) -> str:
    remaining = budget_limit - budget_spent
    return "\n".join([
        "## 予算状況",
        f"- 消費: ${budget_spent:.2f} / ${budget_limit:.2f}（残り: ${remaining:.2f}）",
    ])


def _build_initiative_section(
    active_initiatives: list[InitiativeEntry] | None,
) -> str:
    lines = ["## イニシアチブ"]
    if not active_initiatives:
        lines.append("アクティブなイニシアチブなし")
    else:
        for ini in active_initiatives:
            lines.append(f"- [{ini.status}] {ini.title}: {ini.description}")
    return "\n".join(lines)


def _build_strategy_section(
    strategy_direction: StrategyDirection | None,
) -> str:
    lines = ["## 戦略方針"]
    if strategy_direction is None or not strategy_direction.summary:
        lines.append("戦略方針未設定")
    else:
        lines.append(strategy_direction.summary)
    return "\n".join(lines)


def _build_curated_memory_section(curated_memory_text: str | None) -> str:
    lines = ["## キュレートメモリ（LTM / 絶対に忘れない）"]
    t = (curated_memory_text or "").strip()
    if not t:
        lines.append("（なし）")
        return "\n".join(lines)
    lines.append(t)
    return "\n".join(lines)


def _build_commitments_section(open_commitments: list[CommitmentEntry] | None) -> str:
    lines = ["## 約束/TODO（未完了Commitments）"]
    if not open_commitments:
        lines.append("なし")
        return "\n".join(lines)

    # Due date first, then created_at
    def _key(c: CommitmentEntry):
        due = c.due_date.isoformat() if c.due_date else "9999-99-99"
        return (due, c.created_at.isoformat())

    for c in sorted(open_commitments, key=_key)[:12]:
        title = (c.title or "").strip()
        content = (c.content or "").strip()
        first = content.splitlines()[0] if content else ""
        if len(first) > 140:
            first = first[:140] + "…"
        due = c.due_date.isoformat() if c.due_date else "n/a"
        if title:
            lines.append(f"- [{c.commitment_id}] {title} (due={due}) — {first}")
        else:
            lines.append(f"- [{c.commitment_id}] (due={due}) — {first}")
    return "\n".join(lines)


def _build_daily_memory_section(daily_memory_text: str | None) -> str:
    lines = ["## Dailyメモ（本日）"]
    t = (daily_memory_text or "").strip()
    if not t:
        lines.append("なし")
        return "\n".join(lines)
    lines.append(t)
    return "\n".join(lines)

def _build_rolling_summary_section(rolling_summary: str | None) -> str:
    if rolling_summary and rolling_summary.strip():
        text = rolling_summary.strip()
        if text.startswith("#"):
            return text
        return "\n".join(["## 永続メモリ（要約）", text])
    return "\n".join(["## 永続メモリ（要約）", "要約なし"])


def _build_memory_recall_section(recalled_memories: list[str] | None) -> str:
    lines = ["## 長期記憶（リコール）"]
    if not recalled_memories:
        lines.append("リコールなし")
        return "\n".join(lines)
    for it in recalled_memories[:12]:
        s = (it or "").strip()
        if not s:
            continue
        lines.append(s if s.startswith("-") else f"- {s}")
    return "\n".join(lines)

def _build_slack_thread_section(slack_thread_context: str | None) -> str:
    lines = ["## Slackスレッド（コンテキスト）"]
    if not slack_thread_context or not slack_thread_context.strip():
        lines.append("スレッドコンテキストなし")
        return "\n".join(lines)
    lines.append(slack_thread_context.strip())
    return "\n".join(lines)


def _build_infra_guidance_section() -> str:
    """VPSインフラ操作のガイダンスセクションを構築する."""
    return "\n".join([
        "## VPSインフラ操作",
        "",
        "あなたはVPS上で直接動作しています。<shell>タグで任意のコマンドを実行できます。",
        "Docker, Docker Compose, Traefik等すべてシェルコマンドで操作してください。",
        "",
        "### 環境情報",
        "- OS: Ubuntu, Docker 29.x, Docker Compose v2",
        "- Traefik v2.11 がリバースプロキシとして稼働中（apps-network上）",
        "- Let's Encrypt自動SSL（certresolver=letsencrypt）",
        "- サービスディレクトリ: /opt/apps/{サービス名}/",
        "- Dockerネットワーク: apps-network（external）",
        "",
        "### 新サービスのデプロイ手順",
        "1. ディレクトリ作成: `mkdir -p /opt/apps/{サービス名}`",
        "2. docker-compose.yml を作成（cat > で書き出し）",
        "3. `cd /opt/apps/{サービス名} && docker compose up -d`",
        "4. `docker compose ps` でステータス確認",
        "",
        "### Traefikラベル（docker-compose.yml内で設定）",
        "```yaml",
        "labels:",
        '  - "traefik.enable=true"',
        '  - "traefik.http.routers.{サービス名}.rule=Host(`{ドメイン}`)"',
        '  - "traefik.http.routers.{サービス名}.entrypoints=websecure"',
        '  - "traefik.http.routers.{サービス名}.tls.certresolver=letsencrypt"',
        '  - "traefik.http.services.{サービス名}.loadbalancer.server.port={ポート}"',
        "```",
        "※ メインサービスのみにラベルを付与。apps-networkに接続すること。",
        "",
        "### WordPressデプロイ例",
        "```yaml",
        "services:",
        "  wordpress:",
        "    image: wordpress:latest",
        "    restart: unless-stopped",
        "    environment:",
        "      WORDPRESS_DB_HOST: db:3306",
        "      WORDPRESS_DB_USER: wordpress",
        "      WORDPRESS_DB_PASSWORD: {ランダム生成}",
        "      WORDPRESS_DB_NAME: wordpress",
        "    volumes:",
        "      - wp_data:/var/www/html",
        "    networks:",
        "      - apps-network",
        "    labels:",
        '      - "traefik.enable=true"',
        '      - "traefik.http.routers.{名前}.rule=Host(`{ドメイン}`)"',
        '      - "traefik.http.routers.{名前}.entrypoints=websecure"',
        '      - "traefik.http.routers.{名前}.tls.certresolver=letsencrypt"',
        '      - "traefik.http.services.{名前}.loadbalancer.server.port=80"',
        "  db:",
        "    image: mysql:8.0",
        "    restart: unless-stopped",
        "    environment:",
        "      MYSQL_ROOT_PASSWORD: {ランダム生成}",
        "      MYSQL_DATABASE: wordpress",
        "      MYSQL_USER: wordpress",
        "      MYSQL_PASSWORD: {上と同じ}",
        "    volumes:",
        "      - db_data:/var/lib/mysql",
        "volumes:",
        "  wp_data:",
        "  db_data:",
        "networks:",
        "  apps-network:",
        "    external: true",
        "```",
        "",
        "### ライフサイクル管理コマンド",
        "- 起動: `cd /opt/apps/{名前} && docker compose up -d`",
        "- 停止: `cd /opt/apps/{名前} && docker compose down`",
        "- 再起動: `cd /opt/apps/{名前} && docker compose restart`",
        "- ログ: `cd /opt/apps/{名前} && docker compose logs --tail=100`",
        "- 状態: `cd /opt/apps/{名前} && docker compose ps`",
        "- 全コンテナ一覧: `docker ps --format 'table {{.Names}}\\t{{.Status}}\\t{{.Ports}}'`",
        "",
        "### 注意事項",
        "- パスワードは `openssl rand -base64 24` 等で生成する",
        "- compose.ymlの書き出しは `cat > /opt/apps/{名前}/docker-compose.yml << 'EOF'` を使う",
        "- デプロイ後は必ず `docker compose ps` で正常起動を確認する",
        "- 問題があれば `docker compose logs` でログを確認する",
    ])


def _build_format_section() -> str:
    return "\n".join([
        "## 応答フォーマット",
        "以下のタグを使って応答してください:",
        "",
        "<reply>",
        "Creatorへの返信テキスト",
        "</reply>",
        "",
        "<research>",
        "Web検索したいクエリ（例: 最新の○○ / 価格 / 仕様 / 競合事例 / ニュース）",
        "</research>",
        "※ 外部情報（最新情報・料金・規約・仕様・ニュース等）が必要/不確かな場合は、まず<research>で確認してください。",
        "※ 検索結果は後続の入力として渡されるので、それを踏まえて判断・要約・次アクションに進んでください。",
        "",
        "<shell>",
        "実行するシェルコマンド",
        "</shell>",
        "",
        "<publish>",
        "create_repo:repo_name:description もしくは commit:repo_path:message",
        "</publish>",
        "※ 公開が必要な場合に使用（GitHub等のリポジトリ作成/コミット&push）。",
        "",
        "<memory>",
        "curated: タイトル（任意）",
        "本文（価値観/方針/禁止事項/重要な合意など、絶対に忘れたくない内容）",
        "</memory>",
        "※ 重要な価値観・方向性・ルール・大事な合意は、<memory>で保存してください（失われないように）。",
        "",
        "<memory>",
        "daily: タイトル（任意）",
        "本文（今日の学び/やったこと/次の作戦 など）",
        "</memory>",
        "※ 今日の学び/作戦はdailyに追記して、後から追える形にしてください。",
        "",
        "<commitment>",
        "add: タイトル（任意）",
        "本文（約束/TODO。いつか必ずやること・忘れたくないこと）",
        "</commitment>",
        "",
        "<commitment>",
        "close <commitment_id>: 完了メモ（任意）",
        "</commitment>",
        "※ 約束/TODOはcommitmentとして記録し、完了したらcloseしてください。",
        "",
        "<done>",
        "タスク完了の要約",
        "</done>",
        "",
        "<consult>",
        "Creatorに相談したい内容（質問 + 選択肢 + 推奨 + 影響 + 上限コスト）",
        "</consult>",
        "",
        "<plan>",
        "複雑なタスクを分解する場合に使用。番号付きサブタスクと依存関係を記述:",
        "1. サブタスク1の説明",
        "2. サブタスク2の説明 [depends:1]",
        "3. サブタスク3の説明 [depends:1,2]",
        "</plan>",
        "※ 指示が複数ステップを要する複雑なものの場合に<plan>タグを使用してください。",
        "※ 単純な指示には<plan>タグは不要です。従来のタグで直接応答してください。",
        "",
        "<delegate>",
        "role:タスク説明 model=モデル名",
        "</delegate>",
        "※ model=は省略可能。省略時はデフォルトモデルを使用。",
        "",
        "<control>",
        "pause <task_id>: 理由",
        "cancel <task_id>: 理由",
        "resume <task_id>",
        "</control>",
        "※ タスクの保留/中止/再開など、ステータス変更が必要な場合に使用。",
        "",
        "タグは複数組み合わせ可能です。タグなしの場合は全体がreplyとして扱われます。",
        "",
        "## 重要: 誠実性（虚偽申告禁止）",
        "- 実際に実行していないことを「した/する」と断言しない。",
        "- 「保留/中止/再開」などのステータス変更は、反映されたことを確認してから報告する。不明ならCreatorにtask_id/consult_idの提示を求める。",
        "",
        "## 重要: 外部情報の扱い（風潮）",
        "- 「最新/価格/仕様/規約/ニュース」など時間で変わる事実は、推測せず<research>で確認してから話す。",
        "- 可能な限り日付（published_at/retrieved_at）を添えて解釈し、出典URLを残す。",
    ])
def _build_model_catalog_section(model_catalog_text: str | None) -> str:
    """利用可能なモデルセクションを構築する."""
    if not model_catalog_text:
        return ""
    return "\n".join([
        "## 利用可能なモデル",
        "",
        "社員エージェントに委任する際、タスクに適したモデルを選択できます。",
        "",
        model_catalog_text,
        "",
        "タスク種別ごとの推奨:",
        "- コーディング・分析: coding/analysisカテゴリのモデル",
        "- 簡単なタスク・情報収集: fast/cheapカテゴリのモデル",
        "- 汎用タスク: generalカテゴリのモデル",
    ])




def _truncate(text: str, max_len: int = _RESULT_TRUNCATE_LEN) -> str:
    """文字列を指定長に切り詰める."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "…"


def _build_task_history_section(
    task_history: TaskHistoryContext | None,
) -> str:
    lines = ["## タスク履歴"]
    if task_history is None or (
        not task_history.completed and not task_history.failed and not task_history.running
        and not task_history.paused and not task_history.canceled
    ):
        lines.append("タスク履歴なし")
        return "\n".join(lines)

    if task_history.running:
        lines.append("### 実行中タスク")
        for t in task_history.running:
            lines.append(f"- [{t.task_id}] {t.description}")

    if task_history.completed:
        lines.append("### 最近完了したタスク")
        for t in task_history.completed:
            result = _truncate(t.result) if t.result else "結果なし"
            lines.append(f"- [{t.task_id}] {t.description} → {result}")

    if task_history.failed:
        lines.append("### 最近失敗したタスク")
        for t in task_history.failed:
            error = _truncate(t.error) if t.error else "原因不明"
            lines.append(f"- [{t.task_id}] {t.description} — エラー: {error}")

    if task_history.paused:
        lines.append("### 保留中タスク")
        for t in task_history.paused:
            note = _truncate(t.error) if t.error else "理由なし"
            lines.append(f"- [{t.task_id}] {t.description} — 理由: {note}")

    if task_history.canceled:
        lines.append("### 中止タスク")
        for t in task_history.canceled:
            note = _truncate(t.error) if t.error else "理由なし"
            lines.append(f"- [{t.task_id}] {t.description} — 理由: {note}")

    return "\n".join(lines)


def _build_research_section(
    research_notes: list[ResearchNote] | None,
) -> str:
    lines = ["## リサーチノート"]
    if not research_notes:
        lines.append("リサーチノートなし")
    else:
        for note in research_notes[:10]:
            ts = note.retrieved_at.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{ts}] {note.source_url}\n  {note.summary}")
    return "\n".join(lines)

def _build_conversation_section(
    conversation_history: list[ConversationEntry] | None,
) -> str:
    lines = ["## 会話履歴"]
    if not conversation_history:
        lines.append("会話履歴なし")
    else:
        for entry in conversation_history:
            ts = entry.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            lines.append(f"- [{entry.role}] {ts}: {entry.content}")
    return "\n".join(lines)


def _build_creator_score_section(
    constitution: ConstitutionModel | None,
    creator_reviews: list[CreatorReview] | None,
) -> str:
    lines = ["## 評価（Creatorスコア）"]

    policy = getattr(constitution, "creator_score_policy", None) if constitution else None
    if not policy or not getattr(policy, "enabled", False):
        lines.append("Creatorスコア未設定")
        return "\n".join(lines)

    lines.append(f"- 優先: {policy.priority}")
    lines.append("- 採点軸: 面白さ / コスト効率 / 現実性 / 進化性（各0-25、合計0-100）")

    try:
        axes = getattr(policy, "axes", {}) or {}
        for name in ("面白さ", "コスト効率", "現実性", "進化性"):
            desc = axes.get(name, "")
            if desc:
                short = desc.replace("\n", " ").strip()
                if len(short) > 140:
                    short = short[:140] + "…"
                lines.append(f"- {name}: {short}")
    except Exception:
        pass

    if creator_reviews:
        lines.append("### 直近のCreatorレビュー")
        for r in creator_reviews[-3:]:
            ts = r.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            axis = []
            if r.score_interestingness_25 is not None:
                axis.append(f"面白さ{r.score_interestingness_25}/25")
            if r.score_cost_efficiency_25 is not None:
                axis.append(f"コスト効率{r.score_cost_efficiency_25}/25")
            if r.score_realism_25 is not None:
                axis.append(f"現実性{r.score_realism_25}/25")
            if r.score_evolvability_25 is not None:
                axis.append(f"進化性{r.score_evolvability_25}/25")
            axis_text = " ".join(axis) if axis else "軸スコアなし"

            comment = (r.comment or "").replace("\n", " ").strip()
            if len(comment) > 200:
                comment = comment[:200] + "…"
            if comment:
                lines.append(f"- [{ts}] {r.score_total_100}/100 ({axis_text}) — {comment}")
            else:
                lines.append(f"- [{ts}] {r.score_total_100}/100 ({axis_text})")
    else:
        lines.append("直近レビューなし（Creator日報に対する採点を待っています）")

    return "\n".join(lines)
