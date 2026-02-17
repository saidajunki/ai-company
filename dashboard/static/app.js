// ===========================
// AI会社 ダッシュボード - app.js
// ポーリングエンジン + UI更新ロジック
// ===========================

'use strict';

// --- 定数 ---
const POLL_INTERVAL = 10000; // デフォルト10秒
const MAX_VISIBLE_TASKS = 20;
const MAX_CONTENT_LENGTH = 200;

// --- ユーティリティ関数 ---

/** HTML特殊文字をエスケープ */
const escapeHtml = (str) => {
  if (str == null) return '';
  const s = String(str);
  const map = { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' };
  return s.replace(/[&<>"']/g, (c) => map[c]);
};

/** ISO日時を "MM/DD HH:mm" (JST) にフォーマット */
const formatTime = (isoString) => {
  if (!isoString) return '--';
  try {
    const d = new Date(isoString);
    if (isNaN(d.getTime())) return '--';
    return d.toLocaleString('ja-JP', {
      timeZone: 'Asia/Tokyo',
      month: '2-digit',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      hour12: false,
    }).replace(/\//g, '/');
  } catch {
    return '--';
  }
};

/** 数値を "$X.XXXX" にフォーマット */
const formatUSD = (amount) => {
  if (amount == null || isNaN(amount)) return '--';
  return `$${Number(amount).toFixed(4)}`;
};

// --- UI更新関数 ---

/** ハートビート表示を更新 */
const updateHeartbeat = (heartbeat) => {
  const statusEl = document.getElementById('heartbeat-status');
  const wipEl = document.getElementById('heartbeat-wip');
  const updatedEl = document.getElementById('heartbeat-updated');

  if (!heartbeat) {
    statusEl.textContent = '不明';
    statusEl.className = 'badge badge-gray';
    wipEl.innerHTML = '<li>データなし</li>';
    updatedEl.textContent = '--';
    return;
  }

  // ステータスマッピング
  const statusMap = {
    running: { label: '稼働中', cls: 'badge badge-green' },
    idle: { label: '待機中', cls: 'badge badge-gray' },
    waiting_approval: { label: '承認待ち', cls: 'badge badge-orange badge-pulse' },
  };
  const info = statusMap[heartbeat.status] || { label: heartbeat.status, cls: 'badge badge-gray' };
  statusEl.textContent = info.label;
  statusEl.className = info.cls;

  // WIPリスト
  const wip = heartbeat.current_wip;
  if (Array.isArray(wip) && wip.length > 0) {
    wipEl.innerHTML = wip.map((t) => `<li>${escapeHtml(t)}</li>`).join('');
  } else {
    wipEl.innerHTML = '<li>なし</li>';
  }

  updatedEl.textContent = formatTime(heartbeat.updated_at);
};

/** コスト表示を更新 */
const updateCost = (cost) => {
  const windowEl = document.getElementById('cost-window');
  const totalEl = document.getElementById('cost-total');
  const limitEl = document.getElementById('cost-limit');
  const barEl = document.getElementById('cost-bar');
  const containerEl = barEl?.parentElement;

  if (!cost) {
    windowEl.textContent = '--';
    totalEl.textContent = '--';
    limitEl.textContent = '--';
    if (barEl) {
      barEl.style.width = '0%';
      barEl.className = 'cost-bar';
    }
    return;
  }

  windowEl.textContent = formatUSD(cost.window_60min_usd);
  totalEl.textContent = formatUSD(cost.total_usd);
  limitEl.textContent = formatUSD(cost.budget_limit_usd);

  // プログレスバー
  const pct = Math.min(cost.budget_usage_percent ?? 0, 100);
  barEl.style.width = `${pct}%`;

  // 色クラス: <80% → green(デフォルト), 80-99% → warning, >=100% → danger
  let barClass = 'cost-bar';
  if (cost.budget_usage_percent >= 100) {
    barClass = 'cost-bar danger';
  } else if (cost.budget_usage_percent >= 80) {
    barClass = 'cost-bar warning';
  }
  barEl.className = barClass;

  // aria属性更新
  if (containerEl) {
    containerEl.setAttribute('aria-valuenow', Math.round(pct));
  }
};

/** エージェント一覧テーブルを更新 */
const updateAgents = (agents) => {
  const tbody = document.getElementById('agents-list');
  if (!Array.isArray(agents) || agents.length === 0) {
    tbody.innerHTML = '<tr><td colspan="4">エージェントなし</td></tr>';
    return;
  }

  const statusMap = {
    active: { label: 'アクティブ', cls: 'badge badge-green' },
    inactive: { label: '非アクティブ', cls: 'badge badge-gray' },
  };

  tbody.innerHTML = agents.map((a) => {
    const s = statusMap[a.status] || { label: a.status, cls: 'badge badge-gray' };
    return `<tr>
      <td>${escapeHtml(a.name)}</td>
      <td>${escapeHtml(a.role)}</td>
      <td>${escapeHtml(a.model)}</td>
      <td><span class="${s.cls}">${escapeHtml(s.label)}</span></td>
    </tr>`;
  }).join('');
};

/** タスクサマリーを更新 */
const updateTasksSummary = (summary) => {
  const container = document.getElementById('tasks-summary');
  if (!summary) return;

  const items = [
    { label: '待機', cls: 'badge-yellow', count: summary.pending },
    { label: '実行中', cls: 'badge-blue', count: summary.running },
    { label: '完了', cls: 'badge-green', count: summary.completed },
    { label: '失敗', cls: 'badge-red', count: summary.failed },
  ];

  container.innerHTML = items.map((item) => `
    <div class="summary-item">
      <span class="badge ${item.cls}">${item.label}</span>
      <span class="count">${item.count ?? 0}</span>
    </div>
  `).join('');
};

/** 最近のタスク一覧を更新 */
const updateRecentTasks = (tasks) => {
  const ul = document.getElementById('tasks-list');
  if (!Array.isArray(tasks) || tasks.length === 0) {
    ul.innerHTML = '<li>タスクなし</li>';
    return;
  }

  const statusMap = {
    pending: { label: '待機', cls: 'badge-yellow' },
    running: { label: '実行中', cls: 'badge-blue' },
    completed: { label: '完了', cls: 'badge-green' },
    failed: { label: '失敗', cls: 'badge-red' },
  };

  const visible = tasks.slice(0, MAX_VISIBLE_TASKS);
  ul.innerHTML = visible.map((t) => {
    const s = statusMap[t.status] || { label: t.status, cls: 'badge-gray' };
    return `<li>
      <span class="badge ${s.cls}">${escapeHtml(s.label)}</span>
      ${escapeHtml(t.description)}
    </li>`;
  }).join('');
};

/** イニシアチブ一覧を更新（ステータスごとにグループ化） */
const updateInitiatives = (initiatives) => {
  const container = document.getElementById('initiatives-list');
  if (!Array.isArray(initiatives) || initiatives.length === 0) {
    container.innerHTML = '<p>イニシアチブなし</p>';
    return;
  }

  const statusLabels = {
    in_progress: '進行中',
    consulting: '相談中',
    planned: '計画中',
    completed: '完了',
    abandoned: '中止',
  };

  // ステータスごとにグループ化
  const groups = {};
  for (const ini of initiatives) {
    const st = ini.status || 'unknown';
    if (!groups[st]) groups[st] = [];
    groups[st].push(ini);
  }

  let html = '';
  for (const [status, items] of Object.entries(groups)) {
    const label = statusLabels[status] || status;
    html += `<div class="initiative-group"><h3>${escapeHtml(label)}</h3>`;
    for (const ini of items) {
      html += `<div class="initiative-card">
        <strong>${escapeHtml(ini.title)}</strong>
        <div style="font-size:0.82rem;color:var(--text-secondary);margin-top:0.3rem;">
          ${escapeHtml(ini.description || '')}
        </div>`;
      // 推定スコア表示
      if (ini.estimated_scores) {
        html += '<div class="scores-grid">';
        for (const [key, val] of Object.entries(ini.estimated_scores)) {
          html += `<div><div style="color:var(--text-secondary);font-size:0.72rem;">${escapeHtml(key)}</div><div style="font-weight:700;">${val}</div></div>`;
        }
        html += '</div>';
      }
      html += '</div>';
    }
    html += '</div>';
  }

  container.innerHTML = html;
};

/** 相談事項一覧を更新 */
const updateConsultations = (consultations) => {
  const countEl = document.getElementById('consultations-count');
  const ul = document.getElementById('consultations-list');

  if (!Array.isArray(consultations) || consultations.length === 0) {
    countEl.hidden = true;
    ul.innerHTML = '<li>相談事項なし</li>';
    return;
  }

  // pending件数バッジ
  const pendingCount = consultations.filter((c) => c.status === 'pending').length;
  if (pendingCount > 0) {
    countEl.textContent = pendingCount;
    countEl.hidden = false;
  } else {
    countEl.hidden = true;
  }

  ul.innerHTML = consultations.map((c) => {
    const isPending = c.status === 'pending';
    const borderStyle = isPending ? 'border-left:3px solid var(--color-orange);padding-left:0.5rem;' : '';
    const taskInfo = c.related_task_id
      ? `<span style="font-size:0.75rem;color:var(--text-secondary);margin-left:0.5rem;">タスク: ${escapeHtml(c.related_task_id)}</span>`
      : '';
    return `<li style="${borderStyle}">
      <div>${escapeHtml(c.content)}</div>
      <div style="font-size:0.75rem;color:var(--text-secondary);margin-top:0.2rem;">
        ${formatTime(c.created_at)}${taskInfo}
      </div>
    </li>`;
  }).join('');
};

/** 会話履歴を更新 */
const updateConversations = (conversations) => {
  const container = document.getElementById('conversations-list');
  if (!Array.isArray(conversations) || conversations.length === 0) {
    container.innerHTML = '<p>会話なし</p>';
    return;
  }

  const roleMap = {
    user: { label: 'Creator', cls: 'badge-blue' },
    assistant: { label: 'AI', cls: 'badge-green' },
    system: { label: 'System', cls: 'badge-gray' },
  };

  container.innerHTML = conversations.map((conv) => {
    const r = roleMap[conv.role] || { label: conv.role, cls: 'badge-gray' };
    const content = conv.content || '';
    const truncated = content.length > MAX_CONTENT_LENGTH;
    const visibleText = truncated ? content.slice(0, MAX_CONTENT_LENGTH) + '...' : content;
    // data属性に全文を保持し、クリックで展開
    const dataAttr = truncated ? ` data-full="${escapeHtml(content)}" data-truncated="${escapeHtml(visibleText)}"` : '';
    const clickAttr = truncated ? ' style="cursor:pointer;" onclick="toggleConversation(this)"' : '';

    return `<div class="conversation-entry">
      <span class="role-badge badge ${r.cls}">${r.label}</span>
      <time style="font-size:0.72rem;color:var(--text-secondary);">${formatTime(conv.timestamp)}</time>
      <div class="conversation-content"${dataAttr}${clickAttr}>${escapeHtml(visibleText)}</div>
    </div>`;
  }).join('');
};

/** 会話テキストの展開/折りたたみ */
function toggleConversation(el) {
  const contentEl = el;
  const full = contentEl.getAttribute('data-full');
  const truncated = contentEl.getAttribute('data-truncated');
  if (!full || !truncated) return;

  const isExpanded = contentEl.getAttribute('data-expanded') === 'true';
  if (isExpanded) {
    contentEl.textContent = '';
    contentEl.innerHTML = escapeHtml(truncated);
    contentEl.setAttribute('data-expanded', 'false');
  } else {
    contentEl.textContent = '';
    contentEl.innerHTML = escapeHtml(full);
    contentEl.setAttribute('data-expanded', 'true');
  }
}

// --- ポーリングエンジン ---

let pollingTimer = null;

/** APIからダッシュボードデータを取得して全UIを更新 */
const fetchDashboard = async () => {
  const statusEl = document.getElementById('connection-status');
  const lastUpdatedEl = document.getElementById('last-updated');

  try {
    const res = await fetch('/api/dashboard');
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();

    // 接続ステータス: 成功
    statusEl.textContent = '接続中';
    statusEl.className = 'badge badge-green';

    // 最終更新時刻
    lastUpdatedEl.textContent = formatTime(data.timestamp || new Date().toISOString());

    // 各セクション更新
    updateHeartbeat(data.heartbeat);
    updateCost(data.cost);
    updateAgents(data.agents);
    updateTasksSummary(data.tasks_summary);
    updateRecentTasks(data.recent_tasks);
    updateInitiatives(data.initiatives);
    updateConsultations(data.consultations);
    updateConversations(data.conversations);
  } catch (err) {
    // 接続ステータス: エラー
    statusEl.textContent = '接続エラー';
    statusEl.className = 'badge badge-red';
    console.error('ダッシュボード取得エラー:', err);
  }
};

/** ポーリングを開始 */
const startPolling = () => {
  if (pollingTimer) clearInterval(pollingTimer);
  pollingTimer = setInterval(fetchDashboard, POLL_INTERVAL);
};

// --- 初期化 ---
document.addEventListener('DOMContentLoaded', () => {
  // 初回取得 → ポーリング開始
  fetchDashboard();
  startPolling();
});
