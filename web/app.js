const els = {
  domain: document.getElementById('domain'),
  apiBase: document.getElementById('apiBase'),
  saveConfig: document.getElementById('saveConfig'),
  chatLog: document.getElementById('chatLog'),
  message: document.getElementById('message'),
  send: document.getElementById('send'),
  status: document.getElementById('status'),
  quickQuestions: document.getElementById('quickQuestions'),
};

const state = {
  history: [],
  apiBase: localStorage.getItem('career.apiBase') || '',
  domain: localStorage.getItem('career.domain') || '',
};

const sampleQuestions = [
  '학생의 흥미와 강점을 기준으로 적합한 진로 방향을 제안해줘.',
  '상담 대화 내용을 바탕으로 이번 달 실천 계획 3가지를 추천해줘.',
  '추천 직업군별로 필요한 역량과 학교에서 준비할 방법을 알려줘.',
];

function baseUrl() {
  return (state.apiBase || '').trim() || window.location.origin;
}

function setStatus(msg = '') {
  if (!msg) {
    els.status.classList.add('hidden');
    return;
  }
  els.status.classList.remove('hidden');
  els.status.textContent = msg;
}

function escapeHtml(s) {
  return (s || '').replace(/[&<>'"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', "'": '&#39;', '"': '&quot;' }[c]));
}

function addBubble(role, text, citations = []) {
  const isUser = role === 'user';
  const wrap = document.createElement('div');
  wrap.className = `flex ${isUser ? 'justify-end' : 'justify-start'}`;

  const citeHtml = !isUser && citations.length
    ? `<div class="mt-2 space-y-1">${citations.map((c, i) => `<div class="text-xs text-slate-600 border border-slate-200 rounded-lg p-2"><b>[${i + 1}]</b> ${escapeHtml(c.domain_name)} · ${escapeHtml(c.source_spec || 'unknown')}<br>${escapeHtml(c.excerpt || '')}</div>`).join('')}</div>`
    : '';

  wrap.innerHTML = `
    <div class="max-w-[85%] rounded-2xl px-4 py-3 text-sm ${isUser ? 'bg-indigo-600 text-white' : 'bg-white border border-slate-200'}">
      <div class="whitespace-pre-wrap leading-relaxed">${escapeHtml(text)}</div>
      ${citeHtml}
    </div>
  `;
  els.chatLog.appendChild(wrap);
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

async function sendMessage() {
  const message = (els.message.value || '').trim();
  if (!message) return;

  const domain = (els.domain.value || '').trim();
  addBubble('user', message);
  state.history.push({ role: 'user', content: message });
  els.message.value = '';
  setStatus('상담 답변 생성 중...');
  els.send.disabled = true;

  try {
    const res = await fetch(`${baseUrl()}/chat`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message,
        domain: domain || null,
        top_k: 5,
        history: state.history.slice(-8),
      }),
    });

    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    addBubble('assistant', data.answer || '답변이 비어 있습니다.', data.citations || []);
    state.history.push({ role: 'assistant', content: data.answer || '' });
    setStatus('');
  } catch (e) {
    setStatus(`오류: ${e.message}`);
    addBubble('assistant', '요청 처리 중 오류가 발생했습니다. API 설정과 서버 상태를 확인해주세요.');
  } finally {
    els.send.disabled = false;
  }
}

function renderQuickQuestions() {
  els.quickQuestions.innerHTML = sampleQuestions
    .map((q) => `<button class="w-full text-left rounded-lg border border-slate-200 px-3 py-2 text-sm hover:bg-slate-50">${escapeHtml(q)}</button>`)
    .join('');

  Array.from(els.quickQuestions.querySelectorAll('button')).forEach((btn) => {
    btn.addEventListener('click', () => {
      els.message.value = btn.textContent.trim();
      els.message.focus();
    });
  });
}

function init() {
  els.apiBase.value = state.apiBase;
  els.domain.value = state.domain;

  renderQuickQuestions();
  addBubble('assistant', '안녕하세요! 학생의 흥미·강점·상담기록을 바탕으로 진로탐색을 도와드릴게요.');

  els.saveConfig.addEventListener('click', () => {
    state.apiBase = (els.apiBase.value || '').trim();
    state.domain = (els.domain.value || '').trim();
    localStorage.setItem('career.apiBase', state.apiBase);
    localStorage.setItem('career.domain', state.domain);
    setStatus('설정을 저장했습니다.');
    setTimeout(() => setStatus(''), 1600);
  });

  els.send.addEventListener('click', sendMessage);
  els.message.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') sendMessage();
  });
}

init();
