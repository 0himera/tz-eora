function $(id) { return document.getElementById(id); }

function escapeHtml(str) {
  return str
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#039;');
}

function formatAnswer(text) {
  // Convert [N] to links to #src-N and split into paragraphs
  const paragraphs = text.split(/\n\n+/).map(p => p.trim()).filter(Boolean);
  const htmlParas = paragraphs.map(p => {
    const safe = escapeHtml(p);
    const linked = safe.replace(/\[(\d+)\]/g, (m, n) => `<sup><a class="cite" href="#src-${n}">[${n}]</a></sup>`);
    return `<p>${linked}</p>`;
  });
  return htmlParas.join('\n');
}

async function health() {
  try {
    const r = await fetch('/api/health');
    if (!r.ok) return null;
    return await r.json();
  } catch (e) {
    return null;
  }
}

async function ask(question) {
  const res = await fetch('/api/ask', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({}));
    throw new Error(err.detail || `Ошибка ${res.status}`);
  }
  return await res.json();
}

function setStatus(msg, kind = 'ok') {
  const s = $('status');
  s.classList.remove('hidden', 'ok', 'err');
  s.classList.add(kind === 'ok' ? 'ok' : 'err');
  s.textContent = msg;
}

function clearStatus() {
  const s = $('status');
  s.classList.add('hidden');
}

function renderResult(answer, citations) {
  $('result').classList.remove('hidden');
  $('answer').innerHTML = formatAnswer(answer || '');
  const ul = $('citations');
  ul.innerHTML = '';
  const sorted = [...(citations || [])].sort((a, b) => Number(a.index) - Number(b.index));
  for (const c of sorted) {
    const li = document.createElement('li');
    li.id = `src-${c.index}`;
    const a = document.createElement('a');
    a.href = c.url;
    a.target = '_blank';
    a.rel = 'noopener noreferrer';
    a.textContent = `[${c.index}] ${c.title || c.url}`;
    li.appendChild(a);
    ul.appendChild(li);
  }
}

async function init() {
  const h = await health();
  if (h && h.status === 'ok') {
    setStatus(`Готово. Загрузили источников: ${h.docs}`);
    setTimeout(clearStatus, 1500);
  } else {
    setStatus('Сервис недоступен. Проверьте, что запущен бэкенд.', 'err');
  }

  $('sampleBtn').addEventListener('click', () => {
    $('question').value = 'Что вы можете сделать для ритейлеров?';
  });

  $('askBtn').addEventListener('click', async () => {
    const q = $('question').value.trim();
    if (!q) {
      setStatus('Введите вопрос.', 'err');
      return;
    }
    $('askBtn').disabled = true;
    setStatus('Думаю...');
    try {
      const { answer, citations } = await ask(q);
      clearStatus();
      renderResult(answer, citations);
    } catch (e) {
      setStatus(e.message || 'Ошибка', 'err');
    } finally {
      $('askBtn').disabled = false;
    }
  });
}

window.addEventListener('DOMContentLoaded', init);
