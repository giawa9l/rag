const API = '';

const uploadZone = document.getElementById('uploadZone');
const fileInput = document.getElementById('fileInput');
const status = document.getElementById('status');
const docList = document.getElementById('docList');
const searchInput = document.getElementById('searchInput');
const searchBtn = document.getElementById('searchBtn');
const searchResults = document.getElementById('searchResults');

function showStatus(msg, type) {
    status.textContent = msg;
    status.className = 'status ' + type;
}

uploadZone.addEventListener('click', () => fileInput.click());
uploadZone.addEventListener('dragover', e => {
    e.preventDefault();
    uploadZone.classList.add('dragover');
});
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
    e.preventDefault();
    uploadZone.classList.remove('dragover');
    handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => handleFiles(fileInput.files));

async function handleFiles(files) {
    for (const file of files) {
        showStatus(`上傳中: ${file.name}...`, 'loading');
        const form = new FormData();
        form.append('file', file);
        try {
            const res = await fetch(`${API}/api/upload`, { method: 'POST', body: form });
            if (!res.ok) {
                const err = await res.json();
                throw new Error(err.detail || res.statusText);
            }
            const data = await res.json();
            showStatus(`${file.name} — ${data.chunks} 個片段已索引`, 'success');
        } catch (e) {
            showStatus(`${file.name}: ${e.message}`, 'error');
        }
    }
    loadDocuments();
    fileInput.value = '';
}

async function loadDocuments() {
    try {
        const res = await fetch(`${API}/api/documents`);
        const docs = await res.json();
        if (docs.length === 0) {
            docList.innerHTML = '<li class="empty">尚無文件，請上傳 PDF 或圖片</li>';
            return;
        }
        docList.innerHTML = docs.map(d => `
            <li class="doc-item">
                <div class="info">
                    <span class="name">${d.type === 'pdf' ? '📄' : '🖼'} ${d.name}</span>
                    <span class="meta">${d.chunks} chunks · ${new Date(d.added_at).toLocaleString('zh-TW')}</span>
                </div>
                <button onclick="deleteDoc('${d.name}')">刪除</button>
            </li>
        `).join('');
    } catch (e) {
        docList.innerHTML = `<li class="empty">載入失敗: ${e.message}</li>`;
    }
}

async function deleteDoc(name) {
    if (!confirm(`確定刪除 ${name}？`)) return;
    showStatus(`刪除中: ${name}...`, 'loading');
    try {
        const res = await fetch(`${API}/api/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
        if (!res.ok) throw new Error((await res.json()).detail);
        showStatus(`已刪除 ${name}`, 'success');
    } catch (e) {
        showStatus(e.message, 'error');
    }
    loadDocuments();
}

searchBtn.addEventListener('click', doSearch);
searchInput.addEventListener('keydown', e => { if (e.key === 'Enter') doSearch(); });

async function doSearch() {
    const query = searchInput.value.trim();
    if (!query) return;
    searchResults.innerHTML = '<div class="empty">搜索中...</div>';
    try {
        const res = await fetch(`${API}/api/search`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ query, n_results: 5 }),
        });
        const results = await res.json();
        if (results.length === 0) {
            searchResults.innerHTML = '<div class="empty">無搜索結果</div>';
            return;
        }
        searchResults.innerHTML = results.map(r => `
            <div class="result-item">
                <span class="source">${r.source} · 第${r.page}頁</span>
                <span class="score">相似度: ${(r.score * 100).toFixed(1)}%</span>
                <div class="content">${escapeHtml(r.content)}</div>
            </div>
        `).join('');
    } catch (e) {
        searchResults.innerHTML = `<div class="empty">搜索失敗: ${e.message}</div>`;
    }
}

function escapeHtml(text) {
    const d = document.createElement('div');
    d.textContent = text;
    return d.innerHTML;
}

loadDocuments();
