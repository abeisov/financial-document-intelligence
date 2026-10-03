const state = {
  datasetId: "",
  file: null,
  documentIds: [],
  poller: null,
  maxUploadMb: null,
};

const $ = (id) => document.getElementById(id);

function escapeHtml(value) {
  return String(value ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

function toast(message, kind = "info") {
  const el = $("toast");
  el.textContent = message;
  el.dataset.kind = kind;
  el.classList.add("show", kind);
  window.clearTimeout(el._timer);
  el._timer = window.setTimeout(() => el.classList.remove("show", kind), 4200);
}

async function api(path, options = {}) {
  const response = await fetch(path, options);
  let body = null;
  try {
    body = await response.json();
  } catch {
    body = null;
  }
  if (!response.ok) {
    const detail = body?.detail || `HTTP ${response.status}`;
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
  return body;
}

function setBusy(button, busy, label) {
  if (!button) return;
  if (busy) {
    button.dataset.label = button.textContent;
    button.textContent = label;
  } else if (button.dataset.label) {
    button.textContent = button.dataset.label;
  }
  button.disabled = busy;
}

async function loadHealth() {
  try {
    const body = await api("/api/health");
    const limitMb = body.limits?.max_upload_mb;
    if (limitMb) {
      state.maxUploadMb = limitMb;
      const note = $("upload-limit-note");
      if (note) note.textContent = `До ${limitMb} МБ, один файл за раз`;
    }
    const services = body.services || {};
    const labels = { ragflow: "RAGFlow", qwen: "Qwen", embedder: "Embedder", reranker: "Reranker" };
    const badges = Object.entries(labels).map(([key, label]) => {
      const ready = Boolean(services[key]);
      return `<span class="service-badge ${ready ? "ok" : "fail"}"><i></i>${label}</span>`;
    });
    $("service-badges").innerHTML = badges.join("");
    const readyCount = Object.values(services).filter(Boolean).length;
    $("service-summary").textContent = `${readyCount} из ${Object.keys(services).length} настроены`;
    if (!$("dataset-name").value) {
      $("dataset-name").value = body.defaults?.dataset_name || "financial-documents";
    }
  } catch (error) {
    $("service-summary").textContent = "Не удалось проверить";
    toast(error.message, "error");
  }
}

function renderDatasets(datasets, preferredId = "") {
  const select = $("dataset-select");
  if (!datasets.length) {
    select.innerHTML = '<option value="">Dataset пока нет</option>';
    state.datasetId = "";
    updateUploadButton();
    return;
  }
  select.innerHTML = datasets.map((dataset) => {
    const count = Number(dataset.document_count || 0);
    return `<option value="${escapeHtml(dataset.id)}">${escapeHtml(dataset.name)} · ${count} док.</option>`;
  }).join("");
  const requested = preferredId || state.datasetId;
  if (requested && datasets.some((dataset) => dataset.id === requested)) {
    select.value = requested;
  }
  state.datasetId = select.value;
  updateUploadButton();
}

async function loadDatasets(preferredId = "") {
  const refresh = $("refresh-datasets");
  setBusy(refresh, true, "…");
  try {
    const body = await api("/api/datasets");
    renderDatasets(body.datasets || [], preferredId);
  } catch (error) {
    $("dataset-select").innerHTML = '<option value="">RAGFlow недоступен</option>';
    state.datasetId = "";
    updateUploadButton();
    toast(error.message, "error");
  } finally {
    setBusy(refresh, false);
  }
}

function selectFile(file) {
  if (file && state.maxUploadMb && file.size > state.maxUploadMb * 1024 * 1024) {
    toast(`Файл больше ${state.maxUploadMb} МБ — такой размер не примет сервер`, "error");
    file = null;
  }
  state.file = file || null;
  const zone = $("drop-zone");
  zone.classList.toggle("has-file", Boolean(file));
  if (file) {
    zone.querySelector("strong").textContent = file.name;
    zone.querySelector("small").textContent = `${(file.size / 1024 / 1024).toFixed(2)} МБ`;
  } else {
    zone.querySelector("strong").textContent = "Перетащите файл сюда";
    zone.querySelector("small").textContent = "или нажмите для выбора";
  }
  updateUploadButton();
}

function updateUploadButton() {
  $("upload-button").disabled = !(state.datasetId && state.file);
}

async function createDataset() {
  const name = $("dataset-name").value.trim();
  if (name.length < 2) {
    toast("Введите название Dataset", "error");
    return;
  }
  const button = $("create-dataset");
  setBusy(button, true, "Создаём…");
  try {
    const body = await api("/api/datasets", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    await loadDatasets(body.dataset.id);
    toast(body.created ? "Dataset создан" : "Такой Dataset уже существует", "success");
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

function showDocument(document) {
  $("document-card").classList.remove("hidden");
  $("document-name").textContent = document.name || state.file?.name || "Документ";
  $("document-status").textContent = "Файл загружен, RAGFlow начинает разбор";
  $("document-progress").textContent = "0%";
}

async function uploadDocument() {
  if (!state.datasetId || !state.file) return;
  const button = $("upload-button");
  setBusy(button, true, "Загружаем…");
  const form = new FormData();
  form.append("dataset_id", state.datasetId);
  form.append("file", state.file);
  try {
    const body = await api("/api/documents", { method: "POST", body: form });
    const document = body.documents?.[0];
    if (!document?.id) throw new Error("RAGFlow не вернул ID документа");
    state.documentIds = body.documents.map((item) => item.id);
    showDocument(document);
    $("empty-state").classList.add("hidden");
    $("chat-state").classList.remove("hidden");
    toast("Документ загружен. Ждём индексацию…", "success");
    pollDocument(document.id);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
    updateUploadButton();
  }
}

function parseRun(raw) {
  const value = String(raw ?? "").toUpperCase();
  if (["3", "DONE", "SUCCESS", "COMPLETED"].includes(value)) return "done";
  if (["4", "FAIL", "FAILED", "ERROR"].includes(value)) return "failed";
  if (["2", "CANCEL", "CANCELLED"].includes(value)) return "cancelled";
  return "working";
}

async function pollDocument(documentId) {
  window.clearTimeout(state.poller);
  try {
    const body = await api(`/api/documents/${encodeURIComponent(state.datasetId)}/${encodeURIComponent(documentId)}`);
    const document = body.document;
    const progressValue = Number(document.progress || 0);
    const percent = Math.max(0, Math.min(100, progressValue <= 1 ? progressValue * 100 : progressValue));
    $("document-progress").textContent = `${Math.round(percent)}%`;
    const run = parseRun(document.run);
    if (run === "done" || percent >= 100) {
      $("document-status").textContent = `Готово · ${document.chunk_count || 0} фрагментов`;
      $("document-card").classList.add("complete");
      toast("Индексация завершена — можно задавать вопросы", "success");
      return;
    }
    if (run === "failed" || run === "cancelled") {
      const needsEmbedding = String(document.message || "").includes("No default embedding model");
      const status = needsEmbedding
        ? "В RAGFlow не выбрана embedding-модель"
        : (run === "failed" ? "Ошибка обработки" : "Обработка отменена");
      $("document-status").textContent = status;
      toast(needsEmbedding
        ? "Откройте RAGFlow → Провайдеры моделей и выберите Embedder по умолчанию"
        : (document.message || "RAGFlow не смог обработать документ"), "error");
      return;
    }
    $("document-status").textContent = `RAGFlow обрабатывает · ${Math.round(percent)}%`;
    state.poller = window.setTimeout(() => pollDocument(documentId), 2500);
  } catch (error) {
    $("document-status").textContent = "Не удалось получить статус";
    toast(error.message, "error");
  }
}

function renderList(element, items, emptyText) {
  const rows = Array.isArray(items) ? items : [];
  element.innerHTML = rows.length
    ? rows.map((item) => `<li>${escapeHtml(item)}</li>`).join("")
    : `<li class="muted">${escapeHtml(emptyText)}</li>`;
}

function renderAnswer(body) {
  $("answer-card").classList.remove("hidden");
  $("answer-question").textContent = body.question;
  $("answer-text").textContent = body.answer;
  renderList($("facts-list"), body.observed_facts, "Факты не выделены");
  renderList($("interpretation-list"), body.interpretation, "Интерпретация не требуется");

  const grounding = body.grounding || {};
  const grounded = grounding.citations_valid && grounding.quotes_found && grounding.numbers_grounded;
  const badge = $("answer-grounding");
  badge.textContent = grounded ? "Источники проверены" : "Нужна проверка";
  badge.className = `grounding-badge ${grounded ? "ok" : "warn"}`;

  const sourceById = new Map((body.sources || []).map((source) => [source.source_id, source]));
  const citations = body.citations || [];
  $("sources-list").innerHTML = citations.length ? citations.map((citation, index) => {
    const source = sourceById.get(citation.source_id) || {};
    return `<details class="source-item" ${index === 0 ? "open" : ""}>
      <summary><span>${index + 1}</span><strong>${escapeHtml(citation.document || source.document || "Документ")}</strong><em>стр. ${escapeHtml(citation.page || source.page || "—")}</em></summary>
      <blockquote>${escapeHtml(citation.quote || source.excerpt || "")}</blockquote>
    </details>`;
  }).join("") : '<p class="muted">Подтверждающих фрагментов не найдено.</p>';
  $("answer-card").scrollIntoView({ behavior: "smooth", block: "start" });
}

async function askQuestion(question) {
  const clean = question.trim();
  if (clean.length < 3) {
    toast("Введите вопрос по документу", "error");
    return;
  }
  if (!state.datasetId) {
    toast("Сначала выберите Dataset", "error");
    return;
  }
  const button = $("ask-button");
  setBusy(button, true, "Анализируем…");
  try {
    const body = await api("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        dataset_id: state.datasetId,
        document_ids: state.documentIds,
        question: clean,
      }),
    });
    renderAnswer(body);
  } catch (error) {
    toast(error.message, "error");
  } finally {
    setBusy(button, false);
  }
}

$("dataset-select").addEventListener("change", (event) => {
  state.datasetId = event.target.value;
  state.documentIds = [];
  updateUploadButton();
});
$("refresh-datasets").addEventListener("click", () => loadDatasets());
$("create-dataset").addEventListener("click", createDataset);
$("file-input").addEventListener("change", (event) => selectFile(event.target.files?.[0]));
$("upload-button").addEventListener("click", uploadDocument);
$("question-form").addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion($("question-input").value);
});
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => {
    $("question-input").value = button.dataset.question;
    askQuestion(button.dataset.question);
  });
});

const dropZone = $("drop-zone");
["dragenter", "dragover"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((name) => dropZone.addEventListener(name, (event) => {
  event.preventDefault();
  dropZone.classList.remove("dragging");
}));
dropZone.addEventListener("drop", (event) => selectFile(event.dataTransfer.files?.[0]));

loadHealth();
loadDatasets();
