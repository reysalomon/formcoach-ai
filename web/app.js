// FormCoach AI — логика интерфейса. Вся AI-часть работает на сервере (server.py -> formcoach/),
// здесь только: отправить видео, показать результат, хранить настройки.

const $ = (sel) => document.querySelector(sel);
const $$ = (sel) => [...document.querySelectorAll(sel)];
const PHASE_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]; // ВЕРХ, СПУСК, НИЗ, ПОДЪЁМ
const FALLBACK_DEFAULTS = {
  top_angle: 165, bottom_angle: 85, stay_prob: 0.9,
  max_thigh_angle: 35, max_torso_lean: 30, max_heel_lift: 0.2, min_rep_seconds: 0.8,
};

let defaults = { ...FALLBACK_DEFAULTS };
let settings = loadSettings();
let lastResult = null;
let lastDemo = null; // список ошибок демо — чтобы пересчитать при смене настроек

// ---------------- навигация ----------------
function go(name) {
  $$(".screen").forEach((s) => s.classList.toggle("active", s.id === `screen-${name}`));
  $$(".tabbar button").forEach((b) => b.classList.toggle("active", b.dataset.go === name));
  window.scrollTo({ top: 0 });
  if (name === "about") renderAbout();
}
$$("[data-go]").forEach((b) => b.addEventListener("click", () => go(b.dataset.go)));

// ---------------- сообщения об ошибках ----------------
let toastTimer;
function toast(msg) {
  const t = $("#toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => (t.hidden = true), 6000);
}
$("#toast").addEventListener("click", () => ($("#toast").hidden = true));

const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));

// ---------------- настройки (хранятся в браузере) ----------------
function loadSettings() {
  try {
    return JSON.parse(localStorage.getItem("fc-settings")) || {};
  } catch {
    return {};
  }
}
function saveSettings() {
  try {
    localStorage.setItem("fc-settings", JSON.stringify(settings));
  } catch { /* приватный режим — просто не запоминаем */ }
}
const current = () => ({ ...defaults, ...settings });

function buildSettings() {
  const values = current();
  $$(".field").forEach((f) => {
    const { key, min, max, step, unit, label, help } = f.dataset;
    const id = `set-${key}`;
    f.innerHTML = `
      <div class="field-row"><label for="${id}">${label}</label><output id="${id}-out"></output></div>
      ${help ? `<p class="help">${help}</p>` : ""}
      <input type="range" id="${id}" min="${min}" max="${max}" step="${step}" value="${values[key]}">`;
    const input = f.querySelector("input");
    const out = f.querySelector("output");
    const show = () => (out.textContent = `${+(+input.value).toFixed(2)}${unit}`);
    show();
    input.addEventListener("input", show);
    input.addEventListener("change", () => {
      settings[key] = +input.value;
      saveSettings();
      if (lastDemo !== null) runDemo(lastDemo, false);
    });
  });
}
$("#btn-reset").addEventListener("click", () => {
  settings = {};
  saveSettings();
  buildSettings();
  if (lastDemo !== null) runDemo(lastDemo, false);
});

// ---------------- загрузка видео ----------------
function setStep(n) {
  $$("#load-steps li").forEach((li) => {
    const s = +li.dataset.step;
    li.className = s < n ? "done" : s === n ? "now" : "";
  });
}
function showLoading(on) {
  $("#loading").hidden = !on;
  document.body.style.overflow = on ? "hidden" : "";
}

function uploadVideo(file) {
  if (!file) return;
  if (!file.type.startsWith("video/") && !/\.(mp4|mov|webm|avi|mkv|3gp|m4v)$/i.test(file.name)) {
    toast("Это не видео. Выбери файл mp4 или mov.");
    return;
  }
  if (file.size > 100 * 1024 * 1024) {
    toast("Видео больше 100 МБ. Сними покороче — 20–60 секунд хватает.");
    return;
  }
  const form = new FormData();
  // у снятых на iPhone файлов имя иногда без расширения — серверу нужно расширение
  const name = /\.[a-z0-9]+$/i.test(file.name) ? file.name : `video.${file.type.includes("quicktime") ? "mov" : "mp4"}`;
  form.append("video", file, name);
  form.append("settings", JSON.stringify(current()));

  const xhr = new XMLHttpRequest(); // XHR, а не fetch: у него есть прогресс отправки
  xhr.open("POST", "/api/analyze");
  xhr.responseType = "json";
  showLoading(true);
  $("#load-title").textContent = "Загружаю видео…";
  $("#load-bar").classList.remove("indeterminate");
  $("#load-bar").style.width = "0";
  setStep(0);

  xhr.upload.onprogress = (e) => {
    if (e.lengthComputable) $("#load-bar").style.width = `${(e.loaded / e.total) * 100}%`;
  };
  xhr.upload.onload = () => {
    $("#load-title").textContent = "Анализирую технику…";
    $("#load-bar").classList.add("indeterminate");
    setStep(1);
    // поза — самый долгий шаг; через ~15 с показываем, что дошли до HMM и тренера
    setTimeout(() => !$("#loading").hidden && setStep(2), 15000);
  };
  xhr.onload = () => {
    showLoading(false);
    if (xhr.status === 200 && xhr.response) {
      lastDemo = null;
      render(xhr.response, "Твоё видео");
    } else {
      toast(xhr.response?.detail || `Ошибка сервера (${xhr.status}). Попробуй ещё раз.`);
    }
  };
  xhr.onerror = () => {
    showLoading(false);
    toast("Нет связи с сервером. Проверь интернет и попробуй снова.");
  };
  xhr.send(form);
}
["#input-camera", "#input-gallery"].forEach((sel) =>
  $(sel).addEventListener("change", (e) => {
    uploadVideo(e.target.files[0]);
    e.target.value = ""; // чтобы можно было выбрать тот же файл ещё раз
  }),
);

// ---------------- демо ----------------
$$("#demo-chips .chip").forEach((c) =>
  c.addEventListener("click", () => {
    c.classList.toggle("on");
    c.setAttribute("aria-pressed", c.classList.contains("on"));
  }),
);
$("#btn-demo").addEventListener("click", () => {
  runDemo($$("#demo-chips .chip.on").map((c) => c.dataset.err), true);
});

async function runDemo(errors, navigate) {
  const q = new URLSearchParams({ errors: errors.join(","), settings: JSON.stringify(current()) });
  try {
    const res = await fetch(`/api/demo?${q}`);
    const data = await res.json();
    if (!res.ok) throw new Error(data.detail);
    lastDemo = errors;
    render(data, "Демо · виртуальный спортсмен", navigate);
  } catch (e) {
    toast(e.message || "Нет связи с сервером.");
  }
}

// ---------------- отрисовка результата ----------------
const plural = (n, one, few, many) => {
  const m10 = n % 10, m100 = n % 100;
  return m10 === 1 && m100 !== 11 ? one : m10 >= 2 && m10 <= 4 && (m100 < 12 || m100 > 14) ? few : many;
};

function render(data, source, navigate = true) {
  lastResult = data;
  $("#result-empty").hidden = true;
  $("#result").hidden = false;

  const { reps, good, bad, seconds } = data.summary;
  const share = reps ? good / reps : 0;
  $("#score-source").textContent = source;
  $("#score-good").textContent = good;
  $("#score-total").textContent = reps;
  $("#st-reps").textContent = reps;
  $("#st-bad").textContent = bad;
  $("#st-time").textContent = seconds;
  $("#score-headline").textContent =
    reps === 0 ? "Повторений не найдено" : share === 1 ? "Отличная техника!" : share >= 0.6 ? "Хорошо, но есть ошибки" : "Технику стоит поправить";
  $("#score-sub").textContent = reps === 0
    ? "Проверь, что камера сбоку и видно колени и стопы."
    : `${good} ${plural(good, "правильное", "правильных", "правильных")} из ${reps}`;
  const ring = $("#ring-fg");
  ring.style.stroke = share >= 0.6 || reps === 0 ? "var(--accent)" : "var(--warn)";
  ring.style.strokeDashoffset = 327;
  requestAnimationFrame(() => requestAnimationFrame(() => (ring.style.strokeDashoffset = 327 * (1 - share))));

  renderAdvice(data.reps);
  renderReps(data.reps);
  renderChart(data);
  renderFeedback(data.feedback);
  $("#ml-note").textContent = data.has_ml
    ? "«Итог» — правила агента. «ML» — классификатор, обученный на чужих фото: он хорошо видит грубые ошибки спины и пяток, но не глубину."
    : "В демо-режиме нет реальных кадров, поэтому ML-модель не участвует.";
  if (navigate) go("result");
}

function renderAdvice(reps) {
  const counts = {};
  reps.forEach((r) => r.errors.forEach((e) => (counts[e.key] ??= { n: 0, text: e.text }).n++));
  const list = Object.values(counts).sort((a, b) => b.n - a.n);
  const ul = $("#top-advice");
  ul.classList.toggle("ok", list.length === 0);
  ul.innerHTML = list.length
    ? list.map((e) => `<li><span class="badge">${e.n}×</span><span>${esc(e.text)}</span></li>`).join("")
    : reps.length ? "<li>Ошибок не найдено — так держать!</li>" : "<li>Сними видео сбоку, чтобы было видно всё тело.</li>";
}

const SILHOUETTE = `<svg class="silhouette" viewBox="0 0 64 64" aria-hidden="true"><circle cx="30" cy="10" r="6"/>
  <path d="M27 18h7l6 16 12 1v6l-16-1-4 10 6 10h-7l-8-11 3-10-7-9z"/></svg>`;

function renderReps(reps) {
  const box = $("#reps");
  if (!reps.length) {
    box.innerHTML = `<div class="card muted">Повторений не найдено. Проверь, что камера сбоку и видны колени и стопы.</div>`;
    return;
  }
  const rules = current();
  box.innerHTML = reps.map((r) => {
    const keys = r.errors.map((e) => e.key);
    const verdict = r.good
      ? `<span class="verdict ok">✓ хорошо</span>`
      : `<span class="verdict bad">${r.errors.map((e) => esc(e.short)).join(" · ")}</span>`;
    const heel = r.max_heel === null ? "—" : r.max_heel.toFixed(2);
    let ml = "";
    if (r.ml) {
      const p = r.ml.probs[r.ml.label];
      ml = `<div class="ml-line">ML: <b>${esc(r.ml.label_ru)}</b> ${Math.round(p * 100)}%${r.ml.no_feet ? " · без стоп" : ""}
        <div class="prob ${r.ml.label === "good" ? "" : "bad"}"><i style="width:${p * 100}%"></i></div></div>`;
    }
    return `<article class="rep ${r.good ? "" : "bad"}">
      <div class="rep-img">${r.image ? `<img src="${r.image}" alt="Нижняя точка повторения ${r.number}">` : SILHOUETTE}</div>
      <div class="rep-body">
        <div class="rep-head"><span class="rep-num">#${r.number}</span>${verdict}</div>
        <div class="metrics">
          <span class="${keys.includes("shallow") ? "over" : ""}">Бедро к полу <b>${Math.round(r.min_thigh)}°</b></span>
          <span>Колено <b>${Math.round(r.min_knee)}°</b></span>
          <span class="${keys.includes("lean") ? "over" : ""}">Наклон <b>${Math.round(r.max_lean)}°</b></span>
          <span class="${keys.includes("heels") ? "over" : ""}">Пятка <b>${heel}</b></span>
          <span class="${keys.includes("fast") ? "over" : ""}">Время <b>${r.seconds.toFixed(1)} с</b></span>
          <span>Порог глубины <b>${rules.max_thigh_angle}°</b></span>
        </div>
        ${ml}
      </div>
    </article>`;
  }).join("");
}

function renderChart(data) {
  const { t, knee, phase, bottoms } = data.series;
  const W = 340, H = 200, L = 30, R = 8, T = 10, B = 22;
  const tMax = t[t.length - 1] || 1;
  const x = (v) => L + (v / tMax) * (W - L - R);
  const y = (v) => T + (1 - (v - 40) / (185 - 40)) * (H - T - B);

  // подряд идущие точки одной фазы склеиваем в полосу
  let bands = "";
  for (let i = 0; i < phase.length; ) {
    let j = i;
    while (j + 1 < phase.length && phase[j + 1] === phase[i]) j++;
    const x0 = x(t[i]), x1 = x(j + 1 < t.length ? t[j + 1] : tMax);
    bands += `<rect x="${x0}" y="${T}" width="${Math.max(x1 - x0, 0.5)}" height="${H - T - B}" fill="${PHASE_COLORS[phase[i]]}" opacity=".28"/>`;
    i = j + 1;
  }
  let grid = "";
  [60, 100, 140, 180].forEach((v) => {
    grid += `<line class="grid" x1="${L}" x2="${W - R}" y1="${y(v)}" y2="${y(v)}"/><text x="${L - 5}" y="${y(v) + 4}" text-anchor="end">${v}°</text>`;
  });
  const step = tMax > 40 ? 10 : 5;
  for (let s = 0; s <= tMax; s += step) grid += `<text x="${x(s)}" y="${H - 6}" text-anchor="middle">${s} с</text>`;

  const pts = knee.map((k, i) => (k === null ? null : `${x(t[i]).toFixed(1)},${y(k).toFixed(1)}`));
  const path = pts.reduce((d, p, i) => (p === null ? d : d + (i === 0 || pts[i - 1] === null ? "M" : "L") + p), "");
  const badNums = new Set(data.reps.filter((r) => !r.good).map((r) => r.number));
  const dots = bottoms.map((b) => `<circle class="bottom-dot ${badNums.has(b.n) ? "bad" : ""}" cx="${x(b.t)}" cy="${y(b.knee)}" r="5"/>
    <text class="bottom-label" x="${x(b.t)}" y="${y(b.knee) + 18}" text-anchor="middle">${b.n}</text>`).join("");

  $("#chart").innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="График угла колена по времени с фазами движения">
      ${bands}${grid}<path class="knee" d="${path}"/>${dots}
      <line class="cursor" id="cursor" y1="${T}" y2="${H - B}" visibility="hidden"/></svg>
    <div class="chart-tip" id="chart-tip">Проведи пальцем по графику</div>`;
  $("#legend").innerHTML = data.phase_names.map((n, i) => `<span><i style="background:${PHASE_COLORS[i]}"></i>${n}</span>`).join("");

  // подсказка по касанию: время, угол и фаза в этой точке
  const svg = $("#chart svg");
  const onMove = (e) => {
    const box = svg.getBoundingClientRect();
    const sx = ((e.clientX - box.left) / box.width) * W;
    const tv = Math.min(Math.max(((sx - L) / (W - L - R)) * tMax, 0), tMax);
    let i = 0;
    while (i + 1 < t.length && t[i + 1] <= tv) i++;
    const c = $("#cursor");
    c.setAttribute("x1", x(t[i]));
    c.setAttribute("x2", x(t[i]));
    c.setAttribute("visibility", "visible");
    $("#chart-tip").innerHTML = `<b>${t[i].toFixed(1)} с</b> · угол <b>${knee[i] === null ? "—" : Math.round(knee[i]) + "°"}</b> · фаза <b>${data.phase_names[phase[i]]}</b>`;
  };
  svg.addEventListener("pointermove", onMove);
  svg.addEventListener("pointerdown", onMove);
}

function renderFeedback(items) {
  $("#feedback").innerHTML = items.length
    ? items.map((f) => `<li class="${f.level}"><time>${f.t.toFixed(1)} с</time><span class="msg">${esc(f.message)}</span></li>`).join("")
    : `<li><time></time><span class="msg">Пока нечего сказать.</span></li>`;
}

// ---------------- «Как это работает» ----------------
let info = null;
async function renderAbout() {
  try {
    info ??= await (await fetch("/api/info")).json();
  } catch {
    return;
  }
  if (info.models) {
    const best = Math.max(...info.models.map((m) => m.cv_f1));
    $("#models-table").innerHTML = `<div class="table-wrap"><table class="tbl">
      <tr><th>Модель</th><th class="n">F1 новые клипы</th><th class="n">F1 новые люди</th></tr>
      ${info.models.map((m) => `<tr class="${m.cv_f1 === best ? "best" : ""}"><td>${esc(m.name)}</td><td class="n">${m.cv_f1.toFixed(3)}</td><td class="n">${m.test_f1.toFixed(3)}</td></tr>`).join("")}
    </table></div>`;
  }
  let trans = lastResult?.hmm.trans;
  if (!trans) {
    try {
      const q = new URLSearchParams({ errors: "", settings: JSON.stringify(current()) });
      trans = (await (await fetch(`/api/demo?${q}`)).json()).hmm.trans;
    } catch {
      return;
    }
  }
  const names = lastResult?.phase_names ?? ["ВЕРХ", "СПУСК", "НИЗ", "ПОДЪЁМ"];
  $("#hmm-matrix").innerHTML = `<div class="table-wrap"><table class="tbl">
    <tr><th>из \\ в</th>${names.map((n) => `<th class="n">${n}</th>`).join("")}</tr>
    ${trans.map((row, i) => `<tr><td>${names[i]}</td>${row.map((v) => `<td class="n">${v.toFixed(2)}</td>`).join("")}</tr>`).join("")}
  </table></div>`;
}

// ---------------- запуск ----------------
async function init() {
  try {
    const res = await fetch("/api/info");
    info = await res.json();
    defaults = { ...FALLBACK_DEFAULTS, ...info.defaults };
  } catch { /* сервер недоступен — покажем ошибку при первом действии */ }
  buildSettings();
}
init();

// PWA: офлайн-кеш интерфейса и кнопка «Установить» (Android/Chrome)
if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
let installPrompt = null;
window.addEventListener("beforeinstallprompt", (e) => {
  e.preventDefault();
  installPrompt = e;
  $("#btn-install").hidden = false;
});
$("#btn-install").addEventListener("click", async () => {
  if (!installPrompt) return;
  installPrompt.prompt();
  await installPrompt.userChoice;
  installPrompt = null;
  $("#btn-install").hidden = true;
});
