// VESE HUD front-end: upload -> process -> live overlay + search.
"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  videoId: null,
  det: null,        // detections.json payload
  frameTimes: [],   // sorted t[] for fast lookup
  summary: null,
};

// ---- color per class (stable hash -> hue) ----
const colorCache = {};
function classColor(name) {
  if (colorCache[name]) return colorCache[name];
  let h = 0;
  for (let i = 0; i < name.length; i++) h = (h * 31 + name.charCodeAt(i)) % 360;
  const c = `hsl(${h}, 90%, 60%)`;
  colorCache[name] = c;
  return c;
}

function setSys(txt) { $("sysline").textContent = txt; }
function fmt(sec) {
  sec = Math.max(0, sec);
  const m = Math.floor(sec / 60), s = Math.floor(sec % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

// ---------- upload / process ----------
const dropzone = $("dropzone"), fileInput = $("fileInput");
dropzone.addEventListener("click", () => fileInput.click());
["dragover", "dragenter"].forEach((e) =>
  dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.add("drag"); }));
["dragleave", "drop"].forEach((e) =>
  dropzone.addEventListener(e, (ev) => { ev.preventDefault(); dropzone.classList.remove("drag"); }));
dropzone.addEventListener("drop", (ev) => {
  const f = ev.dataTransfer.files[0];
  if (f) startPipeline(f);
});
fileInput.addEventListener("change", () => {
  if (fileInput.files[0]) startPipeline(fileInput.files[0]);
});

async function startPipeline(file) {
  setSys("UPLOADING FEED…");
  dropzone.classList.add("hidden");
  $("viewer").classList.remove("hidden");
  $("processing").classList.remove("hidden");
  $("procPct").textContent = "0%";

  const fd = new FormData();
  fd.append("file", file);
  const up = await fetch("/api/upload", { method: "POST", body: fd }).then((r) => r.json());
  state.videoId = up.video_id;

  await fetch(`/api/process/${state.videoId}`, { method: "POST" });
  setSys("ANALYZING FEED…");
  pollStatus();
}

async function pollStatus() {
  const s = await fetch(`/api/status/${state.videoId}`).then((r) => r.json());
  if (s.state === "processing") {
    const pct = s.total ? Math.floor((s.done / s.total) * 100) : 0;
    $("procPct").textContent = `${pct}%`;
    setTimeout(pollStatus, 500);
  } else if (s.state === "done") {
    $("procPct").textContent = "100%";
    await loadResults();
  } else if (s.state === "error") {
    setSys("ERROR — SEE CONSOLE");
    $("procPct").textContent = "ERR";
    console.error("Processing failed:", s.error);
  } else {
    setTimeout(pollStatus, 500);
  }
}

// ---------- load processed data ----------
async function loadResults() {
  state.summary = await fetch(`/api/summary/${state.videoId}`).then((r) => r.json());
  state.det = await fetch(`/api/detections/${state.videoId}`).then((r) => r.json());
  state.frameTimes = state.det.frames.map((f) => f.t);

  $("processing").classList.add("hidden");
  setSys(`FEED LOCKED — ${state.summary.num_intervals} EVENTS INDEXED`);
  renderClasses();

  const video = $("video");
  video.src = `/api/video/${state.videoId}`;
  video.load();
  video.play().catch(() => {});
  requestAnimationFrame(drawLoop);
  drawTimeline();
  window.addEventListener("resize", drawTimeline);
}

function renderClasses() {
  const box = $("classList");
  const classes = state.summary.classes;
  const keys = Object.keys(classes);
  if (!keys.length) { box.innerHTML = `<div class="empty">No objects detected.</div>`; return; }
  box.innerHTML = keys.map((k) => `
    <div class="class-row">
      <span class="class-swatch" style="background:${classColor(k)};color:${classColor(k)}"></span>
      <span class="class-name">${k}</span>
      <span class="class-count">${classes[k].count}</span>
    </div>`).join("");
}

// ---------- live overlay ----------
function nearestFrame(t) {
  const ts = state.frameTimes;
  if (!ts.length) return null;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) {
    const mid = (lo + hi) >> 1;
    if (ts[mid] < t) lo = mid + 1; else hi = mid;
  }
  // pick closer of lo and lo-1
  if (lo > 0 && Math.abs(ts[lo - 1] - t) < Math.abs(ts[lo] - t)) lo -= 1;
  return state.det.frames[lo];
}

function drawLoop() {
  const video = $("video"), cv = $("overlay");
  if (video.videoWidth) {
    const rect = video.getBoundingClientRect();
    if (cv.width !== rect.width || cv.height !== rect.height) {
      cv.width = rect.width; cv.height = rect.height;
      cv.style.left = video.offsetLeft + "px";
      cv.style.top = video.offsetTop + "px";
    }
    const ctx = cv.getContext("2d");
    ctx.clearRect(0, 0, cv.width, cv.height);
    const sx = cv.width / state.det.width, sy = cv.height / state.det.height;
    const frame = nearestFrame(video.currentTime);
    if (frame) {
      $("liveCount").textContent = frame.dets.length;
      for (const d of frame.dets) drawBox(ctx, d, sx, sy);
    }
  }
  requestAnimationFrame(drawLoop);
}

function drawBox(ctx, d, sx, sy) {
  const [x1, y1, x2, y2] = d.box;
  const X = x1 * sx, Y = y1 * sy, W = (x2 - x1) * sx, H = (y2 - y1) * sy;
  const col = classColor(d.cls);
  const len = Math.min(18, W / 3, H / 3);

  ctx.strokeStyle = col; ctx.lineWidth = 2;
  ctx.shadowColor = col; ctx.shadowBlur = 8;
  // corner brackets
  const corners = [
    [X, Y, 1, 1], [X + W, Y, -1, 1], [X, Y + H, 1, -1], [X + W, Y + H, -1, -1],
  ];
  for (const [cx, cy, dx, dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(cx, cy + dy * len); ctx.lineTo(cx, cy); ctx.lineTo(cx + dx * len, cy);
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  // faint fill
  ctx.fillStyle = col.replace("60%)", "60%,0.08)").replace("hsl", "hsla");
  ctx.fillRect(X, Y, W, H);
  // label
  const label = `${d.cls.toUpperCase()} #${d.id}`;
  ctx.font = "11px 'Share Tech Mono', monospace";
  const tw = ctx.measureText(label).width + 10;
  ctx.fillStyle = col;
  ctx.fillRect(X, Y - 16, tw, 15);
  ctx.fillStyle = "#04121a";
  ctx.fillText(label, X + 5, Y - 5);
}

// ---------- timeline ----------
function drawTimeline() {
  const cv = $("timeline");
  if (!state.summary) return;
  cv.width = cv.clientWidth; cv.height = 54;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0, 0, cv.width, cv.height);
  const dur = state.summary.duration || 1;
  const ivs = state.summary.intervals;
  // lane per class
  const classes = Object.keys(state.summary.classes);
  const laneH = Math.max(4, Math.min(10, (cv.height - 8) / Math.max(1, classes.length)));
  ivs.forEach((iv) => {
    const lane = classes.indexOf(iv.cls);
    const x = (iv.start_sec / dur) * cv.width;
    const w = Math.max(2, ((iv.end_sec - iv.start_sec) / dur) * cv.width);
    ctx.fillStyle = classColor(iv.cls);
    ctx.globalAlpha = 0.8;
    ctx.fillRect(x, 4 + lane * laneH, w, laneH - 1);
  });
  ctx.globalAlpha = 1;
  // playhead
  const video = $("video");
  if (video.duration) {
    const px = (video.currentTime / dur) * cv.width;
    ctx.strokeStyle = "#fff"; ctx.beginPath();
    ctx.moveTo(px, 0); ctx.lineTo(px, cv.height); ctx.stroke();
  }
  requestAnimationFrame(drawTimeline);
}

// click timeline to seek
$("timeline").addEventListener("click", (e) => {
  if (!state.summary) return;
  const rect = e.target.getBoundingClientRect();
  const frac = (e.clientX - rect.left) / rect.width;
  $("video").currentTime = frac * state.summary.duration;
});

// ---------- search ----------
$("searchForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  const q = $("searchInput").value.trim();
  if (!q || !state.videoId) return;
  const res = await fetch(`/api/search/${state.videoId}?q=${encodeURIComponent(q)}`)
    .then((r) => r.json());
  $("searchMeta").textContent = `${res.count} MATCH(ES) FOR "${q.toUpperCase()}"`;
  renderResults(res.matches);
});

function renderResults(matches) {
  const box = $("results");
  if (!matches.length) { box.innerHTML = `<div class="empty">No matches.</div>`; return; }
  box.innerHTML = matches.map((m, i) => `
    <div class="result" data-t="${m.start_sec}">
      <div class="r-top">
        <span class="r-cls">${m.cls}</span>
        <span class="r-id">ID ${m.track_id}</span>
      </div>
      <div class="r-time">${m.start_time.slice(0, 8)} → ${m.end_time.slice(0, 8)}
        <span style="color:var(--cy-dim)">(${m.duration_sec}s)</span></div>
    </div>`).join("");
  [...box.querySelectorAll(".result")].forEach((el) => {
    el.addEventListener("click", () => {
      const video = $("video");
      video.currentTime = parseFloat(el.dataset.t);
      video.play().catch(() => {});
    });
  });
}
