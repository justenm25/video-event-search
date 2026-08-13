// ============================================================
//  Video Event Search — front-end controller
// ============================================================
"use strict";
const $ = (id) => document.getElementById(id);
const state = { videoId: null, filename: null, det: null, frameTimes: [], summary: null,
  dragging: false, traj: {}, showTraj: false, showSkel: true, showLine: true };

// COCO-17 skeleton connections (pairs of keypoint indices)
const SKELETON = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];

// curated categorical palette (distinguishable on dark)
const PALETTE = ["#8b7fe8","#4a9eff","#4dd6a0","#f0a35e","#e06c9a","#a99cff",
  "#7bd85b","#e06c75","#5cc9d6","#f5d15e","#f08a5e","#38b6a0","#c084fc","#68a8ff"];
const colorCache = {};
function classColor(name) {
  if (colorCache[name]) return colorCache[name];
  let h = 0; for (let i = 0; i < name.length; i++) h = (h * 131 + name.charCodeAt(i)) >>> 0;
  return (colorCache[name] = PALETTE[h % PALETTE.length]);
}
const fmtClock = (s) => {
  s = Math.max(0, s); const m = Math.floor(s / 60), sec = Math.floor(s % 60);
  return `${String(m).padStart(2,"0")}:${String(sec).padStart(2,"0")}`;
};
function setSys(text, cls) {
  $("sysText").textContent = text;
  $("sysline").className = "sysline" + (cls ? " " + cls : "");
}

// ---------- panel tabs ----------
document.querySelectorAll(".ptab[data-tab]").forEach((tab) => {
  tab.addEventListener("click", () => {
    const panel = tab.closest(".panel");
    panel.querySelectorAll(".ptab[data-tab]").forEach((t) => t.classList.remove("active"));
    tab.classList.add("active");
    const target = tab.dataset.tab;
    panel.querySelectorAll(".pane").forEach((p) => p.classList.toggle("active", p.dataset.pane === target));
  });
});

// clock
setInterval(() => { $("cClock").textContent = new Date().toLocaleTimeString(); }, 1000);

// ---------- upload / process ----------
const dropzone = $("dropzone"), fileInput = $("fileInput");
dropzone.addEventListener("click", () => fileInput.click());
["dragover","dragenter"].forEach((e) => dropzone.addEventListener(e,(ev)=>{ev.preventDefault();dropzone.classList.add("drag");}));
["dragleave","drop"].forEach((e) => dropzone.addEventListener(e,(ev)=>{ev.preventDefault();dropzone.classList.remove("drag");}));
dropzone.addEventListener("drop",(ev)=>{ const f=ev.dataTransfer.files[0]; if(f) startPipeline(f); });
fileInput.addEventListener("change",()=>{ if(fileInput.files[0]) startPipeline(fileInput.files[0]); });

// top-bar actions
$("btnImport").addEventListener("click", () => fileInput.click());
$("btnAddPerson").addEventListener("click", () => openIdModal());

const PROC_MSGS = ["Reading video","Detecting objects","Tracking motion",
  "Building timeline","Finalizing"];

async function startPipeline(file) {
  state.filename = file.name;
  setSys("Importing…", "");
  dropzone.classList.add("hidden");
  $("viewer").classList.remove("hidden");
  $("processing").classList.remove("hidden");
  $("procPct").textContent = "0%";
  $("procBar").style.width = "0%";

  const fd = new FormData(); fd.append("file", file);
  const up = await fetch("/api/upload", { method:"POST", body:fd }).then((r)=>r.json());
  state.videoId = up.video_id;
  await fetch(`/api/process/${state.videoId}`, { method:"POST" });
  setSys("Analyzing…", "");
  pollStatus();
}

async function pollStatus() {
  const s = await fetch(`/api/status/${state.videoId}`).then((r)=>r.json());
  if (s.state === "processing") {
    const pct = s.total ? Math.floor((s.done/s.total)*100) : 0;
    $("procPct").textContent = `${pct}%`;
    $("procBar").style.width = `${pct}%`;
    $("procSub").textContent = PROC_MSGS[Math.min(PROC_MSGS.length-1, Math.floor(pct/22))];
    setTimeout(pollStatus, 400);
  } else if (s.state === "done") {
    $("procPct").textContent = "100%"; $("procBar").style.width = "100%"; loadResults();
  } else if (s.state === "error") {
    setSys("Processing failed — see console", "err");
    $("procPct").textContent = "error"; console.error("Processing failed:", s.error);
  } else setTimeout(pollStatus, 400);
}

// ---------- load processed data ----------
async function loadResults() {
  state.summary = await fetch(`/api/summary/${state.videoId}`).then((r)=>r.json());
  state.det = await fetch(`/api/detections/${state.videoId}`).then((r)=>r.json());
  state.frameTimes = state.det.frames.map((f)=>f.t);

  $("processing").classList.add("hidden");
  setSys(`Ready · ${state.summary.num_intervals} events`, "live");
  $("roEvents").textContent = state.summary.num_intervals;
  renderMedia(); renderClasses(); renderChips();
  buildTrajectories(); renderAnalytics();

  const video = $("video");
  video.src = `/api/video/${state.videoId}`;
  video.load();
  video.addEventListener("loadedmetadata", () => { $("tDur").textContent = fmtClock(video.duration); }, { once:true });
  video.play().catch(()=>{});
  requestAnimationFrame(drawLoop);
  requestAnimationFrame(drawTimeline);
}

function renderMedia() {
  const box = $("mediaBin"); if (!box) return;
  const name = state.filename || "clip.mp4";
  const nClasses = Object.keys(state.summary.classes).length;
  box.innerHTML = `<div class="clip-item">
    <div class="clip-thumb"><svg viewBox="0 0 24 24"><path d="M8 5v14l11-7z"/></svg></div>
    <div class="clip-meta">
      <div class="clip-name">${name}</div>
      <div class="clip-sub">${fmtClock(state.summary.duration)} · ${nClasses} object type${nClasses===1?"":"s"}</div>
    </div>
  </div>`;
}

function renderClasses() {
  const box = $("classList"), classes = state.summary.classes, keys = Object.keys(classes);
  if (!keys.length) { box.innerHTML = `<div class="empty">No objects detected.</div>`; return; }
  const max = Math.max(...keys.map((k)=>classes[k].count));
  box.innerHTML = keys.map((k)=>{
    const c = classColor(k), pct = Math.round((classes[k].count/max)*100);
    return `<div class="class-row">
      <div class="cr-top">
        <span class="class-swatch" style="background:${c}"></span>
        <span class="class-name">${k}</span>
        <span class="class-count">×${classes[k].count}</span>
      </div>
      <div class="cr-bar"><i style="width:${pct}%;background:${c}"></i></div>
    </div>`;
  }).join("");
}

function renderChips() {
  const box = $("chips"), keys = Object.keys(state.summary.classes);
  box.innerHTML = keys.map((k)=>`<span class="qchip" data-q="${k}">
    <span class="dot" style="background:${classColor(k)}"></span>${k}</span>`).join("");
  [...box.querySelectorAll(".qchip")].forEach((el)=>el.addEventListener("click",()=>{
    $("searchInput").value = el.dataset.q; runSearch(el.dataset.q);
  }));
}

// ---------- motion analytics ----------
function buildTrajectories() {
  const traj = {};
  for (const f of state.det.frames) {
    for (const d of f.dets) {
      const tr = traj[d.id] || (traj[d.id] = { color: classColor(d.cls), pts: [] });
      const [x1,y1,x2,y2] = d.box;
      tr.pts.push([f.t, (x1+x2)/2, (y1+y2)/2]);
      tr.color = classColor(d.cls);
    }
  }
  state.traj = traj;
}
function renderAnalytics() {
  const a = state.summary.analytics;
  if (a && a.crossings) { $("crL").textContent = a.crossings.l2r; $("crR").textContent = a.crossings.r2l; }
  const img = $("depthImg");
  img.onerror = () => { $("depthWrap").style.display = "none"; };
  img.onload = () => { $("depthWrap").style.display = ""; };
  img.src = `/api/depth/${state.videoId}?t=${Date.now()}`;
}
function drawTrajectories(ctx, sx, sy, tNow) {
  ctx.save(); ctx.lineWidth = 2; ctx.globalAlpha = 0.8;
  for (const tid in state.traj) {
    const tr = state.traj[tid], pts = tr.pts.filter((p)=>p[0] <= tNow);
    if (pts.length < 2) continue;
    ctx.strokeStyle = tr.color;
    ctx.beginPath(); ctx.moveTo(pts[0][1]*sx, pts[0][2]*sy);
    for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i][1]*sx, pts[i][2]*sy);
    ctx.stroke();
  }
  ctx.restore();
}
function drawCountLine(ctx, cv) {
  const a = state.summary && state.summary.analytics;
  if (!a || !a.line) return;
  const x = a.line.pos_frac * cv.width;
  ctx.save(); ctx.strokeStyle = "#f0a35e"; ctx.globalAlpha = 0.7; ctx.lineWidth = 1.5;
  ctx.setLineDash([7,6]); ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,cv.height); ctx.stroke();
  ctx.restore();
}
function drawSkeleton(ctx, kpts, sx, sy, col) {
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 1.75; ctx.fillStyle = "#fff";
  for (const [a,b] of SKELETON) {
    const p = kpts[a], q = kpts[b];
    if (p && q && p[2] > 0.3 && q[2] > 0.3) {
      ctx.beginPath(); ctx.moveTo(p[0]*sx, p[1]*sy); ctx.lineTo(q[0]*sx, q[1]*sy); ctx.stroke();
    }
  }
  for (const p of kpts) if (p[2] > 0.3) { ctx.beginPath(); ctx.arc(p[0]*sx, p[1]*sy, 2.25, 0, 7); ctx.fill(); }
  ctx.restore();
}

// ---------- live detection overlay ----------
function nearestFrame(t) {
  const ts = state.frameTimes; if (!ts.length) return null;
  let lo = 0, hi = ts.length - 1;
  while (lo < hi) { const m=(lo+hi)>>1; if (ts[m] < t) lo=m+1; else hi=m; }
  if (lo > 0 && Math.abs(ts[lo-1]-t) < Math.abs(ts[lo]-t)) lo--;
  return state.det.frames[lo];
}
function drawLoop() {
  const video = $("video"), cv = $("overlay");
  if (video.videoWidth && state.det) {
    const rect = video.getBoundingClientRect();
    if (cv.width !== Math.round(rect.width) || cv.height !== Math.round(rect.height)) {
      cv.width = Math.round(rect.width); cv.height = Math.round(rect.height);
      cv.style.left = video.offsetLeft + "px"; cv.style.top = video.offsetTop + "px";
    }
    const ctx = cv.getContext("2d");
    ctx.clearRect(0,0,cv.width,cv.height);
    const sx = cv.width/state.det.width, sy = cv.height/state.det.height;
    const frame = nearestFrame(video.currentTime);
    if (state.showTraj) drawTrajectories(ctx, sx, sy, video.currentTime);
    if (state.showLine) drawCountLine(ctx, cv);
    if (frame) {
      $("liveCount").textContent = frame.dets.length;
      $("roFrame").textContent = Math.round(video.currentTime * (state.det.fps||30));
      for (const d of frame.dets) drawBox(ctx, d, sx, sy);
    }
    $("roTime").textContent = fmtClock(video.currentTime);
  }
  requestAnimationFrame(drawLoop);
}
function drawBox(ctx, d, sx, sy) {
  const [x1,y1,x2,y2] = d.box;
  const X=x1*sx, Y=y1*sy, W=(x2-x1)*sx, H=(y2-y1)*sy;
  const col = classColor(d.cls);

  ctx.save();
  // segmentation mask outline (if present)
  if (d.mask && d.mask.length > 2) {
    ctx.beginPath(); ctx.moveTo(d.mask[0][0]*sx, d.mask[0][1]*sy);
    for (let i = 1; i < d.mask.length; i++) ctx.lineTo(d.mask[i][0]*sx, d.mask[i][1]*sy);
    ctx.closePath();
    ctx.fillStyle = col + "22"; ctx.fill();
  }
  // clean tracker box
  ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.strokeRect(X, Y, W, H);

  // skeleton (pose)
  if (state.showSkel && d.kpts) drawSkeleton(ctx, d.kpts, sx, sy, col);

  // label tab
  const label = `${d.cls} ${d.id}`
    + (d.attrs ? ` · ${d.attrs.gender}${d.attrs.age}` : "")
    + (d.action ? ` · ${d.action}` : "");
  ctx.font = "600 11px -apple-system, Segoe UI, sans-serif";
  const lw = ctx.measureText(label).width, padX = 6, chipH = 16, chipW = lw + padX*2;
  let ly = Y - chipH; if (ly < 0) ly = Y;
  ctx.fillStyle = col; ctx.fillRect(X, ly, chipW, chipH);
  ctx.fillStyle = "#151515"; ctx.textBaseline = "middle";
  ctx.fillText(label, X + padX, ly + chipH/2 + 0.5);
  ctx.restore();
}

// ---------- timeline ----------
function roundRect(ctx, x, y, w, h, r) {
  r = Math.min(r, h/2, w/2);
  ctx.beginPath();
  ctx.moveTo(x+r, y); ctx.arcTo(x+w, y, x+w, y+h, r); ctx.arcTo(x+w, y+h, x, y+h, r);
  ctx.arcTo(x, y+h, x, y, r); ctx.arcTo(x, y, x+w, y, r); ctx.closePath();
}
function niceStep(dur) {
  const target = dur / 8, steps = [1,2,5,10,15,30,60,120,300,600,900,1800];
  for (const s of steps) if (s >= target) return s;
  return 3600;
}
const TL_GUTTER = 96, TL_RULER = 24;
function drawTimeline() {
  const cv = $("timeline");
  if (!state.summary) { requestAnimationFrame(drawTimeline); return; }
  const dpr = window.devicePixelRatio || 1;
  const w = cv.clientWidth, h = cv.clientHeight;
  if (cv.width !== Math.round(w*dpr) || cv.height !== Math.round(h*dpr)) {
    cv.width = Math.round(w*dpr); cv.height = Math.round(h*dpr);
  }
  const ctx = cv.getContext("2d");
  ctx.setTransform(dpr,0,0,dpr,0,0);
  ctx.clearRect(0,0,w,h);

  const dur = state.summary.duration || 1;
  const classes = Object.keys(state.summary.classes);
  const lanes = Math.max(1, classes.length);
  const laneAreaH = h - TL_RULER;
  const laneH = laneAreaH / lanes;
  const trackW = w - TL_GUTTER;
  const xForT = (t) => TL_GUTTER + (t/dur)*trackW;

  // backgrounds
  ctx.fillStyle = "#1b1b1b"; ctx.fillRect(0,0,w,h);
  ctx.fillStyle = "#232323"; ctx.fillRect(0,0,TL_GUTTER,h);
  ctx.fillStyle = "#262626"; ctx.fillRect(TL_GUTTER,0,trackW,TL_RULER);

  // ruler ticks + timecodes
  const step = niceStep(dur);
  ctx.font = "10px " + "ui-monospace, monospace"; ctx.textBaseline = "middle";
  for (let t = 0; t <= dur + 0.001; t += step) {
    const x = xForT(t);
    ctx.strokeStyle = "#3a3a3a"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, TL_RULER); ctx.stroke();
    ctx.strokeStyle = "rgba(255,255,255,0.04)";
    ctx.beginPath(); ctx.moveTo(x, TL_RULER); ctx.lineTo(x, h); ctx.stroke();
    ctx.fillStyle = "#8a8a8a"; ctx.fillText(fmtClock(t), x + 4, TL_RULER/2);
  }

  // lanes + labels + intervals
  classes.forEach((cls, i) => {
    const y = TL_RULER + i*laneH, c = classColor(cls);
    ctx.strokeStyle = "#2a2a2a"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(w, y); ctx.stroke();
    // track header
    ctx.fillStyle = c; ctx.fillRect(8, y + laneH/2 - 5, 3, 10);
    ctx.fillStyle = "#c8c8c8"; ctx.font = "11px -apple-system, Segoe UI, sans-serif";
    const name = cls.length > 13 ? cls.slice(0,12) + "…" : cls;
    ctx.fillText(name, 18, y + laneH/2);
  });

  state.summary.intervals.forEach((iv) => {
    const lane = classes.indexOf(iv.cls); if (lane < 0) return;
    const y = TL_RULER + lane*laneH, c = classColor(iv.cls);
    const x = xForT(iv.start_sec), x2 = xForT(iv.end_sec);
    const bw = Math.max(3, x2 - x), bh = Math.max(7, laneH - 12);
    roundRect(ctx, x, y + (laneH-bh)/2, bw, bh, 2);
    ctx.fillStyle = c; ctx.globalAlpha = 0.92; ctx.fill(); ctx.globalAlpha = 1;
    ctx.fillStyle = "rgba(255,255,255,0.22)"; ctx.fillRect(x, y + (laneH-bh)/2, bw, 1);
  });

  // gutter divider
  ctx.strokeStyle = "#000"; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(TL_GUTTER, 0); ctx.lineTo(TL_GUTTER, h); ctx.stroke();

  // playhead
  const video = $("video");
  if (video.duration) {
    const px = xForT(video.currentTime);
    ctx.strokeStyle = "#4a9eff"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, h); ctx.stroke();
    ctx.fillStyle = "#4a9eff";
    ctx.beginPath(); ctx.moveTo(px-6, 0); ctx.lineTo(px+6, 0); ctx.lineTo(px, 10); ctx.closePath(); ctx.fill();
  }
  requestAnimationFrame(drawTimeline);
}
function seekTimeline(clientX) {
  const cv = $("timeline"), rect = cv.getBoundingClientRect();
  const trackW = rect.width - TL_GUTTER;
  const frac = Math.min(1, Math.max(0, (clientX - rect.left - TL_GUTTER) / trackW));
  $("video").currentTime = frac * (state.summary.duration || 0);
}
$("timeline").addEventListener("mousedown",(e)=>{ if(!state.summary) return; state.dragging=true; seekTimeline(e.clientX); });
window.addEventListener("mousemove",(e)=>{ if(state.dragging) seekTimeline(e.clientX); });
window.addEventListener("mouseup",()=>{ state.dragging=false; });

// ---------- transport controls ----------
const video = $("video");
function updatePlayIcon() {
  const paused = video.paused;
  $("icoPlay").classList.toggle("hidden", !paused);
  $("icoPause").classList.toggle("hidden", paused);
  $("bigPlay").classList.toggle("hidden", !paused || !video.src);
}
function togglePlay(){ video.paused ? video.play() : video.pause(); }
video.addEventListener("play", updatePlayIcon);
video.addEventListener("pause", updatePlayIcon);
video.addEventListener("timeupdate", ()=>{ $("tCur").textContent = fmtClock(video.currentTime); });
$("btnPlay").addEventListener("click", togglePlay);
$("bigPlay").addEventListener("click", togglePlay);
$("videoWrap").addEventListener("click",(e)=>{ if(e.target===$("overlay")||e.target===video) togglePlay(); });
$("btnRestart").addEventListener("click", ()=>{ video.currentTime=0; video.play(); });
$("btnBack").addEventListener("click", ()=>{ video.pause(); video.currentTime -= 1/(state.det?.fps||30); });
$("btnFwd").addEventListener("click", ()=>{ video.pause(); video.currentTime += 1/(state.det?.fps||30); });
const SPEEDS = [0.25,0.5,1,1.5,2]; let spIdx = 2;
$("btnSpeed").addEventListener("click", ()=>{ spIdx=(spIdx+1)%SPEEDS.length;
  video.playbackRate = SPEEDS[spIdx]; $("btnSpeed").textContent = SPEEDS[spIdx].toFixed(2).replace(/0$/,"")+"×"; });
$("btnFull").addEventListener("click", ()=>{
  const el = $("videoWrap");
  if (document.fullscreenElement) document.exitFullscreen(); else el.requestFullscreen?.();
});
document.addEventListener("keydown",(e)=>{
  if (e.target.tagName === "INPUT") return;
  if (e.code === "Space") { e.preventDefault(); togglePlay(); }
  else if (e.code === "ArrowLeft") { video.currentTime -= 5; }
  else if (e.code === "ArrowRight") { video.currentTime += 5; }
  else if (e.key === "f") { $("btnFull").click(); }
});

// ---------- search ----------
$("searchForm").addEventListener("submit",(e)=>{ e.preventDefault(); runSearch($("searchInput").value.trim()); });
async function runSearch(q) {
  if (!q || !state.videoId) return;
  const res = await fetch(`/api/search/${state.videoId}?q=${encodeURIComponent(q)}`).then((r)=>r.json());
  $("searchMeta").textContent = `${res.count} result${res.count===1?"":"s"} for “${q}”`;
  renderResults(res.matches);
}
function renderResults(matches) {
  const box = $("results");
  if (!matches.length) { box.innerHTML = `<div class="empty">No matches.</div>`; return; }
  box.innerHTML = matches.map((m)=>{
    const c = classColor(m.cls);
    return `<div class="result" data-t="${m.start_sec}" style="border-left-color:${c}">
      <div class="r-top"><span class="r-cls" style="color:${c}">${m.cls}</span>
        <span class="r-id">#${m.track_id}</span></div>
      <div class="r-time">${m.start_time.slice(0,8)} → ${m.end_time.slice(0,8)}
        <span class="r-dur">· ${m.duration_sec}s</span></div>
    </div>`;
  }).join("");
  [...box.querySelectorAll(".result")].forEach((el)=>el.addEventListener("click",()=>{
    video.currentTime = parseFloat(el.dataset.t); video.play().catch(()=>{});
  }));
}

// ---------- people (identities) ----------
let idFiles = [];  // File[] staged for enrollment

async function loadIdentities() {
  try {
    const { identities } = await fetch("/api/identities").then((r) => r.json());
    renderIdentityList(identities);
  } catch (e) { /* non-fatal */ }
}
function renderIdentityList(items) {
  const box = $("identityList");
  if (!items || !items.length) {
    box.innerHTML = `<div class="empty">No people added.</div>`;
    return;
  }
  box.innerHTML = items.map((it) => {
    const c = classColor(it.name);
    const avatar = it.has_thumb
      ? `<img src="/api/identities/${it.id}/thumb" alt="" />`
      : `<span class="id-noimg" style="background:${c}">${(it.name[0]||"?").toUpperCase()}</span>`;
    return `<div class="id-row">
      ${avatar}
      <div class="id-meta">
        <div class="id-name">${it.name}</div>
        <div class="id-faces">${it.num_faces} face${it.num_faces===1?"":"s"} enrolled</div>
      </div>
      <button class="id-del" data-id="${it.id}" title="remove">✕</button>
    </div>`;
  }).join("");
  [...box.querySelectorAll(".id-del")].forEach((el) => el.addEventListener("click", () =>
    deleteIdentity(el.dataset.id)));
}
async function deleteIdentity(id) {
  if (!confirm("Remove this person? Videos analyzed later will no longer be labeled with this name.")) return;
  await fetch(`/api/identities/${id}`, { method: "DELETE" });
  loadIdentities();
}

// dialog open / close
function openIdModal() {
  idFiles = []; refreshIdStaging();
  $("idName").value = ""; setIdMsg("", "");
  $("idModal").classList.remove("hidden");
  $("idName").focus();
}
function closeIdModal() { $("idModal").classList.add("hidden"); }
$("btnCreateIdentity").addEventListener("click", openIdModal);
$("idClose").addEventListener("click", closeIdModal);
$("idModal").addEventListener("click", (e) => { if (e.target === $("idModal")) closeIdModal(); });

// photo staging (dropzone + file input)
const idDrop = $("idDrop"), idFileInput = $("idFiles");
idDrop.addEventListener("click", () => idFileInput.click());
["dragover","dragenter"].forEach((e) => idDrop.addEventListener(e,(ev)=>{ev.preventDefault();idDrop.classList.add("drag");}));
["dragleave","drop"].forEach((e) => idDrop.addEventListener(e,(ev)=>{ev.preventDefault();idDrop.classList.remove("drag");}));
idDrop.addEventListener("drop",(ev)=>addIdFiles(ev.dataTransfer.files));
idFileInput.addEventListener("change",()=>{ addIdFiles(idFileInput.files); idFileInput.value=""; });
function addIdFiles(fileList) {
  for (const f of fileList) if (f.type.startsWith("image/")) idFiles.push(f);
  refreshIdStaging();
}
function refreshIdStaging() {
  $("idCount").textContent = idFiles.length;
  const box = $("idPreview");
  box.innerHTML = idFiles.map((f) => `<img src="${URL.createObjectURL(f)}" alt="" />`).join("");
}
function setIdMsg(text, cls) { const el = $("idMsg"); el.textContent = text; el.className = "id-msg" + (cls?" "+cls:""); }

// enroll
$("idEnroll").addEventListener("click", async () => {
  const name = $("idName").value.trim();
  if (!name) { setIdMsg("Enter a name first.", "err"); return; }
  if (idFiles.length < 1) { setIdMsg("Add at least one photo.", "err"); return; }
  const fd = new FormData();
  fd.append("name", name);
  idFiles.forEach((f) => fd.append("files", f));
  const btn = $("idEnroll");
  btn.disabled = true; setIdMsg("Detecting faces…", "");
  try {
    const res = await fetch("/api/identities", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { setIdMsg(data.detail || "Could not add person", "err"); btn.disabled = false; return; }
    setIdMsg(`${data.num_faces} face${data.num_faces===1?"":"s"} added from ${data.num_photos} photo${data.num_photos===1?"":"s"}`, "ok");
    loadIdentities();
    setTimeout(() => { btn.disabled = false; closeIdModal(); }, 1100);
  } catch (e) {
    setIdMsg("Could not add person — see console", "err"); console.error(e); btn.disabled = false;
  }
});

// ---------- accuracy report ----------
$("evalFile").addEventListener("change", async () => {
  const f = $("evalFile").files[0];
  if (!f || !state.videoId) { if (!state.videoId) $("evalOut").textContent = "Load a video first."; return; }
  const out = $("evalOut"); out.textContent = "Scoring…";
  const fd = new FormData(); fd.append("file", f);
  try {
    const res = await fetch(`/api/evaluate/${state.videoId}`, { method: "POST", body: fd });
    const m = await res.json();
    if (!res.ok) { out.textContent = m.detail || "Evaluation failed"; return; }
    renderEval(m);
  } catch (e) { out.textContent = "Evaluation failed — see console"; console.error(e); }
  $("evalFile").value = "";
});
function renderEval(m) {
  const d = m.detection || {}, t = m.tracking, id = m.identity;
  let g = `<div class="ev-grid">
    <div class="info"><label>Precision</label><b>${d.precision ?? "—"}</b></div>
    <div class="info"><label>Recall</label><b>${d.recall ?? "—"}</b></div>
    <div class="info"><label>mAP@0.5</label><b>${d["mAP@0.5"] ?? "—"}</b></div>
    <div class="info"><label>F1</label><b>${d.f1 ?? "—"}</b></div>`;
  if (t && !t.error) g += `<div class="info"><label>MOTA</label><b>${t.MOTA}</b></div>
    <div class="info"><label>IDF1</label><b>${t.IDF1}</b></div>`;
  if (id) g += `<div class="info"><label>ID acc.</label><b>${id.accuracy}</b></div>`;
  g += `</div><div class="ev-sub">${d.tp||0} TP · ${d.fp||0} FP · ${d.fn||0} FN · ${m.frames_evaluated||0} frames scored</div>`;
  $("evalOut").innerHTML = g;
}

// ---------- overlay toggles ----------
function bindToggle(id, key) {
  const el = $(id);
  el.addEventListener("click", () => { state[key] = !state[key]; el.classList.toggle("on", state[key]); });
}
bindToggle("tgTraj", "showTraj");
bindToggle("tgSkel", "showSkel");
bindToggle("tgLine", "showLine");

// ---------- init ----------
loadIdentities();
$("cClock").textContent = new Date().toLocaleTimeString();
updatePlayIcon();
