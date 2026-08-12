// ============================================================
//  V.E.S.E HUD — front-end controller
// ============================================================
"use strict";
const $ = (id) => document.getElementById(id);
const state = { videoId: null, det: null, frameTimes: [], summary: null, dragging: false,
  traj: {}, showTraj: false, showSkel: true, showLine: true };

// COCO-17 skeleton connections (pairs of keypoint indices)
const SKELETON = [[5,6],[5,7],[7,9],[6,8],[8,10],[5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],[0,5],[0,6]];

// curated categorical palette (distinguishable on dark)
const PALETTE = ["#38f0ff","#ffb454","#4dffb0","#ff6ec7","#a78bfa","#7dff5b",
  "#ff5d5d","#5db4ff","#ffe45e","#ff8a3d","#31d0aa","#c084fc","#f472b6","#67e8f9"];
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

// ---------- boot sequence ----------
const BOOT_LINES = [
  "> initializing V.E.S.E core .............. [<span class='ok'>OK</span>]",
  "> mounting neural detection lattice ...... [<span class='ok'>OK</span>]",
  "> calibrating optical sensors ............ [<span class='ok'>OK</span>]",
  "> linking temporal tracking array ........ [<span class='ok'>OK</span>]",
  "> compute backend ........................ [<span class='ok'>GPU</span>]",
  "> S Y S T E M   O N L I N E",
];
function runBoot() {
  const log = $("bootlog"); let i = 0;
  const tick = () => {
    if (i < BOOT_LINES.length) { log.innerHTML += BOOT_LINES[i++] + "\n"; setTimeout(tick, 340); }
    else setTimeout(endBoot, 700);
  };
  tick();
}
function endBoot() { $("boot").classList.add("gone"); }
$("boot").addEventListener("click", endBoot);

// ---------- system info chips ----------
async function loadInfo() {
  try {
    const info = await fetch("/api/info").then((r) => r.json());
    $("cModel").textContent = info.model.replace(".pt", "");
    $("cRes").textContent = info.imgsz + "px";
    const gpu = info.device.replace(/NVIDIA GeForce /i, "").replace(/ Laptop GPU/i, "");
    $("cGpu").textContent = gpu.length > 14 ? gpu.slice(0, 14) : gpu;
  } catch (e) { /* non-fatal */ }
}
setInterval(() => { $("cClock").textContent = new Date().toLocaleTimeString(); }, 1000);

// ---------- upload / process ----------
const dropzone = $("dropzone"), fileInput = $("fileInput");
dropzone.addEventListener("click", () => fileInput.click());
["dragover","dragenter"].forEach((e) => dropzone.addEventListener(e,(ev)=>{ev.preventDefault();dropzone.classList.add("drag");}));
["dragleave","drop"].forEach((e) => dropzone.addEventListener(e,(ev)=>{ev.preventDefault();dropzone.classList.remove("drag");}));
dropzone.addEventListener("drop",(ev)=>{ const f=ev.dataTransfer.files[0]; if(f) startPipeline(f); });
fileInput.addEventListener("change",()=>{ if(fileInput.files[0]) startPipeline(fileInput.files[0]); });

const PROC_MSGS = ["initializing neural core","scanning frames","locking targets",
  "resolving track identities","folding temporal intervals"];

async function startPipeline(file) {
  setSys("UPLOADING FEED…", "");
  dropzone.classList.add("hidden");
  $("viewer").classList.remove("hidden");
  $("processing").classList.remove("hidden");
  $("procPct").textContent = "0%";

  const fd = new FormData(); fd.append("file", file);
  const up = await fetch("/api/upload", { method:"POST", body:fd }).then((r)=>r.json());
  state.videoId = up.video_id;
  await fetch(`/api/process/${state.videoId}`, { method:"POST" });
  setSys("ANALYZING FEED…", "");
  pollStatus();
}

async function pollStatus() {
  const s = await fetch(`/api/status/${state.videoId}`).then((r)=>r.json());
  if (s.state === "processing") {
    const pct = s.total ? Math.floor((s.done/s.total)*100) : 0;
    $("procPct").textContent = `${pct}%`;
    $("procSub").textContent = PROC_MSGS[Math.min(PROC_MSGS.length-1, Math.floor(pct/22))];
    setTimeout(pollStatus, 400);
  } else if (s.state === "done") {
    $("procPct").textContent = "100%"; loadResults();
  } else if (s.state === "error") {
    setSys("ANALYSIS FAILED — SEE CONSOLE", "err");
    $("procPct").textContent = "ERR"; console.error("Processing failed:", s.error);
  } else setTimeout(pollStatus, 400);
}

// ---------- load processed data ----------
async function loadResults() {
  state.summary = await fetch(`/api/summary/${state.videoId}`).then((r)=>r.json());
  state.det = await fetch(`/api/detections/${state.videoId}`).then((r)=>r.json());
  state.frameTimes = state.det.frames.map((f)=>f.t);

  $("processing").classList.add("hidden");
  setSys(`FEED LOCKED · ${state.summary.num_intervals} EVENTS INDEXED`, "live");
  $("roEvents").textContent = state.summary.num_intervals;
  renderClasses(); renderChips();
  buildTrajectories(); renderAnalytics();

  const video = $("video");
  video.src = `/api/video/${state.videoId}`;
  video.load();
  video.addEventListener("loadedmetadata", () => { $("tDur").textContent = fmtClock(video.duration); }, { once:true });
  video.play().catch(()=>{});
  requestAnimationFrame(drawLoop);
  requestAnimationFrame(drawTimeline);
}

function renderClasses() {
  const box = $("classList"), classes = state.summary.classes, keys = Object.keys(classes);
  if (!keys.length) { box.innerHTML = `<div class="empty">No objects detected.</div>`; return; }
  const max = Math.max(...keys.map((k)=>classes[k].count));
  box.innerHTML = keys.map((k)=>{
    const c = classColor(k), pct = Math.round((classes[k].count/max)*100);
    return `<div class="class-row">
      <div class="cr-top">
        <span class="class-swatch" style="background:${c};color:${c}"></span>
        <span class="class-name">${k}</span>
        <span class="class-count">×${classes[k].count}</span>
      </div>
      <div class="cr-bar"><i style="width:${pct}%;background:${c};box-shadow:0 0 8px ${c}"></i></div>
    </div>`;
  }).join("");
}

function renderChips() {
  const box = $("chips"), keys = Object.keys(state.summary.classes);
  box.innerHTML = keys.map((k)=>`<span class="qchip" data-q="${k}">
    <span class="dot" style="background:${classColor(k)};color:${classColor(k)}"></span>${k}</span>`).join("");
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
  for (const [id, ep] of [["heatImg","heatmap"], ["depthImg","depth"]]) {
    const img = $(id);
    img.onerror = () => { img.closest(".an-heat").style.display = "none"; };
    img.onload = () => { img.closest(".an-heat").style.display = ""; };
    img.src = `/api/${ep}/${state.videoId}?t=${Date.now()}`;
  }
}
function drawTrajectories(ctx, sx, sy, tNow) {
  ctx.save(); ctx.lineWidth = 2; ctx.globalAlpha = 0.85;
  for (const tid in state.traj) {
    const tr = state.traj[tid], pts = tr.pts.filter((p)=>p[0] <= tNow);
    if (pts.length < 2) continue;
    ctx.strokeStyle = tr.color; ctx.shadowColor = tr.color; ctx.shadowBlur = 6;
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
  ctx.save(); ctx.strokeStyle = "#ffb454"; ctx.globalAlpha = 0.7; ctx.lineWidth = 2;
  ctx.setLineDash([8,6]); ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,cv.height); ctx.stroke();
  ctx.restore();
}
function drawSkeleton(ctx, kpts, sx, sy, col) {
  ctx.save(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.fillStyle = "#eafcff";
  ctx.shadowColor = col; ctx.shadowBlur = 4;
  for (const [a,b] of SKELETON) {
    const p = kpts[a], q = kpts[b];
    if (p && q && p[2] > 0.3 && q[2] > 0.3) {
      ctx.beginPath(); ctx.moveTo(p[0]*sx, p[1]*sy); ctx.lineTo(q[0]*sx, q[1]*sy); ctx.stroke();
    }
  }
  ctx.shadowBlur = 0;
  for (const p of kpts) if (p[2] > 0.3) { ctx.beginPath(); ctx.arc(p[0]*sx, p[1]*sy, 2.5, 0, 7); ctx.fill(); }
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
    const phase = 0.6 + 0.4*Math.abs(Math.sin(performance.now()/450));
    if (state.showTraj) drawTrajectories(ctx, sx, sy, video.currentTime);
    if (state.showLine) drawCountLine(ctx, cv);
    if (frame) {
      $("liveCount").textContent = frame.dets.length;
      $("roFrame").textContent = Math.round(video.currentTime * (state.det.fps||30));
      for (const d of frame.dets) drawReticle(ctx, d, sx, sy, phase);
    }
    $("roTime").textContent = fmtClock(video.currentTime);
  }
  requestAnimationFrame(drawLoop);
}
function drawReticle(ctx, d, sx, sy, phase) {
  const [x1,y1,x2,y2] = d.box;
  const X=x1*sx, Y=y1*sy, W=(x2-x1)*sx, H=(y2-y1)*sy;
  const col = classColor(d.cls), len = Math.max(8, Math.min(22, W/3.5, H/3.5));

  ctx.save();
  // segmentation mask outline (if present)
  if (d.mask && d.mask.length > 2) {
    ctx.beginPath(); ctx.moveTo(d.mask[0][0]*sx, d.mask[0][1]*sy);
    for (let i = 1; i < d.mask.length; i++) ctx.lineTo(d.mask[i][0]*sx, d.mask[i][1]*sy);
    ctx.closePath();
    ctx.fillStyle = col + "22"; ctx.fill();
    ctx.strokeStyle = col; ctx.lineWidth = 1.5; ctx.stroke();
  }
  // faint fill
  ctx.fillStyle = col + "14"; ctx.fillRect(X,Y,W,H);
  // corner target brackets
  ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.shadowColor = col; ctx.shadowBlur = 10*phase;
  const corners = [[X,Y,1,1],[X+W,Y,-1,1],[X,Y+H,1,-1],[X+W,Y+H,-1,-1]];
  for (const [cx,cy,dx,dy] of corners) {
    ctx.beginPath();
    ctx.moveTo(cx, cy+dy*len); ctx.lineTo(cx, cy); ctx.lineTo(cx+dx*len, cy); ctx.stroke();
  }
  // center crosshair tick
  const mx=X+W/2, my=Y+H/2, t=4;
  ctx.globalAlpha = 0.7*phase; ctx.beginPath();
  ctx.moveTo(mx-t,my); ctx.lineTo(mx+t,my); ctx.moveTo(mx,my-t); ctx.lineTo(mx,my+t); ctx.stroke();
  ctx.globalAlpha = 1; ctx.shadowBlur = 0;

  // skeleton (pose)
  if (state.showSkel && d.kpts) drawSkeleton(ctx, d.kpts, sx, sy, col);

  // label chip (+ action, if any)
  const label = `${d.cls.toUpperCase()} · ${d.id}`
    + (d.attrs ? ` · ${d.attrs.gender}${d.attrs.age}` : "")
    + (d.action ? ` · ${d.action.toUpperCase()}` : "")
    + (d.depth != null ? ` · D${d.depth}` : "");
  const pctTxt = `${Math.round(d.conf*100)}%`;
  ctx.font = "600 12px Rajdhani, monospace";
  const lw = ctx.measureText(label).width, chipW = lw + 46, chipH = 17;
  let ly = Y - chipH - 2; if (ly < 0) ly = Y + 2;
  ctx.fillStyle = col; ctx.fillRect(X, ly, chipW, chipH);
  ctx.fillStyle = "#04121a"; ctx.textBaseline = "middle";
  ctx.fillText(label, X+5, ly+chipH/2);
  // confidence pill
  ctx.font = "700 10px 'Share Tech Mono', monospace";
  ctx.fillText(pctTxt, X+chipW-34, ly+chipH/2);
  ctx.restore();
}

// ---------- timeline ----------
function drawTimeline() {
  const cv = $("timeline"); if (!state.summary) return;
  const w = cv.clientWidth, h = 66;
  if (cv.width !== w) cv.width = w; cv.height = h;
  const ctx = cv.getContext("2d");
  ctx.clearRect(0,0,w,h);
  const dur = state.summary.duration || 1;
  const classes = Object.keys(state.summary.classes);
  const top = 18, lanes = Math.max(1, classes.length);
  const laneH = Math.max(3, Math.min(9, (h-top-6)/lanes));

  // grid ticks
  ctx.strokeStyle = "rgba(56,224,255,0.12)"; ctx.lineWidth = 1;
  for (let i=0;i<=10;i++){ const x=(i/10)*w; ctx.beginPath(); ctx.moveTo(x,top-4); ctx.lineTo(x,h); ctx.stroke(); }

  state.summary.intervals.forEach((iv)=>{
    const lane = classes.indexOf(iv.cls);
    const x = (iv.start_sec/dur)*w, bw = Math.max(2, ((iv.end_sec-iv.start_sec)/dur)*w);
    ctx.fillStyle = classColor(iv.cls); ctx.globalAlpha = 0.85;
    ctx.fillRect(x, top + lane*laneH, bw, laneH-1);
  });
  ctx.globalAlpha = 1;

  // playhead
  const video = $("video");
  if (video.duration) {
    const px = (video.currentTime/dur)*w;
    ctx.fillStyle = "rgba(56,224,255,0.12)"; ctx.fillRect(0,0,px,h);
    ctx.strokeStyle = "#eafcff"; ctx.lineWidth = 2; ctx.shadowColor = "#38f0ff"; ctx.shadowBlur = 8;
    ctx.beginPath(); ctx.moveTo(px,0); ctx.lineTo(px,h); ctx.stroke(); ctx.shadowBlur = 0;
    ctx.beginPath(); ctx.moveTo(px-5,0); ctx.lineTo(px+5,0); ctx.lineTo(px,7); ctx.closePath();
    ctx.fillStyle = "#eafcff"; ctx.fill();
  }
  requestAnimationFrame(drawTimeline);
}
function seekTimeline(clientX) {
  const cv = $("timeline"), rect = cv.getBoundingClientRect();
  const frac = Math.min(1, Math.max(0, (clientX-rect.left)/rect.width));
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
  $("searchMeta").textContent = `${res.count} MATCH${res.count===1?"":"ES"} · "${q.toUpperCase()}"`;
  renderResults(res.matches);
}
function renderResults(matches) {
  const box = $("results");
  if (!matches.length) { box.innerHTML = `<div class="empty" style="color:var(--muted);padding:10px 0">No matches in feed.</div>`; return; }
  box.innerHTML = matches.map((m)=>{
    const c = classColor(m.cls);
    return `<div class="result" data-t="${m.start_sec}" style="border-left-color:${c}">
      <div class="r-top"><span class="r-cls" style="color:${c}">${m.cls}</span>
        <span class="r-id">TRK ${m.track_id}</span></div>
      <div class="r-time">${m.start_time.slice(0,8)} → ${m.end_time.slice(0,8)}
        <span class="r-dur">· ${m.duration_sec}s</span></div>
    </div>`;
  }).join("");
  [...box.querySelectorAll(".result")].forEach((el)=>el.addEventListener("click",()=>{
    video.currentTime = parseFloat(el.dataset.t); video.play().catch(()=>{});
  }));
}

// ---------- identity registry ----------
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
    box.innerHTML = `<div class="empty">No identities enrolled.</div>`;
    return;
  }
  box.innerHTML = items.map((it) => {
    const c = classColor(it.name);
    const avatar = it.has_thumb
      ? `<img src="/api/identities/${it.id}/thumb" alt="" />`
      : `<span class="id-noimg" style="color:${c}">${(it.name[0]||"?").toUpperCase()}</span>`;
    return `<div class="id-row">
      ${avatar}
      <div class="id-meta">
        <div class="id-name" style="color:${c}">${it.name}</div>
        <div class="id-faces">${it.num_faces} FACE${it.num_faces===1?"":"S"} ENROLLED</div>
      </div>
      <button class="id-del" data-id="${it.id}" title="remove">✕</button>
    </div>`;
  }).join("");
  [...box.querySelectorAll(".id-del")].forEach((el) => el.addEventListener("click", () =>
    deleteIdentity(el.dataset.id)));
}
async function deleteIdentity(id) {
  if (!confirm("Remove this identity? Videos analyzed later will no longer be labeled with this name.")) return;
  await fetch(`/api/identities/${id}`, { method: "DELETE" });
  loadIdentities();
}

// modal open / close
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
  btn.disabled = true; setIdMsg("ENROLLING — DETECTING FACES…", "");
  try {
    const res = await fetch("/api/identities", { method: "POST", body: fd });
    const data = await res.json();
    if (!res.ok) { setIdMsg((data.detail || "Enrollment failed").toUpperCase(), "err"); btn.disabled = false; return; }
    setIdMsg(`${data.num_faces} FACE${data.num_faces===1?"":"S"} ENROLLED FROM ${data.num_photos} PHOTO${data.num_photos===1?"":"S"}`, "ok");
    loadIdentities();
    setTimeout(() => { btn.disabled = false; closeIdModal(); }, 1100);
  } catch (e) {
    setIdMsg("ENROLLMENT FAILED — SEE CONSOLE", "err"); console.error(e); btn.disabled = false;
  }
});

// ---------- model evaluation ----------
$("evalFile").addEventListener("change", async () => {
  const f = $("evalFile").files[0];
  if (!f || !state.videoId) { if (!state.videoId) $("evalOut").textContent = "Load a feed first."; return; }
  const out = $("evalOut"); out.textContent = "SCORING…";
  const fd = new FormData(); fd.append("file", f);
  try {
    const res = await fetch(`/api/evaluate/${state.videoId}`, { method: "POST", body: fd });
    const m = await res.json();
    if (!res.ok) { out.textContent = (m.detail || "evaluation failed").toUpperCase(); return; }
    renderEval(m);
  } catch (e) { out.textContent = "EVALUATION FAILED — SEE CONSOLE"; console.error(e); }
  $("evalFile").value = "";
});
function renderEval(m) {
  const d = m.detection || {}, t = m.tracking, id = m.identity;
  let g = `<div class="ev-grid">
    <div class="ro"><label>PRECISION</label><b>${d.precision ?? "—"}</b></div>
    <div class="ro"><label>RECALL</label><b>${d.recall ?? "—"}</b></div>
    <div class="ro"><label>mAP@0.5</label><b>${d["mAP@0.5"] ?? "—"}</b></div>
    <div class="ro"><label>F1</label><b>${d.f1 ?? "—"}</b></div>`;
  if (t && !t.error) g += `<div class="ro"><label>MOTA</label><b>${t.MOTA}</b></div>
    <div class="ro"><label>IDF1</label><b>${t.IDF1}</b></div>`;
  if (id) g += `<div class="ro"><label>ID ACC</label><b>${id.accuracy}</b></div>`;
  g += `</div><div class="ev-sub">${d.tp||0} TP · ${d.fp||0} FP · ${d.fn||0} FN · ${m.frames_evaluated||0} frames scored</div>`;
  $("evalOut").innerHTML = g;
}

// ---------- analytics toggles ----------
function bindToggle(id, key) {
  const el = $(id);
  el.addEventListener("click", () => { state[key] = !state[key]; el.classList.toggle("on", state[key]); });
}
bindToggle("tgTraj", "showTraj");
bindToggle("tgSkel", "showSkel");
bindToggle("tgLine", "showLine");

// ---------- init ----------
runBoot();
loadInfo();
loadIdentities();
$("cClock").textContent = new Date().toLocaleTimeString();
updatePlayIcon();
