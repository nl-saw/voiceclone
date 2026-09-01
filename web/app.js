/* voiceclone web UI — vanilla JS, no build step */
"use strict";

const $ = (sel) => document.querySelector(sel);

let EMOTIONS = ["neutral", "happy", "sad", "angry", "calm", "excited", "fearful", "surprised"];
let activeVoice = null;
let selectedEmotion = "neutral";
let trainPoll = null;
let activeVoiceSeconds = 0;
const RECOMMENDED_MIN_AUDIO_S = 600; // ~10 min, mirrors backend advisory

function fmtBytes(n) {
  if (n == null) return "—";
  const u = ["B", "KB", "MB", "GB", "TB"];
  let i = 0; n = Number(n);
  while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
  return n.toFixed(n >= 10 || i === 0 ? 0 : 1) + " " + u[i];
}

function updateDataWarning() {
  const el = $("#data-warning");
  if (!el) return;
  if (activeVoiceSeconds > 0 && activeVoiceSeconds < RECOMMENDED_MIN_AUDIO_S) {
    el.hidden = false;
    el.innerHTML = `⚠️ Only <b>${Math.round(activeVoiceSeconds)}s</b> of source audio (recommended ≥ ${RECOMMENDED_MIN_AUDIO_S / 60} min). ` +
      "With this little data, more epochs usually make words <b>worse</b>, not better — 1 epoch is typically the ceiling. " +
      "Zero-shot often sounds cleaner; collect more audio for real fine-tune gains.";
  } else {
    el.hidden = true;
    el.innerHTML = "";
  }
}

// ---------------------------------------------------------------- helpers --
async function api(path, opts = {}) {
  const res = await fetch(path, opts);
  if (!res.ok) {
    let msg = res.statusText;
    try { msg = (await res.json()).detail || msg; } catch {}
    throw new Error(msg);
  }
  return res;
}

function setStatus(el, text, cls = "") {
  el.textContent = text;
  el.className = "status" + (cls ? " " + cls : "");
}

// ---------------------------------------------------------------- voices --
async function loadVoices() {
  const voices = await (await api("/api/voices")).json();
  const list = $("#voice-list");
  list.innerHTML = "";
  if (!voices.length) {
    list.innerHTML = '<span class="hint">No voices yet — drop some audio files below to create one.</span>';
  }
  for (const v of voices) {
    const chip = document.createElement("div");
    chip.className = "voice-chip" + (v.name === activeVoice ? " active" : "");
    chip.innerHTML = `${esc(v.name)} <span class="hint">${v.samples} clips</span>${v.finetuned ? ' <span class="ft">FT✓</span>' : ""}`;
    chip.onclick = () => selectVoice(v.name);
    list.appendChild(chip);
  }
  if (voices.length && !activeVoice) selectVoice(voices[0].name);
}

async function selectVoice(name) {
  activeVoice = name;
  $("#active-voice-label").textContent = name;
  await loadVoices();
  await loadSamples(name);
  loadCheckpoints(name).catch(() => {});
}

async function loadSamples(name) {
  const d = await (await api(`/api/voices/${encodeURIComponent(name)}`)).json();
  activeVoiceSeconds = d.total_seconds || 0;
  updateDataWarning();
  const tbody = $("#samples-table tbody");
  tbody.innerHTML = "";
  for (const s of d.samples) {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${esc(s.id)}</td>
      <td><select data-sid="${s.id}">${EMOTIONS.map(e => `<option ${e === s.emotion ? "selected" : ""}>${e}</option>`).join("")}</select></td>
      <td>${esc(s.language)}</td>
      <td>${s.duration_s.toFixed(1)}s</td>
      <td class="transcript" title="${esc(s.transcript)}">${esc(s.transcript || "(no transcript)")}</td>
      <td><button class="del" data-del="${s.id}" title="delete sample">✕</button></td>`;
    tbody.appendChild(tr);
  }
  if (!d.samples.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="hint">No samples yet.</td></tr>';
  }
}

$("#samples-table").addEventListener("change", async (e) => {
  const sel = e.target.closest("select[data-sid]");
  if (!sel || !activeVoice) return;
  await api(`/api/voices/${encodeURIComponent(activeVoice)}/samples/${sel.dataset.sid}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ emotion: sel.value }),
  });
});

$("#samples-table").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-del]");
  if (!btn || !activeVoice) return;
  if (!confirm(`Delete sample ${btn.dataset.del}?`)) return;
  await api(`/api/voices/${encodeURIComponent(activeVoice)}/samples/${btn.dataset.del}`, { method: "DELETE" });
  loadSamples(activeVoice);
  loadVoices();
});

// ---------------------------------------------------------------- upload --
const dropzone = $("#dropzone");
["dragenter", "dragover"].forEach(ev => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.add("dragover"); }));
["dragleave", "drop"].forEach(ev => dropzone.addEventListener(ev, (e) => { e.preventDefault(); dropzone.classList.remove("dragover"); }));

dropzone.addEventListener("drop", async (e) => {
  const files = [...e.dataTransfer.files].filter(f => /\.(wav|mp3|flac|m4a|ogg|opus|webm)$/i.test(f.name));
  if (!files.length) return;
  await uploadFiles(files);
});

$("#file-input").addEventListener("change", async (e) => {
  const files = [...e.target.files];
  e.target.value = "";
  if (files.length) await uploadFiles(files);
});

async function uploadFiles(files) {
  if (!activeVoice) { setStatus($("#status"), "Select or create a voice first.", "err"); return; }
  const name = prompt("Upload samples to which voice?", activeVoice);
  if (!name) return;
  activeVoice = name.toLowerCase().trim();

  const fd = new FormData();
  files.forEach(f => fd.append("files", f));
  fd.append("lang", $("#up-lang").value);
  fd.append("emotion", $("#up-emotion").value);
  fd.append("note", "");

  setStatus($("#status"), `Uploading ${files.length} file(s) — transcribing with Whisper… (this can take a while)`, "warn");
  try {
    const res = await api(`/api/voices/${encodeURIComponent(activeVoice)}/samples`, { method: "POST", body: fd });
    const d = await res.json();
    const ok = d.reports.filter(r => r.ok).length;
    if (ok === files.length) {
      setStatus($("#status"), `✔ Added ${ok}/${files.length} samples to '${d.voice}'.`, "ok");
    } else {
      const errs = d.reports.filter(r => !r.ok).map(r => `${r.file}: ${r.error || "unknown error"}`);
      setStatus($("#status"), `⚠ Added ${ok}/${files.length} samples to '${d.voice}'. Failed: ${errs.join(" | ")}`, "err");
    }
    selectVoice(activeVoice);
  } catch (err) {
    setStatus($("#status"), `Upload failed: ${err.message}`, "err");
  }
}

// ---------------------------------------------------------------- synthesize --
function renderEmotionChips() {
  const box = $("#emotion-chips");
  box.innerHTML = "";
  for (const e of EMOTIONS) {
    const c = document.createElement("span");
    c.className = "chip" + (e === selectedEmotion ? " active" : "");
    c.textContent = e;
    c.onclick = () => { selectedEmotion = e; renderEmotionChips(); };
    box.appendChild(c);
  }
}

// Read the optional advanced generation settings; empty fields are omitted so
// the server falls back to the model's own defaults (None).
function genParams() {
  const p = {};
  const num = (id) => {
    const raw = document.getElementById(id).value.trim();
    if (raw === "") return null;
    const n = Number(raw);
    return Number.isFinite(n) ? n : null;
  };
  const map = { temperature: "temperature", length_penalty: "length-penalty", repetition_penalty: "repetition-penalty", top_k: "top-k", top_p: "top-p", speed: "speed" };
  for (const [key, id] of Object.entries(map)) {
    const v = num(id);
    if (v != null) p[key] = v;
  }
  return p;
}

$("#gen-btn").addEventListener("click", async () => {
  const text = $("#text").value.trim();
  if (!text) return setStatus($("#status"), "Type some text first.", "err");
  if (!activeVoice) return setStatus($("#status"), "No voice selected.", "err");

  const btn = $("#gen-btn");
  btn.disabled = true;
  $("#player").style.display = "none";
  setStatus($("#status"), `Generating with '${activeVoice}'… (first run loads the model, ~1 min)`, "warn");
  const t0 = performance.now();
  try {
    const res = await api("/api/synthesize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice: activeVoice,
        text,
        emotion: selectedEmotion,
        style: $("#style").value || null,
        language: $("#syn-lang").value,
        mode: $("#mode").value,
        ...genParams(),
      }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const player = $("#player");
    player.src = url;
    player.style.display = "block";

    let meta = {};
    try {
      const b64 = res.headers.get("X-Synthesis-Meta");
      if (b64) meta = JSON.parse(atob(b64));
    } catch {}
    $("#meta").innerHTML = Object.entries(meta).map(([k, v]) => `<b>${esc(k)}:</b> ${esc(String(v))}`).join(" · ");
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    setStatus($("#status"), `✔ Done in ${secs}s — reference clip tagged '${meta.reference_emotion || "?"}'`, "ok");
  } catch (err) {
    setStatus($("#status"), `Synthesis failed: ${err.message}`, "err");
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------- train --
$("#train-btn").addEventListener("click", async () => {
  if (!activeVoice) return setStatus($("#train-status"), "No voice selected.", "err");
  const epochs = parseInt($("#epochs").value || "1", 10);
  let msg = `Fine-tune '${activeVoice}' for ${epochs} epoch(s)? On CPU this can take hours.`;
  if (activeVoiceSeconds > 0 && activeVoiceSeconds < RECOMMENDED_MIN_AUDIO_S) {
    msg += `\n\n⚠ You only have ~${Math.round(activeVoiceSeconds)}s of audio. With little data, more epochs usually make words WORSE. Continue anyway?`;
  }
  if (!confirm(msg)) return;

  const lrRaw = parseFloat($("#lr").value);
  const body = {
    voice: activeVoice,
    epochs,
    grad_accum_steps: parseInt($("#grad-accum").value || "4", 10),
    precision: $("#precision").value,
    force: $("#force").checked,
  };
  if (!isNaN(lrRaw) && lrRaw > 0) body.lr = lrRaw;

  const btn = $("#train-btn");
  btn.disabled = true;
  setStatus($("#train-status"), "Starting fine-tuning…", "warn");
  try {
    const d = await (await api("/api/train", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    })).json();
    if (d.advisory) setStatus($("#train-status"), `⚠ ${d.advisory}\n\nStarting…`, "warn");
    pollTrain(d.job_id);
  } catch (err) {
    setStatus($("#train-status"), `Failed to start: ${err.message}`, "err");
    btn.disabled = false;
  }
});

function pollTrain(jobId) {
  if (trainPoll) clearInterval(trainPoll);
  trainPoll = setInterval(async () => {
    try {
      const d = await (await api(`/api/train/${jobId}`)).json();
      const tail = (d.log_tail || []).slice(-3).join("\n");
      setStatus($("#train-status"), `status: ${d.status}${d.error ? "\n" + d.error : ""}${tail ? "\n" + tail : ""}`, d.status === "failed" ? "err" : d.status === "done" ? "ok" : "warn");
      if (d.status !== "running") {
        clearInterval(trainPoll);
        trainPoll = null;
        $("#train-btn").disabled = false;
        loadVoices();
        if (activeVoice) loadCheckpoints(activeVoice).catch(() => {});
        loadStorage();
      }
    } catch (err) {
      setStatus($("#train-status"), `poll error: ${err.message}`, "err");
    }
  }, 4000);
}

// ---------------------------------------------------------------- checkpoints --
async function loadCheckpoints(name) {
  const d = await (await api(`/api/voices/${encodeURIComponent(name)}/checkpoints`)).json();
  const sel = $("#ckpt-select");
  sel.innerHTML = '<option value="">— none (zero-shot) —</option>';
  for (const r of d.runs) {
    for (const b of r.best_models) {
      const opt = document.createElement("option");
      opt.value = `${r.path}/${b.file}`;
      opt.textContent = `${r.dir} · step ${b.step}`;
      sel.appendChild(opt);
    }
  }
  if (d.registered) {
    // registered path may be stored resolved or not → match tolerantly by tail
    const reg = d.registered.replace(/\/+$/, "");
    const hit = [...sel.options].find(o => o.value === reg || reg.endsWith(o.value) || o.value.endsWith(reg));
    if (hit) sel.value = hit.value;
  }
}

$("#ckpt-apply").addEventListener("click", async () => {
  if (!activeVoice) return setStatus($("#ckpt-status"), "No voice selected.", "err");
  const ckpt = $("#ckpt-select").value;
  if (!ckpt) return setStatus($("#ckpt-status"), "Pick a checkpoint first (or use Clear).", "err");
  try {
    await api(`/api/voices/${encodeURIComponent(activeVoice)}/checkpoint`, {
      method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ checkpoint: ckpt }),
    });
    setStatus($("#ckpt-status"), `✔ ${activeVoice} now uses: ${ckpt.split("/").slice(-2).join("/")}`, "ok");
    loadCheckpoints(activeVoice); loadVoices(); loadStorage();
  } catch (err) { setStatus($("#ckpt-status"), `Failed: ${err.message}`, "err"); }
});

$("#ckpt-clear").addEventListener("click", async () => {
  if (!activeVoice) return setStatus($("#ckpt-status"), "No voice selected.", "err");
  if (!confirm(`Clear fine-tuned checkpoint for '${activeVoice}'? Synthesis will fall back to zero-shot.`)) return;
  try {
    await api(`/api/voices/${encodeURIComponent(activeVoice)}/checkpoint`, { method: "DELETE" });
    setStatus($("#ckpt-status"), `✔ ${activeVoice} registration cleared — using zero-shot.`, "ok");
    loadCheckpoints(activeVoice); loadVoices(); loadStorage();
  } catch (err) { setStatus($("#ckpt-status"), `Failed: ${err.message}`, "err"); }
});

// ---------------------------------------------------------------- storage --
async function loadStorage() {
  let d;
  try { d = await (await api("/api/storage")).json(); } catch { return; }
  const base = d.breakdown.find(b => b.key === "base_model") || { bytes: 0 };
  const samples = d.breakdown.find(b => b.key === "voices_samples") || { bytes: 0 };
  $("#storage-summary").innerHTML =
    `Total <b>${fmtBytes(d.total_bytes)}</b> · Fine-tune artifacts <b>${fmtBytes(d.ft_total_bytes)}</b> · ` +
    `Base model ${fmtBytes(base.bytes)} (protected) · Source samples ${fmtBytes(samples.bytes)} (protected)`;

  const tbody = $("#storage-table tbody");
  tbody.innerHTML = "";
  if (!d.runs.length) {
    tbody.innerHTML = '<tr><td colspan="6" class="hint">No fine-tune runs on disk yet.</td></tr>';
  }
  for (const r of d.runs) {
    const bestStep = r.best_models.length ? Math.max(...r.best_models.map(b => b.step)) : "—";
    const tr = document.createElement("tr");
    const badge = r.registered ? '<span class="badge reg">registered</span>' : "";
    tr.innerHTML = `
      <td>${esc(r.voice)}</td>
      <td title="${esc(r.dir)}">${esc(r.dir)}</td>
      <td class="size">${fmtBytes(r.bytes)}</td>
      <td>${bestStep}</td>
      <td>${badge}</td>
      <td><button class="del" data-delrun="${esc(r.voice)}::${esc(r.dir)}" title="delete this run">✕</button></td>`;
    tbody.appendChild(tr);
  }
  if (activeVoice) $("#cleanup-voice-label").textContent = activeVoice;
}

$("#storage-table").addEventListener("click", async (e) => {
  const btn = e.target.closest("[data-delrun]");
  if (!btn) return;
  const [voice, dir] = btn.dataset.delrun.split("::");
  if (!confirm(`Delete run '${dir}' for voice '${voice}'? This cannot be undone.`)) return;
  try {
    const res = await (await api("/api/storage/clean", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ voice, action: "run", run: dir }),
    })).json();
    setStatus($("#cleanup-status"), `✔ Freed ${fmtBytes(res.freed_bytes)} — deleted ${res.deleted.length} item(s).`, "ok");
    loadStorage(); if (activeVoice === voice) { loadCheckpoints(voice); loadVoices(); }
  } catch (err) { setStatus($("#cleanup-status"), `Cleanup failed: ${err.message}`, "err"); }
});

$("#clean-keep-reg").addEventListener("click", async () => {
  const voice = activeVoice; if (!voice) return setStatus($("#cleanup-status"), "No voice selected.", "err");
  if (!confirm(`For '${voice}': delete all runs EXCEPT the registered one?`)) return;
  try {
    const res = await (await api("/api/storage/clean", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ voice, action: "all-but-registered" }) })).json();
    setStatus($("#cleanup-status"), `✔ Freed ${fmtBytes(res.freed_bytes)} — kept the registered checkpoint.`, "ok");
    loadStorage(); loadCheckpoints(voice); loadVoices();
  } catch (err) { setStatus($("#cleanup-status"), `Cleanup failed: ${err.message}`, "err"); }
});

$("#clean-reset").addEventListener("click", async () => {
  const voice = activeVoice; if (!voice) return setStatus($("#cleanup-status"), "No voice selected.", "err");
  if (!confirm(`RESET '${voice}': delete ALL fine-tune runs + dataset and clear the checkpoint? Synthesis falls back to zero-shot.`)) return;
  try {
    const res = await (await api("/api/storage/clean", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ voice, action: "reset" }) })).json();
    setStatus($("#cleanup-status"), `✔ Reset '${voice}' — freed ${fmtBytes(res.freed_bytes)}, registration cleared.`, "ok");
    loadStorage(); loadCheckpoints(voice); loadVoices();
  } catch (err) { setStatus($("#cleanup-status"), `Cleanup failed: ${err.message}`, "err"); }
});

// ---------------------------------------------------------------- A/B compare --
$("#ab-btn").addEventListener("click", async () => {
  const text = $("#text").value.trim();
  if (!text) return setStatus($("#ab-status"), "Type some text first.", "err");
  if (!activeVoice) return setStatus($("#ab-status"), "No voice selected.", "err");

  const btn = $("#ab-btn");
  btn.disabled = true;
  $("#ab-results").innerHTML = "";

  async function gen(mode) {
    const t0 = performance.now();
    const res = await api("/api/synthesize", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        voice: activeVoice, text, emotion: selectedEmotion,
        style: $("#style").value || null, language: $("#syn-lang").value, mode, ...genParams(),
      }),
    });
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    let meta = {};
    try { const b64 = res.headers.get("X-Synthesis-Meta"); if (b64) meta = JSON.parse(atob(b64)); } catch {}
    return { url, meta, secs: ((performance.now() - t0) / 1000).toFixed(1) };
  }

  function render(label, r) {
    const div = document.createElement("div");
    div.className = "ab-item";
    const m = r.meta || {};
    div.innerHTML = `
      <div class="ab-label">${esc(label)}</div>
      <audio controls src="${r.url}" style="width:100%"></audio>
      <div class="meta">ref: ${esc(m.reference_file || "?")} (${esc(m.reference_emotion || "?")}) · ${esc(m.duration_s != null ? m.duration_s : "?")}s audio · ${r.secs}s elapsed</div>`;
    $("#ab-results").appendChild(div);
  }

  try {
    setStatus($("#ab-status"), "Generating A (zero-shot)… first run loads the model (~1 min)", "warn");
    render("A · zero-shot (base model)", await gen("zero-shot"));
    setStatus($("#ab-status"), "✔ A done. Generating B (fine-tuned)…", "warn");
    try {
      render("B · fine-tuned voice", await gen("finetuned"));
      setStatus($("#ab-status"), "✔ Both ready — listen and compare.", "ok");
    } catch (errB) {
      const div = document.createElement("div");
      div.className = "ab-item";
      div.innerHTML = `<div class="ab-label">B · fine-tuned voice</div><div class="status err">${esc(errB.message)}</div>`;
      $("#ab-results").appendChild(div);
      setStatus($("#ab-status"), "⚠ A done, but B failed (no registered checkpoint?).", "err");
    }
  } catch (errA) {
    setStatus($("#ab-status"), `A/B failed: ${errA.message}`, "err");
  } finally {
    btn.disabled = false;
  }
});

// ---------------------------------------------------------------- init --
function esc(s) {
  return String(s).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

(async function init() {
  try { EMOTIONS = await (await api("/api/emotions")).json(); } catch {}
  $("#up-emotion").innerHTML = EMOTIONS.map(e => `<option>${e}</option>`).join("");
  renderEmotionChips();
  await loadVoices();
  loadStorage().catch(() => {});
})();
