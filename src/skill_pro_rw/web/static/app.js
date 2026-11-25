(() => {
  const state = {
    snapshot: null,
    selectedEpisodeId: null,
    eventFilter: "all",
    liveEvents: [],
  };

  const el = {
    sseBadge: document.getElementById("sseBadge"),
    modeBadge: document.getElementById("modeBadge"),
    verifierBadge: document.getElementById("verifierBadge"),
    iterBadge: document.getElementById("iterBadge"),
    scenarioSelect: document.getElementById("scenarioSelect"),
    ablationSelect: document.getElementById("ablationSelect"),
    episodesInput: document.getElementById("episodesInput"),
    runBatchBtn: document.getElementById("runBatchBtn"),
    resetBtn: document.getElementById("resetBtn"),
    liveRunIndicator: document.getElementById("liveRunIndicator"),
    kpiReward: document.getElementById("kpiReward"),
    kpiResolution: document.getElementById("kpiResolution"),
    kpiReuse: document.getElementById("kpiReuse"),
    kpiMemTokens: document.getElementById("kpiMemTokens"),
    kpiSkillCount: document.getElementById("kpiSkillCount"),
    kpiSavings: document.getElementById("kpiSavings"),
    kpiTokensDetail: document.getElementById("kpiTokensDetail"),
    kpiGateRate: document.getElementById("kpiGateRate"),
    kpiGateCount: document.getElementById("kpiGateCount"),
    episodeList: document.getElementById("episodeList"),
    traceDetail: document.getElementById("traceDetail"),
    rewardChart: document.getElementById("rewardChart"),
    tokenChart: document.getElementById("tokenChart"),
    skillPoolList: document.getElementById("skillPoolList"),
    poolCapacityBadge: document.getElementById("poolCapacityBadge"),
    auditList: document.getElementById("auditList"),
    eventStream: document.getElementById("eventStream"),
    eventFilters: document.getElementById("eventFilters"),
    configModal: document.getElementById("configModal"),
    openConfigBtn: document.getElementById("openConfigBtn"),
    closeConfigBtn: document.getElementById("closeConfigBtn"),
    configForm: document.getElementById("configForm"),
    cfgMode: document.getElementById("cfgMode"),
    cfgVerifierMode: document.getElementById("cfgVerifierMode"),
    cfgExecUrl: document.getElementById("cfgExecUrl"),
    cfgExecModel: document.getElementById("cfgExecModel"),
    cfgProxyUrl: document.getElementById("cfgProxyUrl"),
    cfgProxyModel: document.getElementById("cfgProxyModel"),
    cfgKeyNotice: document.getElementById("cfgKeyNotice"),
  };

  function node(tag, className, text) {
    const d = document.createElement(tag);
    if (className) d.className = className;
    if (text !== undefined) d.textContent = text;
    return d;
  }

  async function api(path, options = {}) {
    const resp = await fetch(path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
    if (!resp.ok) {
      const errBody = await resp.json().catch(() => ({}));
      throw new Error(errBody.detail || `HTTP ${resp.status}`);
    }
    return resp.json();
  }

  async function refreshState() {
    const data = await api("/api/state");
    state.snapshot = data;
    if (!state.liveEvents.length && Array.isArray(data.events)) {
      state.liveEvents = data.events;
    } else if (Array.isArray(data.events)) {
      state.liveEvents = data.events;
    }
    renderAll();
  }

  function renderAll() {
    const snap = state.snapshot;
    if (!snap) return;

    renderHeader(snap);
    renderScenarios(snap);
    renderKPIs(snap);
    renderEpisodes(snap);
    renderCharts(snap);
    renderSkillPool(snap);
    renderAudits(snap);
    renderEvents();
  }

  function renderHeader(snap) {
    const cfg = snap.config;
    el.modeBadge.textContent = `Mode: ${cfg.mode}`;
    el.verifierBadge.textContent = `Verifier: ${cfg.proxy_verifier_mode}`;
    el.iterBadge.textContent = `Iter #${snap.iteration}`;
    el.runBatchBtn.disabled = snap.running;
    el.resetBtn.disabled = snap.running;
    el.liveRunIndicator.textContent = snap.running ? "Running Batch..." : "Idle";
    el.liveRunIndicator.className = snap.running ? "badge badge-warn" : "badge badge-muted";

    el.cfgMode.value = cfg.mode;
    el.cfgVerifierMode.value = cfg.proxy_verifier_mode;
    el.cfgExecUrl.value = cfg.executor_base_url;
    el.cfgExecModel.value = cfg.executor_model;
    el.cfgProxyUrl.value = cfg.proxy_verifier_url;
    el.cfgProxyModel.value = cfg.proxy_verifier_model;
    el.cfgKeyNotice.textContent = cfg.executor_key_configured
      ? "EXECUTOR_API_KEY detected in server environment (kept strictly server-side)."
      : "EXECUTOR_API_KEY not set in environment; Simulator Mode works out-of-the-box.";
  }

  function renderScenarios(snap) {
    if (el.scenarioSelect.options.length > 1) return;
    for (const sc of snap.scenarios) {
      const opt = document.createElement("option");
      opt.value = sc.scenario_id;
      opt.textContent = `[${sc.severity}] ${sc.title}`;
      el.scenarioSelect.appendChild(opt);
    }
  }

  function renderKPIs(snap) {
    const m = snap.metrics;
    el.kpiReward.textContent = m.mean_reward.toFixed(3);
    el.kpiResolution.textContent = `${(m.resolution_rate * 100).toFixed(1)}% resolved across ${m.total_episodes} episodes`;
    el.kpiReuse.textContent = `${(m.skill_reuse_rate * 100).toFixed(1)}%`;
    el.kpiMemTokens.textContent = `${m.total_active_tokens} tok`;
    el.kpiSkillCount.textContent = `${m.active_skill_count} active skills (~${m.avg_tokens_per_skill} tok/skill)`;
    el.kpiSavings.textContent = `${m.token_savings_pct.toFixed(1)}%`;
    el.kpiTokensDetail.textContent = `${m.avg_prompt_tokens} vs ${m.avg_episodic_tokens} episodic tok/ep`;
    el.kpiGateRate.textContent = `${(m.ppo_gate_pass_rate * 100).toFixed(1)}%`;
    el.kpiGateCount.textContent = `${m.ppo_gate_attempts} L_CLIP verification checks`;
  }

  function renderEpisodes(snap) {
    const exps = snap.experiences || [];
    el.episodeList.innerHTML = "";
    if (!exps.length) {
      el.episodeList.appendChild(node("div", "empty-state", "Run a triage batch to watch Skill-MDP rollouts step-by-step."));
      el.traceDetail.innerHTML = "";
      el.traceDetail.appendChild(node("div", "empty-state", "No episode selected yet."));
      return;
    }

    if (!state.selectedEpisodeId || !exps.some((e) => e.episode_id === state.selectedEpisodeId)) {
      state.selectedEpisodeId = exps[0].episode_id;
    }

    for (const ep of exps) {
      const row = node("div", `episode-item ${ep.episode_id === state.selectedEpisodeId ? "selected" : ""}`);
      row.addEventListener("click", () => {
        state.selectedEpisodeId = ep.episode_id;
        renderEpisodes(state.snapshot);
      });

      const top = node("div", "ep-top");
      top.appendChild(node("span", "", ep.scenario_id));
      const badge = node(
        "span",
        `badge ${ep.resolved ? "badge-ok" : "badge-bad"}`,
        `${ep.reward.toFixed(2)}`
      );
      top.appendChild(badge);
      row.appendChild(top);

      const skillsText = ep.skills_used.length ? ep.skills_used.join(", ") : "None (stateless)";
      row.appendChild(
        node("div", "ep-meta", `${ep.ablation_mode} | ${ep.step_count} steps | ${ep.prompt_tokens} tok | ${skillsText}`)
      );
      el.episodeList.appendChild(row);
    }

    const activeEp = exps.find((e) => e.episode_id === state.selectedEpisodeId);
    renderTraceDetail(activeEp);
  }

  function renderTraceDetail(ep) {
    el.traceDetail.innerHTML = "";
    if (!ep) return;

    const header = node("div", "panel-header");
    header.appendChild(node("h3", "", `${ep.scenario_id} (${ep.episode_id})`));
    header.appendChild(
      node(
        "span",
        `badge ${ep.resolved ? "badge-ok" : "badge-bad"}`,
        ep.resolved ? `RESOLVED (${ep.reward.toFixed(2)})` : `UNRESOLVED (${ep.reward.toFixed(2)})`
      )
    );
    el.traceDetail.appendChild(header);

    for (const tr of ep.transitions) {
      const card = node("div", "step-card");
      const head = node("div", "step-head");
      head.appendChild(node("span", "", `Step ${tr.step_index} -> ${tr.action.tool}`));
      const skillBadge = node(
        "span",
        `badge ${tr.skill_name ? "badge-info" : "badge-muted"}`,
        tr.skill_name ? `${tr.skill_name}${tr.skill_done ? " [DONE]" : ""}` : "No Skill"
      );
      head.appendChild(skillBadge);
      card.appendChild(head);

      card.appendChild(node("div", "code-line", `Args: ${JSON.stringify(tr.action.args)} | Prompt: ${tr.prompt_tokens} tok`));
      card.appendChild(node("div", "code-line", `Obs: ${tr.observation_summary}`));
      el.traceDetail.appendChild(card);
    }
  }

  function renderCharts(snap) {
    const exps = [...(snap.experiences || [])].reverse().slice(-20);
    renderRewardChart(exps);
    renderTokenChart(exps);
  }

  function renderRewardChart(exps) {
    const svg = el.rewardChart;
    svg.innerHTML = "";
    const w = 420;
    const h = 150;
    const pad = 18;

    const ns = "http://www.w3.org/2000/svg";
    const grid = document.createElementNS(ns, "line");
    grid.setAttribute("x1", pad);
    grid.setAttribute("x2", w - pad);
    grid.setAttribute("y1", h - pad);
    grid.setAttribute("y2", h - pad);
    grid.setAttribute("stroke", "#233049");
    svg.appendChild(grid);

    if (!exps.length) return;
    const barW = Math.max(8, (w - pad * 2) / exps.length - 4);
    exps.forEach((ep, idx) => {
      const val = Math.max(0, Math.min(1, ep.reward));
      const bh = Math.max(4, val * (h - pad * 2));
      const x = pad + idx * ((w - pad * 2) / exps.length);
      const y = h - pad - bh;

      const rect = document.createElementNS(ns, "rect");
      rect.setAttribute("x", x.toFixed(1));
      rect.setAttribute("y", y.toFixed(1));
      rect.setAttribute("width", barW.toFixed(1));
      rect.setAttribute("height", bh.toFixed(1));
      rect.setAttribute("rx", "3");
      let color = "#10b981";
      if (ep.ablation_mode === "no_skill") color = "#f59e0b";
      else if (ep.ablation_mode === "no_ppo_gate") color = "#f43f5e";
      else if (ep.reward < 0.7) color = "#38bdf8";
      rect.setAttribute("fill", color);
      svg.appendChild(rect);
    });
  }

  function renderTokenChart(exps) {
    const svg = el.tokenChart;
    svg.innerHTML = "";
    const w = 420;
    const h = 150;
    const pad = 18;
    const ns = "http://www.w3.org/2000/svg";

    if (!exps.length) return;
    const maxTok = Math.max(
      100,
      ...exps.map((e) => Math.max(e.prompt_tokens || 0, e.episodic_baseline_tokens || 0))
    );
    const slot = (w - pad * 2) / exps.length;
    const subW = Math.max(4, slot * 0.36);

    exps.forEach((ep, idx) => {
      const baseX = pad + idx * slot;
      const hEpisodic = ((ep.episodic_baseline_tokens || 0) / maxTok) * (h - pad * 2);
      const hActual = ((ep.prompt_tokens || 0) / maxTok) * (h - pad * 2);

      const r1 = document.createElementNS(ns, "rect");
      r1.setAttribute("x", baseX.toFixed(1));
      r1.setAttribute("y", (h - pad - hEpisodic).toFixed(1));
      r1.setAttribute("width", subW.toFixed(1));
      r1.setAttribute("height", Math.max(3, hEpisodic).toFixed(1));
      r1.setAttribute("fill", "#475569");
      r1.setAttribute("rx", "2");
      svg.appendChild(r1);

      const r2 = document.createElementNS(ns, "rect");
      r2.setAttribute("x", (baseX + subW + 2).toFixed(1));
      r2.setAttribute("y", (h - pad - hActual).toFixed(1));
      r2.setAttribute("width", subW.toFixed(1));
      r2.setAttribute("height", Math.max(3, hActual).toFixed(1));
      r2.setAttribute("fill", "#38bdf8");
      r2.setAttribute("rx", "2");
      svg.appendChild(r2);
    });
  }

  function renderSkillPool(snap) {
    const skills = snap.skills || [];
    const activeCount = skills.filter((s) => s.status !== "archived").length;
    el.poolCapacityBadge.textContent = `${activeCount} / ${snap.config.pool_size}`;
    el.skillPoolList.innerHTML = "";

    const order = { active: 0, candidate: 1, archived: 2 };
    const sorted = [...skills].sort((a, b) => (order[a.status] ?? 9) - (order[b.status] ?? 9) || b.version - a.version);

    for (const sk of sorted) {
      const card = node("div", "skill-card");
      const head = node("div", "skill-head");
      head.appendChild(node("strong", "", `${sk.name} v${sk.version}`));

      const statusBadge = node(
        "span",
        `badge ${sk.status === "active" ? "badge-ok" : sk.status === "candidate" ? "badge-warn" : "badge-muted"}`,
        sk.status.toUpperCase()
      );
      head.appendChild(statusBadge);
      card.appendChild(head);

      card.appendChild(
        node(
          "div",
          "skill-metrics",
          `freq=${sk.frequency} | avg_gain=${sk.avg_gain.toFixed(3)} | lcb=${sk.lcb_score.toFixed(3)} | mat=${sk.maturity} | ~${sk.estimated_tokens} tok`
        )
      );

      const body = node("div", "skill-body");
      body.appendChild(node("p", "", `I (When): ${sk.initiation}`));
      const ul = document.createElement("ul");
      for (const step of sk.policy) {
        ul.appendChild(node("li", "", step));
      }
      body.appendChild(ul);
      body.appendChild(node("p", "", `B (Stop): ${sk.termination}`));
      card.appendChild(body);

      const actions = node("div", "skill-actions");
      if (sk.status !== "active") {
        const btn = node("button", "btn btn-secondary btn-xs", "Activate");
        btn.addEventListener("click", () => updateSkillStatus(sk.id, "active"));
        actions.appendChild(btn);
      }
      if (sk.status !== "archived") {
        const btn = node("button", "btn btn-danger btn-xs", "Archive");
        btn.addEventListener("click", () => updateSkillStatus(sk.id, "archived"));
        actions.appendChild(btn);
      }
      card.appendChild(actions);
      el.skillPoolList.appendChild(card);
    }
  }

  function renderAudits(snap) {
    const audits = snap.audit_logs || [];
    el.auditList.innerHTML = "";
    if (!audits.length) {
      el.auditList.appendChild(
        node("div", "empty-state", "No PPO Gate evolution cycles triggered yet. Run at least 4 episodes.")
      );
      return;
    }

    for (const aud of audits) {
      const card = node("div", "audit-card");
      const head = node("div", "audit-head");
      const candLabel = aud.candidate_skill ? `${aud.candidate_skill.name}@v${aud.candidate_skill.version}` : "None";
      head.appendChild(node("strong", "", `Iter ${aud.iteration}: ${aud.parent_skill_name} -> ${candLabel}`));

      const verdict = node(
        "span",
        `badge ${aud.accepted ? "badge-ok" : "badge-bad"}`,
        aud.accepted ? "PPO GATE ACCEPTED" : "PPO GATE REJECTED"
      );
      head.appendChild(verdict);
      card.appendChild(head);

      const g = aud.gate_eval;
      card.appendChild(
        node(
          "div",
          "lclip-meter",
          `Delta_L_CLIP = ${(g.improvement ?? g.l_clip).toFixed(5)} | L_CLIP = ${g.l_clip.toFixed(5)} (margin ${g.margin}) | ratio=[${g.min_ratio.toFixed(2)}, ${g.max_ratio.toFixed(2)}] | clipped=${(g.clipped_fraction * 100).toFixed(1)}% | verifier=${g.verifier_mode}`
        )
      );

      for (const grad of aud.gradients.slice(0, 2)) {
        const deltaText = grad.policy_delta || grad.diagnosis;
        card.appendChild(node("div", "code-line", `Semantic Gradient (${grad.episode_id}): ${deltaText}`));
      }
      el.auditList.appendChild(card);
    }
  }

  function renderEvents() {
    el.eventStream.innerHTML = "";
    const filtered = state.liveEvents.filter(
      (ev) => state.eventFilter === "all" || ev.event_type === state.eventFilter
    );
    if (!filtered.length) {
      el.eventStream.appendChild(node("div", "empty-state", "Waiting for backend events..."));
      return;
    }
    for (const ev of filtered.slice(0, 80)) {
      const row = node("div", "event-row");
      const time = (ev.created_at || "").slice(11, 19);
      row.appendChild(node("span", "muted", time));
      row.appendChild(node("span", "event-type", ev.event_type));
      row.appendChild(node("span", "", JSON.stringify(ev.payload)));
      el.eventStream.appendChild(row);
    }
  }

  async function updateSkillStatus(skillId, status) {
    try {
      await api(`/api/skills/${encodeURIComponent(skillId)}/status`, {
        method: "POST",
        body: JSON.stringify({ status }),
      });
      await refreshState();
    } catch (err) {
      alert(`Failed to update skill status: ${err.message}`);
    }
  }

  function connectSSE() {
    const source = new EventSource("/api/events");
    source.addEventListener("connected", () => {
      el.sseBadge.className = "pill pill-ok";
      el.sseBadge.innerHTML = '<span class="dot"></span>SSE Live';
    });
    source.addEventListener("backend_event", (msg) => {
      try {
        const ev = JSON.parse(msg.data);
        state.liveEvents.unshift(ev);
        if (state.liveEvents.length > 150) state.liveEvents.length = 150;
        renderEvents();
        if (
          ev.event_type === "episode_finished" ||
          ev.event_type === "ppo_gate_decision" ||
          ev.event_type === "batch_completed" ||
          ev.event_type === "batch_failed" ||
          ev.event_type === "pool_maintained" ||
          ev.event_type === "sandbox_reset" ||
          ev.event_type === "config_updated" ||
          ev.event_type === "skill_status_changed"
        ) {
          refreshState().catch(() => {});
        }
      } catch (_) {}
    });
    source.onerror = () => {
      el.sseBadge.className = "pill pill-warn";
      el.sseBadge.innerHTML = '<span class="dot"></span>Reconnecting SSE...';
    };
  }

  // Event listeners
  el.runBatchBtn.addEventListener("click", async () => {
    try {
      el.runBatchBtn.disabled = true;
      await api("/api/run-batch", {
        method: "POST",
        body: JSON.stringify({
          num_episodes: Number(el.episodesInput.value),
          scenario_id: el.scenarioSelect.value || null,
          ablation_mode: el.ablationSelect.value,
          evolve_every: 4,
        }),
      });
      await refreshState();
    } catch (err) {
      alert(`Run failed: ${err.message}`);
      el.runBatchBtn.disabled = false;
    }
  });

  el.resetBtn.addEventListener("click", async () => {
    try {
      await api("/api/reset", { method: "POST" });
      state.liveEvents = [];
      state.selectedEpisodeId = null;
      await refreshState();
    } catch (err) {
      alert(`Reset failed: ${err.message}`);
    }
  });

  el.eventFilters.addEventListener("click", (e) => {
    const btn = e.target.closest(".filter-pill");
    if (!btn) return;
    el.eventFilters.querySelectorAll(".filter-pill").forEach((b) => b.classList.remove("active"));
    btn.classList.add("active");
    state.eventFilter = btn.dataset.filter || "all";
    renderEvents();
  });

  el.openConfigBtn.addEventListener("click", () => el.configModal.classList.remove("hidden"));
  el.closeConfigBtn.addEventListener("click", () => el.configModal.classList.add("hidden"));

  el.configForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    try {
      await api("/api/config", {
        method: "POST",
        body: JSON.stringify({
          mode: el.cfgMode.value,
          proxy_verifier_mode: el.cfgVerifierMode.value,
          executor_base_url: el.cfgExecUrl.value.trim(),
          executor_model: el.cfgExecModel.value.trim(),
          proxy_verifier_url: el.cfgProxyUrl.value.trim(),
          proxy_verifier_model: el.cfgProxyModel.value.trim(),
        }),
      });
      el.configModal.classList.add("hidden");
      await refreshState();
    } catch (err) {
      alert(`Config error: ${err.message}`);
    }
  });

  refreshState();
  connectSSE();
})();
