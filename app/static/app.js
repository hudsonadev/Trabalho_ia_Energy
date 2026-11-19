// EnergyGreen — lógica do dashboard: chamadas à API (/api/*), estado e interações.
// Debounce (~150 ms) nos controles; custo e alertas reutilizam a previsão já em tela,
// sem refazer /api/prever.
"use strict";

const $ = (s) => document.querySelector(s);
const $$ = (s) => Array.from(document.querySelectorAll(s));
const HORA_MS = 3600 * 1000;
const NOMES = { lstm: "LSTM", gru: "GRU", baseline: "Baseline" };

const E = {
  resumo: null,       // resposta de /api/upload ou /api/exemplo
  refs: [],           // instantes selecionáveis na máquina do tempo (ms, UTC "ingênuo")
  idx: 0,
  modelo: "lstm",
  revelado: false,
  prev: null,         // resposta de /api/prever
  custo: null,        // resposta de /api/custo
  alertas: null,      // resposta de /api/alertas
  limite: 2.5,
  tocando: false,
  ordem: { col: "hora", dir: 1 },
  desempenho: null,
  req: 0,
};

// ------------------------------------------------------------------ utilidades
const nf = (d) => new Intl.NumberFormat("pt-BR", { minimumFractionDigits: d, maximumFractionDigits: d });
const fmt = (v, d = 2) => (v === null || v === undefined || Number.isNaN(v) ? "—" : nf(d).format(v));
const brl = (v) => (v === null || v === undefined ? "—" : new Intl.NumberFormat("pt-BR", { style: "currency", currency: "BRL" }).format(v));
const pct = (v, d = 1) => `${fmt(v * 100, d)}%`;

// As datas da série não têm fuso; tratamos tudo como UTC para não sofrer com horário de verão.
const paraMs = (iso) => Date.parse(iso.length === 19 ? iso + "Z" : iso);
const paraIso = (ms) => new Date(ms).toISOString().slice(0, 19);
const hora = (iso) => new Date(paraMs(iso)).getUTCHours();
function rotuloData(ms, comDia = true) {
  const d = new Date(ms);
  const op = { day: "2-digit", month: "2-digit", year: "numeric", timeZone: "UTC" };
  if (comDia) op.weekday = "short";
  return d.toLocaleDateString("pt-BR", op);
}
const rotuloHora = (iso) => `${String(hora(iso)).padStart(2, "0")}h`;
const soma = (a) => a.reduce((s, v) => s + (v ?? 0), 0);
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function debounce(fn, ms = 150) {
  let t;
  return (...args) => { clearTimeout(t); t = setTimeout(() => fn(...args), ms); };
}

function cor(nome) {
  return getComputedStyle(document.documentElement).getPropertyValue(nome).trim();
}

// ------------------------------------------------------------------ feedback
let pendentes = 0;
function ocupado(delta) {
  pendentes = Math.max(0, pendentes + delta);
  $("#carregando").hidden = pendentes === 0;
  document.body.classList.toggle("ocupado", pendentes > 0);
}

function toast(msg, tipo = "info", titulo = null) {
  const el = document.createElement("div");
  el.className = `toast ${tipo}`;
  if (titulo) {
    const t = document.createElement("strong");
    t.textContent = titulo;
    el.appendChild(t);
  }
  el.appendChild(document.createTextNode(msg));
  $("#toasts").appendChild(el);
  setTimeout(() => el.remove(), tipo === "erro" ? 7000 : 3500);
}

async function api(caminho, opcoes = {}) {
  ocupado(+1);
  try {
    const r = await fetch(caminho, opcoes);
    let corpo = null;
    try { corpo = await r.json(); } catch { /* sem corpo JSON */ }
    if (!r.ok) {
      const msg = corpo && corpo.detail ? corpo.detail : `Erro ${r.status} ao chamar ${caminho}.`;
      const erro = new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
      erro.status = r.status;
      throw erro;
    }
    return corpo;
  } catch (e) {
    if (e instanceof TypeError) throw new Error("Não foi possível falar com o servidor. Ele está rodando?");
    throw e;
  } finally {
    ocupado(-1);
  }
}
const postJson = (caminho, dados) =>
  api(caminho, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(dados) });

// ------------------------------------------------------------------ Plotly
const CONFIG_PLOTLY = { displayModeBar: false, responsive: true };

function eixo(extra = {}) {
  return Object.assign({
    gridcolor: cor("--grade"), linecolor: cor("--eixo"), zeroline: false,
    tickfont: { color: cor("--texto-mudo"), size: 11 }, automargin: true,
  }, extra);
}

function layoutBase(extra = {}) {
  return Object.assign({
    paper_bgcolor: "rgba(0,0,0,0)", plot_bgcolor: "rgba(0,0,0,0)",
    font: { family: getComputedStyle(document.body).fontFamily, color: cor("--texto-2"), size: 12 },
    margin: { l: 50, r: 14, t: 36, b: 40 },
    separators: ",.",
    hovermode: "x unified",
    hoverlabel: { bgcolor: cor("--superficie"), bordercolor: cor("--eixo"), font: { color: cor("--texto"), size: 12 } },
    legend: { orientation: "h", x: 0, y: 1.02, yanchor: "bottom", font: { size: 12, color: cor("--texto-2") } },
    xaxis: eixo(), yaxis: eixo(),
  }, extra);
}

const corModelo = (m) => cor(m === "gru" ? "--serie-gru" : "--serie-lstm");

// ------------------------------------------------------------------ entrada de dados
async function usarExemplo() {
  try {
    const r = await api("/api/exemplo");
    await serieCarregada(r, "Dados de exemplo carregados (período de teste do dataset UCI).");
  } catch (e) { toast(e.message, "erro", "Não foi possível carregar o exemplo"); }
}

async function enviarArquivo(arquivo) {
  if (!arquivo) return;
  const fd = new FormData();
  fd.append("arquivo", arquivo);
  try {
    const r = await api("/api/upload", { method: "POST", body: fd });
    await serieCarregada(r, `Arquivo "${arquivo.name}" carregado.`);
  } catch (e) { toast(e.message, "erro", "Arquivo recusado"); }
}

function montarResumo(r) {
  const el = $("#resumo-dados");
  let txt = `Série: ${rotuloData(paraMs(r.inicio), false)} → ${rotuloData(paraMs(r.fim), false)} · ${fmt(r.horas, 0)} h`;
  let dica = `${fmt(r.faltantes_preenchidos, 0)} h preenchidas · média ${fmt(r.media_kw)} kW · pico ${fmt(r.pico_kw)} kW`;
  if (r.colunas_estimadas && r.colunas_estimadas.length) {
    txt += ` · ${r.colunas_estimadas.length} medições estimadas`;
    dica += " · colunas ausentes no CSV, estimadas a partir do treino: " + r.colunas_estimadas.join(", ");
  }
  el.textContent = txt;
  el.title = dica;
  el.hidden = false;
}

function construirRefs(r) {
  const inicio = paraMs(r.inicio), fim = paraMs(r.fim);
  const primeiro = inicio + 168 * HORA_MS;
  const futuro = fim + HORA_MS;
  const refs = [];
  const dia = 24 * HORA_MS;
  let m = Math.ceil(primeiro / dia) * dia;
  if (m !== primeiro) refs.push(primeiro);
  for (; m <= fim; m += dia) refs.push(m);
  if (refs[refs.length - 1] !== futuro) refs.push(futuro);
  E.refs = refs;
  // posição inicial: último dia com as 24 h reais disponíveis (permite "Revelar o real")
  let idx = refs.length - 1;
  for (let i = refs.length - 1; i >= 0; i--) { if (refs[i] + 23 * HORA_MS <= fim) { idx = i; break; } }
  E.idx = idx;
  const s = $("#slider");
  s.max = String(refs.length - 1);
  s.value = String(idx);
  const di = $("#data-ref");
  di.min = paraIso(refs[0]).slice(0, 10);
  di.max = paraIso(futuro).slice(0, 10);
}

async function serieCarregada(r, msg) {
  E.resumo = r;
  E.prev = null;
  pararReproducao();
  montarResumo(r);
  construirRefs(r);
  $("#entrada").hidden = true;
  $("#btn-trocar").hidden = false;
  $("#painel").hidden = false;
  toast(msg, "info");
  await prever();
}

// ------------------------------------------------------------------ máquina do tempo
function refAtual() { return E.refs[E.idx]; }
function ehFuturo() { return E.resumo && refAtual() > paraMs(E.resumo.fim); }

function atualizarControlesTempo() {
  const ms = refAtual();
  $("#slider").value = String(E.idx);
  $("#data-ref").value = paraIso(ms).slice(0, 10);
  const rot = $("#rotulo-ref");
  rot.replaceChildren();
  if (ehFuturo()) {
    const s = document.createElement("span");
    s.className = "futuro";
    s.textContent = `Futuro · ${rotuloData(ms)} · ${rotuloHora(paraIso(ms))}`;
    rot.appendChild(s);
  } else {
    rot.textContent = `${rotuloData(ms)} · ${rotuloHora(paraIso(ms))}`;
  }
  $("#btn-ant").disabled = E.idx <= 0;
  $("#btn-prox").disabled = E.idx >= E.refs.length - 1;
}

async function prever() {
  if (!E.resumo) return;
  atualizarControlesTempo();
  const id = ++E.req;
  try {
    const r = await postJson("/api/prever", { modelo: E.modelo, data_referencia: paraIso(refAtual()) });
    if (id !== E.req) return; // resposta antiga: o usuário já mudou de dia/modelo
    E.prev = r;
    atualizarBotaoRevelar();
    desenharTempo();
    desenharMetricasDia();
    await Promise.all([calcularCusto(), calcularAlertas()]);
    atualizarCartoes();
    desenharTabela();
  } catch (e) {
    if (id === E.req) toast(e.message, "erro", e.status === 503 ? "Modelos indisponíveis" : "Falha na previsão");
    pararReproducao();
  }
}
const preverDebounced = debounce(prever, 150);

function irPara(idx) {
  E.idx = Math.max(0, Math.min(E.refs.length - 1, idx));
  atualizarControlesTempo();
  preverDebounced();
}

function realVisivel() {
  return E.revelado && E.prev && E.prev.real ? E.prev.real : null;
}

function atualizarBotaoRevelar() {
  const b = $("#btn-revelar");
  const temReal = !!(E.prev && E.prev.real);
  b.disabled = !temReal;
  b.title = temReal ? "" : "Período futuro: ainda não há consumo real para comparar.";
  b.setAttribute("aria-pressed", String(E.revelado && temReal));
  b.textContent = E.revelado && temReal ? "Ocultar o consumo real" : "Revelar o consumo real";
}

function desenharTempo() {
  const p = E.prev;
  if (!p) return;
  const ref = paraMs(p.horas[0]);
  const fimPrev = ref + 23 * HORA_MS;
  const real = realVisivel();
  const hover = "%{y:,.2f} kW";
  const traces = [
    { x: p.historico.datetime, y: p.historico.kw, name: "Histórico", mode: "lines",
      line: { color: cor("--serie-hist"), width: 1.5 }, hovertemplate: hover },
    { x: p.horas, y: p.baseline, name: "Baseline (dia anterior)", mode: "lines",
      line: { color: cor("--serie-baseline"), width: 2, dash: "dot" }, hovertemplate: hover },
    { x: p.horas, y: p.previsto, name: `Previsão ${NOMES[p.modelo]}`, mode: "lines",
      line: { color: corModelo(p.modelo), width: 2.5 }, hovertemplate: hover },
  ];
  if (real) {
    traces.push({ x: p.horas, y: real, name: "Real", mode: "lines+markers",
      line: { color: cor("--serie-real"), width: 2 }, marker: { size: 5 }, hovertemplate: hover });
  }
  const layout = layoutBase({
    xaxis: eixo({ type: "date", tickformat: "%d/%m<br>%Hh", hoverformat: "%d/%m/%Y %Hh" }),
    yaxis: eixo({ title: { text: "kW (média horária)", font: { size: 12 } }, rangemode: "tozero" }),
    shapes: [
      { type: "rect", xref: "x", yref: "paper", x0: paraIso(ref - HORA_MS / 2), x1: paraIso(fimPrev + HORA_MS / 2), y0: 0, y1: 1,
        fillcolor: cor("--foco"), opacity: 0.08, line: { width: 0 }, layer: "below" },
      { type: "line", xref: "x", yref: "paper", x0: paraIso(ref - HORA_MS / 2), x1: paraIso(ref - HORA_MS / 2), y0: 0, y1: 1,
        line: { color: cor("--foco"), width: 1, dash: "dash" } },
    ],
    annotations: [
      { xref: "x", yref: "paper", x: paraIso(ref + 11.5 * HORA_MS), y: 0.98, yanchor: "top", showarrow: false,
        text: "próximas 24 h", font: { color: cor("--foco"), size: 12 } },
    ],
  });
  Plotly.react("g-tempo", traces, layout, CONFIG_PLOTLY);
}

function desenharMetricasDia() {
  const el = $("#metricas-dia");
  el.replaceChildren();
  const p = E.prev;
  if (!p) return;
  if (!p.real) {
    const d = document.createElement("p");
    d.className = "rot";
    d.textContent = "Período futuro: ainda não há consumo real para comparar.";
    el.appendChild(d);
    return;
  }
  if (!E.revelado) return;
  const m = p.metricas, b = m.baseline;
  const vence = m.mae < b.mae;
  const dif = (b.mae - m.mae) / b.mae;
  const veredito = document.createElement("p");
  veredito.className = "veredito";
  veredito.textContent = vence
    ? `${NOMES[p.modelo]} errou ${pct(dif, 0)} menos que o baseline neste dia.`
    : `Neste dia o baseline foi melhor: ${NOMES[p.modelo]} errou ${pct(-dif, 0)} a mais.`;
  if (m.horas_com_real < 24) veredito.textContent += ` Apenas ${m.horas_com_real} h com dado real.`;

  const tab = document.createElement("table");
  tab.className = "mini";
  const cab = tab.createTHead().insertRow();
  for (const t of ["", "MAE", "RMSE", "WAPE"]) { const th = document.createElement("th"); th.textContent = t; cab.appendChild(th); }
  const corpo = tab.createTBody();
  const linha = (nome, x, corMarca, ganhou) => {
    const tr = corpo.insertRow();
    if (ganhou) tr.className = "vence";
    const mk = document.createElement("span");
    mk.className = "marcador";
    mk.style.background = corMarca;
    tr.insertCell().append(mk, ` ${nome}`);
    tr.insertCell().textContent = `${fmt(x.mae)} kW`;
    tr.insertCell().textContent = `${fmt(x.rmse)} kW`;
    tr.insertCell().textContent = pct(x.wape, 0);
  };
  linha(NOMES[p.modelo], m, corModelo(p.modelo), vence);
  linha("Baseline", b, cor("--serie-baseline"), !vence);
  el.append(veredito, tab);
}

// ------------------------------------------------------------------ reprodução
function pararReproducao() {
  E.tocando = false;
  const b = $("#btn-play");
  b.querySelector("span").textContent = "Reproduzir";
  b.setAttribute("aria-pressed", "false");
}

async function reproduzir() {
  if (E.tocando) { pararReproducao(); return; }
  if (E.idx >= E.refs.length - 1) E.idx = 0;
  E.tocando = true;
  const b = $("#btn-play");
  b.querySelector("span").textContent = "Pausar";
  b.setAttribute("aria-pressed", "true");
  while (E.tocando && E.idx < E.refs.length - 1) {
    E.idx += 1;
    await prever();
    await sleep(1100);
  }
  pararReproducao();
}

// ------------------------------------------------------------------ custo
function lerNumero(sel) {
  const el = $(sel);
  const v = parseFloat(String(el.value).replace(",", "."));
  const ok = Number.isFinite(v) && v >= 0;
  el.setAttribute("aria-invalid", String(!ok));
  return ok ? v : null;
}

function parametrosCusto() {
  const tarifa = lerNumero("#tarifa"), tarifaPonta = lerNumero("#tarifa-ponta");
  if (tarifa === null || tarifaPonta === null) return null;
  return {
    tarifa, tarifa_ponta: tarifaPonta,
    ponta_inicio: parseInt($("#ponta-inicio").value, 10),
    ponta_fim: parseInt($("#ponta-fim").value, 10),
    deslocar_pct: parseFloat($("#deslocar").value),
  };
}

async function calcularCusto() {
  if (!E.prev) return;
  const par = parametrosCusto();
  if (!par) return;
  try {
    E.custo = await postJson("/api/custo", { previsto: E.prev.previsto, horas: E.prev.horas, ...par });
    desenharCusto();
  } catch (e) { toast(e.message, "erro", "Falha no cálculo de custo"); }
}
const custoDebounced = debounce(async () => { await calcularCusto(); atualizarCartoes(); desenharTabela(); }, 150);

function desenharCusto() {
  const c = E.custo, p = E.prev;
  if (!c || !p) return;
  $("#k-total").textContent = brl(c.custo_total);
  $("#k-ponta").textContent = brl(c.custo_ponta);
  $("#k-ponta-kwh").textContent = `${fmt(c.kwh_ponta, 1)} kWh`;
  $("#k-desl").textContent = brl(c.custo_com_deslocamento);
  $("#k-eco").textContent = brl(c.economia);
  const fora = c.custo_hora.map((v, i) => (c.ponta[i] ? null : v));
  const ponta = c.custo_hora.map((v, i) => (c.ponta[i] ? v : null));
  const traces = [
    { type: "bar", x: p.horas, y: fora, name: "Fora de ponta", marker: { color: cor("--serie-fora"), cornerradius: 4 },
      hovertemplate: "R$ %{y:,.2f}" },
    { type: "bar", x: p.horas, y: ponta, name: "Ponta", marker: { color: cor("--serie-ponta"), cornerradius: 4 },
      hovertemplate: "R$ %{y:,.2f}" },
  ];
  if (parseFloat($("#deslocar").value) > 0) {
    traces.push({ type: "scatter", mode: "lines+markers", x: p.horas, y: c.custo_hora_deslocado, name: "Com deslocamento",
      line: { color: cor("--serie-desl"), width: 2, shape: "hvh" }, marker: { size: 6 }, hovertemplate: "R$ %{y:,.2f}" });
  }
  Plotly.react("g-custo", traces, layoutBase({
    barmode: "overlay", bargap: 0.15,
    xaxis: eixo({ type: "date", tickformat: "%Hh", hoverformat: "%d/%m %Hh" }),
    yaxis: eixo({ title: { text: "R$ por hora", font: { size: 12 } }, tickprefix: "R$ ", rangemode: "tozero" }),
  }), CONFIG_PLOTLY);
}

// ------------------------------------------------------------------ alertas
async function calcularAlertas() {
  if (!E.prev) return;
  const limite = lerNumero("#limite"), margem = lerNumero("#margem");
  if (limite === null || margem === null) return;
  E.limite = limite;
  try {
    E.alertas = await postJson("/api/alertas", {
      horas: E.prev.horas, previsto: E.prev.previsto, real: realVisivel(),
      limite_kw: limite, margem_anomalia_pct: margem,
    });
    desenharAlertas();
  } catch (e) { toast(e.message, "erro", "Falha nos alertas"); }
}
const alertasDebounced = debounce(async () => { await calcularAlertas(); desenharTabela(); }, 150);

function desenharAlertas() {
  const p = E.prev, a = E.alertas;
  if (!p || !a) return;
  const real = realVisivel();
  const traces = [
    { x: p.horas, y: p.previsto, name: `Previsão ${NOMES[p.modelo]}`, mode: "lines",
      line: { color: corModelo(p.modelo), width: 2.5 }, hovertemplate: "%{y:,.2f} kW" },
  ];
  if (real) {
    traces.push({ x: p.horas, y: real, name: "Real", mode: "lines",
      line: { color: cor("--serie-real"), width: 1.5 }, hovertemplate: "%{y:,.2f} kW" });
  }
  if (a.picos.length) {
    traces.push({ x: a.picos.map((q) => q.hora), y: a.picos.map((q) => q.kw), name: "Acima do limite", mode: "markers",
      marker: { color: cor("--critico"), size: 10, line: { color: cor("--superficie"), width: 2 } },
      hovertemplate: "pico %{y:,.2f} kW" });
  }
  if (a.anomalias.length) {
    traces.push({ x: a.anomalias.map((q) => q.hora), y: a.anomalias.map((q) => q.real_kw), name: "Anomalia (real ≫ previsto)",
      mode: "markers", marker: { symbol: "triangle-up", color: cor("--alerta"), size: 13, line: { color: cor("--texto"), width: 1 } },
      hovertemplate: "anomalia %{y:,.2f} kW" });
  }
  const gd = $("#g-alertas");
  // durante o arraste a escala fica parada, senão a linha "foge" do cursor
  const ymax = gd.classList.contains("arrastando-limite") && gd._fullLayout && gd._fullLayout.yaxis
    ? gd._fullLayout.yaxis.range[1]
    : Math.max(E.limite * 1.15, ...p.previsto, ...(real || []).filter((v) => v !== null)) * 1.08;
  Plotly.react("g-alertas", traces, layoutBase({
    xaxis: eixo({ type: "date", tickformat: "%Hh", hoverformat: "%d/%m %Hh" }),
    yaxis: eixo({ title: { text: "kW", font: { size: 12 } }, range: [0, ymax], fixedrange: true }),
    shapes: [{ type: "line", xref: "paper", x0: 0, x1: 1, yref: "y", y0: E.limite, y1: E.limite,
      line: { color: cor("--critico"), width: 2.5, dash: "dash" } }],
    annotations: [{ xref: "paper", x: 1, xanchor: "right", yref: "y", y: E.limite, yanchor: "bottom", showarrow: false,
      text: textoLimite(), font: { color: cor("--critico"), size: 12 } }],
    dragmode: false,
  }), CONFIG_PLOTLY);
  listarAlertas();
}

// Arraste da linha de limite: implementado com pointer events (mouse, caneta e toque) sobre o
// gráfico, em vez da edição de shapes do Plotly, para ter controle do eixo (só vertical) e do passo.
function textoLimite() { return `limite ${fmt(E.limite, 1)} kW · arraste`; }

function moverLinhaLimite() {
  const gd = $("#g-alertas");
  if (!gd._fullLayout || !gd.layout.shapes) return;
  Plotly.relayout(gd, {
    "shapes[0].y0": E.limite, "shapes[0].y1": E.limite,
    "annotations[0].y": E.limite, "annotations[0].text": textoLimite(),
  });
}

function ligarArrasteLimite() {
  const gd = $("#g-alertas");
  const geo = () => {
    const fl = gd._fullLayout;
    if (!fl || !fl.yaxis || !fl._size) return null;
    const r = gd.getBoundingClientRect();
    return { fl, r, topo: r.top + fl._size.t, esq: r.left + fl._size.l, larg: fl._size.w };
  };
  const perto = (ev) => {
    const g = geo();
    if (!g || !E.alertas) return false;
    const py = g.topo + g.fl.yaxis.l2p(E.limite);
    return ev.clientX >= g.esq && ev.clientX <= g.esq + g.larg && Math.abs(ev.clientY - py) <= 12;
  };
  let arrastando = false;
  gd.addEventListener("pointermove", (ev) => {
    if (!arrastando) gd.style.cursor = perto(ev) ? "ns-resize" : "";
  }, true);
  gd.addEventListener("pointerdown", (ev) => {
    if (!perto(ev)) return;
    arrastando = true;
    gd.classList.add("arrastando-limite");
    ev.preventDefault();
    ev.stopPropagation();
    try { gd.setPointerCapture(ev.pointerId); } catch { /* evento sintético */ }
  }, true);
  const mover = (ev) => {
    if (!arrastando) return;
    const g = geo();
    if (!g) return;
    const ymax = g.fl.yaxis.range[1];
    const v = Math.min(ymax, Math.max(0, Math.round(g.fl.yaxis.p2l(ev.clientY - g.topo) * 10) / 10));
    if (v === E.limite) return;
    E.limite = v;
    $("#limite").value = v.toFixed(1);
    moverLinhaLimite();
    alertasDebounced();
  };
  const soltar = () => {
    if (!arrastando) return;
    arrastando = false;
    gd.classList.remove("arrastando-limite");
    gd.style.cursor = "";
  };
  window.addEventListener("pointermove", mover);
  window.addEventListener("pointerup", soltar);
  window.addEventListener("pointercancel", soltar);
}

function listarAlertas() {
  const el = $("#lista-alertas");
  el.replaceChildren();
  const a = E.alertas;
  const n = a.picos.length;
  $("#alertas-titulo").textContent = n === 0
    ? `Nenhuma hora prevista acima de ${fmt(E.limite, 1)} kW`
    : `${n} ${n === 1 ? "hora prevista" : "horas previstas"} acima de ${fmt(E.limite, 1)} kW`;
  const real = realVisivel();
  let fraseReal = "";
  if (real) {
    const acima = E.prev.horas.filter((_, i) => real[i] !== null && real[i] > E.limite).map(rotuloHora);
    fraseReal = acima.length
      ? `O consumo real passou do limite em ${acima.length} ${acima.length === 1 ? "hora" : "horas"}: ${acima.join(", ")}.`
      : "O consumo real também ficou abaixo do limite.";
  }
  $("#alertas-real").textContent = fraseReal;
  const nota = (txt) => { const d = document.createElement("div"); d.className = "nota-alertas"; d.textContent = txt; el.appendChild(d); };
  const item = (classeMarca, titulo, detalhe, valor) => {
    const d = document.createElement("div");
    d.className = "alerta";
    const mk = document.createElement("span"); mk.className = classeMarca;
    const t = document.createElement("div");
    t.textContent = titulo;
    if (detalhe) { const sm = document.createElement("small"); sm.textContent = detalhe; t.appendChild(sm); }
    const v = document.createElement("div"); v.className = "valor-alerta"; v.textContent = valor;
    d.append(mk, t, v);
    el.appendChild(d);
  };
  for (const q of a.picos) {
    item("m-pico", `${rotuloHora(q.hora)} · ${rotuloData(paraMs(q.hora))}`, null, `${fmt(q.kw)} kW (+${fmt(q.excedente_kw)})`);
  }
  if (realVisivel()) {
    if (!a.anomalias.length) nota("Sem desperdício: o consumo real ficou dentro da margem.");
    for (const q of a.anomalias) {
      item("m-anomalia", `${rotuloHora(q.hora)} · real ${fmt(q.real_kw)} kW, previsto ${fmt(q.previsto_kw)} kW`, q.sugestao, "");
    }
  } else if (E.prev && E.prev.real) {
    nota("O desperdício aparece depois de revelar o consumo real.");
  }
}

// ------------------------------------------------------------------ cartões
function atualizarCartoes() {
  const p = E.prev;
  if (!p) return;
  const kwh = soma(p.previsto);
  $("#c-kwh").innerHTML = `${fmt(kwh, 1)}<small>kWh</small>`;
  const ini = paraMs(p.horas[0]);
  $("#c-rot").textContent = hora(p.horas[0]) === 0
    ? `Previsto para ${rotuloData(ini)}`
    : `Previsto para as 24 h a partir de ${rotuloData(ini)} · ${rotuloHora(p.horas[0])}`;

  const anterior = soma(p.historico.kw.slice(-24));
  const varia = anterior > 0 ? (kwh - anterior) / anterior : 0;
  $("#c-var").textContent = `${pct(Math.abs(varia))} ${varia >= 0 ? "acima" : "abaixo"} das 24 h anteriores`;

  const real = realVisivel();
  $("#c-kwh-nota").textContent = real ? `Real: ${fmt(soma(real), 1)} kWh` : "";

  let iMax = 0;
  p.previsto.forEach((v, i) => { if (v > p.previsto[iMax]) iMax = i; });
  const quando = document.createElement("span");
  quando.textContent = ` às ${rotuloHora(p.horas[iMax])}`;
  $("#c-pico").replaceChildren(`${fmt(p.previsto[iMax])} kW`, quando);

  if (E.custo) $("#c-custo").textContent = brl(E.custo.custo_total);
}

// ------------------------------------------------------------------ tabela
function linhasTabela() {
  const p = E.prev, c = E.custo;
  const real = realVisivel();
  const limite = E.limite;
  return p.horas.map((h, i) => ({
    hora: h,
    previsto: p.previsto[i],
    baseline: p.baseline[i],
    real: real ? real[i] : null,
    erro: real && real[i] !== null ? p.previsto[i] - real[i] : null,
    custo: c ? c.custo_hora[i] : null,
    ponta: c ? c.ponta[i] : false,
    pico: p.previsto[i] > limite,
  }));
}

function desenharTabela() {
  if (!E.prev) return;
  const { col, dir } = E.ordem;
  const linhas = linhasTabela().sort((a, b) => {
    const va = a[col], vb = b[col];
    if (va === null) return 1;
    if (vb === null) return -1;
    return (va < vb ? -1 : va > vb ? 1 : 0) * dir;
  });
  const tb = $("#tabela tbody");
  tb.replaceChildren();
  for (const l of linhas) {
    const tr = document.createElement("tr");
    if (l.ponta) tr.classList.add("ponta");
    if (l.pico) tr.classList.add("pico");
    const cel = (txt, cls) => { const td = document.createElement("td"); td.textContent = txt; if (cls) td.className = cls; tr.appendChild(td); };
    cel(`${rotuloData(paraMs(l.hora), false).slice(0, 5)} ${rotuloHora(l.hora)}${l.ponta ? " · ponta" : ""}`);
    cel(fmt(l.previsto, 3), "num prev");
    cel(fmt(l.baseline, 3), "num");
    cel(fmt(l.real, 3), "num");
    cel(l.erro === null ? "—" : (l.erro > 0 ? "+" : "") + fmt(l.erro, 3), "num");
    cel(l.custo === null ? "—" : brl(l.custo), "num");
    tb.appendChild(tr);
  }
  $$("#tabela th").forEach((th) => {
    th.setAttribute("aria-sort", th.dataset.col === col ? (dir > 0 ? "ascending" : "descending") : "none");
  });
}

function ordenarPor(col) {
  E.ordem = E.ordem.col === col ? { col, dir: -E.ordem.dir } : { col, dir: col === "hora" ? 1 : -1 };
  desenharTabela();
}

function exportarCsv() {
  if (!E.prev) return;
  const n = (v, d = 4) => (v === null || v === undefined ? "" : v.toFixed(d).replace(".", ","));
  const linhas = [["hora", "previsto_kw", "baseline_kw", "real_kw", "erro_kw", "custo_rs", "ponta"].join(";")];
  for (const l of linhasTabela()) {
    linhas.push([l.hora.replace("T", " "), n(l.previsto), n(l.baseline), n(l.real), n(l.erro), n(l.custo, 2), l.ponta ? "sim" : "não"].join(";"));
  }
  const blob = new Blob(["﻿" + linhas.join("\r\n")], { type: "text/csv;charset=utf-8" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `previsao_${E.prev.modelo}_${E.prev.horas[0].slice(0, 10)}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("CSV exportado (separador ; e vírgula decimal).");
}

// ------------------------------------------------------------------ desempenho
async function carregarDesempenho() {
  try {
    E.desempenho = await api("/api/desempenho");
    desenharDesempenho();
  } catch (e) {
    $("#desempenho-desc").textContent = e.message;
  }
}

function desenharDesempenho() {
  const d = E.desempenho;
  if (!d || !d.teste) return;
  const modelos = ["lstm", "gru", "baseline"];
  const t = d.teste;
  const melhor = (k, maior = false) => modelos.reduce((a, b) => ((maior ? t[b][k] > t[a][k] : t[b][k] < t[a][k]) ? b : a));
  const best = { mae: melhor("mae"), rmse: melhor("rmse"), wape: melhor("wape"), r2: melhor("r2", true) };
  const tab = $("#tabela-metricas");
  tab.replaceChildren();
  const thead = document.createElement("thead");
  const hr = document.createElement("tr");
  for (const [txt, num] of [["Modelo", false], ["MAE (kW)", true], ["RMSE (kW)", true], ["WAPE", true], ["R²", true], ["Épocas", true], ["Tempo de treino", true]]) {
    const th = document.createElement("th"); th.textContent = txt; if (num) th.className = "num"; hr.appendChild(th);
  }
  thead.appendChild(hr);
  const tb = document.createElement("tbody");
  for (const m of modelos) {
    const tr = document.createElement("tr");
    const add = (txt, k) => { const td = document.createElement("td"); td.textContent = txt; td.className = "num" + (k && best[k] === m ? " melhor" : ""); tr.appendChild(td); };
    const td0 = document.createElement("td");
    const mk = document.createElement("span"); mk.className = "marcador";
    mk.style.background = m === "baseline" ? cor("--serie-baseline") : corModelo(m);
    td0.append(mk, ` ${m === "baseline" ? "Baseline sazonal" : NOMES[m]}`);
    tr.appendChild(td0);
    add(fmt(t[m].mae, 3), "mae"); add(fmt(t[m].rmse, 3), "rmse"); add(pct(t[m].wape), "wape"); add(fmt(t[m].r2, 3), "r2");
    const tr_ = d.treino && d.treino[m];
    add(tr_ ? `${tr_.melhor_epoca} / ${tr_.epocas_executadas}` : "—");
    add(tr_ ? `${fmt(tr_.tempo_treino_s / 60, 1)} min` : "—");
    tb.appendChild(tr);
  }
  tab.append(thead, tb);
  if (d.conjunto_teste) {
    const c = d.conjunto_teste;
    $("#desempenho-desc").textContent = `Conjunto de teste: ${rotuloData(Date.parse(c.inicio.replace(" ", "T") + "Z"), false)} a ` +
      `${rotuloData(Date.parse(c.fim.replace(" ", "T") + "Z"), false)} · ${fmt(c.janelas, 0)} previsões de 24 h · ` +
      "nunca vistas no treino. Épocas = melhor / executadas (early stopping).";
  }

  const corDe = (m) => (m === "baseline" ? cor("--serie-baseline") : corModelo(m));
  const horiz = modelos.map((m) => ({
    x: Array.from({ length: 24 }, (_, i) => i + 1), y: d.erro_por_horizonte[m], mode: "lines+markers",
    name: m === "baseline" ? "Baseline" : NOMES[m], line: { color: corDe(m), width: 2, dash: m === "baseline" ? "dot" : "solid" },
    marker: { size: 5 }, hovertemplate: "%{y:,.3f} kW",
  }));
  Plotly.react("g-horizonte", horiz, layoutBase({
    xaxis: eixo({ title: { text: "horas à frente", font: { size: 12 } }, dtick: 2 }),
    yaxis: eixo({ title: { text: "MAE (kW)", font: { size: 12 } }, rangemode: "tozero" }),
  }), CONFIG_PLOTLY);

  const perdas = [];
  for (const m of ["lstm", "gru"]) {
    const h = d.treino && d.treino[m];
    if (!h) continue;
    const ep = h.perda_treino.map((_, i) => i + 1);
    perdas.push({ x: ep, y: h.perda_treino, name: `${NOMES[m]} treino`, mode: "lines", line: { color: corModelo(m), width: 2 }, hovertemplate: "%{y:.4f}" });
    perdas.push({ x: ep, y: h.perda_validacao, name: `${NOMES[m]} validação`, mode: "lines", line: { color: corModelo(m), width: 2, dash: "dash" }, hovertemplate: "%{y:.4f}" });
  }
  Plotly.react("g-perdas", perdas, layoutBase({
    xaxis: eixo({ title: { text: "época", font: { size: 12 } }, dtick: 2 }),
    yaxis: eixo({ title: { text: "MSE", font: { size: 12 } } }),
  }), CONFIG_PLOTLY);
}

// ------------------------------------------------------------------ tema
function temaEscuro() {
  const t = document.documentElement.dataset.theme;
  if (t) return t === "dark";
  return window.matchMedia("(prefers-color-scheme: dark)").matches;
}

function redesenharTudo() {
  desenharTempo(); desenharCusto(); desenharAlertas(); desenharDesempenho(); desenharTabela();
}

function alternarTema() {
  const novo = temaEscuro() ? "light" : "dark";
  document.documentElement.dataset.theme = novo;
  try { localStorage.setItem("eg-tema", novo); } catch { /* armazenamento indisponível */ }
  redesenharTudo();
}

// ------------------------------------------------------------------ inicialização
function preencherHoras() {
  const ini = $("#ponta-inicio"), fim = $("#ponta-fim");
  for (let h = 0; h < 24; h++) ini.add(new Option(`${String(h).padStart(2, "0")}h`, String(h), false, h === 18));
  for (let h = 1; h <= 24; h++) fim.add(new Option(`${String(h).padStart(2, "0")}h`, String(h), false, h === 21));
}

function ligarEventos() {
  $("#btn-exemplo").addEventListener("click", usarExemplo);
  $("#arquivo").addEventListener("change", (e) => { enviarArquivo(e.target.files[0]); e.target.value = ""; });
  const zona = $("#zona-upload");
  zona.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); $("#arquivo").click(); } });
  ["dragenter", "dragover"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.add("arrastando"); }));
  ["dragleave", "drop"].forEach((ev) => zona.addEventListener(ev, (e) => { e.preventDefault(); zona.classList.remove("arrastando"); }));
  zona.addEventListener("drop", (e) => enviarArquivo(e.dataTransfer.files[0]));
  // evita que soltar um arquivo fora da zona abra o arquivo no navegador
  window.addEventListener("dragover", (e) => e.preventDefault());
  window.addEventListener("drop", (e) => e.preventDefault());

  $("#btn-trocar").addEventListener("click", () => {
    const ent = $("#entrada");
    ent.hidden = !ent.hidden;
    if (!ent.hidden) window.scrollTo({ top: 0, behavior: "smooth" });
  });
  $("#btn-tema").addEventListener("click", alternarTema);
  window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { if (!document.documentElement.dataset.theme) redesenharTudo(); });

  // máquina do tempo
  $("#slider").addEventListener("input", (e) => { pararReproducao(); irPara(parseInt(e.target.value, 10)); });
  $("#btn-ant").addEventListener("click", () => { pararReproducao(); irPara(E.idx - 1); });
  $("#btn-prox").addEventListener("click", () => { pararReproducao(); irPara(E.idx + 1); });
  $("#btn-play").addEventListener("click", reproduzir);
  $("#btn-futuro").addEventListener("click", () => { pararReproducao(); irPara(E.refs.length - 1); });
  $("#data-ref").addEventListener("change", (e) => {
    if (!e.target.value) return;
    const alvo = Date.parse(e.target.value + "T00:00:00Z");
    let i = E.refs.findIndex((r) => r >= alvo);
    if (i < 0) i = E.refs.length - 1;
    if (Math.abs(E.refs[i] - alvo) > 24 * HORA_MS) toast("Data fora do período disponível; usando a mais próxima.", "info");
    pararReproducao();
    irPara(i);
  });
  $("#btn-revelar").addEventListener("click", async () => {
    E.revelado = !E.revelado;
    atualizarBotaoRevelar();
    desenharTempo();
    desenharMetricasDia();
    atualizarCartoes();
    await calcularAlertas();
    desenharTabela();
  });
  $$(".seg").forEach((b) => b.addEventListener("click", () => {
    E.modelo = b.dataset.modelo;
    $$(".seg").forEach((x) => { const on = x === b; x.classList.toggle("ativo", on); x.setAttribute("aria-checked", String(on)); });
    prever();
  }));
  document.addEventListener("keydown", (e) => {
    if (!E.resumo || ["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement.tagName)) return;
    if (e.key === "ArrowLeft") { pararReproducao(); irPara(E.idx - 1); }
    if (e.key === "ArrowRight") { pararReproducao(); irPara(E.idx + 1); }
  });

  // custo
  ["#tarifa", "#tarifa-ponta", "#ponta-inicio", "#ponta-fim"].forEach((s) => $(s).addEventListener("input", custoDebounced));
  $("#deslocar").addEventListener("input", (e) => { $("#deslocar-valor").textContent = `${e.target.value}%`; custoDebounced(); });

  // alertas
  $("#limite").addEventListener("input", alertasDebounced);
  $("#margem").addEventListener("input", alertasDebounced);
  ligarArrasteLimite();

  // tabela
  $$("#tabela th").forEach((th) => {
    th.tabIndex = 0;
    th.addEventListener("click", () => ordenarPor(th.dataset.col));
    th.addEventListener("keydown", (e) => { if (e.key === "Enter") ordenarPor(th.dataset.col); });
  });
  $("#btn-csv").addEventListener("click", exportarCsv);
}

function iniciar() {
  if (typeof Plotly === "undefined") {
    toast("Não foi possível carregar a biblioteca de gráficos (Plotly). Verifique a conexão com a internet.", "erro", "Gráficos indisponíveis");
    window.Plotly = { react: () => {}, newPlot: () => {} };
  }
  preencherHoras();
  ligarEventos();
  carregarDesempenho();
  const qs = new URLSearchParams(location.search);
  if (qs.get("revelar") === "1") E.revelado = true;
  if (qs.get("exemplo") === "1") usarExemplo();
}

document.addEventListener("DOMContentLoaded", iniciar);
