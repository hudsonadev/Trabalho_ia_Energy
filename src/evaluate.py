"""Avaliação no conjunto de teste e geração de figuras.

Métricas em kW (escala original): MAE, RMSE, WAPE, R². Sem MAPE.
Compara LSTM, GRU e baseline; calcula erro por horizonte e por hora do dia.
Atualiza models/metrics.json e grava figuras em reports/figures/.
`python -m src.evaluate` imprime a tabela comparativa.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from src.data import INDICE_ALVO, desnormalizar_alvo, preparar_tudo  # noqa: E402
from src.models import baseline_sazonal  # noqa: E402

RAIZ = Path(__file__).resolve().parent.parent
FIGURAS_DIR = RAIZ / "reports" / "figures"
MODELS_DIR = RAIZ / "models"
METRICS_JSON = MODELS_DIR / "metrics.json"
NOMES = {"lstm": "LSTM", "gru": "GRU", "baseline": "Baseline sazonal"}
CORES = {"lstm": "#2a6fdb", "gru": "#e07b00", "baseline": "#8a8f98", "real": "#1b1f24"}


def calcular_metricas(real: np.ndarray, previsto: np.ndarray) -> dict:
    """Devolve {'mae', 'rmse', 'wape', 'r2'}."""
    real = np.asarray(real, dtype=float).ravel()
    previsto = np.asarray(previsto, dtype=float).ravel()
    erro = previsto - real
    sq_tot = np.sum((real - real.mean()) ** 2)
    return {
        "mae": float(np.mean(np.abs(erro))),
        "rmse": float(np.sqrt(np.mean(erro ** 2))),
        "wape": float(np.sum(np.abs(erro)) / np.sum(np.abs(real))) if np.sum(np.abs(real)) > 0 else float("nan"),
        "r2": float(1 - np.sum(erro ** 2) / sq_tot) if sq_tot > 0 else float("nan"),
    }


def erro_por_horizonte(real: np.ndarray, previsto: np.ndarray) -> list[float]:
    """MAE para cada passo h+1 … h+24. Entradas com forma (N, 24)."""
    return np.mean(np.abs(np.asarray(previsto) - np.asarray(real)), axis=0).astype(float).tolist()


def erro_por_hora_do_dia(real: np.ndarray, previsto: np.ndarray, inicios) -> list[float]:
    """MAE agrupado pela hora do dia do instante previsto (0 … 23)."""
    h0 = np.asarray(inicios.hour)[:, None]
    horas = (h0 + np.arange(real.shape[1])[None, :]) % 24
    absoluto = np.abs(previsto - real)
    return [float(absoluto[horas == h].mean()) for h in range(24)]


def carregar_modelo(nome: str, models_dir: Path = MODELS_DIR):
    from src.models import Config, criar_modelo
    ck = torch.load(models_dir / f"{nome}.pt", map_location="cpu", weights_only=True)
    cfg = Config(**ck["config"])
    modelo = criar_modelo(nome, ck["n_features"], cfg)
    modelo.load_state_dict(ck["state_dict"])
    modelo.eval()
    return modelo


def prever_lote(modelo, X: np.ndarray, lote: int = 512) -> np.ndarray:
    saidas = []
    with torch.no_grad():
        for i in range(0, len(X), lote):
            saidas.append(modelo(torch.from_numpy(X[i:i + lote])).numpy())
    return np.concatenate(saidas)


def previsoes_teste(dados: dict | None = None) -> dict:
    """Previsões (kW) dos três modelos para todas as janelas de teste."""
    dados = dados or preparar_tudo()
    scaler = dados["scaler"]
    X, y = dados["X_teste"], dados["y_teste"]
    real = desnormalizar_alvo(y, scaler)
    entrada_kw = desnormalizar_alvo(X[:, :, INDICE_ALVO], scaler)
    saida = {"real": real, "baseline": baseline_sazonal(entrada_kw), "inicios": dados["inicio_teste"]}
    for nome in ("lstm", "gru"):
        saida[nome] = np.clip(desnormalizar_alvo(prever_lote(carregar_modelo(nome), X), scaler), 0, None)
    return saida


# ---------------------------------------------------------------- figuras
def _salvar(fig, nome: str) -> None:
    FIGURAS_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURAS_DIR / nome, dpi=120)
    plt.close(fig)


def gerar_figuras(prev: dict, resultado: dict) -> list[str]:
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": 0.3, "font.size": 10})
    modelos = ("lstm", "gru", "baseline")

    # 1. curvas de perda
    treino = resultado.get("treino", {})
    if treino:
        fig, eixos = plt.subplots(1, len(treino), figsize=(6 * len(treino), 3.6), squeeze=False)
        for ax, (nome, h) in zip(eixos[0], treino.items()):
            ep = np.arange(1, len(h["perda_treino"]) + 1)
            ax.plot(ep, h["perda_treino"], label="treino", color=CORES[nome])
            ax.plot(ep, h["perda_validacao"], label="validação", color=CORES[nome], linestyle="--")
            ax.axvline(h["melhor_epoca"], color="#999", linestyle=":", label=f"melhor época ({h['melhor_epoca']})")
            ax.set_title(f"{NOMES[nome]} — perda MSE (escala normalizada)")
            ax.set_xlabel("época")
            ax.legend()
        _salvar(fig, "curvas_perda.png")

    # 2. erro por horizonte
    fig, ax = plt.subplots(figsize=(8, 3.8))
    for m in modelos:
        ax.plot(range(1, 25), resultado["erro_por_horizonte"][m], marker="o", ms=3, label=NOMES[m], color=CORES[m])
    ax.set_xlabel("horizonte (horas à frente)")
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Erro por horizonte de previsão — conjunto de teste")
    ax.legend()
    _salvar(fig, "erro_por_horizonte.png")

    # 3. erro por hora do dia
    fig, ax = plt.subplots(figsize=(8, 3.8))
    larg = 0.27
    for i, m in enumerate(modelos):
        ax.bar(np.arange(24) + (i - 1) * larg, resultado["erro_por_hora_do_dia"][m], width=larg,
               label=NOMES[m], color=CORES[m])
    ax.set_xticks(range(24))
    ax.set_xlabel("hora do dia prevista")
    ax.set_ylabel("MAE (kW)")
    ax.set_title("Erro por hora do dia — conjunto de teste")
    ax.legend()
    _salvar(fig, "erro_por_hora.png")

    # 4. métricas comparadas
    fig, eixos = plt.subplots(1, 3, figsize=(10, 3.4))
    for ax, chave, rotulo in zip(eixos, ("mae", "rmse", "wape"), ("MAE (kW)", "RMSE (kW)", "WAPE")):
        vals = [resultado["teste"][m][chave] for m in modelos]
        barras = ax.bar([NOMES[m].split()[0] for m in modelos], vals, color=[CORES[m] for m in modelos])
        ax.bar_label(barras, fmt="%.3f", fontsize=9)
        ax.set_title(rotulo)
    _salvar(fig, "metricas_teste.png")

    # 5. exemplo: uma semana de previsões dia a dia (janelas que começam à meia-noite)
    inicios = prev["inicios"]
    meia_noite = np.where(np.asarray(inicios.hour) == 0)[0]
    escolha = meia_noite[len(meia_noite) // 2: len(meia_noite) // 2 + 7]
    fig, ax = plt.subplots(figsize=(11, 3.8))
    for j, i in enumerate(escolha):
        t = inicios[i] + np.arange(24) * np.timedelta64(1, "h")
        for m in ("real", "lstm", "gru", "baseline"):
            ax.plot(t, prev[m][i], color=CORES[m], lw=1.8 if m == "real" else 1.2,
                    linestyle="--" if m == "baseline" else "-",
                    label=(NOMES.get(m, "Real") if j == 0 else None))
    ax.set_ylabel("kW (média horária)")
    ax.set_title("Exemplo: 7 dias do teste previstos dia a dia (24 h por vez)")
    ax.legend(ncol=4)
    _salvar(fig, "exemplo_previsao.png")
    return sorted(p.name for p in FIGURAS_DIR.glob("*.png"))


def figura_divisao(dados: dict) -> None:
    from src.data import ALVO
    fig, ax = plt.subplots(figsize=(11, 3.4))
    for nome, cor in (("treino", "#2a6fdb"), ("validacao", "#e07b00"), ("teste", "#2b9348")):
        parte = dados[f"df_{nome}"][ALVO].resample("D").mean()
        ax.plot(parte.index, parte.values, color=cor, lw=1, label=nome.replace("validacao", "validação"))
    ax.set_ylabel("kW (média diária)")
    ax.set_title("Consumo ativo global — divisão cronológica 70/15/15")
    ax.legend()
    _salvar(fig, "divisao_serie.png")


def analise_extra(prev: dict) -> dict:
    """Números de apoio à análise crítica (picos, vitórias diárias, suavização, MAPE ilustrativo)."""
    real = prev["real"]
    p90 = float(np.quantile(real, 0.9))
    alto = real > p90
    meia_noite = np.where(np.asarray(prev["inicios"].hour) == 0)[0]
    mae_dia_base = np.abs(prev["baseline"][meia_noite] - real[meia_noite]).mean(axis=1)
    saida = {"p90_real_kw": p90, "dias_avaliados": int(len(meia_noite)),
             "fracao_horas_abaixo_0_3kw": float((real < 0.3).mean()), "real_min_kw": float(real.min()),
             "desvio_padrao_real_kw": float(real.std())}
    for m in ("lstm", "gru", "baseline"):
        erro = prev[m] - real
        d = {"mae_picos_p90": float(np.abs(erro[alto]).mean()), "mae_demais": float(np.abs(erro[~alto]).mean()),
             "vies_picos_p90": float(erro[alto].mean()), "desvio_padrao_previsto_kw": float(prev[m].std()),
             "mape_ilustrativo_pct": float(np.mean(np.abs(erro) / real) * 100)}
        if m != "baseline":
            mae_dia = np.abs(prev[m][meia_noite] - real[meia_noite]).mean(axis=1)
            d["dias_melhor_que_baseline"] = int((mae_dia < mae_dia_base).sum())
        saida[m] = d
    return saida


def avaliar_tudo() -> dict:
    """Avalia os três modelos no teste, salva métricas e figuras."""
    dados = preparar_tudo()
    prev = previsoes_teste(dados)
    resultado = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}
    resultado["teste"] = {m: calcular_metricas(prev["real"], prev[m]) for m in ("lstm", "gru", "baseline")}
    resultado["erro_por_horizonte"] = {m: erro_por_horizonte(prev["real"], prev[m]) for m in ("lstm", "gru", "baseline")}
    resultado["erro_por_hora_do_dia"] = {m: erro_por_hora_do_dia(prev["real"], prev[m], prev["inicios"])
                                         for m in ("lstm", "gru", "baseline")}
    resultado["analise"] = analise_extra(prev)
    te = dados["df_teste"]
    resultado["conjunto_teste"] = {"inicio": str(te.index.min()), "fim": str(te.index.max()),
                                   "horas": int(len(te)), "janelas": int(len(prev["real"]))}
    METRICS_JSON.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    figura_divisao(dados)
    resultado["figuras"] = gerar_figuras(prev, resultado)
    METRICS_JSON.write_text(json.dumps(resultado, indent=2), encoding="utf-8")
    return resultado


def tabela(resultado: dict) -> str:
    linhas = [f"{'Modelo':<18}{'MAE (kW)':>10}{'RMSE (kW)':>11}{'WAPE':>9}{'R²':>8}"]
    for m in ("lstm", "gru", "baseline"):
        r = resultado["teste"][m]
        linhas.append(f"{NOMES[m]:<18}{r['mae']:>10.4f}{r['rmse']:>11.4f}{r['wape']:>9.4f}{r['r2']:>8.4f}")
    return "\n".join(linhas)


if __name__ == "__main__":
    res = avaliar_tudo()
    c = res["conjunto_teste"]
    print(f"Conjunto de teste: {c['inicio']} -> {c['fim']} ({c['horas']} h, {c['janelas']} janelas)\n")
    print(tabela(res))
    base = res["teste"]["baseline"]["mae"]
    for m in ("lstm", "gru"):
        ganho = (base - res["teste"][m]["mae"]) / base * 100
        print(f"{NOMES[m]}: MAE {ganho:+.1f}% em relação ao baseline")
    print("\nMAE por horizonte (h+1, h+6, h+12, h+24):")
    for m in ("lstm", "gru", "baseline"):
        e = res["erro_por_horizonte"][m]
        print(f"  {NOMES[m]:<18} {e[0]:.3f}  {e[5]:.3f}  {e[11]:.3f}  {e[23]:.3f}")
    a = res["analise"]
    print(f"\nPicos (real > p90 = {a['p90_real_kw']:.2f} kW) — MAE nos picos / demais horas / viés nos picos:")
    for m in ("lstm", "gru", "baseline"):
        print(f"  {NOMES[m]:<18} {a[m]['mae_picos_p90']:.3f} / {a[m]['mae_demais']:.3f} / {a[m]['vies_picos_p90']:+.3f}")
    for m in ("lstm", "gru"):
        print(f"{NOMES[m]} melhor que o baseline em {a[m]['dias_melhor_que_baseline']} de {a['dias_avaliados']} dias (previsões à meia-noite)")
    print(f"\nFiguras em {FIGURAS_DIR}: {', '.join(res['figuras'])}")
