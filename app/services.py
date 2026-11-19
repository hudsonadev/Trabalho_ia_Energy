"""Regras de negócio do dashboard: custo/tarifa, alertas de pico e anomalias.

Funções puras (sem FastAPI, sem estado), testadas em tests/test_services.py.
Formatos de entrada e saída em docs/API.md.
"""

from __future__ import annotations

from datetime import datetime


def _hora(h: str) -> int:
    return datetime.fromisoformat(str(h)).hour


def eh_ponta(hora: int, inicio: int, fim: int) -> bool:
    """Faixa de ponta semiaberta [inicio, fim): 18→21 cobre 18h, 19h e 20h.
    Suporta faixas que cruzam a meia-noite (ex.: 22→2)."""
    if inicio == fim:
        return False
    if inicio < fim:
        return inicio <= hora < fim
    return hora >= inicio or hora < fim


def calcular_custo(previsto: list[float], horas: list[str], tarifa: float,
                   tarifa_ponta: float, ponta_inicio: int, ponta_fim: int,
                   deslocar_pct: float = 0.0) -> dict:
    """Custo por hora e total, consumo e custo na ponta, e economia ao deslocar
    `deslocar_pct`% do consumo da ponta para as horas fora de ponta.

    Média horária em kW durante 1 h = kWh. O consumo deslocado é dividido igualmente
    entre as horas fora de ponta, de modo que o kWh total se conserva.
    """
    if len(previsto) != len(horas):
        raise ValueError("'previsto' e 'horas' devem ter o mesmo tamanho.")
    kwh = [max(float(v), 0.0) for v in previsto]
    ponta = [eh_ponta(_hora(h), ponta_inicio, ponta_fim) for h in horas]
    preco = [tarifa_ponta if p else tarifa for p in ponta]
    custo_hora = [k * p for k, p in zip(kwh, preco)]

    kwh_ponta = sum(k for k, p in zip(kwh, ponta) if p)
    n_fora = sum(1 for p in ponta if not p)
    fracao = min(max(deslocar_pct, 0.0), 100.0) / 100.0
    if n_fora == 0 or kwh_ponta == 0:
        fracao = 0.0
    movido = kwh_ponta * fracao
    kwh_desl = [k * (1 - fracao) if p else k + movido / n_fora for k, p in zip(kwh, ponta)]
    custo_desl = [k * p for k, p in zip(kwh_desl, preco)]

    custo_total = sum(custo_hora)
    custo_com = sum(custo_desl)
    r = lambda v: round(v, 4)  # noqa: E731
    return {
        "custo_hora": [r(c) for c in custo_hora],
        "custo_total": r(custo_total),
        "kwh_total": r(sum(kwh)),
        "kwh_ponta": r(kwh_ponta),
        "custo_ponta": r(sum(c for c, p in zip(custo_hora, ponta) if p)),
        "custo_com_deslocamento": r(custo_com),
        "economia": r(custo_total - custo_com),
        # campos extras usados pelo dashboard
        "ponta": ponta,
        "kwh_deslocado": [r(k) for k in kwh_desl],
        "custo_hora_deslocado": [r(c) for c in custo_desl],
    }


def _sugestao(hora: int, desvio_pct: float) -> str:
    texto = f"Consumo {desvio_pct:.0f}% acima do esperado às {hora}h"
    if 0 <= hora < 6:
        return f"{texto} — madrugada: verifique aquecedor, ar-condicionado ou aparelhos esquecidos ligados."
    if 6 <= hora < 12:
        return f"{texto} — manhã: confira chuveiro elétrico, aquecedor de água e máquinas de lavar."
    if 12 <= hora < 18:
        return f"{texto} — verifique equipamentos ligados sem uso (forno, ferro, climatização)."
    return f"{texto} — horário de pico: considere adiar lava-louças, secadora ou forno para fora da ponta."


def detectar_alertas(horas: list[str], previsto: list[float],
                     real: list[float] | None, limite_kw: float,
                     margem_anomalia_pct: float = 50.0) -> dict:
    """Picos previstos acima do limite e, se houver real, horas em que o real
    superou o previsto além da margem (com sugestão de ação em português)."""
    if len(previsto) != len(horas):
        raise ValueError("'previsto' e 'horas' devem ter o mesmo tamanho.")
    picos = [{"hora": h, "kw": round(float(v), 4), "excedente_kw": round(float(v) - limite_kw, 4)}
             for h, v in zip(horas, previsto) if float(v) > limite_kw]
    anomalias = []
    if real is not None:
        if len(real) != len(horas):
            raise ValueError("'real' e 'horas' devem ter o mesmo tamanho.")
        for h, p, r in zip(horas, previsto, real):
            if r is None:
                continue
            base = max(float(p), 0.05)  # evita divisão por quase zero
            desvio = (float(r) - base) / base * 100
            if desvio > margem_anomalia_pct:
                anomalias.append({"hora": h, "real_kw": round(float(r), 4), "previsto_kw": round(float(p), 4),
                                  "desvio_pct": round(desvio, 1), "sugestao": _sugestao(_hora(h), desvio)})
    return {"picos": picos, "anomalias": anomalias}
