"""Gera CSVs sintéticos de demonstração em exemplos/ (semente 42, reprodutível).

Uso: python scripts/gerar_csvs_exemplo.py
São dados SINTÉTICOS, só para testar o upload do dashboard; não são medições reais.
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEMENTE = 42
SAIDA = Path(__file__).resolve().parent.parent / "exemplos"
COLS = ["Global_active_power", "Global_reactive_power", "Voltage", "Global_intensity",
        "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"]


def _perfil(h):
    """Perfil diário (kW) de uma casa: madrugada baixa, pico de manhã e à noite."""
    return (0.45 + 0.9 * np.exp(-((h - 7.5) ** 2) / 3.5) + 1.5 * np.exp(-((h - 19.5) ** 2) / 4.5)
            + 0.4 * np.exp(-((h - 13) ** 2) / 6))


def serie(inicio, horas, rng, freq="h", escala=1.0, picos=False, standby=0.0):
    idx = pd.date_range(inicio, periods=horas, freq=freq)
    h = idx.hour.to_numpy() + idx.minute.to_numpy() / 60
    dia = idx.day_of_week.to_numpy()
    kw = _perfil(h) * escala * np.where(dia >= 5, 1.2, 1.0)
    kw = kw * (1 + rng.normal(0, 0.08, len(idx))) + rng.normal(0, 0.03, len(idx))
    if picos:  # 1 dia a cada 4 com pico forte (chuveiro + forno + ar-condicionado)
        forte = (idx.normalize().day_of_year.to_numpy() % 4 == 0) & (h >= 18) & (h < 21)
        kw = kw + np.where(forte, rng.uniform(2.5, 4.0, len(idx)), 0)
    kw = kw + standby  # consumo fantasma constante, inclusive de madrugada
    kw = np.clip(kw, 0.15, None)
    volt = 241 + rng.normal(0, 2, len(idx)) - 1.2 * kw
    df = pd.DataFrame({
        "Global_active_power": kw,
        "Global_reactive_power": 0.08 + 0.05 * kw + rng.normal(0, 0.01, len(idx)).clip(-0.05),
        "Voltage": volt,
        "Global_intensity": kw * 1000 / volt * 1.05,
        "Sub_metering_1": np.clip(rng.normal(0.4, 0.5, len(idx)) + (h > 18) * kw, 0, None),
        "Sub_metering_2": np.clip(rng.normal(0.6, 0.6, len(idx)) + (h == 12) * 3, 0, None),
        "Sub_metering_3": np.clip(kw * 4 + rng.normal(0, 1, len(idx)), 0, None),
    }, index=idx)
    df.index.name = "datetime"
    return df.round(3)


def main():
    rng = np.random.default_rng(SEMENTE)
    SAIDA.mkdir(exist_ok=True)

    # 1. Casa típica, 30 dias, todas as colunas (fluxo feliz)
    serie("2026-09-01", 24 * 30, rng).to_csv(SAIDA / "01_casa_tipica_30dias.csv")

    # 2. Só o consumo (testa o aviso de colunas estimadas)
    serie("2026-09-01", 24 * 21, rng)[["Global_active_power"]].to_csv(SAIDA / "02_so_consumo_21dias.csv")

    # 3. Formato bruto da UCI: por minuto, separador ';', dd/mm/aaaa, '?' nas falhas
    b = serie("2026-09-20", 24 * 60 * 14, rng, freq="min")
    b.loc[b.sample(frac=0.003, random_state=SEMENTE).index, :] = np.nan
    b.insert(0, "Time", b.index.strftime("%H:%M:%S"))
    b.insert(0, "Date", b.index.strftime("%d/%m/%Y"))
    b.to_csv(SAIDA / "03_formato_uci_por_minuto_14dias.csv", sep=";", index=False, na_rep="?")

    # 4. Com lacunas (queda de energia de 5 h e outra de 2 dias) para testar o preenchimento
    d = serie("2026-08-15", 24 * 28, rng)
    d.iloc[24 * 9 + 10: 24 * 9 + 15] = np.nan
    d.iloc[24 * 18: 24 * 20] = np.nan
    d.to_csv(SAIDA / "04_com_lacunas_28dias.csv", na_rep="?")

    # 5. Picos e desperdício: picos fortes à noite + consumo fantasma alto (alertas do dashboard)
    serie("2026-10-01", 24 * 30, rng, escala=1.1, picos=True, standby=0.45).to_csv(
        SAIDA / "05_picos_e_desperdicio_30dias.csv")

    # 6. Histórico curto (120 h < 168 h): deve ser recusado com mensagem em português
    serie("2026-11-01", 120, rng).to_csv(SAIDA / "06_invalido_historico_curto_5dias.csv")

    for p in sorted(SAIDA.glob("*.csv")):
        print(f"{p.name}  {p.stat().st_size / 1024:.0f} KB")


if __name__ == "__main__":
    main()
