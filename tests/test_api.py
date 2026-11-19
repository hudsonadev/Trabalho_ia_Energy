"""Testes dos endpoints com fastapi.testclient.TestClient."""

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app
from src.predict import MODELS_DIR

TEM_MODELOS = all((MODELS_DIR / n).exists() for n in ("lstm.pt", "gru.pt", "scaler.pkl"))
precisa_modelos = pytest.mark.skipif(not TEM_MODELOS, reason="execute python -m src.train antes")


@pytest.fixture()
def cliente():
    with TestClient(app) as c:
        yield c


def csv_horario(horas: int = 24 * 10, so_alvo: bool = False) -> bytes:
    idx = pd.date_range("2010-01-01", periods=horas, freq="h")
    kw = 1 + 0.8 * np.sin(2 * np.pi * idx.hour / 24) ** 2
    df = pd.DataFrame({"datetime": idx.strftime("%Y-%m-%d %H:%M:%S"), "Global_active_power": kw.round(3)})
    if not so_alvo:
        df["Global_reactive_power"] = 0.1
        df["Voltage"] = 240.0
        df["Global_intensity"] = (kw * 4.2).round(3)
        for i in (1, 2, 3):
            df[f"Sub_metering_{i}"] = float(i)
    return df.to_csv(index=False).encode()


def csv_uci(dias: int = 8) -> bytes:
    idx = pd.date_range("2008-03-01", periods=dias * 24 * 60, freq="min")
    kw = 1 + 0.5 * np.sin(2 * np.pi * idx.hour / 24)
    linhas = ["Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;"
              "Sub_metering_1;Sub_metering_2;Sub_metering_3"]
    for t, v in zip(idx, kw):
        linhas.append(f"{t.day}/{t.month}/{t.year};{t:%H:%M:%S};{v:.3f};0.100;240.00;{v * 4.2:.1f};0;1;17")
    linhas[500] = linhas[500].split(";", 2)[0] + ";" + linhas[500].split(";", 2)[1] + ";?;?;?;?;?;?;"
    return ("\n".join(linhas)).encode()


def enviar(cliente, conteudo: bytes, nome: str = "dados.csv"):
    return cliente.post("/api/upload", files={"arquivo": (nome, conteudo, "text/csv")})


def test_upload_formato_horario(cliente):
    r = enviar(cliente, csv_horario())
    assert r.status_code == 200, r.text
    corpo = r.json()
    assert corpo["horas"] == 240
    assert corpo["inicio"] == "2010-01-01T00:00:00"
    assert corpo["colunas_estimadas"] == []
    assert {"fim", "faltantes_preenchidos", "media_kw", "pico_kw"} <= corpo.keys()


def test_upload_formato_uci(cliente):
    r = enviar(cliente, csv_uci(), "household.txt")
    assert r.status_code == 200, r.text
    assert r.json()["horas"] == 8 * 24
    serie = cliente.get("/api/serie").json()
    assert len(serie["kw"]) == 8 * 24


def test_upload_sem_coluna_alvo(cliente):
    r = enviar(cliente, b"datetime,consumo\n2010-01-01 00:00,1.0\n")
    assert r.status_code == 400
    assert "Global_active_power" in r.json()["detail"]


def test_upload_historico_curto(cliente):
    r = enviar(cliente, csv_horario(100))
    assert r.status_code == 400
    assert "168" in r.json()["detail"]


def test_upload_datas_ilegiveis(cliente):
    r = enviar(cliente, b"datetime,Global_active_power\nontem,1\nhoje,2\namanha,3\n")
    assert r.status_code == 400


def test_upload_vazio(cliente):
    assert enviar(cliente, b"").status_code == 400


@precisa_modelos
def test_prever_sem_real_apos_o_fim(cliente):
    enviar(cliente, csv_horario(24 * 10, so_alvo=True))
    r = cliente.post("/api/prever", json={"modelo": "lstm"})
    assert r.status_code == 200, r.text
    c = r.json()
    assert c["horas"][0] == "2010-01-11T00:00:00"
    assert len(c["previsto"]) == 24 and len(c["baseline"]) == 24
    assert c["real"] is None and c["metricas"] is None
    assert len(c["historico"]["kw"]) == 168


@precisa_modelos
@pytest.mark.parametrize("modelo", ["lstm", "gru", "baseline"])
def test_prever_com_real(cliente, modelo):
    enviar(cliente, csv_horario())
    r = cliente.post("/api/prever", json={"modelo": modelo, "data_referencia": "2010-01-09T00:00:00"})
    assert r.status_code == 200, r.text
    c = r.json()
    assert len(c["real"]) == 24
    assert set(c["metricas"]) >= {"mae", "rmse", "wape"}
    assert c["historico"]["datetime"][-1] == "2010-01-08T23:00:00"


@precisa_modelos
def test_prever_sem_168h_antes(cliente):
    enviar(cliente, csv_horario())
    r = cliente.post("/api/prever", json={"modelo": "gru", "data_referencia": "2010-01-03T00:00:00"})
    assert r.status_code == 400
    assert "168" in r.json()["detail"]


@precisa_modelos
def test_prever_modelo_invalido(cliente):
    enviar(cliente, csv_horario())
    r = cliente.post("/api/prever", json={"modelo": "xgboost"})
    assert r.status_code == 422
    assert "detail" in r.json()


@precisa_modelos
def test_exemplo_e_desempenho(cliente):
    r = cliente.get("/api/exemplo")
    assert r.status_code == 200
    assert r.json()["horas"] > 1000
    d = cliente.get("/api/desempenho").json()
    assert set(d["teste"]) == {"lstm", "gru", "baseline"}


def test_custo(cliente):
    horas = [f"2010-06-15T{h:02d}:00:00" for h in range(24)]
    r = cliente.post("/api/custo", json={"previsto": [1.0] * 24, "horas": horas, "tarifa": 0.85,
                                         "tarifa_ponta": 1.60, "ponta_inicio": 18, "ponta_fim": 21,
                                         "deslocar_pct": 20})
    assert r.status_code == 200
    c = r.json()
    assert c["kwh_total"] == pytest.approx(24)
    assert c["custo_total"] == pytest.approx(21 * 0.85 + 3 * 1.60)
    assert c["economia"] > 0


def test_alertas(cliente):
    horas = [f"2010-06-15T{h:02d}:00:00" for h in range(24)]
    prev = [1.0] * 24
    prev[19] = 3.4
    r = cliente.post("/api/alertas", json={"horas": horas, "previsto": prev, "real": None,
                                           "limite_kw": 3.0, "margem_anomalia_pct": 50})
    assert r.status_code == 200
    assert len(r.json()["picos"]) == 1 and r.json()["anomalias"] == []
    real = [1.0] * 24
    real[14] = 4.1
    r = cliente.post("/api/alertas", json={"horas": horas, "previsto": prev, "real": real,
                                           "limite_kw": 3.0, "margem_anomalia_pct": 50})
    assert r.json()["anomalias"][0]["hora"] == horas[14]


def test_pagina_servida(cliente):
    r = cliente.get("/")
    assert r.status_code == 200
    assert "EnergyGreen" in r.text
