"""Testes de src/data.py com uma série sintética pequena (sem download)."""

import numpy as np
import pandas as pd
import pytest

from src.data import (ALVO, COLUNAS_MEDIDAS, FEATURES, HORIZONTE, INDICE_ALVO, JANELA_ENTRADA,
                      adicionar_features, criar_janelas, dividir, para_horario, preparar_tudo)


def serie_horaria(horas: int = 24 * 60, semente: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(semente)
    idx = pd.date_range("2009-01-01", periods=horas, freq="h", name="datetime")
    base = 1 + 0.6 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(0, 0.1, horas)
    df = pd.DataFrame({c: base * (i + 1) for i, c in enumerate(COLUNAS_MEDIDAS)}, index=idx)
    df[ALVO] = base
    return df


def test_formas_das_janelas():
    valores = np.random.default_rng(1).normal(size=(500, len(FEATURES))).astype(np.float32)
    X, y = criar_janelas(valores, INDICE_ALVO)
    n = 500 - JANELA_ENTRADA - HORIZONTE + 1
    assert X.shape == (n, JANELA_ENTRADA, len(FEATURES))
    assert y.shape == (n, HORIZONTE)


def test_janela_alinha_entrada_e_alvo():
    valores = np.arange(400, dtype=np.float32)[:, None]
    X, y = criar_janelas(valores, 0)
    for i in (0, 17, len(y) - 1):
        assert X[i, -1, 0] == i + JANELA_ENTRADA - 1  # última hora observada
        assert y[i, 0] == i + JANELA_ENTRADA  # primeira hora prevista
        assert y[i, -1] == i + JANELA_ENTRADA + HORIZONTE - 1


def test_janelas_com_passo():
    valores = np.arange(600, dtype=np.float32)[:, None]
    X1, _ = criar_janelas(valores, 0)
    X5, y5 = criar_janelas(valores, 0, passo=5)
    assert len(X5) == len(X1[::5])
    assert np.array_equal(X5[1], X1[5])


def test_serie_curta_nao_gera_janelas():
    X, y = criar_janelas(np.zeros((100, 3), np.float32), 0)
    assert len(X) == 0 and len(y) == 0


def test_divisao_cronologica_sem_sobreposicao():
    df = serie_horaria()
    tr, va, te = dividir(df)
    assert len(tr) + len(va) + len(te) == len(df)
    assert tr.index.max() < va.index.min()
    assert va.index.max() < te.index.min()
    assert abs(len(tr) / len(df) - 0.70) < 0.01


def test_preparar_tudo_sem_vazamento():
    df = serie_horaria()
    dados = preparar_tudo(df)
    tr, te = dados["df_treino"], dados["df_teste"]
    # scaler ajustado apenas no treino
    assert dados["scaler"].mean_[INDICE_ALVO] == pytest.approx(tr[ALVO].mean())
    assert dados["scaler"].mean_[INDICE_ALVO] != pytest.approx(df[ALVO].mean(), abs=1e-9)
    # nenhuma janela de teste (entrada ou alvo) toca o treino ou a validação
    primeira_entrada_teste = dados["inicio_teste"][0] - pd.Timedelta(hours=JANELA_ENTRADA)
    assert primeira_entrada_teste == te.index[0]
    assert primeira_entrada_teste > dados["df_validacao"].index.max()
    # alvos de treino não avançam para a validação
    ultimo_alvo_treino = dados["inicio_treino"][-1] + pd.Timedelta(hours=HORIZONTE - 1)
    assert ultimo_alvo_treino <= tr.index.max()
    assert dados["X_teste"].shape[1:] == (JANELA_ENTRADA, len(FEATURES))


def test_preenchimento_nao_deixa_nan():
    df = serie_horaria(24 * 30)
    minutos = df.resample("min").ffill()
    minutos.iloc[100:200] = np.nan  # lacuna curta (< 6 h)
    inicio_longa = 60 * 24 * 10  # dia 10: já existe uma semana anterior
    minutos.iloc[inicio_longa:inicio_longa + 60 * 30] = np.nan  # lacuna longa (30 h)
    minutos.iloc[-10:] = np.nan
    horario = para_horario(minutos)
    assert not horario.isna().any().any()
    assert horario.attrs["faltantes_preenchidos"] >= 30
    # lacuna longa preenchida com a mesma hora da semana anterior
    ts = minutos.index[inicio_longa + 60 * 10].floor("h")
    assert horario.loc[ts, ALVO] == pytest.approx(horario.loc[ts - pd.Timedelta(hours=168), ALVO])


def test_features_de_calendario():
    df = adicionar_features(serie_horaria(48))
    assert set(FEATURES) <= set(df.columns)
    assert df["fim_de_semana"].isin([0.0, 1.0]).all()
    assert np.allclose(df["hora_sin"] ** 2 + df["hora_cos"] ** 2, 1)
