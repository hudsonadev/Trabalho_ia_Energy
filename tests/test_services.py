"""Testes de app/services.py."""

import pytest

from app.services import calcular_custo, detectar_alertas, eh_ponta

HORAS = [f"2010-06-15T{h:02d}:00:00" for h in range(24)]


def test_custo_tarifa_unica():
    r = calcular_custo([1.0] * 24, HORAS, tarifa=0.5, tarifa_ponta=0.5, ponta_inicio=18, ponta_fim=21)
    assert r["kwh_total"] == pytest.approx(24)
    assert r["custo_total"] == pytest.approx(12)
    assert r["custo_hora"] == [0.5] * 24
    assert r["kwh_ponta"] == pytest.approx(3)


def test_custo_com_ponta():
    r = calcular_custo([1.0] * 24, HORAS, tarifa=1.0, tarifa_ponta=2.0, ponta_inicio=18, ponta_fim=21)
    assert r["custo_ponta"] == pytest.approx(6)
    assert r["custo_total"] == pytest.approx(21 + 6)
    assert r["custo_hora"][18] == 2.0 and r["custo_hora"][21] == 1.0


def test_deslocamento_reduz_custo_e_conserva_kwh():
    prev = [0.5 + (h % 5) * 0.3 for h in range(24)]
    sem = calcular_custo(prev, HORAS, 0.85, 1.60, 18, 21, deslocar_pct=0)
    com = calcular_custo(prev, HORAS, 0.85, 1.60, 18, 21, deslocar_pct=40)
    assert sem["economia"] == pytest.approx(0)
    assert com["custo_com_deslocamento"] < com["custo_total"]
    assert com["economia"] > 0
    assert sum(com["kwh_deslocado"]) == pytest.approx(sum(prev), abs=1e-3)


def test_ponta_cruzando_meia_noite():
    assert eh_ponta(23, 22, 2) and eh_ponta(1, 22, 2)
    assert not eh_ponta(2, 22, 2) and not eh_ponta(12, 22, 2)


def test_picos_acima_do_limite():
    prev = [1.0] * 24
    prev[19], prev[20] = 3.4, 2.9
    r = detectar_alertas(HORAS, prev, None, limite_kw=3.0)
    assert len(r["picos"]) == 1
    assert r["picos"][0]["hora"] == HORAS[19]
    assert r["picos"][0]["excedente_kw"] == pytest.approx(0.4)


def test_anomalias_so_com_real():
    prev = [1.0] * 24
    real = [1.0] * 24
    real[14] = 2.1  # +110%
    real[15] = 1.3  # +30%, abaixo da margem
    r = detectar_alertas(HORAS, prev, real, limite_kw=5, margem_anomalia_pct=50)
    assert [a["hora"] for a in r["anomalias"]] == [HORAS[14]]
    a = r["anomalias"][0]
    assert a["desvio_pct"] == pytest.approx(110)
    assert "14h" in a["sugestao"]


def test_sem_real_nenhuma_anomalia():
    r = detectar_alertas(HORAS, [5.0] * 24, None, limite_kw=1)
    assert r["anomalias"] == []
    assert len(r["picos"]) == 24


def test_tamanhos_diferentes_geram_erro():
    with pytest.raises(ValueError):
        calcular_custo([1.0] * 3, HORAS, 1, 1, 18, 21)
