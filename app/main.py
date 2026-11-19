"""API FastAPI + servidor do dashboard estático.

Endpoints: POST /api/upload, GET /api/exemplo, GET /api/serie, POST /api/prever,
POST /api/custo, POST /api/alertas, GET /api/desempenho.
Monta app/static/ na raiz (index.html em "/").
Executar: uvicorn app.main:app --reload
"""

from __future__ import annotations

import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app import services
from src.data import (ALVO, CACHE_MASCARA, COLUNAS_MEDIDAS, HORIZONTE, JANELA_ENTRADA,
                      carregar_horario, dividir, ler_csv_usuario, resumo_serie)
from src.evaluate import calcular_metricas
from src.predict import MODELS_DIR, ModelosAusentes, Previsor

STATIC_DIR = Path(__file__).resolve().parent / "static"
FMT = "%Y-%m-%dT%H:%M:%S"
MSG_SEM_MODELOS = "Modelos não treinados. Execute `python -m src.train`."

ESTADO: dict = {"previsor": None, "serie": None, "colunas_estimadas": []}


def carregar_previsor() -> None:
    try:
        ESTADO["previsor"] = Previsor()
    except ModelosAusentes:
        ESTADO["previsor"] = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    carregar_previsor()
    yield


app = FastAPI(title="EnergyGreen — Previsão de Consumo de Energia", lifespan=lifespan)


def _traduzir_erro(e: dict) -> str:
    tipo, ctx = e.get("type", ""), e.get("ctx") or {}
    if tipo == "missing":
        return "campo obrigatório"
    if tipo == "literal_error":
        return f"use um destes valores: {ctx.get('expected', '')}".replace(" or ", " ou ")
    if tipo in ("float_parsing", "float_type", "int_parsing", "int_type", "int_from_float"):
        return "deve ser um número"
    if tipo in ("greater_than_equal", "greater_than"):
        return f"deve ser maior ou igual a {ctx.get('ge', ctx.get('gt'))}"
    if tipo in ("less_than_equal", "less_than"):
        return f"deve ser menor ou igual a {ctx.get('le', ctx.get('lt'))}"
    if tipo == "too_short":
        return "a lista não pode ser vazia"
    if tipo in ("json_invalid", "model_attributes_type", "dict_type"):
        return "corpo JSON inválido"
    if tipo in ("list_type", "string_type"):
        return "tipo de valor inválido"
    return "valor inválido"


@app.exception_handler(RequestValidationError)
async def erro_validacao(_req: Request, exc: RequestValidationError):
    partes = []
    for e in exc.errors():
        campo = ".".join(str(p) for p in e.get("loc", []) if p != "body")
        if e.get("type") == "json_invalid":
            campo = "corpo"
        partes.append(f"{campo or 'requisição'}: {_traduzir_erro(e)}")
    return JSONResponse(status_code=422, content={"detail": "Dados inválidos na requisição — " + "; ".join(partes)})


# ---------------------------------------------------------------- utilidades
def _previsor() -> Previsor:
    if ESTADO["previsor"] is None:
        carregar_previsor()
    if ESTADO["previsor"] is None:
        raise HTTPException(503, MSG_SEM_MODELOS)
    return ESTADO["previsor"]


def _serie() -> pd.DataFrame:
    if ESTADO["serie"] is None:
        raise HTTPException(400, "Nenhuma série carregada. Envie um CSV ou use os dados de exemplo.")
    return ESTADO["serie"]


def _iso(indice) -> list[str]:
    return [t.strftime(FMT) for t in indice]


def _lista(valores) -> list:
    return [None if (v is None or (isinstance(v, float) and np.isnan(v))) else round(float(v), 4) for v in valores]


def _definir_serie(df: pd.DataFrame, estimadas: list[str]) -> dict:
    ESTADO["serie"] = df
    ESTADO["colunas_estimadas"] = estimadas
    resp = resumo_serie(df)
    resp["colunas_estimadas"] = estimadas
    return resp


# ---------------------------------------------------------------- endpoints
@app.post("/api/upload")
async def upload(arquivo: UploadFile = File(...)):
    conteudo = await arquivo.read()
    try:
        df = ler_csv_usuario(conteudo)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc
    estimadas = [c for c in COLUNAS_MEDIDAS if c not in df.columns]
    return _definir_serie(df, estimadas)


@app.get("/api/exemplo")
def exemplo():
    try:
        horario = carregar_horario()
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(503, "Dados não preparados. Execute `python -m src.data`.") from exc
    _, _, teste = dividir(horario)
    teste = teste.copy()
    preenchidos = 0
    if CACHE_MASCARA.exists():
        mascara = pd.read_pickle(CACHE_MASCARA)
        preenchidos = int(mascara.reindex(teste.index, fill_value=False).sum())
    teste.attrs = {"faltantes_preenchidos": preenchidos}
    return _definir_serie(teste, [])


@app.get("/api/serie")
def serie(inicio: str | None = None, fim: str | None = None):
    df = _serie()
    try:
        if inicio:
            df = df[df.index >= pd.Timestamp(inicio)]
        if fim:
            df = df[df.index <= pd.Timestamp(fim)]
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, "Datas inválidas em 'inicio'/'fim'. Use AAAA-MM-DDTHH:MM:SS.") from exc
    return {"datetime": _iso(df.index), "kw": _lista(df[ALVO].to_numpy())}


class PedidoPrevisao(BaseModel):
    modelo: Literal["lstm", "gru", "baseline"] = "lstm"
    data_referencia: str | None = None


@app.post("/api/prever")
def prever(pedido: PedidoPrevisao):
    previsor = _previsor()
    df = _serie()
    fim = df.index.max()
    if pedido.data_referencia:
        try:
            ref = pd.Timestamp(pedido.data_referencia).floor("h")
        except (ValueError, TypeError) as exc:
            raise HTTPException(400, "data_referencia inválida. Use AAAA-MM-DDTHH:MM:SS.") from exc
        if ref.tzinfo is not None:
            ref = ref.tz_localize(None)
    else:
        ref = fim + pd.Timedelta(hours=1)
    if ref > fim + pd.Timedelta(hours=1):
        raise HTTPException(400, f"data_referencia além do fim da série ({fim.strftime(FMT)}). "
                                 "Escolha uma data até a hora seguinte ao último dado.")
    historico = df[df.index < ref].iloc[-JANELA_ENTRADA:]
    if len(historico) < JANELA_ENTRADA or historico.index[-1] != ref - pd.Timedelta(hours=1):
        raise HTTPException(400, f"Não há {JANELA_ENTRADA} h de histórico antes de {ref.strftime(FMT)}. "
                                 f"A primeira referência possível é "
                                 f"{(df.index.min() + pd.Timedelta(hours=JANELA_ENTRADA)).strftime(FMT)}.")
    try:
        previsto = previsor.prever(historico, pedido.modelo)
        baseline = previsor.prever(historico, "baseline")
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc

    horas = pd.date_range(ref, periods=HORIZONTE, freq="h")
    real_serie = df[ALVO].reindex(horas)
    real, metricas = None, None
    if real_serie.notna().any():
        real = _lista(real_serie.to_numpy())
        ok = real_serie.notna().to_numpy()
        m = calcular_metricas(real_serie.to_numpy()[ok], previsto[ok])
        metricas = {k: round(m[k], 4) for k in ("mae", "rmse", "wape")}
        mb = calcular_metricas(real_serie.to_numpy()[ok], baseline[ok])
        metricas["baseline"] = {k: round(mb[k], 4) for k in ("mae", "rmse", "wape")}
        metricas["horas_com_real"] = int(ok.sum())
    return {
        "modelo": pedido.modelo,
        "horas": _iso(horas),
        "previsto": _lista(previsto),
        "baseline": _lista(baseline),
        "real": real,
        "metricas": metricas,
        "historico": {"datetime": _iso(historico.index), "kw": _lista(historico[ALVO].to_numpy())},
    }


class PedidoCusto(BaseModel):
    previsto: list[float] = Field(min_length=1)
    horas: list[str] = Field(min_length=1)
    tarifa: float = Field(0.85, ge=0)
    tarifa_ponta: float = Field(1.60, ge=0)
    ponta_inicio: int = Field(18, ge=0, le=23)
    ponta_fim: int = Field(21, ge=0, le=24)
    deslocar_pct: float = Field(0, ge=0, le=100)


@app.post("/api/custo")
def custo(p: PedidoCusto):
    try:
        return services.calcular_custo(p.previsto, p.horas, p.tarifa, p.tarifa_ponta,
                                       p.ponta_inicio, p.ponta_fim % 24, p.deslocar_pct)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


class PedidoAlertas(BaseModel):
    horas: list[str] = Field(min_length=1)
    previsto: list[float] = Field(min_length=1)
    real: list[float | None] | None = None
    limite_kw: float = Field(3.0, ge=0)
    margem_anomalia_pct: float = Field(50, ge=0)


@app.post("/api/alertas")
def alertas(p: PedidoAlertas):
    try:
        return services.detectar_alertas(p.horas, p.previsto, p.real, p.limite_kw, p.margem_anomalia_pct)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


@app.get("/api/desempenho")
def desempenho():
    caminho = MODELS_DIR / "metrics.json"
    if not caminho.exists():
        raise HTTPException(503, MSG_SEM_MODELOS)
    return json.loads(caminho.read_text(encoding="utf-8"))


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
