"""Inferência: carrega os artefatos e prevê a partir de um histórico horário.

É a ponte entre os modelos treinados e a API (app/main.py).
Aplica as mesmas features e o mesmo scaler do treino; devolve kW na escala original.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from src.data import (ALVO, HORIZONTE, JANELA_ENTRADA, adicionar_features,
                      completar_colunas, desnormalizar_alvo)
from src.models import Config, baseline_sazonal, criar_modelo

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELOS_VALIDOS = ("lstm", "gru", "baseline")


class ModelosAusentes(RuntimeError):
    """Artefatos de treino não encontrados em models/."""


class Previsor:
    """Carrega lstm.pt, gru.pt e scaler.pkl uma única vez."""

    def __init__(self, models_dir=None):
        self.models_dir = Path(models_dir) if models_dir else MODELS_DIR
        arquivos = [self.models_dir / n for n in ("lstm.pt", "gru.pt", "scaler.pkl")]
        if not all(a.exists() for a in arquivos):
            raise ModelosAusentes("Modelos não treinados. Execute `python -m src.train`.")
        pacote = joblib.load(self.models_dir / "scaler.pkl")
        self.scaler = pacote["scaler"]
        self.features = pacote["features"]
        self.imputacao = pacote["imputacao"]
        self.modelos = {}
        for nome in ("lstm", "gru"):
            ck = torch.load(self.models_dir / f"{nome}.pt", map_location="cpu", weights_only=True)
            modelo = criar_modelo(nome, ck["n_features"], Config(**ck["config"]))
            modelo.load_state_dict(ck["state_dict"])
            modelo.eval()
            self.modelos[nome] = modelo

    def prever(self, historico: pd.DataFrame, modelo: str = "lstm") -> np.ndarray:
        """`historico`: DataFrame horário com pelo menos 168 linhas terminando
        na hora imediatamente anterior à primeira hora prevista.
        `modelo`: 'lstm' | 'gru' | 'baseline'. Devolve 24 valores em kW.
        Levanta ValueError (mensagem em português) se o histórico for curto."""
        if modelo not in MODELOS_VALIDOS:
            raise ValueError(f"Modelo inválido: '{modelo}'. Use lstm, gru ou baseline.")
        if len(historico) < JANELA_ENTRADA:
            raise ValueError(f"São necessárias {JANELA_ENTRADA} h de histórico antes da referência; "
                             f"há apenas {len(historico)} h.")
        janela = historico.iloc[-JANELA_ENTRADA:]
        if modelo == "baseline":
            return baseline_sazonal(janela[ALVO].to_numpy(), HORIZONTE)
        completo, _ = completar_colunas(janela, self.imputacao)
        valores = self.scaler.transform(adicionar_features(completo)[self.features].to_numpy())
        x = torch.from_numpy(valores.astype(np.float32))[None]
        with torch.no_grad():
            saida = self.modelos[modelo](x).numpy()[0]
        return np.clip(desnormalizar_alvo(saida, self.scaler), 0, None)
