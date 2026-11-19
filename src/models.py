"""Modelos: LSTMForecaster, GRUForecaster e baseline sazonal ingênuo.

Ambos os modelos: entrada (lote, 168, F) -> saída (lote, 24).
2 camadas recorrentes, 64 unidades, dropout 0,2, último estado -> Linear(64, 24).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn


@dataclass
class Config:
    """Hiperparâmetros centralizados (salvos junto com os pesos e em metrics.json)."""
    entrada: int = 168
    horizonte: int = 24
    hidden: int = 64
    camadas: int = 2
    dropout: float = 0.2
    lr: float = 1e-3
    batch: int = 64
    epocas: int = 30
    paciencia: int = 5
    plateau_fator: float = 0.5
    plateau_paciencia: int = 2
    # passo entre janelas de TREINO (validação/teste usam todas). 5 é coprimo de 24, então as
    # janelas continuam começando em todas as horas do dia; reduz o treino em CPU para minutos.
    passo_treino: int = 5
    semente: int = 42

    def dict(self) -> dict:
        return asdict(self)


class _Recorrente(nn.Module):
    celula: type[nn.Module] = nn.LSTM

    def __init__(self, n_features: int, hidden: int = 64, camadas: int = 2,
                 dropout: float = 0.2, horizonte: int = 24):
        super().__init__()
        self.rnn = self.celula(input_size=n_features, hidden_size=hidden, num_layers=camadas,
                               dropout=dropout if camadas > 1 else 0.0, batch_first=True)
        self.saida = nn.Linear(hidden, horizonte)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq, _ = self.rnn(x)
        return self.saida(seq[:, -1, :])  # último estado oculto da última camada


class LSTMForecaster(_Recorrente):
    celula = nn.LSTM


class GRUForecaster(_Recorrente):
    celula = nn.GRU


def baseline_sazonal(historico_kw: np.ndarray, horizonte: int = 24) -> np.ndarray:
    """Previsão = últimas `horizonte` horas observadas (mesmo horário do dia anterior).

    Aceita (T,) -> (horizonte,) ou (N, T) -> (N, horizonte).
    """
    historico_kw = np.asarray(historico_kw, dtype=float)
    return historico_kw[..., -horizonte:].copy()


def criar_modelo(nome: str, n_features: int, config: Config | None = None) -> nn.Module:
    """Fábrica: 'lstm' ou 'gru'."""
    config = config or Config()
    classes = {"lstm": LSTMForecaster, "gru": GRUForecaster}
    if nome not in classes:
        raise ValueError(f"Modelo desconhecido: {nome}. Use 'lstm' ou 'gru'.")
    return classes[nome](n_features, hidden=config.hidden, camadas=config.camadas,
                         dropout=config.dropout, horizonte=config.horizonte)
