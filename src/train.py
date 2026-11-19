"""Treino de LSTM e GRU.

Adam lr 1e-3, MSE, batch 64, até 30 épocas, early stopping (paciência 5,
restaura melhores pesos), ReduceLROnPlateau, semente 42.
Salva em models/: lstm.pt, gru.pt, scaler.pkl e o histórico de perdas em metrics.json.
`python -m src.train` treina os dois modelos.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import FEATURES, INDICE_ALVO, preparar_tudo
from src.models import Config, criar_modelo

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
METRICS_JSON = MODELS_DIR / "metrics.json"


def fixar_semente(semente: int = 42) -> None:
    random.seed(semente)
    np.random.seed(semente)
    torch.manual_seed(semente)


def _perda_media(modelo: nn.Module, loader: DataLoader, criterio) -> float:
    modelo.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for xb, yb in loader:
            total += criterio(modelo(xb), yb).item() * len(xb)
            n += len(xb)
    return total / max(n, 1)


def treinar(nome: str, dados: dict | None = None, config: Config | None = None,
            models_dir: Path = MODELS_DIR) -> dict:
    """Treina um modelo ('lstm' | 'gru'), salva os pesos e devolve o histórico
    (perda de treino e validação por época, melhor época, tempo de treino)."""
    config = config or Config()
    dados = dados or preparar_tudo(passo_treino=config.passo_treino)
    fixar_semente(config.semente)
    torch.set_num_threads(os.cpu_count() or 1)

    gerador = torch.Generator().manual_seed(config.semente)
    treino = DataLoader(TensorDataset(torch.from_numpy(dados["X_treino"]), torch.from_numpy(dados["y_treino"])),
                        batch_size=config.batch, shuffle=True, generator=gerador)
    validacao = DataLoader(TensorDataset(torch.from_numpy(dados["X_validacao"]), torch.from_numpy(dados["y_validacao"])),
                           batch_size=256, shuffle=False)

    modelo = criar_modelo(nome, len(FEATURES), config)
    otim = torch.optim.Adam(modelo.parameters(), lr=config.lr)
    agendador = torch.optim.lr_scheduler.ReduceLROnPlateau(
        otim, mode="min", factor=config.plateau_fator, patience=config.plateau_paciencia)
    criterio = nn.MSELoss()

    hist = {"perda_treino": [], "perda_validacao": [], "lr": [], "tempo_epoca_s": []}
    melhor, melhor_epoca, melhores_pesos, sem_melhora = float("inf"), 0, None, 0
    inicio = time.perf_counter()
    print(f"\n=== Treinando {nome.upper()} ({sum(p.numel() for p in modelo.parameters()):,} parâmetros, "
          f"{len(dados['X_treino']):,} janelas de treino) ===")
    for epoca in range(1, config.epocas + 1):
        t0 = time.perf_counter()
        modelo.train()
        soma, n = 0.0, 0
        for xb, yb in treino:
            otim.zero_grad()
            perda = criterio(modelo(xb), yb)
            perda.backward()
            nn.utils.clip_grad_norm_(modelo.parameters(), 1.0)
            otim.step()
            soma += perda.item() * len(xb)
            n += len(xb)
        perda_tr = soma / n
        perda_va = _perda_media(modelo, validacao, criterio)
        agendador.step(perda_va)
        dt = time.perf_counter() - t0
        hist["perda_treino"].append(perda_tr)
        hist["perda_validacao"].append(perda_va)
        hist["lr"].append(otim.param_groups[0]["lr"])
        hist["tempo_epoca_s"].append(dt)
        marca = ""
        if perda_va < melhor - 1e-6:
            melhor, melhor_epoca, sem_melhora = perda_va, epoca, 0
            melhores_pesos = copy.deepcopy(modelo.state_dict())
            marca = "  * melhor"
        else:
            sem_melhora += 1
        print(f"época {epoca:2d} | treino {perda_tr:.4f} | validação {perda_va:.4f} | "
              f"lr {otim.param_groups[0]['lr']:.0e} | {dt:5.1f}s{marca}", flush=True)
        if sem_melhora >= config.paciencia:
            print(f"Early stopping: {config.paciencia} épocas sem melhora. "
                  f"Restaurando pesos da época {melhor_epoca}.")
            break

    modelo.load_state_dict(melhores_pesos)
    tempo = time.perf_counter() - inicio
    models_dir.mkdir(parents=True, exist_ok=True)
    torch.save({"nome": nome, "config": config.dict(), "n_features": len(FEATURES),
                "state_dict": modelo.state_dict()}, models_dir / f"{nome}.pt")
    hist.update({"melhor_epoca": melhor_epoca, "melhor_perda_validacao": melhor,
                 "epocas_executadas": len(hist["perda_treino"]),
                 "early_stopping": len(hist["perda_treino"]) < config.epocas,
                 "tempo_treino_s": tempo})
    print(f"{nome.upper()} pronto em {tempo:.0f}s (melhor época {melhor_epoca}, val {melhor:.4f}).")
    return hist


def salvar_scaler(dados: dict, models_dir: Path = MODELS_DIR) -> None:
    joblib.dump({"scaler": dados["scaler"], "features": FEATURES, "indice_alvo": INDICE_ALVO,
                 "imputacao": dados["imputacao"]}, models_dir / "scaler.pkl")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description="Treina LSTM e GRU.")
    parser.add_argument("--passo", type=int, default=Config.passo_treino,
                        help="passo entre janelas de treino (1 = todas)")
    parser.add_argument("--epocas", type=int, default=Config.epocas)
    parser.add_argument("--modelos", nargs="+", default=["lstm", "gru"])
    args = parser.parse_args(argv)

    config = Config(passo_treino=args.passo, epocas=args.epocas)
    fixar_semente(config.semente)
    dados = preparar_tudo(passo_treino=config.passo_treino)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    salvar_scaler(dados)

    metricas = json.loads(METRICS_JSON.read_text(encoding="utf-8")) if METRICS_JSON.exists() else {}
    metricas["config"] = config.dict()
    metricas["threads_cpu"] = torch.get_num_threads()
    metricas.setdefault("treino", {})
    metricas["janelas"] = {k: int(len(dados[f"y_{k}"])) for k in ("treino", "validacao", "teste")}
    for nome in args.modelos:
        metricas["treino"][nome] = treinar(nome, dados, config)
        METRICS_JSON.write_text(json.dumps(metricas, indent=2), encoding="utf-8")
    print(f"\nArtefatos salvos em {MODELS_DIR}")


if __name__ == "__main__":
    main()
