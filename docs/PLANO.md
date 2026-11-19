# Plano de construção

## 1. Dataset

**Individual Household Electric Power Consumption** — UCI ML Repository, id 235.

- Download: `https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip` (contém `household_power_consumption.txt`). Confirme a URL ao implementar; se mudou, use o pacote `ucimlrepo` ou avise o usuário.
- 2.075.259 medições por minuto, dez/2006 a nov/2010, uma residência em Sceaux (França).
- Separador `;`. Datas em `dd/mm/yyyy`. Faltantes marcados com `?` (~1,25% das linhas).
- Colunas: `Date`, `Time`, `Global_active_power` (kW, **alvo**), `Global_reactive_power`, `Voltage`, `Global_intensity`, `Sub_metering_1`, `Sub_metering_2`, `Sub_metering_3`.

Confira esses números na execução real e use os valores observados no README.

## 2. Estruturação da série temporal

1. **Limpeza:** unir `Date`+`Time` em índice datetime; `?` → NaN; tipos numéricos.
2. **Reamostragem:** média horária (~34,5 mil pontos). Reduz ruído e viabiliza treino em CPU.
3. **Faltantes:** interpolação temporal para lacunas curtas (até ~6 h); lacunas longas preenchidas com o valor da mesma hora da semana anterior.
4. **Features:** as 7 medições + seno/cosseno da hora do dia + seno/cosseno do dia da semana + flag de fim de semana.
5. **Divisão cronológica:** 70% treino / 15% validação / 15% teste, sem embaralhar a ordem temporal.
6. **Normalização:** `StandardScaler` ajustado **somente no treino**; alvo desnormalizado antes de calcular métricas.
7. **Janelas deslizantes:** X = 168 h (N × 168 × F), y = próximas 24 h de `Global_active_power` (N × 24). Janelas não cruzam a fronteira entre os conjuntos.

## 3. Modelagem

- **Baseline:** sazonal ingênuo — previsão das próximas 24 h = últimas 24 h observadas.
- **LSTMForecaster / GRUForecaster:** 2 camadas recorrentes, 64 unidades, dropout 0,2; último estado oculto → `Linear(64, 24)`.
- Hiperparâmetros centralizados em um único lugar (dataclass ou dict de configuração) e salvos junto com os pesos.

## 4. Treinamento

- Adam (lr 1e-3), perda MSE, batch 64, até 30 épocas.
- *Early stopping* com paciência 5 na perda de validação, restaurando os melhores pesos.
- `ReduceLROnPlateau` (fator 0,5, paciência 2).
- Semente 42. Lotes de treino embaralhados (as janelas são independentes); validação e teste em ordem.
- Artefatos em `models/`: `lstm.pt`, `gru.pt`, `scaler.pkl`, `metrics.json` (config, histórico de perdas por época, métricas de teste dos três modelos, erro por horizonte, tempo de treino).

## 5. Métricas

No conjunto de teste, em kW (escala original):

- **MAE**, **RMSE**, **WAPE** (Σ|erro| / Σ|real|) e **R²**.
- **MAPE fica de fora de propósito:** o consumo chega perto de zero e o MAPE explode. Explique isso na análise crítica.
- Extras: erro por horizonte (h+1 … h+24), erro por hora do dia, curvas de perda treino/validação.

Figuras em `reports/figures/` (matplotlib), usadas no README.

## 6. Aplicação

Contrato completo em [API.md](API.md).

## 7. Ordem de construção

1. Ambiente: `.venv`, instalar dependências. Risco de wheel ausente em Python 3.14 — ver `CLAUDE.md`.
2. `src/data.py` + download. Conferir formas, faltantes, período.
3. `src/models.py` e `src/train.py`. Treinar LSTM e GRU.
4. `src/evaluate.py`. Métricas e figuras.
5. `src/predict.py`, `app/services.py`, `app/main.py`. Testar a API antes do front.
6. Dashboard em `app/static/`: entrada de dados + máquina do tempo primeiro; depois custo, alertas, tabela, desempenho.
7. `tests/`.
8. `notebooks/experimento.ipynb`, executado com saídas.
9. `README.md` com números reais e análise crítica.

## 8. README — seções obrigatórias

1. Contextualização do problema
2. Objetivo da solução
3. Dados: origem e organização
4. Estruturação da série temporal (responde à "Pergunta" do case)
5. Arquitetura / modelagem
6. Processo de treinamento
7. Métricas utilizadas (e por que essas)
8. Resultados obtidos (tabela + figuras)
9. Análise crítica: onde o modelo erra (picos, horizontes longos), comparação com o baseline, limitações (uma única residência, sem clima nem feriados, dados de 2006–2010), melhorias possíveis
10. A aplicação (capturas de tela e como usar)
11. Como reproduzir (comandos)

## 9. Critérios de aceite

- `python -m src.data` imprime tamanho da série horária, faltantes tratados e formas dos tensores.
- `python -m src.train` mostra perda de validação caindo e early stopping atuando; artefatos em `models/`.
- `python -m src.evaluate` gera a tabela de métricas dos três modelos e as figuras.
- Sem vazamento: scaler só do treino; nenhuma janela de teste sobreposta ao treino (há teste automatizado para isso).
- `pytest` passa: `services.py`, endpoints (incluindo CSV inválido e histórico com menos de 168 h) e divisão dos dados.
- `uvicorn app.main:app` → fluxo completo funciona no navegador: carregar exemplo, mover a máquina do tempo, revelar o real, trocar de modelo, alterar tarifa, arrastar o limite de alerta, exportar CSV; repetir com um CSV enviado.
- Notebook roda do início ao fim sem erro.
- README com todas as seções e números iguais aos de `models/metrics.json`.
