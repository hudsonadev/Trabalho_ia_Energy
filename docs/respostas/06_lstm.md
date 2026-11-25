# 6. Usamos LSTM?

**Resposta curta: sim.** A LSTM é um dos dois modelos de Deep Learning do projeto, e foi o que
teve o **melhor resultado**. Também treinamos uma **GRU** para comparar, e as duas foram
comparadas a um **baseline** simples.

---

## O que é uma LSTM

**LSTM** (*Long Short-Term Memory*, "memória longa de curto prazo") é um tipo de rede neural
feita para **sequências**, como texto, áudio ou séries temporais. Ela lê os dados em ordem, um
passo de cada vez, e guarda uma **memória** do que já leu.

A diferença para uma rede recorrente simples são os **portões** (*gates*), que decidem a cada
passo:

- **Portão de esquecimento**: o que da memória pode ser descartado.
- **Portão de entrada**: o que da informação nova vale guardar.
- **Portão de saída**: o que da memória usar para a resposta deste passo.

Isso permite à LSTM lembrar de coisas de muitos passos atrás. No nosso caso: "há 7 dias, nesta
mesma hora, o consumo subiu".

## Por que LSTM faz sentido aqui

Consumo de energia é uma série temporal com padrões que se repetem:

- **diário**: pouco consumo de madrugada, picos de manhã e à noite;
- **semanal**: fins de semana diferentes dos dias úteis.

A análise dos dados mostrou que o consumo de **uma semana atrás** se parece tanto com o de agora
quanto o de **ontem** (autocorrelação de 0,47 em 168 h e 0,44 em 24 h). Por isso a LSTM olha
**168 horas** (7 dias) para trás: ela precisa dessa memória longa.

## Como a nossa LSTM está montada

Código em [src/models.py](../../src/models.py), classe `LSTMForecaster`:

```python
self.rnn = nn.LSTM(12, 64, num_layers=2, dropout=0.2, batch_first=True)
self.saida = nn.Linear(64, 24)

def forward(self, x):                 # x: (lote, 168 horas, 12 variáveis)
    seq, _ = self.rnn(x)
    return self.saida(seq[:, -1, :])  # último estado → 24 horas previstas
```

| Item | Valor |
|---|---|
| Entrada | 168 horas × 12 variáveis |
| Camadas LSTM | 2, com 64 unidades cada |
| Dropout entre as camadas | 20% |
| Saída | camada linear → 24 valores (as próximas 24 horas) |
| Parâmetros treináveis | 54.808 |

## LSTM, GRU e baseline: o resultado

A **GRU** é uma "prima" mais simples da LSTM: tem 2 portões em vez de 3, menos parâmetros, e
costuma ter desempenho parecido. Treinamos as duas para comparar.

| Modelo | Erro médio (MAE) | RMSE | R² |
|---|---|---|---|
| **LSTM** | **0,426 kW** | **0,592 kW** | **0,305** |
| GRU | 0,434 kW | 0,594 kW | 0,300 |
| Baseline (repete o dia anterior) | 0,497 kW | 0,744 kW | −0,099 |

- A **LSTM foi a melhor** em todas as métricas gerais, com 14,3% menos erro médio que o baseline.
- A GRU ficou praticamente empatada, e errou um pouco menos nos picos de consumo.
- No dashboard, dá para alternar entre **LSTM** e **GRU** com um botão e ver a diferença em
  qualquer dia.
