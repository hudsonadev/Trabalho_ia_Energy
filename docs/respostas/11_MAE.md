# 11. MAE — Erro Absoluto Médio

**Em uma frase:** o MAE diz, **em média, quantos kW o modelo erra por hora**.

| Modelo | MAE no teste |
|---|---|
| **LSTM** | **0,426 kW** |
| GRU | 0,434 kW |
| Baseline | 0,497 kW |

Quanto **menor**, melhor. O mínimo possível é 0 (acerto perfeito).

---

## Como se calcula

1. Para cada hora, calcule o erro: `previsto − real`.
2. Tire o sinal (valor absoluto): errar 0,2 para cima ou para baixo conta igual.
3. Faça a média.

```
MAE = média( |previsto − real| )
```

No código ([src/evaluate.py](../../src/evaluate.py)):

```python
mae = np.mean(np.abs(previsto - real))
```

## Exemplo com 4 horas

| Hora | Real (kW) | Previsto (kW) | Erro absoluto |
|---|---|---|---|
| 1 | 1,0 | 0,8 | 0,2 |
| 2 | 0,5 | 0,6 | 0,1 |
| 3 | 2,0 | 1,0 | **1,0** |
| 4 | 0,5 | 0,6 | 0,1 |

MAE = (0,2 + 0,1 + 1,0 + 0,1) ÷ 4 = **0,35 kW**

## Por que o sinal é retirado

Sem o valor absoluto, erros para cima e para baixo se cancelam. Um modelo que erra +1 numa hora
e −1 na outra teria "erro médio zero", o que esconderia que ele errou nas duas.

## Como interpretar o nosso resultado

> "A LSTM erra, em média, **0,43 kW** a cada hora prevista."

Para ter uma noção: o consumo médio no período de teste é **0,94 kW**. A LSTM erra 0,07 kW a
menos por hora que o baseline, o que dá uma redução de **14,3%** no erro médio.

## Vantagens

- **Fácil de explicar**: está na mesma unidade do consumo (kW).
- **Trata todos os erros igual**: não é distorcido por poucos erros enormes.

## Limitação

O MAE não destaca os erros grandes. Errar 1 kW numa hora e acertar o resto dá o mesmo MAE que
errar 0,25 kW em 4 horas. Para o nosso problema, errar um pico pode custar caro (tarifa de
ponta, sobrecarga), por isso olhamos também o **RMSE** (arquivo 12).
