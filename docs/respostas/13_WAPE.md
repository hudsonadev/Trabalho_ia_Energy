# 13. WAPE — Erro Percentual Absoluto Ponderado

**Em uma frase:** o WAPE diz **quanto o modelo errou em proporção ao consumo total**, em
porcentagem.

| Modelo | WAPE no teste |
|---|---|
| **LSTM** | **45,1%** |
| GRU | 45,9% |
| Baseline | 52,6% |

Quanto **menor**, melhor. 0% seria acerto perfeito.

---

## Como se calcula

1. Some todos os erros absolutos.
2. Some todo o consumo real.
3. Divida um pelo outro.

```
WAPE = soma( |previsto − real| ) ÷ soma( |real| )
```

No código ([src/evaluate.py](../../src/evaluate.py)):

```python
wape = np.sum(np.abs(previsto - real)) / np.sum(np.abs(real))
```

## Exemplo com as mesmas 4 horas

| Hora | Real | Previsto | Erro absoluto |
|---|---|---|---|
| 1 | 1,0 | 0,8 | 0,2 |
| 2 | 0,5 | 0,6 | 0,1 |
| 3 | 2,0 | 1,0 | 1,0 |
| 4 | 0,5 | 0,6 | 0,1 |
| **Soma** | **4,0** | | **1,4** |

WAPE = 1,4 ÷ 4,0 = **35%**

## Como interpretar o nosso resultado

> "Somando todas as horas do teste, o erro da LSTM equivale a **45%** de todo o consumo."

Parece alto, mas lembre que são previsões **hora a hora** de **uma única casa**, que é muito
irregular. Se a pergunta fosse "quanto a casa vai gastar no dia inteiro", os erros de horas
diferentes se compensariam e o erro percentual seria bem menor.

O que importa é a comparação: a LSTM tem **7,5 pontos percentuais a menos** que o baseline.

## Por que WAPE e não MAPE?

O **MAPE** é a métrica percentual mais conhecida. Ele calcula o erro percentual **de cada
hora** e tira a média:

```
MAPE = média( |previsto − real| ÷ real )
```

O problema aparece quando o consumo real é **pequeno**. De madrugada a casa consome perto de
0,2 kW. Um erro de apenas 0,2 kW vira um erro de **100%** naquela hora, e poucas horas assim
dominam a média inteira.

| Hora | Real | Previsto | Erro | Erro % (MAPE) |
|---|---|---|---|---|
| madrugada | 0,2 | 0,4 | 0,2 kW | **100%** |
| noite | 2,0 | 2,2 | 0,2 kW | 10% |

O mesmo erro de 0,2 kW pesa 10 vezes mais de madrugada. O MAPE passa a medir **o quanto a casa
consumia pouco**, e não a qualidade do modelo. No nosso teste, 10% das horas estão abaixo de
0,3 kW, o que inflaria o MAPE (daria 59,0% na LSTM e 65,1% no baseline).

O **WAPE resolve isso**: ele soma tudo antes de dividir, então as horas de consumo baixo não
explodem a conta. Responde a mesma pergunta ("quanto erramos em proporção?") de forma estável.

## Vantagens

- Em porcentagem: fácil de comunicar para quem não é técnico.
- Estável mesmo com consumo baixo.
- Permite comparar com outros problemas de escala diferente.
