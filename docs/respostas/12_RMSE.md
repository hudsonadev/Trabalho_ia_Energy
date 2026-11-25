# 12. RMSE — Raiz do Erro Quadrático Médio

**Em uma frase:** o RMSE também mede o erro em kW, mas **pune mais os erros grandes**.

| Modelo | RMSE no teste |
|---|---|
| **LSTM** | **0,592 kW** |
| GRU | 0,594 kW |
| Baseline | 0,744 kW |

Quanto **menor**, melhor. O mínimo possível é 0.

---

## Como se calcula

1. Para cada hora, calcule o erro: `previsto − real`.
2. **Eleve ao quadrado** (isso tira o sinal e aumenta os erros grandes).
3. Faça a média.
4. Tire a **raiz quadrada**, para voltar à unidade original (kW).

```
RMSE = raiz( média( (previsto − real)² ) )
```

No código ([src/evaluate.py](../../src/evaluate.py)):

```python
rmse = np.sqrt(np.mean((previsto - real) ** 2))
```

## Exemplo com as mesmas 4 horas do MAE

| Hora | Real | Previsto | Erro | Erro² |
|---|---|---|---|---|
| 1 | 1,0 | 0,8 | 0,2 | 0,04 |
| 2 | 0,5 | 0,6 | 0,1 | 0,01 |
| 3 | 2,0 | 1,0 | 1,0 | **1,00** |
| 4 | 0,5 | 0,6 | 0,1 | 0,01 |

Média dos quadrados = 1,06 ÷ 4 = 0,265 → RMSE = √0,265 = **0,51 kW**

Compare: o MAE dessas mesmas horas é **0,35 kW**. O RMSE ficou maior por causa da hora 3. Ao
elevar ao quadrado, o erro de 1,0 continuou 1,0, mas os erros pequenos (0,1 → 0,01) quase
sumiram. Resultado: **o erro grande domina a conta**.

## Por que isso importa para energia

Errar um pico custa caro: se o modelo prevê 1 kW e a casa consome 3 kW no horário de ponta, a
conta e o planejamento ficam bem errados. Vários erros pequenos incomodam muito menos. O RMSE
reflete isso.

## O que o nosso resultado mostra

| | MAE | RMSE |
|---|---|---|
| Ganho da LSTM sobre o baseline | 14,3% | **20,5%** |

A LSTM melhora **mais** no RMSE do que no MAE. Isso quer dizer que ela evita principalmente
os **erros grandes** do baseline. Faz sentido: o baseline "copia" os picos de ontem para hoje,
e quando o pico não se repete no mesmo horário, ele erra feio duas vezes (prevê um pico que
não veio e não prevê o que veio).

## Dica: comparar RMSE com MAE

- Se o RMSE for **quase igual** ao MAE, os erros têm tamanho parecido.
- Se o RMSE for **bem maior** que o MAE, há alguns erros grandes no meio.

Na LSTM: 0,592 ÷ 0,426 ≈ 1,4. No baseline: 0,744 ÷ 0,497 ≈ 1,5. O baseline tem erros grandes
mais frequentes.

## Limitação

Como amplia os erros grandes, o RMSE é sensível a poucos casos extremos. Por isso ele é usado
**junto** com o MAE, não no lugar dele.

**Curiosidade:** a função de perda usada no treino é o **MSE**, que é o RMSE antes da raiz.
Ou seja, a rede foi treinada justamente para evitar erros grandes.
