# 4. Com quantas épocas trabalhamos, e por quê?

**Resposta curta:** configuramos um **máximo de 30 épocas**, mas o treino parou sozinho antes,
por **parada antecipada** (*early stopping*):

| Modelo | Épocas executadas | Melhor época (a que foi salva) |
|---|---|---|
| **LSTM** | 11 | **6** |
| **GRU** | 15 | **10** |

---

## O que é uma época

Uma **época** é uma passada completa da rede por todos os exemplos de treino. Em cada época a
rede vê todos os dados uma vez e ajusta seus pesos. Poucas épocas e a rede não aprende o
suficiente; épocas demais e ela começa a **decorar** os dados de treino (overfitting).

## Por que não um número fixo?

Não dá para saber de antemão qual é o número certo. Por isso, em vez de escolher "treine 20
épocas", usamos uma regra automática:

> **Parada antecipada:** ao fim de cada época, medimos o erro nos dados de validação. Se ele
> passar **5 épocas seguidas sem melhorar**, o treino para, e voltamos aos pesos da melhor
> época.

O que aconteceu na prática com a LSTM:

```
Época 6  → menor erro de validação (0,560)   ← melhor, pesos guardados
Época 7  → piorou     (1 sem melhora)
Época 8  → piorou     (2)
Época 9  → piorou     (3)
Época 10 → piorou     (4)
Época 11 → piorou     (5)  → PARA e restaura os pesos da época 6
```

A GRU seguiu a mesma lógica: melhor na época 10, parou na 15.

## Por que esses valores?

| Escolha | Valor | Motivo |
|---|---|---|
| Máximo de épocas | 30 | Um teto de segurança. Suficiente para a rede convergir, sem deixar o treino rodar sem fim na CPU. Nenhum modelo chegou perto. |
| Paciência | 5 épocas | Dá à rede uma chance de sair de uma piora momentânea sem desperdiçar tempo depois que ela já estabilizou. |
| Redução da taxa de aprendizado | metade, após 2 épocas sem melhora | Antes de desistir, a rede tenta passos menores para refinar os pesos. Na LSTM, a taxa caiu de 0,001 para 0,0005 na época 9. |

## Por que a GRU treinou mais épocas?

Cada rede aprende num ritmo próprio. A GRU continuou melhorando na validação por mais tempo
(até a época 10). A parada antecipada se adapta a isso automaticamente: cada modelo treina o
quanto precisa, nem mais nem menos.

## Resumo para apresentar

> "Definimos até 30 épocas, mas usamos early stopping com paciência 5. A LSTM teve o melhor
> resultado na época 6 e parou na 11; a GRU teve o melhor na época 10 e parou na 15. Assim cada
> modelo treinou o necessário, e guardamos os pesos da melhor época, não os da última."
