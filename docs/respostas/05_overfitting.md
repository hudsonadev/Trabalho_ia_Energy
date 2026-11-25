# 5. Houve overfitting?

**Resposta curta:** **começou a haver, e foi contido.** Depois de algumas épocas as redes
começaram a decorar os dados de treino, mas a parada antecipada detectou isso e guardamos os
pesos de **antes** do overfitting. Os modelos finais não estão sobreajustados.

---

## O que é overfitting

**Overfitting** (sobreajuste) é quando a rede **decora** os dados de treino em vez de aprender
o padrão geral. Como um aluno que decora as respostas da lista de exercícios mas não sabe
resolver uma questão nova.

O sinal típico, olhando a curva de loss:

- o erro no **treino** continua caindo;
- o erro na **validação** (dados que a rede não usa para aprender) para de cair ou sobe.

## O que os nossos dados mostram

**LSTM:**

| | Época 6 (melhor) | Época 11 (última) | O que aconteceu |
|---|---|---|---|
| Loss de treino | 0,586 | 0,524 | continuou caindo ↓ |
| Loss de validação | **0,560** | 0,573 | **subiu** ↑ |

**GRU:**

| | Época 10 (melhor) | Época 15 (última) | O que aconteceu |
|---|---|---|---|
| Loss de treino | 0,556 | 0,518 | continuou caindo ↓ |
| Loss de validação | **0,535** | 0,549 | **subiu** ↑ |

Ou seja: **depois da melhor época, as duas redes começaram a sobreajustar.** É exatamente o
padrão descrito acima.

## Como foi controlado

Usamos quatro proteções:

1. **Parada antecipada (early stopping)**: o treino parou 5 épocas depois do melhor ponto da
   validação, e **os pesos da melhor época foram restaurados**. O modelo salvo é o da época 6
   (LSTM) e 10 (GRU), antes do overfitting.
2. **Dropout de 20%**: durante o treino, 20% das conexões entre as camadas são desligadas ao
   acaso. A rede não consegue depender de um caminho específico e é forçada a generalizar.
3. **Rede pequena**: 64 unidades por camada (cerca de 55 mil parâmetros na LSTM). Uma rede
   gigante decoraria os dados com mais facilidade.
4. **Redução da taxa de aprendizado**: quando a validação estaciona, os ajustes ficam menores.

## A prova final: o conjunto de teste

A melhor prova de que o modelo não decorou é testá-lo em dados que ele **nunca viu**, nem no
treino nem na escolha da melhor época. O conjunto de teste (abril a novembro de 2010) serve
para isso:

| Modelo | Erro médio no teste (MAE) |
|---|---|
| LSTM | 0,426 kW |
| Baseline (repetir o dia anterior) | 0,497 kW |

A LSTM erra **14,3% menos** que o baseline em dados totalmente novos. Se ela tivesse apenas
decorado o treino, ela iria mal no teste.

## Um detalhe honesto

O erro de validação ficou **menor** que o de treino nas primeiras épocas. Isso não é um erro:
o dropout fica ligado no treino e desligado na validação, e o período de validação (outono e
inverno de 2009–2010) tem um consumo mais regular. Também não indica vazamento de dados: a
divisão é cronológica, e testes automáticos (`tests/test_data.py`) verificam que nenhuma janela
mistura treino e teste.

## Resumo para apresentar

> "Sim, as redes começaram a sobreajustar depois de algumas épocas: a loss de treino continuou
> caindo e a de validação subiu. O early stopping detectou isso e restaurou os pesos da melhor
> época. Com dropout e uma rede pequena, o modelo final generaliza: no teste, com dados nunca
> vistos, ele erra 14% menos que o baseline."
