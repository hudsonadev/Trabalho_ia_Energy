# 1. A matriz de confusão está implementada?

**Resposta curta: não, e isso está correto.** Matriz de confusão não se aplica a este projeto.

## Por quê?

A matriz de confusão é uma ferramenta de **classificação**: o modelo escolhe entre categorias
(ex.: "spam" ou "não spam", "gato", "cachorro" ou "pássaro"). A matriz conta quantas vezes o
modelo acertou e errou cada categoria:

|                  | Previu "spam" | Previu "não spam" |
|------------------|---------------|-------------------|
| **Era spam**     | acerto        | erro              |
| **Não era spam** | erro          | acerto            |

O nosso problema é de **regressão**: o modelo não escolhe uma categoria, ele prevê **um número**
(quantos kW a casa vai consumir em cada uma das próximas 24 horas). Não existe "acertou a
classe" ou "errou a classe". Existe "errou por quanto". Prever 1,52 kW quando o real foi
1,60 kW não é um erro de categoria, é um erro de 0,08 kW.

Por isso, uma matriz de confusão aqui não teria sentido. O enunciado do trabalho
([docs/ENUNCIADO.md](../ENUNCIADO.md)) também não pede essa matriz.

## O que usamos no lugar

As métricas próprias de regressão, todas calculadas em [src/evaluate.py](../../src/evaluate.py):

| Métrica | O que responde | Resultado da LSTM |
|---|---|---|
| **MAE** (erro absoluto médio) | "Em média, erramos quantos kW por hora?" | 0,426 kW |
| **RMSE** (raiz do erro quadrático médio) | Igual ao MAE, mas pune mais os erros grandes | 0,592 kW |
| **WAPE** (erro percentual ponderado) | "Erramos quanto, em proporção ao consumo total?" | 45,1% |
| **R²** (coeficiente de determinação) | "Quanto da variação do consumo o modelo explica?" | 0,305 |

Para ver **onde** o modelo erra (que é o papel da matriz de confusão na classificação), temos:

- **Erro por horizonte**: o erro da previsão 1 hora à frente comparado ao de 24 horas à frente
  (`reports/figures/erro_por_horizonte.png`).
- **Erro por hora do dia**: em que horários o modelo erra mais (`reports/figures/erro_por_hora.png`).
- **Análise de picos**: o erro nas horas de consumo muito alto, separado do resto.

## E se o professor perguntar?

> "Matriz de confusão é para classificação. Nosso modelo prevê valores contínuos de consumo,
> então é regressão. Avaliamos com MAE, RMSE, WAPE e R², além do erro por horizonte e por hora
> do dia, que cumprem o mesmo papel de mostrar onde o modelo erra."

**Nota:** daria para "forçar" uma matriz de confusão transformando o consumo em faixas
(baixo / médio / alto) e tratando cada faixa como uma classe. Não fizemos isso porque joga
fora informação (1,9 kW e 2,1 kW cairiam em faixas diferentes, 2,1 kW e 6 kW na mesma) e não é
o que o problema pede.
