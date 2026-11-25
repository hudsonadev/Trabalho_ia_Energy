# 2. Como foi feito o treinamento de Deep Learning?

**Resumo:** treinamos duas redes neurais recorrentes (LSTM e GRU) em **PyTorch**, rodando na
**CPU**. Cada rede aprendeu a olhar os **últimos 7 dias** de consumo e prever as **próximas 24
horas**. O código está em [src/train.py](../../src/train.py).

---

## Passo 1: preparar os dados

Uma rede neural não lê um arquivo de texto direto. Antes, os dados passam por uma preparação
([src/data.py](../../src/data.py)):

1. **Limpeza**: as medições marcadas com `?` viram valores faltantes.
2. **Média por hora**: as medições minuto a minuto viram uma média por hora (34.589 horas).
3. **Preenchimento de lacunas**: buracos curtos são interpolados; buracos longos recebem o
   valor da mesma hora na semana anterior.
4. **12 variáveis de entrada**: as 7 medições do aparelho, mais a hora do dia e o dia da
   semana (codificados com seno e cosseno, para a rede saber que 23h e 0h são vizinhas), mais
   uma marca de fim de semana.
5. **Divisão no tempo**: 70% mais antigos para treino, 15% para validação, 15% mais recentes
   para teste. **Sem embaralhar**: o modelo nunca vê o futuro durante o treino.
6. **Normalização**: todas as variáveis são colocadas na mesma escala. A escala é calculada
   **só com os dados de treino**, para não vazar informação do teste.

## Passo 2: montar os exemplos ("janelas")

Cada exemplo de treino é um par:

```
ENTRADA (X): 168 horas × 12 variáveis   →   SAÍDA (y): as 24 horas seguintes de consumo
         (7 dias de histórico)                    (o que a rede deve prever)
```

A rede recebe a entrada e tenta acertar a saída. As 24 horas saem **de uma só vez**
("multi-passo direto"), então o erro de uma hora não contamina a hora seguinte.

## Passo 3: a arquitetura da rede

Definida em [src/models.py](../../src/models.py):

```
168 horas × 12 variáveis
        │
  [Camada LSTM 1: 64 unidades]
        │   dropout 20%
  [Camada LSTM 2: 64 unidades]
        │   pega o último estado (o "resumo" dos 7 dias)
  [Camada Linear: 64 → 24]
        │
  24 valores = previsão das próximas 24 horas
```

A GRU tem a mesma estrutura, só troca a célula LSTM pela GRU.

## Passo 4: o ciclo de treino

A cada **época** (uma passada completa pelos dados de treino):

1. Os exemplos são embaralhados e divididos em **lotes de 64**.
2. Para cada lote, a rede faz a previsão.
3. Calcula-se o erro com a função de perda **MSE** (média dos erros ao quadrado).
4. O **otimizador Adam** ajusta os pesos da rede para diminuir esse erro
   (taxa de aprendizado inicial 0,001).
5. Os gradientes são limitados (norma 1,0) para o treino não "explodir".

Ao fim de cada época, a rede é testada nos dados de **validação** (que ela não usa para
aprender). Esse número decide quando parar (ver o arquivo 4).

## Configuração completa

| Item | Valor |
|---|---|
| Framework | PyTorch 2.15 (CPU, sem GPU) |
| Otimizador | Adam, taxa de aprendizado 0,001 |
| Função de perda | MSE |
| Tamanho do lote | 64 |
| Máximo de épocas | 30 |
| Parada antecipada | após 5 épocas sem melhora na validação |
| Redução da taxa | metade, após 2 épocas sem melhora |
| Dropout | 20% |
| Semente aleatória | 42 (resultados reproduzíveis) |

## Quanto tempo levou

| Modelo | Parâmetros | Tempo de treino |
|---|---|---|
| LSTM | 54.808 | 2,0 minutos |
| GRU | 41.496 | 7,8 minutos |

Para caber em ~10 minutos na CPU, o treino usa uma janela a cada 5 horas (em vez de todas).
Como 5 não divide 24, as janelas ainda começam em todas as horas do dia.

## O que sai do treino

Tudo fica salvo na pasta `models/`:

- `lstm.pt` e `gru.pt`: os pesos das redes treinadas.
- `scaler.pkl`: a normalização (para aplicar a mesma escala em dados novos).
- `metrics.json`: o histórico de perda de cada época e as métricas finais.

## Para refazer

```powershell
python -m src.data      # prepara os dados
python -m src.train     # treina LSTM e GRU
python -m src.evaluate  # calcula as métricas e gera os gráficos
```
