# Redes neurais e como o EnergyGreen as usa

Este documento explica, do zero, o que é uma rede neural e como duas delas (LSTM e GRU) foram treinadas para prever o consumo de energia de uma casa nas próximas 24 horas. Todos os números citados vêm de `models/metrics.json`, gerado por execuções reais.

---

## 1. O que é uma rede neural

Uma rede neural é uma função matemática com muitos "botões" ajustáveis, chamados **pesos** (ou parâmetros). Ela recebe números na entrada, faz uma sequência de contas e devolve números na saída. Treinar é ajustar esses botões até a saída ficar parecida com a resposta correta.

### 1.1 O neurônio

A peça básica é o neurônio artificial:

```
saída = f( w1·x1 + w2·x2 + ... + wn·xn + b )
```

- `x1...xn` são as entradas.
- `w1...wn` são os pesos: dizem o quanto cada entrada importa.
- `b` é o viés (um deslocamento).
- `f` é a **função de ativação**: um filtro não linear (por exemplo, tanh ou sigmoide) que permite à rede aprender relações que não são retas.

### 1.2 Camadas

Vários neurônios lado a lado formam uma **camada**. Empilhando camadas, a rede aprende em níveis: as primeiras captam padrões simples, as seguintes combinam esses padrões em padrões mais abstratos. "Deep learning" (aprendizado profundo) é isso: redes com várias camadas.

### 1.3 Como a rede aprende

O treino repete quatro passos milhares de vezes:

1. **Passe para frente (forward):** a rede recebe exemplos e faz uma previsão.
2. **Perda (loss):** compara-se a previsão com a resposta real. Aqui usamos o **MSE** (erro quadrático médio): quanto mais longe, mais pesa, porque o erro é elevado ao quadrado.
3. **Retropropagação (backpropagation):** calcula-se, para cada peso, o quanto ele contribuiu para o erro (o gradiente).
4. **Atualização:** o otimizador move cada peso um pouquinho na direção que reduz o erro.

Uma passada completa por todos os exemplos de treino é uma **época**. O tamanho do passo de atualização é a **taxa de aprendizado** (*learning rate*).

---

## 2. Por que redes comuns não bastam: dados em sequência

O consumo de energia é uma **série temporal**: o valor das 18h de hoje depende do que aconteceu ontem, no mesmo horário, e na semana passada. A ordem importa. Uma rede comum trata cada entrada como independente e ignora essa ordem.

As **redes recorrentes (RNN)** resolvem isso lendo a sequência passo a passo (hora a hora) e carregando uma **memória** (o *estado oculto*) de um passo para o seguinte. Assim, cada hora é interpretada à luz do que veio antes.

Uma RNN simples, porém, "esquece" rápido: em sequências longas o gradiente some durante o treino (*vanishing gradient*). Duas variantes corrigem isso com **portões** (*gates*), que decidem o que guardar, o que esquecer e o que mostrar:

| | LSTM | GRU |
|---|---|---|
| Portões | 3 (entrada, esquecimento, saída) + célula de memória separada | 2 (atualização e reset), sem célula separada |
| Parâmetros neste projeto | 54.808 | 41.496 |
| Ideia | Mais expressiva, mais pesada | Mais simples, treina mais rápido |

O projeto treina **as duas** e compara os resultados, em vez de apostar em uma.

---

## 3. O problema que o projeto resolve

**Entrada:** as últimas **168 horas** (7 dias) de medições da casa.
**Saída:** a previsão de consumo de cada uma das próximas **24 horas** (`Global_active_power`, em kW).

Isso é previsão **multi-passo direto**: a rede devolve as 24 horas de uma vez, em vez de prever uma hora e realimentar a si mesma (o que acumularia erro).

```
 168 horas x 12 variáveis            24 horas
┌─────────────────────────┐    ┌──────────────────┐
│ h-167 ... h-2  h-1  h0  │ -> │ h+1  h+2 ... h+24│
└─────────────────────────┘    └──────────────────┘
        entrada                       saída
```

---

## 4. Os dados

- **Dataset:** Individual Household Electric Power Consumption (UCI, id 235): medições de uma casa na França, minuto a minuto, agregadas aqui **por hora**.
- **12 variáveis de entrada por hora** (`src/data.py`):
  - 7 medidas: potência ativa (o alvo), potência reativa, tensão, intensidade e os 3 submedidores (cozinha, lavanderia, aquecimento/ar-condicionado).
  - 5 variáveis de calendário: hora e dia da semana codificados como seno/cosseno (para a rede entender que 23h e 0h são vizinhas) e um indicador de fim de semana.
- **Lacunas:** buracos curtos (até 6 h) são interpolados; os maiores não são inventados.

### 4.1 Como evitamos "trapacear" (vazamento de dados)

- **Divisão cronológica:** 70% do começo da série para treino, 15% seguintes para validação, 15% finais para teste. A rede nunca vê o futuro durante o treino.
- **Normalização só com o treino:** o `StandardScaler` (média 0, desvio 1) é ajustado apenas nos dados de treino e depois aplicado à validação e ao teste. Redes aprendem melhor com números em escala parecida.
- **Janelas** de 168 h → 24 h deslizam pela série dentro de cada conjunto, sem misturar conjuntos.

Resultado: 4.805 janelas de treino, 4.997 de validação e 4.998 de teste (o treino usa passo 5 entre janelas, para treinar em minutos numa CPU; validação e teste usam todas).

---

## 5. A arquitetura usada

Definida em `src/models.py`. LSTM e GRU têm exatamente a mesma estrutura, só muda a célula:

```
entrada (lote, 168, 12)
        │
   ┌────▼─────────────────────────┐
   │ Camada recorrente 1 (64 un.) │  lê as 168 horas em ordem
   └────┬─────────────────────────┘
        │  dropout 0,2
   ┌────▼─────────────────────────┐
   │ Camada recorrente 2 (64 un.) │
   └────┬─────────────────────────┘
        │  pega só o ÚLTIMO estado oculto (resumo das 168 h)
   ┌────▼─────────────┐
   │ Linear(64 → 24)  │  transforma o resumo em 24 previsões
   └────┬─────────────┘
        ▼
   saída (lote, 24)
```

- **2 camadas × 64 unidades:** capacidade suficiente para captar rotina diária e semanal sem exagerar.
- **Dropout 0,2:** durante o treino, desliga aleatoriamente 20% das conexões. Força a rede a não decorar e ajuda a evitar overfitting.
- **Camada linear final:** converte o último estado oculto nas 24 horas previstas.

---

## 6. Como o treino foi feito

Implementado em `src/train.py`.

| Item | Valor | Para quê |
|---|---|---|
| Perda | MSE | Penaliza erros grandes |
| Otimizador | Adam, lr = 0,001 | Ajusta os pesos de forma adaptativa |
| Lote (batch) | 64 janelas | Atualiza os pesos a cada 64 exemplos, não um por um |
| Épocas máximas | 30 | Teto de segurança |
| Early stopping | paciência 5 | Para se a perda de validação não melhorar por 5 épocas e **restaura os melhores pesos** |
| ReduceLROnPlateau | fator 0,5, paciência 2 | Reduz o passo pela metade quando a validação estagna |
| Gradient clipping | norma 1,0 | Impede gradientes explosivos em redes recorrentes |
| Semente | 42 | Reprodutibilidade |

### 6.1 O papel de treino vs. validação

A cada época o modelo é treinado nos dados de **treino** e medido nos dados de **validação**, que ele nunca usa para atualizar pesos. A curva de validação é o termômetro de overfitting: se o treino continua melhorando e a validação piora, o modelo está decorando. É nela que o early stopping decide parar.

### 6.2 O que aconteceu de fato

| | LSTM | GRU |
|---|---|---|
| Épocas executadas | 11 (early stopping) | 15 (early stopping) |
| Melhor época | 6 | 10 |
| Perda de validação (melhor) | 0,560 | 0,535 |
| Tempo de treino (CPU) | ~2 min | ~8 min |

Nas duas, a perda de treino continuou caindo, enquanto a de validação estagnou e subiu levemente. Era o sinal para parar, e os pesos da melhor época foram restaurados. A curva está em `reports/figures/curvas_perda.png`.

---

## 7. Como saber se funcionou: comparação com um baseline

Uma rede só vale a pena se vencer uma solução simples. O **baseline sazonal ingênuo** apenas repete as 24 h do dia anterior, sem aprender nada. Resultados no conjunto de **teste** (dados que nenhum modelo viu):

| Modelo | MAE (kW) | RMSE (kW) | WAPE | R² |
|---|---|---|---|---|
| Baseline (dia anterior) | 0,497 | 0,744 | 52,6% | −0,099 |
| GRU | 0,434 | 0,594 | 45,9% | 0,300 |
| **LSTM** | **0,426** | **0,592** | **45,1%** | **0,305** |

- **MAE:** erro médio absoluto, em kW. Quanto menor, melhor.
- **RMSE:** como o MAE, mas pune mais os erros grandes.
- **WAPE:** erro total em relação ao consumo total, em %.
- **R²:** 1 é perfeito; 0 equivale a prever sempre a média; negativo é pior que isso.

### 7.1 Leitura honesta dos resultados

- **As redes vencem o baseline** em todas as métricas: o MAE cai cerca de 14% (LSTM) e o RMSE cerca de 20%. O LSTM e o GRU ficam praticamente empatados.
- **O ganho é real, mas moderado.** Um R² de ~0,30 mostra que a rede captura a rotina, mas está longe de prever tudo. O consumo de uma única casa é muito ruidoso (alguém liga o forno, chega visita, viaja).
- **Ponto fraco conhecido:** a rede **subestima os picos**. Nas horas de consumo mais alto (acima do percentil 90, ~1,92 kW), o erro médio do LSTM é 1,03 kW e o viés é −1,0 kW, ou seja, ela tende a prever bem menos do que o real. Isso é típico de modelos treinados com MSE sobre dados ruidosos: eles "suavizam" para se proteger de errar feio.
- **O erro não cresce com o horizonte:** a previsão da 24ª hora tem erro parecido com a da 1ª. Isso indica que boa parte do que a rede aprende é o padrão diário/semanal, e não a continuidade de curto prazo.

---

## 8. Como a rede é usada no sistema

1. `python -m src.train` salva os pesos (`models/lstm.pt`, `models/gru.pt`), o normalizador (`models/scaler.pkl`) e as métricas (`models/metrics.json`).
2. A API (FastAPI, em `app/`) carrega esses arquivos, recebe as últimas 168 horas, aplica **o mesmo** normalizador, roda a rede e **desnormaliza** a saída de volta para kW.
3. O dashboard mostra a previsão das próximas 24 h e usa esses números nos recursos interativos: máquina do tempo, simulador de custo e alertas de pico e desperdício.

O notebook e a API reutilizam o mesmo código de `src/`, então o pré-processamento do treino e o da previsão em produção são idênticos.

---

## 9. Glossário rápido

- **Peso / parâmetro:** número ajustável que a rede aprende.
- **Época:** uma passada completa pelos dados de treino.
- **Batch:** grupo de exemplos usado em cada atualização de pesos.
- **Loss (perda):** número que mede o quão errada a rede está.
- **Overfitting:** a rede decora o treino e vai mal em dados novos.
- **Early stopping:** parar o treino quando a validação deixa de melhorar.
- **Dropout:** desligar neurônios aleatórios no treino para evitar decoreba.
- **Estado oculto:** a "memória" que a rede recorrente carrega de hora em hora.
- **Baseline:** solução simples de referência para provar que a rede agrega valor.
