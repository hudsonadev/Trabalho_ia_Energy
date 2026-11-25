# 8. Como está funcionando o sistema na web?

**Resumo:** o sistema web tem duas partes. Um **servidor** em Python (**FastAPI**) carrega os
modelos treinados e faz as previsões. Um **dashboard** (HTML, CSS e JavaScript puro, gráficos
com **Plotly.js**) mostra tudo no navegador e deixa o usuário interagir.

Para rodar:

```powershell
uvicorn app.main:app
```

Depois é só abrir **http://localhost:8000**.

---

## Como as partes conversam

```
 ┌───────────────┐   envia CSV / pede previsão   ┌──────────────────┐   usa   ┌──────────────┐
 │   Navegador   │ ────────────────────────────► │   API FastAPI    │ ──────► │  src/        │
 │  (dashboard)  │ ◄──────────────────────────── │  (app/main.py)   │         │  predict.py  │
 └───────────────┘   devolve números em JSON     └──────────────────┘         │  lstm.pt     │
                                                                              │  gru.pt      │
                                                                              └──────────────┘
```

1. O usuário envia um arquivo com o histórico de consumo (ou clica em "Usar dados de exemplo").
2. O navegador manda os dados para a **API**.
3. A API prepara os dados com **o mesmo código usado no treino** (`src/data.py`), para a
   previsão ser feita do mesmo jeito que o modelo foi treinado.
4. A **LSTM** ou a **GRU** faz a previsão das próximas 24 horas.
5. A API devolve os números, e o dashboard desenha os gráficos.

## Os endereços da API

Definidos em [app/main.py](../../app/main.py) e documentados em [docs/API.md](../API.md):

| Endereço | O que faz |
|---|---|
| `POST /api/upload` | Recebe um CSV com o histórico de consumo |
| `GET /api/exemplo` | Carrega os dados de exemplo (o período de teste) |
| `GET /api/serie` | Devolve a série de consumo carregada |
| `POST /api/prever` | Faz a previsão das 24 horas a partir de um momento escolhido |
| `POST /api/custo` | Calcula o custo em reais, com tarifa de ponta |
| `POST /api/alertas` | Lista as horas acima de um limite e as anomalias |
| `GET /api/desempenho` | Devolve as métricas dos modelos no teste |

Erros são sempre devolvidos em português, por exemplo:
`{"detail": "Nenhuma série carregada. Envie um CSV ou use os dados de exemplo."}`.

## O que o dashboard oferece

### 1. Entrada de dados
Arrastar um CSV ou usar o exemplo. Aceita o arquivo bruto da UCI ou um CSV horário simples
(`datetime` + consumo). É preciso ter pelo menos 168 horas (7 dias).

### 2. Máquina do tempo
O recurso principal. O usuário escolhe **qualquer dia** do histórico (por um controle deslizante,
pela data ou pelas setas do teclado) e vê:

- os 7 dias de entrada que o modelo leu;
- a previsão das 24 horas seguintes;
- o baseline, para comparar.

O botão **"Revelar o consumo real"** mostra o que de fato aconteceu e o erro daquele dia.
**"Reproduzir"** avança dia a dia como um vídeo. **"Ir para o futuro"** prevê as 24 horas
depois do fim dos dados. Um botão alterna entre **LSTM** e **GRU**.

### 3. Simulador de custo e tarifa
Transforma a previsão em **reais**. O usuário ajusta o preço do kWh, a tarifa do horário de
ponta e o horário da ponta, e move um controle para "deslocar X% do consumo da ponta". O
gráfico e a economia recalculam na hora.

### 4. Alertas de pico e desperdício
Uma linha tracejada no gráfico marca um **limite de potência**; o usuário **arrasta a linha**
e o sistema lista as horas previstas acima dele. Com o consumo real revelado, aponta
**anomalias** (consumo real muito acima do previsto) com uma sugestão, como "madrugada: verifique
aquecedor ou aparelhos esquecidos ligados".

### 5. Tabela hora a hora
Previsto, baseline, real, erro e custo de cada hora. Dá para ordenar e **exportar para CSV**
(abre direto no Excel em português).

### 6. Desempenho dos modelos
As métricas do teste, o erro por horizonte e as curvas de loss do treino.

Extras: tema claro e escuro, e layout que funciona no celular.

## Arquivos

| Arquivo | Papel |
|---|---|
| [app/main.py](../../app/main.py) | A API: recebe pedidos e devolve respostas |
| [app/services.py](../../app/services.py) | Regras de custo, tarifa e alertas |
| [src/predict.py](../../src/predict.py) | Carrega os modelos e faz a previsão |
| [app/static/index.html](../../app/static/index.html) | Estrutura da página |
| [app/static/styles.css](../../app/static/styles.css) | Visual |
| [app/static/app.js](../../app/static/app.js) | Interações e gráficos |

Não há Node nem etapa de build: o servidor Python entrega os arquivos da página diretamente.
