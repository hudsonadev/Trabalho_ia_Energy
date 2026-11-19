# Aplicação: API e dashboard

Sobe com `uvicorn app.main:app` e abre em `http://localhost:8000`. O FastAPI serve a API em `/api/*` e os arquivos de `app/static/` na raiz. O front é uma página única que consome a API; cada interação atualiza a tela sem recarregar.

Uso local, um usuário: a série carregada fica em memória no servidor. Modelos e scaler são carregados uma vez na inicialização; se `models/` estiver vazio, a API responde 503 com a mensagem "Modelos não treinados. Execute `python -m src.train`."

## Endpoints

Erros sempre em `{"detail": "mensagem em português"}` com status 4xx.

### `POST /api/upload`
`multipart/form-data` com `arquivo` (CSV). Aceita dois formatos:
- **Bruto UCI:** separador `;`, colunas `Date`, `Time`, `Global_active_power`, …
- **Horário:** coluna de data/hora (`datetime`) + `Global_active_power` e, opcionalmente, as demais medições.

Passa pela mesma limpeza de `src/data.py`. Rejeita (400) se faltar a coluna alvo, se as datas forem ilegíveis ou se houver menos de 168 h.

Resposta:
```json
{"inicio": "2010-01-01T00:00:00", "fim": "2010-11-26T21:00:00", "horas": 7918,
 "faltantes_preenchidos": 42, "media_kw": 1.06, "pico_kw": 5.2}
```

### `GET /api/exemplo`
Carrega o trecho de teste do dataset como série da sessão. Mesma resposta do upload.

### `GET /api/serie?inicio=&fim=`
```json
{"datetime": ["..."], "kw": [1.2, 0.9]}
```
Sem parâmetros, devolve a série inteira.

### `POST /api/prever`
```json
{"modelo": "lstm", "data_referencia": "2010-06-15T00:00:00"}
```
`modelo`: `lstm` | `gru` | `baseline`. `data_referencia` é a primeira hora a prever; omitida = logo após o fim da série. Usa as 168 h anteriores. 400 se não houver 168 h antes da referência.

```json
{"modelo": "lstm",
 "horas": ["2010-06-15T00:00:00", "..."],
 "previsto": [0.81, "..."],
 "baseline": [0.77, "..."],
 "real": [0.85, "..."],
 "metricas": {"mae": 0.31, "rmse": 0.44, "wape": 0.28},
 "historico": {"datetime": ["..."], "kw": ["..."]}}
```
`real` e `metricas` são `null` quando a referência está além do fim da série. `historico` traz as 168 h usadas como entrada.

### `POST /api/custo`
```json
{"previsto": [0.81, "..."], "horas": ["..."],
 "tarifa": 0.85, "tarifa_ponta": 1.60,
 "ponta_inicio": 18, "ponta_fim": 21, "deslocar_pct": 20}
```
```json
{"custo_hora": [0.69, "..."], "custo_total": 21.4,
 "kwh_total": 24.9, "kwh_ponta": 5.1, "custo_ponta": 8.2,
 "custo_com_deslocamento": 20.6, "economia": 0.8}
```
Média horária em kW durante 1 h = kWh. O deslocamento move `deslocar_pct`% do consumo das horas de ponta, distribuído igualmente entre as horas fora de ponta.

### `POST /api/alertas`
```json
{"horas": ["..."], "previsto": ["..."], "real": null,
 "limite_kw": 3.0, "margem_anomalia_pct": 50}
```
```json
{"picos": [{"hora": "...", "kw": 3.4, "excedente_kw": 0.4}],
 "anomalias": [{"hora": "...", "real_kw": 4.1, "previsto_kw": 2.0, "desvio_pct": 105,
                "sugestao": "Consumo muito acima do esperado às 14h — verifique equipamentos ligados."}]}
```
`anomalias` fica vazio quando `real` é `null`.

### `GET /api/desempenho`
Conteúdo de `models/metrics.json`: métricas de teste dos três modelos, erro por horizonte e histórico de perdas.

## Dashboard (`app/static/`)

HTML/CSS/JS puro, sem build. Plotly.js por CDN (cdnjs, versão fixa). Interface em português, responsiva, com tema claro/escuro. Estado vazio amigável antes de carregar dados; indicador de carregamento durante chamadas; erros da API exibidos de forma visível.

1. **Entrada de dados** — arrastar/soltar CSV ou botão "Usar dados de exemplo". Mostra período, horas e faltantes preenchidos.
2. **Cartões de resumo** — consumo previsto nas próximas 24 h (kWh), pico previsto e seu horário, custo estimado, variação contra as 24 h anteriores. Reagem a toda interação.
3. **Máquina do tempo** — controle deslizante + seletor de data sobre o histórico. Ao escolher um momento, mostra as 168 h de entrada e a previsão das 24 h seguintes. Botão "Revelar o real" sobrepõe o que aconteceu e mostra MAE/RMSE/WAPE daquele dia. Botões ◀ ▶ (±1 dia) e "Reproduzir" (avança dia a dia). Seletor LSTM / GRU, com o baseline sempre visível para comparação.
4. **Simulador de custo e tarifa** — campos de preço do kWh e tarifa de ponta, faixa de horário de ponta e controle "deslocar X% do consumo da ponta". Custo por hora (barras, ponta destacada), total e economia em R$ recalculam ao vivo.
5. **Alertas de pico e desperdício** — linha de limite (kW) arrastável no gráfico, espelhada em um campo numérico. Lista as horas previstas acima do limite com o excedente. Com o real revelado, marca horas de anomalia e mostra a sugestão.
6. **Tabela de valores previstos** — hora, previsto, baseline, real (quando houver), erro, custo. Ordenável, com exportação em CSV.
7. **Desempenho do modelo** — métricas de teste dos três modelos, erro por horizonte e curvas de treino.

Chamadas disparadas por controles deslizantes devem usar *debounce* (~150 ms). Custo e alertas dependem só da previsão já em tela, então não refaça `/api/prever` ao mudar tarifa ou limite.
