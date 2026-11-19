# EnergyGreen — Previsão de Consumo de Energia com Deep Learning

Trabalho acadêmico (CASE 3). O projeto está **estruturado mas não implementado**: os arquivos em `src/`, `app/` e `tests/` são esqueletos com os contratos. Sua tarefa é construir tudo seguindo os documentos abaixo.

## Leia antes de começar

1. [docs/ENUNCIADO.md](docs/ENUNCIADO.md) — o que o professor exige (critérios de avaliação).
2. [docs/PLANO.md](docs/PLANO.md) — decisões técnicas, ordem de construção e critérios de aceite.
3. [docs/API.md](docs/API.md) — contrato da API e especificação do dashboard.

## Decisões já fechadas com o usuário (não reabrir)

- **Dataset:** Individual Household Electric Power Consumption (UCI, id 235), agregado por hora.
- **Framework:** PyTorch em CPU. A máquina tem Python 3.14.7, sem GPU NVIDIA. Não usar TensorFlow.
- **Modelos:** LSTM e GRU, comparados a um baseline sazonal ingênuo (repete as 24 h do dia anterior).
- **Janela:** 168 h de entrada → 24 h de saída (multi-passo direto).
- **Aplicação:** FastAPI + dashboard próprio em HTML/CSS/JS puro (sem Node, sem build), Plotly.js por CDN.
- **Recursos interativos obrigatórios:** máquina do tempo (replay), simulador de custo e tarifa, alertas de pico e desperdício. O usuário pediu explicitamente algo além de gráficos.
- **Parte escrita:** `README.md` com as seções do enunciado + `notebooks/experimento.ipynb` executado.

## Regras do projeto

- Tudo que o usuário final lê (interface, README, notebook, mensagens de erro) em **português do Brasil**. Código (nomes de funções e variáveis) pode ficar em inglês ou português, mas seja consistente dentro de cada arquivo.
- **Sem vazamento de dados:** divisão cronológica, scaler ajustado só no treino, nenhuma janela de teste sobreposta ao treino.
- **Números reais:** o README e o notebook só trazem métricas que saíram de execuções reais. Se LSTM/GRU não baterem o baseline, reporte isso e investigue — não maquie.
- O notebook e a API **reutilizam `src/`**; não duplicar lógica de pré-processamento.
- `data/` não é versionado; o download é feito por script.
- Semente fixa (42) em `random`, `numpy` e `torch`.

## Comandos

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt

python -m src.data        # baixa e prepara os dados
python -m src.train       # treina LSTM e GRU
python -m src.evaluate    # métricas e figuras
pytest                    # testes
uvicorn app.main:app --reload   # dashboard em http://localhost:8000
```

Se algum pacote não tiver wheel para Python 3.14, **pare e avise o usuário** propondo uma alternativa (ex.: Python 3.12 só para o venv) antes de trocar qualquer decisão acima.
