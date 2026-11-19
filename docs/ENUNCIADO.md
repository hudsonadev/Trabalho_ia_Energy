# Enunciado do trabalho

## Objetivo

Desenvolver a capacidade de estruturar, treinar, validar e interpretar modelos de Deep Learning aplicados a problemas reais, utilizando datasets públicos ou replicáveis, com foco na construção de experimentos funcionais e análise crítica dos resultados obtidos.

## Instruções gerais

O aluno deverá:

- Utilizar um dataset público, aberto ou replicável;
- Explicar a origem e organização dos dados;
- Implementar um modelo de Deep Learning;
- Definir métricas coerentes de avaliação;
- Treinar e validar o modelo;
- Interpretar criticamente os resultados obtidos.

O projeto deverá conter:

- Contextualização do problema;
- Objetivo da solução;
- Arquitetura/modelagem utilizada;
- Processo de treinamento;
- Métricas utilizadas;
- Resultados obtidos.

## CASE 3 — Previsão de Consumo de Energia

Uma empresa deseja prever o consumo energético para reduzir desperdícios e melhorar planejamento operacional.

**Pergunta:** Explique como você estruturaria séries temporais e aplicaria Deep Learning para prever consumo futuro de energia.

Desenvolva uma aplicação que:

- receba dados históricos;
- execute a previsão;
- exiba os valores previstos;
- apresente gráficos comparativos.

Datasets sugeridos:

- Household Power Consumption
- Energy Consumption Dataset

## Mapa: exigência → onde é atendida

| Exigência | Entregável |
|---|---|
| Origem e organização dos dados | README (seção Dados) + notebook (EDA) |
| Resposta à "Pergunta" (estruturar séries temporais) | README (seção Estruturação da série temporal) |
| Modelo de Deep Learning | `src/models.py` (LSTM, GRU) |
| Métricas coerentes | `src/evaluate.py` + README (seção Métricas) |
| Treinar e validar | `src/train.py` + notebook |
| Interpretação crítica | README (seção Análise crítica) |
| Receber dados históricos | Dashboard: upload de CSV / dados de exemplo |
| Executar a previsão | `POST /api/prever` |
| Exibir valores previstos | Dashboard: tabela hora a hora + cartões |
| Gráficos comparativos | Dashboard: real × LSTM × GRU × baseline |
