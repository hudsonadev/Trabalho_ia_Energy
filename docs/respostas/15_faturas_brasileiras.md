# 15. E se enviarmos faturas de energia brasileiras ao sistema?

**Pergunta:** se mandarmos ao sistema faturas de energia brasileiras (por exemplo, 6 contas =
6 meses), como elas serão interpretadas, já que o modelo foi treinado com dados de outro país?

**Resposta curta:** **o modelo não interpreta faturas.** O sistema recusa o arquivo logo de
início. E mesmo que a fatura seja convertida para o formato aceito, o modelo não estaria
analisando o consumo real da casa, e sim uma curva inventada.

---

## 1. O que o sistema espera receber

O modelo responde a uma pergunta bem específica:

> "Dado o consumo **hora a hora** dos últimos **7 dias**, como será o consumo **hora a hora**
> nas próximas **24 horas**?"

Por isso o upload (função `ler_csv_usuario`, em [src/data.py](../../src/data.py)) exige:

| Requisito | Exemplo |
|---|---|
| Uma coluna de data e hora | `2026-10-01 00:00:00`, `2026-10-01 01:00:00`, ... |
| Uma coluna de consumo em kW | `Global_active_power` = `0,27`, `0,22`, ... |
| Pelo menos **168 horas** seguidas | 7 dias × 24 horas |

## 2. O que uma fatura traz

Uma conta de luz traz **um único número de consumo por mês**, por exemplo **379 kWh**. Ela
não diz em que hora nem em que dia esse consumo aconteceu.

| | Fatura | O que o modelo precisa |
|---|---|---|
| 1 mês | **1 número** | **~730 números** (um por hora) |
| 6 meses | **6 números** | **~4.380 números** |
| Unidade | kWh no mês | kW em cada hora |

**O que acontece na prática:** o upload é recusado com uma mensagem de erro (falta a coluna de
data e hora, ou o histórico é curto demais). Não há como a rede "ler" 6 totais mensais: ela
foi construída para sequências de 168 horas.

## 3. E se transformarmos a fatura em horas?

É possível "espalhar" o total do mês pelas horas usando uma curva típica de um dia (pouco
consumo de madrugada, pico à noite). Isso já foi feito no projeto: o arquivo
`data/consumo_estimado_2026-10.csv` distribui uma fatura de **379 kWh** (outubro/2026) em
**744 horas**.

Analisando o arquivo:

- a soma das horas dá exatamente **379 kWh**: o total da fatura está preservado;
- mas só existem **7 dias diferentes** (um molde por dia da semana), **repetidos o mês todo**.

Ou seja, é um perfil **inventado**, não uma medição. Ninguém sabe a que horas aquela casa
realmente consumiu energia.

### Teste com os modelos treinados

Rodamos a LSTM, a GRU e o baseline nesse arquivo, prevendo cada dia a partir dos 7 dias
anteriores (24 dias avaliados):

| Modelo | Erro médio (MAE) |
|---|---|
| **Baseline** (repete o dia anterior) | **0,086 kW** |
| LSTM | 0,098 kW |
| GRU | 0,115 kW |

**Aqui o baseline vence a LSTM**, o contrário do que acontece com dados reais. A explicação:
se o "consumo real" é um molde que se repete, copiar o dia anterior acerta quase tudo.

A LSTM até devolve uma previsão com cara de plausível (14,4 kWh para o dia seguinte, contra
14,0 kWh no último dia), mas ela está apenas **reproduzindo o molde que nós mesmos criamos**.
Não há nenhuma informação nova sobre os hábitos da casa.

**Conclusão:** converter a fatura faz o sistema **rodar**, mas não faz o modelo **analisar**
nada.

## 4. O problema da diferença de país

Mesmo com dados horários de verdade, há outro obstáculo: o modelo aprendeu os hábitos de **uma
casa específica na França**, entre 2006 e 2010.

| | Casa do treino (França) | Casa brasileira típica |
|---|---|---|
| Consumo médio | ~795 kWh/mês | a fatura do exemplo: 379 kWh/mês |
| Grandes cargas | aquecimento e aquecedor de água elétricos | chuveiro elétrico, ar-condicionado, geladeira |
| Tensão da rede | 240 V | 127 V ou 220 V |
| Picos de consumo | rotina e jantar no padrão francês | banho e cozinha, em geral entre 18h e 21h |
| Estação fria | dezembro a fevereiro | junho a agosto |

O que isso causa:

1. **O formato do dia é "francês".** A rede aprendeu em que horas aquela família consumia mais.
   No teste acima, a previsão manteve o pico até 21h–23h, um hábito do treino.
2. **As outras medições são estimadas com valores da França.** O modelo usa 7 medições (tensão,
   corrente, cozinha, lavanderia etc.). Se o arquivo só tem o consumo, as outras são preenchidas
   com **médias da casa francesa**. O dashboard avisa quando isso acontece.
3. **A escala é diferente.** A casa do treino consumia cerca do dobro da casa da fatura. O
   modelo viu poucas situações de consumo tão baixo de forma contínua.
4. **As estações invertidas pesam menos do que parece.** O modelo não recebe o mês, só a hora,
   o dia da semana e os últimos 7 dias. O problema maior são os **hábitos e equipamentos**, não
   o calendário.

**Para prever bem uma casa brasileira, o modelo precisaria ser treinado de novo (ou pelo menos
ajustado) com dados horários de casas brasileiras.** É a limitação "uma única residência",
listada em [10_pontos_fracos.md](10_pontos_fracos.md).

## 5. O que dá para fazer com 6 faturas

| Objetivo | É possível? | Como |
|---|---|---|
| Usar o modelo atual (LSTM/GRU) | ❌ | Faltam dados horários |
| Treinar uma rede neural só com 6 faturas | ❌ | 6 números são pouquíssimos para Deep Learning |
| Comparar os meses e achar o mês fora do padrão | ✅ | Análise simples, uma planilha resolve |
| Calcular média e tendência do consumo | ✅ | Idem |
| Estimar o custo com outra tarifa | ✅ | Total do mês × preço do kWh |
| Usar o modelo de verdade | ✅, com outro tipo de dado | Consumo **por hora**, vindo de medidor inteligente, do portal ou app da distribuidora (algumas oferecem a curva de carga) ou de um medidor instalado no quadro de energia |

## 6. Resumo para apresentar

> "O modelo prevê o consumo hora a hora a partir de 7 dias de medições. Uma fatura só tem o
> total do mês, então não serve como entrada: o sistema recusa o arquivo. Se espalharmos o total
> em horas com um perfil típico, o modelo roda, mas prevê um padrão que nós mesmos inventamos.
> Testamos isso e o baseline venceu a LSTM. Além disso, o modelo aprendeu os hábitos de uma casa
> francesa. Para atender casas brasileiras, precisaríamos de dados horários delas e de um novo
> treino."

---

> ⚠️ **Privacidade:** o arquivo `data/fatura_thopen_2026-10.csv` contém dados pessoais reais
> (nome, CPF, endereço e linha digitável do boleto). A pasta `data/` não vai para o Git
> (`.gitignore`), mas esse arquivo **não deve aparecer** em slides, prints ou vídeos da
> apresentação.
