# 10. Pontos fracos do projeto

Todo modelo tem limites, e mostrar que os conhecemos é parte da avaliação ("análise crítica"
no enunciado). Abaixo estão os pontos fracos reais do EnergyGreen, separados por área. Cada
item traz **o problema**, **por que acontece** e **como poderia ser melhorado**.

---

## A. Pontos fracos do modelo

### 1. Subestima os picos de consumo (o principal)
- **O problema:** nas horas de consumo muito alto (acima de 1,92 kW, os 10% maiores do
  teste), a LSTM erra em média **1,03 kW**, contra **0,36 kW** no restante. E erra sempre
  **para baixo** (viés de −1,01 kW): prevê menos do que realmente acontece.
- **Por quê:** o modelo aprende o "formato médio" do dia. Ele sabe que à noite o consumo
  sobe, mas não tem como saber se alguém vai ligar o forno às 19h10 ou às 20h40. Treinado com
  MSE, a aposta mais segura é suavizar a curva. Prova disso: a variação das previsões
  (desvio-padrão 0,51 kW) é bem menor que a do consumo real (0,71 kW).
- **Impacto:** os alertas de pico do dashboard podem deixar passar picos reais.
- **Melhoria:** prever **faixas** (ex.: "entre 1,2 e 2,8 kW") com perda quantílica, em vez
  de um único valor.

### 2. O ganho sobre o baseline é moderado
- **O problema:** a LSTM erra 14,3% menos que o baseline ("repetir o dia anterior"). É um
  ganho real, mas não enorme. E o baseline ainda **vence em 1 a cada 3 dias** (a LSTM ganha em
  139 de 208 dias).
- **Por quê:** o consumo de uma casa tem muito ruído que nenhum histórico explica.
- **Melhoria:** acrescentar informações externas (ver item 4).

### 3. Explica só 30% da variação do consumo
- **O problema:** o R² da LSTM é **0,305**. Ou seja, 70% da variação do consumo continua sem
  explicação.
- **Por quê:** uma única casa é muito mais imprevisível que uma cidade inteira, onde os
  hábitos individuais se compensam.
- **Observação:** para esse tipo de dado, o valor é coerente. O baseline tem R² negativo.

### 4. Não usa variáveis externas
- **O problema:** o modelo não sabe a **temperatura**, se é **feriado**, se a família está
  de **férias** ou quantas pessoas estão em casa.
- **Por quê:** o dataset não traz essas informações.
- **Impacto:** quedas e picos causados por clima ou férias (como a queda de consumo em
  agosto) pegam o modelo de surpresa.
- **Melhoria:** cruzar com dados meteorológicos e calendário de feriados da França.

### 5. Erra mais nos horários de início de atividade
- **O problema:** a LSTM vence o baseline em 22 das 24 horas, mas **perde às 5h e às 7h**.
  O maior erro é à noite (0,61 kW às 20h).
- **Por quê:** o horário em que a casa "acorda" varia de um dia para o outro.

### 6. A LSTM e a GRU ficaram praticamente empatadas
- **O problema:** a diferença entre as duas é pequena (MAE 0,426 contra 0,434). A LSTM é
  melhor na média, a GRU um pouco melhor nos picos. Não há uma vencedora clara.
- **Melhoria:** testar arquiteturas diferentes (com atenção, TCN, Transformer temporal).

---

## B. Pontos fracos dos dados

### 7. Uma única residência
- O modelo aprendeu os hábitos **de uma família**. Ele não serve para outra casa, uma empresa
  ou uma cidade sem ser treinado de novo com os dados delas.

### 8. Dados antigos (2006–2010)
- Os aparelhos e os hábitos mudaram muito (LED, carro elétrico, home office). O padrão de
  consumo atual é diferente.

### 9. Os períodos de validação e teste são diferentes entre si
- **O problema:** a validação cobre outono e inverno (consumo médio 1,25 kW); o teste cobre
  primavera e verão (0,94 kW); o treino tem média 1,09 kW.
- **Impacto:** a melhor época foi escolhida num período com comportamento diferente do teste.
- **Melhoria:** validação *walk-forward*, avaliando em vários períodos ao longo do ano.

---

## C. Pontos fracos do treinamento

### 10. Treino com 1 a cada 5 janelas
- **O problema:** para caber em ~10 minutos na CPU, o treino usou só **1/5 das janelas**
  possíveis (4.805 de cerca de 24 mil).
- **Melhoria:** treinar com todas as janelas numa máquina com GPU.

### 11. Sem busca de hiperparâmetros
- **O problema:** valores como 64 unidades, 2 camadas, dropout 0,2 e taxa de aprendizado
  0,001 foram escolhidos por serem padrões razoáveis, não por uma busca sistemática.
- **Melhoria:** testar combinações (busca em grade ou Optuna) usando a validação.

### 12. Uma única execução por modelo
- **O problema:** cada modelo foi treinado uma vez, com semente 42. Não sabemos quanto o
  resultado varia com outra semente. Como a diferença LSTM × GRU é pequena, ela pode inverter.
- **Melhoria:** treinar com várias sementes e reportar média e desvio.

---

## D. Pontos fracos da aplicação

### 13. Upload só com o consumo perde precisão
- **O problema:** o modelo usa 7 medições. Se o usuário enviar um CSV só com o consumo, a
  corrente é **estimada** e as outras medições recebem a **média do treino**.
- **Impacto:** a previsão pode piorar. O dashboard avisa quando isso acontece.

### 14. Depende de internet para os gráficos
- O Plotly.js e a fonte são carregados de CDN. Sem internet, a página abre mas **os gráficos
  não aparecem**. Para apresentar offline, seria preciso baixar o Plotly para `app/static/`.

### 15. Feito para demonstração, não para produção
- Não há login, banco de dados nem retreino automático com dados novos. A série carregada
  fica na memória do servidor, compartilhada entre quem acessa.

---

## Resumo para apresentar

> "O principal ponto fraco é que o modelo subestima os picos: ele acerta o ritmo do dia, mas
> não sabe exatamente quando alguém vai ligar um aparelho. O ganho sobre o baseline é real,
> mas moderado, e é limitado por termos uma única casa, dados antigos e nenhuma informação de
> clima ou feriados. Como melhorias, propomos incluir temperatura e feriados, prever faixas em
> vez de um único valor e treinar com todos os dados em GPU."
