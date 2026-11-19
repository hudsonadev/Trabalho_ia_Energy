# Receitas do projeto — EnergyGreen

Identidade aprovada em 2026-11-19. A fonte da verdade são as variáveis no topo de `app/static/styles.css`.

## Paleta ("Laboratório")

| Papel | Claro | Escuro |
|---|---|---|
| Fundo | `#f6f7f8` | `#0e1113` |
| Superfície (só o quadro do gráfico principal, campos e toasts) | `#ffffff` | `#171b1e` |
| Poço / hover | `#eceef0` | `#22272b` |
| Texto / texto 2 / texto mudo | `#0f1418` / `#47515a` / `#66717b` | `#edf1f4` / `#b7c0c8` / `#8a949d` |
| Marca e foco (botão principal, faixa "próximas 24 h") | `#00875a` | `#2dbd85` |
| Crítico (acima do limite) | `#c23434` | `#e66767` |
| Alerta (anomalia, horário de ponta) | `#d99200` | `#e0a21a` |

Séries, com o mesmo papel em todos os gráficos e tabelas: LSTM azul `#2a6fc9`, GRU laranja `#d9582b`, baseline cinza pontilhado, real = cor do texto, histórico = cinza claro.

## Fontes

Archivo (Google Fonts), pesos 400 e 600, com a fonte do sistema como reserva. Tamanhos: 12 / 13 / 15 / 18 / 28 / 48. Números sempre tabulares.

## Superfícies

Uma só superfície elevada na tela: o quadro do gráfico da máquina do tempo (fundo superfície, borda 1px, raio 10, sem sombra). Todo o resto vive direto no fundo, separado por espaço e por fios de 1px. Sombra só no toast, que flutua.

## Componentes

- **Leitura:** rótulo 13px, número 48px (a notícia), dois apoios de 18px.
- **Manchete de seção:** rótulo + número 28px + uma frase com os valores exatos; o gráfico vem depois e os controles por último.
- **Segmentado LSTM/GRU:** o traço colorido é a cor da série no gráfico.
- **Botões:** primário (um por estado), texto (ações secundárias), ícone (SVG de traço 1,75).
- **Lista de alertas:** linhas com fio, marcador igual ao do gráfico (círculo vermelho = acima do limite, triângulo âmbar = anomalia).
- **Tabela:** sem caixa, cabeçalho mudo, números à direita; melhor valor em peso 600, sem estrela.

## Regras locais

- Espaço na escala 4 / 8 / 12 / 16 / 24 / 32 / 48; 48 entre grupos. Raios 6 (controles) e 10 (quadro).
- Gráficos em Plotly.js, cores lidas das variáveis CSS e fonte lida do `body`.
- Ícones em SVG próprio; nada de glifo de texto.
- Funções obrigatórias que não podem sumir: máquina do tempo, simulador de custo e tarifa, alertas de pico e desperdício.
