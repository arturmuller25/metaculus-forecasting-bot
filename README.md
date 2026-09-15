# Bot de previsão para os torneios da Metaculus

Bot que prevê eventos reais nos torneios FutureEval da Metaculus. Premiação de
US$ 50 mil por temporada, três temporadas por ano, mais o MiniBench de US$ 1 mil
a cada duas semanas. Não se aposta capital próprio, e a Metaculus banca os
créditos de LLM dos participantes.

O torneio corrente é o **Fall 2026** (ID 33121), aberto desde setembro.

## Por que este projeto existe

É a única arena verificada onde a evidência diz que LLM tem habilidade real de
previsão e onde dá para ganhar dinheiro sem arriscar capital. Previsão de
eventos em dias ou semanas é diferente de adivinhar direção de preço em 15
minutos, que é onde a pesquisa anterior deste repositório não achou vantagem
nenhuma.

Expectativa honesta: no Q2 de 2025, os bots hobbistas ficaram cerca de 20 pontos
atrás dos forecasters profissionais, e a projeção de paridade fica entre
novembro de 2026 e junho de 2027. Ou seja, dá para competir, não dá para
esperar vitória fácil.

## Instalação

Requer Python 3.12 e [uv](https://docs.astral.sh/uv/).

```bash
cd metaculus-bot
uv sync
cp .env.example .env
```

Depois preencha o `.env`:

1. **`METACULUS_TOKEN`**: crie uma conta em metaculus.com, vá em
   [configurações](https://www.metaculus.com/accounts/settings/), clique em
   "My Forecasting Bots" e depois "Create a Bot", e copie a API Key.
2. **`OPENROUTER_API_KEY`**: peça os créditos gratuitos do torneio pelo
   [formulário](https://forms.gle/aQdYMq9Pisrf1v7d8). O crédito chega como
   chave da OpenRouter. Se preferir pagar do bolso,
   [gere uma chave](https://openrouter.ai/keys) e defina um limite de gasto nela.

## Uso

Sem chave de provedor o bot usa o proxy de LLM da própria Metaculus, que
autentica com o `METACULUS_TOKEN`. Dá para testar o encanamento inteiro antes
de ter crédito, mas a pesquisa sai sem busca web.

```bash
# Smoke test na área de testes de bots. Não publica nada.
uv run python main.py --mode test

# Torneio principal + MiniBench, ainda sem publicar.
uv run python main.py --mode tournament

# Para valer.
uv run python main.py --mode tournament --publish
```

Sem `--publish` o bot calcula tudo e grava os relatórios em `logs/`, mas não
envia nada. Rode assim nas primeiras vezes e leia o raciocínio antes de soltar
na competição.

Modos disponíveis: `test`, `tournament`, `minibench`, `cup`, `market_pulse`.

## Como funciona

Por pergunta, a biblioteca oficial `forecasting-tools` executa:

1. `run_research` uma vez, com um modelo que tem busca web ligada.
2. `_run_forecast_on_*` cinco vezes sobre essa mesma pesquisa.
3. Agrega as cinco previsões.
4. Aplica a calibração, se você tiver configurado.
5. Publica, se `--publish` estiver ligado.

O bot cobre os três tipos de pergunta que o torneio usa: binária, múltipla
escolha e numérica.

### As decisões de projeto, e o que as sustenta

Tudo aqui vem das análises públicas do torneio, não de palpite. Onde a
evidência é fraca ou conflitante, isso está dito.

**O modelo que decide importa mais que o andaime.** Na tabela de correlação da
Metaculus, usar GPT-5.x como modelo de previsão final dá r=+0,42, o sinal
isolado mais forte e o único que se repete em duas temporadas. Claude Opus, na
mesma tabela, dá r=-0,01. Por isso o decisor padrão é GPT-5.4. Se for otimizar
uma coisa só, otimize essa linha.

**Ensemble precisa ser de modelos diferentes, não do mesmo modelo repetido.**
Um experimento controlado em 202 perguntas deste torneio mediu os dois:
modelos heterogêneos melhoraram o Brier de 0,162 para 0,153 (p=0,014), e três
instâncias do mesmo modelo não mudaram nada. O consenso usa média aparada,
descartando os extremos, que é o método descrito pelo vencedor do Q4 2024.

Ressalva honesta: a tabela da Spring 2026 pontua "agregar previsões" em
r=-0,19, e a Metaculus avisa que nenhum resultado daquela temporada sobrevive
à correção de comparações múltiplas. Evidência conflitante. Por isso existe o
interruptor `ENSEMBLE=0` no `.env`. Meça antes de acreditar em qualquer um dos
dois estudos.

**Calibração é o único pós-processamento com significância real.** Platt
scaling tem p=0,00052 em perguntas binárias. Já extremização manual, que
parece a mesma coisa, pontua r=-0,30. Este bot faz a primeira e não faz a
segunda.

**Simplicidade é uma decisão de projeto.** O tema mais repetido na pesquisa
com os 37 participantes da Spring 2026 é que complexidade atrapalhou. Um
deles caiu da 20ª para a 63ª posição depois de adicionar agregação em
log-odds, extremização e mistura com a multidão. Outro removeu uma
arquitetura de 13 papéis. Este bot é deliberadamente pequeno.

### Duas proteções que vieram de erro alheio

**Alarme de 50% exato.** Um participante perdeu a temporada inteira porque o
bot caiu num fallback silencioso e passou a enviar 0,5 em tudo. Meio por cento
cravado quase nunca é conclusão de raciocínio, quase sempre é parser quebrado.
O bot grita no log quando isso acontece.

**As duas convenções de resolução da Metaculus.** Assumir que o evento não
aconteceu quando a pesquisa não mostra que aconteceu, e entender que "antes da
data X" é pergunta sobre o futuro. Ignorar a primeira custou 90 pontos de peer
score a um participante. Ambas estão no prompt.

### Calibração

Começa desligada, em `A=1, B=0`. Não copie coeficiente de ninguém: os seus
dependem do seu modelo e dos seus prompts.

Depois de acumular perguntas binárias resolvidas, monte um CSV assim:

```csv
prediction,outcome
0.73,1
0.20,0
0.55,1
```

E rode:

```bash
uv run python calibration.py resolvidas.csv
```

O script recusa amostra abaixo de 20 perguntas e avisa quando o ajuste não
melhora nada. Se melhorar, ele imprime as duas linhas para colar no `.env`.

Leitura dos coeficientes: `A > 1` significa que o bot é subconfiante e as
previsões deveriam ir mais para os extremos; `A < 1` significa o contrário;
`B` diferente de zero corrige viés sistemático para Yes ou No.

## Rodando sozinho no GitHub Actions

O arquivo `.github/workflows/forecast.yml` roda de hora em hora e publica.
O bot pula perguntas que já respondeu, então a maior parte das execuções não
gasta quase nada.

Para ativar: suba o repositório para o GitHub, vá em Settings, depois Secrets
and variables, depois Actions, e cadastre `METACULUS_TOKEN` e
`OPENROUTER_API_KEY` como secrets. Se você já tiver calibrado, cadastre
`CALIBRATION_A` e `CALIBRATION_B` como *variables*, não como secrets.

A execução manual pela aba Actions deixa você escolher o modo e se publica ou
não. A execução agendada sempre publica no torneio principal.

## Custos

O teto de `MAX_COST_PER_RUN` no `.env` aborta a execução ao estourar, mas não
cobre a etapa de pesquisa: modelos com o sufixo `:online` não reportam custo
para a biblioteca. O limite que realmente segura o gasto é o que você configura
na própria chave da OpenRouter. Configure lá também.

Se o sufixo `:online` der erro no seu provedor, tire ele do `RESEARCH_MODEL`.
O bot continua funcionando, só pesquisa pior.

## Arquivos

| Arquivo | O que faz |
|---|---|
| `bot.py` | a classe do bot: pesquisa, os três tipos de forecast, agregação |
| `main.py` | CLI, escolha de modelos, teto de custo, seleção de torneio |
| `calibration.py` | Platt scaling e o ajuste dos coeficientes a partir de resolvidas |
| `.github/workflows/forecast.yml` | execução automática de hora em hora |

## Origem

Os prompts seguem de perto o template oficial da Metaculus, de propósito: é o
baseline que a casa usa e mede. Vale mudar depois de ter placar próprio, não
antes.
