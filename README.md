# Metaculus Forecasting Bot

An automated forecasting bot for the [Metaculus FutureEval](https://www.metaculus.com/futureeval/)
AI benchmark tournaments: the seasonal bot tournament (Fall 2026) and the
biweekly MiniBench. It finds open questions, researches them, forecasts, and
publishes each forecast with its reasoning. Built on Metaculus's
[forecasting-tools](https://github.com/Metaculus/forecasting-tools) library.

## How it works

For each open question (when publishing, questions it has already forecast are
skipped):

1. **Research.** Several independent search providers run in parallel and
   their results are combined into one briefing, each block labeled with its
   source: GPT-5.4 web search, Claude web search, and AskNews, Exa or
   Perplexity when their API keys are set.
2. **Forecast.** Binary questions are forecast by two models from different
   families (GPT-5.4 and Claude Sonnet 4.6), each reading the same briefing,
   and the two probabilities are averaged. Numeric, discrete and
   multiple-choice questions are forecast by GPT-5.4.
3. **Calibrate.** Optional Platt scaling on binary forecasts (off by default).
4. **Publish.** The forecast is submitted together with a private comment
   containing the reasoning.

OpenAI and Anthropic models are called through OpenRouter. Without an
OpenRouter key the bot falls back to the Metaculus LLM proxy, authenticated
with `METACULUS_TOKEN`.

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
cp .env.example .env
```

Fill in `.env`:

- `METACULUS_TOKEN`: create a bot account under
  [account settings](https://www.metaculus.com/accounts/settings/) >
  "My Forecasting Bots" and copy its API key.
- `OPENROUTER_API_KEY`: tournament participants can request free credits
  through [this form](https://forms.gle/aQdYMq9Pisrf1v7d8), or create a key at
  [openrouter.ai](https://openrouter.ai/keys).
- Optional: `ASKNEWS_API_KEY`, `EXA_API_KEY`, `PERPLEXITY_API_KEY` to add
  research providers.

## Usage

```bash
# Dry run on the Metaculus bot testing area (nothing is published)
uv run python main.py --mode test

# Dry run on the live tournaments
uv run python main.py --mode tournament

# Forecast and publish
uv run python main.py --mode tournament --publish
```

Without `--publish` the bot runs the full pipeline but submits nothing.

| Flag | Meaning |
|---|---|
| `--mode` | `test`, `tournament` (seasonal tournament + MiniBench), `minibench`, `cup`, `market_pulse` |
| `--publish` | submit forecasts and comments to Metaculus |
| `--limit N` | forecast at most N questions, alternating question types |
| `--samples N` | run the forecast step N times per question (default 1) |

## Configuration

All optional; defaults live in `main.py`. See `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `FORECAST_MODEL` | `openrouter/openai/gpt-5.4` | main forecasting model |
| `RESEARCH_MODEL` | `openrouter/openai/gpt-5.4:online` | primary research model (web search) |
| `PARSER_MODEL` | `openrouter/openai/gpt-4o-mini` | extracts structured values from model output |
| `REASONING_EFFORT` | `high` | reasoning effort for forecasting models |
| `RESEARCH_REASONING` | `low` | reasoning effort for research calls |
| `ENSEMBLE_MODELS` | GPT-5.4, Claude Sonnet 4.6 | comma-separated ensemble for binary questions |
| `ENSEMBLE` | on | `0` disables the ensemble |
| `RESEARCH_PROVIDERS` | auto | force a provider list, e.g. `asknews,anthropic-search` |
| `MAX_COST_PER_RUN` | `5.00` | cost cap per run in USD (web search cost is not tracked) |
| `CALIBRATION_A`, `CALIBRATION_B` | `1.0`, `0.0` | Platt scaling coefficients |

## Running on GitHub Actions

`.github/workflows/forecast.yml` runs the bot on a schedule. Each scheduled
run checks for new questions every 20 minutes for about five and a half hours
and publishes to the seasonal tournament and MiniBench; the next run takes over
when it ends. Add `METACULUS_TOKEN` and `OPENROUTER_API_KEY` (plus any optional
provider keys) as repository secrets. Calibration coefficients go in as
repository variables. Manual runs from the Actions tab let you choose the mode
and whether to publish.

## Calibration

Once you have resolved binary questions, put them in a CSV with the columns
`prediction,outcome` and run:

```bash
uv run python calibration.py resolved.csv
```

It fits the Platt coefficients (minimum 20 questions) and prints the two
lines to add to `.env` when the fit improves the Brier score.

## Evaluation scripts

These spend API credits and publish nothing.

| Script | What it measures |
|---|---|
| `eval_limpo.py` | Brier score on post-training-cutoff questions from the [BTF-3](https://huggingface.co/datasets/BTF-2/BTF-3) dataset, with and without research |
| `probe_memorizacao.py` | whether a model remembers the outcomes of past questions |
| `teste_pesquisa.py` | cost and content of the research step under different reasoning settings |

## Files

| File | Contents |
|---|---|
| `bot.py` | the bot: research, forecasting for each question type, aggregation |
| `main.py` | command line, model configuration, tournament selection, cost reporting |
| `calibration.py` | Platt scaling and coefficient fitting |
| `.github/workflows/forecast.yml` | scheduled runs |
