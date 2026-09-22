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
   source: GPT-5.6 Sol web search, Claude Sonnet 5 web search, and AskNews,
   Exa or Perplexity when their API keys are set.
2. **Forecast.** Two models from different families (GPT-5.6 Sol and Claude
   Sonnet 5) forecast every question from the same briefing. Binary
   probabilities are averaged, multiple-choice probabilities are averaged per
   option, and numeric and discrete distributions are combined by the
   pointwise median of their CDFs.
3. **Calibrate.** Optional Platt scaling on binary forecasts (off by default).
4. **Publish.** The forecast is submitted together with a private comment
   containing each model's forecast and reasoning.

Every model's forecast is also appended to `logs/forecasts.jsonl`, so each
model can be scored separately once questions resolve.

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
| `FORECAST_MODEL` | `openrouter/openai/gpt-5.6-sol` | forecasting model when the ensemble is off |
| `RESEARCH_MODEL` | `openrouter/openai/gpt-5.6-sol:online` | primary research model (web search) |
| `PARSER_MODEL` | `openrouter/openai/gpt-4o-mini` | extracts structured values from model output |
| `REASONING_EFFORT` | `high` | reasoning effort for forecasting models |
| `RESEARCH_REASONING` | `low` | reasoning effort for research calls |
| `ENSEMBLE_MODELS` | GPT-5.6 Sol, Claude Sonnet 5 | comma-separated ensemble |
| `ENSEMBLE` | on | `0` disables the ensemble |
| `SHADOW_MODELS` | none | models that forecast every question but are only recorded, never published, as `model[@effort]`, e.g. `openrouter/openai/gpt-5.4` |
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

## Measuring results

```bash
uv run python analyze_results.py                    # uses the Metaculus API only, no LLM cost
uv run python analyze_results.py --fetch-artifacts  # also scores shadow models from Actions runs
```

For every resolved question the bot forecast, it scores the published
forecast and each model separately (Brier for binary and multiple choice,
10th-90th percentile coverage for numeric), shows the Metaculus scores, and
writes `logs/calibration.csv`.

## Calibration

```bash
uv run python calibration.py logs/calibration.csv
```

It fits the Platt coefficients and recommends them only when they beat the
raw forecasts in leave-one-out cross-validation. It needs at least 20 resolved
binary questions, and around 100 to detect a real miscalibration.

## Evaluation scripts

These spend API credits and publish nothing.

| Script | What it measures |
|---|---|
| `eval_post_cutoff.py` | Brier score on post-training-cutoff questions from the [BTF-3](https://huggingface.co/datasets/BTF-2/BTF-3) dataset, with and without research |
| `memorization_probe.py` | whether a model remembers the outcomes of past questions |
| `research_settings_check.py` | cost and content of the research step under different reasoning settings |

## Files

| File | Contents |
|---|---|
| `bot.py` | the bot: research, forecasting for each question type, aggregation |
| `main.py` | command line, model configuration, tournament selection, cost reporting |
| `calibration.py` | Platt scaling and coefficient fitting |
| `analyze_results.py` | scores resolved forecasts, overall and per model |
| `.github/workflows/forecast.yml` | scheduled runs |

## License

MIT. See [LICENSE](LICENSE).
