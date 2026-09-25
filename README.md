# NBA Match Predictor

Predicts NBA game outcomes by scraping raw box scores, engineering a rolling-average feature set, and backtesting a Ridge classifier season-by-season — the way the model would actually be used, forecasting games it has never seen.

## What it does

- **Collects** 17,600+ real NBA box scores spanning 7 seasons (2020–2026) with Python, Playwright, and BeautifulSoup
- **Transforms** raw HTML into a structured, 150+ feature dataset with pandas (team and opponent shooting splits, advanced stats, rolling 10-game team averages, home/away context)
- **Trains and validates** a Ridge classifier with cross-validated feature selection
- **Backtests season-by-season**: the model is trained only on seasons before the one it's tested on, so results reflect real forecasting performance rather than a model that has already seen the answer

## Result

**63%+ accuracy** forecasting game outcomes on unseen future seasons.

## Tech stack

Python · pandas · scikit-learn · Playwright · BeautifulSoup · Jupyter

## Repo structure

| Notebook | Purpose |
|---|---|
| `get_data.ipynb` | Scrapes raw box score HTML for every game in the target seasons |
| `parse_data.ipynb` | Parses the scraped HTML into a clean, structured per-game dataset |
| `predict.ipynb` | Builds rolling features, trains the Ridge classifier, and runs the season-by-season backtest |

## Running it

1. Run `get_data.ipynb` to scrape box scores (this takes a while — it's fetching thousands of individual game pages)
2. Run `parse_data.ipynb` to turn the scraped HTML into `nba_games.csv`
3. Run `predict.ipynb` to build features and reproduce the backtested accuracy

## Notes

The backtesting methodology is the main thing that makes this result trustworthy: every prediction is made using only data available *before* that game was played, with no leakage from future seasons into training data.
