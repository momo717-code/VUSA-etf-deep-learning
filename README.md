# 📈 Next-Day ETF Price-Direction Prediction — VUSA.L (Deep Learning)

A comparative deep-learning study that predicts the **next-day price direction** of the Vanguard S&P 500 UCITS ETF (**VUSA.L**), and tests a concrete question: **does proprietary Bloomberg Terminal data beat free public data?** Motivated by a personal £30,000 holding in the ETF.

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)

> MSc *Deep Learning & Optimisation* project (CS51011), University of Dundee.

---

## The question

Can a model predict whether VUSA.L closes **up or down tomorrow** — and is that easier with **proprietary Bloomberg data** (which adds a 30-day implied-volatility field) than with **free Yahoo Finance data**? Three model families are trained on both datasets, giving six experimental configurations measured against a majority-class baseline.

## Approach

- **Three architectures, compared head-to-head:**
  - **Logistic Regression** (scikit-learn) — linear baseline.
  - **WiderMLP** (PyTorch) — feed-forward net with BatchNorm, Dropout, and the Adam optimiser.
  - **LSTM** (PyTorch) — sequence model over 10-day windows to capture temporal dependence.
- **Two datasets:** public `yfinance` (21 features) vs. proprietary **Bloomberg** export (26 features, incl. `Volatility_30D`). Data to 12 March 2026.
- **Feature engineering** (`pandas_ta`): RSI, MACD, Bollinger Bands (±2σ, 20-day), ATR-14, plus Bloomberg's forward-looking implied-volatility measure.
- **Honest evaluation:** `StandardScaler`, early stopping (patience 12), and both classification metrics **and** a monetary backtest with transaction costs.

## Results

| Model | Data | Val Acc | Test Acc | F1 |
|---|---|:--:|:--:|:--:|
| Logistic Regression | Public | — | 51.8% | 0.643 |
| Logistic Regression | **Bloomberg** | — | **52.8%** | 0.664 |
| WiderMLP | Public | 55.0% | 51.4% | 0.679 |
| **WiderMLP** | **Bloomberg** | **58.6%** | 52.1% | 0.685 |
| LSTM | Public | 48.8% | 49.8% | 0.658 |
| LSTM | Bloomberg | 51.8% | 51.2% | 0.674 |

**Key finding:** the **Bloomberg WiderMLP** reached the highest validation accuracy at **58.6%** — a **+3.6 pp** improvement over the same model on public data — and Bloomberg inputs improved *every* model. The edge is real but modest: next-day equity direction is close to a random walk, so the honest takeaway is that proprietary volatility data provides a **small, consistent** signal rather than a money-printing one. Reporting that straight is the point of the study.

## Run it

The project was built in Google Colab.

```bash
git clone https://github.com/momo717-code/vusa-etf-deep-learning.git
cd vusa-etf-deep-learning
pip install torch scikit-learn pandas numpy matplotlib seaborn yfinance pandas_ta openpyxl
```

- **Public-data path** is fully reproducible — `yfinance` downloads VUSA.L automatically.
- **Bloomberg data is not included** (proprietary / licensed). The `Volatility_30D` experiments require your own Bloomberg export named `vusa_bloomberg_data.csv.xlsx`.

Open `vusa_etf_deep_learning.ipynb` (Colab/Jupyter) or run `vusa_etf_deep_learning.py` cell-by-cell. Full write-up in **`VUSA_Deep_Learning_Report.pdf`**.

## What I learned

- Building and tuning **MLP and LSTM** classifiers in PyTorch with BatchNorm, Dropout, and early stopping.
- Designing a **controlled comparison** (same pipeline, different data) to isolate the value of a feature source.
- **Evaluating honestly** — pairing accuracy/F1 with a transaction-cost-aware backtest, and reporting a modest result truthfully.

---

*⚠️ Educational project, not investment advice.*
