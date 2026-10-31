# Q1 – Fraud detection under operational constraints

Python 3.12 · numpy 2.4.4 · pandas 3.0.2 · scikit-learn 1.8.0 (pinned in `requirements.txt`).

## Commands

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python run_q1.py                                  # parts b, c, d  -> outputs/run_output.txt
python -m unittest discover -s tests -t . -v      # 46 tests       -> outputs/test_output.txt
```

## Files

| Part | File | What it holds |
| --- | --- | --- |
| a | `data/smoke_fixture.csv` | The 12 transactions exactly as given (5 fraud, 7 legitimate) |
| b | `fraud_pipeline.py`, `config.py` | load, schema validation, cleaning, pipeline; all tunables in `config.py` (seed, columns, plausibility rules) |
| c | `metrics.py` | TN, precision, recall, F1, accuracy, all-legitimate baseline |
| d | `review.py` | `select_for_review(scores, ids, budget)` |
| b–d | `tests/` | 46 unit tests |
| e | `outputs/`, `MODEL_RISK_NOTE.md` | captured run and test output; 146-word risk note |

## (b) Design and assumptions

* **Validation** (`validate_schema`) fails loudly, listing every problem: missing columns, missing or duplicate IDs, labels outside {0, 1}, empty file. Extra columns are ignored so an unexpected column (say a chargeback flag) can never become a feature.
* **Cleaning** (`clean_features`) takes feature columns only, so it cannot see the label. Non-numeric, infinite, negative, over-range and fractional-count values become NaN and are counted in a report. No row is dropped and nothing is filled at this stage.
* **No leakage:** the median imputer and the scaler sit inside the sklearn `Pipeline`, so they learn only from the rows passed to `.fit()`. A test confirms held-out rows cannot change fitted statistics. Rows with an unusable label are rejected, not imputed.
* **Reproducibility:** fixed seed, deterministic solver, pinned versions, and two fits produce identical scores (tested).
* **Model:** `log1p` on amount and account age, median imputation with missing indicators, standardisation, L2 logistic regression with balanced class weights.
* **The fixture is a smoke test only.** The pipeline is fitted on all 12 rows to prove it runs; the printed scores are training scores, not evidence of quality. Every fraud has amount ≥ 4,000 and every legitimate row ≤ 2,500, so the data is separable on one column.

### Real data needed before the model can be evaluated

1. **Volume at realistic prevalence:** enough transactions that the test set holds hundreds of fraud cases (at ~1% prevalence that means tens of thousands of transactions or more), so precision and recall have usable confidence intervals.
2. **Time-ordered split:** train on earlier periods, validate and test on later ones, never shuffled, with the final test set touched once.
3. **Matured labels:** chargebacks and confirmed fraud arrive weeks later; recent transactions with unresolved outcomes must be excluded or flagged, otherwise fraud is undercounted.
4. **Label quality and selection bias:** how labels were produced, and whether only reviewed or blocked transactions were labelled.
5. **Richer features:** merchant, device, geography, velocity, customer history. Features must be available at decision time (no post-event fields).
6. **Segment coverage:** new accounts, high-value, cross-border, channels, so performance can be checked per segment.
7. **Operating inputs:** review capacity, cost of a missed fraud versus a false alert, and the latency limit, which set the threshold and budget.

## (c) Held-out batch (from `outputs/run_output.txt`)

TN = 10,000 − 80 − 20 − 320 = **9,580**

| | Model | All-legitimate baseline |
| --- | --- | --- |
| TP / FP / FN / TN | 80 / 320 / 20 / 9,580 | 0 / 0 / 100 / 9,900 |
| Precision | 0.2000 | n/a (no alerts) |
| Recall | 0.8000 | 0.0000 |
| F1 | 0.3200 | 0.0000 |
| Accuracy | 0.9660 | **0.9900** |

The baseline has higher accuracy and catches no fraud, which is why accuracy is the wrong headline metric at 1% prevalence. The model raises 400 alerts (4.0 false alerts per fraud found). Undefined ratios are NaN rather than a silent 0.

## (d) `select_for_review` rules

* Descending score; **ties broken by ascending ID**, so output does not depend on input order. Numeric IDs compare numerically (2 before 10); non-comparable mixed types fall back to a deterministic text comparison.
* **Unique IDs:** a repeated ID is one transaction, ranked by its highest score, so the result can be shorter than `budget`.
* `budget=0` returns `[]`; a budget above the batch size returns every unique ID.
* **Raises** on negative budget, non-integer or bool budget, length mismatch, NaN or inf scores, missing IDs and non-1-D scores. A NaN score is an upstream failure and skipping it would hide a transaction from review.

## Known limits

* A transaction whose features are all invalid is scored from imputed medians only; it will look average, not risky. Monitor the cleaning report for such rows.
* Missingness at scoring time in a column that never had missing values in training is imputed silently (no indicator was learned).
* Class-weighted scores rank risk but are not calibrated probabilities.
