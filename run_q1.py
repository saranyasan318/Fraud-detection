
import pandas as pd

import config
import fraud_pipeline as fp
from metrics import all_legitimate_baseline, classification_metrics, fmt, tn_from_totals
from review import select_for_review

FIXTURE = "data/smoke_fixture.csv"


def section(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")

section("Q1(b) Pipeline smoke test on the 12-row fixture  (NOT a performance estimate)")
r = fp.smoke_test(FIXTURE)
print(f"rows={r['rows']}  fraud={r['frauds']}  legitimate={r['legitimate']}")
print(f"schema valid: unique ids, binary labels  -> OK")
print(f"values blanked by cleaning: {sum(v['non_numeric'] + v['infinite'] + v['out_of_range'] + v['non_integer'] for v in r['cleaning_report'].values())}")
print(f"two independent fits give identical scores: {r['deterministic']}")
print("scores are training-set scores, shown only to prove the pipeline runs end to end:")
print(r["scores"].round(3).to_string())

section("Q1(b) Cleaning demo on a deliberately dirty batch")
dirty = pd.DataFrame({
    "id": ["D1", "D2", "D3", "D4", "D5"],
    "amount": ["250", "-40", "abc", None, "1e400"],
    "account_age_days": ["30", "12", "99999", "15", "7"],
    "failed_logins": ["0", "2.5", "1", "3", None],
    "is_fraud": ["0", "1", "0", "1", "0"],
})
clean_X, report = fp.clean_features(fp.validate_schema(dirty))
print(clean_X.to_string())
print(pd.DataFrame(report).T.to_string())
print("rows kept:", len(clean_X), "of", len(dirty), "(nothing dropped, nothing filled before the pipeline)")

section("Q1(c) Held-out batch: N=10,000, TP=80, FN=20, FP=320")
tn = tn_from_totals(10_000, tp=80, fn=20, fp=320)
model = classification_metrics(tp=80, fp=320, fn=20, tn=tn)
base = all_legitimate_baseline(10_000, positives=model["tp"] + model["fn"])
print(f"TN = 10000 - 80 - 20 - 320 = {tn}")
rows = {"Model": model, "All-legitimate baseline": base}
table = pd.DataFrame({
    name: {"TP": m["tp"], "FP": m["fp"], "FN": m["fn"], "TN": m["tn"],
           "Precision": fmt(m["precision"]), "Recall": fmt(m["recall"]),
           "F1": fmt(m["f1"]), "Accuracy": fmt(m["accuracy"]),
           "Alerts raised": m["alerts"], "Fraud caught": m["tp"]}
    for name, m in rows.items()})
print(table.to_string())
print(f"\nPrevalence {model['prevalence']:.2%}. Baseline accuracy {base['accuracy']:.2%} beats the "
      f"model's {model['accuracy']:.2%} while catching 0 of {base['fn']} frauds.")
print(f"Model workload: {model['alerts']} alerts ({model['alert_rate']:.1%} of batch) to catch "
      f"{model['tp']} frauds -> {model['fp'] / model['tp']:.1f} false alerts per fraud found.")


section("Q1(d) select_for_review demo")
ids = ["T3", "T1", "T2", "T4", "T5"]
scores = [0.90, 0.90, 0.90, 0.40, 0.95]
for budget in (0, 2, 4, 50):
    print(f"budget={budget:<3} -> {select_for_review(scores, ids, budget)}")
print("fixture scores, budget=3 ->", select_for_review(r["scores"].to_numpy(), r["scores"].index, 3))
