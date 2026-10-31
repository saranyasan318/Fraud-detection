# Model-risk note

This is a smoke-tested prototype, not a validated fraud model. Twelve rows (five fraud) cannot support any estimate of precision, recall or generalisation, and the fixture is separable on amount alone, so any score from it would be misleadingly optimistic. Do not deploy it or quote its metrics.

The held-out counts show why accuracy misleads: at 1% prevalence an all-legitimate rule scores 99.0% accuracy against the model's 96.6% while catching no fraud. The model finds 80% of fraud, but only 20% of its alerts are genuine, so review capacity drives cost.

Key risks: label delay and selection bias (only reviewed cases get labels), drift as fraudsters adapt, leakage from post-event features, and unfair impact on new accounts, because account age is a strong feature.

Before release: out-of-time validation, thresholds tied to review budget and loss costs, drift and recall monitoring, human override, and named retraining ownership.
