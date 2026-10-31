

SEED = 20260101                      

ID_COLUMN = "id"
LABEL_COLUMN = "is_fraud"
FEATURES = ["amount", "account_age_days", "failed_logins"]
REQUIRED_COLUMNS = [ID_COLUMN, *FEATURES, LABEL_COLUMN]

LOG_FEATURES = ["amount", "account_age_days"]
LINEAR_FEATURES = ["failed_logins"]


FEATURE_RULES = {
    "amount":           {"min": 0.0, "max": None,     "integer": False},
    "account_age_days": {"min": 0.0, "max": 36_500.0, "integer": False},   # 100 years
    "failed_logins":    {"min": 0.0, "max": None,     "integer": True},
}

# Classifier settings
LOGREG_C = 1.0
LOGREG_MAX_ITER = 1000
