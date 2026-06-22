"""
Gradient-boosted fraud classifier.
Trains on synthetic imbalanced data and persists via joblib.
"""
from __future__ import annotations
import numpy as np
import joblib
from pathlib import Path
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

from app.schemas import TransactionIn

FEATURE_COLS = [
    "amount", "hour_of_day", "day_of_week", "is_online",
    "distance_from_home_km", "velocity_last_1h", "velocity_last_24h",
    "account_age_days", "avg_spend_30d",
]

THRESHOLD = 0.5


class FraudModel:
    def __init__(self):
        self._pipeline: Pipeline | None = None

    @property
    def is_trained(self) -> bool:
        return self._pipeline is not None

    def _tx_to_row(self, tx: TransactionIn) -> list[float]:
        return [
            tx.amount,
            tx.hour_of_day,
            tx.day_of_week,
            float(tx.is_online),
            tx.distance_from_home_km,
            tx.velocity_last_1h,
            tx.velocity_last_24h,
            tx.account_age_days,
            tx.avg_spend_30d,
        ]

    def predict(self, tx: TransactionIn) -> tuple[float, bool]:
        row = np.array([self._tx_to_row(tx)])
        prob = float(self._pipeline.predict_proba(row)[0, 1])
        return prob, prob >= THRESHOLD

    def train_on_synthetic(self, n_samples: int = 20_000) -> dict:
        rng = np.random.default_rng(42)

        # Legitimate transactions
        n_legit = int(n_samples * 0.97)
        X_legit = np.column_stack([
            rng.lognormal(3.5, 1.0, n_legit),   # amount
            rng.integers(8, 21, n_legit),         # hour (daytime)
            rng.integers(0, 7,  n_legit),         # day
            rng.integers(0, 2,  n_legit),         # is_online
            rng.exponential(5,  n_legit),         # distance
            rng.poisson(1,      n_legit),         # vel 1h
            rng.poisson(4,      n_legit),         # vel 24h
            rng.integers(90, 2000, n_legit),      # account age
            rng.lognormal(3.5, 0.8, n_legit),    # avg spend
        ])

        # Fraudulent transactions â€” higher amount, unusual hours, high velocity
        n_fraud = n_samples - n_legit
        X_fraud = np.column_stack([
            rng.lognormal(5.5, 1.5, n_fraud),
            rng.choice([0,1,2,3,22,23], n_fraud),
            rng.integers(0, 7, n_fraud),
            rng.integers(1, 2, n_fraud),          # usually online
            rng.exponential(200, n_fraud),
            rng.poisson(8,  n_fraud),
            rng.poisson(25, n_fraud),
            rng.integers(0, 60, n_fraud),
            rng.lognormal(2.0, 1.0, n_fraud),
        ])

        X = np.vstack([X_legit, X_fraud])
        y = np.concatenate([np.zeros(n_legit), np.ones(n_fraud)])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        self._pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", GradientBoostingClassifier(
                n_estimators=200, max_depth=4, learning_rate=0.05,
                subsample=0.8, min_samples_leaf=20, random_state=42,
            )),
        ])
        self._pipeline.fit(X_train, y_train)

        probs = self._pipeline.predict_proba(X_test)[:, 1]
        auc   = roc_auc_score(y_test, probs)
        print(classification_report(y_test, probs >= THRESHOLD, target_names=["legit", "fraud"]))
        return {"roc_auc": auc, "n_train": len(X_train), "n_test": len(X_test)}

    def save(self, path: Path):
        joblib.dump(self._pipeline, path)

    def load(self, path: Path):
        self._pipeline = joblib.load(path)