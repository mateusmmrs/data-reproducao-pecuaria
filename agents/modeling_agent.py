"""
Machine Learning Modeling Agent
─────────────────────────────────
Builds predictive models for conception rate:
  • Logistic Regression (interpretable baseline)
  • Random Forest Classifier (non-linear)
  • Gradient Boosting Classifier (performance)

Also estimates feature importances and SHAP-style partial effects.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from agents.base_agent import BaseAgent
from utils.helpers import save_json, DATA_PROCESSED, OUTPUTS_MODELS, section


FEATURE_CANDIDATES = [
    "ecc", "ordem_parto", "servicos_concepcao", "dias_abertos",
    "producao_leite_kg", "dias_pos_parto",
]
CATEGORICAL_CANDIDATES = ["raca", "estacao", "grupo_paridade", "ecc_categoria", "fazenda", "tecnico"]
TARGET = "prenhe"


class ModelingAgent(BaseAgent):
    name = "ModelingAgent"

    def _description(self) -> str:
        return "Building Logistic Regression, Random Forest, and Gradient Boosting models"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = context["df_cleaned"].copy()

        if TARGET not in df.columns:
            print("  ⚠  Target variable 'prenhe' not found — skipping modeling.")
            context["modeling"] = {}
            return context

        X, y, feature_names = self._prepare_features(df)
        results: dict[str, Any] = {}

        section("Logistic Regression")
        results["logistic_regression"] = self._train_eval(
            LogisticRegression(max_iter=1000, random_state=42),
            X, y, feature_names, model_name="logistic_regression",
        )

        section("Random Forest")
        results["random_forest"] = self._train_eval(
            RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
            X, y, feature_names, model_name="random_forest",
        )

        section("Gradient Boosting")
        results["gradient_boosting"] = self._train_eval(
            GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, random_state=42),
            X, y, feature_names, model_name="gradient_boosting",
        )

        section("Model Comparison")
        comparison = self._compare(results)
        results["comparison"] = comparison
        print(pd.DataFrame(comparison).T.to_string())

        # Best model feature importances
        best_name = max(comparison, key=lambda k: comparison[k].get("roc_auc", 0))
        results["best_model"] = best_name
        print(f"\n  ✦ Best model: {best_name} (AUC = {comparison[best_name]['roc_auc']:.4f})")

        save_json(results, OUTPUTS_MODELS / "model_results.json")
        context["modeling"] = results
        context["feature_names"] = feature_names
        return context

    # ── feature engineering ────────────────────────────────────────────────────

    def _prepare_features(
        self, df: pd.DataFrame
    ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        num_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
        cat_cols = [c for c in CATEGORICAL_CANDIDATES if c in df.columns]

        X_parts = []
        feature_names: list[str] = []

        # Numeric
        if num_cols:
            X_num = df[num_cols].copy()
            X_num = X_num.apply(lambda s: s.fillna(s.median()))
            X_parts.append(X_num.values.astype(float))
            feature_names.extend(num_cols)

        # Categorical (label-encoded for tree models, dummy for LR)
        for col in cat_cols:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].astype(str).fillna("Unknown"))
            X_parts.append(encoded.reshape(-1, 1).astype(float))
            feature_names.append(col)

        X = np.hstack(X_parts) if X_parts else np.zeros((len(df), 1))
        y = df[TARGET].fillna(0).astype(int).values

        print(f"  Features ({len(feature_names)}): {feature_names}")
        print(f"  Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
        return X, y, feature_names

    # ── training & evaluation ──────────────────────────────────────────────────

    def _train_eval(
        self,
        model,
        X: np.ndarray,
        y: np.ndarray,
        feature_names: list[str],
        model_name: str,
    ) -> dict:
        scaler = StandardScaler()
        pipe = Pipeline([("scaler", scaler), ("model", model)])

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_auc = cross_val_score(pipe, X, y, cv=cv, scoring="roc_auc")

        # Final fit on full data for feature importances
        pipe.fit(X, y)
        y_pred = pipe.predict(X)
        y_prob = pipe.predict_proba(X)[:, 1]

        auc = roc_auc_score(y, y_prob)
        cm = confusion_matrix(y, y_pred).tolist()
        report = classification_report(y, y_pred, output_dict=True)

        print(f"  CV AUC (5-fold): {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
        print(f"  Train AUC      : {auc:.4f}")
        print(classification_report(y, y_pred))

        # Feature importances (tree) or coefficients (LR)
        importances = self._get_importances(pipe["model"], feature_names)

        return {
            "cv_auc_mean": round(float(cv_auc.mean()), 4),
            "cv_auc_std":  round(float(cv_auc.std()), 4),
            "roc_auc":     round(float(auc), 4),
            "confusion_matrix": cm,
            "classification_report": report,
            "feature_importances": importances,
        }

    @staticmethod
    def _get_importances(model, feature_names: list[str]) -> dict[str, float]:
        if hasattr(model, "feature_importances_"):
            imp = model.feature_importances_
        elif hasattr(model, "coef_"):
            imp = np.abs(model.coef_[0])
        else:
            return {}
        total = imp.sum() or 1.0
        return {
            name: round(float(val / total), 4)
            for name, val in sorted(
                zip(feature_names, imp), key=lambda x: -x[1]
            )
        }

    @staticmethod
    def _compare(results: dict) -> dict:
        comparison = {}
        for model_name, res in results.items():
            if model_name == "comparison":
                continue
            comparison[model_name] = {
                "cv_auc":  res.get("cv_auc_mean"),
                "roc_auc": res.get("roc_auc"),
            }
        return comparison

    def _summarize(self, context: dict[str, Any]) -> None:
        best = context.get("modeling", {}).get("best_model")
        if best:
            auc = context["modeling"]["comparison"][best]["roc_auc"]
            print(f"\n  Best: {best} | AUC = {auc}")
