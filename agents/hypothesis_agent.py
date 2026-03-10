"""
Hypothesis Generation Agent
────────────────────────────
Generates testable hypotheses from EDA findings and performs
statistical tests (chi-square, Mann-Whitney U, ANOVA, Fisher's exact).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from agents.base_agent import BaseAgent
from utils.helpers import save_json, DATA_PROCESSED, section


ALPHA = 0.05


class HypothesisAgent(BaseAgent):
    name = "HypothesisAgent"

    def _description(self) -> str:
        return "Generating and statistically testing reproductive-performance hypotheses"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = context["df_cleaned"].copy()
        eda: dict = context.get("eda", {})

        hypotheses = self._define_hypotheses(df, eda)
        tested = self._test_all(df, hypotheses)

        save_json(tested, DATA_PROCESSED / "hypotheses_results.json")
        context["hypotheses"] = tested
        return context

    # ── hypothesis definitions ────────────────────────────────────────────────

    def _define_hypotheses(self, df: pd.DataFrame, eda: dict) -> list[dict]:
        """Return a list of hypothesis descriptors based on available columns."""
        available = set(df.columns)
        hyps: list[dict] = []

        # H1 – BCS vs conception
        if {"ecc", "prenhe"} <= available:
            hyps.append({
                "id": "H1",
                "description": "Animals with ECC ≥ 3.0 have higher conception rates than those with ECC < 3.0",
                "type": "chi_square",
                "factor": "ecc_group",
                "target": "prenhe",
            })

        # H2 – Season vs conception
        if {"estacao", "prenhe"} <= available:
            hyps.append({
                "id": "H2",
                "description": "Conception rate varies significantly by season (Verão / Outono / Inverno / Primavera)",
                "type": "chi_square",
                "factor": "estacao",
                "target": "prenhe",
            })

        # H3 – Parity vs conception
        if {"ordem_parto", "prenhe"} <= available:
            hyps.append({
                "id": "H3",
                "description": "Primiparous animals have lower conception rates than multiparous animals",
                "type": "chi_square",
                "factor": "grupo_paridade",
                "target": "prenhe",
            })

        # H4 – Breed vs conception
        if {"raca", "prenhe"} <= available:
            hyps.append({
                "id": "H4",
                "description": "Conception rate differs significantly among breeds",
                "type": "chi_square",
                "factor": "raca",
                "target": "prenhe",
            })

        # H5 – BCS vs days open (Mann-Whitney)
        if {"ecc", "dias_abertos"} <= available:
            hyps.append({
                "id": "H5",
                "description": "Animals with ECC ≥ 3.0 have fewer days open than those with ECC < 3.0",
                "type": "mann_whitney",
                "factor": "ecc_group",
                "metric": "dias_abertos",
            })

        # H6 – Technician effect
        if {"tecnico", "prenhe"} <= available and df["tecnico"].nunique() > 1:
            hyps.append({
                "id": "H6",
                "description": "Inseminator (technician) identity significantly affects conception rate",
                "type": "chi_square",
                "factor": "tecnico",
                "target": "prenhe",
            })

        # H7 – Season vs calving interval
        if {"estacao", "iep"} <= available:
            hyps.append({
                "id": "H7",
                "description": "Calving interval differs across seasons",
                "type": "anova",
                "factor": "estacao",
                "metric": "iep",
            })

        # H8 – Farm effect
        if {"fazenda", "prenhe"} <= available and df["fazenda"].nunique() > 1:
            hyps.append({
                "id": "H8",
                "description": "Conception rate differs significantly across farms",
                "type": "chi_square",
                "factor": "fazenda",
                "target": "prenhe",
            })

        return hyps

    # ── statistical tests ─────────────────────────────────────────────────────

    def _test_all(self, df: pd.DataFrame, hypotheses: list[dict]) -> list[dict]:
        results = []
        for hyp in hypotheses:
            section(f"{hyp['id']} — {hyp['description']}")
            try:
                result = self._test(df, hyp)
            except Exception as exc:
                result = {**hyp, "error": str(exc), "conclusion": "Erro ao testar"}
            results.append(result)
            self._print_result(result)
        return results

    def _test(self, df: pd.DataFrame, hyp: dict) -> dict:
        df = self._add_derived(df)
        test_type = hyp["type"]

        if test_type == "chi_square":
            return self._chi_square(df, hyp)
        elif test_type == "mann_whitney":
            return self._mann_whitney(df, hyp)
        elif test_type == "anova":
            return self._anova(df, hyp)
        raise ValueError(f"Unknown test type: {test_type}")

    def _add_derived(self, df: pd.DataFrame) -> pd.DataFrame:
        if "ecc" in df.columns and "ecc_group" not in df.columns:
            df["ecc_group"] = (df["ecc"] >= 3.0).map({True: "ECC≥3.0", False: "ECC<3.0"})
        if "grupo_paridade" not in df.columns and "ordem_parto" in df.columns:
            df["grupo_paridade"] = pd.cut(
                df["ordem_parto"], bins=[0, 1, 3, 100],
                labels=["Primípara", "Plurípara (2-3)", "Plurípara (4+)"],
            )
        return df

    def _chi_square(self, df: pd.DataFrame, hyp: dict) -> dict:
        factor, target = hyp["factor"], hyp["target"]
        sub = df[[factor, target]].dropna()
        contingency = pd.crosstab(sub[factor].astype(str), sub[target])
        chi2, p, dof, _ = stats.chi2_contingency(contingency)

        group_rates = sub.groupby(factor, observed=True)[target].mean().round(4).to_dict()
        cramers_v = float(np.sqrt(chi2 / (len(sub) * (min(contingency.shape) - 1))))

        return {
            **hyp,
            "n": len(sub),
            "chi2": round(float(chi2), 4),
            "p_value": round(float(p), 6),
            "dof": int(dof),
            "cramers_v": round(cramers_v, 4),
            "group_rates": group_rates,
            "significant": p < ALPHA,
            "conclusion": (
                f"REJEITADA H0 (p={p:.4f} < {ALPHA}) — {hyp['description']}"
                if p < ALPHA
                else f"Não rejeitada H0 (p={p:.4f} ≥ {ALPHA})"
            ),
        }

    def _mann_whitney(self, df: pd.DataFrame, hyp: dict) -> dict:
        factor, metric = hyp["factor"], hyp["metric"]
        groups = df[factor].dropna().unique()
        if len(groups) < 2:
            return {**hyp, "conclusion": "Insufficient groups"}

        a, b = groups[0], groups[1]
        ga = df.loc[df[factor] == a, metric].dropna()
        gb = df.loc[df[factor] == b, metric].dropna()
        u, p = stats.mannwhitneyu(ga, gb, alternative="two-sided")

        return {
            **hyp,
            "groups": {str(a): {"mean": round(float(ga.mean()), 2), "n": len(ga)},
                       str(b): {"mean": round(float(gb.mean()), 2), "n": len(gb)}},
            "U_statistic": round(float(u), 2),
            "p_value": round(float(p), 6),
            "significant": p < ALPHA,
            "conclusion": (
                f"REJEITADA H0 (p={p:.4f}) — {str(a)} vs {str(b)} difere em '{metric}'"
                if p < ALPHA
                else f"Não rejeitada H0 (p={p:.4f})"
            ),
        }

    def _anova(self, df: pd.DataFrame, hyp: dict) -> dict:
        factor, metric = hyp["factor"], hyp["metric"]
        groups_data = [
            g[metric].dropna().values
            for _, g in df.groupby(factor)
            if len(g[metric].dropna()) >= 5
        ]
        if len(groups_data) < 2:
            return {**hyp, "conclusion": "Insufficient data for ANOVA"}

        f, p = stats.f_oneway(*groups_data)
        group_means = df.groupby(factor)[metric].mean().round(2).to_dict()

        return {
            **hyp,
            "F_statistic": round(float(f), 4),
            "p_value": round(float(p), 6),
            "group_means": group_means,
            "significant": p < ALPHA,
            "conclusion": (
                f"REJEITADA H0 (p={p:.4f}) — '{metric}' varia entre grupos de '{factor}'"
                if p < ALPHA
                else f"Não rejeitada H0 (p={p:.4f})"
            ),
        }

    # ── display ───────────────────────────────────────────────────────────────

    @staticmethod
    def _print_result(r: dict) -> None:
        sig = "✦ SIGNIFICATIVO" if r.get("significant") else "  não significativo"
        print(f"  p-value   : {r.get('p_value', 'N/A')}")
        print(f"  Resultado : {sig}")
        print(f"  Conclusão : {r.get('conclusion', '')}")

    def _summarize(self, context: dict[str, Any]) -> None:
        hyps = context.get("hypotheses", [])
        n_sig = sum(1 for h in hyps if h.get("significant"))
        print(f"\n  {n_sig}/{len(hyps)} hypotheses confirmed significant (α={ALPHA})")
