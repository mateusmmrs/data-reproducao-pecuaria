"""
Data Visualization Agent
─────────────────────────
Produces professional static (matplotlib/seaborn) charts for reproductive
analysis. Each chart has didactic titles, benchmark references, and
data-source annotations for presentation and LinkedIn sharing.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

from agents.base_agent import BaseAgent
from utils.helpers import OUTPUTS_PLOTS, section


# ── Style ─────────────────────────────────────────────────────────────────────
PALETTE = sns.color_palette("Set2")
sns.set_theme(style="whitegrid", palette="Set2", font_scale=1.05)
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "#F8F8F8",
    "axes.edgecolor":   "#CCCCCC",
    "grid.color":       "#EEEEEE",
    "font.family":      "DejaVu Sans",
})

# Benchmarks Embrapa / CNA / ABIEC (usados como linhas de referência)
BENCH = {
    "taxa_concepcao_alvo":  0.60,    # 60% — meta Embrapa Gado de Corte
    "taxa_concepcao_media": 0.55,    # 55% — média nacional ABIEC 2023
    "dias_abertos_alvo":    110,     # 110 d — meta CNA
    "dias_abertos_media":   145,     # 145 d — média nacional
    "iep_ideal":            365,     # 365 d — 1 bezerra/vaca/ano
    "iep_media_nacional":   430,     # 430 d — média Brasil (Embrapa 2023)
    "ecc_ideal":            3.0,     # ECC ideal ao serviço (Embrapa)
}

SOURCE_SYNTH = "Dados sintéticos calibrados | Benchmarks: Embrapa / CNA / ABIEC 2023"
SOURCE_IBGE  = "Fonte: IBGE SIDRA — PPM (Pesquisa da Pecuária Municipal)"


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUTS_PLOTS / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✔ {path.name}")
    return path


def _add_source(fig: plt.Figure, source: str = SOURCE_SYNTH) -> None:
    """Adiciona nota de rodapé com a fonte dos dados."""
    fig.text(0.01, -0.01, source, fontsize=7.5, color="#888888",
             ha="left", va="bottom", style="italic")


class VisualizationAgent(BaseAgent):
    name = "VisualizationAgent"

    def _description(self) -> str:
        return "Generating reproductive-analysis visualizations"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        df: pd.DataFrame = context["df_cleaned"].copy()
        eda: dict = context.get("eda", {})
        modeling: dict = context.get("modeling", {})
        economic: dict = context.get("economic", {})
        hypotheses: list = context.get("hypotheses", [])

        saved_plots: list[str] = []

        section("1. Conception Rate by Group")
        saved_plots += self._conception_by_group(df)

        section("2. BCS Distribution vs Conception")
        saved_plots += self._bcs_vs_conception(df)

        section("3. Parity vs Conception")
        saved_plots += self._parity_vs_conception(df)

        section("4. Correlation Heatmap")
        saved_plots += self._correlation_heatmap(df)

        section("5. Calving Interval Distribution")
        saved_plots += self._iep_distribution(df)

        section("6. Feature Importances")
        saved_plots += self._feature_importances(modeling)

        section("7. Economic Summary")
        saved_plots += self._economic_bar(economic)

        section("8. Hypothesis p-values")
        saved_plots += self._hypothesis_chart(hypotheses)

        section("9. Temporal Trends")
        saved_plots += self._temporal_trends(df)

        section("10. Season vs Key Metrics")
        saved_plots += self._season_metrics(df)

        context["plots"] = saved_plots
        return context

    # ── individual charts ─────────────────────────────────────────────────────

    def _conception_by_group(self, df: pd.DataFrame) -> list[str]:
        plots = []
        group_cols = [c for c in ("raca", "estacao", "fazenda", "tecnico") if c in df.columns]
        if "prenhe" not in df.columns or not group_cols:
            return plots

        n_cols = min(2, len(group_cols))
        n_rows = (len(group_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(7 * n_cols, 5 * n_rows))
        axes = np.array(axes).flatten()

        global_mean = df["prenhe"].mean() * 100

        label_map = {
            "raca": "Raça",
            "estacao": "Estação do Ano",
            "fazenda": "Fazenda",
            "tecnico": "Técnico / Inseminador",
        }

        for idx, col in enumerate(group_cols):
            ax = axes[idx]
            grp = df.groupby(col)["prenhe"].mean().sort_values(ascending=False) * 100
            colors = [("#2ECC71" if v >= BENCH["taxa_concepcao_alvo"] * 100 else
                       "#F39C12" if v >= BENCH["taxa_concepcao_media"] * 100 else
                       "#E74C3C") for v in grp.values]
            bars = ax.bar(grp.index, grp.values, color=colors, edgecolor="white", linewidth=0.8)
            ax.axhline(global_mean, color="#2C3E50", lw=1.5, ls="--",
                       label=f"Média do rebanho: {global_mean:.1f}%")
            ax.axhline(BENCH["taxa_concepcao_alvo"] * 100, color="#27AE60", lw=1.2, ls=":",
                       label=f"Meta Embrapa: {BENCH['taxa_concepcao_alvo']:.0%}")
            ax.set_title(f"Taxa de Concepção por {label_map.get(col, col)}", fontweight="bold")
            ax.set_ylabel("Taxa de Concepção (%)")
            ax.set_ylim(0, 105)
            ax.tick_params(axis="x", rotation=30)
            for bar, val in zip(bars, grp.values):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 1.2, f"{val:.1f}%",
                        ha="center", va="bottom", fontsize=8.5, fontweight="bold")
            ax.legend(fontsize=8)

        for ax in axes[len(group_cols):]:
            ax.set_visible(False)

        fig.suptitle(
            "Taxa de Concepção por Grupo\n"
            "Verde ≥ 60% (meta) · Amarelo ≥ 55% (média nacional) · Vermelho < 55%",
            fontsize=13, fontweight="bold", y=1.02,
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "01_concepcao_por_grupo.png")
        plots.append("01_concepcao_por_grupo.png")
        return plots

    def _bcs_vs_conception(self, df: pd.DataFrame) -> list[str]:
        plots = []
        if not {"ecc", "prenhe"} <= set(df.columns):
            return plots

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Esquerda: violin por resultado
        ax = axes[0]
        for val, label, color in [(0, "Não Prenhe", "#E74C3C"), (1, "Prenhe", "#27AE60")]:
            subset = df.loc[df["prenhe"] == val, "ecc"].dropna()
            parts = ax.violinplot(subset, positions=[val], showmedians=True, showextrema=True)
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
            parts["cmedians"].set_color(color)
        ax.axhline(BENCH["ecc_ideal"], color="#2C3E50", lw=1.5, ls="--",
                   label=f"ECC ideal ao serviço: {BENCH['ecc_ideal']:.1f} (Embrapa)")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Não Prenhe", "Prenhe"])
        ax.set_title(
            "Distribuição do ECC por Resultado Reprodutivo\n"
            "Vacas com maior ECC tendem a conceber mais",
            fontweight="bold",
        )
        ax.set_ylabel("Escore de Condição Corporal (1–5)")
        ax.legend(fontsize=8)

        # Direita: taxa de concepção por faixa de ECC
        ax2 = axes[1]
        df2 = df.copy()
        df2["ecc_bin"] = pd.cut(df2["ecc"], bins=np.arange(1, 5.5, 0.5))
        grp = (df2.groupby("ecc_bin", observed=True)["prenhe"]
               .agg(["mean", "count"]).reset_index())
        grp["mean"] *= 100
        colors2 = ["#27AE60" if v >= 60 else "#F39C12" if v >= 50 else "#E74C3C"
                   for v in grp["mean"]]
        bars = ax2.bar(range(len(grp)), grp["mean"], color=colors2, edgecolor="white")
        ax2.set_xticks(range(len(grp)))
        ax2.set_xticklabels([str(b) for b in grp["ecc_bin"]], rotation=45, ha="right", fontsize=8)
        ax2.axhline(60, color="#27AE60", lw=1.2, ls=":", label="Meta 60% (Embrapa)")
        ax2.set_title(
            "Taxa de Concepção por Faixa de ECC\n"
            "Cada barra mostra a taxa média de prenhez naquela faixa",
            fontweight="bold",
        )
        ax2.set_ylabel("Taxa de Concepção (%)")
        ax2.set_xlabel("Faixa de ECC")
        ax2.set_ylim(0, 100)
        for bar, (_, row) in zip(bars, grp.iterrows()):
            ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                     f"n={int(row['count'])}", ha="center", fontsize=7.5, color="#555")
        ax2.legend(fontsize=8)

        fig.suptitle(
            "Escore de Condição Corporal (ECC) × Concepção\n"
            "O ECC ao serviço é o principal preditor individual de prenhez em bovinos",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "02_ecc_vs_concepcao.png")
        plots.append("02_ecc_vs_concepcao.png")
        return plots

    def _parity_vs_conception(self, df: pd.DataFrame) -> list[str]:
        plots = []
        if not {"ordem_parto", "prenhe"} <= set(df.columns):
            return plots

        fig, axes = plt.subplots(1, 2, figsize=(13, 5))

        # Esquerda: scatter por ordem de parto
        ax = axes[0]
        grp = df.groupby("ordem_parto")["prenhe"].agg(["mean", "count"]).reset_index()
        grp["mean"] *= 100
        sc = ax.scatter(grp["ordem_parto"], grp["mean"],
                        s=grp["count"] / grp["count"].max() * 500,
                        c=grp["mean"], cmap="RdYlGn", vmin=30, vmax=90,
                        alpha=0.85, edgecolors="#888", linewidth=0.7)
        ax.plot(grp["ordem_parto"], grp["mean"], "--", color="#AAAAAA", lw=1, alpha=0.5)
        ax.axhline(BENCH["taxa_concepcao_alvo"] * 100, color="#27AE60", lw=1.2, ls=":",
                   label=f"Meta Embrapa: 60%")
        plt.colorbar(sc, ax=ax, label="Taxa de Concepção (%)")
        ax.set_title(
            "Taxa de Concepção por Ordem de Parto\n"
            "Tamanho do ponto proporcional ao número de animais",
            fontweight="bold",
        )
        ax.set_xlabel("Ordem de Parto (1 = primípara)")
        ax.set_ylabel("Taxa de Concepção (%)")
        ax.legend(fontsize=8)

        # Direita: grupo de paridade (barras)
        ax2 = axes[1]
        if "grupo_paridade" in df.columns:
            grp2 = (df.groupby("grupo_paridade", observed=True)["prenhe"]
                    .agg(["mean", "count"]).reset_index())
            grp2["mean"] *= 100
            bars = ax2.bar(grp2["grupo_paridade"].astype(str), grp2["mean"],
                           color=PALETTE[:len(grp2)], edgecolor="white")
            ax2.axhline(BENCH["taxa_concepcao_alvo"] * 100, color="#27AE60",
                        lw=1.2, ls=":", label="Meta 60%")
            ax2.set_title(
                "Taxa de Concepção por Grupo de Paridade\n"
                "Primíparas vs pluríparas jovens vs pluríparas de alta ordem",
                fontweight="bold",
            )
            ax2.set_ylabel("Taxa de Concepção (%)")
            ax2.set_ylim(0, 100)
            for bar, (_, row) in zip(bars, grp2.iterrows()):
                ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                         f"{row['mean']:.1f}%\n(n={row['count']:.0f})",
                         ha="center", fontsize=8.5, fontweight="bold")
            ax2.legend(fontsize=8)
        else:
            ax2.set_visible(False)

        fig.suptitle(
            "Efeito da Paridade na Taxa de Concepção\n"
            "Pluríparas de 2–3 partos geralmente apresentam melhor desempenho reprodutivo",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "03_paridade_vs_concepcao.png")
        plots.append("03_paridade_vs_concepcao.png")
        return plots

    def _correlation_heatmap(self, df: pd.DataFrame) -> list[str]:
        plots = []
        num = df.select_dtypes("number")
        if num.shape[1] < 3:
            return plots

        # Nomes amigáveis para as colunas
        rename = {
            "prenhe": "Prenhez",
            "ecc": "ECC",
            "ordem_parto": "Ord. Parto",
            "servicos_concepcao": "Serv./Conc.",
            "producao_leite_kg": "Leite (kg/d)",
            "dias_abertos": "Dias Abertos",
            "iep": "IEP (dias)",
            "pev": "PEV (dias)",
            "dias_pos_parto": "Dias pós-parto",
            "ano": "Ano",
        }
        num_renamed = num.rename(columns=rename)
        corr = num_renamed.corr()
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)

        fig, ax = plt.subplots(figsize=(11, 9))
        sns.heatmap(
            corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
            vmin=-1, vmax=1, center=0, square=True, linewidths=0.5,
            annot_kws={"size": 9}, ax=ax,
        )
        ax.set_title(
            "Matriz de Correlação — Variáveis Numéricas\n"
            "Valores próximos de ±1 indicam forte relação linear entre as variáveis",
            fontweight="bold", pad=15,
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "04_correlacao.png")
        plots.append("04_correlacao.png")
        return plots

    def _iep_distribution(self, df: pd.DataFrame) -> list[str]:
        plots = []
        # Gera dois subplots: IEP e Dias Abertos (se disponíveis)
        available = [(c, lbl, alvo, media) for c, lbl, alvo, media in [
            ("iep",         "Intervalo entre Partos (dias)", BENCH["iep_ideal"],    BENCH["iep_media_nacional"]),
            ("dias_abertos","Dias Abertos",                  BENCH["dias_abertos_alvo"], BENCH["dias_abertos_media"]),
        ] if c in df.columns]

        if not available:
            return plots

        fig, axes = plt.subplots(1, len(available), figsize=(10 * len(available), 5))
        if len(available) == 1:
            axes = [axes]

        for ax, (col, label, alvo, media_nac) in zip(axes, available):
            data = df[col].dropna()
            ax.hist(data, bins=40, color="#5C85D6", edgecolor="white", alpha=0.85, label="Distribuição")
            ax.axvline(data.mean(), color="#E74C3C", lw=2,
                       label=f"Média do rebanho: {data.mean():.0f} d")
            ax.axvline(alvo, color="#27AE60", lw=2, ls=":",
                       label=f"Meta Embrapa: {alvo} d")
            ax.axvline(media_nac, color="#F39C12", lw=1.8, ls="--",
                       label=f"Média nacional: {media_nac} d")
            # Área de alerta (excesso)
            ymax = ax.get_ylim()[1] if ax.get_ylim()[1] > 0 else data.value_counts().max()
            ax.axvspan(alvo, data.max() * 1.05, alpha=0.05, color="#E74C3C")
            ax.set_title(
                f"Distribuição — {label}\n"
                f"Área vermelha = dias acima da meta (perdas econômicas)",
                fontweight="bold",
            )
            ax.set_xlabel(label)
            ax.set_ylabel("Nº de Animais")
            ax.legend(fontsize=8)

        fig.suptitle(
            "Distribuição de Dias Abertos e Intervalo entre Partos\n"
            "Cada dia acima da meta representa custo operacional direto (R$ 5–10/animal/dia)",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "05_distribuicao_iep.png")
        plots.append("05_distribuicao_iep.png")
        return plots

    def _feature_importances(self, modeling: dict) -> list[str]:
        plots = []
        if not modeling:
            return plots

        best = modeling.get("best_model")
        if not best:
            return plots

        importances = modeling.get(best, {}).get("feature_importances", {})
        if not importances:
            return plots

        # Nomes legíveis
        label_map = {
            "ecc": "ECC (Condição Corporal)",
            "ordem_parto": "Ordem de Parto",
            "servicos_concepcao": "Serviços/Concepção",
            "producao_leite_kg": "Produção de Leite",
            "dias_abertos": "Dias Abertos",
            "iep": "IEP (Intervalo Parto)",
            "pev": "Período Espera Voluntário",
            "dias_pos_parto": "Dias Pós-Parto",
        }
        names = [label_map.get(k, k.replace("_", " ").title()) for k in importances.keys()]
        vals  = list(importances.values())

        # Ordenar por importância
        pairs = sorted(zip(names, vals), key=lambda x: x[1])
        names, vals = zip(*pairs) if pairs else ([], [])

        colors = ["#27AE60" if v == max(vals) else
                  "#2980B9" if v >= max(vals) * 0.5 else
                  "#95A5A6" for v in vals]

        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(names, vals, color=colors, edgecolor="white")
        ax.set_title(
            f"Variáveis mais Importantes para Predição de Prenhez\n"
            f"Modelo: {best.replace('_', ' ').title()} | AUC = "
            f"{modeling.get('comparison', {}).get(best, {}).get('roc_auc', '—')}",
            fontweight="bold",
        )
        ax.set_xlabel("Importância Relativa (normalizada)")
        for i, (name, val) in enumerate(zip(names, vals)):
            ax.text(val + 0.002, i, f"{val:.3f}", va="center", fontsize=9)

        # Legenda de cores
        patches = [
            mpatches.Patch(color="#27AE60", label="Mais importante"),
            mpatches.Patch(color="#2980B9", label="Importância média"),
            mpatches.Patch(color="#95A5A6", label="Menor importância"),
        ]
        ax.legend(handles=patches, fontsize=8, loc="lower right")
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "06_feature_importances.png")
        plots.append("06_feature_importances.png")
        return plots

    def _economic_bar(self, economic: dict) -> list[str]:
        plots = []
        if not economic:
            return plots

        labels = {
            "days_open":       "Dias Abertos\nExcessivos",
            "conception_rate": "Baixa Taxa de\nConcepção",
            "services":        "Serviços de IA\nAdicionais",
            "calving_interval":"IEP\nExcessivo",
        }
        items = [(labels[k], economic[k].get("total_loss_brl", economic[k].get("value_lost_brl", 0)))
                 for k in labels if k in economic]
        if not items:
            return plots

        items.sort(key=lambda x: -x[1])
        names, values = zip(*items)
        total = sum(values)

        scen = economic.get("improvement_scenario", {})
        roi  = scen.get("roi_pct", 0)
        gain = scen.get("total_gain_brl", 0)

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Esquerda: barras de perda por categoria
        ax = axes[0]
        colors = ["#E74C3C", "#E67E22", "#F1C40F", "#95A5A6"][:len(items)]
        bars = ax.bar(names, values, color=colors, edgecolor="white", linewidth=0.8)
        ax.set_title(
            "Perda Econômica Estimada por Categoria\n"
            "Com base no diferencial entre desempenho atual e meta Embrapa",
            fontweight="bold",
        )
        ax.set_ylabel("Perda Estimada (R$)")
        for bar, val in zip(bars, values):
            pct = val / total * 100 if total else 0
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(values) * 0.01,
                    f"R$ {val:,.0f}\n({pct:.0f}%)",
                    ha="center", va="bottom", fontsize=8.5, fontweight="bold")
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

        # Direita: ROI do cenário de melhoria
        ax2 = axes[1]
        invest = scen.get("estimated_investment_brl", 0)
        scenario_items = [
            ("Investimento\nem IA", invest, "#3498DB"),
            ("Ganho\nestimado", gain, "#27AE60"),
            ("Perda atual\nevitável", total, "#E74C3C"),
        ]
        s_names, s_vals, s_cols = zip(*scenario_items)
        bars2 = ax2.bar(s_names, s_vals, color=s_cols, edgecolor="white", width=0.5)
        ax2.set_title(
            f"Cenário de Melhoria: +10 p.p. na Taxa de Concepção\n"
            f"ROI estimado: {roi:.0f}% — Retorno sobre investimento em manejo reprodutivo",
            fontweight="bold",
        )
        ax2.set_ylabel("R$")
        for bar, val in zip(bars2, s_vals):
            ax2.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + max(s_vals) * 0.01,
                     f"R$ {val:,.0f}", ha="center", va="bottom",
                     fontsize=9, fontweight="bold")
        ax2.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"R${x:,.0f}"))

        fig.suptitle(
            "Análise de Impacto Econômico da Eficiência Reprodutiva\n"
            f"Perda total estimada: R$ {total:,.0f} | Preços: CONAB/BCB ou padrão CNA",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "07_impacto_economico.png")
        plots.append("07_impacto_economico.png")
        return plots

    def _hypothesis_chart(self, hypotheses: list) -> list[str]:
        plots = []
        valid = [h for h in hypotheses if "p_value" in h]
        if not valid:
            return plots

        ids    = [h["id"] for h in valid]
        pvals  = [h["p_value"] for h in valid]
        sigs   = [h.get("significant", False) for h in valid]
        descs  = [h.get("description", h["id"]) for h in valid]

        fig, ax = plt.subplots(figsize=(11, 5))
        colors = ["#27AE60" if s else "#BDC3C7" for s in sigs]
        neg_log_p = [-np.log10(max(p, 1e-300)) for p in pvals]
        bars = ax.bar(ids, neg_log_p, color=colors, edgecolor="white")
        threshold = -np.log10(0.05)
        ax.axhline(threshold, color="#E74C3C", lw=1.5, ls="--",
                   label=f"Limiar α=0.05 (−log₁₀ = {threshold:.1f})")

        # Adicionar p-values no topo das barras
        for bar, pval, sig in zip(bars, pvals, sigs):
            label_p = f"p={pval:.4f}" if pval >= 0.0001 else "p<0.0001"
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    label_p, ha="center", fontsize=7.5,
                    fontweight="bold" if sig else "normal",
                    color="#27AE60" if sig else "#888")

        ax.set_title(
            "Testes de Hipóteses — Significância Estatística\n"
            "Barras acima da linha vermelha: efeito confirmado estatisticamente (α=0.05)",
            fontweight="bold",
        )
        ax.set_ylabel("−log₁₀(p-value)  [quanto maior, mais significativo]")
        ax.set_xlabel("Hipótese testada")
        ax.tick_params(axis="x", rotation=15)

        green_patch = mpatches.Patch(color="#27AE60", label="Significativo (p < 0.05)")
        grey_patch  = mpatches.Patch(color="#BDC3C7", label="Não significativo (p ≥ 0.05)")
        ax.legend(handles=[green_patch, grey_patch,
                            mpatches.Patch(color="white", label="")],
                  fontsize=8)

        fig.tight_layout()
        _add_source(fig, "Testes: Qui-quadrado / Mann-Whitney / ANOVA  |  α = 0.05")
        _save(fig, "08_hipoteses_pvalues.png")
        plots.append("08_hipoteses_pvalues.png")
        return plots

    def _temporal_trends(self, df: pd.DataFrame) -> list[str]:
        plots = []
        if "ano" not in df.columns or "prenhe" not in df.columns:
            return plots
        if df["ano"].nunique() < 2:
            return plots

        grp = df.groupby("ano")["prenhe"].agg(["mean", "count", "std"]).reset_index()
        grp["mean"] *= 100
        grp["std"]  *= 100
        grp["se"]    = grp["std"] / np.sqrt(grp["count"])   # erro padrão

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.fill_between(grp["ano"],
                        grp["mean"] - 1.96 * grp["se"],
                        grp["mean"] + 1.96 * grp["se"],
                        alpha=0.15, color="#3498DB", label="IC 95%")
        ax.plot(grp["ano"], grp["mean"], "o-", color="#2980B9", lw=2.5,
                markersize=8, label="Taxa de concepção")
        ax.axhline(BENCH["taxa_concepcao_alvo"] * 100, color="#27AE60",
                   lw=1.5, ls=":", label=f"Meta Embrapa: 60%")
        ax.axhline(BENCH["taxa_concepcao_media"] * 100, color="#F39C12",
                   lw=1.2, ls="--", label=f"Média nacional: 55% (ABIEC 2023)")
        ax.set_title(
            "Evolução da Taxa de Concepção ao Longo do Tempo\n"
            "A faixa azul representa o intervalo de confiança 95% da estimativa",
            fontweight="bold",
        )
        ax.set_xlabel("Ano")
        ax.set_ylabel("Taxa de Concepção (%)")
        ax.set_ylim(0, 100)
        for _, row in grp.iterrows():
            ax.text(row["ano"], row["mean"] + 1.5,
                    f"{row['mean']:.1f}%\n(n={row['count']:.0f})",
                    ha="center", fontsize=8)
        ax.legend(fontsize=8)
        fig.tight_layout()
        _add_source(fig, f"{SOURCE_SYNTH}  |  ABIEC Anuário 2023")
        _save(fig, "09_tendencia_temporal.png")
        plots.append("09_tendencia_temporal.png")
        return plots

    def _season_metrics(self, df: pd.DataFrame) -> list[str]:
        plots = []
        if "estacao" not in df.columns:
            return plots

        metrics = [c for c in ("prenhe", "ecc", "dias_abertos") if c in df.columns]
        if not metrics:
            return plots

        labels_m = {"prenhe": "Taxa de Concepção (%)",
                    "ecc": "ECC (Condição Corporal)",
                    "dias_abertos": "Dias Abertos"}
        titles_m = {
            "prenhe": "Taxa de Concepção por Estação",
            "ecc": "ECC Médio por Estação",
            "dias_abertos": "Dias Abertos por Estação",
        }

        fig, axes = plt.subplots(1, len(metrics), figsize=(6 * len(metrics), 5))
        if len(metrics) == 1:
            axes = [axes]

        season_order = ["Verão", "Outono", "Inverno", "Primavera"]
        available_seasons = [s for s in season_order if s in df["estacao"].unique()]

        for ax, metric in zip(axes, metrics):
            if metric == "prenhe":
                grp = df.groupby("estacao")[metric].agg(["mean", "count"]).reindex(available_seasons)
                grp["mean"] *= 100
                colors = ["#F39C12" if s in ("Verão",) else
                          "#3498DB" if s in ("Inverno",) else
                          "#27AE60" for s in grp.index]
                bars = ax.bar(grp.index, grp["mean"], color=colors, edgecolor="white")
                ax.axhline(BENCH["taxa_concepcao_alvo"] * 100, color="#2C3E50",
                           lw=1.2, ls=":", label="Meta 60%")
                ax.set_ylabel(labels_m[metric])
                ax.set_ylim(0, 100)
                for bar, (_, row) in zip(bars, grp.iterrows()):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 1.5,
                            f"{row['mean']:.1f}%\n(n={row['count']:.0f})",
                            ha="center", fontsize=8)
                ax.legend(fontsize=8)
            else:
                sns.boxplot(data=df, x="estacao", y=metric, order=available_seasons,
                            ax=ax, hue="estacao", palette="Set2", legend=False)
                ax.set_ylabel(labels_m[metric])
                if metric == "dias_abertos":
                    ax.axhline(BENCH["dias_abertos_alvo"], color="#27AE60",
                               lw=1.2, ls=":", label=f"Meta: {BENCH['dias_abertos_alvo']} d")
                    ax.legend(fontsize=8)

            ax.set_title(titles_m[metric], fontweight="bold")
            ax.set_xlabel("Estação do Ano")
            ax.tick_params(axis="x", rotation=20)

        fig.suptitle(
            "Métricas Reprodutivas por Estação do Ano\n"
            "A sazonalidade influencia diretamente a taxa de concepção em bovinos tropicais",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout()
        _add_source(fig)
        _save(fig, "10_metricas_por_estacao.png")
        plots.append("10_metricas_por_estacao.png")
        return plots

    def _summarize(self, context: dict[str, Any]) -> None:
        plots = context.get("plots", [])
        print(f"\n  {len(plots)} plots saved to {OUTPUTS_PLOTS}")
