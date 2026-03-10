"""
Painel de Análise Reprodutiva Bovina
──────────────────────────────────────
Interactive Streamlit dashboard for bovine reproductive performance analysis.
Works standalone (no prior pipeline run required) by using the IATF simulator.

Run:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Make repo root importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.reproductive_strategy_simulator import (
    IATFSimulator,
    simulate_scenarios,
    BENCHMARKS,
    CONVENTIONAL_DEFAULTS,
    IATF_DEFAULTS,
)

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Painel Reprodutivo Bovino",
    page_icon="🐄",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 2rem; }
.main { padding-top: 1rem; }
[data-testid="stMetricDelta"] { font-size: 0.9rem; }
.block-container { padding-top: 1rem; padding-bottom: 1rem; }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────

st.sidebar.title("Parâmetros do Rebanho")
st.sidebar.markdown("---")

ecc_mean = st.sidebar.slider(
    "ECC médio (Escore de Condição Corporal)",
    min_value=1.0, max_value=5.0, value=2.9, step=0.1,
    help="Escore médio do rebanho ao momento do serviço (escala 1–5)",
)
conception_rate_pct = st.sidebar.slider(
    "Taxa de concepção (%)",
    min_value=20, max_value=90, value=55, step=1,
    help="% de vacas que concebem por serviço",
)
days_open = st.sidebar.slider(
    "Dias abertos médios",
    min_value=60, max_value=300, value=130, step=5,
    help="Média de dias abertos do rebanho",
)
herd_size = st.sidebar.slider(
    "Tamanho do rebanho (vacas expostas)",
    min_value=50, max_value=2000, value=500, step=50,
)
calf_value = st.sidebar.slider(
    "Valor do bezerro (R$)",
    min_value=800, max_value=5000, value=1800, step=100,
    help="Valor de mercado do bezerro desmamado",
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Benchmarks nacionais (Embrapa/CNA/ABIEC)**")
st.sidebar.caption(f"Meta taxa concepção: {BENCHMARKS['taxa_concepcao_meta_embrapa']:.0%}")
st.sidebar.caption(f"Média nacional: {BENCHMARKS['taxa_concepcao_media_nacional']:.0%}")
st.sidebar.caption(f"Dias abertos meta (CNA): {BENCHMARKS['dias_abertos_meta_cna']} dias")
st.sidebar.caption(f"IEP ideal: {BENCHMARKS['iep_ideal_dias']} dias (1 bezerra/vaca/ano)")

# ── Compute simulation ─────────────────────────────────────────────────────────

conception_rate = conception_rate_pct / 100
days_open_cost = BENCHMARKS["custo_dia_aberto_corte_rs"]

sim_result = simulate_scenarios(
    herd_size=herd_size,
    ecc_mean=ecc_mean,
    days_open=float(days_open),
    calf_value=float(calf_value),
    days_open_cost=days_open_cost,
)

sim = IATFSimulator(
    herd_size=herd_size,
    base_conception_rate=conception_rate,
    calf_value=float(calf_value),
    days_open_cost=days_open_cost,
)
conv_params = {**CONVENTIONAL_DEFAULTS, "conception_rate": conception_rate}
iatf_cr = min(0.85, conception_rate + 0.05)
iatf_params_run = {**IATF_DEFAULTS, "conception_rate": iatf_cr}
comparison = sim.compare(conv_params, iatf_params_run)
roi_data = sim.roi_analysis(conv_params, iatf_params_run)

conv = comparison["convencional"]
iatf = comparison["iatf"]

expected_pregnancies = herd_size * conception_rate
calves_per_year = expected_pregnancies  # one calf per pregnancy
excess_days = max(0.0, days_open - BENCHMARKS["dias_abertos_meta_cna"])
days_open_loss = excess_days * days_open_cost * herd_size
cost_per_preg = conv["cost_per_pregnancy"]

# ── Main area ──────────────────────────────────────────────────────────────────

st.title("Painel de Análise Reprodutiva Bovina")
st.markdown(
    "Dashboard para monitoramento de indicadores reprodutivos, "
    "simulação econômica e comparativo de protocolos de IA."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "Indicadores",
    "Impacto Econômico",
    "Simulador IATF",
    "Benchmarks",
])

# ── Tab 1: KPI cards ───────────────────────────────────────────────────────────

with tab1:
    st.subheader("Indicadores do Rebanho")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric(
        "Prenhezes esperadas",
        f"{expected_pregnancies:.0f}",
        delta=f"{conception_rate_pct}% de taxa de concepção",
        delta_color="normal",
    )
    c2.metric(
        "Bezerros produzidos/ano",
        f"{calves_per_year:.0f}",
        delta=f"R$ {calves_per_year * calf_value:,.0f} valor total",
        delta_color="normal",
    )
    c3.metric(
        "Perda estimada (dias abertos)",
        f"R$ {days_open_loss:,.0f}",
        delta=f"{excess_days:.0f} dias acima da meta CNA",
        delta_color="inverse",
    )
    c4.metric(
        "Custo por prenhez",
        f"R$ {cost_per_preg:,.2f}",
        delta=f"Protocolo convencional",
    )

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Resumo dos parâmetros informados**")
        summary_data = {
            "Parâmetro": [
                "Rebanho (vacas expostas)",
                "Taxa de concepção",
                "Dias abertos médios",
                "ECC médio",
                "Valor do bezerro",
                "Custo/dia aberto",
            ],
            "Valor": [
                f"{herd_size} vacas",
                f"{conception_rate_pct}%",
                f"{days_open} dias",
                f"{ecc_mean:.1f}",
                f"R$ {calf_value:,.0f}",
                f"R$ {days_open_cost:.2f}",
            ],
        }
        st.table(pd.DataFrame(summary_data))

    with col_right:
        st.markdown("**Distribuição estimada de resultados**")
        pregnant_n = round(expected_pregnancies)
        empty_n = herd_size - pregnant_n
        pie_fig = px.pie(
            names=["Prenhes", "Vazias"],
            values=[pregnant_n, empty_n],
            color_discrete_sequence=["#27AE60", "#E74C3C"],
            hole=0.4,
        )
        pie_fig.update_traces(textinfo="percent+label")
        pie_fig.update_layout(margin=dict(t=20, b=20, l=0, r=0), height=280)
        st.plotly_chart(pie_fig, use_container_width=True)

# ── Tab 2: Economic Impact ─────────────────────────────────────────────────────

with tab2:
    st.subheader("Impacto Econômico da Eficiência Reprodutiva")

    improved_pregnancies = min(herd_size, round(herd_size * (conception_rate + 0.10)))
    gain_10pp = (improved_pregnancies - expected_pregnancies) * calf_value
    invest_ia = conv["total_cost"]

    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Perda atual (dias abertos excessivos)", f"R$ {days_open_loss:,.0f}",
                 delta="Custo de ineficiência", delta_color="inverse")
    col_b.metric("Ganho potencial (+10 p.p.)", f"R$ {gain_10pp:,.0f}",
                 delta="Vacas adicionalmente prenhes", delta_color="normal")
    col_c.metric("Investimento em IA (convencional)", f"R$ {invest_ia:,.0f}",
                 delta="Total da estação", delta_color="off")

    st.markdown("---")

    # Bar chart
    eco_labels = ["Perda atual\n(dias abertos)", "Ganho potencial\n(+10 p.p.)", "Investimento IA\n(convencional)"]
    eco_values = [days_open_loss, gain_10pp, invest_ia]
    eco_colors = ["#E74C3C", "#27AE60", "#3498DB"]

    bar_eco = go.Figure(data=[
        go.Bar(x=eco_labels, y=eco_values, marker_color=eco_colors, text=[f"R$ {v:,.0f}" for v in eco_values],
               textposition="outside")
    ])
    bar_eco.update_layout(
        title="Comparativo Econômico: Perda vs Ganho vs Investimento",
        yaxis_title="R$",
        yaxis_tickformat=",.0f",
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=400,
    )
    st.plotly_chart(bar_eco, use_container_width=True)

    st.info(
        f"Com {days_open:.0f} dias abertos médios, o rebanho acumula "
        f"**{excess_days:.0f} dias acima da meta CNA (110 dias)** por vaca. "
        f"A perda estimada é de **R$ {days_open_loss:,.0f}/estação** para {herd_size} vacas."
    )

# ── Tab 3: IATF Simulator ──────────────────────────────────────────────────────

with tab3:
    st.subheader("Simulador: Convencional vs IATF")
    st.markdown(
        "Comparativo econômico entre inseminação artificial por detecção de cio "
        "e IATF (inseminação em tempo fixo com protocolo de sincronização)."
    )

    # Side-by-side table
    table_data = {
        "Indicador": [
            "Protocolo",
            "Taxa de concepção",
            "Prenhezes esperadas",
            "Custo total (R$)",
            "Custo por prenhez (R$)",
            "Receita bruta (R$)",
            "Retorno líquido (R$)",
            "ROI (%)",
        ],
        "Convencional": [
            conv["protocol"],
            f"{conv['effective_conception_rate']:.1%}",
            f"{conv['expected_pregnancies']:.0f}",
            f"R$ {conv['total_cost']:,.0f}",
            f"R$ {conv['cost_per_pregnancy']:,.2f}",
            f"R$ {conv['revenue_brl']:,.0f}",
            f"R$ {conv['net_revenue']:,.0f}",
            f"{conv['roi_pct']:.1f}%",
        ],
        "IATF": [
            iatf["protocol"],
            f"{iatf['effective_conception_rate']:.1%}",
            f"{iatf['expected_pregnancies']:.0f}",
            f"R$ {iatf['total_cost']:,.0f}",
            f"R$ {iatf['cost_per_pregnancy']:,.2f}",
            f"R$ {iatf['revenue_brl']:,.0f}",
            f"R$ {iatf['net_revenue']:,.0f}",
            f"{iatf['roi_pct']:.1f}%",
        ],
    }
    st.table(pd.DataFrame(table_data))

    st.markdown("---")

    col_x, col_y = st.columns(2)

    with col_x:
        # Cost per pregnancy comparison
        cpp_fig = go.Figure(data=[
            go.Bar(
                name="Custo/Prenhez",
                x=["Convencional", "IATF"],
                y=[conv["cost_per_pregnancy"], iatf["cost_per_pregnancy"]],
                marker_color=["#3498DB", "#9B59B6"],
                text=[f"R$ {conv['cost_per_pregnancy']:,.2f}", f"R$ {iatf['cost_per_pregnancy']:,.2f}"],
                textposition="outside",
            )
        ])
        cpp_fig.update_layout(
            title="Custo por Prenhez (R$)",
            yaxis_title="R$",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=350,
        )
        st.plotly_chart(cpp_fig, use_container_width=True)

    with col_y:
        # ROI comparison
        roi_fig = go.Figure(data=[
            go.Bar(
                name="ROI",
                x=["Convencional", "IATF"],
                y=[conv["roi_pct"], iatf["roi_pct"]],
                marker_color=["#E67E22", "#27AE60"],
                text=[f"{conv['roi_pct']:.1f}%", f"{iatf['roi_pct']:.1f}%"],
                textposition="outside",
            )
        ])
        roi_fig.update_layout(
            title="Retorno sobre Investimento — ROI (%)",
            yaxis_title="%",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=350,
        )
        st.plotly_chart(roi_fig, use_container_width=True)

    # Recommendation
    rec = comparison["recommendation"]
    if rec["level"] == "positive":
        st.success(f"**Recomendação:** {rec['text']}")
    elif rec["level"] == "neutral":
        st.info(f"**Recomendação:** {rec['text']}")
    else:
        st.warning(f"**Recomendação:** {rec['text']}")

    diff = comparison["diferenca"]
    st.markdown(
        f"**Diferencial IATF vs Convencional:** "
        f"{diff['pregnancies_gained']:+.1f} prenhezes | "
        f"R$ {diff['net_revenue_diff_brl']:+,.0f} retorno líquido | "
        f"{diff['roi_pct_diff']:+.1f} p.p. ROI"
    )

# ── Tab 4: Benchmarks ─────────────────────────────────────────────────────────

with tab4:
    st.subheader("Posição do Rebanho vs Benchmarks Nacionais")
    st.markdown("Comparativo com referências Embrapa, CNA e ABIEC 2023.")

    benchmark_rows = [
        {
            "Indicador": "Taxa de Concepção",
            "Rebanho Atual": f"{conception_rate:.1%}",
            "Meta Embrapa": f"{BENCHMARKS['taxa_concepcao_meta_embrapa']:.0%}",
            "Média Nacional": f"{BENCHMARKS['taxa_concepcao_media_nacional']:.0%}",
            "Status": (
                "Acima da meta" if conception_rate >= BENCHMARKS["taxa_concepcao_meta_embrapa"]
                else "Acima da média" if conception_rate >= BENCHMARKS["taxa_concepcao_media_nacional"]
                else "Abaixo da média"
            ),
        },
        {
            "Indicador": "Dias Abertos",
            "Rebanho Atual": f"{days_open} d",
            "Meta Embrapa": f"{BENCHMARKS['dias_abertos_meta_cna']} d (CNA)",
            "Média Nacional": f"{BENCHMARKS['dias_abertos_media_nacional']} d",
            "Status": (
                "Dentro da meta" if days_open <= BENCHMARKS["dias_abertos_meta_cna"]
                else "Acima da média" if days_open <= BENCHMARKS["dias_abertos_media_nacional"]
                else "Acima da média nacional"
            ),
        },
        {
            "Indicador": "ECC ao serviço",
            "Rebanho Atual": f"{ecc_mean:.1f}",
            "Meta Embrapa": f"{BENCHMARKS['ecc_ideal_ao_servico']:.1f}",
            "Média Nacional": "2.5–3.0",
            "Status": (
                "Ideal" if ecc_mean >= BENCHMARKS["ecc_ideal_ao_servico"]
                else "Abaixo do ideal" if ecc_mean >= 2.5
                else "Baixo — risco reprodutivo"
            ),
        },
    ]

    bm_df = pd.DataFrame(benchmark_rows)
    st.table(bm_df)

    st.markdown("---")
    st.markdown("**Progresso em relação às metas**")

    # Progress bar for conception rate
    prog_cr = min(1.0, conception_rate / BENCHMARKS["taxa_concepcao_meta_embrapa"])
    st.markdown(f"**Taxa de Concepção** — {conception_rate:.1%} / meta {BENCHMARKS['taxa_concepcao_meta_embrapa']:.0%}")
    st.progress(prog_cr)

    # Progress bar for ECC
    prog_ecc = min(1.0, ecc_mean / BENCHMARKS["ecc_ideal_ao_servico"])
    st.markdown(f"**ECC** — {ecc_mean:.1f} / ideal {BENCHMARKS['ecc_ideal_ao_servico']:.1f}")
    st.progress(prog_ecc)

    # Days open inverse (lower is better)
    prog_da = min(1.0, BENCHMARKS["dias_abertos_meta_cna"] / max(days_open, 1))
    st.markdown(
        f"**Dias Abertos** — {days_open} d / meta ≤ {BENCHMARKS['dias_abertos_meta_cna']} d "
        f"(barra cheia = dentro da meta)"
    )
    st.progress(prog_da)

    st.markdown("---")
    with st.expander("Referências e fontes"):
        st.markdown("""
| Indicador | Fonte | Ano |
|-----------|-------|-----|
| Taxa de concepção meta | Embrapa Gado de Corte | 2023 |
| Taxa de concepção média nacional | ABIEC Anuário | 2023 |
| Dias abertos meta | CNA | 2023 |
| ECC ideal ao serviço | Embrapa | 2023 |
| IEP ideal | Embrapa / literatura | — |
| Custo/dia aberto (corte) | CNA / literatura | 2023 |
| Valor bezerro referência | CONAB / mercado | 2024 |
        """)

# ── Footer ─────────────────────────────────────────────────────────────────────

st.markdown("---")
st.caption(
    "Painel de Análise Reprodutiva Bovina  |  "
    "Benchmarks: Embrapa / CNA / ABIEC 2023  |  "
    "Valores em R$ — referência 2024"
)
