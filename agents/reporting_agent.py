"""
Reporting Agent
────────────────
Assembles all pipeline outputs into a structured analytical report
saved as both HTML (interactive) and a plain text/markdown summary.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from agents.base_agent import BaseAgent
from utils.helpers import OUTPUTS_REPORTS, OUTPUTS_PLOTS, format_currency, section


HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="pt-BR">
<head>
  <meta charset="utf-8"/>
  <title>Relatório Reprodutivo</title>
  <style>
    :root {{
      --primary: #1a3a5c;
      --accent: #2980b9;
      --success: #27ae60;
      --warning: #f39c12;
      --danger: #e74c3c;
      --light: #f4f6f8;
    }}
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{
      font-family: "Segoe UI", Arial, sans-serif;
      background: #eef2f7;
      color: #333;
      line-height: 1.6;
    }}
    header {{
      background: linear-gradient(135deg, var(--primary), var(--accent));
      color: white;
      padding: 2.5rem 2rem;
      text-align: center;
    }}
    header h1 {{ font-size: 2.2rem; margin-bottom: .4rem; }}
    header p  {{ font-size: 1rem; opacity: .85; }}
    main {{ max-width: 1100px; margin: 2rem auto; padding: 0 1.5rem 4rem; }}
    section {{ background: white; border-radius: 10px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
               padding: 1.8rem 2rem; margin-bottom: 2rem; }}
    h2 {{ color: var(--primary); font-size: 1.35rem; border-left: 4px solid var(--accent);
          padding-left: .8rem; margin-bottom: 1.2rem; }}
    h3 {{ color: var(--accent); font-size: 1.05rem; margin: 1rem 0 .5rem; }}
    .kpi-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(190px, 1fr)); gap: 1rem; }}
    .kpi-card {{
      background: var(--light);
      border-radius: 8px;
      padding: 1.1rem;
      text-align: center;
      border-top: 3px solid var(--accent);
    }}
    .kpi-card .value {{ font-size: 1.9rem; font-weight: 700; color: var(--primary); }}
    .kpi-card .label {{ font-size: .8rem; color: #666; margin-top: .3rem; }}
    table {{
      width: 100%; border-collapse: collapse; font-size: .92rem; margin-top: .8rem;
    }}
    th {{ background: var(--primary); color: white; padding: .6rem .9rem; text-align: left; }}
    td {{ padding: .55rem .9rem; border-bottom: 1px solid #e8e8e8; }}
    tr:nth-child(even) td {{ background: var(--light); }}
    .badge {{
      display: inline-block; padding: .2rem .6rem; border-radius: 12px;
      font-size: .78rem; font-weight: 600;
    }}
    .badge-success {{ background: #d5f5e3; color: var(--success); }}
    .badge-danger  {{ background: #fde8e8; color: var(--danger);  }}
    .badge-warning {{ background: #fef9e7; color: var(--warning); }}
    .plot-grid {{
      display: grid; grid-template-columns: repeat(auto-fill, minmax(480px, 1fr)); gap: 1.2rem;
    }}
    .plot-grid img {{
      width: 100%; border-radius: 8px; box-shadow: 0 1px 5px rgba(0,0,0,.12);
    }}
    .highlight {{
      background: linear-gradient(90deg, #d5f5e3, #e8f8f5);
      border-left: 4px solid var(--success);
      padding: .9rem 1.1rem; border-radius: 0 8px 8px 0; margin: .8rem 0;
    }}
    footer {{ text-align: center; color: #888; font-size: .85rem; margin-top: 3rem; }}
  </style>
</head>
<body>
<header>
  <h1>Relatório de Desempenho Reprodutivo</h1>
  <p>Análise Automatizada de Dados Pecuários &nbsp;|&nbsp; {date}</p>
</header>
<main>

{sections}

</main>
<footer>
  <p>Gerado automaticamente pelo Sistema Multi-Agente de Análise Reprodutiva</p>
</footer>
</body>
</html>
"""


class ReportingAgent(BaseAgent):
    name = "ReportingAgent"

    def _description(self) -> str:
        return "Assembling final analytical report (HTML + Markdown)"

    # ── public ────────────────────────────────────────────────────────────────

    def execute(self, context: dict[str, Any]) -> dict[str, Any]:
        eda       = context.get("eda", {})
        modeling  = context.get("modeling", {})
        economic  = context.get("economic", {})
        hypotheses = context.get("hypotheses", [])
        quality   = context.get("quality_report", {})
        plots     = context.get("plots", [])

        html_sections = []

        section("Executive Summary")
        html_sections.append(self._executive_summary(eda, economic, quality))

        section("KPI Dashboard")
        html_sections.append(self._kpi_section(eda))

        section("Data Quality")
        html_sections.append(self._quality_section(quality))

        section("Hypothesis Results")
        html_sections.append(self._hypothesis_section(hypotheses))

        section("Predictive Models")
        html_sections.append(self._model_section(modeling))

        section("Economic Impact")
        html_sections.append(self._economic_section(economic))

        section("Visualizations")
        html_sections.append(self._plots_section(plots))

        section("Recommendations")
        html_sections.append(self._recommendations(eda, hypotheses, economic))

        html = HTML_TEMPLATE.format(
            date=datetime.now().strftime("%d/%m/%Y %H:%M"),
            sections="\n".join(html_sections),
        )

        report_path = OUTPUTS_REPORTS / "relatorio_reprodutivo.html"
        report_path.write_text(html, encoding="utf-8")
        print(f"\n  ✔ HTML report → {report_path}")

        md_path = OUTPUTS_REPORTS / "relatorio_reprodutivo.md"
        md_path.write_text(self._markdown_summary(eda, economic, hypotheses, modeling), encoding="utf-8")
        print(f"  ✔ Markdown summary → {md_path}")

        context["report_path"]   = report_path
        context["markdown_path"] = md_path
        return context

    # ── HTML sections ─────────────────────────────────────────────────────────

    def _executive_summary(self, eda: dict, economic: dict, quality: dict) -> str:
        kpis = eda.get("kpis", {})
        cr   = kpis.get("taxa_concepcao_global", 0)
        iep  = kpis.get("iep_medio_dias", 0)
        da   = kpis.get("dias_abertos_medio", 0)
        loss = economic.get("total_annual_loss", 0)
        score = quality.get("quality_score", 0)

        status_cr = "badge-success" if cr >= 0.60 else ("badge-warning" if cr >= 0.45 else "badge-danger")
        status_iep = "badge-success" if iep <= 390 else ("badge-warning" if iep <= 450 else "badge-danger")

        return f"""
<section>
  <h2>📋 Sumário Executivo</h2>
  <p>Esta análise avaliou o desempenho reprodutivo do rebanho com base em dados históricos
     de inseminações, diagnósticos de gestação e métricas produtivas. Os resultados são
     apresentados abaixo, seguidos de recomendações práticas para melhoria da eficiência
     reprodutiva e impacto econômico.</p>
  <div style="margin-top:1.2rem">
    <div class="highlight">
      <strong>Taxa de concepção:</strong>
      <span class="badge {status_cr}">{cr:.1%}</span>
      &nbsp;|&nbsp;
      <strong>IEP médio:</strong>
      <span class="badge {status_iep}">{iep:.0f} dias</span>
      &nbsp;|&nbsp;
      <strong>Dias abertos:</strong> {da:.0f} dias
      &nbsp;|&nbsp;
      <strong>Qualidade dos dados:</strong> {score:.1f}/100
    </div>
    <div class="highlight" style="background: linear-gradient(90deg, #fde8e8, #fdede8); border-color: #e74c3c;">
      <strong>Perda econômica estimada:</strong> {format_currency(loss)} / ciclo reprodutivo
    </div>
  </div>
</section>"""

    def _kpi_section(self, eda: dict) -> str:
        kpis = eda.get("kpis", {})
        cards = {
            "Taxa de Concepção":    (f"{kpis.get('taxa_concepcao_global', 0):.1%}", ""),
            "Serv./Concepção":      (f"{kpis.get('servicos_por_concepcao', 0):.2f}", ""),
            "IEP Médio (dias)":     (f"{kpis.get('iep_medio_dias', 0):.0f}", ""),
            "Dias Abertos":         (f"{kpis.get('dias_abertos_medio', 0):.0f}", ""),
            "ECC Médio":            (f"{kpis.get('ecc_medio', 0):.2f}", ""),
            "Leite Médio (kg/dia)": (f"{kpis.get('producao_leite_media_kg', 0):.1f}", ""),
        }
        cards_html = "\n".join(
            f'<div class="kpi-card"><div class="value">{v}</div><div class="label">{k}</div></div>'
            for k, (v, _) in cards.items()
        )
        return f"""
<section>
  <h2>📊 Indicadores-Chave de Desempenho (KPIs)</h2>
  <div class="kpi-grid">{cards_html}</div>
</section>"""

    def _quality_section(self, quality: dict) -> str:
        score = quality.get("quality_score", 0)
        missing = quality.get("missing", {})
        rows = "\n".join(
            f"<tr><td>{col}</td><td>{info['count']}</td><td>{info['pct']:.1f}%</td></tr>"
            for col, info in missing.items()
        ) or "<tr><td colspan='3'>Sem valores ausentes</td></tr>"

        color = "#27ae60" if score >= 90 else ("#f39c12" if score >= 70 else "#e74c3c")
        return f"""
<section>
  <h2>🔍 Qualidade dos Dados</h2>
  <p><strong>Score de qualidade:</strong>
     <span style="color:{color}; font-size:1.4rem; font-weight:700">{score:.1f}/100</span>
  </p>
  <h3>Valores Ausentes por Coluna</h3>
  <table>
    <tr><th>Coluna</th><th>Quantidade</th><th>%</th></tr>
    {rows}
  </table>
</section>"""

    def _hypothesis_section(self, hypotheses: list) -> str:
        rows = ""
        for h in hypotheses:
            sig  = h.get("significant", False)
            badge = "badge-success" if sig else "badge-warning"
            label = "Confirmada" if sig else "Não confirmada"
            p = h.get("p_value", "—")
            p_str = f"{p:.4f}" if isinstance(p, float) else str(p)
            rows += f"""
<tr>
  <td><strong>{h['id']}</strong></td>
  <td>{h['description']}</td>
  <td>{p_str}</td>
  <td><span class="badge {badge}">{label}</span></td>
</tr>"""

        return f"""
<section>
  <h2>🧪 Hipóteses Testadas</h2>
  <table>
    <tr><th>ID</th><th>Hipótese</th><th>p-value</th><th>Resultado</th></tr>
    {rows}
  </table>
</section>"""

    def _model_section(self, modeling: dict) -> str:
        if not modeling:
            return "<section><h2>🤖 Modelos Preditivos</h2><p>Nenhum modelo gerado.</p></section>"

        best = modeling.get("best_model", "—")
        comparison = modeling.get("comparison", {})

        rows = "\n".join(
            f"<tr><td>{'⭐ ' if k == best else ''}{k.replace('_',' ').title()}</td>"
            f"<td>{v.get('cv_auc','—')}</td><td>{v.get('roc_auc','—')}</td></tr>"
            for k, v in comparison.items()
        )

        imp = modeling.get(best, {}).get("feature_importances", {})
        imp_rows = "\n".join(
            f"<tr><td>{k}</td><td>{v:.4f}</td><td>"
            f"<div style='background:#2980b9;height:12px;width:{v*100:.0f}%;border-radius:3px'></div></td></tr>"
            for k, v in list(imp.items())[:10]
        )

        return f"""
<section>
  <h2>🤖 Modelos Preditivos — Concepção</h2>
  <h3>Comparação de Modelos</h3>
  <table>
    <tr><th>Modelo</th><th>CV AUC (5-fold)</th><th>Train AUC</th></tr>
    {rows}
  </table>
  <h3>Importância das Variáveis (melhor modelo: {best.replace('_',' ').title()})</h3>
  <table>
    <tr><th>Variável</th><th>Importância</th><th>Barra</th></tr>
    {imp_rows}
  </table>
</section>"""

    def _economic_section(self, economic: dict) -> str:
        if not economic:
            return "<section><h2>💰 Impacto Econômico</h2><p>Não calculado.</p></section>"

        total = economic.get("total_annual_loss", 0)
        scen  = economic.get("improvement_scenario", {})
        roi   = scen.get("roi_pct", 0)
        gain  = scen.get("total_gain_brl", 0)
        invest = scen.get("estimated_investment_brl", 0)

        sections_html = []
        items = {
            "Dias Abertos Excessivos": economic.get("days_open", {}),
            "Taxa de Concepção Baixa": economic.get("conception_rate", {}),
            "Serviços Adicionais":     economic.get("services", {}),
            "IEP Excessivo":           economic.get("calving_interval", {}),
        }
        for name, data in items.items():
            loss = data.get("total_loss_brl", data.get("value_lost_brl", 0))
            sections_html.append(f"<tr><td>{name}</td><td>{format_currency(loss)}</td></tr>")

        return f"""
<section>
  <h2>💰 Impacto Econômico Estimado</h2>
  <table>
    <tr><th>Categoria</th><th>Perda Estimada</th></tr>
    {''.join(sections_html)}
    <tr style="font-weight:700">
      <td>TOTAL</td><td style="color:#e74c3c">{format_currency(total)}</td>
    </tr>
  </table>
  <h3>Cenário de Melhoria (+10 p.p. na taxa de concepção)</h3>
  <div class="highlight">
    Ganho estimado: <strong>{format_currency(gain)}</strong>
    &nbsp;|&nbsp; Investimento: <strong>{format_currency(invest)}</strong>
    &nbsp;|&nbsp; ROI: <strong style="color:#27ae60">{roi:.1f}%</strong>
  </div>
</section>"""

    def _plots_section(self, plots: list[str]) -> str:
        if not plots:
            return "<section><h2>📈 Visualizações</h2><p>Nenhum gráfico gerado.</p></section>"

        imgs = "\n".join(
            f'<img src="../plots/{p}" alt="{p}"/>'
            for p in plots
        )
        return f"""
<section>
  <h2>📈 Visualizações</h2>
  <div class="plot-grid">{imgs}</div>
</section>"""

    def _recommendations(self, eda: dict, hypotheses: list, economic: dict) -> str:
        kpis = eda.get("kpis", {})
        cr   = kpis.get("taxa_concepcao_global", 1.0)
        da   = kpis.get("dias_abertos_medio", 0)
        ecc  = kpis.get("ecc_medio", 3.5)
        spc  = kpis.get("servicos_por_concepcao", 1.0)

        recs: list[str] = []

        if cr < 0.55:
            recs.append("🔴 <strong>Taxa de concepção abaixo de 55%</strong> — revisar qualidade do sêmen, protocolo hormonal e detecção de cio.")
        elif cr < 0.65:
            recs.append("🟡 <strong>Taxa de concepção entre 55–65%</strong> — avaliar sincronização de ovulação e capacitação dos inseminadores.")
        else:
            recs.append("🟢 <strong>Taxa de concepção satisfatória (≥65%)</strong> — manter o protocolo atual e monitorar continuamente.")

        if ecc < 2.8:
            recs.append("🔴 <strong>ECC médio baixo (< 2.8)</strong> — implementar estratégia nutricional no periparto para melhorar condição corporal.")

        if da > 150:
            recs.append("🔴 <strong>Dias abertos médios acima de 150 dias</strong> — antecipar o início do protocolo reprodutivo pós-parto.")

        if spc > 1.8:
            recs.append("🟡 <strong>SPC > 1.8</strong> — investigar técnica de inseminação e qualidade do material genético.")

        sig_hyps = [h for h in hypotheses if h.get("significant")]
        for h in sig_hyps:
            if "ecc" in h.get("factor", "").lower():
                recs.append("✅ <strong>BCS confirmado como fator significativo</strong> — priorizar manejo nutricional pré-IA.")
            if "tecnico" in h.get("factor", "").lower():
                recs.append("✅ <strong>Efeito do inseminador confirmado</strong> — padronizar técnica e promover treinamento periódico.")
            if "estacao" in h.get("factor", "").lower():
                recs.append("✅ <strong>Sazonalidade confirmada</strong> — ajustar período de monta/IA conforme estação mais favorável.")

        items_html = "\n".join(f"<li style='margin:.5rem 0'>{r}</li>" for r in recs) or "<li>Dados insuficientes para recomendações.</li>"
        return f"""
<section>
  <h2>💡 Recomendações</h2>
  <ul style="list-style:none; padding:0">
    {items_html}
  </ul>
</section>"""

    # ── Markdown ──────────────────────────────────────────────────────────────

    def _markdown_summary(self, eda: dict, economic: dict, hypotheses: list, modeling: dict) -> str:
        kpis = eda.get("kpis", {})
        total = economic.get("total_annual_loss", 0)
        best  = modeling.get("best_model", "—")
        best_auc = modeling.get("comparison", {}).get(best, {}).get("roc_auc", "—")
        n_sig = sum(1 for h in hypotheses if h.get("significant"))

        lines = [
            "# Relatório de Desempenho Reprodutivo",
            f"\n_Gerado em {datetime.now().strftime('%d/%m/%Y %H:%M')}_\n",
            "## KPIs Principais",
            f"| Indicador | Valor |",
            f"|---|---|",
            f"| Taxa de Concepção | {kpis.get('taxa_concepcao_global', 0):.1%} |",
            f"| Serviços por Concepção | {kpis.get('servicos_por_concepcao', 0):.2f} |",
            f"| IEP Médio | {kpis.get('iep_medio_dias', 0):.0f} dias |",
            f"| Dias Abertos | {kpis.get('dias_abertos_medio', 0):.0f} dias |",
            f"| ECC Médio | {kpis.get('ecc_medio', 0):.2f} |",
            "",
            "## Hipóteses",
            f"- **{n_sig}/{len(hypotheses)}** hipóteses confirmadas como estatisticamente significativas (α=0.05)",
            "",
            "## Melhor Modelo Preditivo",
            f"- **{best.replace('_', ' ').title()}** — AUC = {best_auc}",
            "",
            "## Impacto Econômico",
            f"- **Perda total estimada:** {format_currency(total)}",
            "",
        ]
        return "\n".join(lines)

    def _summarize(self, context: dict[str, Any]) -> None:
        print(f"\n  Report: {context.get('report_path')}")
