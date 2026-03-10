"""
Project Manager Agent — Pipeline Orchestrator
═══════════════════════════════════════════════
Coordinates the full 8-agent pipeline for livestock reproductive
data analysis.  Pass a context dict to control behaviour:

    context = {
        "raw_path":    "data/raw/my_dataset.csv",   # optional
        "econ_params": {"custo_ia": 80.0},           # optional overrides
    }

The context is enriched by each agent and passed forward.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from agents import (
    DataEngineerAgent,
    DataQualityAgent,
    EDAAgent,
    HypothesisAgent,
    ModelingAgent,
    EconomicImpactAgent,
    VisualizationAgent,
    ReportingAgent,
)
from utils.helpers import log_step, ROOT


PIPELINE: list = [
    DataEngineerAgent,
    DataQualityAgent,
    EDAAgent,
    HypothesisAgent,
    ModelingAgent,
    EconomicImpactAgent,
    VisualizationAgent,
    ReportingAgent,
]


def run_pipeline(context: dict[str, Any] | None = None) -> dict[str, Any]:
    """
    Execute all agents in sequence.
    Returns the fully enriched context dict.
    """
    if context is None:
        context = {}

    banner = "═" * 70
    print(f"\n{banner}")
    print("  SISTEMA MULTI-AGENTE — ANÁLISE REPRODUTIVA PECUÁRIA")
    print(f"  {len(PIPELINE)} agentes serão executados em sequência")
    print(f"{banner}\n")

    t0 = time.perf_counter()
    timings: dict[str, float] = {}

    for AgentClass in PIPELINE:
        agent = AgentClass()
        t_start = time.perf_counter()
        context = agent.run(context)
        elapsed = time.perf_counter() - t_start
        timings[agent.name] = round(elapsed, 2)

    total = time.perf_counter() - t0

    print(f"\n{'═' * 70}")
    print("  PIPELINE CONCLUÍDO COM SUCESSO")
    print(f"{'═' * 70}")
    print(f"\n  Tempo total: {total:.1f}s\n")
    for name, t in timings.items():
        print(f"    {name:35s} {t:6.2f}s")

    print("\n  Outputs gerados:")
    print(f"    Dataset limpo    → {context.get('cleaned_path', 'N/A')}")
    print(f"    Relatório HTML   → {context.get('report_path', 'N/A')}")
    print(f"    Gráficos         → {ROOT / 'outputs' / 'plots'}")
    print(f"    Modelos          → {ROOT / 'outputs' / 'models'}")
    print(f"\n{'═' * 70}\n")

    return context


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Pipeline de Análise Reprodutiva Pecuária")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Caminho para o dataset (CSV/Excel/Parquet). "
                             "Se omitido, dados sintéticos são gerados.")
    parser.add_argument("--custo-dia-aberto", type=float, default=None,
                        help="Custo por dia aberto (R$) — sobrescreve padrão.")
    args = parser.parse_args()

    ctx: dict[str, Any] = {}
    if args.dataset:
        ctx["raw_path"] = args.dataset
    if args.custo_dia_aberto:
        ctx["econ_params"] = {"custo_dia_aberto_leite": args.custo_dia_aberto,
                              "custo_dia_aberto_corte": args.custo_dia_aberto * 0.6}

    run_pipeline(ctx)
