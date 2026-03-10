# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generates synthetic demo data automatically)
python orchestrator.py

# Run with a real dataset
python orchestrator.py --dataset data/raw/my_dataset.csv

# Run with custom economic parameters
python orchestrator.py --dataset data/raw/my_dataset.csv --custo-dia-aberto 10.0
```

## Architecture

This is an 8-agent sequential pipeline for livestock reproductive data analysis, targeting veterinarians and animal reproduction specialists.

### Pipeline flow

```
orchestrator.py  ←→  Each agent enriches and passes `context` dict forward
     │
     ├─ DataEngineerAgent       Load + normalise columns + engineer features
     ├─ DataQualityAgent        Missing values, duplicates, outliers, imputation
     ├─ EDAAgent                KPIs, group rates, correlations, distributions
     ├─ HypothesisAgent         Chi-square / Mann-Whitney / ANOVA statistical tests
     ├─ ModelingAgent           Logistic Regression + Random Forest + Gradient Boosting
     ├─ EconomicImpactAgent     Days-open losses, missed pregnancies, ROI scenarios
     ├─ VisualizationAgent      10 matplotlib/seaborn charts → outputs/plots/
     └─ ReportingAgent          HTML report + Markdown summary → outputs/reports/
```

### Key design patterns

- **Shared context dict** — all agents receive and return the same `context` dict; each agent adds its key (e.g. `df_cleaned`, `eda`, `modeling`, `economic`, `plots`).
- **`BaseAgent`** — every agent extends `agents/base_agent.py`. Implement `execute(context) → context`; the `run()` method wraps it with logging and a summary call.
- **No dataset required** — `DataEngineerAgent` generates 1,200-row synthetic data when no file is provided, covering all downstream agents.
- **Column normalisation** — `COLUMN_MAP` in `data_engineer.py` maps raw/Portuguese/English column names to canonical names. Extend this map for new datasets.

### Domain context (animal reproduction)

Key variables handled: `ecc` (Body Condition Score), `ordem_parto` (parity), `prenhe` (pregnancy result 0/1), `dias_abertos` (days open), `iep` (calving interval), `servicos_concepcao` (services per conception), `estacao` (season), `raca` (breed), `tecnico` (inseminator), `fazenda` (farm).

Economic defaults live in `agents/economic_impact_agent.py → DEFAULT_PARAMS`. Override via `context["econ_params"]`.

### Outputs

| Path | Content |
|------|---------|
| `data/processed/dataset_engineered.csv` | After DataEngineerAgent |
| `data/processed/dataset_cleaned.csv` | After DataQualityAgent |
| `data/processed/eda_results.json` | KPIs and correlations |
| `data/processed/hypotheses_results.json` | Statistical test results |
| `data/processed/economic_impact.json` | Economic calculations |
| `outputs/models/model_results.json` | Model metrics and importances |
| `outputs/plots/*.png` | 10 charts |
| `outputs/reports/relatorio_reprodutivo.html` | Full interactive report |
| `outputs/reports/relatorio_reprodutivo.md` | Markdown summary |
