"""
Main entry point — shorthand for running the full pipeline.

Usage:
    python main.py
    python main.py --dataset data/raw/meu_rebanho.csv
"""

from orchestrator import run_pipeline
import sys

if __name__ == "__main__":
    # Pass sys.argv parsing down to orchestrator
    import orchestrator
    import runpy
    runpy.run_module("orchestrator", run_name="__main__", alter_sys=True)
