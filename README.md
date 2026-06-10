# Cascaded Neurosymbolic Code Generation for Niche DSLs

This repository contains the code and artifacts used in the paper:

"Cascaded Neurosymbolic Code Generation for Niche DSLs: Preserving Chain-of-Thought in Grammar-Constrained Decoding"

Authors: Ruben Ruiz-Torrubiano, Himanshu Buckchash, Sarita Paudel, and Deepak Dhungana

Overview
-	Code and experiments for cascaded neurosymbolic code generation applied to niche DSLs.
-	Implements methods and evaluation scripts used to reproduce results from the paper.

Repository contents (top-level)
-	`requirements.txt` — Python dependencies.
-	`dual_phase_gcd.py`, `evaluate_dual_phase.py` — example/experiment scripts.
-	`generate_tables.py` — utilities to produce LaTeX tables (e.g., `table_pass_k.tex`).
-	`minizinc_parser.py`, `temp_baseline.mzn`, `temp_eval.mzn`, `temp_stub.mzn`, `minizinc_benchmark.json` — MiniZinc-related files and benchmarks.
-	`benchmark_test.json`, `evaluation_pass_5_results.json`, `evaluate_dual_phase.py` — evaluation artifacts and scripts.
-	`mcts/` — directory with MCTS generator and evaluation code:
-		`mcts/mcts_generator.py` — generator implementation.
-		`mcts/evaluate.py`, `mcts/evaluation_results.json` — evaluation harness and results.
-		`mcts/baselines.py` — baseline implementations.
-		`mcts/Modelfile*` — model configuration examples.
-		`mcts/ollama_example.py` — example integration.

Quick start
-	Create and activate a Python virtual environment, then install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

-	Run evaluation or demo scripts (examples):

```bash
python evaluate_dual_phase.py
python generate_tables.py
```

Notes
-	This repository corresponds to the experiments and code described in the cited paper. See the scripts above for entry points used in the evaluation.

