import json
import os
import statistics

# =====================================================================
# Configuration
# =====================================================================
RESULTS_FILE = "evaluation_pass_5_results.json"

METHODS = [
    "Zero-Shot",
    "One-Shot (No GCD)",
    "One-Shot (GCD Only)",
    "Dual-Phase (Proposed)"
]

def generate_latex_table():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run the benchmark first.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
        
    details = data.get("details", [])
    total_prompts = len(details)
    
    if total_prompts == 0:
        print("Error: No prompt details found in the JSON file.")
        return

    # Track how many times each method passes at k=1, k=3, and k=5
    pass_counts = {m: {1: 0, 3: 0, 5: 0} for m in METHODS}
    
    # Track generation times and token counts to calculate mean and std dev
    generation_times = {m: [] for m in METHODS}
    tokens_spent = {m: [] for m in METHODS}

    # Tracking for Dual-Phase Routing Analytics
    dual_total_samples = 0
    dual_phase2_triggered = 0
    dual_phase2_success = 0

    for item in details:
        evaluations = item.get("evaluations", {})
        
        for method in METHODS:
            if method not in evaluations:
                continue
                
            samples = evaluations[method].get("samples", [])
            
            # Extract metrics for all generated samples of this method
            for s in samples:
                # Time and tokens
                t = s.get("generation_time")
                if t is not None:
                    generation_times[method].append(t)
                    
                tok = s.get("tokens_spent")
                if tok is not None:
                    tokens_spent[method].append(tok)
                
                # Check individual sample success
                compiles = s.get("compiles", False)
                judge_score = s.get("judge_score", 0.0)
                sample_passed = (compiles and judge_score >= 0.8)

                # Tracking internal routing for the Proposed Dual-Phase method
                if method == "Dual-Phase (Proposed)":
                    dual_total_samples += 1
                    fast_path_success = s.get("fast_path_success", False)
                    
                    if not fast_path_success:
                        dual_phase2_triggered += 1
                        if sample_passed:
                            dual_phase2_success += 1

            # Helper to check if a pass exists in the first K samples
            def passed_in_first_k(k: int) -> bool:
                for s in samples[:k]:
                    compiles = s.get("compiles", False)
                    judge_score = s.get("judge_score", 0.0)
                    if compiles and judge_score >= 0.8:
                        return True
                return False

            if passed_in_first_k(1): pass_counts[method][1] += 1
            if passed_in_first_k(3): pass_counts[method][3] += 1
            if passed_in_first_k(5): pass_counts[method][5] += 1

    # Convert counts to percentages
    pass_rates = {}
    overhead_stats = {}
    
    for m in METHODS:
        pass_rates[m] = {
            k: (pass_counts[m][k] / total_prompts) * 100
            for k in [1, 3, 5]
        }
        
        times = generation_times[m]
        mean_time = statistics.mean(times) if times else 0.0
        std_time = statistics.stdev(times) if len(times) > 1 else 0.0
            
        toks = tokens_spent[m]
        mean_toks = statistics.mean(toks) if toks else 0.0
        std_toks = statistics.stdev(toks) if len(toks) > 1 else 0.0
            
        overhead_stats[m] = {
            "time_mean": mean_time, "time_std": std_time,
            "tok_mean": mean_toks, "tok_std": std_toks
        }
        
    dual_rates = pass_rates["Dual-Phase (Proposed)"]

    def get_delta_str(method_name, baseline_val, dual_val):
        if method_name == "Dual-Phase (Proposed)": return "-"
        if baseline_val == 0: return "N/A"
        improvement = ((dual_val - baseline_val) / baseline_val) * 100
        return f"+{improvement:.1f}\\%" if improvement > 0 else f"{improvement:.1f}\\%"

    # Dual-Phase Analytical Probabilities
    prob_phase2_occurrence = (dual_phase2_triggered / dual_total_samples * 100) if dual_total_samples > 0 else 0.0
    cond_success_phase2 = (dual_phase2_success / dual_phase2_triggered * 100) if dual_phase2_triggered > 0 else 0.0

    latex = []

    # =====================================================================
    # TABLE 1: Accuracy (pass@k)
    # =====================================================================
    latex.append("% --- TABLE 1: Accuracy ---")
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Evaluation of code generation strategies on the MiniZinc benchmark. Results denote the $pass@k$ accuracy (\\%), defined as syntactic, semantic, and compiler-verified success. The $\\Delta$ columns represent the relative improvement of the proposed Dual-Phase architecture over the respective baseline.}")
    latex.append("  \\label{tab:pass_at_k}")
    latex.append("  \\begin{tabular}{l r r r r r r}")
    latex.append("    \\toprule")
    latex.append("    \\textbf{Method} & \\textbf{pass@1} & \\textbf{$\\Delta$} & \\textbf{pass@3} & \\textbf{$\\Delta$} & \\textbf{pass@5} & \\textbf{$\\Delta$} \\\\")
    latex.append("    \\midrule")
    
    for m in METHODS:
        p1, p3, p5 = pass_rates[m][1], pass_rates[m][3], pass_rates[m][5]
        d1, d3, d5 = get_delta_str(m, p1, dual_rates[1]), get_delta_str(m, p3, dual_rates[3]), get_delta_str(m, p5, dual_rates[5])
        row_title = f"\\textbf{{{m}}}" if m == "Dual-Phase (Proposed)" else m
        latex.append(f"    {row_title} & {p1:.1f} & {d1} & {p3:.1f} & {d3} & {p5:.1f} & {d5} \\\\")
        
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    latex.append("\n\\vspace{1em}\n")

    # =====================================================================
    # TABLE 2: Computational Overhead (Time & Tokens)
    # =====================================================================
    latex.append("% --- TABLE 2: Computational Overhead ---")
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Computational overhead of the evaluated methodologies. The table reports the mean generation time in seconds and the mean total tokens generated per sample, alongside their respective standard deviations (Std).}")
    latex.append("  \\label{tab:computational_overhead}")
    latex.append("  \\begin{tabular}{l r r r r}")
    latex.append("    \\toprule")
    latex.append("    \\textbf{Method} & \\textbf{Mean Time (s)} & \\textbf{Time Std (s)} & \\textbf{Mean Tokens} & \\textbf{Tokens Std} \\\\")
    latex.append("    \\midrule")
    
    for m in METHODS:
        mean_t, std_t = overhead_stats[m]["time_mean"], overhead_stats[m]["time_std"]
        mean_tok, std_tok = overhead_stats[m]["tok_mean"], overhead_stats[m]["tok_std"]
        row_title = f"\\textbf{{{m}}}" if m == "Dual-Phase (Proposed)" else m
        latex.append(f"    {row_title} & {mean_t:.2f} & {std_t:.2f} & {mean_tok:.1f} & {std_tok:.1f} \\\\")
        
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")
    latex.append("\n\\vspace{1em}\n")

    # =====================================================================
    # TABLE 3: Dual-Phase Internal Routing Analytics
    # =====================================================================
    latex.append("% --- TABLE 3: Dual-Phase Routing Analytics ---")
    latex.append("\\begin{table}[htbp]")
    latex.append("  \\centering")
    latex.append("  \\caption{Internal routing analytics for the Dual-Phase Cascaded Architecture. The table details the probability that a sample required the Phase 2 GCD fallback (i.e., the optimistic draft failed compilation), and the conditional probability that Phase 2 successfully repaired the sequence to yield a passing model.}")
    latex.append("  \\label{tab:dual_phase_analytics}")
    latex.append("  \\begin{tabular}{r r}")
    latex.append("    \\toprule")
    latex.append("    \\textbf{Routing Metric} & \\textbf{Rate (\\%)} \\\\")
    latex.append("    \\midrule")
    latex.append(f"    Phase 2 Trigger Rate ($P(\\text{{Phase 2}} | \\text{{Phase 1 Fails}})$) & {prob_phase2_occurrence:.1f}\\% \\\\")
    latex.append(f"    Conditional Success Rate ($P(\\text{{Pass}} | \\text{{Phase 2 Triggered}})$) & {cond_success_phase2:.1f}\\% \\\\")
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")

    # Output to console
    print("\n--- Generated LaTeX Tables ---\n")
    print("\n".join(latex))
    print("\n------------------------------\n")
    
    with open("tables_results.tex", "w") as f:
        f.write("\n".join(latex))
    print("Tables successfully saved to 'tables_results.tex'.")

if __name__ == "__main__":
    generate_latex_table()