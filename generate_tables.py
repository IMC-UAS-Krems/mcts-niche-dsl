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

    for item in details:
        evaluations = item.get("evaluations", {})
        
        for method in METHODS:
            if method not in evaluations:
                continue
                
            samples = evaluations[method].get("samples", [])
            
            # Extract metrics for all generated samples of this method
            for s in samples:
                t = s.get("generation_time")
                if t is not None:
                    generation_times[method].append(t)
                    
                tok = s.get("tokens_spent")
                if tok is not None:
                    tokens_spent[method].append(tok)
            
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
        # Pass rates
        pass_rates[m] = {
            k: (pass_counts[m][k] / total_prompts) * 100
            for k in [1, 3, 5]
        }
        
        # Time stats
        times = generation_times[m]
        if times:
            mean_time = statistics.mean(times)
            std_time = statistics.stdev(times) if len(times) > 1 else 0.0
        else:
            mean_time = 0.0
            std_time = 0.0
            
        # Token stats
        toks = tokens_spent[m]
        if toks:
            mean_toks = statistics.mean(toks)
            std_toks = statistics.stdev(toks) if len(toks) > 1 else 0.0
        else:
            mean_toks = 0.0
            std_toks = 0.0
            
        overhead_stats[m] = {
            "time_mean": mean_time, 
            "time_std": std_time,
            "tok_mean": mean_toks,
            "tok_std": std_toks
        }
        
    dual_rates = pass_rates["Dual-Phase (Proposed)"]

    # Helper function to calculate the relative percentage improvement
    def get_delta_str(method_name, baseline_val, dual_val):
        if method_name == "Dual-Phase (Proposed)":
            return "-"
        if baseline_val == 0:
            return "N/A"
        improvement = ((dual_val - baseline_val) / baseline_val) * 100
        if improvement > 0:
            return f"+{improvement:.1f}\\%"
        else:
            return f"{improvement:.1f}\\%"

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
        p1 = pass_rates[m][1]
        p3 = pass_rates[m][3]
        p5 = pass_rates[m][5]
        
        # Calculate increments for each metric
        d1 = get_delta_str(m, p1, dual_rates[1])
        d3 = get_delta_str(m, p3, dual_rates[3])
        d5 = get_delta_str(m, p5, dual_rates[5])
        
        # Format the method name
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
        mean_t = overhead_stats[m]["time_mean"]
        std_t = overhead_stats[m]["time_std"]
        mean_tok = overhead_stats[m]["tok_mean"]
        std_tok = overhead_stats[m]["tok_std"]
        
        row_title = f"\\textbf{{{m}}}" if m == "Dual-Phase (Proposed)" else m
        
        latex.append(f"    {row_title} & {mean_t:.2f} & {std_t:.2f} & {mean_tok:.1f} & {std_tok:.1f} \\\\")
        
    latex.append("    \\bottomrule")
    latex.append("  \\end{tabular}")
    latex.append("\\end{table}")

    # Output to console
    print("\n--- Generated LaTeX Tables ---\n")
    print("\n".join(latex))
    print("\n------------------------------\n")
    
    # Optionally save to file
    with open("tables_results.tex", "w") as f:
        f.write("\n".join(latex))
    print("Tables successfully saved to 'tables_results.tex'.")

if __name__ == "__main__":
    generate_latex_table()