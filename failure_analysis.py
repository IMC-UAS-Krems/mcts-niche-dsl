import json
import os
import tempfile
import subprocess
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

# =====================================================================
# Configuration
# =====================================================================
RESULTS_FILE = "evaluation_pass_5_results.json"
OUTPUT_IMAGE = "error_categorization_all.png"

# Includes the ablation baseline added in previous steps
METHODS = [
    "Zero-Shot",
    "One-Shot (No GCD)",
    "One-Shot (GCD Only)",
    "One-Shot (CoT + Always GCD)",
    "Dual-Phase (Proposed)"
]

CATEGORIES = [
    "Pure Syntax Errors", 
    "Structural/Schema Errors", 
    "Semantic Logic Errors"
]

# Compiler Cache to speed up the OS-level checks for identical failed code strings
_compiler_cache = {}

def get_compile_error_type(code: str) -> str:
    """
    Determines if a compilation failure is a Pure Syntax Error or a Structural/Schema Error.
    Uses caching to avoid redundant disk I/O.
    """
    if not code.strip():
        return "Pure Syntax Errors"
        
    if code in _compiler_cache:
        return _compiler_cache[code]
        
    fd, temp_path = tempfile.mkstemp(suffix=".mzn")
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(code)
            
        result = subprocess.run(
            ["minizinc", "--model-check-only", temp_path],
            capture_output=True, text=True, timeout=5
        )
        
        stderr = result.stderr.lower()
        if "syntax error" in stderr or "parse error" in stderr:
            res = "Pure Syntax Errors"
        else:
            res = "Structural/Schema Errors"
            
        _compiler_cache[code] = res
        return res
            
    except Exception:
        return "Pure Syntax Errors"
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def analyze_and_visualize():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run the benchmark first.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
        
    details = data.get("details", [])

    # Initialize tracking structures
    error_counts = {m: {c: 0 for c in CATEGORIES} for m in METHODS}
    total_failures = {m: 0 for m in METHODS}

    print("Analyzing failed samples across all methods...")

    for item in tqdm(details, desc="Processing Prompts"):
        evaluations = item.get("evaluations", {})
        
        for method in METHODS:
            if method not in evaluations:
                continue
                
            samples = evaluations[method].get("samples", [])
            
            for s in samples:
                compiles = s.get("compiles", False)
                judge_score = s.get("judge_score", 0.0)
                
                # Check if it is a PASS
                if compiles and judge_score >= 0.8:
                    continue 
                    
                total_failures[method] += 1
                code = s.get("code", "")
                
                # GATE 3: Semantic Logic (Compiled fine, but failed intent)
                if compiles and judge_score < 0.8:
                    error_counts[method]["Semantic Logic Errors"] += 1
                    
                # GATE 1 & 2: Syntax vs Structural Schema
                elif not compiles:
                    error_category = get_compile_error_type(code)
                    error_counts[method][error_category] += 1

    # Print Text Summary
    print("\n" + "="*50)
    print("FAILURE ANALYSIS RESULTS")
    print("="*50)
    for m in METHODS:
        print(f"\n{m} (Total Failures: {total_failures[m]}):")
        for cat in CATEGORIES:
            count = error_counts[m][cat]
            pct = (count / total_failures[m] * 100) if total_failures[m] > 0 else 0
            print(f"  - {cat}: {count} ({pct:.1f}%)")

    # =====================================================================
    # Visualization (Grouped Bar Chart)
    # =====================================================================
    # We want X-axis = Categories, Bars inside = Methods
    x = np.arange(len(CATEGORIES))
    width = 0.15 # Width of the bars
    multiplier = 0 # To offset the bars

    fig, ax = plt.subplots(figsize=(14, 7))

    # Academic-friendly qualitative color palette
    colors = ['#d62728', '#9467bd', '#8c564b', '#1f77b4', '#2ca02c']

    for i, m in enumerate(METHODS):
        counts = [error_counts[m][c] for c in CATEGORIES]
        
        # Calculate percentage for text annotation
        percentages = [(c / total_failures[m] * 100) if total_failures[m] > 0 else 0 for c in counts]
        
        # Calculate bar positions
        offset = width * multiplier
        bars = ax.bar(x + offset, counts, width, label=m, color=colors[i], edgecolor='black', zorder=3)
        
        # Annotate bars
        for bar, count, pct in zip(bars, counts, percentages):
            if count > 0: # Only annotate non-zero bars to avoid clutter
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (max([max(error_counts[method].values()) for method in METHODS]) * 0.01),
                        f"{int(count)}\n({pct:.1f}%)",
                        ha='center', va='bottom', fontsize=8, rotation=0)
        
        multiplier += 1

    # Formatting
    ax.set_title('Error Taxonomy of Failed Generations Across Methodologies', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Absolute Number of Failed Samples', fontsize=13)
    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels(CATEGORIES, fontsize=12, fontweight='bold')
    ax.legend(title="Generation Method", title_fontsize='11', fontsize=10, loc='upper left', bbox_to_anchor=(1, 1))
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Adjust limits to fit labels
    global_max = max([max(error_counts[m].values()) for m in METHODS])
    ax.set_ylim(0, global_max * 1.15)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved as '{OUTPUT_IMAGE}'.")
    
    plt.show()

if __name__ == "__main__":
    analyze_and_visualize()