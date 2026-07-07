import json
import os
import tempfile
import subprocess
import matplotlib.pyplot as plt

# =====================================================================
# Configuration
# =====================================================================
RESULTS_FILE = "evaluation_pass_5_results.json"
OUTPUT_IMAGE = "error_categorization.png"
TARGET_METHOD = "Dual-Phase (Proposed)"

def get_compile_error_type(code: str) -> str:
    """
    Determines if a compilation failure is a Pure Syntax Error or a Structural/Schema Error.
    """
    if not code.strip():
        return "Pure Syntax Errors"
        
    fd, temp_path = tempfile.mkstemp(suffix=".mzn")
    try:
        with os.fdopen(fd, 'w') as f:
            f.write(code)
            
        result = subprocess.run(
            ["minizinc", "--model-check-only", temp_path],
            capture_output=True, text=True, timeout=5
        )
        
        stderr = result.stderr.lower()
        # If the compiler throws a syntax/parse error, it failed Gate 1
        if "syntax error" in stderr or "parse error" in stderr:
            return "Pure Syntax Errors"
        else:
            # If it's a type error, uninitialized variable, etc., it failed Gate 2
            return "Structural/Schema Errors"
            
    except Exception:
        # Fallback in case of timeout or fatal crash
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

    # Counters for our Error Taxonomy
    error_counts = {
        "Pure Syntax Errors": 0,
        "Structural/Schema Errors": 0,
        "Semantic Logic Errors": 0
    }
    
    total_samples = 0
    total_failures = 0

    print(f"Analyzing failed samples for '{TARGET_METHOD}'...")

    for item in details:
        evaluations = item.get("evaluations", {})
        if TARGET_METHOD not in evaluations:
            continue
            
        samples = evaluations[TARGET_METHOD].get("samples", [])
        
        for s in samples:
            total_samples += 1
            compiles = s.get("compiles", False)
            judge_score = s.get("judge_score", 0.0)
            
            # Check if it is a failure
            if compiles and judge_score >= 0.8:
                continue # Passed, skip
                
            total_failures += 1
            code = s.get("code", "")
            
            # --- GATE 3 FAILURE (Semantic Logic) ---
            if compiles and judge_score < 0.8:
                error_counts["Semantic Logic Errors"] += 1
                
            # --- GATE 1 or 2 FAILURE (Syntax vs Schema) ---
            elif not compiles:
                error_category = get_compile_error_type(code)
                error_counts[error_category] += 1

    print("\n--- Failure Analysis Results ---")
    print(f"Total Samples Generated: {total_samples}")
    print(f"Total Failed Samples: {total_failures}")
    for cat, count in error_counts.items():
        pct = (count / total_failures * 100) if total_failures > 0 else 0
        print(f"- {cat}: {count} ({pct:.1f}%)")

    # =====================================================================
    # Visualization (Bar Chart)
    # =====================================================================
    categories = list(error_counts.keys())
    counts = list(error_counts.values())
    
    # Calculate percentages for labels
    percentages = [(c / total_failures * 100) if total_failures > 0 else 0 for c in counts]

    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Academic color palette mapping to the severity of the error
    colors = ['#d62728', '#ff7f0e', '#ffbb78']  # Red, Orange, Light Orange
    
    bars = ax.bar(categories, counts, color=colors, edgecolor='black', zorder=3)
    
    # Formatting
    ax.set_title(f"Error Taxonomy of Failed Generations\n({TARGET_METHOD})", fontsize=14, fontweight='bold', pad=15)
    ax.set_ylabel("Number of Failed Samples", fontsize=12)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
    
    # Add counts and percentages on top of bars
    for bar, count, pct in zip(bars, counts, percentages):
        yval = bar.get_height()
        # Offset slightly above the bar
        ax.text(bar.get_x() + bar.get_width()/2, yval + (max(counts)*0.02), 
                f"{int(count)}\n({pct:.1f}%)", 
                ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Extend Y-axis slightly so labels don't get cut off
    ax.set_ylim(0, max(counts) * 1.15 if max(counts) > 0 else 10)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved as '{OUTPUT_IMAGE}'.")
    
    plt.show()

if __name__ == "__main__":
    analyze_and_visualize()