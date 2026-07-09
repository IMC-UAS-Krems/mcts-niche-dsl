import json
import os
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# Configuration
# =====================================================================
RESULTS_FILE = "evaluation_pass_5_results.json"
BENCHMARK_FILE = "minizinc_benchmark.json"
OUTPUT_IMAGE = "accuracy_by_category.png"

METHODS = [
    "Zero-Shot",
    "One-Shot (No GCD)",
    "One-Shot (GCD Only)",
    "One-Shot (CoT + Always GCD)",
    "Dual-Phase (Proposed)"
]

# Raw category strings as they appear in the JSON
CATEGORIES = ["Basic", "Arithmetic", "DataStructures", "Logic", "Optimization"]
# Formatted strings for the X-axis labels
DISPLAY_CATEGORIES = ["Basic", "Arithmetic", "Data Structures", "Logic", "Optimization"]

def visualize_category_performance():
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(BENCHMARK_FILE):
        print(f"Error: Ensure both '{RESULTS_FILE}' and '{BENCHMARK_FILE}' are in the directory.")
        return

    # 1. Load the files
    with open(BENCHMARK_FILE, "r") as f:
        benchmark_data = json.load(f)
        
    with open(RESULTS_FILE, "r") as f:
        results_data = json.load(f)
        
    # Create a mapping of Intent -> Category to easily tag the results
    category_map = {item["nl"]: item.get("category", "Basic") for item in benchmark_data}

    details = results_data.get("details", [])
    
    # Tracking structures
    total_prompts = {c: 0 for c in CATEGORIES}
    pass_counts = {m: {c: 0 for c in CATEGORIES} for m in METHODS}

    print("Aggregating pass@5 performance by category...")

    # 2. Extract Data
    for item in details:
        intent = item.get("intent", "")
        category = category_map.get(intent, "Basic")
        
        if category in total_prompts:
            total_prompts[category] += 1
            
        evaluations = item.get("evaluations", {})
        
        for method in METHODS:
            if method not in evaluations:
                continue
                
            samples = evaluations[method].get("samples", [])
            
            # pass@5 Check: Did any of the first 5 samples pass?
            passed = False
            for s in samples[:5]:
                compiles = s.get("compiles", False)
                judge_score = s.get("judge_score", 0.0)
                if compiles and judge_score >= 0.8:
                    passed = True
                    break
                    
            if passed:
                pass_counts[method][category] += 1

    # 3. Calculate Percentages
    pass_rates = {m: {} for m in METHODS}
    for m in METHODS:
        for c in CATEGORIES:
            total = total_prompts[c]
            count = pass_counts[m][c]
            pass_rates[m][c] = (count / total * 100) if total > 0 else 0.0

    # Print Text Summary
    print("\n" + "="*70)
    print("pass@5 ACCURACY BY TASK CATEGORY")
    print("="*70)
    counts_str = ", ".join([f"{d_cat}: {total_prompts[c]}" for c, d_cat in zip(CATEGORIES, DISPLAY_CATEGORIES)])
    print(f"Instances per category -> {counts_str}")
    print("-" * 70)
    for m in METHODS:
        rates = pass_rates[m]
        rates_str = " | ".join([f"{rates[c]:5.1f}%" for c in CATEGORIES])
        print(f"{m:>28} | {rates_str}")

    # =====================================================================
    # Visualization (Grouped Bar Chart)
    # =====================================================================
    x = np.arange(len(CATEGORIES))
    width = 0.15 # Width of the bars
    multiplier = 0 # To offset the bars

    # Use a wider figure to accommodate 5 categories cleanly
    fig, ax = plt.subplots(figsize=(15, 7))

    # Academic qualitative color palette (matches previous charts)
    colors = ['#d62728', '#9467bd', '#8c564b', '#1f77b4', '#2ca02c']

    for i, m in enumerate(METHODS):
        rates = [pass_rates[m][c] for c in CATEGORIES]
        
        offset = width * multiplier
        bars = ax.bar(x + offset, rates, width, label=m, color=colors[i], edgecolor='black', zorder=3)
        
        # Annotate bars with the percentage
        for bar, rate in zip(bars, rates):
            if rate > 0: # Avoid cluttering 0% bars
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{rate:.0f}%", ha='center', va='bottom', fontsize=9)
        
        multiplier += 1

    # Formatting
    ax.set_title('pass@5 Accuracy by Task Category', fontsize=16, fontweight='bold', pad=20)
    ax.set_ylabel('Success Rate (%)', fontsize=13)
    
    # Center the X-axis labels under the groups
    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels(DISPLAY_CATEGORIES, fontsize=12, fontweight='bold')
    
    # Legend and Grid
    # Placed outside the plot to prevent obscuring the bars
    ax.legend(title="Generation Method", title_fontsize='12', fontsize=11, 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Set Y-axis limit to slightly above 100% to fit the text labels
    ax.set_ylim(0, 115)

    # Adjust layout to make room for the legend below
    plt.subplots_adjust(bottom=0.25)
    
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved as '{OUTPUT_IMAGE}'.")
    
    plt.show()

if __name__ == "__main__":
    visualize_category_performance()