import json
import os
import matplotlib.pyplot as plt
import numpy as np

# =====================================================================
# Configuration
# =====================================================================
RESULTS_FILE = "evaluation_pass_5_results.json"
BENCHMARK_FILE = "minizinc_benchmark.json"
OUTPUT_IMAGE = "accuracy_by_difficulty.png"

METHODS = [
    "Zero-Shot",
    "One-Shot (No GCD)",
    "One-Shot (GCD Only)",
    "One-Shot (CoT + Always GCD)",
    "Dual-Phase (Proposed)"
]

DIFFICULTIES = ["easy", "medium", "hard"]

def visualize_difficulty_performance():
    if not os.path.exists(RESULTS_FILE) or not os.path.exists(BENCHMARK_FILE):
        print(f"Error: Ensure both '{RESULTS_FILE}' and '{BENCHMARK_FILE}' are in the directory.")
        return

    # 1. Load the files
    with open(BENCHMARK_FILE, "r") as f:
        benchmark_data = json.load(f)
        
    with open(RESULTS_FILE, "r") as f:
        results_data = json.load(f)
        
    # Create a mapping of Intent -> Difficulty to easily tag the results
    # (Assuming the "nl" field in benchmark exactly matches the "intent" in results)
    difficulty_map = {item["nl"]: item.get("difficulty", "medium") for item in benchmark_data}

    details = results_data.get("details", [])
    
    # Tracking structures
    total_prompts = {d: 0 for d in DIFFICULTIES}
    pass_counts = {m: {d: 0 for d in DIFFICULTIES} for m in METHODS}

    print("Aggregating pass@5 performance by difficulty...")

    # 2. Extract Data
    for item in details:
        intent = item.get("intent", "")
        # Fallback to 'medium' if mapping fails, though it shouldn't
        difficulty = difficulty_map.get(intent, "medium").lower()
        
        if difficulty in total_prompts:
            total_prompts[difficulty] += 1
            
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
                pass_counts[method][difficulty] += 1

    # 3. Calculate Percentages
    pass_rates = {m: {} for m in METHODS}
    for m in METHODS:
        for d in DIFFICULTIES:
            total = total_prompts[d]
            count = pass_counts[m][d]
            pass_rates[m][d] = (count / total * 100) if total > 0 else 0.0

    # Print Text Summary
    print("\n" + "="*50)
    print("pass@5 ACCURACY BY DIFFICULTY")
    print("="*50)
    print(f"Total instances - Easy: {total_prompts['easy']}, Medium: {total_prompts['medium']}, Hard: {total_prompts['hard']}")
    print("-" * 50)
    for m in METHODS:
        rates = pass_rates[m]
        print(f"{m:>28} | Easy: {rates['easy']:5.1f}% | Med: {rates['medium']:5.1f}% | Hard: {rates['hard']:5.1f}%")

    # =====================================================================
    # Visualization (Grouped Bar Chart)
    # =====================================================================
    x = np.arange(len(DIFFICULTIES))
    width = 0.15 # Width of the bars
    multiplier = 0 # To offset the bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Same academic qualitative color palette
    colors = ['#d62728', '#9467bd', '#8c564b', '#1f77b4', '#2ca02c']

    for i, m in enumerate(METHODS):
        rates = [pass_rates[m][d] for d in DIFFICULTIES]
        
        offset = width * multiplier
        bars = ax.bar(x + offset, rates, width, label=m, color=colors[i], edgecolor='black', zorder=3)
        
        # Annotate bars with the percentage
        for bar, rate in zip(bars, rates):
            if rate > 0: # Avoid cluttering 0% bars
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                        f"{rate:.0f}%", ha='center', va='bottom', fontsize=9)
        
        multiplier += 1

    # Formatting
    ax.set_title('pass@5 Accuracy by Task Difficulty', fontsize=15, fontweight='bold', pad=20)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    
    # Center the X-axis labels under the groups
    ax.set_xticks(x + width * (len(METHODS) - 1) / 2)
    ax.set_xticklabels([d.capitalize() for d in DIFFICULTIES], fontsize=12, fontweight='bold')
    
    # Legend and Grid
    ax.legend(title="Generation Method", title_fontsize='12', fontsize=11, 
              loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3)
    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Set Y-axis limit to slightly above 100% to fit the text labels
    ax.set_ylim(0, 115)

    plt.tight_layout()
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"\nSuccess! Visualization saved as '{OUTPUT_IMAGE}'.")
    
    plt.show()

if __name__ == "__main__":
    visualize_difficulty_performance()