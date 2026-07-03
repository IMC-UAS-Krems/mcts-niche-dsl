import json
import matplotlib.pyplot as plt
from collections import Counter
import os

# =====================================================================
# Configuration
# =====================================================================
DATASET_FILE = "minizinc_benchmark.json"
OUTPUT_IMAGE = "dataset_statistics.png"

def visualize_statistics():
    if not os.path.exists(DATASET_FILE):
        print(f"Error: '{DATASET_FILE}' not found. Please ensure the dataset file exists.")
        return

    # 1. Load the Dataset
    with open(DATASET_FILE, "r") as f:
        dataset = json.load(f)

    # 2. Extract Data
    categories = [item.get("category") for item in dataset if "category" in item]
    difficulties = [item.get("difficulty") for item in dataset if "difficulty" in item]

    # 3. Count Frequencies
    cat_counts = Counter(categories)
    diff_counts = Counter(difficulties)

    # 4. Sort Data for Visualization
    # Sort categories by frequency (descending)
    cat_sorted = dict(sorted(cat_counts.items(), key=lambda item: item[1], reverse=True))
    cat_labels = list(cat_sorted.keys())
    cat_values = list(cat_sorted.values())

    # Sort difficulties logically (Easy -> Medium -> Hard)
    diff_order = ["easy", "medium", "hard"]
    diff_labels = [d for d in diff_order if d in diff_counts]
    diff_values = [diff_counts[d] for d in diff_labels]

    # 5. Set up the Matplotlib Figure (1 Row, 2 Columns)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Plot 1: Categories ---
    bars_cat = axes[0].bar(cat_labels, cat_values, color='#4C72B0', edgecolor='black', zorder=3)
    axes[0].set_title('Task Distribution by Category', fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Number of Instances', fontsize=12)
    axes[0].tick_params(axis='x', rotation=30)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Annotate bar values
    for bar in bars_cat:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # --- Plot 2: Difficulties ---
    # Use traffic-light colors for difficulties (Green, Orange, Red)
    diff_colors = ['#55A868', '#F1A340', '#C44E52']
    bars_diff = axes[1].bar(diff_labels, diff_values, color=diff_colors, edgecolor='black', zorder=3)
    axes[1].set_title('Task Distribution by Difficulty', fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Number of Instances', fontsize=12)
    
    # Capitalize the x-axis labels (Easy, Medium, Hard)
    axes[1].set_xticklabels([label.capitalize() for label in diff_labels])
    axes[1].grid(axis='y', linestyle='--', alpha=0.7, zorder=0)

    # Annotate bar values
    for bar in bars_diff:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2, yval + 0.5, int(yval), 
                     ha='center', va='bottom', fontsize=11, fontweight='bold')

    # 6. Final Layout Adjustments
    plt.tight_layout()
    
    # Save the figure in high resolution for the paper
    plt.savefig(OUTPUT_IMAGE, dpi=300, bbox_inches='tight')
    print(f"Success! Figure saved as '{OUTPUT_IMAGE}'")
    
    # Display the plot
    plt.show()

if __name__ == "__main__":
    visualize_statistics()