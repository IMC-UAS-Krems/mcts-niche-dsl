import json
import os
import plotly.graph_objects as go

# =====================================================================
# Configuration
# =====================================================================
RESULTS_FILE = "evaluation_pass_5_results.json"
OUTPUT_HTML = "sankey_diagram.html"
OUTPUT_IMAGE = "sankey_diagram.png" # Requires 'kaleido' package

def generate_sankey():
    if not os.path.exists(RESULTS_FILE):
        print(f"Error: {RESULTS_FILE} not found. Please run the benchmark first.")
        return

    with open(RESULTS_FILE, "r") as f:
        data = json.load(f)
        
    details = data.get("details", [])

    # -----------------------------------------------------------------
    # 1. Extract Routing Data from Dual-Phase Samples
    # -----------------------------------------------------------------
    # Because of pass@k short-circuiting, the total number of samples 
    # will be less than (100 prompts * 5). We count actual generated samples.
    
    fp_pass = 0  # Fast Path -> Semantic Pass
    fp_fail = 0  # Fast Path -> Semantic Fail
    p2_pass = 0  # Phase 2 -> Semantic Pass
    p2_fail = 0  # Phase 2 -> Semantic Fail

    for item in details:
        evaluations = item.get("evaluations", {})
        dual_eval = evaluations.get("Dual-Phase (Proposed)", {})
        samples = dual_eval.get("samples", [])
        
        for s in samples:
            # Did it take the Optimistic Bypass (Fast Path)?
            is_fast_path = s.get("fast_path_success", False)
            
            # Did it ultimately pass the semantic judge?
            compiles = s.get("compiles", False)
            judge_score = s.get("judge_score", 0.0)
            is_pass = (compiles and judge_score >= 0.8)
            
            if is_fast_path:
                if is_pass:
                    fp_pass += 1
                else:
                    fp_fail += 1
            else:
                if is_pass:
                    p2_pass += 1
                else:
                    p2_fail += 1

    total_samples = fp_pass + fp_fail + p2_pass + p2_fail
    total_fp = fp_pass + fp_fail
    total_p2 = p2_pass + p2_fail
    total_pass = fp_pass + p2_pass
    total_fail = fp_fail + p2_fail

    print(f"Total Samples Generated: {total_samples}")
    print(f"  -> Fast Path: {total_fp} (Pass: {fp_pass}, Fail: {fp_fail})")
    print(f"  -> Phase 2 GCD: {total_p2} (Pass: {p2_pass}, Fail: {p2_fail})")

    # -----------------------------------------------------------------
    # 2. Build the Sankey Diagram
    # -----------------------------------------------------------------
    # Node Indices:
    # 0: Total Generated Samples
    # 1: Fast Path (Phase 1 Draft)
    # 2: Phase 2 (GCD Fallback)
    # 3: Final Pass
    # 4: Final Fail
    
    node_labels = [
        f"Generated Samples<br>({total_samples})", 
        f"Optimistic Fast-Path<br>({total_fp})", 
        f"Phase 2 GCD Fallback<br>({total_p2})", 
        f"Semantic Pass<br>({total_pass})", 
        f"Semantic Fail<br>({total_fail})"
    ]
    
    # Publication-ready color palette
    node_colors = [
        "#1f77b4", # Blue (Source)
        "#2ca02c", # Greenish (Fast Path)
        "#ff7f0e", # Orange (Phase 2)
        "#2ca02c", # Green (Pass)
        "#d62728"  # Red (Fail)
    ]

    source = [0, 0, 1, 1, 2, 2]
    target = [1, 2, 3, 4, 3, 4]
    values = [total_fp, total_p2, fp_pass, fp_fail, p2_pass, p2_fail]
    
    # Link colors (semi-transparent matching the source node)
    link_colors = [
        "rgba(31, 119, 180, 0.4)",  # 0 -> 1
        "rgba(31, 119, 180, 0.4)",  # 0 -> 2
        "rgba(44, 160, 44, 0.4)",   # 1 -> 3
        "rgba(44, 160, 44, 0.2)",   # 1 -> 4 (Faded green-to-red)
        "rgba(255, 127, 14, 0.4)",  # 2 -> 3
        "rgba(255, 127, 14, 0.4)"   # 2 -> 4
    ]

    fig = go.Figure(data=[go.Sankey(
        node=dict(
            pad=30,
            thickness=30,
            line=dict(color="black", width=1.0),
            label=node_labels,
            color=node_colors
        ),
        link=dict(
            source=source,
            target=target,
            value=values,
            color=link_colors
        )
    )])

    fig.update_layout(
        title_text="Dual-Phase Cascaded Architecture: Internal Routing & Resolution Flow",
        title_font_size=18,
        font=dict(size=14, color="black"),
        width=900,
        height=500,
        plot_bgcolor='white',
        paper_bgcolor='white'
    )

    # -----------------------------------------------------------------
    # 3. Export
    # -----------------------------------------------------------------
    # Save as interactive HTML
    # fig.write_html(OUTPUT_HTML)
    # print(f"\nInteractive diagram saved to '{OUTPUT_HTML}'.")

    # Try to save as a static image for the paper
    try:
        fig.write_image(OUTPUT_IMAGE, scale=2.0)
        print(f"High-resolution static image saved to '{OUTPUT_IMAGE}'.")
    except Exception as e:
        print(f"Note: Could not save static PNG (requires 'kaleido' package). Error: {e}")
        
    # Open in browser
    fig.show()

if __name__ == "__main__":
    generate_sankey()