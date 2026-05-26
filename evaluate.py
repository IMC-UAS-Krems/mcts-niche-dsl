import json
import os
from tqdm import tqdm
from typing import List
from dotenv import load_dotenv

load_dotenv()


# Import your components from their respective files
# (Adjust the import names based on what you named your Python files)
from mcts_generator import MiniZincEnvironment, OllamaLLMHeuristic, NeurosymbolicMCTS
from baselines import (
    baseline_1_zero_shot, 
    baseline_2_one_shot_grammar, 
    baseline_3_grammar_constrained, 
    evaluate_generated_code
)

# =====================================================================
# Configuration
# =====================================================================
BENCHMARK_FILE = "benchmark_test.json"
RESULTS_FILE = "evaluation_results.json"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")

# K-sampling settings
# If K=3, we generate 3 samples per method. If ANY of the 3 pass, the prompt is marked as solved.
K_SAMPLES = 3  

# =====================================================================
# Helper: Evaluate a Single Code Sample
# =====================================================================
def is_successful_generation(prompt: str, code: str, llm_judge: OllamaLLMHeuristic) -> bool:
    """Checks Syntax, Compilation, and Intent."""
    
    # 1. Use the unified evaluator from baselines.py for Syntax and Compilation
    eval_results = evaluate_generated_code(code, prompt)
    
    if not eval_results["syntax_pass"] or not eval_results["compile_pass"]:
        return False
        
    # 2. Use the LLM Judge for Semantic Intent
    # We must parse the AST first to feed to the judge
    try:
        from minizinc_parser import parse_model
        ast = parse_model(eval_results["raw_code"])
        reward = llm_judge.evaluate_code(prompt=prompt, code=eval_results["raw_code"], ast=ast)
        
        # A reward >= 0.8 means the code accurately reflects the prompt
        return reward >= 0.8
    except Exception:
        return False

# =====================================================================
# Generators Wrappers (Generating K samples)
# =====================================================================
def run_mcts_k_times(prompt: str, judge: OllamaLLMHeuristic, k: int) -> List[str]:
    samples = []
    # Extract entities once to save time
    extracted_data = judge.extract_entities(prompt=prompt)
    
    for _ in range(k):
        env = MiniZincEnvironment(target_prompt=prompt, llm_judge=judge, extracted_entities=extracted_data)
        # Note: num_simulations can be lowered slightly here to speed up K-sampling
        mcts = NeurosymbolicMCTS(env=env, llm_policy=judge, c_puct=1.5)
        initial_ast = ("<Model>",)
        
        code = mcts.generate_code(initial_ast, max_steps=100, num_simulations=100)
        samples.append(code)
    return samples

# =====================================================================
# Main Evaluation Loop
# =====================================================================
def run_benchmark():
    print(f"Loading Benchmark from {BENCHMARK_FILE}...")
    with open(BENCHMARK_FILE, "r") as f:
        dataset = json.load(f)

    # Initialize the LLM Judge (Reused across all evaluations)
    minizinc_aliases = {
        "\\/": "Logical OR (either/or)", "/\\": "Logical AND (both)",
        "->": "Logical Implication (if/then)", "==": "Equality", "!=": "Inequality"
    }

    print(f"Initializing LLM Judge with model '{OLLAMA_MODEL}'...")
    llm_judge = OllamaLLMHeuristic(
        prompt="", # Prompt is updated dynamically during eval
        model=OLLAMA_MODEL,
        dsl_name="MiniZinc",
        dsl_description="constraint programming",
        action_aliases=minizinc_aliases
    )

    # Tracking Success Rates
    results = {
        "Zero-Shot": {"successes": 0, "failures": 0},
        "One-Shot":  {"successes": 0, "failures": 0},
        "GCD":       {"successes": 0, "failures": 0},
        "MCTS":      {"successes": 0, "failures": 0}
    }
    
    detailed_log = []

    print(f"Starting pass@{K_SAMPLES} evaluation on {len(dataset)} prompts...\n")
    
    # We use tqdm for a progress bar, as 100 prompts * 4 methods * K samples takes a long time!
    for i, item in enumerate(tqdm(dataset, desc="Evaluating Prompts")):
        prompt = item["nl"]
        target_code = item["code"]
        llm_judge.prompt = prompt # Update judge context
        
        prompt_log = {"id": i, "prompt": prompt, "evaluations": {}}

        # --- Evaluate Methods ---
        methods = {
            "Zero-Shot": lambda: [baseline_1_zero_shot(prompt, OLLAMA_MODEL) for _ in range(K_SAMPLES)],
            "One-Shot":  lambda: [baseline_2_one_shot_grammar(prompt, OLLAMA_MODEL) for _ in range(K_SAMPLES)],
            "GCD":       lambda: [baseline_3_grammar_constrained(prompt) for _ in range(K_SAMPLES)],
            "MCTS":      lambda: run_mcts_k_times(prompt, llm_judge, K_SAMPLES)
        }

        for method_name, generate_func in methods.items():
            try:
                samples = generate_func()
                
                # pass@k logic: Check if AT LEAST ONE sample is correct
                passed = False
                for sample_code in samples:
                    if is_successful_generation(prompt, sample_code, llm_judge):
                        passed = True
                        break # We only need one pass to satisfy pass@k
                        
                if passed:
                    results[method_name]["successes"] += 1
                else:
                    results[method_name]["failures"] += 1
                    
                prompt_log["evaluations"][method_name] = {"pass": passed, "samples": samples}
                
            except Exception as e:
                print(f"\n[Error] {method_name} failed on prompt {i}: {e}")
                results[method_name]["failures"] += 1
                prompt_log["evaluations"][method_name] = {"pass": False, "error": str(e)}

        detailed_log.append(prompt_log)
        
        # Save intermediate results so you don't lose data if it crashes
        with open(RESULTS_FILE, "w") as f:
            json.dump({"aggregate": results, "details": detailed_log}, f, indent=2)

    # --- Print Final Metrics ---
    print("\n" + "="*50)
    print(f"FINAL pass@{K_SAMPLES} ACCURACY ({len(dataset)} Prompts)")
    print("="*50)
    for method, metrics in results.items():
        total = metrics["successes"] + metrics["failures"]
        accuracy = (metrics["successes"] / total) * 100 if total > 0 else 0
        print(f"{method:>12}: {accuracy:.1f}% ({metrics['successes']}/{total})")

if __name__ == "__main__":
    run_benchmark()
