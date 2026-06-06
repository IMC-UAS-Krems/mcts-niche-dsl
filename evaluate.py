import json
import os
import concurrent.futures
from tqdm import tqdm
from typing import List, Dict, Any
from dotenv import load_dotenv
from minizinc_parser import minizinc_aliases

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
BENCHMARK_FILE = "minizinc_benchmark.json"
RESULTS_FILE = "evaluation_results.json"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
OLLAMA_JUDGE_MODEL = os.getenv("OLLAMA_JUDGE_MODEL", "qwen3.5:latest")

# K-sampling settings
# If K=3, we generate 3 samples per method. If ANY of the 3 pass, the prompt is marked as solved.
K_SAMPLES = 3  
MAX_WORKERS = 4 

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
def run_mcts_k_times(prompt: str, judge: OllamaLLMHeuristic, search: OllamaLLMHeuristic, k: int) -> List[str]:
    samples = []
    # Extract entities once to save time
    extracted_data = search.extract_entities(prompt=prompt)
    
    for _ in range(k):
        # Clear the cache so the LLM gets queried freshly for each run!
        search.cache = {k: v for k, v in search.cache.items() if k[0] == "eval"}

        env = MiniZincEnvironment(target_prompt=prompt, llm_judge=judge, extracted_entities=extracted_data)
        # Note: num_simulations can be lowered slightly here to speed up K-sampling
        mcts = NeurosymbolicMCTS(env=env, llm_policy=search, c_puct=1.5)
        initial_ast = ("<Model>",)
        
        code = mcts.generate_code(initial_ast, max_steps=100, num_simulations=30)
        samples.append(code)
    return samples

# =====================================================================
# Worker Function (Executes 1 Prompt completely)
# =====================================================================
def evaluate_single_prompt(item: dict, index: int) -> dict:
    """Runs all 4 methods for K samples on a single prompt."""
    prompt = item["nl"]
    
    print(f"Initializing LLM Judge with model '{OLLAMA_JUDGE_MODEL}'...")
    llm_judge = OllamaLLMHeuristic(
        prompt="", # Prompt is updated dynamically during eval
        model=OLLAMA_JUDGE_MODEL,
        dsl_name="MiniZinc",
        dsl_description="constraint programming",
        action_aliases=minizinc_aliases
    )

    print(f"Using LLM semantic model '{OLLAMA_MODEL}'...")
    llm_search = OllamaLLMHeuristic(
        prompt="", # Prompt is updated dynamically during eval
        model=OLLAMA_MODEL,
        dsl_name="MiniZinc",
        dsl_description="constraint programming",
        action_aliases=minizinc_aliases
    )

    prompt_log = {"id": index, "prompt": prompt, "evaluations": {}}
    local_results = {
        "Zero-Shot": {"successes": 0, "failures": 0},
        "One-Shot":  {"successes": 0, "failures": 0},
        "GCD":       {"successes": 0, "failures": 0},
        "MCTS":      {"successes": 0, "failures": 0}
    }

    methods = {
        "Zero-Shot": lambda: [baseline_1_zero_shot(prompt, OLLAMA_MODEL) for _ in range(K_SAMPLES)],
        "One-Shot":  lambda: [baseline_2_one_shot_grammar(prompt, OLLAMA_MODEL) for _ in range(K_SAMPLES)],
        "GCD":       lambda: [baseline_3_grammar_constrained(prompt) for _ in range(K_SAMPLES)],
        "MCTS":      lambda: run_mcts_k_times(prompt, llm_judge, llm_search, K_SAMPLES)
    }

    for method_name, generate_func in methods.items():
        try:
            samples = generate_func()
            passed = any(is_successful_generation(prompt, code, llm_judge) for code in samples)
            
            if passed:
                local_results[method_name]["successes"] += 1
            else:
                local_results[method_name]["failures"] += 1
                
            prompt_log["evaluations"][method_name] = {"pass": passed, "samples": samples}
            
        except Exception as e:
            local_results[method_name]["failures"] += 1
            prompt_log["evaluations"][method_name] = {"pass": False, "error": str(e)}

    return {"index": index, "results": local_results, "log": prompt_log}

# =====================================================================
# Main Parallel Execution
# =====================================================================
def run_benchmark():
    print(f"Loading Benchmark from {BENCHMARK_FILE}...")
    with open(BENCHMARK_FILE, "r") as f:
        dataset = json.load(f)

    # Global tracking
    global_results = {
        "Zero-Shot": {"successes": 0, "failures": 0},
        "One-Shot":  {"successes": 0, "failures": 0},
        "GCD":       {"successes": 0, "failures": 0},
        "MCTS":      {"successes": 0, "failures": 0}
    }
    detailed_log = []

    print(f"Starting pass@{K_SAMPLES} parallel evaluation on {len(dataset)} prompts (Workers: {MAX_WORKERS})...\n")
    
    # ThreadPoolExecutor handles the parallelism
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Submit all tasks to the pool
        futures = {executor.submit(evaluate_single_prompt, item, i): i for i, item in enumerate(dataset)}
        
        # Process them as they complete
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(dataset), desc="Evaluating Prompts"):
            try:
                task_output = future.result()
                
                # Safely merge local thread results into global tracking
                for method in global_results:
                    global_results[method]["successes"] += task_output["results"][method]["successes"]
                    global_results[method]["failures"]  += task_output["results"][method]["failures"]
                
                detailed_log.append(task_output["log"])
                
                # Incremental Save
                with open(RESULTS_FILE, "w") as f:
                    json.dump({"aggregate": global_results, "details": detailed_log}, f, indent=2)
                    
            except Exception as e:
                print(f"\n[Fatal Worker Error] Task failed: {e}")

    # --- Print Final Metrics ---
    print("\n" + "="*50)
    print(f"FINAL pass@{K_SAMPLES} ACCURACY ({len(dataset)} Prompts)")
    print("="*50)
    for method, metrics in global_results.items():
        total = metrics["successes"] + metrics["failures"]
        accuracy = (metrics["successes"] / total) * 100 if total > 0 else 0
        print(f"{method:>12}: {accuracy:.1f}% ({metrics['successes']}/{total})")

if __name__ == "__main__":
    run_benchmark()
