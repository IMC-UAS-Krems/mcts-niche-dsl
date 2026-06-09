import os
import json
import torch
import subprocess
import tempfile
import requests
from tqdm import tqdm
from typing import List, Tuple, Callable, Dict, Any

try:
    import outlines
    from outlines.types import CFG
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("Please install dependencies: pip install outlines transformers torch")

# =====================================================================
# Configuration
# =====================================================================
BENCHMARK_FILE = "benchmark_test.json"
RESULTS_FILE = "evaluation_pass_5_results.json"
K_SAMPLES = 5
TEMPERATURE = 0.6  # Required for diverse k-sampling

# Generator Model (HuggingFace)
GENERATOR_MODEL_NAME = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
# Judge Model (Ollama)
JUDGE_MODEL_NAME = os.getenv("OLLAMA_MODEL", "qwen3.5") 

# =====================================================================
# Semantic Judge Check
# =====================================================================
def judge_semantic_alignment(dsl_name: str, intent: str, generated_code: str, golden_code: str) -> Tuple[float, str]:
    """Uses Qwen 3.5 via Ollama to score semantic alignment against the golden solution."""
    api_url = "http://localhost:11434/api/generate"
    
    sys_instruction = (
        f"You are an expert {dsl_name} code reviewer. You must compare the 'Generated Code' against the 'Golden Solution'. "
        "Return a JSON object with strictly two keys:\n"
        "1. 'reasoning': Explain if the logic, variables, and constraints match exactly.\n"
        "2. 'score': A float between 0.0 and 1.0 (1.0 = perfect semantic match, 0.0 = logically broken or different intent)."
    )
    
    user_msg = f"User Intent: {intent}\n\nGolden Solution:\n{golden_code}\n\nGenerated Code:\n{generated_code}"
    
    payload = {
        "model": JUDGE_MODEL_NAME,
        "prompt": f"{sys_instruction}\n\n{user_msg}",
        "format": "json",
        "stream": False,
        "options": {"temperature": 0.0}
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=30)
        output = json.loads(response.json().get("thinking", "{}"))
        score = float(output.get("score", 0.0))
        reasoning = output.get("reasoning", "No reasoning provided.")
        # print(f"Judge Reasoning:\n{output.get('reasoning', 'No reasoning provided.')}\nScore: {score}\n---")
        return max(0.0, min(1.0, score)), reasoning
    except Exception:
        return 0.0, "Judge failed to evaluate the code."

# =====================================================================
# Prompt Builders (DSL-Agnostic)
# =====================================================================
def build_zero_shot_prompt(intent: str, dsl_config: dict) -> str:
    dsl_name = dsl_config["name"]
    return f"Write {dsl_name} code for the following intent. Output ONLY the code. Do not use markdown formatting, code blocks, or explanations.\nIntent: {intent}\nCode:\n"

def build_one_shot_prompt(intent: str, dsl_config: dict, include_think: bool) -> str:
    dsl_name = dsl_config["name"]
    aliases_str = "\n".join([f"- '{k}': {v}" for k, v in dsl_config["aliases"].items()])
    
    prompt = (
        f"You are an expert {dsl_name} programmer.\n"
        "Strictly adhere to the following EBNF grammar rules:\n"
        f"```ebnf\n{dsl_config['ebnf'].strip()}\n```\n\n"
        f"OPERATORS:\n{aliases_str}\n\n"
        f"EXAMPLE:\nUser Intent: {dsl_config['example_intent']}\n"
    )
    
    if include_think:
        prompt += f"<think>\n{dsl_config['example_think']}\n</think>\n"
        
    prompt += f"```{dsl_name.lower()}\n{dsl_config['example_code']}\n```\n\nUser Intent: {intent}\n"
    return prompt

# =====================================================================
# Generation Methods (Updated for Outlines v0.1.0+)
# =====================================================================
def generate_zero_shot(model, intent: str, dsl_config: dict, compiler_fn: Callable) -> Tuple[str, dict]:
    prompt = build_zero_shot_prompt(intent, dsl_config)
    
    # Pass generation parameters directly
    output = model(prompt, max_new_tokens=150, temperature=TEMPERATURE, do_sample=True)
    if isinstance(output, list): output = output[0]
    
    code = output.replace(f"```{dsl_config['name'].lower()}", "").replace("```", "").strip()
    # print(f"Generated Code (Zero-Shot):\n{code}\n---")
    return code, {}

def generate_one_shot_no_gcd(model, intent: str, dsl_config: dict, compiler_fn: Callable) -> Tuple[str, dict]:
    prompt = build_one_shot_prompt(intent, dsl_config, include_think=True) + "<think>\n"
    
    output = model(prompt, max_new_tokens=3000, temperature=TEMPERATURE, do_sample=True)
    if isinstance(output, list): output = output[0]
    
    code_marker = f"```{dsl_config['name'].lower()}"
    new_text = output #[len(prompt):]
    
    if code_marker in new_text:
        code = new_text.split(code_marker)[1].split("```")[0].strip()
    else:
        code = new_text.split("</think>")[-1].strip()
        
    # print(f"Generated Code (One-Shot No GCD):\n{code}\n---")
    return code, {}

def generate_one_shot_only_gcd(model, intent: str, dsl_config: dict, compiler_fn: Callable) -> Tuple[str, dict]:
    code_marker = f"```{dsl_config['name'].lower()}\n"
    prompt = build_one_shot_prompt(intent, dsl_config, include_think=False) + code_marker
    
    output = model(prompt, CFG(dsl_config["ebnf"]), max_new_tokens=150, temperature=TEMPERATURE, do_sample=True)
    if isinstance(output, list): output = output[0]
    
    if code_marker in output:
        code = output.split(code_marker)[-1].strip()
    else:
        code = output.strip()
        
    return code, {}

def generate_dual_phase(model, intent: str, dsl_config: dict, compiler_fn: Callable) -> Tuple[str, dict]:
    """The Proposed Architecture: CoT + Optimistic Bypass + GCD Fallback"""
    prompt = build_one_shot_prompt(intent, dsl_config, include_think=True) + "<think>\n"
    code_marker = f"```{dsl_config['name'].lower()}"
    
    # Phase 1: Unconstrained Reasoning & Draft
    phase_1_out = model(prompt, max_new_tokens=400, temperature=TEMPERATURE, do_sample=True)
    if isinstance(phase_1_out, list): phase_1_out = phase_1_out[0]
    
    new_text = phase_1_out[len(prompt):]
    reasoning_text = new_text.split("</think>")[0].strip() if "</think>" in new_text else new_text.strip()
    
    # Try Optimistic Extraction
    draft_code = ""
    if code_marker in new_text:
        draft_code = new_text.split(code_marker)[1].split("```")[0].strip()
        
    # Check Fast Path
    if draft_code and compiler_fn(draft_code):
        return draft_code, {"fast_path_success": True}
        
    # Phase 2: GCD Fallback
    phase_2_prompt = prompt + reasoning_text + f"\n</think>\n{code_marker}\n"
    
    constrained_out = model(phase_2_prompt, CFG(dsl_config["ebnf"]), max_new_tokens=150, temperature=TEMPERATURE, do_sample=True)
    if isinstance(constrained_out, list): constrained_out = constrained_out[0]
    
    if f"{code_marker}\n" in constrained_out:
        final_code = constrained_out.split(f"{code_marker}\n")[-1].strip()
    else:
        final_code = constrained_out.strip()
        
    return final_code, {"fast_path_success": False}

# =====================================================================
# Main Evaluation Loop
# =====================================================================
def run_benchmark():
    
    # -----------------------------------------------------------------
    # INJECT DSL SPECIFICS HERE (MiniZinc Example)
    # -----------------------------------------------------------------
    def minizinc_compiler(code: str) -> tuple[bool, str]:
        """Compiler implementation for MiniZinc."""
        fd, temp_path = tempfile.mkstemp(suffix=".mzn")
        try:
            with os.fdopen(fd, 'w') as f:
                f.write(code)
            result = subprocess.run(["minizinc", "--model-check-only", temp_path], capture_output=True, timeout=3)
            if result.returncode != 0:
                return False, result.stderr.decode()
            return True, "Success! No type or semantic errors."
        except Exception as e:
            return False, f"Compiler execution failed: {e}"
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    MINIZINC_DSL_CONFIG = {
        "name": "MiniZinc",
        "ebnf": r"""
            ?start: model
            model: var_decls constraints solve output_opt
            var_decls: var_decl | var_decl var_decls
            var_decl: "var " type ": " IDENT ";" "\n" | "array[" int_lit ".." int_lit "] of var " type ": " IDENT ";" "\n"
            type: "int" | "bool" | int_lit ".." int_lit | "set of " int_lit ".." int_lit
            constraints: constraint | constraint constraints
            constraint: "constraint " expr ";" "\n"
            expr: logic_expr
            logic_expr: comp_expr | comp_expr " " logic_op " " logic_expr
            comp_expr: math_expr | math_expr " " comp_op " " math_expr | "sum(" IDENT ") " comp_op " " math_expr
            math_expr: term | term " " math_op " " math_expr
            term: IDENT | int_lit
            math_op: "+" | "-" | "*" | "/" | "mod"
            comp_op: "==" | ">" | "<" | "!=" | "<=" | ">=" | "in"
            logic_op: "\\/" | "/\\" | "->"
            solve: "solve satisfy;" "\n" | "solve maximize " IDENT ";" "\n" | "solve minimize " IDENT ";" "\n"
            output_opt: "output [show(" IDENT ")];" "\n" | ""
            IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/
            int_lit: /-?[0-9]+/
        """,
        "aliases": {
            "\\/": "Logical OR", "/\\": "Logical AND", "->": "Implication",
            "==": "Equality", "!=": "Inequality", "mod": "Modulo"
        },
        "example_intent": "Find an integer y exactly equal to 10.",
        "example_think": "1. Need integer 'y'.\n2. Constraint: y == 10.\n3. Solve satisfy.",
        "example_code": "var int: y;\nconstraint y == 10;\nsolve satisfy;"
    }
    # -----------------------------------------------------------------

    # 1. Load HF Model (Singleton)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {GENERATOR_MODEL_NAME} on {device}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        GENERATOR_MODEL_NAME, 
        device_map=device, 
        attn_implementation="eager" # Safe fallback for CPU/Older GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(GENERATOR_MODEL_NAME)
    
    if hasattr(outlines, "from_transformers"):
        model = outlines.from_transformers(hf_model, tokenizer)
    else:
        model = outlines.models.Transformers(hf_model, tokenizer)

    # 2. Load Dataset
    with open(BENCHMARK_FILE, "r") as f:
        dataset = json.load(f)

    results = {
        "Zero-Shot": {"pass": 0, "fail": 0},
        "One-Shot (No GCD)": {"pass": 0, "fail": 0},
        "One-Shot (GCD Only)": {"pass": 0, "fail": 0},
        "Dual-Phase (Proposed)": {"pass": 0, "fail": 0, "fast_path_count": 0}
    }
    
    detailed_log = []

    print(f"\nStarting pass@{K_SAMPLES} evaluation on {len(dataset)} prompts...\n")
    
    for i, item in enumerate(tqdm(dataset)):
        intent = item["nl"]
        golden_code = item["code"]
        
        prompt_log = {"id": i, "intent": intent, "evaluations": {}}

        methods = {
            "Zero-Shot": generate_zero_shot,
            "One-Shot (No GCD)": generate_one_shot_no_gcd,
            "One-Shot (GCD Only)": generate_one_shot_only_gcd,
            "Dual-Phase (Proposed)": generate_dual_phase
        }

        for method_name, gen_func in methods.items():
            method_passed = False
            sample_logs = []

            if method_name == "Dual-Phase (Proposed)":
                MINIZINC_DSL_CONFIG["name"] = "MaskedName"  # Masking for judge to prevent bias
            else:
                MINIZINC_DSL_CONFIG["name"] = "MiniZinc"
            
            for k in range(K_SAMPLES):
                # 1. Generate code and metadata
                try:
                    generated_code, metadata = gen_func(
                        model, intent, MINIZINC_DSL_CONFIG, minizinc_compiler
                    )
                except Exception as e:
                    generated_code, metadata = "", {"error": str(e)}

                # 2. Evaluate Compile & Semantic Alignment
                compiles, compiler_output = minizinc_compiler(generated_code)
                score = 0.0
                
                if compiles:
                    score, reasoning = judge_semantic_alignment(
                        MINIZINC_DSL_CONFIG["name"], intent, generated_code, golden_code
                    )
                    if score >= 0.85:
                        method_passed = True
                
                log_entry = {
                    "sample": k + 1,
                    "code": generated_code,
                    "compiles": compiles,
                    "compiler_output": compiler_output,
                    "judge_score": score,
                    "reasoning": reasoning if compiles else "Did not compile, skipping judge evaluation."
                }
                
                # Attach extra metadata (like fast_path_success for Dual-Phase)
                log_entry.update(metadata)
                sample_logs.append(log_entry)
                
                # If pass@k is satisfied, short-circuit
                if method_passed:
                    # Tally fast-path success tracking for analytics
                    if method_name == "Dual-Phase (Proposed)" and metadata.get("fast_path_success"):
                        results[method_name]["fast_path_count"] += 1
                    # break
                    
            if method_passed:
                results[method_name]["pass"] += 1
            else:
                results[method_name]["fail"] += 1
                
            prompt_log["evaluations"][method_name] = {"passed": method_passed, "samples": sample_logs}

        detailed_log.append(prompt_log)
        
        # Incremental Save
        with open(RESULTS_FILE, "w") as f:
            json.dump({"summary": results, "details": detailed_log}, f, indent=2)

    # --- Print Final Metrics ---
    print("\n" + "="*50)
    print(f"FINAL pass@{K_SAMPLES} ACCURACY ({len(dataset)} Prompts)")
    print("="*50)
    for method, metrics in results.items():
        total = metrics["pass"] + metrics["fail"]
        accuracy = (metrics["pass"] / total) * 100 if total > 0 else 0.0
        output_str = f"{method:>25}: {accuracy:.1f}% ({metrics['pass']}/{total})"
        
        if "fast_path_count" in metrics:
            fp_percentage = (metrics['fast_path_count'] / metrics['pass']) * 100 if metrics['pass'] > 0 else 0.0
            output_str += f" | Bypassed GCD successfully in {fp_percentage:.1f}% of passes"
            
        print(output_str)

if __name__ == "__main__":
    run_benchmark()