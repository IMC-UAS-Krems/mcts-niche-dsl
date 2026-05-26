import os
import json
import requests
import subprocess
from typing import Optional
from minizinc_parser import parse_model # Re-use your parser from the previous steps

# =====================================================================
# 1. Unified Evaluator
# =====================================================================
def evaluate_generated_code(code: str, target_prompt: str) -> dict:
    """Evaluates a baseline's output for Syntactic and Semantic Correctness."""
    
    # Clean up standard LLM markdown artifacts (very common in unconstrained baselines)
    clean_code = code.strip()
    if clean_code.startswith("```"):
        clean_code = "\n".join(clean_code.split("\n")[1:])
    if clean_code.endswith("```"):
        clean_code = "\n".join(clean_code.split("\n")[:-1])
    clean_code = clean_code.strip()

    result = {
        "raw_code": clean_code,
        "syntax_pass": False,
        "compile_pass": False
    }

    # 1. Syntactic Check (Lark Parser)
    try:
        parse_model(clean_code)
        result["syntax_pass"] = True
    except Exception as e:
        result["error"] = f"Syntax Error: {str(e)[:100]}..."
        return result

    # 2. Semantic/Type Check (MiniZinc CLI)
    try:
        with open("temp_baseline.mzn", "w") as f:
            f.write(clean_code)
        
        proc = subprocess.run(
            ["minizinc", "--model-check-only", "temp_baseline.mzn"],
            capture_output=True, text=True, timeout=5
        )
        if proc.returncode == 0:
            result["compile_pass"] = True
        else:
            result["error"] = f"Type/Semantic Error: {proc.stderr.strip()[:100]}..."
    except Exception as e:
        result["error"] = f"Compiler CLI Error: {e}"

    return result


# =====================================================================
# Baseline 1: Zero-Shot LLM
# =====================================================================
def baseline_1_zero_shot(prompt: str, model: str = "qwen2.5-coder:1.5b") -> str:
    """
    Standard autoregressive generation. No grammar, no examples.
    Expected outcome: Often hallucinates syntax, includes markdown, or uses 
    standard Python/C++ constructs instead of MiniZinc.
    """
    api_url = "http://localhost:11434/api/generate"
    
    sys_instruction = (
        "You are an expert MiniZinc programmer. "
        "Write MiniZinc code to fulfill the User Intent. "
        "Output ONLY the raw MiniZinc code. Do not use markdown formatting, code blocks, or explanations."
    )
    
    payload = {
        "model": model,
        "prompt": f"{sys_instruction}\n\nUser Intent: {prompt}",
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    print("[Baseline 1] Running Zero-Shot...")
    response = requests.post(api_url, json=payload)
    return response.json().get("response", "")


# =====================================================================
# Baseline 2: One-Shot Grammar-Informed LLM
# =====================================================================
def baseline_2_one_shot_grammar(prompt: str, model: str = "qwen2.5-coder:1.5b") -> str:
    """
    Prompt engineering approach. The LLM is given a description of the EBNF 
    and one golden example.
    Expected outcome: Better syntax, but frequently makes semantic type errors 
    or slightly violates the strict grammar rules.
    """
    api_url = "http://localhost:11434/api/generate"
    
    sys_instruction = (
        "You are an expert MiniZinc programmer. You must strictly follow this simplified grammar structure:\n"
        "1. <VarDecls> : e.g., 'var int: x;'\n"
        "2. <Constraints> : e.g., 'constraint x > 5;'\n"
        "3. <Solve> : e.g., 'solve satisfy;'\n\n"
        "EXAMPLE:\n"
        "Intent: Declare a boolean b and satisfy.\n"
        "Code:\nvar bool: b;\nsolve satisfy;\n\n"
        "Write MiniZinc code to fulfill the User Intent. "
        "Output ONLY the raw MiniZinc code. Do not use markdown."
    )
    
    payload = {
        "model": model,
        "prompt": f"{sys_instruction}\n\nUser Intent: {prompt}",
        "stream": False,
        "options": {"temperature": 0.2}
    }
    
    print("[Baseline 2] Running One-Shot Grammar-Informed...")
    response = requests.post(api_url, json=payload)
    return response.json().get("response", "")


# =====================================================================
# Baseline 3: Grammar-Constrained Decoding (GCD) via Outlines
# =====================================================================
def baseline_3_grammar_constrained(prompt: str) -> str:
    """
    Uses outlines to compile the EBNF grammar into a Finite State Machine.
    The logits of the LLM are masked at every token step.
    Expected outcome: 100% Syntactic Pass Rate. However, because it lacks lookahead, 
    it often falls into "Semantic Dead-Ends" (e.g., forcing a boolean into an integer constraint).
    """
    print("[Baseline 3] Loading Transformers and Outlines (this may take a moment)...")
    
    try:
        import outlines
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        return f"Import Error: {e}. Please run `pip install outlines transformers torch`"

    model_name = "Qwen/Qwen2.5-Coder-1.5B"
    try:
        print(f"[Baseline 3] Downloading/Loading {model_name} from HuggingFace...")
        # Load the model natively into HuggingFace first
        hf_model = AutoModelForCausalLM.from_pretrained(model_name, device_map="cpu")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Use the latest Outlines API to wrap the model
        if hasattr(outlines, "from_transformers"):
            model = outlines.from_transformers(hf_model, tokenizer)
        else:
            return "Error: Please update outlines to the latest version (`pip install -U outlines`)."
            
    except Exception as e:
        return f"Error loading model: {e}"

    # We provide the Lark EBNF format (from your original MINIZINC_GRAMMAR)
    ebnf_grammar = r"""
        ?start: model
        model: var_decl constraint solve
        
        var_decl: "var " type ": " IDENT ";" "\n"
        type: "int" | "bool"
        
        constraint: "constraint " IDENT " " op " " int_lit ";" "\n"
        op: "==" | ">" | "<" | "!="
        
        solve: "solve satisfy;" "\n" | "solve maximize " IDENT ";" "\n"
        
        IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/
        int_lit: /[0-9]+/
    """
    
    print("[Baseline 3] Compiling FSM from Grammar...")
    try:
        # Import the CFG type wrapper from the new API
        from outlines.types import CFG
        
        prompt_text = f"Write a MiniZinc model to fulfill this intent: {prompt}\nCode:\n"
        
        # NEW API: You simply call the model with the prompt and the CFG type wrapper
        sequence = model(prompt_text, CFG(ebnf_grammar))# , max_tokens=100)
        
        # Clean up output parsing
        if isinstance(sequence, list): sequence = sequence[0]
        if "Code:\n" in sequence: sequence = sequence.split("Code:\n")[1]
            
        return sequence
    except Exception as e:
        import traceback
        return f"Outlines Generation Error: {e}\n{traceback.format_exc()}"

# =====================================================================
# Execution & Comparison
# =====================================================================
if __name__ == "__main__":
    test_prompt = "Write a MiniZinc model to find an integer a that is exactly equal to 10."
    ollama_model_name = os.getenv("OLLAMA_MODEL", "qwen2.5-coder:1.5b")
    
    print(f"--- EVALUATING BASELINES ---")
    print(f"Target Prompt: '{test_prompt}'\n")

    # 1. Zero-Shot
    b1_code = baseline_1_zero_shot(test_prompt, ollama_model_name)
    b1_eval = evaluate_generated_code(b1_code, test_prompt)
    
    # 2. One-Shot
    b2_code = baseline_2_one_shot_grammar(test_prompt, ollama_model_name)
    b2_eval = evaluate_generated_code(b2_code, test_prompt)
    
    # 3. Constrained Decoding
    # Note: If your machine lacks RAM to load transformers alongside Ollama, 
    # you can comment this block out.
    b3_code = baseline_3_grammar_constrained(test_prompt)
    b3_eval = evaluate_generated_code(b3_code, test_prompt)

    # --- PRINT RESULTS ---
    print("\n" + "="*50)
    print("RESULTS COMPARISON")
    print("="*50)

    print("\n[Baseline 1: Zero-Shot]")
    print(f"Code:\n{b1_eval['raw_code']}")
    print(f"Syntax Pass:  {b1_eval['syntax_pass']}")
    print(f"Compile Pass: {b1_eval['compile_pass']}")
    if "error" in b1_eval: print(f"Error: {b1_eval['error']}")

    print("\n[Baseline 2: One-Shot Grammar-Informed]")
    print(f"Code:\n{b2_eval['raw_code']}")
    print(f"Syntax Pass:  {b2_eval['syntax_pass']}")
    print(f"Compile Pass: {b2_eval['compile_pass']}")
    if "error" in b2_eval: print(f"Error: {b2_eval['error']}")

    print("\n[Baseline 3: Grammar-Constrained Decoding]")
    print(f"Code:\n{b3_eval['raw_code']}")
    print(f"Syntax Pass:  {b3_eval['syntax_pass']}")
    print(f"Compile Pass: {b3_eval['compile_pass']}")
    if "error" in b3_eval: print(f"Error: {b3_eval['error']}")
    
    print("\n" + "="*50)
    print("MCTS Performance (Hypothetical from your architecture):")
    print("Syntax Pass:  True (Guaranteed by Action Space Masking)")
    print("Compile Pass: True (Guaranteed by Dead-End Compiler Checks)")
    print("Semantic:     Driven by Search-based Lookahead and Rollouts")