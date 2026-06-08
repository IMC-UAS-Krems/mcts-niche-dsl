import os
import torch
import outlines
from outlines.types import CFG
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess

# =====================================================================
# 1. MiniZinc EBNF Grammar (Strict Constraint for Phase 2)
# =====================================================================
MINIZINC_EBNF = r"""
    ?start: model
    model: var_decls constraints solve
    
    var_decls: var_decl | var_decl var_decls
    var_decl: "var " type ": " IDENT ";" "\n"
    type: "int" | "bool" | int_lit ".." int_lit
    
    constraints: constraint | constraint constraints
    constraint: "constraint " expr ";" "\n"
    
    expr: base_bool | base_bool " " logic_op " " base_bool
    base_bool: math_expr | math_expr " " comp_op " " math_expr
    math_expr: term | term " " math_op " " term
    term: IDENT | int_lit
    
    math_op: "+" | "-" | "*" | "mod"
    comp_op: "==" | ">" | "<" | "!=" | "<=" | ">="
    logic_op: "\/" | "/\\" | "->"
    
    solve: "solve satisfy;" "\n" | "solve maximize " IDENT ";" "\n" | "solve minimize " IDENT ";" "\n"
    
    IDENT: /[a-zA-Z_][a-zA-Z0-9_]*/
    int_lit: /-?[0-9]+/
"""

# =====================================================================
# 2. Evaluation Helper
# =====================================================================
def check_compilation(code: str) -> tuple[bool, str]:
    """Verifies semantic type safety using the MiniZinc compiler."""
    try:
        with open("temp_test.mzn", "w") as f:
            f.write(code)
        result = subprocess.run(
            ["minizinc", "--model-check-only", "temp_test.mzn"],
            capture_output=True, text=True, timeout=3
        )
        if result.returncode == 0:
            return True, "Success! No type or semantic errors."
        return False, result.stderr.strip()
    except Exception as e:
        return False, f"Compiler execution failed: {e}"
    finally:
        if os.path.exists("temp_test.mzn"):
            os.remove("temp_test.mzn")


def build_dual_phase_prompt(target_intent: str, aliases: dict, examples: list) -> str:
    """Dynamically builds the system prompt injecting aliases and few-shot examples."""
    
    # 1. Build the Semantic Cheat Sheet
    aliases_str = "MINIZINC OPERATOR CHEAT SHEET:\n"
    for operator, meaning in aliases.items():
        aliases_str += f"- '{operator}' : {meaning}\n"
        
    # 2. Build the Few-Shot Examples Block
    examples_str = "EXAMPLES:\n"
    for i, ex in enumerate(examples):
        examples_str += f"User Intent: {ex['nl']}\n"
        # Provide a brief, synthetic thought process to prime the CoT engine
        examples_str += f"<think>\nAnalyze intent: Identify variables and constraints. Map to MiniZinc syntax.\n</think>\n"
        examples_str += f"```minizinc\n{ex['code']}\n```\n\n"

    # 3. Assemble the final prompt
    sys_prompt = (
        "You are an expert MiniZinc programmer. You must translate the User Intent into valid MiniZinc code.\n"
        "First, reason deeply about the required variables, types, and logic. Enclose your reasoning in <think> and </think> tags.\n"
        "Then, write the corresponding MiniZinc code block strictly adhering to the syntax.\n\n"
        f"{aliases_str}\n"
        f"{examples_str}"
        f"User Intent: {target_intent}\n"
    )
    
    return sys_prompt

from minizinc_parser import MINIZINC_ALIASES, minizinc_few_shot_examples

# =====================================================================
# 3. Main Dual-Phase Architecture
# =====================================================================
def run_dual_phase_prototype():
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name} on {device}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="cpu", 
        attn_implementation="eager" # Safe fallback for CPU/Older GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize the modern Outlines model wrapper
    if hasattr(outlines, "from_transformers"):
        model = outlines.from_transformers(hf_model, tokenizer)
    else:
        model = outlines.models.Transformers(hf_model, tokenizer)
    
    # ---------------------------------------------------------
    # The Prompt Setup (Dynamic Injection)
    # ---------------------------------------------------------
    target_intent = "Declare two booleans a and c, constrain that either a or c is true, and satisfy."
    
    # Inject our external configurations
    sys_prompt = build_dual_phase_prompt(
        target_intent=target_intent,
        aliases=MINIZINC_ALIASES,
        examples=minizinc_few_shot_examples
    )
    
    print("\n" + "="*50)
    print(f"TARGET INTENT: {target_intent}")
    print("="*50)

    # ---------------------------------------------------------
    # PHASE 1: CoT Reasoning (Unconstrained)
    # ---------------------------------------------------------
    print("\n[Phase 1] Generating Unconstrained Reasoning (DeepSeek-R1 <think> block)...")
    
    phase_1_prompt = sys_prompt + "<think>\n"
    print("\n--- Phase 1 Prompt ---")
    print(phase_1_prompt)
    print("----------------------")
    
    phase_1_output = model(phase_1_prompt, max_new_tokens=3000)
    print("\n--- Phase 1 Raw Output ---")
    print(phase_1_output)
    print("-------------------------")
    
    if isinstance(phase_1_output, list):
        phase_1_output = phase_1_output[0]
        
    new_text = phase_1_output[len(phase_1_prompt):].strip()
    reasoning_text = new_text.split("</think>")[0].strip()
        
    print("\n--- DeepSeek-R1 Internal Thoughts ---")
    print(reasoning_text)
    print("-------------------------------------")

    # ---------------------------------------------------------
    # PHASE 2: Strict CFG Generation (Constrained)
    # ---------------------------------------------------------
    print("\n[Phase 2] Generating Constrained Code (CFG Logits Masking)...")
    
    phase_2_prompt = phase_1_prompt + reasoning_text + "\n</think>\n```minizinc\n"
    
    constrained_code = model(phase_2_prompt, CFG(MINIZINC_EBNF), max_new_tokens=1500)
    
    if isinstance(constrained_code, list):
        constrained_code = constrained_code[0]
        
    if "```minizinc\n" in constrained_code:
        final_code = constrained_code.split("```minizinc\n")[-1].strip()
    else:
        final_code = constrained_code.strip()
        
    if not (final_code.startswith("var") or final_code.startswith("array")):
        final_code = final_code.split("</think>")[-1].replace("```minizinc", "").replace("```", "").strip()

    print("\n--- Final Constrained MiniZinc Code ---")
    print(final_code)
    print("---------------------------------------")

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    print("\n[Verification] Running MiniZinc Compiler Check...")
    is_valid, msg = check_compilation(final_code)
    print(f"Compiler Result: {msg}")

if __name__ == "__main__":
    run_dual_phase_prototype()