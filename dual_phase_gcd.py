import os
import torch
import outlines
from outlines.types import CFG
from transformers import AutoModelForCausalLM, AutoTokenizer
import subprocess
from minizinc_parser import parse_model, MINIZINC_GRAMMAR

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
        "IMPORTANT: You MUST close your reasoning with </think> BEFORE writing the code block.\n" 
        "Then, write the corresponding MiniZinc code block strictly adhering to the syntax.\n\n"
        f"{aliases_str}\n"
        f"{examples_str}"
        f"User Intent: {target_intent}\n"
    )
    
    return sys_prompt

from minizinc_parser import MINIZINC_ALIASES, minizinc_few_shot_examples

def build_dsl_agnostic_prompt(
    dsl_name: str,
    target_intent: str,
    ebnf_grammar: str,
    aliases: dict,
    examples: list
) -> str:
    """Dynamically builds a completely DSL-agnostic prompt using the strict EBNF grammar and examples."""
    
    # 1. Build the Semantic Cheat Sheet
    aliases_str = f"{dsl_name.upper()} OPERATOR CHEAT SHEET:\n"
    for operator, meaning in aliases.items():
        aliases_str += f"- '{operator}' : {meaning}\n"
        
    # 2. Build the Few-Shot / One-Shot Examples Block
    examples_str = "EXAMPLES:\n"
    for i, ex in enumerate(examples):
        examples_str += f"User Intent: {ex['nl']}\n"
        examples_str += f"<think>\nAnalyze intent: Identify components and trace them through the {dsl_name} EBNF grammar derivation rules.\n</think>\n"
        examples_str += f"```{dsl_name.lower()}\n{ex['code']}\n```\n\n"

    # 3. Assemble the final grammar-informed prompt
    sys_prompt = (
        f"You are an expert {dsl_name} programmer. You must translate the User Intent into valid {dsl_name} code.\n\n"
        f"You MUST strictly adhere to the following EBNF grammar. Do not use any syntax or keywords outside of these derivation rules:\n"
        f"```ebnf\n{ebnf_grammar.strip()}\n```\n\n"
        f"First, reason deeply about the required components, types, and logic. Enclose your reasoning in <think> and </think> tags.\n"
        "IMPORTANT: You MUST close your reasoning with </think> BEFORE writing the code block.\n"
        f"Then, write the corresponding {dsl_name} code block strictly adhering to the provided EBNF grammar.\n\n"
        f"{aliases_str}\n"
        f"{examples_str}"
        f"User Intent: {target_intent}\n"
    )
    
    return sys_prompt

# =====================================================================
# 3. Main Dual-Phase Architecture
# =====================================================================
def run_dual_phase_prototype():
    model_name = "Qwen/Qwen3-8B"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Loading {model_name} on {device}...")
    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map=device, 
        attn_implementation="eager" # Safe fallback for CPU/Older GPUs
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Initialize the modern Outlines model wrapper
    if hasattr(outlines, "from_transformers"):
        model = outlines.from_transformers(hf_model, tokenizer)
    else:
        model = outlines.models.Transformers(hf_model, tokenizer)
    
    # =========================================================
    # EXTERNAL DSL INJECTION DATA
    # =========================================================
    DSL_NAME = "MaskedLanguage"
    
    ALIASES = {
        "\\/": "Logical OR (either/or)",
        "/\\": "Logical AND (both)",
        "->": "Logical Implication (if/then)",
        "==": "Equality (exactly equal)",
        "!=": "Inequality (not equal)",
        "mod": "Modulo (remainder of division)"
    }
    
    EXAMPLES = [
        {
            "nl": "Find an integer y exactly equal to 10.",
            "code": "var int: y;\nconstraint y == 10;\nsolve satisfy;"
        },
        {
            "nl": "Find an array arr of 3 integers from 1 to 5. Constrain the sum of arr to equal 10.",
            "code": "array[1..3] of var 1..5: arr;\nconstraint sum(arr) == 10;\nsolve satisfy;\n"
        }
    ]
    
    # target_intent = "Declare two booleans a and c, constrain that either a or c is true, and satisfy."
    target_intent = "Create a model with an array arr of 4 integers from 1 to 3. The sum of arr must be less than 8."

    sys_prompt = build_dsl_agnostic_prompt(
        dsl_name=DSL_NAME,
        target_intent=target_intent,
        ebnf_grammar=MINIZINC_GRAMMAR,
        aliases=ALIASES,
        examples=EXAMPLES
    )
    
    code_marker = f"```{DSL_NAME.lower()}"
    
    print("\n" + "="*50)
    print(f"TARGET INTENT: {target_intent}")
    print("="*50)

    # ---------------------------------------------------------
    # PHASE 1: CoT Reasoning & Optimistic Draft
    # ---------------------------------------------------------
    print(f"\n[Phase 1] Generating Unconstrained Reasoning and {DSL_NAME} Draft...")
    
    phase_1_prompt = sys_prompt + "<think>\n"
    phase_1_output = model(phase_1_prompt, max_new_tokens=4000) 
    
    if isinstance(phase_1_output, list):
        phase_1_output = phase_1_output[0]
        
    new_text = phase_1_output # [len(phase_1_prompt):].strip()
    
    reasoning_text = ""
    draft_code = ""
    
    # --- PARSE REASONING AND DRAFT CODE DYNAMICALLY ---
    if "</think>" in new_text:
        parts = new_text.split("</think>")
        reasoning_text = parts[0].strip()
        post_think_text = parts[1].strip()
        
        if code_marker in post_think_text:
            draft_code = post_think_text.split(code_marker)[1].split("```")[0].strip()
        elif "```" in post_think_text:
            draft_code = post_think_text.split("```")[1].strip()
        else:
            draft_code = post_think_text.strip()
            
    elif code_marker in new_text:
        parts = new_text.split(code_marker)
        reasoning_text = parts[0].strip()
        draft_code = parts[1].split("```")[0].strip()
    else:
        reasoning_text = new_text.strip()
        
    reasoning_text = reasoning_text.replace("</think>", "").strip()
        
    print("\n--- Internal Thoughts ---")
    print(reasoning_text)
    print("-------------------------")

    # ---------------------------------------------------------
    # OPTIMISTIC EVALUATION (The Fast Path)
    # ---------------------------------------------------------
    if draft_code:
        print(f"\n[Optimistic Bypass] Model generated draft code. Evaluating...")
        try:
            # 1. Syntactic Gate (Using your specific parser logic)
            parse_model(draft_code)
            
            # 2. Semantic Compiler Gate
            is_valid, msg = check_compilation(draft_code)
            
            if is_valid:
                print("\n✅ FAST PATH SUCCESS! Draft code is perfectly valid.")
                print(f"\n--- Final {DSL_NAME} Code (Phase 1 Fast-Path) ---")
                print(draft_code)
                print("-----------------------------------------------")
                return 
            else:
                print(f"❌ Compiler rejected draft: {msg}")
                print("Falling back to Phase 2 Constraints...")
                
        except Exception as e:
            print(f"❌ Syntax parser rejected draft: {str(e)[:100]}...")
            print("Falling back to Phase 2 Constraints...")
    else:
        print("\n[Optimistic Bypass] No draft code detected. Proceeding to Phase 2...")

    # ---------------------------------------------------------
    # PHASE 2: Strict CFG Generation (Constrained Fallback)
    # ---------------------------------------------------------
    print(f"\n[Phase 2] Generating Constrained Code (CFG Logits Masking)...")
    
    # Dynamically format the Phase 2 prompt injection
    phase_2_prompt = phase_1_prompt + reasoning_text + f"\n</think>\n{code_marker}\n"
    
    constrained_code = model(phase_2_prompt, CFG(MINIZINC_GRAMMAR), max_new_tokens=150)
    
    if isinstance(constrained_code, list):
        constrained_code = constrained_code[0]
        
    if f"{code_marker}\n" in constrained_code:
        final_code = constrained_code.split(f"{code_marker}\n")[-1].strip()
    else:
        final_code = constrained_code.strip()
        
    # Generic fallback cleaner
    if not (final_code.startswith("var") or final_code.startswith("array")):
        final_code = final_code.split("</think>")[-1].replace(code_marker, "").replace("```", "").strip()

    print(f"\n--- Final Constrained {DSL_NAME} Code ---")
    print(final_code)
    print("---------------------------------------")

    # ---------------------------------------------------------
    # Verification
    # ---------------------------------------------------------
    print(f"\n[Verification] Running {DSL_NAME} Compiler Check...")
    is_valid, msg = check_compilation(final_code)
    print(f"Compiler Result: {msg}")

if __name__ == "__main__":
    run_dual_phase_prototype()