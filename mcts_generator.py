import math
import json
import requests
from typing import List, Dict, Tuple, Any, Optional
from minizinc_parser import parse_model, ast_to_json_serializable, minizinc_few_shot_examples
import os
from dotenv import load_dotenv
import subprocess
import random


load_dotenv()

# =====================================================================
# 1. Core MCTS Implementation
# =====================================================================

class MCTSNode:
    def __init__(self, state: Tuple[str, ...], prior_prob: float, parent: Optional['MCTSNode'] = None, action_taken: Tuple[str, ...] = None):
        self.state = state
        self.prior_prob = prior_prob
        self.parent = parent
        self.action_taken = action_taken
        
        self.children: Dict[Tuple[str, ...], MCTSNode] = {}
        self.visit_count: int = 0
        self.total_value: float = 0.0

    @property
    def q_value(self) -> float:
        if self.visit_count == 0: return 0.0
        return self.total_value / self.visit_count

    def is_expanded(self) -> bool:
        return len(self.children) > 0

    def get_best_child(self, c_puct: float) -> Tuple[Tuple[str, ...], 'MCTSNode']:
        best_score = -float('inf')
        best_action = None
        best_child = None

        for action, child in self.children.items():
            u_value = c_puct * child.prior_prob * math.sqrt(self.visit_count) / (1 + child.visit_count)
            puct_score = child.q_value + u_value

            if puct_score > best_score:
                best_score = puct_score
                best_action = action
                best_child = child

        return best_action, best_child

    def expand(self, action_probs: Dict[Tuple[str, ...], float], env: 'MiniZincEnvironment'):
        for action, prob in action_probs.items():
            if action not in self.children:
                next_state = env.apply_action(self.state, action)
                self.children[action] = MCTSNode(
                    state=next_state, prior_prob=prob, parent=self, action_taken=action
                )

    def backpropagate(self, value: float):
        self.visit_count += 1
        self.total_value += value
        if self.parent:
            self.parent.backpropagate(value)

class NeurosymbolicMCTS:
    def __init__(self, env: 'MiniZincEnvironment', llm_policy: 'OllamaLLMHeuristic', c_puct: float = 1.5):
        self.env = env
        self.llm = llm_policy
        self.c_puct = c_puct

    def fast_rollout(self, state: Tuple[str, ...]) -> float:
        """
        Fast-forwards from a partial AST to a terminal AST by making random valid choices.
        This allows the simulation to reach the compiler check without wasting LLM compute.
        """
        current_state = state
        depth = 0
        max_depth = 40 # Safety net to prevent infinite loops in the grammar
        
        while not self.env.is_terminal(current_state) and depth < max_depth:
            valid_actions = self.env.get_valid_actions(current_state)
            if not valid_actions:
                break
            
            # Pick a random valid grammar expansion
            action = random.choice(valid_actions)
            current_state = self.env.apply_action(current_state, action)
            depth += 1
            
        # If the rollout successfully generated a terminal string, RUN THE COMPILER CHECK
        if self.env.is_terminal(current_state):
            return self.env.compute_reward(current_state)
        else:
            return 0.0 # Failed to reach a terminal state within depth limit
    
    def fast_safe_rollout(self, state: Tuple[str, ...]) -> tuple[bool, str]:
        max_retries = 5
        previous_errors = [] # Stores tuples of (failed_code, compiler_error)

        for attempt in range(max_retries):
            current_state = state
            depth = 0
            max_depth = 40 
            
            while not self.env.is_terminal(current_state) and depth < max_depth:
                valid_actions = self.env.get_valid_actions(current_state)
                if not valid_actions: break
                
                idx = self.env._get_leftmost_nt(current_state)
                current_nt = current_state[idx] if idx != -1 else None

                # If we are at the root of the rollout and have failed previously, 
                # ask the LLM to adjust its priorities based on the compiler error!
                if previous_errors:
                    action_probs = self.llm.predict_with_feedback(current_state, valid_actions, previous_errors)
                else:
                    action_probs, _ = self.llm.predict_and_evaluate(current_state, valid_actions)
                
                # Combine Semantic Probs, Anti-Recursion Penalty, and Retry Noise
                scores = {}
                for action in valid_actions:
                    base_score = action_probs.get(action, 0.0)
                    
                    # 1. Anti-Recursion Penalty
                    if current_nt and current_nt in action:
                        base_score -= 100.0 
                        
                    # 2. Tie-Breaking Noise (Only active during retries to force exploration)
                    noise = random.uniform(0.0, 0.05) if attempt > 0 else 0.0
                    
                    scores[action] = base_score + noise
                    
                best_action = max(scores, key=scores.get)
                current_state = self.env.apply_action(current_state, best_action)
                depth += 1
                
            # Evaluate the completed stub
            if self.env.is_terminal(current_state):
                code = "".join(current_state)
                # Assumes check_compilation_with_feedback returns (bool, error_msg_str)
                # print(f"  [Rollout Attempt {attempt+1}] Generated Code:\n{code}")
                is_valid, error_msg = self.env.check_compilation_with_feedback(code)
                # print(f"  [Rollout Attempt {attempt+1}] Compilation Feedback: {'Valid' if is_valid else 'Invalid'}; Error: {error_msg}")
                
                if is_valid:
                    return True, ""
                else:
                    # Save the error to inform the next retry loop
                    previous_errors.append((code, error_msg))
            else:
                previous_errors.append(("".join(current_state), "Rollout depth exceeded. Left unresolved non-terminals."))
        
        # If all retries failed, return False and the last error message
        return False, previous_errors[-1][1] if previous_errors else "Unknown Rollout Error"

    import random

    def fast_stochastic_rollout(self, state: Tuple[str, ...], max_trials: int = 15) -> bool:
        """
        Attempts to find a compiling stub by rapidly generating pseudo-random completions.
        Bypasses the LLM entirely to maximize MCTS simulation throughput.
        """
        for attempt in range(max_trials):
            current_state = state
            depth = 0
            max_depth = 40 
            
            while not self.env.is_terminal(current_state) and depth < max_depth:
                valid_actions = self.env.get_valid_actions(current_state)
                if not valid_actions: 
                    break
                
                idx = self.env._get_leftmost_nt(current_state)
                current_nt = current_state[idx] if idx != -1 else None
                
                # --- ANTI-RECURSION FILTER ---
                # Strictly filter out any action that loops back on the current non-terminal
                safe_actions = [a for a in valid_actions if current_nt not in a]
                
                # Fallback just in case the grammar design forces a recursive step
                if not safe_actions:
                    safe_actions = valid_actions
                
                # --- STOCHASTIC SELECTION ---
                # Randomly pick a safe action. This ensures every trial generates a different stub.
                # Because we removed recursion, the stub will naturally be minimal/short.
                action = random.choice(safe_actions)
                
                current_state = self.env.apply_action(current_state, action)
                depth += 1
                
            # Evaluate the completed stub
            if self.env.is_terminal(current_state):
                code = "".join(current_state)
                # Use the fast compiler check (no LLM overhead)
                is_valid, _ = self.env.check_compilation_with_feedback(code)
                
                if is_valid:
                    return True # Success! We proved this MCTS branch is semantically viable.
                    
        # If we exhausted all trials and none compiled, it's highly likely a dead-end
        return False    

    def search(self, initial_state: Tuple[str, ...], num_simulations: int = 50, rollout_weight: float = 0.5) -> Tuple[str, ...]:
        root = MCTSNode(state=initial_state, prior_prob=1.0)

        terminal_reached = False
        for _ in range(num_simulations):
            node = root
            # 1. Selection
            while node.is_expanded() and not self.env.is_terminal(node.state):
                action, node = node.get_best_child(self.c_puct)

            # print(f"[DEBUG SEARCH] Node Selected: {node.state}")
            # 2. Evaluation & Expansion
            if not self.env.is_terminal(node.state):
                valid_actions = self.env.get_valid_actions(node.state)
                if len(valid_actions) == 1:
                    # If there's only one valid action, skip the LLM and directly expand
                    action_probs, llm_value = {valid_actions[0]: 1.0}, 1.0
                else:
                    # print(f"[DEBUG SEARCH] Valid actions: {valid_actions}")
                    action_probs, llm_value = self.llm.predict_and_evaluate(node.state, valid_actions)
                    # print(f"[DEBUG SEARCH] Action probabilities: {action_probs}, State value: {value}")
                
                node.expand(action_probs, self.env)

                # Execute the fast stochastic stub trials
                is_viable = self.fast_stochastic_rollout(node.state, max_trials=50)
                
                if is_viable:
                    final_value = llm_value # Trust the LLM intent heuristic
                    # print(f"  [Rollout Success] Found a viable stub. LLM State Value: {llm_value}")
                else:
                    final_value = 0.0 # Branch is a semantic dead-end
                    # print(f"  [Rollout Failed] No valid compilation found.")
            else:
                # 3. Terminal Reward
                base_reward = self.env.compute_reward(node.state)
                
                # Penalize verbosity: subtract a tiny fraction based on AST length
                # e.g., if a state takes 30 steps, penalty is 0.03. If 50 steps, 0.05.
                length_penalty = len(node.state) * 0.001 
                
                final_value = max(0.0, base_reward - length_penalty)
                terminal_reached = True

            # 4. Backpropagation
            node.backpropagate(final_value)

        if not terminal_reached:
            print("[MCTS Warning] No terminal state reached during simulations. Final selection may be suboptimal.")

        # Proportional Action Sampling based on Visit Counts
        actions = list(root.children.keys())
        visit_counts = [child.visit_count for child in root.children.values()]
        print(f"[MCTS Search Completed] Root visit count: {root.visit_count}, Action visit counts: {visit_counts}")
        print(f"[MCTS Search Completed] Q-values: {[child.q_value for child in root.children.values()]}")
        print(f"[MCTS Search Completed] Actions: {actions}")
        
        # Calculate probabilities proportional to visit counts
        total_visits = sum(visit_counts)
        if total_visits > 0:
            probabilities = [v / total_visits for v in visit_counts]
            # Set probabilities to zero for any action that leads to a known dead-end (zero reward) to prevent selection)
            for i, action in enumerate(actions):
                child = root.children[action]
                if child.visit_count > 0 and child.q_value == 0.0:
                    probabilities[i] = 0.0
            # Ensure probabilities does not contain all zeros (which would cause random.choices to fail)
            if sum(probabilities) == 0.0:
                probabilities = [1.0 / len(actions) for _ in actions] # Fallback to uniform if all are zero
            # Probabilistically sample the next action
            best_action = random.choices(actions, weights=probabilities, k=1)[0]
        else:
            best_action = random.choice(actions)
            
        best_child = root.children[best_action]
        
        # Return both the action AND its average reward (Q-value) for dead-end detection
        print(f"[MCTS] Best action selected: {best_action} with Q-value: {best_child.q_value}")
        return best_action, best_child.q_value

    def generate_code(self, initial_state: Tuple[str, ...], max_steps: int = 40, num_simulations: int = 50) -> str:
        current_state = initial_state
        step = 0
        
        while not self.env.is_terminal(current_state) and step < max_steps:
            print(f"\n[Generation Step {step}] Current AST: {current_state}")
            best_action, expected_reward = self.search(current_state, num_simulations)
            
            # DEAD-END DETECTION
            if expected_reward == 0.0 and step > 0:
                print(f"\n[FATAL] MCTS realized all forward paths from this state result in compilation errors.")
                print(f"Trapped at state: {''.join(current_state)}")
                print("Halting generation to prevent hallucinating broken code.")
                break
                
            current_state = self.env.apply_action(current_state, best_action)
            step += 1
            
        return "".join(current_state)


# =====================================================================
# 2. MiniZinc Grammar Environment
# =====================================================================
class MiniZincEnvironment:
    """Handles the symbolic derivation of MiniZinc Code and evaluates it via AST matching."""
    
    def __init__(self, target_prompt: str, llm_judge: 'OllamaLLMHeuristic', extracted_entities: dict = None):
        self.target_prompt = target_prompt
        self.llm_judge = llm_judge
        
        # 1. Parse extracted entities
        extracted_entities = extracted_entities or {}
        idents = extracted_entities.get("identifiers", [])
        int_lits = extracted_entities.get("integer_literals", [])
        
        # 2. Provide safe fallbacks if the LLM extraction failed or found nothing
        if not idents: idents = ["x", "y", "b", "arr", "s", "z", "nums", "a", "c"]
        if not int_lits: int_lits = ["0", "1", "2", "3", "4", "5", "6", "10"]
            
        # Ensure all elements are strings for the grammar
        idents = [str(i) for i in idents]
        int_lits = [str(val) for val in int_lits]
        self.extracted_idents = idents  # Store for heuristic use in get_valid_actions
        
        print(f"[Environment] Bound Identifiers: {idents}")
        print(f"[Environment] Bound Literals: {int_lits}")

        # 3. Dynamically construct the grammar
        self.grammar = {
            # Phased structure now includes an optional Output phase
            "<Model>": [
                ["<VarDecls>", "<Constraints>", "<Solve>", "<OutputOpt>"]
            ],
            
            # --- Variables Phase ---
            "<VarDecls>": [
                ["<VarDecl>", "<VarDecls>"], # Recursive
                ["<VarDecl>"]                # Base case
            ],
            "<VarDecl>": [
                ["var ", "<Type>", ": ", "<Ident>", ";\n"],
                ["array[", "<IntLit>", "..", "<IntLit>", "] of var ", "<Type>", ": ", "<Ident>", ";\n"]
            ],
            "<Type>": [
                ["int"], 
                ["bool"], 
                ["<IntLit>", "..", "<IntLit>"],             # e.g., 0..10
                ["set of ", "<IntLit>", "..", "<IntLit>"]   # e.g., set of 1..5
            ],
            
            # --- Constraints Phase ---
            "<Constraints>": [
                ["<Constraint>", "<Constraints>"], # Recursive
                ["<Constraint>"]                   # Base case
            ],
            "<Constraint>": [
                ["constraint ", "<Expr>", ";\n"]
            ],
            
            # --- Stratified Expressions (Bounded Depth, No Infinite Recursion) ---
            
            # Level 1: Logical Combinations (e.g., A \/ B, A -> B)
            "<Expr>": [
                ["<BaseBool>"],
                ["<BaseBool>", " ", "<LogicOp>", " ", "<BaseBool>"]
            ],
            
            # Level 2: Boolean Evaluations / Comparisons (e.g., X > Y, sum(arr) == 3, or just a boolean 'b')
            "<BaseBool>": [
                ["<MathExpr>"],                                          # For pure boolean variables (e.g., 'b')
                ["<MathExpr>", " ", "<CompOp>", " ", "<MathExpr>"],      # e.g., x + y <= 10
                ["sum(", "<Ident>", ")", " ", "<CompOp>", " ", "<MathExpr>"] # e.g., sum(arr) == 3
            ],
            
            # Level 3: Arithmetic Operations (e.g., X + Y)
            "<MathExpr>": [
                ["<Term>"],
                ["<Term>", " ", "<MathOp>", " ", "<Term>"]
            ],
            
            # Level 4: Atoms
            "<Term>": [
                ["<Ident>"], 
                ["<IntLit>"]
            ],

            # Operators
            "<MathOp>":  [["+"], ["-"], ["*"], ["/"], ["mod"]],
            "<CompOp>":  [[">"], ["<"], ["=="], ["!="], ["<="], [">="], ["in"]],
            "<LogicOp>": [["->"], ["\\/"], ["/\\"]],
            
            # --- Solve Phase ---
            "<Solve>": [
                ["solve satisfy;\n"], 
                ["solve maximize ", "<Ident>", ";\n"],
                ["solve minimize ", "<Ident>", ";\n"]
            ],
            
            # --- Output Phase (Optional) ---
            "<OutputOpt>": [
                ["output [show(", "<Ident>", ")];\n"], 
                [""]  
            ],
            
            
            # --- Pruned Terminal Nodes ---
            "<Ident>": [[str(i)] for i in idents],
            "<IntLit>": [[str(val)] for val in int_lits]
        }


    def _get_leftmost_nt(self, state: tuple) -> int:
        for i, symbol in enumerate(state):
            if symbol.startswith("<") and symbol.endswith(">"):
                return i
        return -1

    def get_valid_actions(self, state: tuple) -> list:
        idx = self._get_leftmost_nt(state)
        if idx == -1: return []
        
        nt = state[idx]
        
        # HEURISTIC 1: Bound Variable Declarations
        if nt == "<VarDecls>":
            var_count = sum(1 for s in state if s == "<VarDecl>")
            # Assuming self.extracted_idents is saved during __init__
            max_vars = max(len(getattr(self, 'extracted_idents', [])), 3) 
            if var_count >= max_vars:
                return [tuple(["<VarDecl>"])] # Force base case
                
        # HEURISTIC 2: Bound Constraints
        if nt == "<Constraints>":
            constraint_count = sum(1 for s in state if s == "<Constraint>")
            # Limit to 4 constraints to keep MCTS search horizon manageable
            if constraint_count >= 4:
                return [tuple(["<Constraint>"])] # Force base case
                
        return [tuple(prod) for prod in self.grammar[nt]]

    def apply_action(self, state: tuple, action: tuple) -> tuple:
        idx = self._get_leftmost_nt(state)
        return state[:idx] + action + state[idx+1:]

    def is_terminal(self, state: tuple) -> bool:
        return self._get_leftmost_nt(state) == -1

    def check_compilation_only(self, code: str) -> bool:
        """Runs purely the syntactic (Lark) and semantic (MiniZinc CLI) checks."""
        try:
            # 1. Syntax
            parse_model(code)
            
            # 2. Semantics (Type checking)
            with open("temp_stub.mzn", "w") as f:
                f.write(code)
            import subprocess
            result = subprocess.run(
                ["minizinc", "--model-check-only", "temp_stub.mzn"],
                capture_output=True, text=True, timeout=2
            )
            print(f"[Compilation Check] Return code: {result.returncode}, Stderr: {result.stderr.strip()}")
            return result.returncode == 0
        except Exception as e:
            print(f"  [Compilation Check Failed] Error: {e}")
            return False

    def check_compilation_with_feedback(self, code: str) -> tuple[bool, str]:
        try:
            from minizinc_parser import parse_model
            parse_model(code)
            with open("temp_stub.mzn", "w") as f:
                f.write(code)
            import subprocess
            result = subprocess.run(
                ["minizinc", "--model-check-only", "temp_stub.mzn"],
                capture_output=True, text=True, timeout=2
            )
            if result.returncode == 0:
                return True, ""
            return False, result.stderr.strip()
        except Exception as e:
            return False, str(e)

    def compute_reward(self, state: tuple) -> float:
        """
        First guarantees syntactic validity using Lark.
        If valid, uses the LLM to judge semantic correctness against the prompt.
        """
        code = "".join(state)
        
        # 1. Syntactic Gatekeeper
        try:
            ast = parse_model(code)
        except Exception:
            # Code is syntactically invalid - zero reward
            return 0.0
            
        # We write the code to a temporary file and run `minizinc --model-check-only`
        # This instantly catches "type error: bool compared to int"
        try:
            with open("temp_eval.mzn", "w") as f:
                f.write(code)
                
            # Run MiniZinc in compile-only/check mode. 
            # (Assumes 'minizinc' is installed and in your system PATH)
            result = subprocess.run(
                ["minizinc", "--model-check-only", "temp_eval.mzn"],
                capture_output=True,
                text=True,
                timeout=2 # Prevent infinite hangs
            )
            
            if result.returncode != 0:
                # The MiniZinc compiler found a type error or semantic issue!
                # print(f"[Semantic Error Caught]: {result.stderr}")
                return 0.0
                
        except Exception as e:
            # If the subprocess fails for environmental reasons, fallback to 0.0
            # print(f"[Subprocess Error]: {e}")
            return 0.0
        
        # 2. Semantic Evaluation via LLM Judge
        reward = self.llm_judge.evaluate_code(
            prompt=self.target_prompt, 
            code=code, 
            ast=ast
        )
        
        # 3. Ensure a syntactically valid script always gets at least a baseline reward (0.1)
        #    This prevents the MCTS from abandoning perfectly valid parsing branches entirely.
        return max(0.1, min(reward, 1.0))

# =====================================================================
# 3. Neural Component: Local Ollama LLM Heuristic
# =====================================================================

class OllamaLLMHeuristic:
    """Uses a local Ollama LLM to predict probabilities for valid grammar expansions."""
    
    def __init__(self, prompt: str, model: str = "llama3", 
                 dsl_name: str = "generic", dsl_description: str = "programming", 
                 action_aliases: dict = None, few_shot_examples: list = None):
        self.prompt = prompt
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.token = os.getenv("OLLAMA_API_KEY") # Optional API key for authentication
        self.cache = {} # Cache to store LLM responses for seen states
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

        # --- Injected DSL Knowledge ---
        self.dsl_name = dsl_name
        self.dsl_description = dsl_description
        self.action_aliases = action_aliases or {}

        # --- Few-Shot Injection ---
        self.few_shot_examples = few_shot_examples or []
        self.examples_str = ""
        if self.few_shot_examples:
            self.examples_str = f"\nHere are some reference examples mapping intents to {self.dsl_name} code:\n"
            for ex in self.few_shot_examples:
                self.examples_str += f"- Intent: {ex['nl']}\n  Code: {ex['code']}\n"

    def predict_and_evaluate(self, state: Tuple[str, ...], valid_actions: List[Tuple[str, ...]]) -> Tuple[Dict[Tuple[str, ...], float], float]:
        # 1. Short-circuit: If there's only 1 valid grammar rule, bypass the LLM completely.
        if len(valid_actions) == 1:
            return {valid_actions[0]: 1.0}, 0.5

        # 2. Caching: Check if we have evaluated this exact state + actions before
        cache_key = (state, tuple(valid_actions))
        if cache_key in self.cache:
            return self.cache[cache_key]

        state_str = "".join(state)
        actions_dict = {}
        for i, action in enumerate(valid_actions):
            action_str = "".join(action)
            # If the action contains a known cryptic symbol, append the explanation
            explanation = self.action_aliases.get(action_str.strip(), "")
            if explanation:
                actions_dict[str(i)] = f"'{action_str}' ({explanation})"
            else:
                actions_dict[str(i)] = f"'{action_str}'"
        
        # Prepare the prompt for JSON mode
        sys_instruction = (
            f"You are a coding assistant guiding a {self.dsl_name} code generator. "
            "Evaluate the given 'Partial Code' against the 'User Intent'. "
            f"{self.examples_str}\n"
            "You are provided with 'Valid Next Actions' to replace the leftmost '<...>' placeholder. "
            "Return a JSON object with strictly two keys:\n"
            "1. 'action_scores': A dictionary mapping the action index (string) to a score (1.0 to 10.0) based on how likely it solves the intent.\n"
            "2. 'state_value': A float between 0.0 and 1.0 estimating how promising the current Partial Code is.\n"
            "Score 0.0 if the variable name is wrong, or if the integer value does not match the prompt.\n"
        )
        
        user_msg = (
            f"User Intent: {self.prompt}\n"
            f"Partial Code: {state_str}\n"
            f"Valid Next Actions: {json.dumps(actions_dict)}\n"
        )
        
        payload = {
            "model": self.model,
            "prompt": f"{sys_instruction}\n\n{user_msg}",
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.5 # Some randomness to encourage exploration, but not too much!
            }
        }

        try:
            # print(f"  [LLM Requesting...] Evaluating {len(valid_actions)} actions...")
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            # print(f"[DEBUG RESPONSE] Response: {response.json()}")
            
            # Extract JSON output
            llm_output = json.loads(response.json()["response"])
            
            scores = llm_output.get("action_scores", {})
            state_value = float(llm_output.get("state_value", 0.5))
            
            # Map indices back to actions and calculate total score for Softmax/Normalization
            action_probs = {}
            total_score = 0.0
            
            for i, action in enumerate(valid_actions):
                # Default to a score of 1.0 if the LLM hallucinated/missed an index
                score = float(scores.get(str(i), 1.0))
                action_probs[action] = score
                total_score += score
                
            # Normalize to sum up to 1.0
            if total_score > 0:
                for a in action_probs:
                    action_probs[a] /= total_score
            else:
                raise ValueError("Total score is 0.")

        except Exception as e:
            # print(f"  [Ollama Error / Timeout]: {e}. Falling back to uniform probabilities.")
            prob = 1.0 / len(valid_actions)
            action_probs = {action: prob for action in valid_actions}
            state_value = 0.5

        # Save to cache and return
        self.cache[cache_key] = (action_probs, state_value)
        # print(f"[DEBUG PRIORS] State: {''.join(state)}")
        # print(f"[DEBUG PRIORS] Probs: {action_probs}")
        return action_probs, state_value
    
    def evaluate_code(self, prompt: str, code: str, ast: dict) -> float:
        """
        LLM-as-a-Judge: Evaluates the terminal MiniZinc code against the natural language intent.
        """
        # Cache terminal state evaluations to save massive amounts of compute during MCTS rollouts
        cache_key = ("eval", code)
        if cache_key in self.cache:
            return self.cache[cache_key]

        sys_instruction = (
            f"You are a strict, expert {self.dsl_name} code evaluator. "
            f"You will be given a User Intent, the generated {self.dsl_name} Code, and its corresponding parsed AST (Abstract Syntax Tree). "
            f"{self.examples_str}\n"
            "Your job is to determine how accurately the code implements the User Intent. "
            "Return a JSON object with a single key 'reward' mapping to a float between 0.0 and 1.0. "
            "1.0 means perfect semantic match. 0.0 means it completely fails to fulfill the user's requirements."
        )
        
        ast_json = ast_to_json_serializable(ast)
        user_msg = (
            f"User Intent: {prompt}\n"
            f"{self.dsl_name} Code: {code}\n"
            f"Parsed AST: {json.dumps(ast_json)}\n"
        )
        
        payload = {
            "model": self.model,
            "prompt": f"{sys_instruction}\n\n{user_msg}",
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0 # Strict deterministic evaluation
            }
        }
        
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
            llm_output = json.loads(response.json()["response"])
            reward = float(llm_output.get("reward", 0.0))
            # print(f"[DEBUG JUDGE] Code: {''.join(code)}")
            # print(f"[DEBUG JUDGE] Reward Assigned: {reward}")
            
            # Bound the reward
            reward = max(0.0, min(reward, 1.0))

        except Exception as e:
            print(f"  [Evaluation Error]: {e}")
            # If the LLM fails, return a baseline reward indicating syntax passed but semantics are unknown
            reward = 0.1 

        self.cache[cache_key] = reward
        return reward
    
    def extract_entities(self, prompt: str) -> dict:
        """
        Pre-processes the prompt to extract variable names and literals.
        Returns a dictionary like {"identifiers": ["y"], "integer_literals": ["10"]}
        """
        sys_instruction = (
            f"You are a helpful assistant for a {self.dsl_description} system. "
            "Extract the variable names and integer literal values from the given user prompt. "
            "Return a JSON object with strictly two keys:\n"
            "- 'identifiers': A list of strings representing variable names (e.g., ['x', 'y']).\n"
            "- 'integer_literals': A list of strings representing integer numbers (e.g., ['5', '10']).\n"
            "If none are found, return empty lists."
        )
        
        payload = {
            "model": self.model,
            "prompt": f"{sys_instruction}\n\nUser Prompt: {prompt}",
            "format": "json",
            "stream": False,
            "options": {
                "temperature": 0.0 # Deterministic extraction
            }
        }
        
        print("\n[Extraction] Analyzing prompt for entities...")
        try:
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            extracted = json.loads(response.json()["response"])
            
            # Ensure lists are returned even if the LLM hallucinates format
            return {
                "identifiers": extracted.get("identifiers", []),
                "integer_literals": extracted.get("integer_literals", [])
            }
        except Exception as e:
            print(f"[Extraction Error]: {e}")
            return {"identifiers": [], "integer_literals": []}

    def evaluate_compiler_error(self, state: tuple, error_msg: str) -> float:
        cache_key = ("error_eval", tuple(state), error_msg)
        if cache_key in self.cache: return self.cache[cache_key]

        sys_instruction = (
            f"You are an expert {self.dsl_name} debugger. "
            "A code generator produced a partial snippet, which was automatically completed into a stub to check viability. "
            f"{self.examples_str}\n"  # <--- INJECTED HERE
            "The compiler returned an error on the stub. "
            "Analyze if the compiler error is caused by a fundamental flaw in the 'Partial Code' prefix, "
            "or if it is merely an artifact of a poor automatic completion. "
            "Return a JSON object with a single key 'viability_score' mapping to a float between 0.0 and 1.0. "
            "Score 0.0 if the Partial Code is irreversibly broken. Score > 0.5 if the Partial Code is fine and the error is just a completion artifact."
        )
        
        user_msg = f"Partial Code: {''.join(state)}\nCompiler Error: {error_msg}"
        payload = {
            "model": self.model,
            "prompt": f"{sys_instruction}\n\n{user_msg}",
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.0}
        }
        try:
            import requests, json
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            score = float(json.loads(response.json()["thinking"]).get("viability_score", 0.0))
        except Exception:
            score = 0.0
            
        self.cache[cache_key] = score
        return score
    
    def predict_with_feedback(self, state: Tuple[str, ...], valid_actions: List[Tuple[str, ...]], previous_errors: list) -> Dict[Tuple[str, ...], float]:
        """
        Re-evaluates the valid actions based on the compiler errors from previous failed rollout attempts.
        """
        # Cache based on the number of previous errors to avoid repeating the exact same feedback loop
        cache_key = ("feedback", state, tuple(valid_actions), len(previous_errors))
        if cache_key in self.cache: 
            return self.cache[cache_key]

        state_str = "".join(state)
        actions_dict = {str(i): "".join(a) for i, a in enumerate(valid_actions)}

        # Format the feedback history
        errors_str = ""
        for i, (code, err) in enumerate(previous_errors):
            errors_str += f"\nAttempt {i+1}:\nGenerated Code:\n{code}\nCompiler Error:\n{err}\n"

        sys_instruction = (
            f"You are an expert {self.dsl_name} debugger guiding a code generator. "
            "We are at a 'Partial Code' state and need to choose the 'Valid Next Action'. "
            "Previously, we tried completing this code, but it resulted in compiler errors. "
            "Review the past attempts and compiler errors to understand what went wrong. "
            "Then, score the 'Valid Next Actions' to steer the generation away from the error and towards a correct solution. "
            "Return a JSON object with a single key 'action_scores' mapping the action index (string) to a score (1.0 to 10.0)."
        )

        import json
        user_msg = (
            f"User Intent: {self.prompt}\n"
            f"Partial Code: {state_str}\n"
            f"--- PREVIOUS FAILED ATTEMPTS ---{errors_str}"
            f"--- END PREVIOUS ATTEMPTS ---\n"
            f"Valid Next Actions: {json.dumps(actions_dict)}\n"
        )
        
        payload = {
            "model": self.model,
            "prompt": f"{sys_instruction}\n\n{user_msg}",
            "format": "json",
            "stream": False,
            "options": {"temperature": 0.3} # Slightly higher temp to encourage changing its mind
        }

        try:
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            llm_output = json.loads(response.json()["thinking"]) # or "response"
            
            scores = llm_output.get("action_scores", {})
            action_probs = {}
            total_score = 0.0
            for i, action in enumerate(valid_actions):
                score = float(scores.get(str(i), 1.0))
                action_probs[action] = score
                total_score += score
                
            if total_score > 0:
                for a in action_probs: action_probs[a] /= total_score
            else:
                raise ValueError("Total score is 0.")

        except Exception as e:
            # Fallback to uniform if LLM fails formatting
            prob = 1.0 / len(valid_actions)
            action_probs = {action: prob for action in valid_actions}

        self.cache[cache_key] = action_probs
        return action_probs

# =====================================================================
# 4. Test Execution
# =====================================================================
if __name__ == "__main__":
    nl_prompt = "Write a MiniZinc model to find an integer a that is exactly equal to 10."
    # nl_prompt = "Declare two booleans a and c, constrain that either a or c, and satisfy."
    print(f"NL Prompt: '{nl_prompt}'")

    model = os.getenv("OLLAMA_MODEL", "llama3")
    print(f"Using Ollama Model: {model}")

    # Define the domain-specific aliases here
    minizinc_aliases = {
        "\\/": "Logical OR (either/or)",
        "/\\": "Logical AND (both)",
        "->": "Logical Implication (if/then)",
        "==": "Equality (exactly equal)",
        "!=": "Inequality (not equal)"
    }

    # 1. Initialize the Neural component (LLM)
    llm = OllamaLLMHeuristic(
        prompt=nl_prompt, 
        model=model,
        dsl_name="MiniZinc",
        dsl_description="constraint programming",
        action_aliases=minizinc_aliases,
        few_shot_examples=minizinc_few_shot_examples
    ) 
    
    extracted_data = llm.extract_entities(prompt=nl_prompt)

    # 2. Initialize the Environment, passing the LLM in as the judge
    env = MiniZincEnvironment(target_prompt=nl_prompt, llm_judge=llm, extracted_entities=extracted_data)
    
    # 3. Instantiate MCTS
    mcts = NeurosymbolicMCTS(env=env, llm_policy=llm, c_puct=1.5)
    
    initial_ast = ("<Model>",)
    final_code = mcts.generate_code(initial_ast, max_steps=200, num_simulations=200)
    
    print("\n--- Final Generated MiniZinc Code ---")
    print(final_code)