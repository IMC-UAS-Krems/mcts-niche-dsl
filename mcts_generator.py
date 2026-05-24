import math
import json
import requests
from typing import List, Dict, Tuple, Any, Optional
from minizinc_parser import parse_model, ast_to_json_serializable
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
    
    def fast_safe_rollout(self, state: Tuple[str, ...]) -> bool:
        current_state = state
        depth = 0
        max_depth = 40 
        
        while not self.env.is_terminal(current_state) and depth < max_depth:
            valid_actions = self.env.get_valid_actions(current_state)
            if not valid_actions: break
            
            # Use the LLM to pick the MOST LIKELY valid action, not just the shortest.
            # (Because the state is cached, this is extremely fast and doesn't spam the API)
            action_probs, _ = self.llm.predict_and_evaluate(current_state, valid_actions)
            best_action = max(action_probs, key=action_probs.get)
            
            current_state = self.env.apply_action(current_state, best_action)
            depth += 1
            
        if self.env.is_terminal(current_state):
            code = "".join(current_state)
            return self.env.check_compilation_only(code) 
        
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
                # print(f"[DEBUG SEARCH] Valid actions: {valid_actions}")
                action_probs, llm_value = self.llm.predict_and_evaluate(node.state, valid_actions)
                # print(f"[DEBUG SEARCH] Action probabilities: {action_probs}, State value: {value}")
                node.expand(action_probs, self.env)

                is_viable = self.fast_safe_rollout(node.state)
                
                if is_viable:
                    # The prefix is semantically sound. Trust the LLM's intuition for intent.
                    final_value = llm_value 
                else:
                    # The LLM guided us into a compiler error (e.g., bool == int). 
                    # Overwrite the LLM and instantly kill this search branch.
                    final_value = 0.0 
            else:
                # 3. Terminal Reward
                # print(f"[DEBUG SEARCH] Terminal node reached: {node.state}, evaluating reward...")
                final_value = self.env.compute_reward(node.state)
                terminal_reached = True

            # 4. Backpropagation
            node.backpropagate(final_value)

        if not terminal_reached:
            print("[MCTS Warning] No terminal state reached during simulations. Final selection may be suboptimal.")

        # Find the best child
        best_action, best_child = max(root.children.items(), key=lambda item: item[1].visit_count)
        
        # Return both the action AND its average reward (Q-value)
        return best_action, best_child.q_value

    def generate_code(self, initial_state: Tuple[str, ...], max_steps: int = 40, num_simulations: int = 50) -> str:
        current_state = initial_state
        step = 0
        
        while not self.env.is_terminal(current_state) and step < max_steps:
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
            
            # --- Expressions (Flattened to avoid infinite left-recursion) ---
            "<Expr>": [
                ["<Term>"],                                                                   # e.g., b
                ["<Term>", " ", "<CompOp>", " ", "<Term>"],                                   # e.g., x > 1, x in s
                ["<Term>", " ", "<MathOp>", " ", "<Term>", " ", "<CompOp>", " ", "<Term>"],   # e.g., x + y <= 10, x mod 2 == 0
                ["<Term>", " ", "<LogicOp>", " ", "<Term>"],                                  # e.g., a \/ c
                ["<Term>", " ", "<LogicOp>", " ", "<Term>", " ", "<CompOp>", " ", "<Term>"],  # e.g., b -> z > 2
                ["sum(", "<Ident>", ")", " ", "<CompOp>", " ", "<Term>"]                      # e.g., sum(arr) == 3
            ],
            
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
            return result.returncode == 0
        except Exception:
            return False

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
    
    def __init__(self, prompt: str, model: str = "llama3"):
        self.prompt = prompt
        self.model = model
        self.api_url = "http://localhost:11434/api/generate"
        self.token = os.getenv("OLLAMA_API_KEY") # Optional API key for authentication
        self.cache = {} # Cache to store LLM responses for seen states
        self.headers = {"Authorization": f"Bearer {self.token}"} if self.token else {}

    def predict_and_evaluate(self, state: Tuple[str, ...], valid_actions: List[Tuple[str, ...]]) -> Tuple[Dict[Tuple[str, ...], float], float]:
        # 1. Short-circuit: If there's only 1 valid grammar rule, bypass the LLM completely.
        if len(valid_actions) == 1:
            return {valid_actions[0]: 1.0}, 0.5

        # 2. Caching: Check if we have evaluated this exact state + actions before
        cache_key = (state, tuple(valid_actions))
        if cache_key in self.cache:
            return self.cache[cache_key]

        # Dictionary to alias cryptic symbols into semantic meaning
        semantic_map = {
            "\\/": "Logical OR (either/or)",
            "/\\": "Logical AND (both)",
            "->": "Logical Implication (if/then)",
            "==": "Equality (exactly equal)",
            "!=": "Inequality (not equal)"
        }

        state_str = "".join(state)
        actions_dict = {}
        for i, action in enumerate(valid_actions):
            action_str = "".join(action)
            # If the action contains a known cryptic symbol, append the explanation
            explanation = semantic_map.get(action_str.strip(), "")
            if explanation:
                actions_dict[str(i)] = f"'{action_str}' ({explanation})"
            else:
                actions_dict[str(i)] = f"'{action_str}'"
        
        # Prepare the prompt for JSON mode
        sys_instruction = (
            "You are a coding assistant guiding a code generator. "
            "Evaluate the given 'Partial Code' against the 'User Intent'. "
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
                "temperature": 0.2 # Low temperature for more deterministic logic evaluation
            }
        }

        try:
            # print(f"  [LLM Requesting...] Evaluating {len(valid_actions)} actions...")
            response = requests.post(self.api_url, headers=self.headers, json=payload, timeout=30)
            response.raise_for_status()
            # print(f"[DEBUG RESPONSE] Response: {response.json()}")
            
            # Extract JSON output
            llm_output = json.loads(response.json()["thinking"])
            
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
            "You are a strict, expert MiniZinc code evaluator. "
            "You will be given a User Intent, the generated MiniZinc Code, and its corresponding parsed AST (Abstract Syntax Tree). "
            "Your job is to determine how accurately the code implements the User Intent. "
            "Return a JSON object with a single key 'reward' mapping to a float between 0.0 and 1.0. "
            "1.0 means perfect semantic match. 0.0 means it completely fails to fulfill the user's requirements."
        )
        
        ast_json = ast_to_json_serializable(ast)
        user_msg = (
            f"User Intent: {prompt}\n"
            f"MiniZinc Code: {code}\n"
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
            
            llm_output = json.loads(response.json()["thinking"])
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
            "You are a helpful assistant for a constraint programming system. "
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
            extracted = json.loads(response.json()["thinking"])
            
            # Ensure lists are returned even if the LLM hallucinates format
            return {
                "identifiers": extracted.get("identifiers", []),
                "integer_literals": extracted.get("integer_literals", [])
            }
        except Exception as e:
            print(f"[Extraction Error]: {e}")
            return {"identifiers": [], "integer_literals": []}


# =====================================================================
# 4. Test Execution
# =====================================================================
if __name__ == "__main__":
    # nl_prompt = "Write a MiniZinc model to find an integer a that is exactly equal to 10."
    nl_prompt = "Declare two booleans a and c, constrain that either a or c, and satisfy."
    print(f"NL Prompt: '{nl_prompt}'")

    model = os.getenv("OLLAMA_MODEL", "llama3")
    print(f"Using Ollama Model: {model}")
    # 1. Initialize the Neural component (LLM)
    llm = OllamaLLMHeuristic(prompt=nl_prompt, model=model) 
    
    extracted_data = llm.extract_entities(prompt=nl_prompt)

    # 2. Initialize the Environment, passing the LLM in as the judge
    env = MiniZincEnvironment(target_prompt=nl_prompt, llm_judge=llm, extracted_entities=extracted_data)
    
    # 3. Instantiate MCTS
    mcts = NeurosymbolicMCTS(env=env, llm_policy=llm, c_puct=1.5)
    
    initial_ast = ("<Model>",)
    final_code = mcts.generate_code(initial_ast, max_steps=200, num_simulations=500)
    
    print("\n--- Final Generated MiniZinc Code ---")
    print(final_code)