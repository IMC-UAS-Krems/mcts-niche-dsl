import math
import json
import requests
from typing import List, Dict, Tuple, Any, Optional
from minizinc_parser import parse_model, ast_to_json_serializable

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

    def search(self, initial_state: Tuple[str, ...], num_simulations: int = 50) -> Tuple[str, ...]:
        root = MCTSNode(state=initial_state, prior_prob=1.0)

        for _ in range(num_simulations):
            node = root
            # 1. Selection
            while node.is_expanded() and not self.env.is_terminal(node.state):
                action, node = node.get_best_child(self.c_puct)

            # 2. Evaluation & Expansion
            if not self.env.is_terminal(node.state):
                valid_actions = self.env.get_valid_actions(node.state)
                action_probs, value = self.llm.predict_and_evaluate(node.state, valid_actions)
                node.expand(action_probs, self.env)
            else:
                # 3. Terminal Reward
                value = self.env.compute_reward(node.state)

            # 4. Backpropagation
            node.backpropagate(value)

        # Return the most visited action (most robust choice)
        return max(root.children.items(), key=lambda item: item[1].visit_count)[0]

    def generate_code(self, initial_state: Tuple[str, ...], max_steps: int = 20, num_simulations: int = 50) -> str:
        current_state = initial_state
        step = 0
        
        print(f"\n--- Starting Generation ---")
        while not self.env.is_terminal(current_state) and step < max_steps:
            best_action = self.search(current_state, num_simulations)
            current_state = self.env.apply_action(current_state, best_action)
            step += 1
            print(f"Step {step}: {''.join(current_state)}")
            
        return "".join(current_state)


# =====================================================================
# 2. MiniZinc Grammar Environment
# =====================================================================
class MiniZincEnvironment:
    """Handles the symbolic derivation of MiniZinc Code and evaluates it via AST matching."""
    
    def __init__(self, target_prompt: str, llm_judge: 'OllamaLLMHeuristic'):
        self.target_prompt = target_prompt
        self.llm_judge = llm_judge # Inject the LLM Judge
        
        # Simplified EBNF Grammar mapping
        self.grammar = {
            "<Model>": [[ "<VarDecl>", " ", "<Constraint>", " ", "<Solve>" ]],
            "<VarDecl>": [[ "var ", "<Type>", ": ", "<Ident>", ";" ]],
            "<Type>": [[ "int" ], [ "bool" ]],
            "<Ident>": [[ "x" ], [ "y" ]],
            "<Constraint>": [[ "constraint ", "<Expr>", " ", "<Op>", " ", "<Expr>", ";" ]],
            "<Expr>": [[ "<Ident>" ], [ "<IntLit>" ]],
            "<IntLit>": [[ "0" ], [ "5" ], [ "10" ]],
            "<Op>": [[ ">" ], [ "<" ], [ "==" ]],
            "<Solve>": [[ "solve satisfy;" ], [ "solve maximize ", "<Ident>", ";" ]]
        }

    def _get_leftmost_nt(self, state: tuple) -> int:
        for i, symbol in enumerate(state):
            if symbol.startswith("<") and symbol.endswith(">"):
                return i
        return -1

    def get_valid_actions(self, state: tuple) -> list:
        idx = self._get_leftmost_nt(state)
        if idx == -1: return[]
        nt = state[idx]
        return [tuple(prod) for prod in self.grammar[nt]]

    def apply_action(self, state: tuple, action: tuple) -> tuple:
        idx = self._get_leftmost_nt(state)
        return state[:idx] + action + state[idx+1:]

    def is_terminal(self, state: tuple) -> bool:
        return self._get_leftmost_nt(state) == -1

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
        self.cache = {} # Cache to store LLM responses for seen states

    def predict_and_evaluate(self, state: Tuple[str, ...], valid_actions: List[Tuple[str, ...]]) -> Tuple[Dict[Tuple[str, ...], float], float]:
        # 1. Short-circuit: If there's only 1 valid grammar rule, bypass the LLM completely.
        if len(valid_actions) == 1:
            return {valid_actions[0]: 1.0}, 0.5

        # 2. Caching: Check if we have evaluated this exact state + actions before
        cache_key = (state, tuple(valid_actions))
        if cache_key in self.cache:
            return self.cache[cache_key]

        state_str = "".join(state)
        actions_dict = {str(i): "".join(a) for i, a in enumerate(valid_actions)}
        
        # Prepare the prompt for JSON mode
        sys_instruction = (
            "You are a coding assistant guiding a code generator. "
            "Evaluate the given 'Partial Code' against the 'User Intent'. "
            "You are provided with 'Valid Next Actions' to replace the leftmost '<...>' placeholder. "
            "Return a JSON object with strictly two keys:\n"
            "1. 'action_scores': A dictionary mapping the action index (string) to a score (1.0 to 10.0) based on how likely it solves the intent.\n"
            "2. 'state_value': A float between 0.0 and 1.0 estimating how promising the current Partial Code is.\n"
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
            response = requests.post(self.api_url, json=payload, timeout=30)
            response.raise_for_status()
            
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
            
            llm_output = json.loads(response.json()["response"])
            reward = float(llm_output.get("reward", 0.0))
            
            # Bound the reward
            reward = max(0.0, min(reward, 1.0))

        except Exception as e:
            # print(f"  [Evaluation Error]: {e}")
            # If the LLM fails, return a baseline reward indicating syntax passed but semantics are unknown
            reward = 0.1 

        self.cache[cache_key] = reward
        return reward


# =====================================================================
# 4. Test Execution
# =====================================================================
if __name__ == "__main__":
    nl_prompt = "Write a MiniZinc model to find an integer y that is exactly equal to 10."
    print(f"NL Prompt: '{nl_prompt}'")

    # 1. Initialize the Neural component (LLM)
    llm = OllamaLLMHeuristic(prompt=nl_prompt, model="llama3") 
    
    # 2. Initialize the Environment, passing the LLM in as the judge
    env = MiniZincEnvironment(target_prompt=nl_prompt, llm_judge=llm)
    
    # 3. Instantiate MCTS
    mcts = NeurosymbolicMCTS(env=env, llm_policy=llm, c_puct=1.5)
    
    initial_ast = ("<Model>",)
    final_code = mcts.generate_code(initial_ast, max_steps=20, num_simulations=20)
    
    print("\n--- Final Generated MiniZinc Code ---")
    print(final_code)