import random
import requests
import json
import os
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from minizinc_parser import parse_model, build_parser
from lark import Lark

# Load environment variables from .env file
load_dotenv()

class OllamaClient:
    """Client for Ollama API to query open-source LLMs."""
    
    def __init__(self, base_url: str = "http://localhost:11434", model: str = "mistral", 
                 api_key: Optional[str] = None, auth_token: Optional[str] = None):
        """Initialize Ollama client with optional authentication.
        
        Args:
            base_url: Base URL for Ollama API
            model: Model name to use
            api_key: API key for authentication (added to 'Authorization: Bearer' header)
            auth_token: Alternative authentication token
        """
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.endpoint_candidates = [
            f"{self.base_url}/api/generate",
            f"{self.base_url}/v1/generate",
            f"{self.base_url}/api/completions",
            f"{self.base_url}/v1/completions",
            f"{self.base_url}/v1/chat/completions",
        ]
        self.endpoint = self.endpoint_candidates[0]
        self.headers = {"Content-Type": "application/json"}
        
        # Add authentication headers if provided
        token = api_key or auth_token
        if token:
            self.headers["Authorization"] = f"Bearer {token}"
    
    def _parse_response(self, data: Any) -> str:
        """Parse different Ollama/OpenAI-style response formats."""
        if isinstance(data, dict):
            if 'response' in data:
                return data['response'] or ""
            if 'output' in data:
                output = data['output']
                if isinstance(output, list):
                    return ''.join(str(item) for item in output)
                return str(output)
            if 'choices' in data and isinstance(data['choices'], list) and data['choices']:
                first = data['choices'][0]
                if isinstance(first, dict):
                    if 'message' in first and isinstance(first['message'], dict):
                        return first['message'].get('content', '')
                    if 'text' in first:
                        return first.get('text', '')
                    if 'output' in first:
                        return str(first['output'])
        return str(data)

    def generate(self, prompt: str, stream: bool = False) -> str:
        """Generate text using Ollama API."""
        last_error = None
        for endpoint in self.endpoint_candidates:
            if endpoint.endswith("/v1/chat/completions"):
                payload = {
                    "model": self.model,
                    "messages": [{"role": "user", "content": prompt}],
                }
            elif endpoint.endswith("/v1/completions"):
                payload = {
                    "model": self.model,
                    "input": prompt,
                }
            else:
                payload = {
                    "model": self.model,
                    "prompt": prompt,
                    "stream": stream,
                }

            try:
                response = requests.post(endpoint, json=payload, headers=self.headers, timeout=30)
            except requests.exceptions.RequestException as e:
                last_error = e
                continue

            if response.status_code == 404:
                last_error = requests.exceptions.HTTPError(
                    f"404 Not Found: {response.text}"
                )
                continue
            if response.status_code == 403:
                raise requests.exceptions.HTTPError(
                    f"403 Forbidden: Check API key and authentication. Response: {response.text}"
                )
            try:
                response.raise_for_status()
            except requests.exceptions.RequestException as e:
                last_error = requests.exceptions.HTTPError(
                    f"{e} - {response.text}"
                )
                continue

            self.endpoint = endpoint
            if stream:
                result = ""
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            result += data.get('response', '')
                        except json.JSONDecodeError:
                            result += line.decode('utf-8', errors='ignore')
                return result
            else:
                data = response.json()
                return self._parse_response(data)

        if last_error:
            print(f"Ollama API error: {last_error}")
        else:
            print(f"Ollama API error: no endpoint responded successfully. Tried: {self.endpoint_candidates}")
        return ""
    
    def rank_actions(self, nl_prompt: str, actions: List[str]) -> Dict[str, float]:
        """Rank candidate actions by likelihood given the NL prompt."""
        prompt = f"""Given the natural language intent: "{nl_prompt}"
        
Rank these MiniZinc modeling actions by relevance (0.0 to 1.0):
{chr(10).join(f"- {action}" for action in actions)}

Return only the ranking as JSON like: {{"action1": 0.8, "action2": 0.3}}"""
        
        response = self.generate(prompt)
        try:
            # Extract JSON from response
            import re
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group())
        except:
            pass
        
        # Fallback: equal probability
        return {action: 1.0 / len(actions) for action in actions}
    
    def evaluate_model(self, nl_prompt: str, minizinc_code: str) -> float:
        """Evaluate how well generated code matches the NL intent (0.0 to 1.0)."""
        prompt = f"""Given the natural language specification:
"{nl_prompt}"

And the generated MiniZinc code:
```
{minizinc_code}
```

On a scale of 0 to 1, how well does the code match the specification?
Respond with only a number between 0 and 1."""
        
        response = self.generate(prompt).strip()
        try:
            return float(response)
        except:
            return 0.5  # Default neutral score


class MCTSNode:
    def __init__(self, partial_ast: Dict[str, Any], parent: Optional['MCTSNode'] = None, nl_prompt: str = ""):
        self.partial_ast = partial_ast  # Current AST state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.nl_prompt = nl_prompt
        self.untried_actions: List[Any] = self.get_possible_actions()

    def get_possible_actions(self) -> List[Any]:
        """Get valid grammar productions to expand the current partial AST."""
        # Check what's missing in the current model
        items = self.partial_ast.get('items', [])
        item_types = set(item.get('type') for item in items)
        
        actions = []
        if 'var_decl' not in item_types:
            actions.append("add_var_decl")
        if 'constraint' not in item_types:
            actions.append("add_constraint")
        if 'solve' not in item_types:
            actions.append("add_solve")
        
        return actions

    def is_terminal(self) -> bool:
        """Check if this node represents a complete MiniZinc model."""
        items = self.partial_ast.get('items', [])
        item_types = set(item.get('type') for item in items)
        return 'solve' in item_types

    def expand(self) -> 'MCTSNode':
        """Expand by choosing an untried action."""
        if not self.untried_actions:
            return self  # No more actions
        action = self.untried_actions.pop(0)
        new_ast = self.apply_action(action)
        child = MCTSNode(new_ast, self, self.nl_prompt)
        self.children.append(child)
        return child

    def apply_action(self, action: str) -> Dict[str, Any]:
        """Apply a grammar action to create a new partial AST."""
        new_ast = self.partial_ast.copy()
        if 'items' not in new_ast:
            new_ast['items'] = []
        new_ast['items'] = new_ast['items'].copy()
        
        if action == "add_var_decl":
            new_ast['items'].append({
                "type": "var_decl",
                "name": "x",
                "decl": {"type": "var_range", "lo": 1, "hi": 3},
                "value": None
            })
        elif action == "add_constraint":
            new_ast['items'].append({
                "type": "constraint",
                "expr": {"type": "binop", "op": ">", "left": "x", "right": 1}
            })
        elif action == "add_solve":
            new_ast['items'].append({"type": "solve", "mode": "satisfy"})
        
        return new_ast

    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        """Select the best child using UCT formula."""
        if not self.children:
            return self
        return max(self.children, key=lambda child: (child.value / max(child.visits, 1)) + c * (self.visits ** 0.5) / max(child.visits, 1))

    def update(self, reward: float):
        """Backpropagate the reward."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.update(reward)

class MCTS:
    def __init__(self, root_ast: Dict[str, Any], nl_prompt: str = "", llm_client: Optional[OllamaClient] = None):
        self.root = MCTSNode(root_ast, nl_prompt=nl_prompt)
        self.nl_prompt = nl_prompt
        self.llm_client = llm_client

    def search(self, iterations: int) -> MCTSNode:
        """Perform MCTS search for the given number of iterations."""
        for _ in range(iterations):
            node = self.select(self.root)
            if not node.is_terminal() and node.untried_actions:
                node = node.expand()
            reward = self.simulate(node)
            node.update(reward)
        return self.root.best_child(c=0)  # Return best child without exploration

    def select(self, node: MCTSNode) -> MCTSNode:
        """Select a node to expand using UCT."""
        while node.children and not node.is_terminal():
            node = node.best_child()
        return node

    def simulate(self, node: MCTSNode) -> float:
        """Simulate a rollout from the current node with LLM guidance."""
        current_node = node
        depth = 0
        max_depth = 10
        
        while not current_node.is_terminal() and depth < max_depth:
            actions = current_node.untried_actions
            if not actions:
                break
            
            # Use LLM to rank actions if available
            if self.llm_client:
                action_scores = self.llm_client.rank_actions(self.nl_prompt, actions)
                # Choose action with highest score
                action = max(actions, key=lambda a: action_scores.get(a, 0.5))
            else:
                # Random selection as fallback
                action = random.choice(actions)
            
            current_node.untried_actions.remove(action)
            new_ast = current_node.apply_action(action)
            current_node = MCTSNode(new_ast, current_node.parent, self.nl_prompt)
            depth += 1
        
        # Generate code and evaluate with LLM if available
        code = ast_to_code(current_node.partial_ast)
        
        if self.llm_client and self.nl_prompt:
            reward = self.llm_client.evaluate_model(self.nl_prompt, code)
        else:
            # Reward: 1 if terminal, 0 otherwise
            reward = 1.0 if current_node.is_terminal() else 0.0
        
        return reward

def generate_code(nl_prompt: str, iterations: int = 1000, use_llm: bool = False, 
                 ollama_url: str = "http://localhost:11434", ollama_model: str = "mistral",
                 api_key: Optional[str] = None, auth_token: Optional[str] = None) -> str:
    """Generate MiniZinc code using MCTS guided by NL prompt.
    
    Args:
        nl_prompt: Natural language description of the desired MiniZinc model
        iterations: Number of MCTS iterations to perform
        use_llm: Whether to use LLM guidance via Ollama
        ollama_url: Base URL for Ollama API
        ollama_model: Model name to use in Ollama
        api_key: API key for Ollama authentication (overrides env var)
        auth_token: Alternative authentication token
    
    Returns:
        Generated MiniZinc code as a string
    """
    # Start with empty model
    root_ast = {"type": "model", "items": []}
    
    llm_client = None
    if use_llm:
        try:
            # Use provided key or load from environment
            final_api_key = api_key or os.getenv("OLLAMA_API_KEY")
            final_auth_token = auth_token or os.getenv("OLLAMA_AUTH_TOKEN")
            
            llm_client = OllamaClient(base_url=ollama_url, model=ollama_model, 
                                     api_key=final_api_key, auth_token=final_auth_token)
            # Test connection
            llm_client.generate("test", stream=False)
        except Exception as e:
            print(f"Warning: Could not connect to Ollama: {e}")
            llm_client = None
    
    mcts = MCTS(root_ast, nl_prompt=nl_prompt, llm_client=llm_client)
    best_node = mcts.search(iterations)
    return ast_to_code(best_node.partial_ast)

def ast_to_code(ast: Dict[str, Any]) -> str:
    """Convert AST back to MiniZinc code string."""
    if ast['type'] != 'model':
        raise ValueError("AST must be a model")
    code_lines = []
    for item in ast.get('items', []):
        line = item_to_code(item)
        if line:
            code_lines.append(line + ";")
    return '\n'.join(code_lines)

def item_to_code(item: Dict[str, Any]) -> str:
    """Convert an item dict to code string."""
    itype = item['type']
    if itype == 'var_decl':
        decl_str = ti_expr_to_code(item['decl'])
        name = item['name']
        value_str = f" = {expr_to_str(item['value'])}" if item.get('value') is not None else ""
        return f"{decl_str}: {name}{value_str}"
    elif itype == 'assign':
        return f"{item['name']} = {expr_to_str(item['value'])}"
    elif itype == 'constraint':
        return f"constraint {expr_to_str(item['expr'])}"
    elif itype == 'solve':
        mode = item['mode']
        expr_str = f" {expr_to_str(item['expr'])}" if 'expr' in item else ""
        return f"solve {mode}{expr_str}"
    elif itype == 'output':
        return f"output {expr_to_str(item['expr'])}"
    elif itype == 'include':
        return f"include {item['path']}"
    return ""  # Unknown item

def ti_expr_to_code(ti_expr: Dict[str, Any]) -> str:
    """Convert type-inst expr to string."""
    if ti_expr['type'] == 'var_range':
        return f"var {ti_expr['lo']}..{ti_expr['hi']}"
    elif ti_expr['type'] == 'base_ti_expr':
        parts = []
        if 'var' in ti_expr.get('values', []):
            parts.append('var')
        if 'set_of' in ti_expr.get('values', []):
            parts.append('set of')
        parts.append(ti_expr.get('base_type', 'int'))  # Default
        return ' '.join(parts)
    # Add more cases as needed
    return str(ti_expr)  # Placeholder

def expr_to_str(expr: Any) -> str:
    """Convert expression AST to string."""
    if isinstance(expr, (int, float)):
        return str(expr)
    if isinstance(expr, str):
        return expr
    if isinstance(expr, bool):
        return 'true' if expr else 'false'
    if not isinstance(expr, dict):
        return str(expr)
    
    etype = expr['type']
    if etype == 'bool':
        return 'true' if expr['value'] else 'false'
    elif etype == 'binop':
        left = expr_to_str(expr['left'])
        op = expr['op']
        right = expr_to_str(expr['right'])
        return f"{left} {op} {right}"
    elif etype == 'call':
        name = expr['name']
        args = ', '.join(expr_to_str(arg) for arg in expr.get('args', []))
        return f"{name}({args})"
    elif etype == 'set':
        elements = ', '.join(expr_to_str(e) for e in expr.get('elements', []))
        return f"{{{elements}}}"
    elif etype == 'array':
        elements = ', '.join(expr_to_str(e) for e in expr.get('elements', []))
        return f"[{elements}]"
    elif etype == 'if':
        cond = expr_to_str(expr['cond'])
        then_part = expr_to_str(expr['then'])
        parts = [f"if {cond} then {then_part}"]
        if 'elif' in expr:
            for elif_part in expr['elif']:
                econd = expr_to_str(elif_part['cond'])
                ethen = expr_to_str(elif_part['then'])
                parts.append(f"elseif {econd} then {ethen}")
        if 'else' in expr:
            else_part = expr_to_str(expr['else'])
            parts.append(f"else {else_part}")
        parts.append("endif")
        return ' '.join(parts)
    # Add more expression types as needed
    return str(expr)  # Fallback

if __name__ == "__main__":
    # Example usage
    nl = "Declare x from 1 to 3, constrain x > 1, and solve to satisfy the constraints."
    
    print("Generating MiniZinc code with LLM guidance...")
    code = generate_code(nl, iterations=10, use_llm=True, ollama_model=os.getenv('OLLAMA_MODEL'))
    print("Generated code:")
    print(code)
    print()
    
    print("To use LLM guidance, ensure Ollama is running at http://localhost:11434")
    print("Example with LLM (requires Ollama):")
    print("  code = generate_code(nl, iterations=50, use_llm=True, ollama_model='mistral')")
