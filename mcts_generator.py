import random
from typing import List, Dict, Any, Optional
from minizinc_parser import parse_model, build_parser
from lark import Lark

class MCTSNode:
    def __init__(self, partial_ast: Dict[str, Any], parent: Optional['MCTSNode'] = None):
        self.partial_ast = partial_ast  # Current AST state
        self.parent = parent
        self.children: List['MCTSNode'] = []
        self.visits = 0
        self.value = 0.0
        self.untried_actions: List[Any] = self.get_possible_actions()

    def get_possible_actions(self) -> List[Any]:
        """Get valid grammar productions to expand the current partial AST."""
        # This is a simplified version; in practice, we'd analyze the AST for non-terminals
        # For now, return dummy actions
        return ["var_decl", "constraint", "solve"]  # Placeholder

    def is_terminal(self) -> bool:
        """Check if this node represents a complete MiniZinc model."""
        # Check if AST has solve item and no incomplete parts
        return 'solve' in [item.get('type') for item in self.partial_ast.get('items', [])]

    def expand(self) -> 'MCTSNode':
        """Expand by choosing an untried action."""
        action = self.untried_actions.pop()
        new_ast = self.apply_action(action)
        child = MCTSNode(new_ast, self)
        self.children.append(child)
        return child

    def apply_action(self, action: str) -> Dict[str, Any]:
        """Apply a grammar action to create a new partial AST."""
        # Placeholder: add a dummy item based on action
        new_ast = self.partial_ast.copy()
        if 'items' not in new_ast:
            new_ast['items'] = []
        if action == "var_decl":
            new_ast['items'].append({"type": "var_decl", "name": "x", "decl": {"type": "var_range", "lo": 1, "hi": 3}})
        elif action == "constraint":
            new_ast['items'].append({"type": "constraint", "expr": {"type": "binop", "op": ">", "left": "x", "right": 1}})
        elif action == "solve":
            new_ast['items'].append({"type": "solve", "mode": "satisfy"})
        return new_ast

    def best_child(self, c: float = 1.4) -> 'MCTSNode':
        """Select the best child using UCT formula."""
        return max(self.children, key=lambda child: child.value / child.visits + c * (self.visits ** 0.5) / child.visits)

    def update(self, reward: float):
        """Backpropagate the reward."""
        self.visits += 1
        self.value += reward
        if self.parent:
            self.parent.update(reward)

class MCTS:
    def __init__(self, root_ast: Dict[str, Any]):
        self.root = MCTSNode(root_ast)

    def search(self, iterations: int) -> MCTSNode:
        """Perform MCTS search for the given number of iterations."""
        for _ in range(iterations):
            node = self.select(self.root)
            if not node.is_terminal():
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
        """Simulate a rollout from the current node."""
        # Placeholder: random rollout
        depth = 0
        while not node.is_terminal() and depth < 10:
            if node.untried_actions:
                action = random.choice(node.untried_actions)
                node.untried_actions.remove(action)
                new_ast = node.apply_action(action)
                node = MCTSNode(new_ast, node.parent)
            depth += 1
        # Reward: 1 if terminal, 0 otherwise
        return 1.0 if node.is_terminal() else 0.0

def generate_code(nl_prompt: str, iterations: int = 1000) -> str:
    """Generate MiniZinc code using MCTS guided by NL prompt."""
    # Start with empty model
    root_ast = {"type": "model", "items": []}
    mcts = MCTS(root_ast)
    best_node = mcts.search(iterations)
    # Convert AST back to code (placeholder)
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
    nl = "Declare x from 1 to 3, constrain x > 1, solve satisfy."
    code = generate_code(nl)
    print("Generated code:")
    print(code)