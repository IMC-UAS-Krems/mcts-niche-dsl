from lark import Lark, Transformer, Token, Tree

MINIZINC_GRAMMAR = r"""
start: model
model: (item SEMICOLON)*
item: include_item | var_decl_item | assign_item | constraint_item | solve_item | output_item
include_item: INCLUDE string_literal
var_decl_item: ti_expr ":" IDENT ("=" expr)?
assign_item: IDENT "=" expr
constraint_item: CONSTRAINT expr
solve_item: SOLVE (SATISFY | MINIMIZE expr | MAXIMIZE expr)
output_item: OUTPUT expr
?ti_expr: base_ti_expr | array_ti_expr
?base_ti_expr: VAR SET_OF int_literal DOTS int_literal
             | VAR int_literal DOTS int_literal
             | VAR? SET_OF? base_type
array_ti_expr: ARRAY "[" index_set "]" OF base_ti_expr
index_set: int_literal DOTS int_literal | INT
base_type: BOOL | INT | FLOAT
?expr: expr_atom expr_binop_tail?
expr_binop_tail: bin_op expr
?expr_atom: "(" expr ")" | IDENT | bool_literal | int_literal | float_literal | set_literal | array_literal | if_then_else_expr | call_expr
bin_op: PLUS | MINUS | STAR | SLASH | DIV | MOD | LT | GT | LE | GE | EQ | NE | LAND | LOR | ARROW | IFF | IN | SUBSET | SUPERSET
bool_literal: TRUE | FALSE
set_literal: "{" [expr ("," expr)*] "}"
array_literal: "[" [expr ("," expr)*] "]"
if_then_else_expr: IF expr THEN expr (ELSEIF expr THEN expr)* (ELSE expr)? ENDIF
call_expr: IDENT "(" [expr ("," expr)*] ")"
annotations: ("::" IDENT ("(" [expr ("," expr)*] ")")? )*
INCLUDE: "include"
CONSTRAINT: "constraint"
SOLVE: "solve"
SATISFY: "satisfy"
MINIMIZE: "minimize"
MAXIMIZE: "maximize"
OUTPUT: "output"
VAR: "var"
SET_OF: "set of"
ARRAY: "array"
OF: "of"
IF: "if"
THEN: "then"
ELSEIF: "elseif"
ELSE: "else"
ENDIF: "endif"
TRUE: "true"
FALSE: "false"
DIV: "div"
MOD: "mod"
IN: "in"
SUBSET: "subset"
SUPERSET: "superset"
LAND: "\/\\"
LOR: "\\/"
ARROW: "->"
IFF: "<->"
PLUS: "+"
MINUS: "-"
STAR: "*"
SLASH: "/"
LT: "<"
GT: ">"
LE: "<="
GE: ">="
EQ: "=="
NE: "!="
DOTS: ".."
SEMICOLON: ";"
INT: "int"
BOOL: "bool"
FLOAT: "float"
STRING: /"[^"]*"/
IDENT: /[A-Za-z_][A-Za-z0-9_]*/
int_literal: /[0-9]+/
float_literal: /[0-9]+\.[0-9]+/
string_literal: STRING
%ignore /[ \t\f\r\n]+/
%ignore /%[^\n]*/
"""

minizinc_few_shot_examples = [
    {
        "nl": "Write a model to find an integer x that is strictly greater than 5. Satisfy the constraints.",
        "code": "var int: x;\nconstraint x > 5;\nsolve satisfy;\n"
    },
    {
        "nl": "Declare a boolean variable b. Constrain that b is true.",
        "code": "var bool: b;\nconstraint b;\nsolve satisfy;\n"
    },
    {
        "nl": "Find an array arr of 3 integers from 1 to 5. Constrain the sum of arr to equal 10.",
        "code": "array[1..3] of var 1..5: arr;\nconstraint sum(arr) == 10;\nsolve satisfy;\n"
    },
    {
        "nl": "Declare integer y between 0 and 50. Maximize y.",
        "code": "var 0..50: y;\nconstraint y > 0;\nsolve maximize y;\n"
    },
    {
        "nl": "Declare a set s from 1 to 10 and integer z. Constrain z in s.",
        "code": "var set of 1..10: s;\nvar int: z;\nconstraint z in s;\nsolve satisfy;\n"
    },
    {
        "nl": "Find an integer a equal to 10 and output a.",
        "code": "var int: a;\nconstraint a == 10;\nsolve satisfy;\noutput [show(a)];\n"
    },
    {
        "nl": "Find an integer x from 1 to 20 where x modulo 2 equals 0. Minimize x.",
        "code": "var 1..20: x;\nconstraint x mod 2 == 0;\nsolve minimize x;\n"
    },
    {
        "nl": "Declare boolean b and integer x. If b is true, x must be 5.",
        "code": "var bool: b;\nvar int: x;\nconstraint b -> x == 5;\nsolve satisfy;\n"
    },
    {
        "nl": "Declare integers x and y from 0 to 10. Constrain x + y <= 15.",
        "code": "var 0..10: x;\nvar 0..10: y;\nconstraint x + y <= 15;\nsolve satisfy;\n"
    },
    {
        "nl": "Find booleans a and c such that either a or c is true.",
        "code": "var bool: a;\nvar bool: c;\nconstraint a \\/ c;\nsolve satisfy;\n"
    }
]

class MiniZincTransformer(Transformer):
    def model(self, items):
        filtered = [item for item in items if not isinstance(item, Token) or item.type != 'SEMICOLON']
        return {"type": "model", "items": filtered}
    def item(self, value): return value[0]
    def include_item(self, values): return {"type": "include", "path": values[1][1:-1]}
    def var_decl_item(self, values):
        ti_expr, name = values[0], values[1]
        expr = values[3] if len(values) > 2 else None
        return {"type": "var_decl", "decl": ti_expr, "name": name, "value": expr}
    def assign_item(self, values): return {"type": "assign", "name": values[0], "value": values[2]}
    def constraint_item(self, values): return {"type": "constraint", "expr": values[1] if len(values) > 1 else values[0]}
    def solve_item(self, values):
        if values[1].type == "SATISFY": return {"type": "solve", "mode": "satisfy"}
        return {"type": "solve", "mode": values[1].type.lower(), "expr": values[2]}
    def output_item(self, values): return {"type": "output", "expr": values[1]}
    def bool_literal(self, values): return {"type": "bool", "value": values[0].type == "TRUE"}
    def int_literal(self, token): return int(token[0])
    def float_literal(self, token): return float(token[0])
    def string_literal(self, token): return token[0][1:-1]
    def IDENT(self, token): return str(token)
    def expr(self, values):
        if len(values) == 1: return values[0]
        left, tail = values[0], values[1]
        return {"type": "binop", "op": tail["op"], "left": left, "right": tail["expr"]}
    def expr_binop_tail(self, values): return {"op": values[0], "expr": values[1]}
    def bin_op(self, token): return str(token[0])
    def set_literal(self, values): return {"type": "set", "elements": values}
    def array_literal(self, values): return {"type": "array", "elements": values}
    def if_then_else_expr(self, values):
        expr = {"type": "if", "cond": values[1], "then": values[3]}
        rest, i = values[4:], 0
        while i < len(rest):
            if rest[i].type == "ELSEIF":
                expr.setdefault("elif",[]).append({"cond": rest[i+1], "then": rest[i+3]})
                i += 4
            elif rest[i].type == "ELSE":
                expr["else"] = rest[i+1]
                i += 2
            else: i += 1
        return expr
    def call_expr(self, values): return {"type": "call", "name": values[0], "args": values[2::2]}
    def base_ti_expr(self, values):
        if len(values) == 4 and isinstance(values[0], Token) and values[0].type == 'VAR' and isinstance(values[2], Token) and values[2].type == 'DOTS':
            return {"type": "var_range", "lo": values[1], "hi": values[3]}
        if len(values) == 1: return values[0]
        return {"type": "base_ti_expr", "values": values}

def build_parser():
    return Lark(MINIZINC_GRAMMAR, start="start", parser="lalr", transformer=MiniZincTransformer())

def parse_model(text: str):
    parser = build_parser()
    tree = parser.parse(text)
    return tree.children[0] if tree.data == 'start' else tree

def ast_to_json_serializable(node):
    if isinstance(node, Tree):
        return {"data": node.data, "children": [ast_to_json_serializable(child) for child in node.children]}
    if isinstance(node, Token):
        return str(node)
    if isinstance(node, dict):
        return {k: ast_to_json_serializable(v) for k, v in node.items()}
    if isinstance(node, list):
        return [ast_to_json_serializable(v) for v in node]
    return node

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Parse MiniZinc subset models with Lark")
    parser.add_argument("model_file", help="Path to a MiniZinc model file")
    args = parser.parse_args()

    with open(args.model_file, "r", encoding="utf-8") as f:
        text = f.read()

    ast = parse_model(text)
    import json
    print(json.dumps(ast, indent=2))
