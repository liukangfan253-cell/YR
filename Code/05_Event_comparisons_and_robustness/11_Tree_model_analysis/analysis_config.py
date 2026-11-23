"""Read saved SHAP script configuration without executing training code."""

import ast
from pathlib import Path


def read_analysis_config(path, project_root):
    def read_value(node):
        # The training entries use str(ROOT / 'relative/path') for file I/O.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == 'str' and len(node.args) == 1 and not node.keywords):
            relative = node.args[0]
            if (isinstance(relative, ast.BinOp) and isinstance(relative.op, ast.Div)
                    and isinstance(relative.left, ast.Name) and relative.left.id == 'ROOT'):
                return str(Path(project_root) / ast.literal_eval(relative.right))
        return ast.literal_eval(node)

    values = {}
    for assignment in ast.parse(path.read_text(encoding='utf-8-sig')).body:
        if not isinstance(assignment, ast.Assign):
            continue
        try:
            value = read_value(assignment.value)
        except (TypeError, ValueError, SyntaxError):
            continue
        for target in assignment.targets:
            if isinstance(target, ast.Name):
                values[target.id] = value
    return values
