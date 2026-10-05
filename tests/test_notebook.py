import ast
import json

import nbformat

from conftest import NOTEBOOK


def test_notebook_schema_syntax_and_clean_outputs():
    notebook = nbformat.read(NOTEBOOK, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            compile(cell.source, str(NOTEBOOK), "exec")
            assert cell.execution_count is None
            assert not cell.outputs


def test_default_configuration_is_portable():
    notebook = json.loads(NOTEBOOK.read_text())
    source = "\n".join("".join(c.get("source", [])) for c in notebook["cells"])
    assignments = {}
    for node in ast.parse(source).body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            assignments.setdefault(node.target.id, node.value)
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name):
                    assignments.setdefault(target.id, node.value)
    assert ast.literal_eval(assignments["DATA_FILE_OVERRIDE"]) is None
    assert ast.literal_eval(assignments["G_RUN_DIRECTORY"]) is None
    assert ast.literal_eval(assignments["RUN_MODEL"]) is True
