import json
from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")

NOTEBOOK = Path(__file__).resolve().parents[1] / "notebooks" / "CCBL_StuffPlus_Model.ipynb"


@pytest.fixture
def model_namespace(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("CCBL_TRACKMAN_FILE", raising=False)
    namespace = {"__name__": "notebook_test"}
    notebook = json.loads(NOTEBOOK.read_text())
    for cell in notebook["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] == "code" and not source.lstrip().startswith(("if ", "print(")):
            exec(compile(source, str(NOTEBOOK), "exec"), namespace)
    return namespace
