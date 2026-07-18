from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path


def _function_hashes(path: Path) -> dict[str, str]:
    tree = ast.parse(path.read_text(encoding='utf-8'))
    return {
        node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def test_moved_core_function_bodies_are_unchanged():
    project = Path(__file__).resolve().parents[1]
    expected = json.loads((project / 'tests' / 'logic_hashes.json').read_text())
    source = project / 'src' / 'cori_analysis'
    for module, expected_hashes in expected.items():
        actual = _function_hashes(source / module)
        for name, expected_hash in expected_hashes.items():
            assert actual[name] == expected_hash, f'Logic changed in {module}:{name}'
