"""Check import hygiene (code-style §6).

Rules, all from §6:
  * imports are absolute (no `from .` relative imports);
  * the four blocks appear in order — `from __future__`, then third-party
    `import`, third-party `from`, first-party `import`, first-party `from`;
  * within each blank-line-separated block, imports rise in (logical) length;
  * the names inside `from X import a, b, c` rise in length too.

Imports are parsed with `ast`, so a parenthesised multi-line import is measured
by its logical single-line length (what it would be unwrapped) and still sorts
as the longest. First-party packages are detected from the project layout, so
nothing is hard-coded.

    python tools/pre_commit/check_imports.py FILE ...
"""

from __future__ import annotations

import ast
import argparse

from pathlib import Path
from typing import NamedTuple


class Imp(NamedTuple):
    line: int
    end: int
    logical: str
    root: str
    is_from: bool
    level: int
    names: list[str]


########################################
#             First-party              #
########################################


def first_party_roots(base_path: Path = Path(".")) -> set[str]:
    # ── A top-level name is first-party if it resolves to a local package or
    #    module on disk — detected, not hard-coded. ──
    roots: set[str] = set()
    for base in (base_path / "src", base_path):
        if not base.is_dir():
            continue
        for child in base.iterdir():
            if child.is_dir() and (child / "__init__.py").exists():
                roots.add(child.name)
            elif child.suffix == ".py":
                roots.add(child.stem)
    return roots


########################################
#               Parsing                #
########################################


def alias_str(alias: ast.alias) -> str:
    return alias.name + (f" as {alias.asname}" if alias.asname else "")


def parse_imports(source: str) -> list[Imp]:
    # ── Module-level imports, each reduced to its logical single-line form ──
    imports: list[Imp] = []
    for node in ast.parse(source).body:
        end = node.end_lineno or node.lineno
        if isinstance(node, ast.Import):
            names = [alias_str(a) for a in node.names]
            root = node.names[0].name.split(".")[0]
            logical = "import " + ", ".join(names)
            is_from, level = False, 0
        elif isinstance(node, ast.ImportFrom):
            names = [alias_str(a) for a in node.names]
            module = node.module or ""
            root = module.split(".")[0]
            prefix = "." * node.level
            logical = f"from {prefix}{module} import " + ", ".join(names)
            is_from, level = True, node.level
        else:
            continue
        imports.append(
            Imp(node.lineno, end, logical, root, is_from, level, names),
        )
    return imports


def blocks_of(imports: list[Imp]) -> list[list[Imp]]:
    # ── Group imports separated by a blank line into the same block ──
    blocks: list[list[Imp]] = []
    current: list[Imp] = []
    for imp in imports:
        if current and imp.line > current[-1].end + 1:
            blocks.append(current)
            current = []
        current.append(imp)
    if current:
        blocks.append(current)
    return blocks


def category(imp: Imp, first_party: set[str]) -> int:
    # ── 0 future · 1 import 3p · 2 from 3p · 3 import 1p · 4 from 1p ──
    if imp.root == "__future__":
        return 0
    fp = imp.root in first_party
    if imp.is_from:
        return 4 if fp else 2
    return 3 if fp else 1


########################################
#               Checking               #
########################################


def check_file(path: Path, first_party: set[str]) -> bool:
    try:
        imports = parse_imports(path.read_text())
    except SyntaxError:
        return False  # leave syntax errors to ruff / python
    bad = False

    # ── Absolute imports only: a leading dot is a relative import ──
    for imp in imports:
        if imp.level > 0:
            print(
                f"{path}:{imp.line}: relative import "
                f"(use an absolute import) (code-style §6)",
            )
            bad = True

    # ── Blocks in category order: future, 3p import/from, 1p import/from ──
    prev_cat = -1
    for imp in imports:
        cat = category(imp, first_party)
        if cat < prev_cat:
            print(
                f"{path}:{imp.line}: import out of block order "
                f"(code-style §6): {imp.logical!r}",
            )
            bad = True
        prev_cat = cat

    # ── Within each block, single-line imports rise in length ──
    for block in blocks_of(imports):
        for a, b in zip(block, block[1:], strict=False):
            if len(b.logical) < len(a.logical):
                print(
                    f"{path}:{b.line}: import ladder breaks "
                    f"({len(a.logical)} -> {len(b.logical)}) (code-style §6)",
                )
                bad = True

    # ── A parenthesised multi-line import is set off as its own block ──
    for block in blocks_of(imports):
        if len(block) > 1:
            for imp in block:
                if imp.end > imp.line:
                    print(
                        f"{path}:{imp.line}: multi-line import needs a "
                        f"blank line around it (code-style §6)",
                    )
                    bad = True

    # ── Names inside `from X import a, b, c` rise in length too ──
    for imp in imports:
        for x, y in zip(imp.names, imp.names[1:], strict=False):
            if len(y) < len(x):
                print(
                    f"{path}:{imp.line}: import names not laddered "
                    f"({x!r} > {y!r}) (code-style §6)",
                )
                bad = True
                break

    return bad


########################################
#             Entry point              #
########################################


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        description="Check import hygiene (code-style §6).",
    )
    p.add_argument("files", nargs="*", type=Path)
    args = p.parse_args(argv)

    first_party = first_party_roots()
    bad = False
    for path in args.files:
        bad = check_file(path, first_party) or bad
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
