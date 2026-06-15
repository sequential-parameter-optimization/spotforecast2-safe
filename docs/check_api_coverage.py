#!/usr/bin/env python
"""docs/check_api_coverage.py — fail if any public API symbol is undocumented.

Why this exists
---------------
The docstring-quality agents guarantee that every public symbol *has* a
Google-style docstring with a live ``{python}`` Examples block.  They do **not**
check whether the symbol is actually *listed in ``_quarto.yml`` and rendered on
the site*.  The "API Reference In Sync" CI job only catches *drift* between the
generated and committed reference — it cannot flag a symbol that was never
included, and inherited members never appear at all unless
``options.include_inherited`` is set.  So a fully-documented public method (e.g.
``MultiTask.build_exogenous_features``, inherited from ``BaseTask``) could be
missing from the website with every existing check green.

This script closes that gap.  It compares the **rendered** API (quartodoc's
``objects.json`` inventory) against the **expected** public API (discovered by
dynamic inspection of the real runtime objects) and fails on any gap.

Two checks
----------
* **Member coverage** — for every documented class, every public method /
  classmethod / staticmethod / property *defined within the spot\\* ecosystem*
  (i.e. not inherited from a third-party base such as ``object`` or sklearn)
  must have a rendered entry.  This is what catches missing inherited methods.
* **Listing coverage** — for every module that ``_quarto.yml`` documents by
  individual symbols, every public class / function *defined in that module*
  must be listed (or otherwise rendered).  This catches "added a new public
  function but forgot to add it to ``_quarto.yml``".

Run AFTER ``docs/quartodoc_build.py`` (which writes ``objects.json``):

    uv run python docs/quartodoc_build.py && uv run quartodoc interlinks
    uv run python docs/check_api_coverage.py

Exit code 0 = clean, 1 = coverage gaps (printed), 2 = setup error.  Deterministic,
no network.
"""
from __future__ import annotations

import importlib
import inspect
import json
import sys
from pathlib import Path

import yaml

# ── Configuration (per repo) ──────────────────────────────────────────────
PACKAGE = "spotforecast2_safe"

# A member/symbol counts as "public API we own" only when the class or function
# that DEFINES it lives in one of these top-level packages.  This excludes
# members inherited from third-party bases (object, sklearn, pydantic, ...),
# which quartodoc does not render either — so requiring them would be a false
# positive.  spotforecast2-safe is the base package, so its own prefix suffices.
ECOSYSTEM_PREFIXES = ("spotforecast2_safe",)

REPO_ROOT = Path(__file__).resolve().parent.parent
QUARTO_YML = REPO_ROOT / "_quarto.yml"
OBJECTS_JSON = REPO_ROOT / "objects.json"

# Intentionally-undocumented public symbols: fully-qualified rendered name ->
# reason.  Keep short and justified; prefer documenting over skipping.  Mirrors
# the justified-skip discipline of the docstring-example agents.
SKIP: dict[str, str] = {}

# Cross-package base classes (``module.Qualname``) whose inherited members are
# documented on their OWN canonical API page in a sibling package's site and
# which quartodoc/griffe cannot render onto subclass pages here.  A member
# inherited from one of these owners is therefore "documented" already.
# spotforecast2-safe is the base package and owns its forecaster bases locally,
# so nothing needs to be deferred elsewhere.  See ADR
# 2026-06-15-api-coverage-gate.md.
DOCUMENTED_ELSEWHERE: dict[str, str] = {}


# ── Helpers ────────────────────────────────────────────────────────────────
def load_rendered() -> dict:
    """Return the quartodoc inventory split by role.

    Returns a dict with ``all`` (set of every rendered dotted name),
    ``classes`` and ``modules`` (sets filtered by role).
    """
    if not OBJECTS_JSON.exists():
        sys.exit(
            f"[api-coverage] {OBJECTS_JSON} not found — run "
            "`uv run python docs/quartodoc_build.py` first."
        )
    data = json.loads(OBJECTS_JSON.read_text(encoding="utf-8"))
    items = data["items"]
    return {
        "all": {it["name"] for it in items},
        "classes": {it["name"] for it in items if it["role"] == "class"},
        "modules": {it["name"] for it in items if it["role"] == "module"},
    }


def resolve(dotted: str):
    """Resolve a dotted path (module / class / function) to its live object."""
    parts = dotted.split(".")
    for i in range(len(parts), 0, -1):
        try:
            obj = importlib.import_module(".".join(parts[:i]))
        except ImportError:
            continue
        for attr in parts[i:]:
            obj = getattr(obj, attr)
        return obj
    raise ImportError(f"cannot resolve {dotted!r}")


def _member_kind(value) -> str | None:
    """Classify a class-level value as a documented behavioural member, or None."""
    if isinstance(value, (staticmethod, classmethod)):
        return value.__class__.__name__
    if isinstance(value, property):
        return "property"
    if inspect.isfunction(value):
        return "method"
    return None  # data attributes are out of scope for this gate


def ecosystem_public_members(cls) -> dict[str, tuple[str, str]]:
    """Public behavioural members of ``cls`` defined within the ecosystem.

    Walks the MRO and collects public (no leading underscore) methods and
    properties, skipping classes defined outside ``ECOSYSTEM_PREFIXES`` (object,
    sklearn bases, ...).  Returns ``{member_name: (display, owner_fqn)}`` where
    ``owner_fqn`` is the ``module.Qualname`` of the class that defines it.
    """
    found: dict[str, tuple[str, str]] = {}
    for klass in cls.__mro__:
        if not getattr(klass, "__module__", "").startswith(ECOSYSTEM_PREFIXES):
            continue
        owner = f"{klass.__module__}.{klass.__qualname__}"
        for name, value in vars(klass).items():
            if name.startswith("_") or name in found:
                continue
            kind = _member_kind(value)
            if kind is not None:
                found[name] = (f"{kind} in {owner}", owner)
    return found


def quarto_contents(path: Path) -> list[str]:
    """All leaf dotted paths listed under ``quartodoc.sections[*].contents``."""
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    leaves: list[str] = []

    def walk(node):
        if isinstance(node, str):
            leaves.append(node)
        elif isinstance(node, dict):
            if "name" in node and isinstance(node["name"], str):
                leaves.append(node["name"])
            for key in ("contents", "package"):
                if key in node:
                    walk(node[key])
        elif isinstance(node, list):
            for item in node:
                walk(item)

    for section in cfg.get("quartodoc", {}).get("sections", []):
        walk(section.get("contents", []))
    return leaves


# ── Checks ───────────────────────────────────────────────────────────────
def check_member_coverage(rendered: dict) -> list[str]:
    """Every ecosystem-public member of a documented class must be rendered.

    quartodoc records a class under both the alias path used in ``_quarto.yml``
    (which carries the member anchors) and its canonical module path (which may
    not).  So members are grouped by the resolved class *object*: a member is a
    real gap only when it is absent under *every* alias path of that class.
    """
    # class object -> {alias_path: missing member names under that path}
    per_class: dict[type, dict[str, set[str]]] = {}
    canonical: dict[type, str] = {}
    for cls_path in sorted(rendered["classes"]):
        try:
            cls = resolve(cls_path)
        except Exception as exc:  # noqa: BLE001 — report, don't crash
            return [f"{cls_path}: cannot import to verify members ({exc})"]
        if not inspect.isclass(cls):
            continue
        members = ecosystem_public_members(cls)
        missing = {
            name
            for name, (_disp, owner) in members.items()
            if f"{cls_path}.{name}" not in rendered["all"]
            and f"{cls_path}.{name}" not in SKIP
            and owner not in DOCUMENTED_ELSEWHERE
        }
        per_class.setdefault(cls, {})[cls_path] = missing
        # Prefer the most-qualified path for the human-readable report.
        canonical[cls] = max(canonical.get(cls, ""), cls_path, key=len)

    gaps: list[str] = []
    for cls, by_path in per_class.items():
        # Missing under ALL alias paths == genuinely unrendered.
        real_missing = set.intersection(*by_path.values()) if by_path else set()
        origins = ecosystem_public_members(cls)
        for name in sorted(real_missing):
            gaps.append(f"{canonical[cls]}.{name}  ({origins.get(name, ('',''))[0]})")
    return gaps


def check_listing_coverage(rendered: dict) -> list[str]:
    """Public classes/functions in partially-listed modules must be listed."""
    listed_by_module: dict[str, set[str]] = {}
    wholesale: set[str] = set()
    for leaf in quarto_contents(QUARTO_YML):
        # Content paths in _quarto.yml are relative to `package:`.
        dotted = leaf if leaf.startswith(f"{PACKAGE}.") else f"{PACKAGE}.{leaf}"
        try:
            importlib.import_module(dotted)
            wholesale.add(dotted)  # whole module documented -> all members included
            continue
        except ImportError:
            # Not a module path (likely a module.symbol leaf); fall back to
            # symbol-level listing checks below.
            ...
        module, _, symbol = dotted.rpartition(".")
        if module:
            listed_by_module.setdefault(module, set()).add(symbol)

    gaps: list[str] = []
    for module, listed in sorted(listed_by_module.items()):
        if module in wholesale:
            continue
        try:
            mod = importlib.import_module(module)
        except ImportError as exc:
            gaps.append(f"{module}: cannot import to verify listing ({exc})")
            continue
        for name, value in vars(mod).items():
            if name.startswith("_"):
                continue
            if not (inspect.isclass(value) or inspect.isfunction(value)):
                continue
            if getattr(value, "__module__", None) != module:
                continue  # imported/re-exported here, defined elsewhere
            fq = f"{module}.{name}"
            if name in listed or fq in rendered["all"] or fq in SKIP:
                continue
            kind = "class" if inspect.isclass(value) else "function"
            gaps.append(f"{fq}  (public {kind} not listed in _quarto.yml)")
    return gaps


def main() -> None:
    rendered = load_rendered()
    member_gaps = check_member_coverage(rendered)
    listing_gaps = check_listing_coverage(rendered)

    if not member_gaps and not listing_gaps:
        print(f"[api-coverage] OK — every public {PACKAGE} symbol is documented.")
        sys.exit(0)

    if member_gaps:
        print("[api-coverage] FAIL — documented classes missing public members:")
        for gap in member_gaps:
            print(f"  - {gap}")
    if listing_gaps:
        print("[api-coverage] FAIL — public symbols missing from _quarto.yml:")
        for gap in listing_gaps:
            print(f"  - {gap}")
    print(
        f"\n[api-coverage] {len(member_gaps) + len(listing_gaps)} gap(s). "
        "Document the symbol (add to _quarto.yml / set options.include_inherited) "
        "or add a justified entry to SKIP."
    )
    sys.exit(1)


if __name__ == "__main__":
    main()
