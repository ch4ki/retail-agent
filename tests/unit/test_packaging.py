"""What the project declares it depends on.

`langchain-core` is imported in eleven modules and `langsmith` in
`obs/tracing.py`, but both arrived transitively through `langchain`. A
transitive edge is not a promise: a future `langchain` release can drop it and
the failure lands at import time, in production, on a machine nobody is
watching.
"""

import re
import tomllib
from pathlib import Path

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"
LANGGRAPH_JSON = Path(__file__).resolve().parents[2] / "langgraph.json"


def _dependencies() -> dict[str, str]:
    """Package name -> the full requirement string it was declared with."""
    data = tomllib.loads(PYPROJECT.read_text())
    found = {}
    for raw in data["project"]["dependencies"]:
        spec = raw.split("#")[0].strip()
        name = re.split(r"[><=!\[]", spec, maxsplit=1)[0].strip()
        found[name] = spec
    return found


def test_directly_imported_langchain_packages_are_declared():
    declared = _dependencies()

    for package in ("langchain", "langchain-core", "langgraph", "langsmith"):
        assert package in declared, (
            f"{package} is imported directly but not declared; it is being "
            f"resolved transitively"
        )


def test_semver_langchain_packages_carry_an_upper_bound():
    """These three follow strict semver, so a 2.0 can break the lock silently."""
    declared = _dependencies()

    for package in ("langchain", "langchain-core", "langgraph"):
        assert "<2.0" in declared[package], (
            f"{package} is declared as {declared[package]!r} with no upper bound"
        )


def test_provider_packages_stay_unbounded():
    """Dedicated integration packages ship compatibility fixes alongside core
    releases. Pinning them is how a project ends up unable to upgrade at all."""
    declared = _dependencies()

    for package in ("langchain-openai", "langchain-google-genai", "langchain-ollama"):
        assert "<" not in declared[package], (
            f"{package} is pinned to {declared[package]!r}; integration packages "
            f"should track latest"
        )


def test_langgraph_declares_a_python_version():
    """Absent, it defaults to 3.11 — below this project's floor, so every
    `langgraph build`, `up` and `deploy` fails dependency resolution."""
    import json

    config = json.loads(LANGGRAPH_JSON.read_text())

    assert "python_version" in config


def test_langgraph_python_version_satisfies_requires_python():
    import json

    config = json.loads(LANGGRAPH_JSON.read_text())
    requires = tomllib.loads(PYPROJECT.read_text())["project"]["requires-python"]

    declared = tuple(int(part) for part in config["python_version"].split("."))
    floor = tuple(int(part) for part in requires.removeprefix(">=").strip().split("."))

    assert declared >= floor, (
        f"langgraph.json builds on {config['python_version']} but pyproject "
        f"requires {requires}"
    )
