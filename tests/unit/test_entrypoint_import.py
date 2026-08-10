"""Importing the deployment entrypoint must construct nothing.

`studio.py` used to end with `graph = build_studio_graph()`. Importing it read
settings, wrote to os.environ through `configure_tracing`, constructed every
model client and opened a BigQuery client — so missing credentials surfaced as
an import error, and the schema snapshot was pinned for the life of the
process.

A subprocess, because the assertion is about what import does and every other
test in this suite has already imported the module.
"""

import subprocess
import sys

CHECK = """
import sys
import retail_agent.agent.studio as studio

assert not hasattr(studio, "graph"), \\
    "studio still exports a module-level graph built at import"
assert studio._process_deps.cache_info().misses == 0, \\
    "dependencies were constructed at import"
assert "langchain_google_genai" not in sys.modules, \\
    "a provider client was constructed at import"
assert "retail_agent.datasources.bigquery" in sys.modules, \\
    "BigQuerySource should still be imported, just not called"
print("clean")
"""


def test_importing_the_entrypoint_constructs_nothing():
    result = subprocess.run(
        [sys.executable, "-c", CHECK],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "clean" in result.stdout
