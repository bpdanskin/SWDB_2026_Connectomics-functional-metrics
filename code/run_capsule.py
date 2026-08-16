"""Capsule entry point: build the V1DD stimulus-metrics asset and describe it.

Three steps, in order:

1. Execute ``V1DD Stimulus Metrics.ipynb`` with ``SWDB_OUTPUT_TARGET=results``, so its
   tables land in ``/results/<asset-name>_<stamp>/`` and CodeOcean captures them.
2. Run ``metadata.py`` against that directory, writing the three AIND sidecars beside the
   tables.
3. Execute the validation notebook, whose artifacts go to ``/scratch`` and are therefore
   *not* part of the asset — checking the asset is not part of it.

The executed notebooks are written to ``/results`` alongside the asset directory rather
than inside it: they are a record of the run, not data.

Interactive use is unaffected. The notebooks default to ``/scratch`` and only write to
``/results`` because this script sets the environment variable, so opening one in
JupyterLab cannot accidentally produce something that looks like a captured asset.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SWDB_RESULTS_DIR", "/results"))
ASSET_PREFIX = "409828_V1DD_stimulus_metrics"
INPUT_ASSET = Path(os.environ.get("SWDB_INPUT_ASSET", "/data/409828_V1DD_Filtered"))

PROCESSING_NB = CODE / "supplement" / "V1DD Stimulus Metrics.ipynb"
VALIDATION_NB = CODE / "validation" / "V1DD Stimulus Metrics Validation.ipynb"


def execute_notebook(path: Path, env: dict) -> None:
    """Run a notebook in place, writing the executed copy to /results."""
    print(f"\n=== executing {path.name}", flush=True)
    subprocess.run(
        [sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute",
         "--ExecutePreprocessor.timeout=-1", "--output-dir", str(RESULTS), str(path)],
        check=True, env=env)


def latest_asset() -> Path:
    """The run directory the processing notebook just created."""
    sys.path.insert(0, str(CODE / "utils"))
    from provenance import latest_run                              # noqa: E402

    found = latest_run(str(RESULTS), ASSET_PREFIX)
    if found is None:
        raise FileNotFoundError(
            f"no {ASSET_PREFIX}_<stamp> directory under {RESULTS} after running the "
            "processing notebook -- it did not produce an asset."
        )
    return Path(found)


def run() -> int:
    RESULTS.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, SWDB_OUTPUT_TARGET="results")

    execute_notebook(PROCESSING_NB, env)

    asset_dir = latest_asset()
    print(f"\n=== asset: {asset_dir}", flush=True)

    print("\n=== writing AIND metadata sidecars", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(CODE / "metadata.py"),
         "--asset-dir", str(asset_dir),
         "--input-asset", str(INPUT_ASSET),
         "--results-dir", str(RESULTS)],
        check=True, env=env)

    # Validation last, and non-fatal: a failed check should surface in its artifacts and
    # in the executed notebook rather than destroy an asset that took hours to build.
    # The asset is already written and described by this point.
    try:
        execute_notebook(VALIDATION_NB, env)
    except subprocess.CalledProcessError as exc:
        print(f"\n!! validation notebook failed ({exc.returncode}); the asset is still "
              f"in {asset_dir}. Read its output above and in /scratch.", flush=True)

    print(f"\n=== done. asset in {asset_dir}", flush=True)
    for f in sorted(asset_dir.iterdir()):
        print(f"    {f.name:<52} {f.stat().st_size / 1024:>9.1f} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
