"""Capsule entry point: build the V1DD stimulus-metrics asset and describe it.

Refuses to start unless the code version is known — see ``resolve_code_version``. Then
three steps, in order:

1. Execute ``V1DD Stimulus Metrics.ipynb`` with ``SWDB_OUTPUT_TARGET=results``, so its
   tables land in ``/results/<asset-name>_<stamp>/`` and CodeOcean captures them.
2. Execute the validation notebook, whose artifacts go to ``/scratch`` and are therefore
   *not* part of the asset — checking the asset is not part of it.
3. Run ``metadata.py`` against that directory, writing the three AIND sidecars beside the
   tables. Last, so ``processing.json`` can record that validation ran and how it went.

The executed notebooks are written to ``/results`` alongside the asset directory rather
than inside it: they are a record of the run, not data.

Interactive use is unaffected. The notebooks default to ``/scratch`` and only write to
``/results`` because this script sets the environment variable, so opening one in
JupyterLab cannot accidentally produce something that looks like a captured asset.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent
RESULTS = Path(os.environ.get("SWDB_RESULTS_DIR", "/results"))
ASSET_PREFIX = "409828_V1DD_stimulus_metrics"
INPUT_ASSET = Path(os.environ.get("SWDB_INPUT_ASSET", "/data/409828_V1DD_Filtered"))
#: Where the validation notebook writes. Outside the asset on purpose -- checking the
#: asset is not part of it -- but `metadata.py` reads it to record that checking happened.
VALIDATION_DIR = Path(os.environ.get("SWDB_VALIDATION_DIR",
                                     "/scratch/v1dd_stimulus_metrics_validation"))

#: What a usable code version looks like: a git object name, short or full. The point of
#: `SWDB_CODE_VERSION` is that the value it stamps into the asset can be checked out
#: again, so a value that cannot name a commit is not a version. Deliberately not a hard
#: 40, because a short sha is a legitimate thing to set by hand.
_SHA_RE = re.compile(r"^[0-9a-fA-F]{7,40}$")

PROCESSING_NB = CODE / "supplement" / "V1DD Stimulus Metrics.ipynb"
VALIDATION_NB = CODE / "validation" / "V1DD Stimulus Metrics Validation.ipynb"


def resolve_code_version() -> str:
    """The commit this run is built from, or raise before any work starts.

    Checked *first*, ahead of the seven-hour processing notebook, because the alternative
    is finding out at the end. `metadata.py` degrades gracefully when the version is
    unknown — it writes `commit_hash: null` and aind-data-schema emits a warning that
    scrolls past in a log thousands of lines long. That is right for a library and wrong
    for a published asset, so the entry point is strict where the library is lenient.

    A reproducible run copies `code/` without `.git`, so `git rev-parse` only answers in a
    checkout. There is no bypass flag on purpose: setting `SWDB_CODE_VERSION` to a value
    you choose is already the escape hatch, and it is an honest one — whatever you set is
    what the asset claims. A second "skip the check" switch would just restore `null`.

    The value must still *look* like a commit (7-40 hex). Being set is not the same as
    being usable: a malformed `ENV` line in the Dockerfile set it to `"= <sha>"`, which is
    neither empty nor whitespace, so it passed the old guard and shipped.
    """
    env = os.environ.get("SWDB_CODE_VERSION", "").strip()
    if env:
        if not _SHA_RE.match(env):
            raise SystemExit(
                f"\n!! refusing to start: SWDB_CODE_VERSION={env!r} is not a commit.\n"
                "\n"
                "   Expected 7-40 hex characters. This exact check exists because the\n"
                "   2026-09-01 asset shipped processing.json with\n"
                "\n"
                '       "version": "= 17cacea5a61c6b596324d6911a879b15f3ed98c4"\n'
                "\n"
                "   Docker\'s legacy `ENV <key> <value>` form takes everything after the\n"
                "   first space as the value, so\n"
                "\n"
                "       ENV SWDB_CODE_VERSION = <sha>      # wrong: value is '= <sha>'\n"
                "       ENV SWDB_CODE_VERSION=<sha>        # right\n"
                "\n"
                "   The old guard rejected empty and whitespace but not malformed, so the\n"
                "   bad value passed and the asset claims a version nothing can check out.\n"
            )
        return env
    try:
        out = subprocess.run(["git", "-C", str(CODE.parent), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10, check=True)
        sha = out.stdout.strip()
        if sha:
            return sha
    except Exception:                                             # noqa: BLE001
        pass
    raise SystemExit(
        "\n!! refusing to start: the code version is unknown.\n"
        "\n"
        "   This capsule has no git checkout -- a reproducible run copies code/ without\n"
        "   .git -- so the commit cannot be derived. Without it the published asset ships\n"
        "   processing.json with commit_hash: null and nothing ties it to the code that\n"
        "   produced it.\n"
        "\n"
        "   Set SWDB_CODE_VERSION in the capsule environment before launching:\n"
        "\n"
        "       git -C <fork> rev-parse HEAD\n"
        "\n"
        "   Checked here rather than at the end so you lose a second, not seven hours.\n"
    )


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

    # Resolved once and passed down, so the entry point and metadata.py cannot disagree
    # about which commit this is: metadata.py reads the variable before trying git.
    version = resolve_code_version()
    print(f"=== code version {version}", flush=True)
    env = dict(os.environ, SWDB_OUTPUT_TARGET="results", SWDB_CODE_VERSION=version)

    execute_notebook(PROCESSING_NB, env)

    asset_dir = latest_asset()
    print(f"\n=== asset: {asset_dir}", flush=True)

    # Validation before metadata, and non-fatal. Non-fatal because a failed check should
    # surface in its artifacts rather than destroy an asset that took hours to build --
    # the tables are already written by this point. Before, because `processing.json`
    # records the validation run, and metadata written first can only ever describe a
    # validation that had not happened yet. The first full run shipped exactly that:
    # one process where there should have been two.
    try:
        execute_notebook(VALIDATION_NB, env)
    except subprocess.CalledProcessError as exc:
        print(f"\n!! validation notebook failed ({exc.returncode}); the asset is still "
              f"in {asset_dir}. Read its output above and in {VALIDATION_DIR}.",
              flush=True)

    print("\n=== writing AIND metadata sidecars", flush=True)
    subprocess.run(
        [sys.executable, "-u", str(CODE / "metadata.py"),
         "--asset-dir", str(asset_dir),
         "--input-asset", str(INPUT_ASSET),
         "--results-dir", str(RESULTS),
         "--validation-dir", str(VALIDATION_DIR)],
        check=True, env=env)

    print(f"\n=== done. asset in {asset_dir}", flush=True)
    for f in sorted(asset_dir.iterdir()):
        print(f"    {f.name:<52} {f.stat().st_size / 1024:>9.1f} KB", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(run())
