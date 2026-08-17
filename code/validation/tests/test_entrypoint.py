"""The capsule entry point's pre-flight checks.

Only the parts that can be exercised without mounted data: the code-version gate, which
decides whether a seven-hour run starts at all, and the paths it hands downstream. The
notebook execution itself is not tested here -- it needs the asset.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from harness import REPO, check, fails, summary

# Imported in a subprocess rather than here: the environment is the thing under test, and
# a module read once at import would freeze whichever value happened to be set.


def resolve(env_overrides, cwd=None):
    """Call `resolve_code_version` in a subprocess with a controlled environment."""
    env = {k: v for k, v in os.environ.items() if k != "SWDB_CODE_VERSION"}
    env.update(env_overrides)
    src = (
        "import sys; sys.path.insert(0, r'%s')\n"
        "import run_stimulus_metrics as m\n"
        "print('VERSION=' + m.resolve_code_version())\n" % (REPO / "code")
    )
    return subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                          env=env, cwd=str(cwd) if cwd else None)


print("[1] an explicit version is taken verbatim")
proc = resolve({"SWDB_CODE_VERSION": "deadbeefcafe"})
check("exits cleanly", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
check("uses the value given", "VERSION=deadbeefcafe" in proc.stdout, proc.stdout.strip())

print("\n[2] whitespace-only is not a version")
proc = resolve({"SWDB_CODE_VERSION": "   "})
# Falls through to git, which succeeds in this checkout -- so assert it did not take the
# blank, rather than asserting failure.
check("blank does not become the version", "VERSION=   " not in proc.stdout,
      proc.stdout.strip()[:80])

print("\n[3] unset falls back to the checkout")
proc = resolve({})
check("resolves from git here", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
sha = proc.stdout.strip().replace("VERSION=", "")
check("looks like a full sha", len(sha) == 40 and all(c in "0123456789abcdef" for c in sha),
      sha)

print("\n[4] no git and no variable is a hard stop, before any work")
# A directory that is not a checkout stands in for the reproducible run, where code/ is
# copied without .git. `git -C` on a non-repo fails, which is the case that shipped a
# null commit_hash.
sandbox = Path(tempfile.mkdtemp(prefix="entry_"))
src = (
    "import sys; sys.path.insert(0, r'%s')\n"
    "import run_stimulus_metrics as m\n"
    "m.CODE = __import__('pathlib').Path(r'%s') / 'code'\n"
    "print('VERSION=' + m.resolve_code_version())\n" % (REPO / "code", sandbox)
)
env = {k: v for k, v in os.environ.items() if k != "SWDB_CODE_VERSION"}
proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
check("does not exit 0", proc.returncode != 0, str(proc.returncode))
check("never prints a version", "VERSION=" not in proc.stdout, proc.stdout.strip()[:80])
msg = (proc.stdout or "") + (proc.stderr or "")
check("says what to set", "SWDB_CODE_VERSION" in msg)
check("says why it matters", "commit_hash" in msg)
check("says when it checked", "seven hours" in msg)

print("\n[5] the version reaches metadata.py through the environment")
full = (REPO / "code" / "run_stimulus_metrics.py").read_text(encoding="utf-8")
# Order is asserted over the body of `run()` only. Comparing against the whole file would
# pick up the docstring, which names the same steps and would make the check pass or fail
# on prose rather than on what executes.
text = full[full.index("def run()"):]
check("run() stamps SWDB_CODE_VERSION into the child environment",
      "SWDB_CODE_VERSION=version" in text)
check("resolved before the processing notebook runs",
      text.index("resolve_code_version()") < text.index("execute_notebook(PROCESSING_NB"))
check("validation runs before metadata, so processing.json can record it",
      text.index("execute_notebook(VALIDATION_NB")
      < text.index('CODE / "metadata.py"'))
check("metadata gets the validation directory", "--validation-dir" in text)

summary()
