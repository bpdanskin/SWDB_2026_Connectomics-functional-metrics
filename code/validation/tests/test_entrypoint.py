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

from harness import REPO, check, fails, skip, summary

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
# Only where there *is* a checkout. A reproducible run copies code/ without .git, so in
# the capsule these two assert that a fallback works in the exact environment it cannot
# work in -- and they duly failed there, printing "unit tests failed" over a clean asset.
# The condition under test is the repository, so ask git rather than looking for a
# directory: a worktree or a submodule has no `.git` directory and is still a checkout.
in_checkout = subprocess.run(
    ["git", "-C", str(REPO), "rev-parse", "--git-dir"],
    capture_output=True, text=True).returncode == 0

if not in_checkout:
    skip("resolves from git here", "no git repository -- reproducible run, code/ has no .git")
    skip("looks like a full sha", "no git repository -- nothing for git to resolve")
else:
    proc = resolve({})
    check("resolves from git here", proc.returncode == 0, (proc.stderr or "").strip()[-200:])
    sha = proc.stdout.strip().replace("VERSION=", "")
    check("looks like a full sha",
          len(sha) == 40 and all(c in "0123456789abcdef" for c in sha), sha)

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

print("\n[6] a malformed version is rejected, not stamped into the asset")
# The value the 2026-09-01 asset actually shipped. It is neither empty nor whitespace, so
# the original guard passed it and processing.json recorded `"= 17cacea..."` as a version.
proc = resolve({"SWDB_CODE_VERSION": "= 17cacea5a61c6b596324d6911a879b15f3ed98c4"})
check("the shipped malformed value does not exit 0", proc.returncode != 0, str(proc.returncode))
msg = (proc.stdout or "") + (proc.stderr or "")
check("never prints it as a version", "VERSION== 17cacea" not in proc.stdout,
      proc.stdout.strip()[:80])
check("names the Dockerfile ENV trap", "ENV SWDB_CODE_VERSION=" in msg)
proc = resolve({"SWDB_CODE_VERSION": "v2.1-release"})
check("a non-hex label is not a commit", proc.returncode != 0, str(proc.returncode))
proc = resolve({"SWDB_CODE_VERSION": "abc123"})
check("six hex is too short to name a commit", proc.returncode != 0, str(proc.returncode))
proc = resolve({"SWDB_CODE_VERSION": "0DF1CF7228401B8024EDF8378E985E801379F31A"})
check("an upper-case full sha is accepted", proc.returncode == 0,
      (proc.stderr or "").strip()[-200:])

print("\n[7] the Dockerfile pins a version the guard would accept")
# The malformed line passed every test last time because no test read the Dockerfile --
# the check ran in a different shape than production, which is how all three provenance
# defects got out. Read the line the container actually sets.
docker = (REPO / "environment" / "Dockerfile").read_text(encoding="utf-8")
env_lines = [ln.strip() for ln in docker.splitlines()
             if ln.strip().startswith("ENV") and "SWDB_CODE_VERSION" in ln]
check("the Dockerfile sets SWDB_CODE_VERSION exactly once", len(env_lines) == 1,
      str(env_lines))
line = env_lines[0] if env_lines else ""
check("uses ENV key=value, not the legacy spaced form",
      line.startswith("ENV SWDB_CODE_VERSION="), line)
pinned = line.split("=", 1)[1].strip() if "=" in line else ""
check("pins something shaped like a commit",
      7 <= len(pinned) <= 40 and all(c in "0123456789abcdefABCDEF" for c in pinned), pinned)
# Not asserted equal to HEAD: the sha is bumped by hand and the commit that bumps it
# necessarily comes after the value it writes, so equality would be red every commit.
# Existence catches the cases that matter -- a typo, or a sha from another repository.
if not in_checkout:
    skip("the pinned commit exists", "no git repository -- cannot resolve a commit here")
else:
    found = subprocess.run(["git", "-C", str(REPO), "cat-file", "-e", f"{pinned}^{{commit}}"],
                           capture_output=True, text=True)
    check("the pinned commit exists", found.returncode == 0, pinned)

print("\n[8] provenance.git_sha reads the same variable metadata.py does")
# Two sidecars in one asset disagreeing about the same fact: the 2026-09-01 run shipped
# stimulus_metrics_provenance.json with git_sha: null beside a processing.json carrying a
# version, because git_sha() only ever shelled out to git.
src = (
    "import sys; sys.path.insert(0, r'%s')\n"
    "import provenance\n"
    "print('SHA=' + str(provenance.git_sha(repo=r'%s')))\n" % (REPO / "code" / "utils", "/")
)
env = {k: v for k, v in os.environ.items() if k != "SWDB_CODE_VERSION"}
# repo='/' so git cannot answer even in a checkout: the variable must be what carries it.
proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                      env=dict(env, SWDB_CODE_VERSION="deadbeefcafe"))
check("takes SWDB_CODE_VERSION when git cannot answer",
      "SHA=deadbeefcafe" in proc.stdout, proc.stdout.strip()[:80])
proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True, env=env)
check("still None with neither git nor the variable", "SHA=None" in proc.stdout,
      proc.stdout.strip()[:80])
proc = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                      env=dict(env, SWDB_CODE_VERSION="   "))
check("whitespace does not become the sha", "SHA=   " not in proc.stdout,
      proc.stdout.strip()[:80])

summary()
