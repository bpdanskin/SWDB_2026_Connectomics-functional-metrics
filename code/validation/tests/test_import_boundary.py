"""The pipeline must never import the machinery that checks it.

`code/utils` is the analysis pipeline; `code/validation` is how we know it is right. The
dependency runs one way only. Without something enforcing that, the boundary erodes in a
month: someone reaches for `checkpoint` inside a metric function, and now the pipeline
cannot run without the validation code, and the split has bought nothing.

This is a grep with a rationale, and it is cheap enough to keep.
"""
import ast

from harness import REPO, check, fails, summary

UTILS = REPO / "code" / "utils"
VALIDATION = REPO / "code" / "validation"

validation_modules = {p.stem for p in VALIDATION.glob("*.py")}
utils_modules = {p.stem for p in UTILS.glob("*.py")}
print(f"  utils:      {sorted(utils_modules)}")
print(f"  validation: {sorted(validation_modules)}")

print("\n[1] no module in code/utils imports one from code/validation")
offenders = []
for path in sorted(UTILS.glob("*.py")):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            names = [node.module] if node.module else []
        else:
            continue
        for name in names:
            if name and name.split(".")[0] in validation_modules:
                offenders.append(f"{path.name}:{node.lineno} imports {name}")
check("the boundary holds", not offenders, "; ".join(offenders))

print("\n[2] the split put things where it said it would")
check("utils has no checkpoints module", "checkpoints" not in utils_modules)
check("validation has checkpoints", "checkpoints" in validation_modules)
check("utils has provenance (the pipeline writes its own)", "provenance" in utils_modules)
check("validation has compare", "compare" in validation_modules)
check("validation has schema_report", "schema_report" in validation_modules)

print("\n[3] the pipeline modules import cleanly with only code/utils on the path")
# Not merely 'no validation import' -- the pipeline must actually *work* without the
# validation directory present at all, which is a stronger claim.
import subprocess
import sys

script = (
    "import sys; sys.path = [p for p in sys.path if 'validation' not in p];"
    f"sys.path.insert(0, r'{UTILS}');"
    "import paths, provenance, trial_responses, v1dd_nwb, stimulus_metrics;"
    "print('ok')"
)
proc = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
check("all five pipeline modules import with validation off the path",
      proc.returncode == 0 and "ok" in proc.stdout,
      (proc.stderr or "").strip().splitlines()[-1] if proc.returncode else "")

summary()
