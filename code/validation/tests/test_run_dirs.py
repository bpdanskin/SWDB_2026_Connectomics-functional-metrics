"""Run-stamped output directories.

A re-run that overwrites its predecessor destroys the only evidence that a change left
the numbers alone — which is the gate every step of this port depends on. That is not
hypothetical: it is exactly what happened to the P1 diff.
"""
import os
import shutil
import tempfile
import time

from harness import check, fails, load, summary

prov = load("provenance")

NAME = "v1dd_1196_coreg_functional_metrics"

print("[1] run_stamp: sortable, filesystem-safe, second resolution")
s = prov.run_stamp(0)
check("shape is YYYY-MM-DD_HH-MM-SS", len(s) == 19 and s[4] == s[7] == "-" and s[10] == "_",
      s)
check("no character that needs quoting in a path",
      not (set(s) & set(' :/\\?*"<>|')), s)
early, late = prov.run_stamp(1_000_000_000), prov.run_stamp(2_000_000_000)
check("lexicographic order matches chronological order", early < late, f"{early} < {late}")

print("\n[2] run_dir composes the path without creating it")
root = tempfile.mkdtemp(prefix="runs_")
d = prov.run_dir(root, NAME, "2026-08-14_22-00-00")
check("path is <root>/<name>_<stamp>",
      os.path.basename(d) == f"{NAME}_2026-08-14_22-00-00", os.path.basename(d))
check("does not create it (the caller decides)", not os.path.exists(d))

print("\n[3] list_runs / latest_run")
check("empty root -> no runs", prov.list_runs(root, NAME) == [])
check("empty root -> latest is None", prov.latest_run(root, NAME) is None)
check("missing root does not raise", prov.list_runs(os.path.join(root, "nope"), NAME) == [])

stamps = ["2026-08-14_09-00-00", "2026-08-14_22-00-00", "2026-08-15_07-30-00"]
for st in stamps:
    os.makedirs(prov.run_dir(root, NAME, st))
runs = prov.list_runs(root, NAME)
check("finds all three", len(runs) == 3, str(len(runs)))
check("oldest first", [os.path.basename(r).rsplit("_", 2)[-2:] for r in runs]
      == [st.split("_") for st in stamps])
check("latest_run is the newest stamp",
      os.path.basename(prov.latest_run(root, NAME)).endswith(stamps[-1]))

print("\n[4] ordering comes from the stamp, not from mtime")
# These artifacts travel by being copied between machines, which rewrites mtimes. If the
# ordering came from the filesystem, the 'previous run' could silently be the wrong one.
newest = prov.run_dir(root, NAME, stamps[-1])
os.utime(newest, (0, 0))                       # make the newest look ancient
check("still ordered by stamp after mtimes are scrambled",
      os.path.basename(prov.latest_run(root, NAME)).endswith(stamps[-1]))

print("\n[5] only this asset's runs, only well-formed stamps")
os.makedirs(os.path.join(root, "some_other_asset_2026-08-14_09-00-00"))
os.makedirs(os.path.join(root, f"{NAME}_not-a-stamp"))
os.makedirs(os.path.join(root, f"{NAME}_2026-08-14"))          # partial stamp
open(os.path.join(root, f"{NAME}_2026-08-14_10-00-00"), "w").close()   # a file, not a dir
check("other assets ignored", len(prov.list_runs(root, NAME)) == 3,
      str([os.path.basename(r) for r in prov.list_runs(root, NAME)]))
check("the other asset resolves on its own name",
      len(prov.list_runs(root, "some_other_asset")) == 1)

print("\n[6] two runs in the same second do not collide by accident")
a = prov.run_dir(root, NAME)
time.sleep(0.05)
b = prov.run_dir(root, NAME)
check("a stamp taken twice within a second is identical -- so take it once per run",
      a == b, "documented behaviour, not a bug: the notebook binds RUN_STAMP once")

shutil.rmtree(root, ignore_errors=True)
summary()
