"""Run every validation test and write a machine-readable summary.

Each test runs in its own subprocess, so one that crashes the interpreter or leaves a
module monkeypatched cannot affect the next. Exit codes: 0 pass, 1 fail, 2 skipped for
data that is not mounted.

    python code/validation/tests/run_all.py                 # just report
    python code/validation/tests/run_all.py /scratch/...     # also write tests.json there

Writing the JSON is what makes this usable from the capsule: the run happens where the
data is and the result gets read somewhere else.
"""
import json
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]

SKIPPED, PASSED, FAILED = 2, 0, 1


def main() -> int:
    tests = sorted(p for p in HERE.glob("test_*.py"))
    save_dir = sys.argv[1] if len(sys.argv) > 1 else None

    results = []
    started = time.time()
    for path in tests:
        t0 = time.time()
        proc = subprocess.run([sys.executable, str(path)], capture_output=True, text=True,
                              cwd=str(HERE))
        out = proc.stdout or ""
        status = {PASSED: "pass", FAILED: "fail", SKIPPED: "skip"}.get(proc.returncode,
                                                                      "error")
        n_pass = out.count("  PASS  ")
        n_fail = out.count("  FAIL  ")
        results.append({
            "test": path.name, "status": status, "returncode": proc.returncode,
            "checks_passed": n_pass, "checks_failed": n_fail,
            "seconds": round(time.time() - t0, 2),
            # Keep the tail only: enough to diagnose, small enough to commit.
            "tail": (out.strip().splitlines() or [""])[-12:] if status != "pass" else [],
            "stderr_tail": (proc.stderr or "").strip().splitlines()[-6:],
        })
        mark = {"pass": "ok  ", "skip": "skip", "fail": "FAIL", "error": "ERR "}[status]
        print(f"  {mark} {path.name:<30} {n_pass:>3} passed"
              + (f", {n_fail} failed" if n_fail else "")
              + f"   {results[-1]['seconds']:>5.1f}s")
        if status in ("fail", "error"):
            for ln in results[-1]["tail"]:
                print(f"         {ln}")

    n = {s: sum(1 for r in results if r["status"] == s)
         for s in ("pass", "skip", "fail", "error")}
    summary = {
        "n_tests": len(results), **{f"n_{k}": v for k, v in n.items()},
        "checks_passed": sum(r["checks_passed"] for r in results),
        "checks_failed": sum(r["checks_failed"] for r in results),
        "wall_seconds": round(time.time() - started, 1),
        "python": sys.version.split()[0],
        "results": results,
    }
    print()
    print(f"  {n['pass']} passed, {n['skip']} skipped, {n['fail']} failed, "
          f"{n['error']} errored  ({summary['checks_passed']} checks)")

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_path = os.path.join(save_dir, "tests.json")
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"  wrote {out_path}")

    return 1 if (n["fail"] or n["error"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
