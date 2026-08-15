"""Compare two run directories and print what moved.

    python code/validation/diff_runs.py <old_run_dir> <new_run_dir> [--atol 1e-9]
    python code/validation/diff_runs.py --last /scratch v1dd_1196_coreg_functional_metrics

The gate for every refactor step in this port. A step that is supposed to change nothing
should print no changed columns; a step that is supposed to change something should change
exactly what was intended and nothing else — which is a claim you can only make if the
previous run still exists to compare against.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "utils"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from compare import diff_run_dirs                    # noqa: E402
from provenance import list_runs                     # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="old run directory, or the scratch root with --last")
    ap.add_argument("b", nargs="?", help="new run directory, or the asset name with --last")
    ap.add_argument("--last", action="store_true",
                    help="compare the two most recent runs of an asset")
    ap.add_argument("--atol", type=float, default=0.0,
                    help="absolute tolerance; 0 (default) means bit-for-bit")
    ap.add_argument("--json", dest="json_path", help="also write the full report here")
    args = ap.parse_args()

    if args.last:
        runs = list_runs(args.a, args.b)
        if len(runs) < 2:
            print(f"need two runs of {args.b!r} under {args.a}, found {len(runs)}")
            return 2
        old, new = runs[-2], runs[-1]
    else:
        old, new = args.a, args.b

    print(f"old  {old}\nnew  {new}\natol {args.atol}\n")
    report = diff_run_dirs(old, new, atol=args.atol)
    s = report["summary"]

    for name in sorted(report["files"]):
        r = report["files"][name]
        bits = []
        if r["only_in_b"]:
            bits.append(f"+{','.join(r['only_in_b'])}")
        if r["only_in_a"]:
            bits.append(f"-{','.join(r['only_in_a'])}")
        if r["rows_only_a"] or r["rows_only_b"]:
            bits.append(f"rows {r['rows_only_a']}/{r['rows_only_b']} unmatched")
        status = "CHANGED" if r["changed"] else ("schema" if bits else "same")
        print(f"  {status:<8} {name:<44} {len(r['identical']):>2} identical"
              + (f"  {' '.join(bits)}" if bits else ""))
        for col, d in sorted(r["changed"].items()):
            extra = (f"max {d['max_abs_diff']:.3e}  median {d['median_abs_diff']:.3e}"
                     if d.get("max_abs_diff") is not None else d.get("dtype", ""))
            print(f"           {col:<34} {d['n_differing']:>6} rows "
                  f"({d['frac_differing']:.2%})  {extra}")

    for label in ("only_in_old", "only_in_new"):
        if report[label]:
            print(f"  {label}: {report[label]}")

    print(f"\n  {len(s['files_identical'])} identical, {len(s['files_changed'])} changed, "
          f"{len(s['files_schema_changed'])} with schema changes, of {s['n_files']} files")

    if args.json_path:
        os.makedirs(os.path.dirname(os.path.abspath(args.json_path)), exist_ok=True)
        with open(args.json_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
            fh.write("\n")
        print(f"  wrote {args.json_path}")

    return 1 if s["files_changed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
