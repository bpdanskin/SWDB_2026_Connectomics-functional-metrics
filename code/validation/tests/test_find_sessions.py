"""Session discovery across both NWB storage formats.

The bug being fixed: globbing only `*/*.nwb.zarr` silently drops sessions stored as plain
HDF5 `.nwb`, and the symptom is a shorter session list rather than an error.
"""

from harness import check, fails, load, require_dataset, summary
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
vn = load("v1dd_nwb")


TMP = Path(tempfile.mkdtemp(prefix="findsess"))


def make_asset(zarr_names, hdf5_names, both_names=()):
    root = TMP / f"asset{len(list(TMP.iterdir()))}"
    for n in zarr_names:
        (root / n / f"{n}.nwb.zarr").mkdir(parents=True)
        (root / n / f"{n}.nwb.zarr" / ".zgroup").write_text("{}", encoding="utf-8")
    for n in hdf5_names:
        (root / n).mkdir(parents=True)
        (root / n / f"{n}.nwb").write_bytes(b"\x89HDF\r\n\x1a\n")
    for n in both_names:
        (root / n / f"{n}.nwb.zarr").mkdir(parents=True)
        (root / n / f"{n}.nwb").write_bytes(b"\x89HDF\r\n\x1a\n")
    return root


print("[1] the actual bug: a mixed asset")
# 21 zarr sessions + 2 hdf5, matching the reported asset
root = make_asset([f"z{i:02d}" for i in range(21)], ["h00", "h01"])
old_way = sorted(root.glob("*/*.nwb.zarr"))
new_way = vn.find_sessions(root)
check("old glob finds only the Zarr sessions", len(old_way) == 21, str(len(old_way)))
check("find_sessions finds all 23", len(new_way) == 23, str(len(new_way)))
fmts = pd.Series([vn.nwb_format(p) for p in new_way]).value_counts().to_dict()
check("and reports the format split", fmts == {"zarr": 21, "hdf5": 2}, str(fmts))
check("one path per session directory",
      len({p.parent for p in new_way}) == len(new_way))

print("\n[2] a session holding both formats")
root2 = make_asset(["z0"], ["h0"], both_names=["dual"])
paths = vn.find_sessions(root2)
check("still one path per session", len(paths) == 3, str(len(paths)))
dual = [p for p in paths if p.parent.name == "dual"][0]
check("prefers Zarr by default", vn.nwb_format(dual) == "zarr", str(dual.name))
dual_h = [p for p in vn.find_sessions(root2, prefer="hdf5") if p.parent.name == "dual"][0]
check("prefer='hdf5' flips it", vn.nwb_format(dual_h) == "hdf5", str(dual_h.name))
try:
    vn.find_sessions(root2, prefer="parquet")
    check("rejects an unknown preference", False)
except ValueError:
    check("rejects an unknown preference", True)

print("\n[3] a stray .nwb inside a Zarr directory is not a session")
root3 = make_asset(["z0"], [])
(root3 / "z0" / "z0.nwb.zarr" / "decoy.nwb").write_bytes(b"\x89HDF\r\n\x1a\n")
paths3 = vn.find_sessions(root3)
check("decoy ignored", len(paths3) == 1 and vn.nwb_format(paths3[0]) == "zarr",
      str([str(p.name) for p in paths3]))

print("\n[4] layout fallbacks and failures")
flat = TMP / "flat"
(flat / "deep" / "nested").mkdir(parents=True)
(flat / "deep" / "nested" / "s.nwb.zarr").mkdir()
check("rglob fallback finds a deeper layout", len(vn.find_sessions(flat)) == 1)
empty = TMP / "empty"
empty.mkdir()
try:
    vn.find_sessions(empty)
    check("raises on an asset with no sessions", False)
except FileNotFoundError as e:
    check("raises on an asset with no sessions", "no *.nwb.zarr or *.nwb" in str(e))
try:
    vn.find_sessions(TMP / "does_not_exist")
    check("raises on a missing directory", False)
except FileNotFoundError as e:
    check("raises on a missing directory", "does not exist" in str(e))

print("\n[5] nwb_format dispatch")
check("zarr suffix", vn.nwb_format("/a/b/x.nwb.zarr") == "zarr")
check("hdf5 suffix", vn.nwb_format("/a/b/x.nwb") == "hdf5")
check("works on Path objects", vn.nwb_format(Path("/a/b/x.nwb.zarr")) == "zarr")

print("\n[6] open_session picks the right reader")
import inspect
src = inspect.getsource(vn.open_session)
check("dispatches to NWBZarrIO for zarr", "NWBZarrIO" in src)
check("dispatches to NWBHDF5IO for hdf5", "NWBHDF5IO" in src)
check("both branches return (nwbfile, io)", src.count("io.read(), io") == 1)

print("\n[7] peek_session degrades on an unreadable file")
info = vn.peek_session(root.parent / "asset0" / "h00" / "h00.nwb")   # not a real NWB
check("returns a row rather than raising", isinstance(info, dict))
check("records the error", info["error"] is not None, str(info["error"])[:60])
check("still reports name and format",
      info["name"] == "h00" and info["format"] == "hdf5")
check("identifiers are NaN, so the row is visibly incomplete",
      bool(np.isnan(info["column"])))

shutil.rmtree(TMP, ignore_errors=True)

summary()
