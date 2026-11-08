"""Convert the IndPenSim 100-batch CSV (Mendeley 10.17632/pdnjz7zz5x.1) into per-batch files.
Usage: python -I code/01_convert.py <csv_path> <out_dir>

The CSV header is misaligned: its multi-line flag header adds names, so the last two named meta columns
('Batch ID', 'Fault flag') actually hold the first two Raman channels. Columns are therefore taken BY POSITION:
  0-30  process + offline variables (31)
  31 Fault_ref (1 for batches 91-100) | 32 Control_ref (0 recipe, 1 operator) | 33 PAT_ref (1 none, 2 Raman/PAT control)
  34 Batch number (1-100) | 35 Batch number (duplicate) | 36 Fault flag (per time step)
  37-2236 Raman spectrum, 2200 channels (wavenumbers 2400..201); 2237-2238 are empty (all NaN) and are dropped after conversion
Writes out_dir/process.parquet and out_dir/raman_<batch>.npy (float32, rows aligned with process rows of that batch)."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

csv, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
names = pd.read_csv(csv, nrows=0).columns.tolist()
assert len(names) == 2239, len(names)
proc_names = [n.strip() for n in names[:31]] + ["Fault_ref", "Control_ref", "PAT_ref", "Batch", "Batch_dup", "Fault_flag"]
proc_parts, raman_parts = [], {}
for ch in pd.read_csv(csv, chunksize=20000, low_memory=False):
    a = ch.apply(pd.to_numeric, errors="coerce").to_numpy(np.float32)
    proc = pd.DataFrame(a[:, :37], columns=proc_names)
    proc_parts.append(proc)
    ram = a[:, 37:]
    for b in np.unique(proc["Batch"].to_numpy()):
        m = proc["Batch"].to_numpy() == b
        raman_parts.setdefault(int(b), []).append(ram[m])
proc = pd.concat(proc_parts, ignore_index=True)
assert (proc["Batch"] == proc["Batch_dup"]).all()
proc.drop(columns="Batch_dup").to_parquet(out / "process.parquet")
for b, parts in sorted(raman_parts.items()):
    np.save(out / f"raman_{b:03d}.npy", np.concatenate(parts))
print("rows", len(proc), "batches", len(raman_parts), "raman channels", ram.shape[1])
