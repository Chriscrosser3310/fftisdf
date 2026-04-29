import sys
import pickle
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
from pyscf.pbc import gto, scf, df

a = 1.7834
lv = np.ones((3, 3)) * a
lv -= np.diag([a, a, a])
atom = [("C", [0.00000, 0.00000, 0.00000])]
atom += [("C", [0.5 * a, 0.5 * a, 0.5 * a])]

cell = gto.Cell()
cell.unit = "A"
cell.atom = atom
cell.a = lv
cell.basis = "gth-dzvp"
cell.pseudo = "gth-pbe"
cell.ke_cutoff = 40.0
cell.verbose = 0
cell.build()

nk = int(sys.argv[1])
kmesh = np.array([nk, nk, nk])
klabel = f"{kmesh[0]}x{kmesh[1]}x{kmesh[2]}"
kpts = cell.make_kpts(kmesh)
kpts_int = np.round(cell.get_scaled_kpts(kpts) * kmesh).astype(int) % kmesh


mf = scf.KHF(cell, kpts)
mf.conv_tol = 1e-6
mf.max_cycle = 50
mf.verbose = 4
mf.exxdiv = None
mf.with_df = df.FFTDF(cell, kpts)
mf.with_df.verbose = 0
mf.with_df.build()
print("Running SCF ...", flush=True)
mf.kernel()

scf_pkl = f"data/SCF_diamond_{klabel}_{cell.basis}_ke{cell.ke_cutoff}.pkl"
with open(scf_pkl, "wb") as f:
    pickle.dump(mf, f)
print("Saved SCF pickle =", scf_pkl, flush=True)
