import sys
import signal
signal.signal(signal.SIGINT, signal.SIG_DFL)

import numpy as np
from pyscf.pbc import gto, scf, df

import fft
import utils
fft.isdf.CHOLESKY_MAX_SIZE = 20000


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

kmesh = np.array([2, 2, 2])
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

# you can run the following code to check pyscf MP2, but somehow it's very slow
#from pyscf.pbc.mp import kmp2
#mp = kmp2.KMP2(mf)
#emp2_pyscf, t2_pyscf = mp.kernel(with_t2=False)
#print("PySCF KMP2: E = %16.8e" % emp2_pyscf)

print("Running SCF ...", flush=True)
mf.kernel()

C = np.asarray(mf.mo_coeff)
nkpts, nao, nmo = C.shape
nocc = cell.nelectron // 2
o = slice(None, nocc)
v = slice(nocc, None)
Cocc = np.array(C[:, :, o], order="C", copy=True)
Cvir = np.array(C[:, :, v], order="C", copy=True)
nvir = Cvir.shape[2]

cisdf = 10.0

print("")
print("cell.ke_cutoff =", cell.ke_cutoff)
print("cell.mesh =", cell.mesh)
print("kmesh =", kmesh)
print("nao =", nao)
print("nocc =", nocc)
print("nvir =", nvir)
print("cisdf =", cisdf)
print("")

# technically the program optimizes (ov|ov), which is different from (ov|vo) (this is what appears in LCC, etc), but numerically they fits equally well, not sure is it a coincidence.

C_ovov = [Cocc, Cvir, Cocc, Cvir]
C_ovvo = [Cocc, Cvir, Cvir, Cocc]
fftdf = mf.with_df
eri_ovov_7d_fft = fftdf.ao2mo_7d(C_ovov, kpts=kpts)
eri_ovvo_7d_fft = fftdf.ao2mo_7d(C_ovvo, kpts=kpts)
norm_ovov_7d_fft = np.linalg.norm(eri_ovov_7d_fft)
print("ovov ||FFTDF|| = %16.8e" % norm_ovov_7d_fft, flush=True)
norm_ovvo_7d_fft = np.linalg.norm(eri_ovvo_7d_fft)
print("ovvo ||FFTDF|| = %16.8e" % norm_ovvo_7d_fft, flush=True)
emp2_fft = utils.mp2_from_ovov_7d(eri_ovov_7d_fft, mf.mo_energy, nocc, kpts_int, kmesh)
print("FFTDF MP2: E = %16.8e" % emp2_fft, flush=True)
print()

isdf_ao = fft.ISDF(cell, kpts)
isdf_ao.verbose = 0
isdf_ao.build(cisdf=cisdf)
eri_ovov_7d_isdf_ao = isdf_ao.ao2mo_7d(C_ovov, kpts=kpts)
eri_ovvo_7d_isdf_ao = isdf_ao.ao2mo_7d(C_ovvo, kpts=kpts)
diff_isdf_ao_ovov = np.linalg.norm(eri_ovov_7d_isdf_ao - eri_ovov_7d_fft) / norm_ovov_7d_fft
diff_isdf_ao_ovvo = np.linalg.norm(eri_ovvo_7d_isdf_ao - eri_ovvo_7d_fft) / norm_ovvo_7d_fft
print('X norm', np.linalg.svd(isdf_ao.inpv_kpt)[1].max())
print()
print("ovov ||ISDF(AO) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ao_ovov, flush=True)
print("ovvo ||ISDF(AO) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ao_ovvo, flush=True)
print('ISDF(AO) Coulomb norm', np.linalg.svd(isdf_ao.coul_kpt)[1].max() / nkpts)
emp2_isdf_ao = utils.mp2_from_ovov_7d(eri_ovov_7d_isdf_ao, mf.mo_energy, nocc, kpts_int, kmesh)
print("ISDF(AO) MP2: E = %16.8e" % emp2_isdf_ao, flush=True)
print()

isdf_ov = fft.ISDF(cell, kpts, ov=(Cocc, Cvir))
isdf_ov.verbose = 0
for reg in [0, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6]:
    isdf_ov.build(cisdf=cisdf, reg=reg)
    eri_ovov_7d_isdf_ov = isdf_ov.ao2mo_7d(C_ovov, kpts=kpts)
    eri_ovvo_7d_isdf_ov = isdf_ov.ao2mo_7d(C_ovvo, kpts=kpts)
    diff_isdf_ov_ovov = np.linalg.norm(eri_ovov_7d_isdf_ov - eri_ovov_7d_fft) / norm_ovov_7d_fft
    diff_isdf_ov_ovvo = np.linalg.norm(eri_ovvo_7d_isdf_ov - eri_ovvo_7d_fft) / norm_ovvo_7d_fft
    print(f"ovov ||ISDF(OV, {str(reg):6s}) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ov_ovov, flush=True)
    print(f"ovvo ||ISDF(OV, {str(reg):6s}) - FFTDF|| / ||FFTDF|| = %16.8e" % diff_isdf_ov_ovvo, flush=True)
    print(f'ISDF(OV, {str(reg):6s}) Coulomb norm', np.linalg.svd(isdf_ov.coul_kpt)[1].max() / nkpts)
    emp2_isdf_ov = utils.mp2_from_ovov_7d(eri_ovov_7d_isdf_ov, mf.mo_energy, nocc, kpts_int, kmesh)
    print(f"ISDF(OV, {str(reg):6s}) MP2: E = %16.8e" % emp2_isdf_ov, flush=True)
    print()
