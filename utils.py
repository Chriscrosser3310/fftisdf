import numpy as np


def eri7d_to_full(tensor_7d, kpts_int, kmesh):
    nkpts, _, _, n1, n2, n3, n4 = tensor_7d.shape
    out = np.zeros((nkpts * n1, nkpts * n2, nkpts * n3, nkpts * n4), dtype=tensor_7d.dtype)
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}
    for k1 in range(nkpts):
        for k2 in range(nkpts):
            for k3 in range(nkpts):
                k4_int = (kpts_int[k1] - kpts_int[k2] + kpts_int[k3]) % kmesh
                k4 = kpt_map[tuple(k4_int)]
                s1 = slice(k1 * n1, (k1 + 1) * n1)
                s2 = slice(k2 * n2, (k2 + 1) * n2)
                s3 = slice(k3 * n3, (k3 + 1) * n3)
                s4 = slice(k4 * n4, (k4 + 1) * n4)
                out[s1, s2, s3, s4] = tensor_7d[k1, k2, k3].copy()
    return out


def mp2_from_ovov_7d(eri_ovov_7d, mo_energy, nocc, kpts_int, kmesh):
    nkpts = eri_ovov_7d.shape[0]
    nvir = eri_ovov_7d.shape[4]
    mo_energy = np.asarray(mo_energy)
    e_occ = mo_energy[:, :nocc]
    e_vir = mo_energy[:, nocc:nocc+nvir]
    kpt_map = {tuple(k): i for i, k in enumerate(kpts_int)}

    emp2 = 0.0
    for ki in range(nkpts):
        for kj in range(nkpts):
            for ka in range(nkpts):
                kb_int = (kpts_int[ki] - kpts_int[ka] + kpts_int[kj]) % kmesh
                kb = kpt_map[tuple(kb_int)]

                eia = e_occ[ki, :, None] - e_vir[ka]
                ejb = e_occ[kj, :, None] - e_vir[kb]
                eijab = eia[:, None, :, None] + ejb[None, :, None, :]

                gijab = eri_ovov_7d[ki, ka, kj].transpose(0, 2, 1, 3) / nkpts
                gijba = eri_ovov_7d[ki, kb, kj].transpose(0, 2, 1, 3) / nkpts

                t2 = np.conj(gijab / eijab)
                edi = np.einsum("ijab,ijab", t2, gijab, optimize=True).real
                exi = -np.einsum("ijab,ijba", t2, gijba, optimize=True).real
                emp2 += 2.0 * edi + exi

    emp2 /= nkpts
    return emp2


def mp2_from_ovov_full(eri_ovov_full, mo_energy, nocc):
    mo_energy = np.asarray(mo_energy)
    nkpts = mo_energy.shape[0]
    nvir = eri_ovov_full.shape[1] // nkpts

    e_occ = mo_energy[:, :nocc].reshape(nkpts * nocc)
    e_vir = mo_energy[:, nocc:nocc+nvir].reshape(nkpts * nvir)

    eijab = (
        e_occ[:, None, None, None]
        + e_occ[None, :, None, None]
        - e_vir[None, None, :, None]
        - e_vir[None, None, None, :]
    )

    gijab = eri_ovov_full.swapaxes(1, 2) / nkpts
    t2 = np.conj(gijab / eijab)

    emp2 = (
        2.0 * np.einsum("ijab,ijab", t2, gijab, optimize=True)
        - np.einsum("ijab,ijba", t2, gijab, optimize=True)
    ).real / nkpts
    return emp2
