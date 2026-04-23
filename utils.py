import numpy as np
import torch
import builtins
from inspect import signature, Parameter

has_profile = hasattr(builtins, 'profile')
if not has_profile:
    profile = lambda x: x

profile_enabled = False


def modify_signature(func, args_to_add, args_to_delete, **kwargs):
    original_sig = signature(func)
    new_params = [value for value in original_sig.parameters.values() if value.name not in args_to_delete]
    for arg in args_to_add:
        new_params.append(Parameter(arg, Parameter.POSITIONAL_ONLY))
    for key, value in kwargs.items():
        new_params.append(Parameter(key, Parameter.POSITIONAL_OR_KEYWORD, default=value))
    order = [Parameter.POSITIONAL_ONLY, Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY, Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD]
    params_reordered = sorted(new_params, key=lambda p: order.index(p.kind))
    new_sig = original_sig.replace(parameters=params_reordered)

    def decorator(func_input):
        func_input.__signature__ = new_sig
        return func_input

    return decorator


def enable_profile():
    global profile_enabled
    profile_enabled = True


def disable_profile():
    global profile_enabled
    profile_enabled = False


def maybe_profile(func):
    func_profile = profile(func)

    @modify_signature(func, [], [])
    def newfunc(*args, **kwargs):
        if profile_enabled:
            return func_profile(*args, **kwargs)
        else:
            return func(*args, **kwargs)
    return newfunc

    
@maybe_profile
def c_loss_F_norm(logc, X_t, W_t):
    X2 = X_t.abs()**2
    a = X2.sum(dim=tuple(i for i in range(X_t.ndim) if i != 1)).to(logc.dtype)

    W2 = W_t.abs()**2
    b = W2.sum(dim=tuple(i for i in range(W_t.ndim) if i not in (W_t.ndim - 2, W_t.ndim - 1))).to(logc.dtype)

    c2 = torch.exp(2.0 * logc)
    c4_inv = torch.exp(-4.0 * logc)
    sx2 = torch.dot(a, c2)
    sw2 = (b * c4_inv[:, None] * c4_inv[None, :]).sum()
    return sx2 * sx2 * torch.sqrt(sw2)


@maybe_profile
def c_loss_2_norm(logc, X_t, W_t):
    c = torch.exp(logc)
    X = X_t * c[None, :, None]
    W = W_t / (c[None, :, None]**2 * c[None, None, :]**2)
    s_X = torch.linalg.svdvals(X).max()
    s_W = torch.linalg.eigvalsh(W).abs().max()
    return s_X**4 * s_W


@maybe_profile
def thc_ovvo_build_Lbar(Xo_A, Xv_A, Xo_B, Xv_B, kconserv2):
    nkpts, naux = Xo_A.shape[:2]
    if not torch.is_tensor(kconserv2):
        kconserv2 = torch.as_tensor(kconserv2, dtype=torch.long, device=Xo_A.device)
    else:
        kconserv2 = kconserv2.to(device=Xo_A.device, dtype=torch.long)

    O = torch.einsum("pIi,pKi->pIK", Xo_A, Xo_B.conj())
    V = torch.einsum("qIa,qKa->qIK", Xv_A.conj(), Xv_B)

    s_index = kconserv2.reshape(-1)
    Lpq = O[:, None] * V[None, :]
    Lbar = torch.zeros((nkpts, naux, naux), dtype=Xo_A.dtype, device=Xo_A.device)
    return Lbar.index_add(0, s_index, Lpq.reshape(nkpts * nkpts, naux, naux))


@maybe_profile
def torch_lstsq(a, b, tol=1e-10, reg=None):
    u, s, vh = torch.linalg.svd(a, full_matrices=False)
    r = s[None, :] * s[:, None]
    m = torch.abs(r) > tol * tol

    t = u.conj().T @ b @ u
    if reg is None:
        t = torch.where(m, t / r, torch.zeros_like(t))
    else:
        t = torch.where(m, t * r / (r * r + reg * reg), torch.zeros_like(t))

    v = vh.conj().T
    return v @ t @ vh


@maybe_profile
def torch_lstsq_oinv(a, b, reg=None):
    dtype = a.dtype
    a = a.to(dtype=torch.complex128)
    b = b.to(dtype=torch.complex128)
    if reg is not None:
        reg = reg.to(dtype=torch.float64)
    if a.ndim == 3:
        if reg is None:
            x = torch.linalg.solve(a, b)
            result = torch.linalg.solve(a.transpose(-1, -2), x.transpose(-1, -2)).transpose(-1, -2)

        eye = torch.eye(a.shape[-1], dtype=a.dtype, device=a.device)
        ah = a.conj().transpose(-1, -2)
        o = ah @ a + reg * eye
        rhs = ah @ b @ a
        x = torch.linalg.solve(o, rhs)
        result = torch.linalg.solve(o.transpose(-1, -2), x.transpose(-1, -2)).transpose(-1, -2)
    else:
        if reg is None:
            result = torch.linalg.solve(a.T, torch.linalg.solve(a, b).T).T

        eye = torch.eye(a.shape[0], dtype=a.dtype, device=a.device)
        ah = a.conj().T
        o = ah @ a + reg * eye
        rhs = ah @ b @ a
        result = torch.linalg.solve(o.T, torch.linalg.solve(o, rhs).T).T
    return result.to(dtype=dtype)


@maybe_profile
def thc_ovvo_solve_w_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kconserv2, reg=None):
    L_ref = thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, kconserv2)
    L = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kconserv2)
    rhs = (L_ref.transpose(-1, -2) @ W_ref.conj() @ L_ref.conj()).conj()
    return torch_lstsq_oinv(L, rhs, reg=reg)


@maybe_profile
def thc_ovvo_solve_w_error2_from_mo(Xo_ref, Xv_ref, W_ref, Xo, Xv, kconserv2, ref_norm2, reg=None):
    L_ref = thc_ovvo_build_Lbar(Xo_ref, Xv_ref, Xo, Xv, kconserv2)
    L = thc_ovvo_build_Lbar(Xo, Xv, Xo, Xv, kconserv2)
    rhs = (L_ref.transpose(-1, -2) @ W_ref.conj() @ L_ref.conj()).conj()
    W = torch_lstsq_oinv(L, rhs, reg=reg)

    ab = torch.einsum("sKL,sKL->", W.conj(), rhs).real
    bb = torch.einsum("sIJ,sKL,sIK,sJL->", W.conj(), W, L, L.conj()).real
    error2 = (ref_norm2 + bb - 2.0 * ab).real
    return W, error2


@maybe_profile
def thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kconserv2):
    Lbar = thc_ovvo_build_Lbar(Xo_A, Xv_A, Xo_B, Xv_B, kconserv2)
    return torch.einsum(
        "sIJ,sKL,sIK,sJL->",
        W_A.conj(),
        W_B,
        Lbar,
        Lbar.conj(),
    )


@maybe_profile
def thc_ovvo_error2_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kconserv2):
    aa = thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_A, Xv_A, W_A, kconserv2)
    bb = thc_ovvo_inner_from_mo(Xo_B, Xv_B, W_B, Xo_B, Xv_B, W_B, kconserv2)
    ab = thc_ovvo_inner_from_mo(Xo_A, Xv_A, W_A, Xo_B, Xv_B, W_B, kconserv2)
    return (aa + bb - 2.0 * ab.real).real


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

