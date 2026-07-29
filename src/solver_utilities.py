"""
solver_utilities.py
====================
Reusable building blocks for the 3D MUSL elasto-plastic + damage MPM scheme
(README.md, steps 22-34). These are the *problem-independent* pieces:

    Grid / I-O (re-exported from their canonical modules so a driver only needs
    one import surface):
        NodeState3D, get_mpm3d_shape, build_particle_element_map   (src.solver3d)
        write_particles_vtp_3d, write_pvd                          (src.vtk_export)

    Constitutive routines (defined here):
        _von_mises_dev   – von Mises from a deviatoric stress (Voigt)
        _von_mises_full  – von Mises from a full Cauchy stress (Voigt)
        _jc_flow_stress  – Johnson-Cook flow stress with damage + thermal softening
        _pressure        – damaged equation of state (linear or Mie-Grüneisen)
        _constitutive_update – per-particle update of F, V, stress, eps_p and D

Boundary conditions are deliberately NOT in this module: Dirichlet / Neumann
conditions are problem specific and belong in each main driver script (e.g.
bone_fracture.py), which also owns the MUSL time-stepping loop.

Stress is stored in Voigt form: [xx, yy, zz, xy, yz, xz].
"""
import numpy as np

# ── Grid bookkeeping + VTK writers — single source of truth elsewhere ─────────
from src.solver3d import NodeState3D, get_mpm3d_shape, build_particle_element_map
from src.vtk_export import write_particles_vtp_3d, write_pvd

__all__ = [
    "NodeState3D", "get_mpm3d_shape", "build_particle_element_map",
    "write_particles_vtp_3d", "write_pvd",
    "_von_mises_dev", "_von_mises_full", "_jc_flow_stress", "_pressure",
    "_constitutive_update",
]


# ── Stress invariants ─────────────────────────────────────────────────────────

def _von_mises_dev(s):
    """Von Mises equivalent stress from a 3D *deviatoric* stress in Voigt form
    s = [s_xx, s_yy, s_zz, s_xy, s_yz, s_xz]:  sigma_eq = sqrt(3/2 s:s)."""
    return np.sqrt(1.5 * (
        s[0]**2 + s[1]**2 + s[2]**2 + 2.0 * (s[3]**2 + s[4]**2 + s[5]**2)
    ))


def _von_mises_full(s):
    """Von Mises equivalent stress from a full Cauchy stress in Voigt form."""
    sxx, syy, szz, sxy, syz, sxz = s
    return np.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )


# ── Constitutive laws ─────────────────────────────────────────────────────────

def _jc_flow_stress(mat, eps_p, eps_dot, D, Tstar):
    """Johnson-Cook flow stress including damage and (optional) thermal softening
    — README step 29d:
        sigma_f = [A + B eps_p^n][1 + C ln(eps_dot*)][1 - (T*)^m](1 - D)
    """
    eps_dot_star = max(eps_dot / mat.eps_dot_0, 1.0)
    hardening    = mat.A + mat.B * max(eps_p, 0.0) ** mat.n
    rate         = 1.0 + mat.C * np.log(eps_dot_star)
    thermal      = 1.0
    if mat.thermal_enabled and Tstar > 0.0:
        thermal = max(1.0 - Tstar ** mat.m_th, 0.0)
    return hardening * rate * thermal * (1.0 - D)


def _pressure(mat, detF, e, D, eos):
    """README step 30 — pressure p_hat (the scalar added to the deviatoric
    stress so that sigma = s_dev + p_hat I; tension positive).

    eos = 'linear':  p_hat = -K (1 - detF)(1 - D)
    eos = 'mg'    :  damaged Mie-Grüneisen Hugoniot.
    """
    if eos == 'linear':
        return -mat.K * (1.0 - detF) * (1.0 - D)

    # Mie-Grüneisen with damage (README step 30).
    eta_base = 1.0 / detF                  # rho / rho0
    eta = eta_base * (1.0 - D) if eta_base > 1.0 else eta_base
    mu  = eta - 1.0
    denom = (eta - mat.S_alpha * mu) ** 2
    if denom < 1e-20:
        return -mat.Gamma0 * e
    p_H = mat.initial_density * (1.0 - D) * mat.c0**2 * mu * (
        eta - 0.5 * mat.Gamma0 * mu) / denom
    return -(p_H + mat.Gamma0 * e)


def _constitutive_update(mat, pid, particles, dtime, Lp, eos='linear', Tstar=0.0):
    """Per-particle elasto-plastic + damage update — README steps 24-33.

    Updates in place: deformation gradient, volume, deviatoric/total stress,
    equivalent plastic strain, internal energy, damage-initiation variable and
    the damage variable D. Damage is coupled back through G'=(1-D)G (29a), the
    flow stress (29d) and the EOS (30)."""
    I3 = np.eye(3)

    # 24-26: deformation gradient, volume, density
    F    = (I3 + Lp * dtime) @ particles.deformation_gradient[pid].reshape(3, 3)
    detF = np.linalg.det(F)
    particles.deformation_gradient[pid] = F.ravel()
    particles.volume[pid] = detF * particles.initial_volume[pid]

    # 27: strain-rate tensor and polar decomposition via SVD
    Dp         = 0.5 * (Lp + Lp.T)
    U_s, _, Vt = np.linalg.svd(F)
    R          = U_s @ Vt

    # 28: un-rotated deviatoric strain rate and equivalent strain rate
    dp     = R.T @ Dp @ R
    d_vol  = (dp[0, 0] + dp[1, 1] + dp[2, 2]) / 3.0
    dp_dev = dp - d_vol * I3
    eps_dot = np.sqrt(2.0 / 3.0 * (
        dp_dev[0, 0]**2 + dp_dev[1, 1]**2 + dp_dev[2, 2]**2
        + 2.0 * (dp_dev[0, 1]**2 + dp_dev[1, 2]**2 + dp_dev[0, 2]**2)
    ))

    D_old = particles.D[pid]

    # 29a: damaged shear modulus
    Gp = (1.0 - D_old) * mat.G

    # 29b: elastic trial deviatoric stress (Voigt)
    s       = particles.stress_dev[pid]
    d_voigt = np.array([dp_dev[0, 0], dp_dev[1, 1], dp_dev[2, 2],
                        dp_dev[0, 1], dp_dev[1, 2], dp_dev[0, 2]])
    s_trial = s + 2.0 * Gp * dtime * d_voigt

    # 29c: trial von Mises stress
    sigma_trial_eq = _von_mises_dev(s_trial)

    # 29d: JC flow stress (damage + thermal)
    eps_p   = particles.eps_p[pid]
    sigma_f = _jc_flow_stress(mat, eps_p, eps_dot, D_old, Tstar)

    # 29e-f: elastic step or plastic radial return
    delta_eps_p = 0.0
    if sigma_trial_eq <= sigma_f or Gp <= 0.0:
        particles.stress_dev[pid] = s_trial
    else:
        delta_eps_p               = (sigma_trial_eq - sigma_f) / (3.0 * Gp)
        particles.eps_p[pid]      = eps_p + delta_eps_p
        particles.stress_dev[pid] = (sigma_f / sigma_trial_eq) * s_trial
    s_upd = particles.stress_dev[pid]

    # 30: internal-energy increment (Taylor-Quinney) + damaged pressure
    if eos == 'mg':
        particles.e[pid] += detF * mat.chi * sigma_f * delta_eps_p
    p_hat = _pressure(mat, detF, particles.e[pid], D_old, eos)

    # 31: assemble un-rotated stress and rotate back to the global frame
    sigma_unrot = np.array([
        [s_upd[0] + p_hat, s_upd[3],          s_upd[5]         ],
        [s_upd[3],          s_upd[1] + p_hat, s_upd[4]         ],
        [s_upd[5],          s_upd[4],          s_upd[2] + p_hat],
    ])
    sigma_rot = R @ sigma_unrot @ R.T
    particles.stress[pid] = np.array([
        sigma_rot[0, 0], sigma_rot[1, 1], sigma_rot[2, 2],
        sigma_rot[0, 1], sigma_rot[1, 2], sigma_rot[0, 2],
    ])

    # 33: Johnson-Cook damage update (only accumulates with plastic flow)
    if mat.damage_enabled and delta_eps_p > 0.0:
        sigma_eq   = max(_von_mises_full(particles.stress[pid]), 1e-12)
        sigma_star = -p_hat / sigma_eq                              # 33a
        eps_dot_d  = max(delta_eps_p / (dtime * mat.eps_dot_0), 1.0)
        eps_f = ((mat.D1 + mat.D2 * np.exp(mat.D3 * sigma_star))    # 33b
                 * (1.0 + mat.D4 * np.log(eps_dot_d))
                 * (1.0 + mat.D5 * Tstar))
        eps_f = max(eps_f, 1e-8)
        particles.D_init[pid] += delta_eps_p / eps_f               # 33c
        if particles.D_init[pid] >= 1.0:                           # 33d
            particles.D[pid] = min(10.0 * (particles.D_init[pid] - 1.0), 1.0)
        # else: D stays at its previous value (0 until initiation).
