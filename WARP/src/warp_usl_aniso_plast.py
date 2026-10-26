# Explicit USL MPM with two damage fields and J2 plasticity on the GPU -- bulk damage d (AT2),
# anisotropic interfacial damage alpha and plane-stress J2 plasticity with linear isotropic
# hardening acting on the effective (undegraded) stress. The elastoplastic Yvonnet plate.
# cpGIMP basis, small-strain plane stress, nominal traction, pure FLIP. SI units.
#
#   sigma0  = C_a(alpha) : (eps - eps_p),   C_a(alpha) = CA1 + M2 (1 - alpha)^2
#   q       = sqrt(3/2 sigma0^T P sigma0) <= sigma_y + k_h p   (closest-point return, full C_a)
#   sigma   = (1 - d)^2 sigma0,   H = max(H, 1/2 sigma0 . eps_e),   eps_e = eps - eps_p
#   G_a     = (1 - a_p) (1 - d_p)^2 eps_e : M2 : eps_e
#
# eps_p is the plastic strain (Voigt, engineering shear; eps_p33 = -(eps_p11 + eps_p22) is
# implied and never stored), p the equivalent plastic strain, sigma_y the yield stress and k_h the
# hardening modulus. Theory: plastic_aniso_layer_PF_MPM_explicit.tex, section 6 and Algorithm 1.
#
# One step is that of run_warp_aniso_solver, kernel for kernel, with two differences:
# damage_residual_two_field reads the elastic strain eps_e (from the previous step's stress
# update) where it read eps, and the last kernel is stress_update_aniso_plast.
# Kernel conventions: docs/parallel.md. The max d, max alpha, max p, external-work, viscous-
# dissipation and plastic-work histories are accumulated per step on the device and read once
# at the end.
import time as _time

import numpy as np
import warp as wp

from .warp_gimp import update_gimp_lp_2d
from .warp_usl_aniso import (damage_residual_two_field, g2p_aniso, grid_two_field_project,
                             grid_update_aniso, p2g_aniso, reset_grid_aniso)
from .vtk_export import write_particles_vtp_aniso_plast

# Plane-stress von Mises projection in Voigt form (11, 22, 12), engineering shear:
# sigma^T P sigma = |dev sigma|^2, with s_33 = -(s_11 + s_22)/3 included.
P_VM = wp.constant(wp.mat33(2.0 / 3.0, -1.0 / 3.0, 0.0,
                            -1.0 / 3.0, 2.0 / 3.0, 0.0,
                            0.0, 0.0, 2.0))

# Newton on the plastic multiplier: stop at |r| <= RETURN_TOL sigma_y, or after RETURN_MAX_ITER
# iterations, keeping the last iterate.
RETURN_TOL = wp.constant(1.0e-5)
RETURN_MAX_ITER = wp.constant(25)


@wp.func
def von_mises_ps(s: wp.vec3):
    """Plane-stress equivalent stress q = sqrt(3/2 s^T P s)."""
    return wp.sqrt(1.5 * wp.dot(s, P_VM * s))


@wp.func
def plane_stress_return(s_tr: wp.vec3, C: wp.mat33, sigma_y: wp.float32, k_h: wp.float32,
                        p_n: wp.float32):
    """Closest-point return of a yielding trial stress s_tr onto q = sigma_y + k_h (p_n + dp).

    Solves r(dgam) = q(sigma)(1 - 2/3 k_h dgam) - (sigma_y + k_h p_n) = 0 by Newton from
    dgam = 0, with sigma(dgam) = (I + dgam C P)^-1 s_tr. Returns (sigma, dgam) as a vec4; the
    returned sigma always belongs to the returned dgam.
    """
    CP = C * P_VM
    yield_n = sigma_y + k_h * p_n
    tol = RETURN_TOL * sigma_y
    two_thirds_kh = (2.0 / 3.0) * k_h

    dgam = wp.float32(0.0)
    A_inv = wp.identity(n=3, dtype=wp.float32)
    s = wp.vec3(s_tr)
    q = von_mises_ps(s)
    r = q - yield_n
    it = wp.int32(0)
    while it < RETURN_MAX_ITER and wp.abs(r) > tol:
        fac = 1.0 - two_thirds_kh * dgam
        ds = -(A_inv * (CP * s))
        dq = 1.5 * wp.dot(P_VM * s, ds) / q
        dr = dq * fac - two_thirds_kh * q
        dgam = wp.max(0.0, dgam - r / dr)

        A_inv = wp.inverse(wp.identity(n=3, dtype=wp.float32) + dgam * CP)
        s = A_inv * s_tr
        q = von_mises_ps(s)
        r = q * (1.0 - two_thirds_kh * dgam) - yield_n
        it += 1

    return wp.vec4(s[0], s[1], s[2], dgam)


@wp.kernel
def stress_update_aniso_plast(F: wp.array(dtype=wp.mat33),
                              L: wp.array(dtype=wp.mat22),
                              d: wp.array(dtype=wp.float32),
                              alpha: wp.array(dtype=wp.float32),
                              vol0: wp.array(dtype=wp.float32),
                              CA1: wp.mat33,
                              M2: wp.mat33,
                              sigma_y: wp.float32,
                              k_h: wp.float32,
                              dt: wp.float32,
                              step: wp.int32,
                              eps: wp.array(dtype=wp.vec3),
                              eps_p: wp.array(dtype=wp.vec3),
                              p_eq: wp.array(dtype=wp.float32),
                              eps_e: wp.array(dtype=wp.vec3),
                              sigma0: wp.array(dtype=wp.vec3),
                              sigma: wp.array(dtype=wp.vec3),
                              H: wp.array(dtype=wp.float32),
                              J: wp.array(dtype=wp.float32),
                              w_p: wp.array(dtype=wp.float32),
                              p_max: wp.array(dtype=wp.float32)):
    """Total-strain small-strain update with a plane-stress J2 return (Voigt, engineering shear):

      eps += sym(L) dt,  sigma_tr = C_a(alpha) : (eps - eps_p),
      if q(sigma_tr) > sigma_y + k_h p:  closest-point return with the full C_a(alpha),
          d eps_p = dgam P sigma0,  dp = 2/3 dgam q(sigma0),
      eps_e = eps - eps_p,  sigma = (1 - d)^2 sigma0  (no kappa),
      H = max(H, 1/2 sigma0 . eps_e),  J = det F.

    sigma0 is recomputed from the stored strains every step because C_a depends on alpha. A
    yielding particle adds its plastic work V0 sigma . d eps_p (degraded Cauchy stress) to
    w_p[step]; every particle writes its equivalent plastic strain into p_max[step].
    """
    i = wp.tid()
    Lp = L[i]
    eps_i = eps[i] + wp.vec3(dt * Lp[0, 0], dt * Lp[1, 1], dt * (Lp[0, 1] + Lp[1, 0]))
    one_a = 1.0 - alpha[i]
    one_d = 1.0 - d[i]
    C = CA1 + M2 * (one_a * one_a)

    eps_p_i = eps_p[i]
    p_i = p_eq[i]
    s0 = C * (eps_i - eps_p_i)
    yielded = von_mises_ps(s0) > sigma_y + k_h * p_i
    deps_p = wp.vec3(0.0, 0.0, 0.0)
    if yielded:
        ret = plane_stress_return(s0, C, sigma_y, k_h, p_i)
        s0 = wp.vec3(ret[0], ret[1], ret[2])
        dgam = ret[3]
        deps_p = dgam * (P_VM * s0)
        eps_p_i = eps_p_i + deps_p
        p_i = p_i + (2.0 / 3.0) * dgam * von_mises_ps(s0)
        eps_p[i] = eps_p_i
        p_eq[i] = p_i

    eps_e_i = eps_i - eps_p_i
    s = (one_d * one_d) * s0
    eps[i] = eps_i
    eps_e[i] = eps_e_i
    sigma0[i] = s0
    sigma[i] = s

    psi0 = 0.5 * wp.dot(s0, eps_e_i)
    if psi0 > H[i]:
        H[i] = psi0

    J[i] = wp.determinant(F[i])

    if yielded:
        wp.atomic_add(w_p, step, vol0[i] * wp.dot(s, deps_p))
    wp.atomic_max(p_max, step, p_i)


def run_warp_aniso_plast_solver(mesh, particles, CA1, M2, omega_a, gc_d, ell_d, gc_alpha,
                                ell_alpha, eta_d, eta_alpha, dtime, time, traction, lp0,
                                vtk_output_dir, vtk_interval, sigma_y, k_h):
    """Explicit USL MPM with bulk and interfacial damage of an anisotropic layered elastoplastic
    body under nominal traction, entirely on the device. SI units, plane stress.

    The arguments are those of run_warp_aniso_solver, plus
    sigma_y      — yield stress (Pa) of the effective stress;
    k_h          — hardening modulus (Pa), linear isotropic hardening.

    Both damage fields, the history field, the plastic strain and the equivalent plastic strain
    start at zero. Output (VTK, energy budget) happens at t = 0 and every `vtk_interval` steps,
    plus the last step; the elastic energy is 1/2 sum V0 sigma . eps_e. Returns the dict of
    run_warp_aniso_solver plus 'plastic_work' (cumulative, at the output times), 'p_max' (per
    step) and 'plastic_strain_eq' (final p per particle); 'miss' includes the plastic work.
    """
    if not np.isclose(mesh.deltax, mesh.deltay):
        raise ValueError("cpGIMP needs square background cells")

    dx = float(mesh.deltax)
    n_grid_x, n_grid_y = mesh.numx, mesh.numy
    n_nodes = (n_grid_x + 1) * (n_grid_y + 1)
    n_p = particles.count
    nsteps = int(np.floor(time / dtime))
    Lx, Ly = n_grid_x * dx, n_grid_y * dx

    f32 = np.float32
    F0 = np.zeros((n_p, 3, 3), f32)
    F0[:, 0, 0] = particles.deformation_gradient[:, 0]
    F0[:, 0, 1] = particles.deformation_gradient[:, 1]
    F0[:, 1, 0] = particles.deformation_gradient[:, 2]
    F0[:, 1, 1] = particles.deformation_gradient[:, 3]
    F0[:, 2, 2] = 1.0

    # ----- upload once -----
    x = wp.array(particles.positions.astype(f32), dtype=wp.vec2)
    v = wp.array(particles.velocities.astype(f32), dtype=wp.vec2)
    mass = wp.array(particles.mass.astype(f32), dtype=wp.float32)
    vol0 = wp.array(particles.volume.astype(f32), dtype=wp.float32)
    t_p = wp.array(np.asarray(traction, f32), dtype=wp.vec2)
    F = wp.array(F0, dtype=wp.mat33)
    lp0_d = wp.array(np.asarray(lp0, f32), dtype=wp.vec2)
    lp = wp.zeros(n_p, dtype=wp.vec2)
    L = wp.zeros(n_p, dtype=wp.mat22)
    J = wp.ones(n_p, dtype=wp.float32)
    eps = wp.zeros(n_p, dtype=wp.vec3)
    eps_p = wp.zeros(n_p, dtype=wp.vec3)
    p_eq = wp.zeros(n_p, dtype=wp.float32)
    eps_e = wp.zeros(n_p, dtype=wp.vec3)
    sigma0 = wp.zeros(n_p, dtype=wp.vec3)
    sigma = wp.zeros(n_p, dtype=wp.vec3)
    d = wp.zeros(n_p, dtype=wp.float32)
    alpha = wp.zeros(n_p, dtype=wp.float32)
    H = wp.zeros(n_p, dtype=wp.float32)
    crack_energy_d = wp.zeros(n_p, dtype=wp.float32)
    crack_energy_a = wp.zeros(n_p, dtype=wp.float32)

    grid_m = wp.zeros(n_nodes, dtype=wp.float32)
    grid_mv = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_f = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_fext = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_pseudo_mass = wp.zeros(n_nodes, dtype=wp.float32)
    grid_dnum = wp.zeros(n_nodes, dtype=wp.float32)
    grid_anum = wp.zeros(n_nodes, dtype=wp.float32)
    grid_Fd = wp.zeros(n_nodes, dtype=wp.float32)
    grid_Fa = wp.zeros(n_nodes, dtype=wp.float32)
    grid_d = wp.zeros(n_nodes, dtype=wp.float32)
    grid_alpha = wp.zeros(n_nodes, dtype=wp.float32)
    grid_v_old = wp.zeros(n_nodes, dtype=wp.vec2)
    grid_v_new = wp.zeros(n_nodes, dtype=wp.vec2)

    d_max = wp.zeros(nsteps, dtype=wp.float32)
    alpha_max = wp.zeros(nsteps, dtype=wp.float32)
    p_max = wp.zeros(nsteps, dtype=wp.float32)
    w_ext = wp.zeros(nsteps, dtype=wp.float32)
    d_visc = wp.zeros(nsteps, dtype=wp.float32)
    w_p = wp.zeros(nsteps, dtype=wp.float32)

    CA1_d = wp.mat33(*np.asarray(CA1, f32).ravel())
    M2_d = wp.mat33(*np.asarray(M2, f32).ravel())
    omega_d = wp.mat22(*np.asarray(omega_a, f32).ravel())
    gc_d_f, ell_d_f, gc_a_f, ell_a_f = f32(gc_d), f32(ell_d), f32(gc_alpha), f32(ell_alpha)
    eta_d_f, eta_a_f = f32(eta_d), f32(eta_alpha)
    inv_eta_d, inv_eta_a = f32(1.0 / eta_d), f32(1.0 / eta_alpha)
    sigma_y_f, k_h_f = f32(sigma_y), f32(k_h)
    dt_f, dx_f = f32(dtime), f32(dx)
    clamp = (f32(dx), f32(Lx - dx), f32(dx), f32(Ly - dx))

    wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])

    mass_h = particles.mass
    vol0_h = particles.volume
    vtk_entries = []
    out_steps, kinetic, elastic, crack_bulk, crack_interf = [], [], [], [], []

    def output(step):
        t = step * dtime
        xh, vh = x.numpy().astype(np.float64), v.numpy().astype(np.float64)
        sh, eps_eh = sigma.numpy().astype(np.float64), eps_e.numpy().astype(np.float64)
        out_steps.append(step)
        kinetic.append(0.5 * np.sum(mass_h * np.sum(vh ** 2, axis=1)))
        elastic.append(0.5 * np.sum(vol0_h * np.sum(sh * eps_eh, axis=1)))
        crack_bulk.append(np.sum(crack_energy_d.numpy().astype(np.float64)))
        crack_interf.append(np.sum(crack_energy_a.numpy().astype(np.float64)))

        if vtk_output_dir is not None:
            fname = write_particles_vtp_aniso_plast(xh, vh, sh, d.numpy(), alpha.numpy(),
                                                    H.numpy(), p_eq.numpy(), step, t,
                                                    vtk_output_dir)
            vtk_entries.append((t, fname))

    output(0)
    wp.synchronize()
    t_start = _time.perf_counter()

    for step in range(nsteps):
        wp.launch(reset_grid_aniso, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_dnum,
                          grid_anum, grid_Fd, grid_Fa])
        wp.launch(p2g_aniso, dim=n_p,
                  inputs=[x, v, mass, vol0, J, sigma, t_p, d, alpha, lp, dx_f, n_grid_x,
                          grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_dnum,
                          grid_anum])
        wp.launch(grid_two_field_project, dim=n_nodes,
                  inputs=[grid_pseudo_mass, grid_dnum, grid_anum, grid_d, grid_alpha])
        # G_a is driven by the elastic strain eps_e of the previous step's stress update
        wp.launch(damage_residual_two_field, dim=n_p,
                  inputs=[x, vol0, d, alpha, H, eps_e, lp, grid_d, grid_alpha, dx_f, n_grid_x,
                          gc_d_f, ell_d_f, gc_a_f, ell_a_f, omega_d, M2_d,
                          grid_Fd, grid_Fa, crack_energy_d, crack_energy_a])
        wp.launch(grid_update_aniso, dim=n_nodes,
                  inputs=[grid_m, grid_mv, grid_f, grid_fext, grid_pseudo_mass, grid_Fd, grid_Fa,
                          dt_f, inv_eta_d, inv_eta_a, step, grid_d, grid_alpha,
                          grid_v_old, grid_v_new, w_ext])
        wp.launch(g2p_aniso, dim=n_p,
                  inputs=[x, v, F, L, d, alpha, vol0, lp, grid_v_old, grid_v_new, grid_d,
                          grid_alpha, dx_f, n_grid_x, dt_f, eta_d_f, eta_a_f, step,
                          d_max, alpha_max, d_visc])
        wp.launch(update_gimp_lp_2d, dim=n_p, inputs=[F, lp, lp0_d, x, *clamp])
        wp.launch(stress_update_aniso_plast, dim=n_p,
                  inputs=[F, L, d, alpha, vol0, CA1_d, M2_d, sigma_y_f, k_h_f, dt_f, step,
                          eps, eps_p, p_eq, eps_e, sigma0, sigma, H, J, w_p, p_max])

        if (step + 1) % vtk_interval == 0 or step + 1 == nsteps:
            output(step + 1)

    wp.synchronize()
    wall_time = _time.perf_counter() - t_start

    # ----- one final copy for post-processing -----
    Fh = F.numpy()
    particles.positions = x.numpy().astype(np.float64)
    particles.velocities = v.numpy().astype(np.float64)
    particles.deformation_gradient = np.column_stack(
        [Fh[:, 0, 0], Fh[:, 0, 1], Fh[:, 1, 0], Fh[:, 1, 1]]).astype(np.float64)

    out_steps = np.array(out_steps)
    w_ext_cum = np.concatenate([[0.0], np.cumsum(w_ext.numpy().astype(np.float64))])[out_steps]
    d_visc_cum = np.concatenate([[0.0], np.cumsum(d_visc.numpy().astype(np.float64))])[out_steps]
    w_p_cum = np.concatenate([[0.0], np.cumsum(w_p.numpy().astype(np.float64))])[out_steps]
    kinetic, elastic = np.array(kinetic), np.array(elastic)
    crack_bulk, crack_interf = np.array(crack_bulk), np.array(crack_interf)

    return {
        'vtk_entries':       vtk_entries,
        'energy_times':      out_steps * dtime,
        'kinetic':           kinetic,
        'elastic':           elastic,
        'crack_bulk':        crack_bulk,
        'crack_interf':      crack_interf,
        'visc':              d_visc_cum,
        'plastic_work':      w_p_cum,
        'w_ext':             w_ext_cum,
        'miss':              kinetic + elastic + crack_bulk + crack_interf + d_visc_cum + w_p_cum
                             - w_ext_cum,
        'step_times':        (np.arange(nsteps) + 1) * dtime,
        'd_max':             d_max.numpy().astype(np.float64),
        'alpha_max':         alpha_max.numpy().astype(np.float64),
        'p_max':             p_max.numpy().astype(np.float64),
        'bulk_damage':       d.numpy().astype(np.float64),
        'interf_damage':     alpha.numpy().astype(np.float64),
        'history':           H.numpy().astype(np.float64),
        'plastic_strain_eq': p_eq.numpy().astype(np.float64),
        'nsteps':            nsteps,
        'wall_time':         wall_time,
    }
