import os
import numpy as np
import pyvista as pv


def write_particles_vtp(xp, vp, s, s_dev, eps_p, D, step, t, out_dir):
    """Write one .vtp file for a single MPM time step."""
    os.makedirs(out_dir, exist_ok=True)

    pts   = np.column_stack([xp, np.zeros(len(xp))])
    cloud = pv.PolyData(pts)

    cloud["velocity"] = np.column_stack([vp, np.zeros(len(xp))])

    cloud["stress_xx"] = s[:, 0]
    cloud["stress_yy"] = s[:, 1]
    cloud["stress_xy"] = s[:, 2]

    # Von Mises from deviatoric stress — correct for plane strain (sigma_zz != 0).
    # s_zz = -(s_xx + s_yy) from the traceless condition; full 3D formula:
    # sigma_vm = sqrt(3/2 * (s_xx^2 + s_yy^2 + s_zz^2 + 2*s_xy^2))
    s_zz = -(s_dev[:, 0] + s_dev[:, 1])
    cloud["von_mises"] = np.sqrt(1.5 * (
        s_dev[:, 0]**2 + s_dev[:, 1]**2 + s_zz**2 + 2.0 * s_dev[:, 2]**2
    ))

    cloud["eps_p"]  = eps_p
    cloud["damage"] = D

    fname = f"particles_{step:05d}.vtp"
    cloud.save(os.path.join(out_dir, fname))
    return fname


def write_pvd(out_dir, entries):
    """Write a .pvd collection file linking all time steps for ParaView.

    entries: list of (timestep_float, filename_str) tuples
    """
    pvd_path = os.path.join(out_dir, "simulation.pvd")
    with open(pvd_path, "w") as f:
        f.write('<?xml version="1.0"?>\n')
        f.write('<VTKFile byte_order="LittleEndian" type="Collection" version="0.1">\n')
        f.write('<Collection>\n')
        for t, fname in entries:
            f.write(f'  <DataSet file="{fname}" groups="" part="0" timestep="{t:.6e}"/>\n')
        f.write('</Collection>\n')
        f.write('</VTKFile>\n')
    return pvd_path


def write_particles_vtp_elastic(xp, vp, sigma, J, step, t, out_dir):
    """Write one .vtp file for an elastic Warp step: velocity, von_mises, pressure, J.

    sigma: (n, 3, 3) Cauchy stress, including sigma_zz (plane strain).
    """
    os.makedirs(out_dir, exist_ok=True)

    pts   = np.column_stack([xp, np.zeros(len(xp))])
    cloud = pv.PolyData(pts)

    cloud["velocity"] = np.column_stack([vp, np.zeros(len(xp))])

    p   = -np.trace(sigma, axis1=1, axis2=2) / 3.0
    dev = sigma + p[:, None, None] * np.eye(3)
    cloud["von_mises"] = np.sqrt(1.5 * np.sum(dev * dev, axis=(1, 2)))
    cloud["pressure"]  = p
    cloud["J"]         = J

    fname = f"particles_{step:05d}.vtp"
    cloud.save(os.path.join(out_dir, fname))
    return fname


def write_particles_vtp_pf(xp, vp, sigma, nu, phi, H, step, t, out_dir):
    """Write one .vtp file for a phase-field Warp step: velocity, von_mises, damage, history.

    sigma: (n, 3) degraded Cauchy stress in Voigt order (xx, yy, xy), small-strain plane strain;
    sigma_zz = nu (sigma_xx + sigma_yy) enters the von Mises stress.
    """
    os.makedirs(out_dir, exist_ok=True)

    pts   = np.column_stack([xp, np.zeros(len(xp))])
    cloud = pv.PolyData(pts)

    cloud["velocity"] = np.column_stack([vp, np.zeros(len(xp))])

    s_xx, s_yy, s_xy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    s_zz = nu * (s_xx + s_yy)
    cloud["von_mises"] = np.sqrt(0.5 * ((s_xx - s_yy)**2 + (s_yy - s_zz)**2 + (s_zz - s_xx)**2)
                                 + 3.0 * s_xy**2)
    cloud["damage"]  = phi
    cloud["history"] = H

    fname = f"particles_{step:05d}.vtp"
    cloud.save(os.path.join(out_dir, fname))
    return fname


def write_particles_vtp_aniso(xp, vp, sigma, d, alpha, H, step, t, out_dir):
    """Write one .vtp file for a two-field phase-field Warp step: velocity, von_mises,
    bulk_damage, interfacial_damage, history.

    sigma: (n, 3) degraded Cauchy stress in Voigt order (xx, yy, xy), small-strain plane stress
    (sigma_zz = 0).
    """
    os.makedirs(out_dir, exist_ok=True)

    pts   = np.column_stack([xp, np.zeros(len(xp))])
    cloud = pv.PolyData(pts)

    cloud["velocity"] = np.column_stack([vp, np.zeros(len(xp))])

    s_xx, s_yy, s_xy = sigma[:, 0], sigma[:, 1], sigma[:, 2]
    cloud["von_mises"] = np.sqrt(s_xx**2 - s_xx * s_yy + s_yy**2 + 3.0 * s_xy**2)
    cloud["bulk_damage"]        = np.asarray(d)
    cloud["interfacial_damage"] = np.asarray(alpha)
    cloud["history"]            = np.asarray(H)

    fname = f"particles_{step:05d}.vtp"
    cloud.save(os.path.join(out_dir, fname))
    return fname
