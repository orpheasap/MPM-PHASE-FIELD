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


def write_particles_vtp_3d(xp, vp, s, step, t, out_dir, eps_p=None, damage=None):
    """Write one .vtp file for a single 3D MPM time step.

    s is Cauchy stress in Voigt form: [xx, yy, zz, xy, yz, xz].
    eps_p, damage: optional per-particle scalar fields (plasto-elastic solver).
    """
    os.makedirs(out_dir, exist_ok=True)

    cloud = pv.PolyData(np.ascontiguousarray(xp, dtype=float))
    cloud["velocity"] = np.ascontiguousarray(vp, dtype=float)

    cloud["stress_xx"] = s[:, 0]
    cloud["stress_yy"] = s[:, 1]
    cloud["stress_zz"] = s[:, 2]
    cloud["stress_xy"] = s[:, 3]
    cloud["stress_yz"] = s[:, 4]
    cloud["stress_xz"] = s[:, 5]

    # von Mises from the full Cauchy stress tensor
    sxx, syy, szz = s[:, 0], s[:, 1], s[:, 2]
    sxy, syz, sxz = s[:, 3], s[:, 4], s[:, 5]
    cloud["von_mises"] = np.sqrt(
        0.5 * ((sxx - syy)**2 + (syy - szz)**2 + (szz - sxx)**2)
        + 3.0 * (sxy**2 + syz**2 + sxz**2)
    )
    p = -(sxx + syy + szz) / 3.0
    cloud["pressure"] = p

    if eps_p is not None:
        cloud["eps_p"] = np.asarray(eps_p, dtype=float)
    if damage is not None:
        cloud["damage"] = np.asarray(damage, dtype=float)

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
