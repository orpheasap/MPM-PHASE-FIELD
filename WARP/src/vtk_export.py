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
