import os
import numpy as np
import pyvista as pv


def write_particles_vtp(xp, vp, s, eps, step, t, out_dir,
                        phase=None, history=None):
    os.makedirs(out_dir, exist_ok=True)
    pts = np.column_stack([xp, np.zeros(len(xp))])
    cloud = pv.PolyData(pts)
    cloud["velocity"] = np.column_stack([vp, np.zeros(len(xp))])
    cloud["stress_xx"] = s[:, 0]
    cloud["stress_yy"] = s[:, 1]
    cloud["stress_xy"] = s[:, 2]
    cloud["von_mises"] = np.sqrt(
        s[:, 0]**2 - s[:, 0] * s[:, 1] + s[:, 1]**2 + 3.0 * s[:, 2]**2
    )
    cloud["strain_xx"] = eps[:, 0]
    cloud["strain_yy"] = eps[:, 1]
    cloud["strain_xy"] = eps[:, 2]
    if phase is not None:
        cloud["phase"] = np.asarray(phase)
    if history is not None:
        cloud["history"] = np.asarray(history)
    fname = f"particles_{step:05d}.vtp"
    cloud.save(os.path.join(out_dir, fname))
    return fname


def write_particles_vtp_aniso(xp, vp, stress, phase_d, phase_a, history,
                              step, t, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    pts = np.column_stack([xp, np.zeros(len(xp))])
    cloud = pv.PolyData(pts)
    cloud["velocity"] = np.column_stack([vp, np.zeros(len(xp))])
    cloud["von_mises"] = np.sqrt(
        stress[:, 0]**2 - stress[:, 0] * stress[:, 1] + stress[:, 1]**2
        + 3.0 * stress[:, 2]**2
    )
    cloud["phase_d"] = np.asarray(phase_d)
    cloud["phase_a"] = np.asarray(phase_a)
    cloud["history"] = np.asarray(history)
    fname = f"particles_{step:05d}.vtp"
    cloud.save(os.path.join(out_dir, fname))
    return fname


def write_pvd(out_dir, entries):
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
