"""
Bone Particle Carving + Visualization
--------------------------------------
Loads an STL mesh, generates a regular grid of candidate particles
inside the bounding box, tests each point for containment, and
renders the surviving (interior) particles alongside the bone mesh.

Dependencies:
    pip install trimesh numpy pyvista
"""

import numpy as np
import trimesh
import pyvista as pv


# ─────────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────────

STL_PATH    = "Sawbone.STL"   # ← path to your STL file
RESOLUTION  = 5          # particle spacing in mesh units (mm). 3 = fine/slow, 5 = demo
PARTICLE_RADIUS = 0.6      # visual size of each particle sphere in the plot

# Output file for the carved particle positions (millimetres). bone_fracture.py
# reads this file to initialise its particle set.
PARTICLE_OUT = "bone_particles_mm.csv"
SHOW_VIEWER  = False       # set True to open the interactive 3D PyVista window

# Colors
COLOR_MESH      = "#d4a97a"   # warm bone colour for the mesh
COLOR_PARTICLES = "#2ec4b6"   # teal for particles
OPACITY_MESH    = 0.25        # mesh transparency so particles show through


# ─────────────────────────────────────────────
# 2. LOAD MESH
# ─────────────────────────────────────────────

print(f"Loading mesh from '{STL_PATH}' ...")
mesh = trimesh.load_mesh(STL_PATH)

if not mesh.is_watertight:
    print("  ⚠  Mesh is not watertight — attempting repair ...")
    trimesh.repair.fill_holes(mesh)
    trimesh.repair.fix_winding(mesh)
    trimesh.repair.fix_normals(mesh)
    print(f"  Watertight after repair: {mesh.is_watertight}")
else:
    print("  ✓ Mesh is watertight.")


# ─────────────────────────────────────────────
# 3. GENERATE CANDIDATE PARTICLE GRID
# ─────────────────────────────────────────────

bmin, bmax = mesh.bounds
print(f"\nBounding box: {bmin} → {bmax}")

x = np.arange(bmin[0], bmax[0], RESOLUTION)
y = np.arange(bmin[1], bmax[1], RESOLUTION)
z = np.arange(bmin[2], bmax[2], RESOLUTION)

grid = np.stack(np.meshgrid(x, y, z, indexing="ij"), axis=-1).reshape(-1, 3)

# Tiny jitter to avoid floating-point edge cases on the surface
grid += np.random.default_rng(0).uniform(-1e-4, 1e-4, grid.shape)

print(f"Candidate particles (full grid): {len(grid):,}")


# ─────────────────────────────────────────────
# 4. CONTAINMENT TEST
# ─────────────────────────────────────────────

print("Testing containment (this may take a moment for dense grids) ...")
# Batched containment test — keeps peak memory bounded when no embree/pyembree
# ray backend is available (the naive ray caster allocates O(points × faces)).
CHUNK = 1000
inside = np.zeros(len(grid), dtype=bool)
for start in range(0, len(grid), CHUNK):
    stop = min(start + CHUNK, len(grid))
    inside[start:stop] = mesh.contains(grid[start:stop])
    print(f"  containment {stop:,}/{len(grid):,}", end="\r")
print()
particles = grid[inside]

print(f"Particles inside bone: {len(particles):,}  "
      f"({100 * len(particles) / len(grid):.1f}% of grid)")


# ─────────────────────────────────────────────
# 5. EXPORT PARTICLE POSITIONS (mm)
# ─────────────────────────────────────────────
# Save the interior particle centres so the MPM driver (bone_fracture.py) can
# rebuild the particle set without re-running the (slow) containment test.
# Units are millimetres — exactly the STL units. A header carries the spacing,
# from which the per-particle reference volume is recovered as RESOLUTION**3.

# First header line carries the grid resolution (mm) on its own so the driver
# script can read it with a single readline(). Remaining lines are metadata.
header = (
    f"resolution_mm={RESOLUTION}\n"
    f"Bone interior particle positions [mm]  (stl={STL_PATH}, count={len(particles)})\n"
    f"x_mm,y_mm,z_mm"
)
np.savetxt(PARTICLE_OUT, particles, delimiter=",", header=header, comments="# ",
           fmt="%.6f")
print(f"\n✓ Wrote {len(particles):,} particle positions (mm) → '{PARTICLE_OUT}'")
print(f"  spacing = {RESOLUTION} mm  →  per-particle volume = {RESOLUTION**3} mm³")

if not SHOW_VIEWER:
    print("  (SHOW_VIEWER = False — skipping the interactive 3D window.)")
    raise SystemExit(0)


# ─────────────────────────────────────────────
# 6. VISUALIZE WITH PYVISTA
# ─────────────────────────────────────────────

print("\nLaunching 3D viewer ...")

# Convert trimesh → pyvista PolyData
vertices = np.array(mesh.vertices)
faces_np = np.array(mesh.faces)
# pyvista face format: [3, v0, v1, v2, ...]
faces_pv = np.hstack([np.full((len(faces_np), 1), 3), faces_np]).ravel()
pv_mesh  = pv.PolyData(vertices, faces_pv)

# Particle cloud as a PolyData point cloud
pv_particles = pv.PolyData(particles)

# ── Plotter setup ──────────────────────────────
pl = pv.Plotter(window_size=(1280, 800))
pl.set_background("#0d0d0d", top="#1a1a2e")

# Bone mesh (semi-transparent)
pl.add_mesh(
    pv_mesh,
    color=COLOR_MESH,
    opacity=OPACITY_MESH,
    smooth_shading=True,
    label="Bone mesh",
)

# Particles rendered as spheres via Glyph
sphere = pv.Sphere(radius=PARTICLE_RADIUS)
glyphs = pv_particles.glyph(geom=sphere, scale=False, orient=False)
pl.add_mesh(
    glyphs,
    color=COLOR_PARTICLES,
    smooth_shading=True,
    label=f"Interior particles ({len(particles):,})",
)

# Axes + legend
pl.add_axes(line_width=3)
pl.add_legend(bcolor="#111111", border=True, size=(0.25, 0.12))

# Info text overlay
pl.add_text(
    f"Bone Particle Carving\n"
    f"Resolution : {RESOLUTION} units\n"
    f"Particles  : {len(particles):,}",
    position="upper_left",
    font_size=11,
    color="white",
    shadow=True,
)

pl.camera_position = "iso"
pl.show(title="Bone Particle Visualization")