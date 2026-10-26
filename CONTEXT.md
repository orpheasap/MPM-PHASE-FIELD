# MPM Phase Field

Material point method simulations of fracture in elastic plates, where damage evolves as a
phase field. This file is the project's glossary: one agreed meaning per term, so that the
vocabulary in drivers, backbone routines, issues and discussion stays consistent.

## Language

### Discretisation

**Background grid**:
The fixed Cartesian mesh of cells and nodes that momentum and damage are solved on. It
never moves and never deforms; it is scratch space that is reset every step, and it carries
no state between steps.
_Avoid_: Eulerian mesh, computational mesh, grid mesh
_Not to be confused with_: the particle mesh, which is a seeding device and not a solution
space.

**Particle mesh**:
The mesh laid over the material at the start of a run purely to place particles and give
each one an initial volume. It is discarded once seeding is done — nothing is ever solved
on it, and it plays no part in the time loop.
_Avoid_: material mesh, Lagrangian mesh
_Not to be confused with_: the background grid, which persists for the whole run and is
where the equations are actually assembled.

**Particle**:
A material point carrying the state of the body: position, mass, volume, deformation,
stress and the damage fields. Particles move with the material and are the only carrier of
state that survives a step.
_Avoid_: material point, marker, node

**Particle domain**:
The rectangular patch of material a particle is taken to occupy, described by a half-width
in each direction. A particle is a point for the physics but a patch for the interpolation.
Whether the patch is held fixed at its seeded size or convected with the deformation is a
choice a driver makes. Near the edge of the background grid, or against a rigid wall, the
patch may be clamped — shrunk so that it does not cross the edge — without changing the
particle's mass or volume; only its reach for interpolation shrinks.
_Avoid_: particle volume, particle footprint, cell of the particle

**Convected particle domain**:
A particle domain whose half-widths follow the stretch of the material in each direction,
so that the patch lengthens or thins as the particle deforms. It is the alternative to a
domain held at its seeded size.
_Avoid_: deformed domain, updated domain

**Support**:
The region over which a particle's shape function is nonzero — equivalently, the set of
grid nodes that particle can exchange anything with. Support is wider than the particle
domain: it is the domain spread by the reach of the interpolation, so a particle with a
domain one cell wide can still have a support several cells across.
_Avoid_: stencil, influence region, footprint

**Connectivity**:
The map from a background cell to the grid nodes that a particle in that cell may reach.
It is a property of the grid and the choice of interpolation, not of the particles, so it
is built once and reused. Near the grid border it holds fewer nodes, because the missing
neighbours do not exist.
_Avoid_: element map, node list, stencil map

**Pad ring**:
A band of background cells added around the outside of the material, containing no
particles at any point in the run. Its purpose is to give particles near the material's
surface real nodes to reach, so that nothing a particle deposits falls off the end of the
grid.
_Avoid_: ghost cells, halo, buffer zone

**Notch**:
The pre-existing crack introduced into the plate before the run starts, formed by leaving
a region unseeded so that no material occupies it. It is a hole in the body, not a damaged
region: it exists in the initial geometry rather than emerging from the damage fields.
_Avoid_: pre-crack, slit, initial flaw

**Rigid wall**:
An immovable, frictionless boundary that material may separate from but never pass through.
It acts on the background grid, not on particles: on the wall's row of nodes it blocks only
motion into the wall, leaving motion along it and away from it free. Because the grid
carries it, contact begins once a particle's support reaches the wall, slightly before the
material itself touches.
_Avoid_: floor, rigid floor, contact boundary
_Not to be confused with_: a fixed (Dirichlet) boundary, which holds the material in place
and would not let it rebound.

### Solution fields

**Pseudo-mass**:
The nodal weight used to normalise the damage fields, accumulated from particle volumes
rather than particle masses. It plays the role that lumped mass plays for momentum, but
for the damage problem, which has no inertia of its own.
_Avoid_: phase mass, damage mass, lumped mass

**Bulk damage**:
The damage field describing degradation of the material itself, running from intact to
fully broken. It is the field whose localised bands are read as the crack path.
_Avoid_: phase field, macro damage, d

**Interfacial damage**:
A second, separate damage field describing degradation of the internal interfaces within
the material — the debonding of an oriented microstructure rather than failure of the bulk
around it. It evolves on its own toughness and length scale, and drives the material's
stiffness anisotropically, whereas bulk damage degrades it uniformly.
_Avoid_: micro damage, interface phase field, alpha
_Not to be confused with_: bulk damage. A point can be fully debonded at its interfaces
while its surrounding bulk is still intact.

**History field**:
The running maximum, over a particle's past, of its undamaged elastic energy density. It is
what drives bulk damage, and because it never decreases, unloading cannot heal a crack.
_Avoid_: H, driving force, psi-plus

### Plasticity

**Plastic strain**:
The part of a particle's strain that is permanent — what would remain if the stress were
removed. The total strain is split into an elastic part, which alone carries stress and stored
energy, and this plastic part, which neither stores energy nor drives damage. It changes volume
not at all.
_Avoid_: permanent strain, inelastic strain
_Not to be confused with_: equivalent plastic strain, which is a scalar measure of how much
plastic flow has accumulated, not the strain itself.

**Equivalent plastic strain**:
The scalar running total of plastic flow at a particle: it only ever grows, whatever direction
the plastic strain moves in, and it is what sets how far the yield stress has risen.
_Avoid_: accumulated plastic strain, p, PEEQ

**Yield stress**:
The equivalent stress at which an intact, never-yielded particle first flows plastically.
Yielding is judged on the undegraded (effective) stress, so bulk damage does not move the
yield point.
_Avoid_: sigma_y, flow stress

**Hardening modulus**:
The rate at which the yield stress rises with equivalent plastic strain — linear isotropic
hardening, the same in every direction.
_Avoid_: H (the history field already owns that letter), plastic modulus, tangent modulus

### Energy accounting

**Crack surface energy**:
The energy stored in the damage field itself — the smeared stand-in for the surface energy of
a real crack, obtained by weighting a damage field's regularised crack density by its
toughness. There is one per damage field, so a two-field run has a bulk and an interfacial
crack surface energy, each on its own toughness and length scale. It is a state function: it
depends only on the current damage fields, never on the path taken to reach them.
_Avoid_: dissipated energy, damage energy, surface energy
_Not to be confused with_: viscous dissipation, which is also irreversible but is a cost of
the numerical scheme rather than a property of the crack.

**Viscous dissipation**:
The energy consumed by the micro-forces that turn the damage stationarity conditions into
evolution equations. It is an artefact of the regularisation, not a physical fracture cost,
and it vanishes as the damage rates go to zero — so a run in which it is comparable to the
crack surface energy is one whose viscosities are distorting the physics rather than merely
stabilising the march. Unlike crack surface energy it accumulates over time, and so cannot be
recovered from a snapshot of the fields.
_Avoid_: dissipated energy, damping energy, numerical dissipation
_Not to be confused with_: crack surface energy.

**Plastic work**:
The work the stress does on the plastic strain, summed over the run. It lumps together the part
of plastic flow that is dissipated as heat and the part locked into hardening; neither comes
back as elastic energy. Like viscous dissipation it accumulates over time and cannot be read
from a snapshot of the fields.
_Avoid_: plastic dissipation, plastic energy
_Not to be confused with_: viscous dissipation, which belongs to the damage fields, not to the
deformation.

**Energy budget**:
The named set of terms — kinetic, stored elastic, the crack surface energies, viscous
dissipation and, in an elastoplastic run, plastic work — that is weighed against the work done by the applied loads. The comparison is a
drift diagnostic rather than a conservation law: the model deliberately contains steps that
neither store nor account for energy, so the budget is expected to miss, and it is the *size*
of the miss that carries the information.
_Avoid_: dissipated energy, energy balance, energy conservation, closure
