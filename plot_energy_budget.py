# %%
# =============================================================================
#  Post-processing for the Yvonnet plate  --  plots straight from a checkpoint.
#
#  Every figure that used to live at the bottom of `Yvonnet_plate.py` and
#  `resume_Yvonnet.py` is here instead.  Those drivers now only run the
#  simulation and write `checkpoint.npz`; this script reads that file and draws.
#
#  The point of the split is that plotting no longer costs a simulation.  The
#  checkpoint carries the FULL time history from step 0 -- `resume_Yvonnet.py`
#  loads the series, appends to them and re-saves the whole arrays -- so one
#  checkpoint plots the entire run however many resume batches produced it.
#  Particle fields (positions, d, alpha) are final-state only, which is all the
#  damage maps need.
#
#  Usage
#  -----
#      python plot_energy_budget.py                       # default checkpoint
#      python plot_energy_budget.py path/to/checkpoint.npz
#      python plot_energy_budget.py --save figs           # write PNGs, no window
#
#  Or open it in the VS Code interactive window / Spyder and run the cells.
#
#  Energies are per unit out-of-plane thickness [J/m]; time is plotted in us.
# =============================================================================

import os
import sys

import numpy as np
import matplotlib.pyplot as plt


DEFAULT_CHECKPOINT = os.path.join('vtk_output_Yvonnet', 'checkpoint.npz')

# Series the energy budget needs.  A checkpoint written before the budget
# existed cannot be back-filled: the cumulative terms depend on a history that
# was never recorded, and zeros would plot as a real flat baseline.
ENERGY_KEYS = ('ea', 'fda', 'faa', 'va', 'wa')


# %%
# =============================================================================
#  Loading
# =============================================================================
def load_checkpoint(path=DEFAULT_CHECKPOINT):
    """Read a Yvonnet checkpoint into a plain dict of arrays.

    Fails loudly on a checkpoint that predates the energy budget rather than
    inventing the missing series.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"No checkpoint at '{path}'.\n"
            "Run Yvonnet_plate.py (or resume_Yvonnet.py) first."
        )

    ckpt = np.load(path, allow_pickle=True)

    missing = [k for k in ENERGY_KEYS if k not in ckpt.files]
    if missing:
        raise KeyError(
            f"Checkpoint '{path}' predates the energy budget: missing {missing}.\n"
            "These series cannot be reconstructed after the fact (the cumulative "
            "terms depend on history that was never recorded).\n"
            "Re-run Yvonnet_plate.py from step 0 to produce a checkpoint that "
            "carries them."
        )

    data = {
        'params':    ckpt['params'].item(),
        't':         float(ckpt['t']),
        'nsteps':    int(ckpt['istep_start']),
        'positions': ckpt['positions'],
        'phase':     ckpt['phase'],      # bulk damage d
        'alpha':     ckpt['alpha'],      # interfacial damage alpha
    }
    for key in ('ta', 'ka', 'sa', 'dda', 'aa') + ENERGY_KEYS:
        data[key] = np.asarray(ckpt[key], dtype=float)
    return data


def describe(data, path):
    """One-block summary of what was loaded."""
    print(f"Checkpoint:  {path}")
    print(f"  Steps recorded:   {len(data['ta'])}  (istep_start = {data['nsteps']})")
    print(f"  Final time:       {data['t']*1e6:.3f} us")
    print(f"  Particles:        {len(data['positions'])}")
    print(f"  theta:            {data['params']['theta_deg']} deg")


# %%
# =============================================================================
#  Figures
# =============================================================================
def plot_diagnostics(data):
    """Kinetic / strain energy and the two damage maxima -- the original 1x4."""
    t_us = data['ta'] * 1e6

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(t_us, data['ka']);  axes[0].set_title('Kinetic energy')
    axes[1].plot(t_us, data['sa']);  axes[1].set_title('Strain energy')
    axes[2].plot(t_us, data['dda']); axes[2].set_title('Max bulk d')
    axes[3].plot(t_us, data['aa']);  axes[3].set_title('Max interfacial alpha')
    for ax in axes:
        ax.set_xlabel('Time [us]')
    fig.tight_layout()
    return fig


def plot_energy_budget(data):
    """Elastic, viscous, crack surface, and the budget weighed against W_ext.

    The last panel is a DRIFT DIAGNOSTIC, not a conservation check: the history
    field, the irreversibility clamp, the O(dt) work rule and the one-step lag
    on the crack surface terms all preclude exact closure.
    """
    t_us   = data['ta'] * 1e6
    e_kin  = data['ka']
    e_el   = data['ea']
    e_d    = data['fda']
    e_a    = data['faa']
    e_visc = data['va']
    w_ext  = data['wa']

    fig, axe = plt.subplots(1, 4, figsize=(22, 4.5))

    axe[0].plot(t_us, e_el, color='tab:blue')
    axe[0].set_title('Total elastic energy')

    axe[1].plot(t_us, e_visc, color='tab:red')
    axe[1].set_title('Viscous dissipation')

    axe[2].plot(t_us, e_d,       label='bulk  $E_d$',             color='tab:orange')
    axe[2].plot(t_us, e_a,       label=r'interfacial $E_\alpha$', color='tab:green')
    axe[2].plot(t_us, e_d + e_a, label='total',                   color='k', lw=2)
    axe[2].set_title('Crack surface energy'); axe[2].legend(fontsize=8)

    axe[3].stackplot(t_us, e_kin, e_el, e_d, e_a, e_visc,
                     labels=['kinetic', 'elastic', 'crack surf. (bulk)',
                             r'crack surf. (interf.)', 'viscous'],
                     colors=['tab:purple', 'tab:blue', 'tab:orange',
                             'tab:green', 'tab:red'], alpha=0.8)
    axe[3].plot(t_us, w_ext, 'k--', lw=2, label='$W_{ext}$')
    axe[3].set_title('Budget vs external work\n(drift diagnostic, not conservation)')
    axe[3].legend(fontsize=8, loc='upper left')

    for ax in axe:
        ax.set_xlabel('Time [us]')
        ax.set_ylabel('Energy per thickness [J/m]')
    fig.tight_layout()
    return fig


def plot_damage_maps(data):
    """Final bulk d and interfacial alpha scattered over the particle set."""
    xp        = data['positions']
    theta_deg = data['params']['theta_deg']

    fig, (axd, axa) = plt.subplots(2, 1, figsize=(14, 10))
    scd = axd.scatter(xp[:, 0], xp[:, 1], s=1.0, c=data['phase'],
                      cmap='inferno', vmin=0, vmax=1, rasterized=True)
    fig.colorbar(scd, ax=axd, label='bulk damage d'); axd.set_aspect('equal')
    axd.set_title(f'Bulk damage d  (theta = {theta_deg} deg)')

    sca = axa.scatter(xp[:, 0], xp[:, 1], s=1.0, c=data['alpha'],
                      cmap='viridis', vmin=0, vmax=1, rasterized=True)
    fig.colorbar(sca, ax=axa, label='interfacial damage alpha'); axa.set_aspect('equal')
    axa.set_title(f'Interfacial damage alpha  (theta = {theta_deg} deg)')
    fig.tight_layout()
    return fig


# %%
# =============================================================================
#  Numeric summary -- the same closing report the drivers used to print
# =============================================================================
def print_budget_summary(data):
    ka, ea = data['ka'], data['ea']
    fda    = data['fda']
    faa    = data['faa']
    va, wa = data['va'], data['wa']

    print("\nEnergy budget at the last recorded step  [J/m]")
    print(f"  Kinetic                E_kin   = {ka[-1]:.6e}")
    print(f"  Elastic (degraded)     E_el    = {ea[-1]:.6e}")
    print(f"  Crack surface, bulk    E_d     = {fda[-1]:.6e}")
    print(f"  Crack surface, interf. E_a     = {faa[-1]:.6e}")
    print(f"  Crack surface, total   E_crack = {fda[-1] + faa[-1]:.6e}")
    print(f"  Viscous dissipation    E_visc  = {va[-1]:.6e}")
    print(f"  External work          W_ext   = {wa[-1]:.6e}")

    accounted = ka[-1] + ea[-1] + fda[-1] + faa[-1] + va[-1]
    drift     = accounted - wa[-1]
    print(f"  Sum - W_ext (drift)            = {drift:.6e}"
          + (f"   ({100.0*drift/wa[-1]:+.2f} % of W_ext)" if abs(wa[-1]) > 0.0 else ""))
    print("  NOTE: drift diagnostic only -- the history field, the irreversibility")
    print("        clamp, the O(dt) work rule and the one-step lag on E_d/E_a make")
    print("        exact closure impossible.")


# %%
# =============================================================================
#  Entry point
# =============================================================================
FIGURES = {
    'diagnostics':   plot_diagnostics,
    'energy_budget': plot_energy_budget,
    'damage_maps':   plot_damage_maps,
}


def main(argv):
    save_dir = None
    if '--save' in argv:
        i = argv.index('--save')
        save_dir = argv[i + 1] if len(argv) > i + 1 else 'figs'
        argv = argv[:i] + argv[i + 2:]

    path = argv[0] if argv else DEFAULT_CHECKPOINT

    data = load_checkpoint(path)
    describe(data, path)
    print_budget_summary(data)

    figs = {name: fn(data) for name, fn in FIGURES.items()}

    if save_dir is None:
        plt.show()
    else:
        os.makedirs(save_dir, exist_ok=True)
        for name, fig in figs.items():
            out = os.path.join(save_dir, f'{name}.png')
            fig.savefig(out, dpi=150)
            print(f"  wrote {out}")


if __name__ == '__main__':
    main(sys.argv[1:])
