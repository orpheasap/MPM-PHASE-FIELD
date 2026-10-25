import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon


def plot_mesh(ax, nodes, elements, color='black', linewidth=1.0, alpha=0.6):
    """
    Plot a mesh using node coordinates and connectivity.
    
    Parameters:
    ax: Matplotlib axis object
    nodes: Node coordinate array (nnodes, 2)
    elements: Element connectivity array (nelems, nnodes_per_elem)
    elem_type: Element type ('Q4', 'T3', etc.)
    color: Line color
    linewidth: Line width
    alpha: Transparency
    """
    # 4-node quadrilateral elements
    for elem in elements:
        # Close the loop: node order 0->1->2->3->0
        node_indices = np.array([elem[0], elem[1], elem[2], elem[3], elem[0]])
        coords = nodes[node_indices]
        ax.plot(coords[:, 0], coords[:, 1], color=color, linewidth=linewidth, alpha=alpha)


def plot_particles(ax, xp, markersize=8, color='red', alpha=0.7, y_scale=1.0, y_ref=None):
    """
    Plot particle positions as scatter points.
    
    Parameters:
    ax: Matplotlib axis object
    xp: Particle position array (nparticles, 2)
    markersize: Size of markers
    color: Marker color
    alpha: Transparency
    y_scale: Vertical scale factor for particle displacements
    y_ref: Reference positions used to compute displacement
    """
    xp_plot = xp.copy()
    if y_ref is not None:
        xp_plot[:, 1] = y_ref[:, 1] + (xp[:, 1] - y_ref[:, 1]) * y_scale
    else:
        xp_plot[:, 1] = xp_plot[:, 1] * y_scale
    ax.scatter(xp_plot[:, 0], xp_plot[:, 1], s=markersize**2, c=color, marker='o', 
               alpha=alpha, edgecolors='none', label='Particles')


def plot_mpm_domain(nodes, elements, xp, figsize=(12, 8), particle_y_scale=1.0, particle_y_ref=None):
    """
    Create a complete visualization of the MPM domain with mesh and particles.
    
    Parameters:
    nodes: Node coordinate array from mesh
    elements: Element connectivity from mesh
    xp: Particle positions
    figsize: Figure size tuple
    particle_y_scale: Vertical magnification factor for particle positions
    
    Returns:
    fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot computational mesh
    plot_mesh(ax, nodes, elements, color='black', 
              linewidth=0.8, alpha=0.5)
    
    # Plot particles
    plot_particles(ax, xp, markersize=6, color='red', alpha=0.8,
                   y_scale=particle_y_scale, y_ref=particle_y_ref)
    
    # Set labels and formatting
    ax.set_xlabel('X', fontsize=12)
    ax.set_ylabel('Y', fontsize=12)
    ax.set_title('Material Point Method Domain', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    return fig, ax
