import os
import numpy as np
import shutil
from src.material import Material
from src.mesh2D import Mesh
from src.particle import ParticleSet
from src.lagrange_basis import lagrange_basis_Q4
from src.quadrature import gauss_2D
from src.solver_utilities import NodeState, get_mpm2d_shape, build_particle_element_map
from src.vtk_export import write_pvd, write_particles_vtp_aniso

E, nu, rho = 32.0e9, 0.2, 2450.0
material = Material(E, nu, rho, stressState='PLANE_STRESS')
theta_deg = 60.0; theta = np.deg2rad(theta_deg); xi = 30.0
c, s = np.cos(theta), np.sin(theta); pref = E/(1.0-nu**2)
A1 = pref*np.array([[0.8437,0,0],[0,0,0],[0,0,0]])
A2 = pref*np.array([[0.1563,nu,0],[nu,1,0],[0,0,(1-nu)/2]])
T_sig = np.array([[c*c,s*s,-2*c*s],[s*s,c*c,2*c*s],[c*s,-c*s,c*c-s*s]])
CA1 = T_sig@A1@T_sig.T; M2 = T_sig@A2@T_sig.T
def C_aniso(a): return CA1 + M2*(1.0-a)**2
P_a = np.array([[c*c,s*c],[s*c,s*s]]); omega_a = np.eye(2)+xi*P_a
gc_d = 3.0; gc_alpha = gc_d/4.0
_hs = float(os.environ.get("PF_H_SCALE","1.0"))
Lx, Ly = 0.1, 0.04; hx = 0.25e-3*_hs; hy = 0.25e-3*_hs
numx = int(round(Lx/hx)); numy = int(round(Ly/hy)); h_e = min(hx,hy)
ell_d = ell_alpha = 2.0*h_e
mesh = Mesh(Lx, Ly, numx, numy)
noX, noY, ngp = numx, numy, 1
W, Q = gauss_2D(ngp); pmesh = Mesh(Lx,Ly,noX,noY)
pCount_max = noX*noY*len(W)
all_pos = np.zeros((pCount_max,2)); all_vol = np.zeros(pCount_max); all_mass = np.zeros(pCount_max)
pid=0
for e in range(pmesh.elemCount):
    sctr=pmesh.element[e,:]; pts=pmesh.node[sctr,:]
    for q in range(len(W)):
        N,dNdxi=lagrange_basis_Q4(Q[q,:]); J0=dNdxi.T@pts; detJ0=np.linalg.det(J0); a=W[q]*detJ0
        all_pos[pid,:]=N.T@pts; all_vol[pid]=a; all_mass[pid]=a*rho; pid+=1
notch_length=0.05; notch_row_lo=numy//2-1; notch_row_hi=numy//2
def _in_notch(x,y):
    iy=int(np.floor(y/hy)); return (x<=notch_length) and (iy==notch_row_lo or iy==notch_row_hi)
keep=np.array([not _in_notch(all_pos[p,0],all_pos[p,1]) for p in range(pCount_max)],dtype=bool)
pCount=int(np.sum(keep)); particles=ParticleSet(pCount)
particles.positions[:]=all_pos[keep]; particles.volume[:]=all_vol[keep]; particles.mass[:]=all_mass[keep]
particles.deformation_gradient[:]=np.tile([1.,0.,0.,1.],(pCount,1)); particles.set_initial_state()
particles.phase[:]=0.0; particles.history[:]=0.0; alpha_p=np.zeros(pCount)
sigma=float(os.environ.get("PF_SIGMA",str(1.0e6))); neumann_traction_y=np.zeros(pCount)
for p in range(pCount):
    iy=int(np.floor(particles.positions[p,1]/hy)); iy=max(0,min(iy,numy-1))
    if iy==numy-1: particles.neumann_particles[p]=True; neumann_traction_y[p]=sigma/hy
    elif iy==0: particles.neumann_particles[p]=True; neumann_traction_y[p]=-sigma/hy
particles.pElems,particles.mpoints=build_particle_element_map(particles.positions,mesh)
node_state=NodeState(mesh.nodeCount)
C_max=float(np.max(np.linalg.eigvalsh(C_aniso(0.0)))); c_d=np.sqrt(C_max/rho); dt_u=h_e/c_d
ndim=2
eta_d=2.0*ndim*gc_d*ell_d/(h_e*c_d); eta_alpha=2.0*ndim*gc_alpha*ell_alpha*(1.0+xi)/(h_e*c_d)
dt_d_crit=eta_d*h_e**2/(2*ndim*gc_d*ell_d); dt_a_crit=eta_alpha*h_e**2/(2*ndim*gc_alpha*ell_alpha*(1+xi))
CFL=float(os.environ.get("PF_CFL","0.5")); dtime=CFL*min(dt_u,dt_d_crit,dt_a_crit)
nsteps=int(os.environ.get("PF_NSTEPS","4"))
print(f"dt_u={dt_u:.4e} dt_d={dt_d_crit:.4e} dt_a={dt_a_crit:.4e} eta_d={eta_d:.3e} eta_a={eta_alpha:.3e}")
shutil.rmtree('vtk_output_aniso',ignore_errors=True)
tol=1e-24; tol_pm=1e-30; vtk_dir='vtk_output_aniso'; vtk_interval=1
ta,ka,sa,dda,aa=[],[],[],[],[]; vtk_entries=[]; I_mat=np.eye(2); mpoints=particles.mpoints; t=0.0
alpha_grid=np.zeros(mesh.nodeCount); alpha_num=np.zeros(mesh.nodeCount); F_alpha=np.zeros(mesh.nodeCount)
for istep in range(nsteps):
    node_state.reset(); alpha_grid.fill(0.); alpha_num.fill(0.); F_alpha.fill(0.)
    for e in range(mesh.elemCount):
        esctr=mesh.element[e]
        for pid in mpoints[e]:
            stress=particles.stress[pid]; d_p=particles.phase[pid]; a_p=alpha_p[pid]; V0=particles.initial_volume[pid]
            for idn in esctr:
                dx=particles.positions[pid]-mesh.node[idn]; N,dNdx=get_mpm2d_shape(dx,hx,hy)
                node_state.mass[idn]+=N*particles.mass[pid]
                node_state.momentum[idn]+=N*particles.mass[pid]*particles.velocities[pid]
                node_state.internal_force[idn,0]-=particles.volume[pid]*(stress[0]*dNdx[0]+stress[2]*dNdx[1])
                node_state.internal_force[idn,1]-=particles.volume[pid]*(stress[2]*dNdx[0]+stress[1]*dNdx[1])
                if particles.neumann_particles[pid]: node_state.external_force[idn,1]+=neumann_traction_y[pid]*N*particles.volume[pid]
                node_state.pseudo_mass[idn]+=N*V0; node_state.phase_num[idn]+=N*V0*d_p; alpha_num[idn]+=N*V0*a_p
    has_pm=node_state.pseudo_mass>tol_pm
    node_state.phase[has_pm]=node_state.phase_num[has_pm]/node_state.pseudo_mass[has_pm]
    alpha_grid[has_pm]=alpha_num[has_pm]/node_state.pseudo_mass[has_pm]
    for e in range(mesh.elemCount):
        esctr=mesh.element[e]
        for pid in mpoints[e]:
            V0=particles.initial_volume[pid]; d_p=particles.phase[pid]; H_p=particles.history[pid]; eps_p=particles.strain[pid]
            grad_d=np.zeros(2); grad_a=np.zeros(2); shp=[]
            for idn in esctr:
                dx=particles.positions[pid]-mesh.node[idn]; N,dNdx=get_mpm2d_shape(dx,hx,hy)
                grad_d+=dNdx*node_state.phase[idn]; grad_a+=dNdx*alpha_grid[idn]; shp.append((idn,N,dNdx))
            g_d=(1.0-d_p)**2; G_alpha=(1.0-alpha_p[pid])*g_d*float(eps_p@(M2@eps_p)); ogA=omega_a@grad_a
            for idn,N,dNdx in shp:
                node_state.damage_force[idn]+=((gc_d/ell_d)*d_p*N+gc_d*ell_d*(dNdx@grad_d)-2.0*(1.0-d_p)*N*H_p)*V0
                F_alpha[idn]+=((gc_alpha/ell_alpha)*alpha_p[pid]*N+gc_alpha*ell_alpha*(dNdx@ogA)-G_alpha*N)*V0
    node_state.momentum+=(node_state.internal_force+node_state.external_force)*dtime
    d_grid_new=node_state.phase.copy(); a_grid_new=alpha_grid.copy()
    d_grid_new[has_pm]-=(dtime/eta_d)*node_state.damage_force[has_pm]/node_state.pseudo_mass[has_pm]
    a_grid_new[has_pm]-=(dtime/eta_alpha)*F_alpha[has_pm]/node_state.pseudo_mass[has_pm]
    for e in range(mesh.elemCount):
        for pid in mpoints[e]:
            Lp=np.zeros((2,2)); d_new=0.0; a_new=0.0
            for idn in mesh.element[e]:
                dx=particles.positions[pid]-mesh.node[idn]; N,dNdx=get_mpm2d_shape(dx,hx,hy)
                if node_state.mass[idn]>tol:
                    particles.velocities[pid]+=dtime*N*(node_state.internal_force[idn]+node_state.external_force[idn])/node_state.mass[idn]
                    particles.positions[pid]+=dtime*N*node_state.momentum[idn]/node_state.mass[idn]
                    vI=node_state.momentum[idn]/node_state.mass[idn]
                else: vI=np.zeros(2)
                Lp+=np.outer(vI,dNdx); d_new+=N*d_grid_new[idn]; a_new+=N*a_grid_new[idn]
            d_new=min(max(d_new,0.),1.); a_new=min(max(a_new,0.),1.)
            d_new=max(particles.phase[pid],d_new); a_new=max(alpha_p[pid],a_new)
            particles.phase[pid]=d_new; alpha_p[pid]=a_new
            F=(I_mat+Lp*dtime)@particles.deformation_gradient[pid].reshape(2,2)
            particles.deformation_gradient[pid]=F.reshape(4); particles.volume[pid]=np.linalg.det(F)*particles.initial_volume[pid]
            dEps=0.5*dtime*(Lp+Lp.T); dEps_vec=np.array([dEps[0,0],dEps[1,1],2.0*dEps[0,1]]); particles.strain[pid]+=dEps_vec
            eps_vec=particles.strain[pid]; sig0=C_aniso(a_new)@eps_vec
            particles.eff_stress[pid]=sig0; particles.stress[pid]=((1.0-d_new)**2)*sig0
            psi=0.5*float(eps_vec@sig0)
            if psi>particles.history[pid]: particles.history[pid]=psi
    ta.append(t); ka.append(0.0); sa.append(0.0); dda.append(float(np.max(particles.phase))); aa.append(float(np.max(alpha_p)))
    if istep%vtk_interval==0:
        fn=write_particles_vtp_aniso(particles.positions,particles.velocities,particles.stress,particles.phase,alpha_p,particles.history,istep,t,vtk_dir)
        vtk_entries.append((t,fn))
    _,mpoints=build_particle_element_map(particles.positions,mesh); t+=dtime
np.savez_compressed(os.path.join(vtk_dir,'checkpoint.npz'),
    pCount=np.array(pCount),positions=particles.positions,velocities=particles.velocities,mass=particles.mass,
    volume=particles.volume,deformation_gradient=particles.deformation_gradient,stress=particles.stress,
    eff_stress=particles.eff_stress,strain=particles.strain,phase=particles.phase,alpha=alpha_p,history=particles.history,
    initial_positions=particles.initial_positions,initial_volume=particles.initial_volume,
    neumann_particles=particles.neumann_particles,neumann_traction_y=neumann_traction_y,
    theta_deg=np.array(theta_deg),xi=np.array(xi),eta_d=np.array(eta_d),eta_alpha=np.array(eta_alpha),
    t=np.array(t),istep_start=np.array(nsteps),ta=np.array(ta),ka=np.array(ka),sa=np.array(sa),
    dda=np.array(dda),aa=np.array(aa),vtk_times=np.array([e[0] for e in vtk_entries]),
    vtk_fnames=np.array([e[1] for e in vtk_entries]))
write_pvd(vtk_dir,vtk_entries)
print("MAIN done maxd",max(dda),"maxa",max(aa),"steps",nsteps)
