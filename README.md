# mpm-3pb
A uni project for simulating 3 point bending with the material point method in python

![damage](screenshots/damage_3pb)

![damage](screenshots/von_misses_3pb)

# MUSL with Elasto-Plastic Constitutive Model (Johnson-Cook + Mie-Grüneisen)

---

## Algorithm: Solution Procedure of Explicit MPM with Elasto-Plastic Material (MUSL-EP)

---

**1: Initialization**

**2:** Set up the, set time $t = 0$

**3:** Set up particle data: $\mathbf{x}_p^0,\ \mathbf{v}_p^0,\ \boldsymbol{\sigma}_p^0,\ \mathbf{F}_p^0,\ V_p^0,\ m_p,\ \rho_p^0$

**4:** Set up constitutive data per particle: $\varepsilon_p^{p,0} = 0,\ D_p^0 = 0,\ D_{\text{init},p}^0 = 0,\ {\boldsymbol{\sigma}'}^{d,0}_p = \mathbf{0},\ e_p^0 = C_v \rho_p^0 (T^0 - T_r)$

**5: end**

---

**6: while** $t < t_f$ **do**

---

### Reset Grid

**7:** Reset grid quantities: $m_I^t = 0,\ (m\mathbf{v})_I^t = \mathbf{0},\ \mathbf{f}_I^{\text{ext},t} = \mathbf{0},\ \mathbf{f}_I^{\text{int},t} = \mathbf{0}$

---

### Mapping from Particles to Nodes (P2G)

**8:** Compute nodal mass: $m_I^t = \sum_p \phi_I(\mathbf{x}_p^t)\, m_p$

**9:** Compute nodal momentum: $(m\mathbf{v})_I^t = \sum_p \phi_I(\mathbf{x}_p^t)\, (m\mathbf{v})_p^t$

**10:** Compute external force: $\mathbf{f}_I^{\text{ext},t} = \sum_p \phi_I(\mathbf{x}_p)\, m_p\, \mathbf{b}(\mathbf{x}_p)$

**11:** Compute internal force: $\mathbf{f}_I^{\text{int},t} = -\sum_p V_p^t\, \boldsymbol{\sigma}_p^t\, \nabla\phi_I(\mathbf{x}_p^t)$

**12:** Compute nodal force: $\mathbf{f}_I^t = \mathbf{f}_I^{\text{ext},t} + \mathbf{f}_I^{\text{int},t}$

**13: end**

---

### Update the Momenta

**14:** Update momenta: $(m\tilde{\mathbf{v}})_I^{t+\Delta t} = (m\mathbf{v})_I^t + \mathbf{f}_I^t\, \Delta t$

**15:** Fix Dirichlet nodes $I$ e.g. $(m\mathbf{v})_I^t = \mathbf{0}$ and $(m\tilde{\mathbf{v}})_I^{t+\Delta t} = \mathbf{0}$

---

### Update Particle Velocities and Grid Velocities (Double Mapping)

**16:** Get nodal velocities: $\tilde{\mathbf{v}}_I^{t+\Delta t} = (m\tilde{\mathbf{v}})_I^{t+\Delta t} / m_I^t$

**17:** Update particle positions: $\mathbf{x}_p^{t+\Delta t} = \mathbf{x}_p^t + \Delta t \sum_I \phi_I(\mathbf{x}_p^t)\, \tilde{\mathbf{v}}_I^{t+\Delta t}$

**18:** Update particle velocities: $\mathbf{v}_p^{t+\Delta t} = \alpha\!\left(\mathbf{v}_p^t + \sum_I \phi_I(\mathbf{x}_p^t)\!\left[\tilde{\mathbf{v}}_I^{t+\Delta t} - \mathbf{v}_I^t\right]\right) + (1-\alpha)\sum_I \phi_I(\mathbf{x}_p^t)\,\tilde{\mathbf{v}}_I^{t+\Delta t}$

**19:** Update grid velocities: $(m\mathbf{v})_I^{t+\Delta t} = \sum_p \phi_I(\mathbf{x}_p^t)\,(m\mathbf{v})_p^{t+\Delta t}$

**20:** Fix Dirichlet nodes $(m\mathbf{v})_I^{t+\Delta t} = \mathbf{0}$

**21: end**

---

### Update Particles (G2P) and Constitutive Update

**22:** Get nodal velocities: $\mathbf{v}_I^{t+\Delta t} = (m\mathbf{v})_I^{t+\Delta t} / m_I^t$

**23:** Compute velocity gradient: $\mathbf{L}_p^{t+\Delta t} = \sum_I \nabla\phi_I(\mathbf{x}_p^t)\, \mathbf{v}_I^{t+\Delta t}$

**24:** Update deformation gradient: $\mathbf{F}_p^{t+\Delta t} = \left(\mathbf{I} + \mathbf{L}_p^{t+\Delta t}\,\Delta t\right)\mathbf{F}_p^t$

**25:** Update volume: $V_p^{t+\Delta t} = \det\mathbf{F}_p^{t+\Delta t}\, V_p^0$

**26:** Update density: $\rho_p^{t+\Delta t} = m_p / V_p^{t+\Delta t}$

**27:** Strain rate and polar decomposition: $\mathbf{D}_p = 0.5(\mathbf{L}_p^{t+\Delta t} + {\mathbf{L}_p^{t+\Delta t}}^{\mathrm{T}})$, $\quad\mathbf{F}_p^{t+\Delta t} = \mathbf{R}_p\mathbf{U}_p$ via SVD $\mathbf{F} = \mathbf{U}_s\boldsymbol{\Sigma}\mathbf{V}^T \Rightarrow \mathbf{R}_p = \mathbf{U}_s\mathbf{V}^T$

**28:** Un-rotated strain rate and deviatoric part: $\mathbf{d}_p = \mathbf{R}_p^{\mathrm{T}}\mathbf{D}_p\mathbf{R}_p$, $\quad\mathbf{d}_p^d = \mathbf{d}_p - \tfrac{1}{3}\mathrm{tr}(\mathbf{d}_p)\,\mathbf{I}$, $\quad\dot{\varepsilon}_p = \sqrt{\tfrac{2}{3}\,\mathbf{d}_p^d:\mathbf{d}_p^d}$

**29: Plasticity update** — compute ${\boldsymbol{\sigma}'}_{p,t+\Delta t}^{d}$ and $\varepsilon_p^{p,t+\Delta t}$

> **29a.** Damaged shear modulus: $G' = (1 - D_p^t)\,G$

> **29b.** Elastic trial deviatoric stress: ${\boldsymbol{\sigma}'}_{\text{trial}}^{d} = {\boldsymbol{\sigma}'}_{p,t}^{d} + 2G'\,\Delta t\,\mathbf{d}_p^d$

> **29c.** Trial von Mises stress (plane strain — the out-of-plane deviatoric component $s_{zz}^d = -(s_{xx}^d+s_{yy}^d)$ is included in the contraction): $\sigma_{\text{trial}}'^{\,\text{eq}} = \sqrt{\tfrac{3}{2}\,{\boldsymbol{\sigma}'}_{\text{trial}}^{d} : {\boldsymbol{\sigma}'}_{\text{trial}}^{d}}$

> **29d.** JC flow stress, with $\dot{\varepsilon}_p^* = \max(\dot{\varepsilon}_p/\dot{\varepsilon}_0,\,1)$ so that the rate term is always $\geq 1$: $\sigma_f = \left[A + B\!\left(\varepsilon_p^{p,t}\right)^n\right]\!\left[1 + C\ln\dot{\varepsilon}_p^*\right]\!\left[1-(T^*)^m\right](1-D_p^t)$

> **29e. If** $\sigma_{\text{trial}}'^{\,\text{eq}} < \sigma_f$ **(elastic):** ${\boldsymbol{\sigma}'}_{p,t+\Delta t}^{d} = {\boldsymbol{\sigma}'}_{\text{trial}}^{d}$, $\quad\varepsilon_p^{p,t+\Delta t} = \varepsilon_p^{p,t}$

> **29f. Else (plastic — radial return):** $\Delta\varepsilon_p = (\sigma_{\text{trial}}'^{\,\text{eq}} - \sigma_f)/(3G')$, $\quad\varepsilon_p^{p,t+\Delta t} = \varepsilon_p^{p,t} + \Delta\varepsilon_p$, $\quad{\boldsymbol{\sigma}'}_{p,t+\Delta t}^{d} = (\sigma_f / \sigma_{\text{trial}}'^{\,\text{eq}})\,{\boldsymbol{\sigma}'}_{\text{trial}}^{d}$

> **29g. End if**

**30: Pressure update** — compute $\hat{p}_{p,t+\Delta t}$ via EOS

> Internal energy increment — from $e = C_v\rho_0(T-T_r)$ and $C_p \approx C_v$:
>
> $$\Delta e_p = \rho_0 C_v\,\Delta T_p = \rho_0 C_v\cdot\frac{\chi\,\sigma_f\,\Delta\varepsilon_p}{\rho_p^{t+\Delta t}\,C_p} \approx \frac{\rho_0}{\rho_p^{t+\Delta t}}\,\chi\,\sigma_f\,\Delta\varepsilon_p = \det(\mathbf{F}_p)\,\chi\,\sigma_f\,\Delta\varepsilon_p$$
>
> $e_p = e_p^{old}+\Delta e_p$

> *Mie-Grüneisen:* $\eta_p = \rho_p^{t+\Delta t}(1-D_p^t)/\rho_p^0\ (\hat{p}>0)$, $\quad\eta_p = \rho_p^{t+\Delta t}/\rho_p^0\ \text{(otherwise)}$, $\quad-\hat{p}_p = \dfrac{\rho_0(1-D_p^t)c_0^2(\eta_p-1)\!\left[\eta_p - \frac{\Gamma_0}{2}(\eta_p-1)\right]}{\left[\eta_p - S_\alpha(\eta_p-1)\right]^2} + \Gamma_0\,e_p$

> *Linear EOS (alternative):* $\hat{p}_p = -K(1-\det\mathbf{F}_p^{t+\Delta t})(1-D_p^t)$

**31:** Assemble and rotate stress: $\boldsymbol{\sigma'}_{p,t+\Delta t} = {\boldsymbol{\sigma}'}_{p,t+\Delta t}^{d} + \hat{p}_{p,t+\Delta t}\,\mathbf{I}$, $\quad\boldsymbol{\sigma}_{p,t+\Delta t} = \mathbf{R}_p\,\boldsymbol{\sigma'}_{p,t+\Delta t}\,\mathbf{R}_p^{\mathrm{T}}$

**32:** Adiabatic temperature update (if active): $\Delta T_p = \chi\,\sigma_f\,\Delta\varepsilon_p\,/\,(\rho_p^{t+\Delta t} C_p)$, $\quad T_p^{t+\Delta t} = T_p^t + \Delta T_p$



**33: Damage update** — compute $D_p^{t+\Delta t}$

> **33a.** Stress triaxiality: $\sigma^* = -\hat{p}_{p,t+\Delta t} / \sigma_{\text{eq},p}^{t+\Delta t}$

> **33b.** JC strain at failure, with $\dot{\varepsilon}_p^* = \max(\Delta\varepsilon_p/(\Delta t\,\dot{\varepsilon}_0),\,1)$: $\varepsilon_f = \left[D_1 + D_2\exp(D_3\sigma^*)\right]\!\left[1 + D_4\ln\dot{\varepsilon}_p^*\right]\!\left[1 + D_5 T^*\right]$
 
> **33c.** Damage initiation variable: $D_{\text{init},p}^{t+\Delta t} = D_{\text{init},p}^{t} + \Delta\varepsilon_p / \varepsilon_f$

> **33d. If** $D_{\text{init},p}^{t+\Delta t} \geq 1$: $D_p^{t+\Delta t} = \min\!\left(10(D_{\text{init},p}^{t+\Delta t} - 1),\ 1\right)$

> **33e. Else:** $D_p^{t+\Delta t} = 0$

> **33f. End if**

**34: end**

---

### Advance Time

**35:** Advance time $t = t + \Delta t$

**36: Error calculation**: if needed (e.g. for convergence tests)

**37: end while**

---

## Variable Reference

| Symbol | Description |
|---|---|
| $\phi_I(\mathbf{x}_p^t)$ | Shape function of node $I$ evaluated at particle $p$ |
| $\mathbf{F}_p$ | Deformation gradient |
| $\mathbf{R}_p,\ \mathbf{U}_p$ | Rotation and stretch tensors from polar decomposition of $\mathbf{F}_p$ |
| $\mathbf{D}_p$ | Symmetric strain rate tensor |
| $\mathbf{d}_p^d$ | Un-rotated deviatoric strain rate |
| ${\boldsymbol{\sigma}'}^d_p$ | Un-rotated deviatoric Cauchy stress |
| $\varepsilon_p^p$ | Equivalent plastic strain |
| $\dot{\varepsilon}_p$ | Equivalent strain rate $= \sqrt{\tfrac{2}{3}\mathbf{d}_p^d:\mathbf{d}_p^d}$ |
| $\dot{\varepsilon}_p^*$ | Normalized strain rate $= \max(\dot{\varepsilon}_p / \dot{\varepsilon}_0,\,1)$ |
| $T^*$ | Homologous temperature $= (T - T_r)/(T_m - T_r)$ |
| $A, B, n, C, m$ | Johnson-Cook flow stress constants |
| $D_1,\ldots,D_5$ | Johnson-Cook damage constants |
| $D_{\text{init}}$ | Damage initiation variable |
| $D$ | Damage variable $\in [0,1]$ |
| $G$ | Shear modulus |
| $K$ | Bulk modulus |
| $c_0$ | Bulk speed of sound |
| $\Gamma_0$ | Grüneisen Gamma (reference state) |
| $S_\alpha$ | Linear Hugoniot slope coefficient |
| $\chi$ | Taylor-Quinney coefficient |
| $\alpha$ | FLIP/PIC blending parameter |

---

## Notes

- The **double mapping** (steps 16–21) is the distinguishing feature of MUSL vs USL/USF.
- The **polar decomposition** in step 27 removes rigid-body rotations before the constitutive update, making the hypoelastic stress integration objective.
- The **radial return** in step 29f is exact for von Mises plasticity with isotropic hardening under the assumption that $\sigma_f$ is constant during the return step.
- The **damage variable** $D$ enters the shear modulus ($G' = (1-D)G$), the EOS, and the flow stress, providing full coupling between damage and the mechanical response.
