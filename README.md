# MPM-PHASE-FIELD
Implementation of phase field fracture and anisotropy with the material point method (MPM).

# PHASE FIELD
## Weak form of the damage sub–problem
Using the degradation function $g(\phi)=(1-\phi)^2$:

$$\int_{\Omega_0} \left\lbrace -2(1-\phi)\,\mathcal{H}\,\delta\phi + \frac{G_c}{\ell}\,\phi\,\delta\phi + G_c\ell\,\nabla\phi\cdot\nabla\delta\phi + \eta\,\dot{\phi}\,\delta\phi \right\rbrace\,\mathrm{d}V = 0$$

After discretization:

$$\int_{\Omega_0} \left\lbrace -2(1-\phi)\,N_I\,\mathcal{H} + \frac{G_c}{\ell}\,\phi\,N_I + G_c\ell\,\nabla N_I\cdot\nabla\phi + \eta\,\dot{\phi}\,N_I \right\rbrace\,\mathrm{d}V = 0$$

The field variable can be updated explicitly:

$$\phi_I^{\,t+\Delta t} \;=\; \phi_I^{\,t} \;-\; \frac{\Delta t}{\eta\,m_I}\,F_I$$

$$m_I := \int_{\Omega_0} N_I\,\mathrm{d}V$$

$$F_I = \int_{\Omega_0} \left\lbrace -2(1-\phi^t)\,N_I\,\mathcal{H} + \frac{G_c}{\ell}\,\phi^t\,N_I + G_c\ell\,\nabla N_I\cdot\nabla_0\phi^{t} \right\rbrace\,\mathrm{d}V$$

# ANISOTROPIC PHASE FIELD
In this formulation, there are two phase fields, the bulk fracture PF $d$ and the interfacial fracture PF $a$.
## Weak forms

$$\int_{\Omega_0} \left\lbrace \eta_d\,\dot{d}\,\delta d + \frac{g_c^{d}}{\ell_d}\,d\,\delta d + g_c^{d}\ell_d\,\nabla d\cdot\nabla\delta d - 2(1-d)\,\mathcal{H}\,\delta d \right\rbrace\,\mathrm{d}V = 0$$

$$\int_{\Omega_0} \left\lbrace \eta_\alpha\,\dot{\alpha}\,\delta\alpha + \frac{g_c^{\alpha}}{\ell_\alpha}\,\alpha\,\delta\alpha + g_c^{\alpha}\ell_\alpha\,\left(\boldsymbol{\omega}^{\alpha}\nabla\alpha\right)\cdot\nabla\delta\alpha + \frac{1}{2}g(d)\,\boldsymbol{\varepsilon}\!:\!\frac{\partial \mathcal{C}_\alpha}{\partial \alpha}\!:\!\boldsymbol{\varepsilon}\,\delta\alpha \right\rbrace\,\mathrm{d}V = 0$$

And after discretization the field variables can be updated explicitly:

$$d_I^{\,t+\Delta t}=d_I^{t}-\frac{\Delta t}{\eta_d\,m_I^{d}}F_I^{d}, \qquad \alpha_I^{\,t+\Delta t}=\alpha_I^{t}-\frac{\Delta t}{\eta_\alpha\,m_I^{\alpha}}F_I^{\alpha}.$$

$$F_I^{d} = \int_{\Omega_0} \left\lbrace \frac{g_c^{d}}{\ell_d}\,d^{t}\,N_I + g_c^{d}\ell_d\,\nabla N_I\cdot\nabla d^{t} - 2(1-d^{t})\,N_I\,\mathcal{H} \right\rbrace\,\mathrm{d}V$$

$$F_I^{\alpha} = \int_{\Omega_0} \left\lbrace \frac{g_c^{\alpha}}{\ell_\alpha}\,\alpha^{t}\,N_I + g_c^{\alpha}\ell_\alpha\,\nabla N_I\cdot\left(\boldsymbol{\omega}^{\alpha}\nabla\alpha^{t}\right) + \frac{1}{2}\,g(d^{t})\,\boldsymbol{\varepsilon}\!:\!\frac{\partial \mathcal{C}_\alpha}{\partial \alpha}\!:\!\boldsymbol{\varepsilon}\,N_I \right\rbrace\,\mathrm{d}V$$

$$m_I^{d}=m_I^{\alpha}=\int_{\Omega_0}N_I\,\mathrm{d}V \;\xrightarrow{\text{MPM}}\; \sum_p N_I(\mathbf{x}_p)\,V_p^0,$$

## Symbol Appendix

| Symbol | Description | SI units |
|---|---|---|
| $\eta$, $\eta_d$, $\eta_\alpha$ | (Artificial) viscosity — regularizes the explicit update | Pa·s |
| $G_c$, $g_c^{d}$, $g_c^{\alpha}$ |Fracture toughness | J/m² |
| $\ell$, $\ell_d$, $\ell_\alpha$ | Phase-field regularization length scale | m |
| $\mathcal{H}$ | Crack driving force (history variable, max. elastic strain-energy density) | Pa |
| $\boldsymbol{\omega}^{\alpha}$ | Anisotropy/orientation structure tensor   | - |
| $\mathcal{C}_\alpha$ | Anisotropic elastic stiffness tensor | Pa |