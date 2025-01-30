# Fe-Mg diffusion in olvine 

Simon Matthews, University of Iceland (simonm@hi.is)

Implementation of the model outlined by Dauphas et al. (2010) for calculating the fractionation of Mg and Fe isotopes during diffusion.

## Notes

The equation to be solved:

$ \frac{\partial X(t, r)}{\partial t}  = \frac{1}{r^2} \frac{\partial}{\partial r} \left[ r^2 \, \, \alpha \, \, D(T, P, X, fO_2(T, P)) \, \, \frac{\partial X(t, r)}{\partial r} \right]$

With boundary conditions:

$X(0, r) = X_{Fe0}$

$X(t, a) = \frac{1}{1+t} (X_{Fe0} - X_{Fe1} + X_{Fe1}) $

*Note that this equation is not implemented, but it doesn't seem to affect the answers, might make the first time steps more numerically stable*

$\frac{\partial X(t, r=0)}{\partial r} = 0 $

## For 1D diffusion

The equations required for Crank Nicholson scheme:

If $D$ were constant:

$ \frac{u_j^{n+1} - u_j^n}{\Delta t} = \frac{D}{2} \left[ \frac{\left( u^{n+1}_{j+1} - 2u_j^{n+1} + u^{n+1}_{j-1} \right) + \left(u^n_{j+1} - 2 u^n_j + u^n_{j-1} \right)}{\left(\Delta x \right)^2} \right]$ 

where $n$ is the time step and $j$ is the spatial step.

Incorporating the variation of $D$ with $r$:

$ \frac{u_j^{n+1} - u_j^n}{\Delta t} = \frac{1}{2} \left[ \frac{D_{j+1/2} \left[ \left( u^{n+1}_{j+1} - u_j^{n+1} \right) + \left( u^{n}_{j+1} - u_j^{n} \right) \right] - D_{j-1/2} \left[ \left( u_j^{n+1} - u^{n+1}_{j-1} \right) + \left( u_j^{n} - u^{n}_{j-1} \right) \right]}{\left(\Delta x \right)^2} \right]$ 

where,

$ D_{j+1/2} = \frac{1}{2} \left[ D(u^{n}_{j+1}) + D(u^n_j) \right] $

This can be rearranged, setting:

$ \kappa = \frac{\Delta t}{2(\Delta x)^2} $

then:

$ u^{n+1}_{j-1} \left( \kappa D^{n+1}_{j-1/2} \right) - u_j^{n+1} \left( 1 + \kappa \left[ D^{n+1}_{j+1/2} + D^{n+1}_{j-1/2} \right] \right) + \kappa u^{n+1}_{j+1} D^{n+1}_{j+1/2}$

$ =  - u_{j-1}^n \left( \kappa D_{j-1/2}^n \right) - u_j^n \left( 1 - \kappa \left[ D^n_{j+1/2} + D^n_{j-1/2} \right] \right) - \kappa u^n_{j+1} D^n_{j+1/2}$

which can be set up as a set of linear equations:
 
For each time step $n$:

$
\begin{bmatrix}
  -(1 + \kappa (D_{1+1/2}^{n+1} + D_{1 - 1/2}^{n+1})) & \kappa D^{n+1}_{1+1/2} & 0 & ...\\
 \kappa D_{2 - 1/2}^{n+1} & - (1 + \kappa (D_{2+1/2}^{n+1} + D_{2 - 1/2}^{n+1})) & \kappa D^{n+1}_{2 - 1/2} & ... \\
 ... & ... & ... & ... 
 \end{bmatrix}
 \begin{bmatrix}
 u_1^{n+1} \\ u_2^{n+1} \\ ...
 \end{bmatrix}
 =\begin{bmatrix}
  -u_1^{n} \kappa D_{1+1/2}^{n} - u_0^{n} ( 1 - \kappa (D^n_{0+1/2} + D^n_{0+1/2})) - u^{n}_{1} \kappa D^n_{1+1/2} \\ 
  -u_0^{n} \kappa D_{1 - 1/2}^{n} - u_1 (1 - \kappa (D_{1+1/2}^{n} + D_{1-1/2}^{n})) - u_2^{n} \kappa D_{2+1/2}^{n} \\ 
  ...
 \end{bmatrix}
 $

Where the first term on the RHS reflects the symmetric boundary condition

## For diffusion with spherical symmetry

- Divide $\kappa$, or perhaps $\kappa_j$ by $r_j^2$
- Multiply $D_{j+1/2}^{n+1}$ and $D_{j-1/2}^{n+1}$ by $r_j^2$ (actually the appropriate $j$ values inside the brackets). I think this is fine, because it is effectively using the treatment of D as a variable that depends on $r$.

For the latter change, this doesn't cancel out:

$ \frac{\kappa}{r^2_j} \cdot \frac{1}{2} \left( r_{j+1}^2 D^n_{j+1} + r_j^2 D^n_{j} \right) = \kappa \cdot \frac{1}{2} \left( \frac{r_{j+1}^2}{r_j^2} D^n_{j+1} + D_j^n \right)$

Then taking the fraction from inside the bracket:

$ \frac{r_{j+1}^2}{r_j^2} = \frac{(r_j + \Delta r)^2}{r_j^2} = \frac{r_j^2 + 2 (\Delta r) r_j + (\Delta r)^2}{r_j^2} = 1 + \frac{2 \Delta r}{r_j} + \frac{(\Delta r)^2}{r_j^2} $

Substituting back in:

$\kappa \cdot \frac{1}{2} \left( \left[ 1 + \frac{2 \Delta r}{r_j} + \frac{(\Delta r)^2}{r_j^2} \right] D^n_{j+1} + D_j^n \right)$

This will generally increase the effective diffusivity, which is exactly what we would expect for diffusion in a sphere.
