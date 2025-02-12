# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     custom_cell_magics: kql
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: base
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Diffusion Model from Dauphas et al. (2010)
# This file is an attempt at converting the model to python. I have tried to make it as direct a translation as possible, so things are not necessarily written in the way that would be best in python, but in a way that can be linked directly back to the mathematica script provided.
#
# Simon Matthews (simonm@hi.is)
# September 2024

# %%
import numpy as np
import sympy as sym
import matplotlib.pyplot as plt

# %% [markdown]
# ## Fe-Mg isotope geospeedometry in olivine
#
# ### Constants

# %%
R = 8.314472

# %%
cooling = 'linear' # linear or exponential
if cooling == 'linear':
    q = 1
else:
    q = 0
c = 30 # Cooling parameter for K d-1 for linear cooling, in d-1 for exponential cooling
T0 = 1853 # initial temperature in K
beta_Mg = 0.05 # exponent in D2/D1=(m1/m2)**beta for Mg, Richter et al. 2009
beta_Fe = 0.05 # exponent in D2/D1=(m1/m2)**beta for Mg, Richter et al. 2009
delta_O2 = 0 # Log fO2 relative to the NNO buffer
P = 10**5 # Pressure in Pa
a = 500e-6 # m, grain radius
XFe1 = 0.11 # Surface Fe concentration, identical for 54 and 56
XMg1 = 0.89 # Surface Mg concentration, identical for 24 and 26
XFe0 = 0.06 # Initial Fe concentration, identical for 54 and 56
XMg0 = 0.94 # Initial Mg concentration, identical for 24 and 26


# %% [markdown]
# ### Definition of functions

# %%
def T(t):
    if cooling == 'linear':
        if T0 - t * c / (3600 * 24) > 298.15:
            return T0 - t * c / (3600 * 24)
        else:
            return 298.15
    else:
        return 298.15 + (T0 - 298.15) * np.exp(- t * c / 3600 * 24)


# %%
def fO2(T, P):
    """
    oxygen fugacity in Pa with T in K and P in Pa; Huebner & Sato 1970; Chou 1987; Herd 2008
    """
    return 101325 * 10**delta_O2 * 10**(9.36 - 24930 / T + 0.046 * (P / 1e5 - 1) / T)


# %%
def d(T, P, X, f):
    """
    Fe-Mg diffusion coefficient in olivine in m2s-1, T in K, P pressure in Pa, 
    f absolute oxygen fugacity in Pa, X mole fraction of the fayalite component
    Dohmen & Chakraborty 2007
    """
    if f > 10**-10:
        return (
            10**(-9.21 - (201000 + (P-1e5)*7e-6)/(2.303*R*T) 
                 + 1/6 * np.log10(f/1e-7) + 3*(X-0.1))
        )
    else:
        return (
            10**(-8.91 - (220000 + (P-1e5)*7e-6) / (2.303*R*T) + 3 * (X-0.1))
        )


# %% [markdown]
# ### Calculation of diffusion and cooling timescales

# %%
# Diffusion timescale in seconds:
tau = a**2 / d(T0, P, XFe0, fO2(T0,P))

if cooling == 'linear':
    initial_cooling_rate = c
else:
    initial_cooling_rate = (T0 - 298.15) * c
print("initial cooling rate for " + cooling + " model: {:.4f} K.d-1".format(initial_cooling_rate))

if cooling == 'linear':
    denominator = c
else:
    denominator = (T0 - 298.15) * c
print("cooling timescale: {:.4f} d".format((T0 - 1173.15) / denominator))

print("diffusion timescale: {:.4f} d".format(tau / (3600 * 24)))


# %% [markdown]
# ## Solving the PDE
#
# Matlab has the ability to solve the diffusion equation. Unfortunately, neither scipy nor numpy have this ability. This means we must code up an algorithm to solve the equation. Here follows some notes that I produced in the process of working out how to do this.

# %% [markdown]
# The equation to be solved:
#
# $ \frac{\partial X(t, r)}{\partial t}  = \frac{1}{r^2} \frac{\partial}{\partial r} \left[ r^2 \, \, \alpha \, \, D(T, P, X, fO_2(T, P)) \, \, \frac{\partial X(t, r)}{\partial r} \right]$
#
# With boundary conditions:
#
# $X(0, r) = X_{Fe0}$
#
# $X(t, a) = \frac{1}{1+t} (X_{Fe0} - X_{Fe1} + X_{Fe1}) $
#
# $\frac{\partial X(t, r=0)}{\partial r} = 0 $

# %% [markdown]
# ## For 1D diffusion
#
# The equations required for Crank Nicholson scheme:
#
# If $D$ were constant:
#
# $ \frac{u_j^{n+1} - u_j^n}{\Delta t} = \frac{D}{2} \left[ \frac{\left( u^{n+1}_{j+1} - 2u_j^{n+1} + u^{n+1}_{j-1} \right) + \left(u^n_{j+1} - 2 u^n_j + u^n_{j-1} \right)}{\left(\Delta x \right)^2} \right]$ 
#
# where $n$ is the time step and $j$ is the spatial step.

# %% [markdown]
# Incorporating the variation of $D$ with $r$:
#
# $ \frac{u_j^{n+1} - u_j^n}{\Delta t} = \frac{1}{2} \left[ \frac{D_{j+1/2} \left[ \left( u^{n+1}_{j+1} - u_j^{n+1} \right) + \left( u^{n}_{j+1} - u_j^{n} \right) \right] - D_{j-1/2} \left[ \left( u_j^{n+1} - u^{n+1}_{j-1} \right) + \left( u_j^{n} - u^{n}_{j-1} \right) \right]}{\left(\Delta x \right)^2} \right]$ 
#
# where,
#
# $ D_{j+1/2} = \frac{1}{2} \left[ D(u^{n}_{j+1}) + D(u^n_j) \right] $

# %% [markdown]
# This can be rearranged, setting:
#
# $ \kappa = \frac{\Delta t}{2(\Delta x)^2} $
#
# then:
#
# $ u^{n+1}_{j-1} \left( \kappa D^{n+1}_{j-1/2} \right) - u_j^{n+1} \left( 1 + \kappa \left[ D^{n+1}_{j+1/2} + D^{n+1}_{j-1/2} \right] \right) + \kappa u^{n+1}_{j+1} D^{n+1}_{j+1/2}$
#
# $ =  - u_{j-1}^n \left( \kappa D_{j-1/2}^n \right) - u_j^n \left( 1 - \kappa \left[ D^n_{j+1/2} + D^n_{j-1/2} \right] \right) - \kappa u^n_{j+1} D^n_{j+1/2}$
#
# which can be set up as a set of linear equations:
#
# For each time step $n$:
#
# $
# \begin{bmatrix}
#  -(1 + \kappa (D_{1+1/2}^{n+1} + D_{1 - 1/2}^{n+1})) & \kappa D^{n+1}_{1+1/2} & 0 & ...\\
# \kappa D_{2 - 1/2}^{n+1} & - (1 + \kappa (D_{2+1/2}^{n+1} + D_{2 - 1/2}^{n+1})) & \kappa D^{n+1}_{2 - 1/2} & ... \\
# ... & ... & ... & ... 
# \end{bmatrix}
# \begin{bmatrix}
# u_1^{n+1} \\ u_2^{n+1} \\ ...
# \end{bmatrix}
# =\begin{bmatrix}
#  -u_1^{n} \kappa D_{1+1/2}^{n} - u_0^{n} ( 1 - \kappa (D^n_{0+1/2} + D^n_{0+1/2})) - u^{n}_{1} \kappa D^n_{1+1/2} \\ 
#  -u_0^{n} \kappa D_{1 - 1/2}^{n} - u_1 (1 - \kappa (D_{1+1/2}^{n} + D_{1-1/2}^{n})) - u_2^{n} \kappa D_{2+1/2}^{n} \\ 
#  ...
# \end{bmatrix}
# $
#
# Where the first term on the RHS reflects the symmetric boundary condition

# %% [markdown]
# ## For diffusion with spherical symmetry
#
# - Divide $\kappa$, or perhaps $\kappa_j$ by $r_j^2$
# - Multiply $D_{j+1/2}^{n+1}$ and $D_{j-1/2}^{n+1}$ by $r_j^2$ (actually the appropriate $j$ values inside the brackets). I think this is fine, because it is effectively using the treatment of D as a variable that depends on $r$.
#
# For the latter change, this doesn't cancel out:
#
# $ \frac{\kappa}{r^2_j} \cdot \frac{1}{2} \left( r_{j+1}^2 D^n_{j+1} + r_j^2 D^n_{j} \right) = \kappa \cdot \frac{1}{2} \left( \frac{r_{j+1}^2}{r_j^2} D^n_{j+1} + D_j^n \right)$
#
# Then taking the fraction from inside the bracket:
#
# $ \frac{r_{j+1}^2}{r_j^2} = \frac{(r_j + \Delta r)^2}{r_j^2} = \frac{r_j^2 + 2 (\Delta r) r_j + (\Delta r)^2}{r_j^2} = 1 + \frac{2 \Delta r}{r_j} + \frac{(\Delta r)^2}{r_j^2} $
#
# Substituting back in:
#
# $\kappa \cdot \frac{1}{2} \left( \left[ 1 + \frac{2 \Delta r}{r_j} + \frac{(\Delta r)^2}{r_j^2} \right] D^n_{j+1} + D_j^n \right)$
#
# This will generally increase the effective diffusivity, which is exactly what we would expect for diffusion in a sphere.
#

# %% [markdown]
#
# ## Older notes- need sorting
#
# This needs to be rearranged such that it has the format of a set of simultaneous linear equations at each time step $n$, solving for the step $n+1$:
#
#

# %% [markdown]
# `f4`, `f6`, `m4`, and `m6` is the solution to the diffusion pde for $^{54}Fe$, $^{56}Fe$, and $^{26}Mg$ respectively.
#
# `alpha` is a coefficient that accounts for differences in diffusivities of isotopes
#
# Note that the /. operator in mathematic means replace with, i.e., (x+10)/.x->1 is the same as (1+10)
#
# The D[...] operator is a partial derivative.

# %% [markdown]
# ## Code for running 1D calculations

# %%
def construct_linalg_problem_1d(u_prev, T_prev, kappa, alpha=1.0, Fe=True):
    """
    Construct the matrices describing the linear algebra problem
    for each time step.

    Note that at the present time the numerical handling of the compositional
    dependence of D is a fudge, though it likely makes a sufficiently small
    difference that this isn't a problem.

    Parameters
    ----------
    u_prev : numpy.ndarray
        The concentrations from the previous step. The last element of 
        u should represent the constant composition at the domain
        boundary.
    T_prev : float
        The temperature of the previous step
    x_boundary : float
        The composition at the boundary of the domain
    kappa : float
        The constant $\frac{\Delta t}{2(\Delta x)^2} where t is time
        and x is distance
    alpha : float, default: 1.0
        Adjustment factor for D calculated from beta and isotope mass ratios
    Fe : bool, default: True
        Fe or Mg?
    """
    usteps = np.shape(u_prev)[0]

    # LHS
    A = np.zeros([usteps, usteps])

    fo2_prev = fO2(T_prev, P)
    D_prev = np.zeros(np.shape(u_prev))
    for j in range(usteps):
        if Fe:
            D_prev[j] = d(T_prev, P, u_prev[j], fo2_prev)
        else:
            D_prev[j] = d(T_prev, P, 1.0 - u_prev[j], fo2_prev)
    D_prev = D_prev * alpha
    
    for j in range(usteps):
        if j == 0:
            A[j,j] = - (1 + kappa * (0.5 * (D_prev[j] + D_prev[j+1]) + 0.5 * (D_prev[j] + D_prev[j+1])))
            A[j, j+1] = kappa * 0.5 * (D_prev[j] + D_prev[j+1])*2

        elif j == usteps - 1:
            A[j,j] = 1 # The final composition is fixed, and therefore known already
            
        else:
            A[j, j-1] = kappa * 0.5 * (D_prev[j] + D_prev[j-1])
            A[j,j] = - (1 + kappa * (0.5 * (D_prev[j] + D_prev[j+1]) + 0.5 * (D_prev[j] + D_prev[j-1])))
            A[j, j+1] = kappa * 0.5 * (D_prev[j] + D_prev[j+1])


    

    # RHS
    B = np.zeros(np.shape(u_prev))

    for j in range(usteps):
        # for j=0, j+1 == j-1 by symmetry
        if j == 0:
            B[j] = (
                - u_prev[j+1] * kappa * 0.5 * (D_prev[j+1] + D_prev[j])
                - u_prev[j] * (1 - kappa * (0.5 * (D_prev[j+1] + D_prev[j]) + 0.5 * (D_prev[j] + D_prev[j+1])))
                - u_prev[j+1] * kappa * 0.5 * (D_prev[j+1] + D_prev[j])
            )

        elif j == usteps - 1:
            B[j] = u_prev[j]
        
        else:
            B[j] = (
                - u_prev[j-1] * kappa * 0.5 * (D_prev[j-1] + D_prev[j])
                - u_prev[j] * (1 - kappa * (0.5 * (D_prev[j+1] + D_prev[j]) + 0.5 * (D_prev[j] + D_prev[j-1])))
                - u_prev[j+1] * kappa * 0.5 * (D_prev[j+1] + D_prev[j])
            )
    
    return A, B




# %%
def run_diffusion_model_1d(alpha, Fe=True, xsteps=100, tsteps=10000):
    """

    Note that many of the parameters are being set by global variables.
    While this is not great python practice, it stays true to the format
    of the original Mathematica script this code is emulating.

    Parameters
    ----------
    alpha : float, default: 1.0
        Adjustment factor for D calculated from beta and isotope mass ratios
    Fe : bool, default: True
        Fe or Mg?
    """

    if Fe:
        X0 = XFe0
        X1 = XFe1
    else:
        X0 = XMg0
        X1 = XMg1

    # Set up starting arrays:
    u0 = np.array([X0]*(xsteps-1) + [X1])
    time = np.linspace(0, 3*tau, tsteps+1)

    x_results = np.zeros([tsteps+1, xsteps])
    x_results[0,:] = u0

    Tt = np.zeros(tsteps+1)
    for i in range(tsteps):
        Tt[i] = T(time[i])
        
    u_prev = u0

    kappa = (time[1] - time[0]) / (2 * (a / (xsteps-1))**2)
    print(np.log10(kappa))

    for i in range(tsteps):
        i+=1 # Run from i=1 to tsteps+1
        A, B = construct_linalg_problem_1d(u_prev, Tt[i-1], kappa, alpha=alpha, Fe=Fe)
        u = np.linalg.solve(A, B)
        x_results[i,:] = u
        u_prev = u
    
    return x_results


# %%
xFe54 = run_diffusion_model_1d((54.0/56.0)**beta_Fe)

# %%
fig, ax = plt.subplots()

for i in range(np.shape(xFe54)[0]):
    ax.plot(range(np.shape(xFe54)[1]), xFe54[i,])

plt.show()


# %% [markdown]
# ## Code for running 3D calculations

# %%
def construct_linalg_problem_3d(u_prev, T_prev, kappa, dr, alpha=1.0, Fe=True, d=d):
    """
    Construct the matrices describing the linear algebra problem
    for each time step.

    Note that at the present time the numerical handling of the compositional
    dependence of D is a fudge, though it likely makes a sufficiently small
    difference that this isn't a problem.

    Parameters
    ----------
    u_prev : numpy.ndarray
        The concentrations from the previous step. The last element of 
        u should represent the constant composition at the domain
        boundary.
    T_prev : float
        The temperature of the previous step
    x_boundary : float
        The composition at the boundary of the domain
    kappa : float
        The constant $\frac{\Delta t}{2(\Delta x)^2} where t is time
        and x is distance
    dr : float
        The stepsize in distance
    alpha : float, default: 1.0
        Adjustment factor for D calculated from beta and isotope mass ratios
    Fe : bool, default: True
        Fe or Mg?
    d : function
        The function to calculate the diffusivity.
    """
    usteps = np.shape(u_prev)[0]

    # LHS
    A = np.zeros([usteps, usteps])

    fo2_prev = fO2(T_prev, P)
    D_prev = np.zeros(np.shape(u_prev))
    for j in range(usteps):
        if Fe:
            D_prev[j] = d(T_prev, P, u_prev[j], fo2_prev)
        else:
            D_prev[j] = d(T_prev, P, 1.0 - u_prev[j], fo2_prev)
    D_prev = D_prev * alpha

    # For convenience, r_j^2
    rj2 = ((j-1) * dr)**2
    
    # For convenience r_{j+1}^2
    rjp2 = (j * dr)**2

    # For convenience r_{j-1}^2
    rjm2 = ((j-2) * dr)**2

    for j in range(usteps):
        if j == 0:
            A[j,j] = - (1 + kappa / rj2 * (0.5 * (D_prev[j] * rj2 + D_prev[j+1] * rjp2) + 0.5 * (D_prev[j] * rj2 + D_prev[j+1] * rjp2))) 
            A[j, j+1] = kappa / rj2 * 0.5 * (D_prev[j] * rj2 + D_prev[j+1] * rjp2)*2

        elif j == usteps - 1:
            A[j,j] = 1 # The final composition is fixed, and therefore known already
            
        else:
            A[j, j-1] = kappa / rj2 * 0.5 * (D_prev[j] * rj2 + D_prev[j-1] * rjm2)
            A[j,j] = - (1 + kappa / rj2 * (0.5 * (D_prev[j] * rj2 + D_prev[j+1] * rjp2) + 0.5 * (D_prev[j] * rj2 + D_prev[j-1] * rjm2)))
            A[j, j+1] = kappa / rj2 * 0.5 * (D_prev[j] * rj2 + D_prev[j+1] * rjp2)



    # RHS
    B = np.zeros(np.shape(u_prev))

    for j in range(usteps):
        # for j=0, j+1 == j-1 by symmetry
        if j == 0:
            B[j] = (
                - u_prev[j+1] * kappa / rj2 * 0.5 * (D_prev[j+1] * rjp2 + D_prev[j] * rj2)
                - u_prev[j] * (1 - kappa / rj2 * (0.5 * (D_prev[j+1] * rjp2 + D_prev[j] * rj2) + 0.5 * (D_prev[j] * rj2 + D_prev[j+1] * rjp2)))
                - u_prev[j+1] * kappa / rj2 * 0.5 * (D_prev[j+1] * rjp2 + D_prev[j] * rj2)
            )

        elif j == usteps - 1:
            B[j] = u_prev[j] 
        else:
            B[j] = (
                - u_prev[j-1] * kappa / rj2 * 0.5 * (D_prev[j-1] * rjm2 + D_prev[j] * rj2)
                - u_prev[j] * (1 - kappa / rj2 * (0.5 * (D_prev[j+1] * rjp2 + D_prev[j] * rj2) + 0.5 * (D_prev[j] * rj2 + D_prev[j-1] * rjm2))) 
                - u_prev[j+1] * kappa / rj2 * 0.5 * (D_prev[j+1] * rjp2 + D_prev[j] * rj2)
            )
    
    return A, B




# %%
def run_diffusion_model_3d(alpha, Fe=True, xsteps=100, tsteps=10000, d=d, T=T):
    """

    Note that many of the parameters are being set by global variables.
    While this is not great python practice, it stays true to the format
    of the original Mathematica script this code is emulating.

    Parameters
    ----------
    alpha : float, default: 1.0
        Adjustment factor for D calculated from beta and isotope mass ratios
    Fe : bool, default: True
        Fe or Mg?
    """

    if Fe:
        X0 = XFe0
        X1 = XFe1
    else:
        X0 = XMg0
        X1 = XMg1

    # Set up starting arrays:
    u0 = np.array([X0]*(xsteps-1) + [X1])
    time = np.linspace(0, 3*tau, tsteps+1)

    x_results = np.zeros([tsteps+1, xsteps])
    x_results[0,:] = u0

    Tt = np.zeros(tsteps+1)
    for i in range(tsteps):
        Tt[i] = T(time[i])
        
    u_prev = u0

    kappa = (time[1] - time[0]) / (2 * (a / (xsteps-1))**2)
    print(np.log10(kappa))

    for i in range(tsteps):
        i+=1 # Run from i=1 to tsteps+1
        A, B = construct_linalg_problem_3d(u_prev, Tt[i-1], kappa, dr=a/(xsteps -1), alpha=alpha, Fe=Fe, d=d)
        u = np.linalg.solve(A, B)
        x_results[i,:] = u
        u_prev = u
    
    return x_results


# %%
def constD(*args, **kwargs):
    return 2.4e-14


# %%
def constT(t):
    return 1853.0


# %%
xFe54_sphere = run_diffusion_model_3d((54.0/56.0)**beta_Fe, d=constD, T=constT)

# %%
fig, ax = plt.subplots()

for i in range(np.shape(xFe54_sphere)[0]):
    ax.plot(range(np.shape(xFe54_sphere)[1]), xFe54_sphere[i,])

plt.show()

# %%
fig, ax = plt.subplots()

timestep = 30

# for i in range(np.shape(xFe54_sphere)[0]):
ax.plot(np.array(range(np.shape(xFe54)[1]))/ np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54[timestep,:], c='k')
ax.plot(- np.array(range(np.shape(xFe54)[1])) / np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54[timestep,:], c='k')

ax.plot(np.array(range(np.shape(xFe54_sphere)[1])) / np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54_sphere[timestep,:], ls='--', c='C1')
ax.plot(- np.array(range(np.shape(xFe54_sphere)[1])) / np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54_sphere[timestep,:], ls='--', c='C1')

ax.set_title("time = {:.1f} days".format(3*tau/(np.shape(xFe54)[0]-1)*timestep/3600/24))

ax.axvline(0, c='k', lw=1)

plt.show()

# %%
fig, ax = plt.subplots()

timestep = 145

# for i in range(np.shape(xFe54_sphere)[0]):
ax.plot(np.array(range(np.shape(xFe54)[1]))/ np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54[timestep,:], c='k')
ax.plot(- np.array(range(np.shape(xFe54)[1])) / np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54[timestep,:], c='k')

ax.plot(np.array(range(np.shape(xFe54_sphere)[1])) / np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54_sphere[timestep,:], ls='--', c='C1')
ax.plot(- np.array(range(np.shape(xFe54_sphere)[1])) / np.shape(xFe54)[1] * a * 1e6, 
        1-xFe54_sphere[timestep,:], ls='--', c='C1')

ax.set_title("time = {:.1f} days".format(3*tau/(np.shape(xFe54)[0]-1)*timestep/3600/24))

ax.axvline(0, c='k', lw=1)

plt.show()

# %%

# %%

# %%

# %%

# %%

# %%
tau

# %%
xres = run_diffusion_model(1.0)

# %%
xres

# %%
fig, ax = plt.subplots()

for i in range(np.shape(xres)[0]):
    ax.plot(range(np.shape(xres)[1]-75), xres[i,75:])

plt.show()

# %%

T_prev = 1473.0
x_boundary = 0.8
x_internal = 0.9
kappa = 1e12

u_prev = np.array([x_internal]*100)
u_prev[-1] = x_boundary

A, B = construct_linalg_problem(u_prev, T_prev, kappa)

# %%
B

# %%
A

# %%
u = np.linalg.solve(A, B)

# %%
u

# %%
A

# %% [markdown]
# Maybe a time dependence for $D$ can be incorporated:
#
# $ \frac{u_j^{n+1} - u_j^n}{\Delta t} = \frac{1}{2} \left[ \frac{D_{j+1/2}^{n+1} \left( u^{n+1}_{j+1} - u_j^{n+1} \right) + D_{j+1/2}^{n} \left( u^{n}_{j+1} - u_j^{n} \right) - D_{j-1/2}^{n+1} \left( u_j^{n+1} - u^{n+1}_{j-1} \right) - D_{j-1/2}^{n} \left( u_j^{n} - u^{n}_{j-1} \right)}{\left(\Delta x \right)^2} \right]$ 
#
# where,
#
# $D_{j+1/2}^n = \frac{1}{2} \left[ D(u^{n}_{j+1}) + D(u^n_j) \right] ; \, \, \, D_{j-1/2}^n = \frac{1}{2} \left[ D(u^{n}_{j}) + D(u^n_{j-1}) \right]$
#
#
# $D_{j+1/2}^{n+1} = \frac{1}{2} \left[ D(u^{n+1}_{j+1}) + D(u^{n+1}_j) \right] ; \, \, \, D_{j-1/2}^{n+1} = \frac{1}{2} \left[ D(u^{n+1}_{j}) + D(u^{n+1}_{j-1}) \right] $
#
# $D(u^{n+1}_{j}) = D(u^n_j) + (u_j^{n+1} - u_j^n) \, \, \frac{\partial D}{\partial u} \Bigr|_{j,n} $
#
# $D(u^{n+1}_{j+1}) = D(u^n_{j+1}) + (u_{j+1}^{n+1} - u_{j+1}^n) \, \, \frac{\partial D}{\partial u} \Bigr|_{j+1,n}$
#
# $ D(u^{n+1}_{j-1}) = D(u^n_{j-1}) + (u_{j-1}^{n+1} - u_j^n) \, \, \frac{\partial D}{\partial u} \Bigr|_{j-1,n} $

# %% [markdown]
#
