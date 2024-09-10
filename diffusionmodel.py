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

# %% [markdown]
# ## Fe-Mg isotope geospeedometry in olivine
#
# ### Constants

# %%
R = 8.314472

# %%

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
a = 300e-6 # m, grain radius
XFe1 = 0.11 # Surface Fe concentration, identical for 54 and 56
XMg1 = 0.89 # Surface Mg concentration, identical for 24 and 26
XFe0 = 0.06 # Initial Fe concentration, identical for 54 and 56
XMg0 = 0.94 # Initial Mg concentration, identical for 24 and 26

# %% [markdown]
# ### Definition of functions

# %%
# def T(t):
#     if cooling == 'linear':
#         if T0 - t * c / (3600 * 24) > 298.15:
#             return T0 - t * c / (3600 * 24) > 298.15
#         else:
#             return 298.15
#     else:
#         return 298.15 + (T0 - 298.15) * np.exp(- t * c / 3600 * 24)

# %%
# t, T0, c, q = sym.symbols('t T0 c q')
t = sym.symbols('t')


Tlin = sym.Piecewise(
       (T0 - t * c / (3600 * 24), T0 - t * c / (3600 * 24) > 298.15),
       (298, T0 - t * c / (3600 * 24) <= 298.15)
)

Texp = 298.15 + (T0 - 298.15) * sym.exp(- t * c / 3600 * 24)

T = sym.Piecewise(
    (Tlin, q>=1),
    (Texp, q<1)
)

T

# %%
# def fO2(T, P):
#     """
#     oxygen fugacity in Pa with T in K and P in Pa; Huebner & Sato 1970; Chou 1987; Herd 2008
#     """
#     return 101325 * 10**delta_O2 * 10**(9.36 - 24930 / T + 0.046 * (P / 1e5 - 1) / T)

# %%
fO2 = 101325 * 10**delta_O2 * 10**(9.36 - 24930 / T + 0.046 * (P / 1e5 - 1) / T)

fO2

# %%
# def d(T, P, X, f):
#     """
#     Fe-Mg diffusion coefficient in olivine in m2s-1, T in K, P pressure in Pa, 
#     f absolute oxygen fugacity in Pa, X mole fraction of the fayalite component
#     Dohmen & Chakraborty 2007
#     """
#     if f > 10**-10:
#         return (
#             10**(-9.21 - (201000 + (P-1e5)*7e-6)/(2.303*R*T) 
#                  + 1/6 * np.log10(f/1e-7) + 3*(X-0.1))
#         )
#     else:
#         return (
#             10**(-8.91 - (220000 + (P-1e5)*7e-6) / (2.303*R*T) + 3 * (X-0.1))
#         )

# %%
r = sym.symbols('r')
X = sym.symbols('X')


# X = sym.Function('X')(t, r)

d = sym.Piecewise(
    (10**(-9.21 - (201000 + (P-1e5)*7e-6) / (2.303*R*T) + 1/6 * sym.log(fO2/1e-7, 10) + 3*(X-0.1)), fO2 > 10**-10),
    (10**(-8.91 - (220000 + (P-1e5)*7e-6) / (2.303*R*T) + 3 * (X-0.1)), fO2 <= 10**-10)
)

d

# %% [markdown]
# ### Calculation of diffusion and cooling timescales

# %%
# # Diffusion timescale in seconds:
# tau = a**2 / d(T0, P, XFe0, fO2(T0,P))

# if cooling == 'linear':
#     initial_cooling_rate = c
# else:
#     initial_cooling_rate = (T0 - 298.15) * c
# print("initial cooling rate for " + cooling + " model: {:.4f} K.d-1".format(initial_cooling_rate))

# if cooling == 'linear':
#     denominator = c
# else:
#     denominator = (T0 - 298.15) * c
# print("cooling timescale: {:.4f} d".format((T0 - 1173.15) / denominator))

# print("diffusion timescale: {:.4f} d".format(tau / (3600 * 24)))

# %%
tau = a**2 / d
tau

# %%
float(tau.subs({'T':T0, 'X':XFe0, 't':0}))

# %%
if q == 1:
    initial_cooling_rate = c
else:
    initial_cooling_rate = (T0 - 298.15) * c
print("initial cooling rate for " + cooling + " model: {:.4f} K.d-1".format(initial_cooling_rate))

if q == 1:
    denominator = c
else:
    denominator = (T0 - 298.15) * c
print("cooling timescale: {:.4f} d".format((T0 - 1173.15) / denominator))

print("diffusion timescale: {:.4f} d".format(float(tau.subs({'T':T0, 'X':XFe0, 't':0})) / (3600 * 24)))

# %%
X = sym.Function('X')(t,r)

d = d.subs({'X':sym.Function('X')(t,r)})
d

# %% [markdown]
# `f4`, `f6`, `m4`, and `m6` is the solution to the diffusion pde for $^{54}Fe$, $^{56}Fe$, and $^{26}Mg$ respectively.
#
# `alpha` is a coefficient that accounts for differences in diffusivities of isotopes
#
# Note that the /. operator in mathematic means replace with, i.e., (x+10)/.x->1 is the same as (1+10)
#
# The D[...] operator is a partial derivative.

# %%
alpha = sym.symbols('alpha')
dXdt = 1 / (r + 0.000001e-6) * sym.diff(r**2 * alpha * d * sym.diff(X, r), r)

dXdt

# %%
eq = sym.Eq(sym.diff(X,t), dXdt)

# %%
# sym.pdsolve?

# %%
# sym.pdsolve?

# %%
sym.pdsolve(eq, X(t,r), ics={X(0,r): XFe0, X(t,a): (1/(1+t)*(XFe0-XFe1)+XFe1), X(t,r).diff(r).subs({r:0}):0})

# %%
X(0,0)

# %%
