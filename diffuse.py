"""
Functions for solving PDE describing Mg and Fe diffusion in a sphere,
and resulting isotope fractionation.

Simon Matthews (simonm@hi.is) 
University of Iceland
"""

import numpy as np




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