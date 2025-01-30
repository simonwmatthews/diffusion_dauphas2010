"""
Functions for solving PDE describing Mg and Fe diffusion in a sphere,
and resulting isotope fractionation.

Simon Matthews (simonm@hi.is) 
University of Iceland
"""

import numpy as np

R = 8.314472

# ==== HELPER CLASSES ====================================================================

class DvariableDauphas2010:
    """
    Implementation of the variable olivine Fe-Mg diffusion coefficient
    used by Dauphas et al. (2010), after Dohmen & Chakraborty (2007).
    """
    def __init__(self):
        pass
    def __call__(self, T, P, X, fO2):
        """
        Calculate the diffusion coefficient at specified conditions

        Parameters
        ----------
        T : float
            Temperature in K
        P : float
            Pressure in Pa
        X : float
            The mole fraction of the fayalite component
        fO2 : float
            The absolute oxygen fugacity in Pa
        
        Returns
        -------
        float
            The diffusivity in m2s-1
        """
        if fO2 > 10**-10:
            return (
                10**(-9.21 - (201000 + (P-1e5)*7e-6)/(2.303*R*T) 
                    + 1/6 * np.log10(fO2/1e-7) + 3*(X-0.1))
            )
        else:
            return (
                10**(-8.91 - (220000 + (P-1e5)*7e-6) / (2.303*R*T) + 3 * (X-0.1))
            )

class Dconstant:
    """
    Implements a constant diffusion coefficient.

    Parameters
    ----------
    D : float
        The partition coefficient in m2s-1
    """
    def __init__(self, D):
        self.D = D
    
    def __call__(self, *args, **kwargs):
        return self.D
    
class TevolutionLinearDauphas2010:
    """
    Implementation of the linearly time-dependent temperature used by 
    Dauphas et al. (2010).

    Parameters
    ----------
    T0 : float
        The initial temperature in K
    c : float
        The cooling parameter in Kd-1.
    """
    def __init__(self, T0, c):
        self.T0 = T0
        self.c = c
    
    def __call__(self, t):
        """
        Return the temperature at a given time

        Parameters
        ----------
        t : float
            Time in s
        
        Returns
        -------
        float
            Temperature in K
        """
        T = self.T0 - t * self.c / (3600 * 24)
        if T > 298.15:
            return T
        else:
            return 298.15
        
class TevolutionExponentialDauphas2010:
    """
    Implementation of the exponentially time-dependent temperature used by 
    Dauphas et al. (2010).

    Parameters
    ----------
    T0 : float
        The initial temperature in K
    c : float
        The cooling parameter in d-1.
    """
    def __init__(self, T0, c):
        self.T0 = T0
        self.c = c
    
    def __call__(self, t):
        """
        Return the temperature at a given time

        Parameters
        ----------
        t : float
            Time in s
        
        Returns
        -------
        float
            Temperature in K
        """
        return 298.15 + (self.T0 - 298.15) * np.exp(- t * self.c / 3600 * 24)

class Tconstant:
    """
    Implements a constant temperature for the calculation.

    Parameters
    ----------
    T : float
        Temperature in K
    """
    def __init__(self, T):
        self.T = T
    
    def __call__(self, t):
        """
        Returns a constant temperature at all times

        Parameter
        ---------
        t : float
            Placeholder for interface consistency.
        
        Returns
        -------
        float
            The constant temperature in K.
        """
        return self.T

class fO2Dauphas2010:
    """
    Implements the variable fO2 used by Dauphas et al. (2010), after
    Huebner & Sato (1970), Chou (1987), and Herd (2008).

    Parameters
    ----------
    DeltafO2 : float
        The Delta log(fO2) value relative to the NNO buffer
    """
    def __init__(self, DeltafO2):
        self.DeltafO2 = DeltafO2
        pass

    def __call__(self, T, P):
        """
        Calculates the fO2 from T, P

        Parameters
        ----------
        T : float
            Temperature in K
        P : float
            Pressure in Pa
        
        Returns
        -------
        float
            The absolute fO2 value in Pa

        """
        return 101325 * 10**self.DeltafO2 * 10**(9.36 - 24930 / T + 0.046 * (P / 1e5 - 1) / T)



# ==== SET UP LINEAR ALGEBRA PROBLEM ========================================

def construct_linalg_problem(u_prev, T_prev, kappa, dr, P, D, fO2, alpha=1.0, Fe=True):
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
    P : float
        The pressure in Pa
    D : callable
        The callable to calculate the diffusivity.
    fO2 : callable
        The callable to the fO2
    alpha : float, default: 1.0
        Adjustment factor for D calculated from beta and isotope mass ratios
    Fe : bool, default: True
        Fe or Mg?
    
    Returns
    -------
    np.array
        The LHS of the linear algebra problem
    
    np.array
        The RHS of the linear algebra problem

    """
    usteps = np.shape(u_prev)[0]

    # LHS
    A = np.zeros([usteps, usteps])

    fo2_prev = fO2(T_prev, P)
    D_prev = np.zeros(np.shape(u_prev))
    for j in range(usteps):
        if Fe:
            D_prev[j] = D(T_prev, P, u_prev[j], fo2_prev)
        else:
            D_prev[j] = D(T_prev, P, 1.0 - u_prev[j], fo2_prev)
    D_prev = D_prev * alpha

    

    for j in range(usteps):
        # For convenience, r_j^2
        rj2 = ((j) * dr)**2 + 1e-12
        
        # For convenience r_{j+1}^2
        rjp2 = ((j+1) * dr)**2 + 1e-12

        # For convenience r_{j-1}^2
        rjm2 = ((j-1) * dr)**2 + 1e-12
        
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
        # For convenience, r_j^2
        rj2 = ((j) * dr)**2 + 1e-12
        
        # For convenience r_{j+1}^2
        rjp2 = ((j+1) * dr)**2 + 1e-12

        # For convenience r_{j-1}^2
        rjm2 = ((j-1) * dr)**2 + 1e-12

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

# === Run the diffusion model ===========================================================

def run_diffusion_model(X0, X1, a, alpha, D, T, P, fO2, Fe=True, n_tau_end=3, xsteps=100, tsteps=10000):
    """

    Note that many of the parameters are being set by global variables.
    While this is not great python practice, it stays true to the format
    of the original Mathematica script this code is emulating.

    Parameters
    ----------
    X0 : float
        The initial concentration of Fe/Mg in the crystal interior in mol fraction
    X1 : float
        The concentraion of Fe/Mg at the crystal edge in mol fraction
    a : float
        Grain radius in m
    alpha : float, default: 1.0
        Adjustment factor for D calculated from beta and isotope mass ratios
    D : callable
        The callable to calculate the diffusivity.
    T : callable
        The callable to calculate T as a function of time
    P : float
        The pressure in Pa
    fO2 : callable
        The callable to the fO2
    Fe : bool, default: True
        Fe or Mg?
    n_tau_end : float
        The length of the simulation (in multiples of the diffusion timescale)
    xsteps : int, default: 100
        The number of steps in space along the radius
    tsteps : int, default: 10000
        The number of timesteps
    """

    # Calculate tau
    if Fe:
        tau = a**2 / D(T(0), P, X0, fO2(T(0),P))
    else:
        tau = a**2 / D(T(0), P, 1.0-X0, fO2(T(0),P))
    print(f'Diffusion Timescale: {tau/3600/24:.4f} days')

    # Set up starting arrays:
    u0 = np.array([X0]*(xsteps-1) + [X1])
    time = np.linspace(0, n_tau_end*tau, tsteps+1)

    x_results = np.zeros([tsteps+1, xsteps])
    x_results[0,:] = u0

    Tt = np.zeros(tsteps+1)
    for i in range(tsteps):
        Tt[i] = T(time[i])
        
    u_prev = u0

    kappa = (time[1] - time[0]) / (2 * (a / (xsteps-1))**2)

    for i in range(tsteps):
        i+=1 # Run from i=1 to tsteps+1
        A, B = construct_linalg_problem(u_prev, Tt[i-1], kappa, a/(xsteps -1), P, D, fO2, alpha=alpha, Fe=Fe)
        u = np.linalg.solve(A, B)
        x_results[i,:] = u
        u_prev = u
    
    return x_results

# === Provide a complete wrapper around the model ===================================


