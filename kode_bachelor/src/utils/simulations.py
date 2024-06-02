# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

class GeometricBrownianMotion:
    def __init__(self):
        """
        

        Returns
        -------
        None.

        """
        
    def gbm(self, S0,r,sigma, T, n, num_paths): 
        """
       Generate simulated paths for the stock price using the Geometric Brownian Motion model.
       
       Parameters:
        S0 : float
            Initial stock price.
        r : float
            Risk-free interest rate (annual).
        sigma : float
            Volatility of the stock.
        T : float
            Total time span of the simulation (in years).
        n : int
            Number of intervals the time span T is divided into.
        num_paths : int
            Number of simulated paths.

       
       Returns:
       np.ndarray
            A numpy array of shape (n+1, num_paths) containing simulated stock price paths. Each column represents
            a separate simulation/path and each row corresponds to a time increment.
       """
        
        dt=T/n
        paths = np.zeros((n+1,num_paths))
        paths[0] = S0
        for t in range(1,n+1):
            rand = np.random.standard_normal(num_paths)
            paths[t] = paths[t - 1]*np.exp((r - 0.5 * sigma ** 2)*dt + sigma *np.sqrt(dt)*rand)
        return paths
    
    def heston(self, s0, v0, kappa, theta, sigma, rho, r, T, n, N):
        
        dt = T / n
        S = np.zeros((n + 1, N))
        V = np.zeros((n + 1, N))
        S[0, :] = s0
        V[0, :] = v0
    
        # Generate stock price and volatility paths
        for t in range(1, n + 1):
    
            z1 = np.random.standard_normal(N)
            z2 = rho * z1 + np.sqrt(1 - rho ** 2) * np.random.standard_normal(N)
            # Compute Gamma(t)^2
            exp_kappa_dt = np.exp(-kappa * dt)
            V_t = V[t - 1, :]
            term1 = 0.5 * sigma**2 * kappa**(-1) * V_t * (1 - np.exp(-2 * kappa * dt))
            term2 = (exp_kappa_dt * V_t + (1 - exp_kappa_dt) * theta)**2
            gamma_squared = dt**(-1) * np.log(1 + term1 / term2)
    
            V[t, :] = np.maximum(
                0, (exp_kappa_dt * V_t + (1 - exp_kappa_dt) * theta) * np.exp(-0.5 * gamma_squared * dt + np.sqrt(gamma_squared) * np.sqrt(dt) * z1)
            )
            lnS = np.log(S[t - 1, :]) + (r - 0.5 * V[t - 1, :]) * dt + np.sqrt(V[t - 1, :]) *  np.sqrt(dt) * z2
            S[t, :] = np.exp(lnS)
            
    
        return S,V
    
    
    
    
    