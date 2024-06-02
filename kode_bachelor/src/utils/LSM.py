import numpy as np
import pandas as pd
from src.utils.polynomials import Polynomials
from scipy import stats

class LSM:
    "This is the LSM algorithm"
    def __init__(self):
        "Initialization of the LSM class"
    def LSM_algorithm(self,paths, strike, r, option_type, M,T,n,Method, Reg_Method):
        """
        LSM Algorithm for option pricing.
        
        Parameters:
        paths : ndarray
            Simulated paths
        strike : float
            Strike price
        r : float 
            Short-term interest rate
        option_type : str 
            'Put' or 'Call'
        M : int 
            Degree of Laguerre polynomials
        T : float 
            Total time period (in years)
        n : int
            Number of steps
        Method : str
            'LSM'
        Reg_Method : str 
            'Laguerre' or 'Polynomium'
        
        Returns:
        float: Option price
        ndarray: Regression coefficients
        """
        # Defining all possible cashflows
        cash_flows = np.zeros_like(paths)
        if option_type == "Call":
            cash_flows = np.maximum(paths - strike, 0)
        else:
            cash_flows = np.maximum(strike - paths, 0)
        # Defining discounted cashflows
        discounted_cash_flows = cash_flows[-1]
        # discount factor
        dt = T / n

        if Method == 'LSM':
            # Making the LSM algorithm
            for t in range(n-1, 0, -1): 
                # Finding all ITM paths
                if option_type == 'Call':
                    in_the_money = paths[t] > strike
                else:  # for 'put'
                    in_the_money = paths[t] < strike
                # Getting prices for ITM paths
                
                if np.any(in_the_money): 
                    # Generating Laguerre polynomials of degree M
                    if Reg_Method == 'Laguerre':
                        X = paths[t, in_the_money] / strike
                        Y = discounted_cash_flows[in_the_money] * np.exp(-r * dt) / strike
                        normalizing = strike
                        Xs = Polynomials().laguerre_basis(X, M)

                    if Reg_Method == 'Polynomium':
                        X = paths[t, in_the_money] 
                        Y = discounted_cash_flows[in_the_money] * np.exp(-r * dt) 
                        normalizing = 1
                        Xs = Polynomials().power_basis(X, M)

                    # Regression coefficients is calculated using OLS
                    XtX_inv = np.linalg.pinv(Xs.T @ Xs) 
                    XtY = Xs.T @ Y
                    reg_coef = XtX_inv @ XtY
                    # Predict conditional expected payoffs
                    conditional_exp = Xs @ reg_coef * normalizing
                    
                    # Checking wheter to early exercise or not
                    exercised_early = conditional_exp < cash_flows[t, in_the_money]
                    # Calculate discounted cashflows
                    discounted_cash_flows[in_the_money] = np.where(exercised_early, cash_flows[t, in_the_money], discounted_cash_flows[in_the_money] * np.exp(-r * dt))
                    discounted_cash_flows[~in_the_money] = discounted_cash_flows[~in_the_money] * np.exp(-r * dt)
        
        option_price = np.mean(discounted_cash_flows) * np.exp(-r * dt)
        
        se = stats.sem(discounted_cash_flows)
        return option_price, se
    
    
    
    
    
    