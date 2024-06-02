import numpy as np
import pandas as pd
from src.utils.polynomials import Polynomials 
from scipy.stats import sem


def LSM_Delta_LSM(paths, strike, r, option_type, M,T,n,Method):
    """
    paths: Simulated paths
    Strike: Strike price
    r: Short-term interest rate
    option_type: Put or Call
    M: degree of Laguerre polynomials
    T: Total time period (in years)
    n: number of steps
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

    regression_coefficients = np.zeros((n-1, M+1))

    if Method == 'LSM':
        # Making the LSM algorithm
        for t in range(n-1, 0, -1): 
            # Finding all ITM paths
            if option_type == 'Call':
                in_the_money = paths[t] > strike
            else:  # for 'put'
                in_the_money = paths[t] < strike
            # Getting prices for ITM paths
            X = paths[t, in_the_money]
            if np.any(in_the_money): 
                # Get cashflows for ITM paths
                Y = discounted_cash_flows[in_the_money] * np.exp(-r * dt) 
                
                # Generating Laguerre polynomials of degree M
                Xs = Polynomials().power_basis(X, M)
                # Create regression
                # Regression coefficients is calculated using OLS
                XtX_inv = np.linalg.pinv(Xs.T @ Xs)
                XtY = Xs.T @ Y
                reg_coef = XtX_inv @ XtY
                # Predict conditional expected payoffs
                conditional_exp = Xs @ reg_coef 

                regression_coefficients[t-1, :] = reg_coef
                
                # Checking wheter to early exercise or not
                exercised_early = conditional_exp < cash_flows[t, in_the_money]
                # Calculate discounted cashflows
                discounted_cash_flows[in_the_money] = np.where(exercised_early, cash_flows[t, in_the_money], discounted_cash_flows[in_the_money] * np.exp(-r * dt))
                discounted_cash_flows[~in_the_money] = discounted_cash_flows[~in_the_money] * np.exp(-r * dt)
    
    if Method == 'Delta_LSM':   
        # Making the LSM algorithm
        optimal_price = paths[-1] 
        
        for t in range(n-1, 0, -1): 
            # Finding all ITM paths
            if option_type == 'Call':
                in_the_money = paths[t] > strike
            else:  # for 'put'
                in_the_money = paths[t] < strike
            
            # Getting prices for ITM paths
            X = paths[t, in_the_money] 
            if np.any(in_the_money):
                # Get cashflows for ITM paths
                Y = discounted_cash_flows[in_the_money] * np.exp(-r * dt) 
                # Generating Laguerre polynomials of degree M
                Xs = Polynomials().power_basis(X, M)
                Xss = Polynomials().weighted_power_basis(X, M)
                
                # Create regression
                # Regressionskoefficienter beregnet ved OLS
            
                h  = np.where(optimal_price[in_the_money] < strike, -1, 0)
                Z = np.array(optimal_price[in_the_money]) / np.array(X) * np.array(h)
                
                l = np.linalg.norm(-Y)**2 / np.linalg.norm(Z)**2
                
                XtX = Xs.T @ Xs
                lXssT_Xss = l * (Xss.T @ Xss)
                X_inv = np.linalg.pinv(XtX + lXssT_Xss)
                
                reg_coef = X_inv @ (Xs.T @ Y + l * (Xss.T @ Z))
                # Predict conditional expected payoffs
                conditional_exp = Xs @ reg_coef 

                regression_coefficients[t-1, :] = reg_coef

                # Checking wheter to early exercise or not
                exercised_early = conditional_exp < cash_flows[t, in_the_money]
                # Calculate discounted cashflows
                discounted_cash_flows[in_the_money] = np.where(exercised_early, cash_flows[t, in_the_money], discounted_cash_flows[in_the_money] * np.exp(-r * dt))
                discounted_cash_flows[~in_the_money] = discounted_cash_flows[~in_the_money] * np.exp(-r * dt)
                optimal_price[in_the_money] = np.where(exercised_early, paths[t, in_the_money], optimal_price[in_the_money])

    see = sem(discounted_cash_flows)
    option_price = np.mean(discounted_cash_flows) * np.exp(-r * dt)
    return option_price, regression_coefficients, see

