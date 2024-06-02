# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from src.utils.polynomials import Polynomials 
from src.utils.simulations import GeometricBrownianMotion as GBM




class Dual_LSM:
    
    def __init__(self):
        """        
        Initializes an instance of the Dual_LSM class
        """
 
    def get_regression(self,paths, strike, r, option_type, M,T,n,Method):
        """
        Get lower price for American option and regression coefficients.
        
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
            Degree of polynomials
        T : float 
            Total time period (in years)
        n : int
            Number of steps
        Method : str
            'LSM' or 'Delta_LSM'
        
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
        # Define regression coefficients matrix
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
                    XtX_inv = np.linalg.inv(Xs.T @ Xs)
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
                    X_inv = np.linalg.inv(XtX + lXssT_Xss)
                    
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
    
        option_price = np.mean(discounted_cash_flows) * np.exp(-r * dt)
        return regression_coefficients, option_price
    
    
    
    def nested_simulation(self, S, r, sigma, t, T, n, strike, num_paths, M, reg_coef, option_type):
        """
        Simulates and evaluates option pricing using nested simulation 
        using the regression coefficients from the lower price.
    
        Returns:
        float: The estimated price of the option based on the nested simulation.
   
        """
        
        # Update number of steps and total time period
        nn = n-t
        TT  = T*nn/n
        # Simulate new nested paths
        nested_paths = GBM().gbm(S, r, sigma, TT, nn, num_paths)
       
        # Defining all possible cashflows
        cash_flows = np.zeros_like(nested_paths)
        if option_type == "Call":
            cash_flows = np.maximum(nested_paths - strike, 0)
        else:
            cash_flows = np.maximum(strike - nested_paths, 0)
        # Defining discounted cashflows
        discounted_cash_flows = np.zeros_like(nested_paths)
        
        # discount factor
        dt = T / n 
        reg_nested = reg_coef[t:,]
        
        # Make forward pass for each path
        for t in range(1,nn): 
            reg_coeff = reg_nested[t-1,:]
            
            in_the_money = nested_paths[t] < strike
            
            not_exercised  = np.all(discounted_cash_flows[:t,:] == 0, axis=0)
            if np.any(in_the_money & not_exercised):
                X = nested_paths[t, in_the_money & not_exercised] 
                Xs = Polynomials().power_basis(X, M)
                conditional_exp = Xs @ reg_coeff 
                
                exercise_now = cash_flows[t, in_the_money & not_exercised] > conditional_exp
                # Update cashflows and exercise policy
                discounted_cash_flows[t, in_the_money & not_exercised] = np.where(exercise_now, cash_flows[t, in_the_money & not_exercised]*np.exp(-r * dt * t), 0)
                #exercise_policy[t, in_the_money] = exercise_now
        
        # Seeing whether exercise is need in period T
        not_exercised_last  = np.all(discounted_cash_flows[:nn,:] == 0, axis=0)
        discounted_cash_flows[nn,not_exercised_last] = np.where(discounted_cash_flows[nn,not_exercised_last] == 0, cash_flows[nn, not_exercised_last]*np.exp(-r * dt * nn), 0)
        
        # Calculate price and print
        option_value = np.sum(discounted_cash_flows) / nested_paths.shape[1]
        return option_value
    
    
    def Dual_LSM(self, paths_dual, strike, r, sigma, option_type, M, T, n, reg_coef, num_paths):
        """
        Calculate the upper bound for the American option price.
        
        Returns:
        float: The estimated upper bound.
   
        """
        
        # discount factor
        dt = T / n
        
        # Define regression coefficient used for the Primal LSM
        reg_dual = reg_coef
        
        # Defining all possible cashflows
        cash_flows = np.zeros_like(paths_dual)
        if option_type == "Call":
            cash_flows = np.maximum(paths_dual - strike, 0)
        else:
            cash_flows = np.maximum(strike - paths_dual, 0)
        # Defining discounted cashflows
        discount_factors = np.exp(-r * dt * np.arange(n + 1))
        discount_cf = cash_flows * discount_factors.reshape(-1, 1)
        # Define the martingale used to find the upper bound
        Martingale = np.zeros_like(paths_dual)
        
        # Define matrix to store expected values in
        expect_value = np.zeros_like(cash_flows)
        # Get expected value for time 1, which is calculated in time 0
        for i in range(paths_dual.shape[1]):  # Loop over each path
            expect_value[0, i] = self.nested_simulation(paths_dual[0, i], r, sigma, 0, T, n, strike, num_paths, M, reg_dual, option_type) 
        # Make forward pass for each path
        for t in range(1, n):
            in_the_money = paths_dual[t] < strike
            in_the_money_indices = np.where(in_the_money)[0]
            out_of_the_money_indices = np.where(~in_the_money)[0]
            X = paths_dual[t, in_the_money]
            Xs = Polynomials().power_basis(X, M)
            conditional_exp = Xs @ reg_dual[t-1]
            
            # Handling in-the-money paths
            for idx, real_idx in enumerate(in_the_money_indices):
                price = self.nested_simulation(X[idx], r, sigma, t, T, n, strike, num_paths, M, reg_dual, option_type) * np.exp(-r * dt * t)
                if cash_flows[t, real_idx] > conditional_exp[idx]:
                    Martingale[t, real_idx] = Martingale[t-1, real_idx] + discount_cf[t, real_idx] - expect_value[t-1, real_idx]
                else:
                    Martingale[t, real_idx] = Martingale[t-1, real_idx] + price - expect_value[t-1, real_idx]
                expect_value[t, real_idx] = price
                
            # Handling out-of-the-money paths
            x_cont = paths_dual[t, ~in_the_money]
            for idx, real_idx in enumerate(out_of_the_money_indices):
                price = self.nested_simulation(x_cont[idx], r, sigma, t, T, n, strike, num_paths, M, reg_dual, option_type) * np.exp(-r * dt * t)
                Martingale[t, real_idx] = Martingale[t-1, real_idx] + price - expect_value[t-1, real_idx]
                expect_value[t, real_idx] = price
        for i in range(paths_dual.shape[1]):
            Martingale[n,i] = Martingale[n-1,i] + cash_flows[n,i]*np.exp(-r * dt * n) - expect_value[n-1,i]
        
        # Getting the maximum for each path of discounted cashflow minus the respective martingale
        final_values = np.max(discount_cf - Martingale, axis=0)
        
        option_value_upper_bound = np.mean(final_values)
        return option_value_upper_bound








