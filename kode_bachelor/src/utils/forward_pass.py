import numpy as np
from src.utils.polynomials import Polynomials
from scipy.stats import sem



class forward_pass:
    def __init__(self):
        """
        Initializes the parameters for the finite difference method and precomputes step sizes for time and asset price space.
        
        Parameters:
            S0 : float
                Initial stock price.
            K :float
                Strike price of the option.
            r : float
                Risk-free interest rate (annual).
            sigma : float 
                Volatility of the stock.
            T : float 
                Total time to maturity of the option (in years).
            Smax : float 
                Maximum stock price considered in the model.
            Smin : float
                Minimum stock price considered in the model.
            m : int
                Number of spatial divisions in the stock price range.
            n : int
                Number of time steps in the time dimension.
        """
    
    def forward_pass(self, paths_OOS, strike, r, option_type, M,T,n,reg_coef):
        # paths - Simulated paths
        # Strike - Strike price
        # r - Short-term interest rate
        # option_type - Put or Call
        # M - degree of Laguerre polynomials
        # T - Total time period (in years)
        # n - number of steps
    
        # Define regression coefficients
        regression_coefficients = reg_coef
    
        cash_flows_forward = np.zeros_like(paths_OOS)
        if option_type == "Call":
            cash_flows_forward = np.maximum(paths_OOS - strike, 0)
        else:
            cash_flows_forward = np.maximum(strike - paths_OOS, 0)
        
        discounted_cash_flows_forward = np.zeros_like(paths_OOS)
    
        dt = T/n
    
        for t in range(1,n): 
            reg_coef = regression_coefficients[t-1]
            
            # Finding all ITM paths
            if option_type == 'Call':
                in_the_money = paths_OOS[t] > strike
            else:  # for 'put'
                in_the_money = paths_OOS[t] < strike
            
            not_exercised  = np.all(discounted_cash_flows_forward[:t,:] == 0, axis=0)
    
            if np.any(in_the_money & not_exercised):
    
                X = paths_OOS[t, in_the_money & not_exercised] 
    
                Xs = Polynomials().power_basis(X, M)
    
                conditional_exp = Xs @ reg_coef 
                
                exercise_now = cash_flows_forward[t, in_the_money & not_exercised] > conditional_exp
    
                # Update cashflows and exercise policy
                discounted_cash_flows_forward[t, in_the_money & not_exercised] = np.where(exercise_now, cash_flows_forward[t, in_the_money & not_exercised]*np.exp(-r * dt * t), 0)
        
        not_exercised_last  = np.all(discounted_cash_flows_forward[:n,:] == 0, axis=0)
        discounted_cash_flows_forward[n,not_exercised_last] = np.where(discounted_cash_flows_forward[n,not_exercised_last] == 0, cash_flows_forward[n, not_exercised_last]*np.exp(-r * dt * n), 0)
       
        see = sem(discounted_cash_flows_forward)
        option_value = np.sum(discounted_cash_flows_forward) / paths_OOS.shape[1]
        return option_value, see
        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    