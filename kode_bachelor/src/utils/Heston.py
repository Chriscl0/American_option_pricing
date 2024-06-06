import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression

from datetime import datetime, timedelta
from scipy.stats import norm

class Heston_model:
    """
    Class to simulate paths for the Heston model and getting a lower and upper bound for the option price using LSM and Delta LSM
    """
    def __init__(self, kappa, theta, rho):
        """
        Initialize the Heston model class.
        """
        self.kappa = kappa
        self.theta = theta
        self.rho = rho
        
        
    def simulate_heston_paths(self, s0, v0, kappa, theta, sigma, rho, r, T, n, N):
        """
        Simulate stock price and volatility paths using the Heston model.

        Parameters:
        - s0: Initial stock price
        - v0: Initial volatility
        - kappa: Rate at which volatility reverts to theta
        - theta: Long-term volatility mean
        - sigma: Volatility of volatility
        - rho: Correlation between stock and volatility shocks
        - r: Risk-free interest rate
        - T: Time to maturity
        - n: Number of time steps
        - N: Number of simulation paths

        Returns:
        - S: Simulated stock prices
        - V: Simulated volatilities
        """
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
        
        return S, V

    def Heston_LSM(self, S, V, r, T, K, n):
        """
        Calculate the option price using the Least Squares Monte Carlo (LSM) algorithm with polynomial regression for backward recursion.

        Parameters:
        - S: Simulated stock prices
        - V: Simulated volatilities
        - r: Risk-free interest rate
        - dt: Time step
        - K: Strike price
        - n: Number of time steps
        - N: Number of simulation paths

        Returns:
        - option_value: Estimated option price
        """
        dt = T/n
        
        payoff = np.maximum(K -  S[-1, :], 0)

        regression_coefficients = np.zeros((n-1, 10))

        for t in range(n - 1, 0, -1):
            in_the_money = S[t, :] < K
            if np.any(in_the_money):
                X = S[t, in_the_money]
                Y = payoff[in_the_money] * np.exp(-r * dt)
                y = V[t, in_the_money]
                
                # Polynomial regression
                Xs = np.column_stack([X**i * y**j for i in range(4) for j in range(4-i)])
                XtX = Xs.T @ Xs
                XtY = Xs.T @ Y
                reg_coef = np.linalg.solve(XtX, XtY)  # Solve the linear equation
                continuation_values = Xs @ reg_coef

                regression_coefficients[t-1, :] = reg_coef

                early_exercise = np.maximum(K - X , 0) > continuation_values
                payoff[in_the_money] = np.where(early_exercise, np.maximum(K - X, 0), Y)
                payoff[~in_the_money] = payoff[~in_the_money]* np.exp(-r * dt)
                        
        # Calculate the option value using Monte Carlo simulation
        discount_factor = np.exp(-r * dt)
        option_value = discount_factor * np.mean(payoff)
        return option_value, regression_coefficients
    
    def Heston_Delta_LSM(self, S, V, r, T, K, n):
        """
        Calculate the option price using the Delta LSM algorithm with polynomial regression for backward recursion.

        Parameters:
        - S: Simulated stock prices
        - V: Simulated volatilities
        - r: Risk-free interest rate
        - dt: Time step
        - K: Strike price
        - n: Number of time steps
        - N: Number of simulation paths

        Returns:
        - option_value: Estimated option price
        """
        dt = T/n
        
        regression_coefficients = np.zeros((n-1, 10))

        payoff = np.maximum(K -  S[-1, :], 0)
        
        optimal_price = S[-1, :]

        Z = np.array(S[-1, :]) / np.array(S[-2, :])
        
        for t in range(n - 1, 0, -1):
            in_the_money = S[t, :] < K
  
            if np.any(in_the_money):
                
                X = S[t, in_the_money]
                Y = payoff[in_the_money] * np.exp(-r * dt)
                y = V[t, in_the_money]
                
                # Polynomial regression
                Xs = np.column_stack([X**i * y**j for i in range(4) for j in range(4-i)])
                Xss = np.column_stack([i * X**(i-1) * y**j if i > 0 else np.zeros(X.shape) for i in range(4) for j in range(4-i)])

                h  = np.where(optimal_price[in_the_money] < K, -1, 0)
    
                delta = np.array(Z[in_the_money])*np.array(h)

                l = np.linalg.norm(-Y)**2 / np.linalg.norm(delta)**2
                    
                XtX = Xs.T @ Xs
                lXssT_Xss = l * (Xss.T @ Xss)
                X_inv = np.linalg.pinv(XtX + lXssT_Xss)

                reg_coef = X_inv @ (Xs.T @ Y + l * (Xss.T @ delta))

                continuation_values = Xs @ reg_coef

                regression_coefficients[t-1, :] = reg_coef

                early_exercise = np.maximum(K - X , 0) > continuation_values
                payoff[in_the_money] = np.where(early_exercise, np.maximum(K - X, 0), Y)
                payoff[~in_the_money] = payoff[~in_the_money]* np.exp(-r * dt)
                
                price_ratio = S[t] / S[t-1]

                Z[in_the_money] = np.where(early_exercise,
                           price_ratio[in_the_money], 
                           Z[in_the_money] * price_ratio[in_the_money]) 

                optimal_price[in_the_money] = np.where(early_exercise, 
                                                       S[t,in_the_money], 
                                                       optimal_price[in_the_money])
                
                Z[~in_the_money] = Z[~in_the_money] * price_ratio[~in_the_money]               


        # Calculate the option value using Monte Carlo simulation
        discount_factor = np.exp(-r * dt)
        option_value = discount_factor * np.mean(payoff)
        
        return option_value, regression_coefficients
    
    def Heston_Delta_LSM_backup(self, S, V, r, T, K, n):
        """
        Calculate the option price using the Delta LSM algorithm with polynomial regression for backward recursion.

        Parameters:
        - S: Simulated stock prices
        - V: Simulated volatilities
        - r: Risk-free interest rate
        - dt: Time step
        - K: Strike price
        - n: Number of time steps
        - N: Number of simulation paths

        Returns:
        - option_value: Estimated option price
        """
        dt = T/n
        
        regression_coefficients = np.zeros((n-1, 10))

        payoff = np.maximum(K -  S[-1, :], 0)
        optimal_price = S[-1] 

        for t in range(n - 1, 0, -1):
            in_the_money = S[t, :] < K
            if np.any(in_the_money):
                X = S[t, in_the_money]
                Y = payoff[in_the_money] * np.exp(-r * dt)
                y = V[t, in_the_money]
                # Polynomial regression
                Xs = np.column_stack([X**i * y**j for i in range(4) for j in range(4-i)])
                xss = np.column_stack([i * X**(i-1) * y**j if i > 0 else np.zeros(X.shape) for i in range(4) for j in range(4-i)])

                h  = np.where(optimal_price[in_the_money] < K, -1, 0)
                Z = np.array(optimal_price[in_the_money]) / np.array(X) * np.array(h)

                l = np.linalg.norm(-Y)**2 / np.linalg.norm(Z)**2
                    
                XtX = Xs.T @ Xs
                lXssT_Xss = l * (xss.T @ xss)
                X_inv = np.linalg.pinv(XtX + lXssT_Xss)

                reg_coef = X_inv @ (Xs.T @ Y + l * (xss.T @ Z))

                continuation_values = Xs @ reg_coef

                regression_coefficients[t-1, :] = reg_coef

                early_exercise = np.maximum(K - X , 0) > continuation_values
                payoff[in_the_money] = np.where(early_exercise, np.maximum(K - X, 0), Y)
                payoff[~in_the_money] = payoff[~in_the_money]* np.exp(-r * dt)

                optimal_price[in_the_money] = np.where(early_exercise, S[t,in_the_money],optimal_price[in_the_money])
                    
                       
        # Calculate the option value using Monte Carlo simulation
        discount_factor = np.exp(-r * dt)
        option_value = discount_factor * np.mean(payoff)
        return option_value, regression_coefficients
    
    def nested_simulation(self, S, V, r, sigma, t, T, n, strike, nested_sim_n, reg_coef):
        """
        S: Price right now
        r: Short-term interest rate
        sigma: Volatility 
        t: The period we are in right now
        T: Total time period (in years)
        n: number of steps
        strike: Strike price
        num_paths: Number of paths in simulation
        reg_coeff: Regression coefficient from the Primal LSM
        option_type: Put or Call
        """
        
        # Update number of steps and total time period
        nn = n-t
        TT  = T*nn/n
        # Simulate new nested paths
        nested_paths, V = Heston_model(self.kappa, self.theta, self.rho).simulate_heston_paths(S, V, self.kappa, self.theta, sigma, self.rho, r, TT, nn, nested_sim_n)
    
        # Defining all possible cashflows
        cash_flows = np.zeros_like(nested_paths)
        cash_flows = np.maximum(strike - nested_paths, 0)
        # Defining discounted cashflows
        discounted_cash_flows = np.zeros_like(nested_paths)
        
        # discount factor
        dt = T / n 
        reg_nested = reg_coef[t:,]
        
        # Make forward pass for each path
        for i in range(1,nn): 
            reg_coeff = reg_nested[i-1,:]
            
            in_the_money = nested_paths[i] < strike
            
            not_exercised  = np.all(discounted_cash_flows[:i,:] == 0, axis=0)
            if np.any(in_the_money & not_exercised):
                X = nested_paths[i, in_the_money & not_exercised] 
                y = V[i, in_the_money & not_exercised] 
                Xs = np.column_stack([X**i * y**j for i in range(4) for j in range(4-i)])
                conditional_exp = Xs @ reg_coeff 
                
                exercise_now = cash_flows[i, in_the_money & not_exercised] > conditional_exp
                # Update cashflows and exercise policy
                discounted_cash_flows[i, in_the_money & not_exercised] = np.where(exercise_now, cash_flows[i, in_the_money & not_exercised]*np.exp(-r * dt * i), 0)
        
        # Seeing whether exercise is need in period T
        not_exercised_last  = np.all(discounted_cash_flows[:nn,:] == 0, axis=0)
        discounted_cash_flows[nn,not_exercised_last] = np.where(discounted_cash_flows[nn,not_exercised_last] == 0, cash_flows[nn, not_exercised_last]*np.exp(-r * dt * nn), 0)
        
        # Calculate price and print
        option_value = np.sum(discounted_cash_flows) / nested_paths.shape[1]
        return option_value
    
    def Dual_LSM(self, S, V, strike, r,  sigma, option_type, nested_sim_n, T, n, reg_coef):
        """
        strike: Strike 
        paths_dual: The simulated paths used for the dual LSM
        r: Short-term interest rate
        sigma: Volatility 
        option_type: Put or Call
        M: degree of polynomials
        T: Total time period (in years)
        n: number of steps
        reg_coeff: Regression coefficient from the Primal LSM
        num_paths: Number of paths in simulation
        """
        
        # discount factor
        dt = T / n
        
        # Define regression coefficient used for the Primal LSM
        reg_dual = reg_coef
        
        # Defining all possible cashflows
        cash_flows = np.zeros_like(S)
        if option_type == "Call":
            cash_flows = np.maximum(S - strike, 0)
        else:
            cash_flows = np.maximum(strike - S, 0)
        # Defining discounted cashflows
        discount_factors = np.exp(-r * dt * np.arange(n + 1))
        discount_cf = cash_flows * discount_factors.reshape(-1, 1)
        # Define the martingale used to find the upper bound
        Martingale = np.zeros_like(S)
        
        # Define matrix to store expected values in
        expect_value = np.zeros_like(cash_flows)
        # Get expected value for time 1, which is calculated in time 0
        for i in range(S.shape[1]):  # Loop over each path
            expect_value[0, i] = self.nested_simulation(S[0, i], V[0,i], r, sigma, 0, T, n, strike, nested_sim_n, reg_dual) 
        # Make forward pass for each path
        for t in range(1, n):
            in_the_money = S[t] < strike
            in_the_money_indices = np.where(in_the_money)[0]
            out_of_the_money_indices = np.where(~in_the_money)[0]

            X = S[t, in_the_money]
            y = V[t, in_the_money]
            Xs = np.column_stack([X**i * y**j for i in range(4) for j in range(4-i)])
            conditional_exp = Xs @ reg_dual[t-1]
            
            # Handling in-the-money paths
            for idx, real_idx in enumerate(in_the_money_indices):
                price = self.nested_simulation(X[idx], y[idx], r, sigma, t, T, n, strike, nested_sim_n, reg_dual)* np.exp(-r * dt * t)
                if cash_flows[t, real_idx] > conditional_exp[idx]:
                    Martingale[t, real_idx] = Martingale[t-1, real_idx] + discount_cf[t, real_idx] - expect_value[t-1, real_idx]
                else:
                    Martingale[t, real_idx] = Martingale[t-1, real_idx] + price - expect_value[t-1, real_idx]
                expect_value[t, real_idx] = price
                
            # Handling out-of-the-money paths
            x_cont = S[t, ~in_the_money]
            y_cont = V[t, ~in_the_money]
            for idx, real_idx in enumerate(out_of_the_money_indices):
                price = self.nested_simulation(x_cont[idx], y_cont[idx], r, sigma, t, T, n, strike, nested_sim_n, reg_dual) * np.exp(-r * dt * t)
                Martingale[t, real_idx] = Martingale[t-1, real_idx] + price - expect_value[t-1, real_idx]
                expect_value[t, real_idx] = price
        for i in range(S.shape[1]):
            Martingale[n,i] = Martingale[n-1,i] + cash_flows[n,i]*np.exp(-r * dt * n) - expect_value[n-1,i]
        
        # Getting the maximum for each path of discounted cashflow minus the respective martingale
        final_values = np.max(discount_cf - Martingale, axis=0)
        
        option_value_upper_bound = np.mean(final_values)
        return option_value_upper_bound

