import numpy as np
from scipy.linalg import inv 

class ImplicitFiniteDifference:

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

    def finite_diff(self, S0, K, r, sigma, T, Smax, Smin, m, n):

        delta_T = T / n
        delta_S = (Smax - Smin) / m    


        a_ls = np.zeros(m + 1)
        b_ls = np.zeros(m + 1)
        c_ls = np.zeros(m + 1)
        for j in range(m + 1):
            a_ls[j] = r/2 * j * delta_T - sigma ** 2 * j ** 2 * delta_T / 2
            b_ls[j] = 1 + sigma ** 2 * j ** 2 * delta_T + r * delta_T
            c_ls[j] = - r/2 * j * delta_T - sigma ** 2 * j ** 2 * delta_T / 2


        coeff_matrix = np.zeros((m - 1, m - 1))
       
        c_m_1 = c_ls[m - 1] 
        coeff_matrix[0, 0] = b_ls[m - 1]
        coeff_matrix[0, 1] = a_ls[m - 1]
        
        coeff_matrix[m - 2, m - 3] = c_ls[1]
        coeff_matrix[m - 2, m - 2] = b_ls[1]
        a1 = a_ls[1] 

       
        for j in range(1, m - 2):  
            for k in range(j - 1, j + 2):  
                
                if k == j - 1:
                    coeff_matrix[j, k] = c_ls[m - j - 1]
                elif k == j:
                    coeff_matrix[j, k] = b_ls[m - j - 1]
                elif k == j + 1:
                    coeff_matrix[j, k] = a_ls[m - j - 1]

        f_matrix = np.zeros((m + 1, n + 1))

        
        for i in range(0, n):
            
            f_matrix[0, i] = max(K - Smax, 0)
            
            f_matrix[m, i] = max(K - Smin, 0)

        
        for j in range(m + 1):  
            
            f_matrix[m - j, n] = max(K - (Smin + j * delta_S), 0)

        
        for i in range(n - 1, -1, -1): 

            
            known_f_ls = f_matrix[1:m, i + 1] 
            

            known_f_ls[0] = known_f_ls[0] - c_m_1 * f_matrix[0, i]
            known_f_ls[m -2] = known_f_ls[m - 2] - a1 * f_matrix[m, i]

            unknown_f_ls = inv(coeff_matrix) @ known_f_ls


            for j in range(1, m):   
                    
                exercise_value = max(max(K - (Smin + (m - j) * delta_S), 0),0)
                
                intrinsic_value = unknown_f_ls[j - 1]
                f_matrix[j, i] = max(exercise_value, intrinsic_value)
                    
                f_matrix[j, i] = max(f_matrix[j, i], 0)

       
        index_j = int(m * (S0 - Smin) / (Smax - Smin)) 
        
        target_j = m - index_j
        option_value = f_matrix[target_j, 0]

        return option_value
