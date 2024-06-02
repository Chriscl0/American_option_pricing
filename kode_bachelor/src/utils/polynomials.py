# Import relevant packages
import numpy as np
import pandas as pd

from sklearn.linear_model import LinearRegression
from numpy.polynomial.laguerre import lagfit, lagval
from scipy.special import eval_laguerre

class Polynomials:
    def __init__(self):
        """Initialize the class with the maximum degree for the Laguerre polynomials."""

    def laguerre_basis(self, x, degree):
        """
        Generate Laguerre basis functions for a given x up to the specified degree.

        Parameters:
        x : array
            Input values for which the Laguerre basis functions are calculated.

        degree : The maximum degree for the Laguerre polynomials
        
        Returns:
        np.array
            Array of shape (len(x), degree+1) containing the Laguerre basis functions evaluated at x.
        """
        
        basis_functions = []
        for i in range(degree + 1):
            basis_function = np.exp(-x / 2) * eval_laguerre(i, x)
            basis_functions.append(basis_function)
        
        return np.column_stack(basis_functions)
    
    def laguerre_derivatives(self, x, degree):
        """
        Generate Laguerre derivatives up to the given degree, where degree <= 3.
        
        
        Parameters:
        x : array
            Input values for which the Laguerre basis functions are calculated.

        degree : The maximum degree for the Laguerre polynomials
        
        Returns:
        np.array
            Array of shape (len(x), degree+1) containing the Laguerre basis functions evaluated at x.
       """
        
        if degree > 3:
            raise ValueError("Degree must be 3 or less.")
    
        # Base function, exp(-x/2)
        weight_pol = np.exp(-x / 2)
        
        derivatives = []
        
        # Calculate derivatives based on the degree
        if degree >= 0:
            # First derivative
            d1 = weight_pol * (-1 / 2)
            derivatives.append(d1)
        if degree >= 1:
            # Second derivative: 
            d2 = weight_pol * (-1 / 2) * (1 - x) -weight_pol
            derivatives.append(d2)
        if degree >= 2:
            # Third derivative: 
            d3 = weight_pol * (-1 / 2) * (1 - 2*x + 0.5 * x**2) +weight_pol*(x-2)
            derivatives.append(d3)
        if degree == 3:
            # Fourth derivative: 
            d4 = weight_pol * (-1 / 2) * (1 - 3 * x + 1.5 * x**2 - x**3 / 6) + weight_pol*(-3+3*x-0.5 * x**2)
            derivatives.append(d4)
    
        return np.column_stack(derivatives)
    
    
    def power_basis(self, x, degree):
        """
        Generate power basis polynomials for a given x up to the specified degree.

        Parameters:
        x : array
            Input vector of data for which to calculate the power basis.

        degree : The maximum degree for the polynomials
        
        Returns:
        np.array
            A matrix Xs with shape [n, degree+1] where n is the length of x, and each column corresponds
            to x raised to the power of the column index (i.e., x^0, x^1, ..., x^degree).
        """
        
        n = x.size
        Xs = np.ones((n, degree + 1))  # Initializes matrix with ones for x^0
        for i in range(1, degree + 1):
            Xs[:, i] = x**i  # Fills each column with x raised to the ith power
        return Xs
    
    def weighted_power_basis(self, x, degree):
       """
       Generate weighted power basis polynomials for a given x up to the specified degree.

       Parameters:
       x : array
           Input vector of data for which to calculate the weighted power basis.
           
       degree : The maximum degree for the polynomials    
           
       Returns:
       np.array
           A matrix Xs with shape [n, degree+1] where n is the length of x. The first column
           is zeros, and each subsequent column corresponds to i*x^(i-1).
       """
       n = x.size
       Xs = np.zeros((n, degree + 1))  # Initializes matrix with zeros for x^0 (which remains zero)
       for i in range(1, degree + 1):
           Xs[:, i] = i * x ** (i - 1)  # Fills each column with i * x^(i-1)
       return Xs
    
    
    
    
    