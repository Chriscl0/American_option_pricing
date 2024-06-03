"""
Here is all the functions used to get lower and upper bound for the 
american put option price using NN. The code used for the Heston model is almost
the same. The differences can be seen throughout the code, where you need to 
replace the line for GBM with the one after the following:

# Heston
# -------------


The code is made with inspiration from JiahaoWu27- American-Option-Pricing
Link:
https://github.com/JiahaoWu27/American-Option-Pricing/blob/main/generals.py
"""


import torch
import numpy as np
import torch
from torch.autograd import grad
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TKAgg')
import csv
import os.path
import os
import time


class Network(nn.Module):
    def __init__(self, n_in, n_neu, n_out):
        """
        
        Parameters
        ----------
        n_in : 
            number of input neurons.
        n_neu : 
            number of neurons in hidden layer.
        n_out : 
            number of output-neurons.

        """
        # Makes it possible for PyTorch to define and train neural networks
        super(Network, self).__init__() 
        
        # Initializes a 'Modulelist' with input layers and a activation function
        self.layers = nn.ModuleList([nn.Linear(n_in, n_neu[0]), 
                                     nn.Softplus(),]) 
        # A loop, which add hidden layers and activation functions to self.layers
        for i in range(len(n_neu)-1): 
            self.layers.append(nn.Linear(n_neu[i],n_neu[i+1]))
            # Add Sofplus activation function
            self.layers.append(nn.Softplus()) 
        # Append the output layer 
        self.layers.append(nn.Linear(n_neu[-1], n_out)) 
        
    # Defines the forward pass in the network. This method is used to compute the output given the input x.
    def forward(self, x): 
        # Iterate over all the layers
        for i in range(len(self.layers)):
            # Use every layer on the input
            x = self.layers[i](x) 
        return x
    
class NN:
    def __init__(self, model_conti, model_mg,generator, opt,loss_f):
        """
        Parameters:
        ----------
        model_conti : 
            Neural network model that computes the continuation values.
        model_mg :
            Neural network model that computes Psi values for dM calculation.
        """
        self.model_conti = model_conti
        self.model_mg = model_mg
        self.generator = generator
        self.opt = opt
        self.loss_f = loss_f

    def valuenn(self, features, multiplier):
        """
        Computes the continuation values and dM using two neural network models.
    
        Parameters:
        ----------
        features :
            Input features to the neural network models.
        multiplier : 
            Multiplier values to be applied to the model's outputs.
        model_conti : 
            Neural network model that computes the continuation values.
        model_mg :
            Neural network model that computes Psi values for dM calculation.
    
        Returns:
        ----------
        tuple: A tuple containing:
            - conti[:, 0] : 
                The first column of the continuation values.
            - dM : 
                The derived value computed from the sum of Psi values and multiplier.
                
        """
        # Get the continuation values from the continuation model
        conti = self.model_conti(features)
        
        # Get the Psi values from the martingale
        Psi = self.model_mg(features)
        
        # Compute dM by summing the products of Psi values and multiplier
        dM = torch.sum(Psi*multiplier, 1, False) 
        return conti[:, 0], dM
    
    def modelnn_dict(self, exe_step, sub_step, pathname, action):
        if action == 'save':
            torch.save(self.model_conti.state_dict(), pathname+'_Time' +
                        str(exe_step)+'_Sub'+str(sub_step)+'_Conti')
            torch.save(self.model_mg.state_dict(), pathname+'_Time' +
                        str(exe_step)+'_Sub'+str(sub_step)+'_Mg')
        if action == 'load':
            self.model_conti.load_state_dict(torch.load(
                    pathname+'_Time'+str(exe_step)+'_Sub'+str(sub_step)+'_Conti'))
            self.model_mg.load_state_dict(torch.load(
                    pathname+'_Time'+str(exe_step)+'_Sub'+str(sub_step)+'_Mg'))
    
    def train_model(self, current_step, current_sub, X, Y, pathname, **kwargs):
        """
        Trains a model on the given dataset.

        Parameters:
        ----------    
        valuenn: 
            Neural network used to compute continuous values and discrete changes.
        modelnn_dict: 
            Dictionary that handles saving and loading model parameters.
        current_step: 
            The current time step in the training process.
        current_sub: 
            The current sub-step in the training process.
        X: 
            Input features for the training.
        Y: 
            Target values for the training.
        pathname: 
            Path to save the model parameters.
        **kwargs: 
            Additional keyword arguments

        Returns:
        ----------
        - conti
        - dM
        - epoch
        """    
        epoch, patience_now = 0, 0
        
        best_mse = 10**8
        
        while patience_now < kwargs['patience'] and (epoch < kwargs['max_epoch']):
            
            train_feature, train_label, val_feature, val_label = random_split(
                X, Y, kwargs['N_train'], self.generator)
            for i in range(kwargs['N_batch']):
                ind_l, ind_r = i*kwargs['batch_size'], (i+1)*kwargs['batch_size']
                
                # Heston
                #conti, dM = self.valuenn(train_feature[ind_l: ind_r, 0: 2], 
                #                      train_feature[ind_l: ind_r, 2::])
                conti, dM = self.valuenn(train_feature[ind_l: ind_r, 0: 1],
                                      train_feature[ind_l: ind_r, 1::])
                # Set cashflow according to (17) in project
                cash_flow = conti + dM
                
                # Resets the gradients from the earlier iteration.
                self.opt.zero_grad()
                loss = self.loss_f(cash_flow, train_label[ind_l: ind_r]) 
                
                # Backpropagation. Calculates the gradients.
                loss.backward()
                
                # Updates the models parameters based on the calculated gradients.
                
                self.opt.step()
                
            epoch += 1
            
            # Heston
            #conti, dM = self.valuenn(val_feature[:, 0: 2], 
            #                    val_feature[:, 2::])
            conti, dM = self.valuenn(val_feature[:, 0: 1],
                                val_feature[:, 1::])
            cash_flow = conti + dM
            loss = self.loss_f(cash_flow, val_label)
            
    
            val_loss = loss.detach()
            if val_loss < best_mse:
                best_mse = val_loss
                
                torch.save(self.model_conti.state_dict(), pathname+'_Time' +
                            str(current_step)+'_Sub'+str(current_sub)+'_Conti')
                torch.save(self.model_mg.state_dict(), pathname+'_Time' +
                            str(current_step)+'_Sub'+str(current_sub)+'_Mg')
                
                patience_now = 0
            else:
                patience_now += 1
        
        self.model_conti.load_state_dict(torch.load(
                pathname+'_Time'+str(current_step)+'_Sub'+str(current_sub)+'_Conti'))
        self.model_mg.load_state_dict(torch.load(
                pathname+'_Time'+str(current_step)+'_Sub'+str(current_sub)+'_Mg'))  
          
        # Heston  
        #conti, dM = self.valuenn(X[:, 0: 2], X[:, 2::]) 
        conti, dM = self.valuenn(X[:, 0: 1], X[:, 1::]) 
           
        return conti.detach(), dM.detach(), epoch # Detach is used to stop the gradient calculations
    
    # Heston
    #def Training(Stock, Vol, dW_S, dW_V, Exe, pathname, **kwargs):
    def Training(self, Stock, dW, Exe, pathname, **kwargs):
        """
        Handles the training process over multiple time steps. "Epochs" refer 
        to the number of times the entire dataset is passed through the training 
        algorithm. An epoch involves a complete pass through all the training data, 
        meaning that each data point has had the opportunity to update the 
        model's weights once.

        Parameters:
        ----------  
        Stock: 
            Stock prices used in training.
        dW: 
            Random fluctuations used in the training.
        Exe: 
            Execution matrix for decision making.
        pathname: 
            Path to save the model parameters.
        valuenn: 
            Neural network used to compute continuous values and discrete changes.
        modelnn_dict: 
            Dictionary that handles saving and loading model parameters.
        **kwargs: 
            Additional keyword arguments

        Returns:
        ----------
        Epochs: 
            Array containing the number of epochs for each training step.
        """
        cash_flow_train, upper_bound_train = (torch.clone(Exe[:, -1]) for _ in range(2))    
        Epochs = np.empty(kwargs['total_step'])
        for i in range(kwargs['N_step']-1, -1, -1):
            for j in range(kwargs['sub_step']-1, -1, -1):
                step = i*kwargs['sub_step']+j
                cash_flow_train *= kwargs['discount']
                
                # Heston
                # X = torch.cat((Stock[:,step,:], Vol[:,step,:], dW_S[:, step, :], dW_S[:, step, :]**2-1,
                #               dW_V[:, step, :], dW_V[:, step, :]**2-1), dim=1) 
                X = torch.cat((Stock[:,step,:], dW[:, step, :], dW[:, step, :]**2-1), dim=1)
                conti, mg_pred, Epochs[step] = self.train_model(i, j, X, cash_flow_train, pathname, **kwargs)
                upper_bound_train = upper_bound_train*kwargs['discount']-mg_pred
                cash_flow_train -= mg_pred
            
            cash_flow_train, upper_bound_train = NN_decision().decision_backward(
                conti, Exe[:, i], cash_flow_train, upper_bound_train)
            
        return Epochs


    def Testing(self,pathname, device, S_mean, S_std, **kwargs):
        """
        Tests the trained model on a test dataset.

        Parameters:
        ----------  
        pathname: 
            Path to load the model parameters.
        device: 
            Device (CPU or GPU) to perform testing.
        S_mean: 
            Mean of stock prices used in testing.
        S_std: 
            Standard deviation of stock prices used in testing.
        valuenn: 
            Neural network used to compute continuous values and discrete changes.
        modelnn_dict: 
            Dictionary that handles saving and loading model parameters.
        **kwargs: 
            Additional keyword arguments

        Returns:
        ---------- 
        Mean values of the cash flows and upper bounds from the testing process.
        """
        M_test, cash_flow_test, upper_bound_test = (
            torch.zeros((kwargs['N_test']), device=device) for _ in range(3))
        
        # Heston
        #Stock, Vol, dW_S, dW_V, exe_now, _, _, _, _ = paths_Heston(
        #    device, False, kwargs['sub_step'], kwargs['N_test'], kwargs['S0_test'],
        #    kwargs['V0_test'], S_mean[0: kwargs['sub_step'], :], S_std[0: kwargs['sub_step'], :], 
        #    V_mean[0: kwargs['sub_step'], :], V_std[0: kwargs['sub_step'], :], **kwargs)   
        
        Stock, dW, exe_now, _, _ = simulation_NN().paths(
            device, False, kwargs['sub_step'], kwargs['N_test'], kwargs['S0_test'], 
            S_mean[0: kwargs['sub_step'], :], S_std[0: kwargs['sub_step'], :], **kwargs)    
        
        exe_now[:, 0] = -np.inf
        for i in range(0, kwargs['N_step']):  
            for j in range(kwargs['sub_step']):
                self.modelnn_dict(i, j, pathname, 'load')
                
                # Heston
                # dWs = torch.cat((dW_S[:, j, :], dW_S[:, j, :]**2-1,
                #                 dW_V[:, j, :], dW_V[:, j, :]**2-1), dim=1)
                
                dWs = torch.cat((dW[:, j, :], dW[:, j, :]**2-1), dim=1)
                with torch.no_grad():
                    # Heston
                    # conti, M_incre = self.valuenn(
                    #    torch.cat((Stock[:, j, :], Vol[:, j, :]), dim=1), dWs) 
                    conti, M_incre = self.valuenn(Stock[:, j], dWs)
                    
                if j == 0:
                    cash_flow_test,upper_bound_test=NN_decision().decision_forward(
                        i, conti.detach(), kwargs['discount']**kwargs['sub_step'], exe_now[:, 0],
                        cash_flow_test, upper_bound_test, M_test)
                M_test += M_incre.detach()*(kwargs['discount']**(i*kwargs['sub_step']+j))
                
            if i < kwargs['N_step']-1:
                
                # Heston
                #Stock, Vol, dW_S, dW_V, exe_now, _, _, _, _= paths_Heston(
                #    device, False, kwargs['sub_step'], kwargs['N_test'], Stock[:, -1, :], 
                #    Vol[:, -1, :], S_mean[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], 
                #    S_std[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :],
                #    V_mean[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], 
                #    V_std[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :],  **kwargs) 
                
                Stock, dW, exe_now, _, _ = simulation_NN().paths(
                    device, False, kwargs['sub_step'], kwargs['N_test'], Stock[:, -1, :], 
                    S_mean[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], 
                    S_std[(i+1)*kwargs['sub_step']:(i+2)*kwargs['sub_step'], :], **kwargs)    
        cash_flow_test, upper_bound_test = NN_decision().decision_forward(
            kwargs['N_step'], -np.inf, kwargs['discount']**kwargs['sub_step'], exe_now[:, 1], 
            cash_flow_test, upper_bound_test, M_test)
        
        return torch.mean(cash_flow_test).item(), torch.mean(upper_bound_test).item()


class simulation_NN:
    def __init__(self):
        """
        Class for simulation of GBM and Heston
        """
        
    def payoffs(self, option_type, x, K):    
        if option_type == 'put':
           # the payoff function is (K-S_{t})^{+})
           y = K-torch.squeeze(x, -1)

           y [y <= 0] = 0

        return y
    
    def paths(self, device, first, steps, N, S0, S_mean, S_std, drift, sigma_dt, generator = None, **kwargs):
        """
       Simulates paths for the underlying asset prices using a geometric Brownian motion model.
    
       Parameters:
       ----------
       device : 
           The device to perform computations on (CPU or GPU).
       first : 
           A flag indicating if this is the first run 
       steps : 
           The number of time steps
       N : 
           The number of pathS
       S0 : 
           The initial stock price.
       S_mean : 
           The mean of the stock prices 
       S_std : 
           The standard deviation of the stock prices
       drift : 
           The drift term 
       sigma_dt : 
           Volatility
       generator : torch.Generator, optional
           A random number generator for reproducibility (default is None).
       kwargs : dict
           - 'sub_step'
           - 'option_type'
           - 'strike'
    
       Returns:
       ----------
       tuple:
           A tuple containing:
           - S : 
               Simulated paths of the stock prices.
           - dW : 
               The generated random walks.
           - Exe : 
               The payoff values at exercise points.
           - S_mean : 
               The mean of the stock prices (for normalization).
           - S_std :
               The standard deviation of the stock prices (for normalization).
    
        """
        
        S = torch.zeros((N, steps+1, 1), device=device)
        dW = torch.randn(size=(int(N), steps, 1), generator = generator, device = device)
        
        S[:, 0, :] = S0
        lnS = torch.log(S0)
        
        for step in range(steps):
            lnS = lnS + drift + sigma_dt*dW[:, step, :]
            S[:, step+1, :] = torch.exp(lnS)
        Exe_mask = np.arange(0, steps+1, kwargs['sub_step'])
        Exe = self.payoffs(kwargs['option_type'], S[:, Exe_mask, :], kwargs['strike'])
        
        if first == True:
            S_mean = S[:,:-1,:].mean(dim = 0)
            S_std = S[:,:-1,:].std(dim = 0)
            
            S_std[0,:] = 1
        S[:,:-1,:] = (S[:,:-1,:]-S_mean)/S_std
        return S, dW, Exe, S_mean, S_std
    
    def paths_Heston(self, device, first, steps, N, S0, V0, S_mean, S_std, V_mean, V_std, N_stock, r, theta, kappa, sigma, dt, rho1, rho2, generator = None, **kwargs):
        S, V = (torch.zeros((N, steps+1, N_stock), device=device) for _ in range(2))
        dW_V = torch.randn(size=(int(N), steps, N_stock), generator = generator, device = device)
        dW_S = rho1*dW_V + rho2*torch.randn(size=(int(N), steps, N_stock), generator = generator, device = device)
        S[:, 0, :], V[:, 0, :] = S0, V0
        lnS = torch.log(S0)
        for i in range(steps):
            
            V_t = V[:, i, :]
            exp_kappa_dt = np.exp(-kappa * dt)
            term1 = 0.5 * sigma**2 * kappa**(-1) * V_t * (1 - np.exp(-2 * kappa * dt))
            term2 = (exp_kappa_dt * V_t + (1 - exp_kappa_dt) * theta)**2
            gamma_squared = dt**(-1) * np.log(1 + term1 / term2)
            
            V[:, i+1, :] = np.maximum(
                    0, (exp_kappa_dt * V_t + (1 - exp_kappa_dt) * theta) * np.exp(-0.5 * gamma_squared * dt + torch.sqrt(gamma_squared)* torch.sqrt(dt) *dW_V[:, i, :])
                )
            
            #V[:, i+1, :] = V[:, i, :] + kappa*(theta - V[:, i, :]) + sigma*torch.sqrt(V[:, i, :])*dW_V[:, i, :]
            #V[:, i+1, :] [V[:, i+1, :] < 0] = 0
            lnS +=  (r - V[:, i, :]/2)*dt + torch.sqrt(V[:, i, :])*torch.sqrt(dt)*dW_S[:, i, :]
            S[:, i+1, :] = torch.exp(lnS)
        Exe_mask = np.arange(0, steps+1, kwargs['sub_step'])
        Exe = self.payoffs(kwargs['option_type'], S[:, Exe_mask, :], kwargs['strike'])
        if first == True:
            S_mean, S_std = S[:,:-1,:].mean(dim = 0), S[:,:-1,:].std(dim = 0)
            V_mean, V_std = V[:,:-1,:].mean(dim = 0), V[:,:-1,:].std(dim = 0)
            S_std[0,:], V_std[0,:] = 1, 1
        S[:,:-1,:] = (S[:,:-1,:]-S_mean)/S_std
        V[:,:-1,:] = (V[:,:-1,:]-V_mean)/V_std
        return S, V, dW_S, dW_V, Exe, S_mean, S_std, V_mean, V_std

class NN_decision:
    def __init__(self):
        """
        Initialization of the decision class.
        """

    def decision_backward(self, ex_conti, ex_now, cash_flow, upper_bound):
        """
        This function updates the cash flow and upper bound based on the comparison 
        between current exercise values and continuation values.
    
        Parameters:
        ----------
        ex_conti : 
            Continuation values.
        ex_now : 
            Current exercise values.
        cash_flow : 
            Current cash flow values.
        upper_bound : 
            Upper bound values.
    
        Returns:
        ----------
        tuple: 
            Updated cash flow and upper bound.
        """
        
        # always substract martingale increments
        mask = ((ex_conti<ex_now) & (ex_now>0))
        cash_flow[mask] = ex_now[mask]
        # Is updated where the current exercise value is greater than or equal to the upper bound. 
        upper_bound[(ex_now>=upper_bound)]=ex_now[(ex_now>=upper_bound)]
        
    
        return cash_flow, upper_bound


    def decision_forward(self, ite, ex_conti, discount, exe_now, cash_flow, upper_bound, 
                         m_test):
        """
        Updates the cash flow and upper bound values based on the current iteration,
        exercise continuation values, discount factor, current exercise values, 
        and other parameters.
    
        Parameters:
        ----------
        ite : 
            Current iteration.
        ex_conti : 
            Continuation values.
        discount : 
            Discount factor.
        exe_now :  
            Current exercise values.
        cash_flow :  
            Current cash flow values.
        upper_bound : 
            Upper bound values.
        m_test :  
            Martingale values.
    
        Returns:
        ----------
        tuple: 
            Updated cash flow and upper bound.
        """
        # cash_flow, m_test and upper_bound are always discounted back at 0
        # Compute the current exercise value at iteration 0, considering discount and m_test
        exe_now_at0 = (exe_now)*(discount**ite)-m_test 
        
        # Update the upper bound where the current exercise value at iteration 0 is greater
        upper_bound[upper_bound < exe_now_at0] = exe_now_at0[upper_bound < exe_now_at0] 
        # Create a mask where the continuation value is less than the current exercise value 
        # and cash flow is still zero
        mask = ((ex_conti<exe_now) & (cash_flow==0)) 
        # Update cash flow for positions where the mask is True
        cash_flow[mask] = exe_now_at0[mask]
        
        return cash_flow, upper_bound

    def num_free_variable(self, inputs, neuron, outputs):
        """
        Calculates the number of free variables in a neural network.
    
        Parameters:
        ----------
        inputs :
            Number of input features.
        neuron : 
            List where each element represents the number of neurons in each hidden layer.
        outputs : 
            Number of output neurons.
    
        Returns:
        ----------
        int: 
            Total number of free variables (parameters) in the network.
        """
        # Initialize the number of layers
        layer = len(neuron)
        
        # Calculate the number of parameters between the input layer and the first hidden layer
        num_var = (inputs + 1) * neuron[0] 
        
        # Calculate the number of parameters between all subsequent hidden layers
        for i in range(layer - 1):
            num_var += (neuron[i]+1)*neuron[i+1]
        
        # Calculate the number of parameters between the last hidden layer and the output layer
        num_var += (neuron[-1]+1)*outputs
        
        return num_var

class prep_sim:
    def __init__(self):
        """
        Initialization of simulation class
        """
    def prep_kwargs(self, S0, r, sigma, T, my_device, my_option, my_training):
        """
        Prepares and updates keyword arguments for option pricing and training.
    
        Parameters:
        ----------
        S0 : 
            Initial stock price
        r : 
            Risk-free interest rate
        sigma : 
            Volatility
        T : 
            Time to maturity
        my_device  :
            Device to perform computations on (CPU or GPU).
        my_option : 
            Dictionary containing option parameters.
        my_training : 
            Dictionary containing training parameters.
    
        Returns:
        ----------
        tuple: 
            Updated my_option and my_training dictionaries.
        """
        
        # Calculate total number of steps in training
        my_training['total_step'] = my_training['sub_step'] * my_option['N_step']
        
        # Calculate time increment per step
        dt = torch.tensor(T / my_training['total_step'], device = my_device)
        
        # Update option parameters
        my_option['discount'] = torch.exp(-r * dt)
        my_option['drift'] = (r - sigma**2 / 2) * dt
        my_option['sigma_dt'] = sigma * torch.sqrt(dt)
        
        # Initialize stock prices for testing
        my_training['S0_test'] = S0*torch.ones((my_training['N_test'], 1),
                                                device=my_device)
        
        # Initialize stock prices for training
        my_training['S0_train'] = S0*torch.ones((my_training['N_path'], 1),
                                                device=my_device)
            
        # Calculate number of training paths
        my_training['N_train'] = int(my_training['N_path']*(1-my_training['val']))
            
        # Calculate number of batches
        my_training['N_batch'] = int(my_training['N_train']/my_training['batch_size'])    
        
        return my_option, my_training
    
    def prep_kwargs_Heston(self, S0, V0, T, my_device, version, my_option, my_training):
        my_training['total_step'] = my_training['sub_step']*my_option['N_step']
        dt = torch.tensor(T/my_training['total_step'], device = my_device)
        my_option ['dt'] = dt
        my_option['discount'] = torch.exp(-my_option['r']*dt)
        my_option['kappa'] = my_option['kappa']
        my_option['sigma'] = my_option['sigma']
        my_option['rho2'] = torch.sqrt(1-my_option['rho1']**2) 
        
        my_training['S0_test'] = S0*torch.ones((my_training['N_test'], my_option['N_stock']),
                                                device=my_device)
        my_training['V0_test'] = V0*torch.ones((my_training['N_test'], my_option['N_stock']),
                                                device=my_device)

        my_training['S0_train'] = S0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                                device=my_device)
        my_training['V0_train'] = V0*torch.ones((my_training['N_path'], my_option['N_stock']),
                                                device=my_device)
        my_training['N_train'] = int(my_training['N_path']*(1-my_training['val']))
        my_training['N_batch'] = int(my_training['N_train']/my_training['batch_size'])    
        
        return my_option, my_training

def random_split(feature, label, train_size, generator):
    """
    Splits the dataset into training and validation sets randomly.

    Parameters:
    ----------
    feature : 
        The input features of the dataset.
    label : 
        The labels corresponding to the input features.
    train_size : 
        The number of samples to include in the training set.
    generator : 
        The random number generator.

    Returns:
    ----------
    tuple:
        A tuple containing:
        - Training features
        - Training labels 
        - Validation features 
        - Validation labels 
    """
    
    whole_size = feature.shape[0]
    indices = torch.randperm(whole_size, generator=generator).tolist()  
    return feature[indices[0 : train_size]], label[indices[0 : train_size]], feature[indices[train_size::]], label[indices[train_size::]]


