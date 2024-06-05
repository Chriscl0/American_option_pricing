# =========================================================================== #
#                      Import all the relevant packages                       #               
# =========================================================================== #

# For working with dataframes, vectors and conducting mathematical operations
import numpy as np
import pandas as pd

# For visualisation
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns

# To check time for the algorithms
import time

# To create network
import networkx as nx
# To get stock data
import yfinance as yf

# To get P-value
from scipy.stats import chi2


from scipy.stats import norm

from sklearn.linear_model import LinearRegression
from numpy.polynomial.laguerre import lagfit, lagval
from scipy.special import eval_laguerre
from aleatory.processes import BrownianMotion

# Import functions we are going to use
from src.utils.LSM import LSM
from src.utils.polynomials import Polynomials
from src.utils.simulations import GeometricBrownianMotion as GBM
from src.utils.dual_lsm import Dual_LSM as Dual
from src.utils.finite import ImplicitFiniteDifference as IFD
from src.utils.Heston import Heston_model
from src.utils.forward_pass import forward_pass
from src.utils.delta_LSM import *


# For NN
from NN_functions import *
import torch
import torch.nn as nn

import os

# For sobol sequence
from scipy.stats.qmc import Sobol

# =========================================================================== #
#             Make plot for Geometric Brownian Motion (Figure 1)              #               
# =========================================================================== #

# Parameters
S0 = 36  # Initial stock price
r = 0.06  # Risk-free rate
sigma = 0.2  # Volatility
T = 1  # Time to maturity
n = 50  # Number of time steps
num_paths = 100  # Number of paths

# Time points
dt = T / n
t = np.linspace(0, T, n + 1)

# Generate GBM paths
paths = np.zeros((n + 1, num_paths))
paths[0] = S0

for i in range(1, n + 1):
    z = np.random.standard_normal(num_paths)  # Random variates
    paths[i] = paths[i - 1] * np.exp((r - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)

# Plotting
plt.figure(figsize=(10, 6))
plt.title('Geometric Brownian Motion Stier', fontsize=16, fontweight='bold')
plt.xlabel('T')
plt.ylabel('Aktie Prisen')
for i in range(num_paths):
    plt.plot(t, paths[:, i], linewidth=1)
plt.grid(True)
plt.show()


# =========================================================================== #
#  Compare our LSM implementation with Longstaff and Schwartz paper (Table 1) #               
# =========================================================================== #

r = 0.06 # Risk-free rate
strike=40 # Strike price
n = 50 # number of steps
num_paths = 100000 # Number of simulation paths
Method = "LSM" # Method used
Reg_Method = "Polynomium" # Polynomium type
option_type="Put" # option type
M = 3 # Polynomium Degree
num_simulations = 100 # Number of simulation for each parameter combination


option_data = [] 

# Iterate over all combinations
for S0 in [36, 38, 40, 42, 44]:
    for sigma in [0.2, 0.4]:
        for T in [1, 2]:
            prices = []  
            se_paths = []
            for _ in range(num_simulations):
                n_T = n * T  
                paths_gbm = GBM().gbm(S0, r, sigma, T, n_T, num_paths)
                option_price, se = LSM().LSM_algorithm(paths_gbm, strike, r, option_type, M, T, n_T, Method, Reg_Method)
                prices.append(option_price) 
                se_paths.append(se)
            # Calculate mean
            average_price = np.mean(prices)
            se = np.mean(se_paths)
            # Append the mean prices and se
            option_data.append({
                "S0": S0,
                "sigma": sigma,
                "T": T,
                "Option Price": average_price,
                "se": se
            })

# Create DataFrame
options_df = pd.DataFrame(option_data)

# Show DataFrame
print(options_df)


# =========================================================================== #
#                      Primal and Dual LSM (Table 2)                          #               
# =========================================================================== #
# Start out by getting the benchmark
# input variable
K = 40
r = 0.06
q = 0 
Smax = 100
Smin = 0

m = 100 # for stock price partition
n = 1000 # for time partition

results_ifd = [] 

# Iterate over all combinations
for S0 in (36, 38, 40, 42, 44):
    for sigma in (0.2, 0.4):
        for T in (1, 2):
            Benchmark = IFD().finite_diff(S0, K, r, sigma, T, Smax, Smin, m, n)
            results_ifd.append({
                "S0": S0,
                "sigma": sigma,
                "T": T,
                "Benchmark": Benchmark
            })
            
# Create Dataframe
results_ifd_df = pd.DataFrame(results_ifd)

# Show DataFrame
print(results_ifd_df)

# Calculate lower and upper bound for the price
# Parameter ranges
S0_values = [36, 38, 40, 42, 44]
sigma_values = [0.2, 0.4]
T_values = [1, 2]
methods =  ["LSM"]
strike = 40
option_type = "Put"
M = 3
r = 0.06

# Number of simulations 
num_paths = 250000
num_paths_dual = 2000
num_paths_nested = 500
n = 50

results_lsm_dual = []


# Iterate over all combinations
for S0 in S0_values:
    for sigma in sigma_values:
        for T in T_values:
            n_T = n * T  # Adjust n based on T
            paths = GBM().gbm(S0, r, sigma, T, n_T, num_paths)
            paths_dual = GBM().gbm(S0, r, sigma, T, n_T, num_paths_dual)
            
            BM = results_ifd_df[(results_ifd_df['S0'] == S0) & (results_ifd_df['sigma'] == sigma) & (results_ifd_df['T'] == T)]["Benchmark"].iloc[0]
            BM = "{:.3f}".format(BM)

            record = {"S0": S0, "sigma": sigma, "T": T}
            record["BM"] = BM
            
            for Method in methods:
                reg_coef, price = Dual().get_regression(paths, strike, r, option_type, M, T, n_T, Method)
                dual_result = Dual().Dual_LSM(paths_dual, strike, r, sigma, option_type, M, T, n_T, reg_coef, num_paths_nested)
                record["LSM price"] = price
                record["LSM dual price"] = dual_result

            # Append to table
            results_lsm_dual.append(record)

# Create pandas DataFrame
results_lsm_dual_df = pd.DataFrame(results_lsm_dual)

print(results_lsm_dual_df)


# =========================================================================== #
#               LSM and Delta LSM for IS and OOS (Table 3)                  #               
# =========================================================================== #

r = 0.06  # Drift (average return per unit time)
n = 50  # number of steps
num_paths = 100000  # Number of simulation paths for each batch
num_simulations = 100  # Number of simulations to average

option_prices_lsm = []

for S0 in [36,38,40,42,44]:
    for sigma in [0.2,0.4]:
        for T in [1,2]:
            for Method in ['LSM','Delta_LSM']:
                simulation_prices = []
                forward = []
                se_lsm = []
                se_forward = []
                
                for i in range(num_simulations):
                    paths_gbm = GBM().gbm(S0, r, sigma, T, n*T, num_paths)
                    paths_gbm_oos = GBM().gbm(S0, r, sigma, T, n*T, num_paths)
                    option_price, reg_coef, se = LSM_Delta_LSM(paths=paths_gbm, strike=40, r=r, option_type="Put", M=3, T=T, n=n*T, Method = Method)
                    price_oos, se_for = forward_pass().forward_pass(paths_OOS=paths_gbm_oos, strike=40, r=r, option_type = "Put", M = 3, T=T ,n=n*T , reg_coef=reg_coef)
                    
                    simulation_prices.append(option_price)
                    se_lsm.append(se)

                    forward.append(price_oos)
                    se_forward.append(se_for)

                average_price = np.mean(simulation_prices) 
                forward_avg_price = np.mean(forward) 
                se_lsm_avg = np.mean(se_lsm)
                se_forward_avg = np.mean(se_forward)

                option_prices_lsm.append((S0, sigma, T, Method, average_price, se_lsm_avg, forward_avg_price, se_forward_avg))

# Opret en DataFrame
columns = ['S0', 'Sigma', 'T', 'Method', 'LSM/Delta LSM IS', 'SE LSM IS', 'LSM/Delta LSM OOS', 'SE OOS']
df = pd.DataFrame(option_prices_lsm, columns=columns)

# =========================================================================== #
#     Make plot for relative error for LSM and Delta LSM (Figure 8)       #               
# =========================================================================== #

error_lsm_is = 10000*(df[df['Method']== 'LSM']['LSM/Delta LSM IS'].values-results_ifd_df['Benchmark'].values)/results_ifd_df['Benchmark'].values
error_lsm_oos =  10000*(df[df['Method']== 'LSM']['LSM/Delta LSM OOS'].values-results_ifd_df['Benchmark'].values)/results_ifd_df['Benchmark'].values
error_delta_lsm_is = 10000*(df[df['Method']== 'Delta_LSM']['LSM/Delta LSM IS'].values-results_ifd_df['Benchmark'].values)/results_ifd_df['Benchmark'].values
error_delta_lsm_oos = 10000*(df[df['Method']== 'Delta_LSM']['LSM/Delta LSM OOS'].values-results_ifd_df['Benchmark'].values)/results_ifd_df['Benchmark'].values

plt.figure(figsize=(10,6))
#plt.plot(range(1,21), error_lsm_is, label = 'LSM IS', color = 'red', linestyle = '-')
plt.plot(range(1,21), error_lsm_oos, label = 'LSM OOS', color = 'red', linestyle = '--')
#plt.plot(range(1,21), error_delta_lsm_is, label = 'Delta LSM IS', color = 'blue', linestyle = '-')
plt.plot(range(1,21), error_delta_lsm_oos, label = 'Delta LSM OOS', color = 'blue', linestyle = '--')

plt.axhline(y=np.mean(error_lsm_is), color = 'red',linestyle = '-', label = 'LSM IS Mean')
plt.axhline(y=np.mean(error_lsm_oos), color = 'red',linestyle = '--', label = 'LSM OOS Mean')
plt.axhline(y=np.mean(error_delta_lsm_is), color = 'blue',linestyle = '-', label = 'Delta LSM IS Mean')
plt.axhline(y=np.mean(error_delta_lsm_oos), color = 'blue',linestyle = '--', label = 'Delta LSM OOS Mean')


plt.xticks(range(1,21,1))
plt.xlabel('Scenarie')
plt.ylabel('Relative fejl i basis point')
plt.title('Relative fejl for LSM og Delta LSM')
plt.legend()
plt.show()


# =========================================================================== #
#               Primal and Dual LSM and Delta LSM (Table 4)                 #               
# =========================================================================== #

# Parameters
strike = 40
option_type = "Put"
M = 3
r = 0.06

# Number of simulations 
num_paths = 250000
num_paths_dual = 2000
num_paths_nested = 500
n = 50

results_table_4 = []


# Iterate over all combinations
for S0 in [36, 38, 40, 42, 44]:
    for sigma in [0.2, 0.4]:
        for T in [1, 2]:
            n_T = n * T  # Adjust n based on T
            paths = GBM().gbm(S0, r, sigma, T, n_T, num_paths)
            paths_dual = GBM().gbm(S0, r, sigma, T, n_T, num_paths_dual)

            record = {"S0": S0, "sigma": sigma, "T": T}
            
            for Method in ["LSM", "Delta LSM"]:
                
                start_time = time.time()  # Start timer
        
                target_price = results_ifd_df[(results_ifd_df['S0'] == S0) & (results_ifd_df['sigma'] == sigma) & (results_ifd_df['T'] == T)]["Benchmark"].iloc[0]

                while True: 
                    paths = GBM().gbm(S0, r, sigma, T, n_T, num_paths)
                    paths_dual = GBM().gbm(S0, r, sigma, T, n_T, num_paths_dual)
                    reg_coef, price = Dual().get_regression(paths, strike, r, option_type, M, T, n_T, Method)
                    
                    if target_price - 5 <= price <= target_price + 5:
                        start_time = time.time() 
                        result = Dual().Dual_LSM(paths_dual, strike, r, sigma, option_type, M, T, n_T, reg_coef, num_paths_nested)
                        break
                    else:
                        # To make sure the lower bound is close to the mean estimated in table 3. 
                        print(f"Price {price} is not within the target range. Retrying get_regression.")
                        
                elapsed_time = time.time() - start_time  # End timer
                
                # Append results to the list
                results_table_4.append({
                    'S0': S0,
                    'sigma': sigma,
                    'T': T,
                    'Method': Method,
                    'Lower bound': price,
                    'Upper bound': result,
                    'Time': elapsed_time
                    })


# Create pandas DataFrame
df_results = pd.DataFrame(results_table_4)


df_results[(df_results['sigma'] == sigma) & (df_results['T'] == T) & (df_results['Method'] == "LSM")]["Lower bound"].values

results_ifd_df[(results_ifd_df['sigma'] == sigma) & (results_ifd_df['T'] == T)]["Benchmark"].values

# =========================================================================== #
#                Make plot for price interval (Figure 2,9,10,11)              #               
# =========================================================================== #

# Define the parameter combinations based on your specifications
combinations = [(0.2, 1), (0.4, 1), (0.2, 2), (0.4, 2)]

# Loop over each combination to create plots
for index, (sigma_val, T_val) in enumerate(combinations):
    # Defining the data
    spot_prices = [36, 38, 40, 42, 44]
    
    # LSM
    lsm_lower = df_results[(df_results['sigma'] == sigma) & (df_results['T'] == T) & (df_results['Method'] == "LSM")]["Lower bound"].values
    
    lsm_upper = df_results[(df_results['sigma'] == sigma) & (df_results['T'] == T) & (df_results['Method'] == "LSM")]["Upper bound"].values
    
    lsm_diff = (lsm_upper - lsm_lower)   
    
    # Delta LSM
    delta_lsm_lower = df_results[(df_results['sigma'] == sigma) & (df_results['T'] == T) & (df_results['Method'] == "Delta LSM")]["Lower bound"].values
        
    delta_lsm_upper = df_results[(df_results['sigma'] == sigma) & (df_results['T'] == T) & (df_results['Method'] == "Delta LSM")]["Upper bound"].values
    
    delta_diff = (delta_lsm_upper - delta_lsm_lower)  
    
    benchmark = results_ifd_df[(results_ifd_df['sigma'] == sigma) & (results_ifd_df['T'] == T)]["Benchmark"].values
    
    # Setup for subplots: 1 row, 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))
    
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.size'] = 12
    
    plt.subplots_adjust(top=0.5)

    # Plotting on the first subplot
    ax1 = axs[0]
    ax2 = ax1.twinx()
    ax2.bar(spot_prices, lsm_diff, color='lightblue', alpha=0.6, label='Prisinterval bredde', width=1)
    ax1.plot(spot_prices, benchmark, label='Benchmark', color='blue', linewidth=0.75, zorder=10)
    ax1.fill_between(spot_prices, lsm_lower, lsm_upper, color='darkgreen', alpha=0.3, label='Pris interval', edgecolor='darkgreen', linewidth=2, zorder=5)
    ax1.set_title('LSM', fontsize=18)
    ax1.set_xlabel('$S_0$ (Spot Pris)')
    ax1.set_ylabel('V($0,S_0$)')
    ax1.grid(True)

    ax2.set_zorder(ax1.get_zorder()-1)
    ax1.set_frame_on(False)

    # Adjust y-axis limits to create space at the top
    ax1.set_ylim([min(benchmark)-0.2, max(benchmark) * 1.2])
    ax2.set_ylim([0, max(lsm_diff) * 1.25])

    # Combine legends for the first subplot
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    combined_lines_1 = lines_1 + lines_2
    combined_labels_1 = labels_1 + labels_2
    ax1.legend(combined_lines_1, combined_labels_1, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)

    # Plotting on the second subplot
    ax3 = axs[1]
    ax4 = ax3.twinx()
    ax4.bar(spot_prices, delta_diff, color='lightblue', alpha=0.6, label='Prisinterval bredde', width=1, zorder=1)
    ax3.plot(spot_prices, benchmark, label='Benchmark', color='blue', linewidth=0.75, zorder=10)
    ax3.fill_between(spot_prices, delta_lsm_lower, delta_lsm_upper, color='darkgreen', alpha=0.3, label='Pris interval', edgecolor='darkgreen', linewidth=2, zorder=1)
    ax3.set_title('Delta LSM', fontsize=18)
    ax3.set_xlabel('$S_0$ (Spot Pris)')
    ax3.grid(True)
    
    ax4.set_zorder(ax3.get_zorder()-1)
    ax3.set_frame_on(False)
    
    # Adjust y-axis limits to create space at the top
    ax3.set_ylim([min(benchmark)-0.2, max(benchmark) * 1.2])
    ax4.set_ylim([0, max(lsm_diff) * 1.25])
    
    # Combine legends for the second subplot
    lines_3, labels_3 = ax3.get_legend_handles_labels()
    lines_4, labels_4 = ax4.get_legend_handles_labels()
    combined_lines_2 = lines_3 + lines_4
    combined_labels_2 = labels_3 + labels_4
    ax3.legend(combined_lines_2, combined_labels_2, loc='upper center', bbox_to_anchor=(0.5, 1), ncol=3)

    # Adjust layout
    plt.suptitle('Sammenligning af Øvre og Nedre grænse for LSM og Delta LSM', fontsize=20, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.subplots_adjust(top=0.86) 
    
    # Show the plot
    plt.show()


# =========================================================================== #
#                  Make plot for Neural Network (Figure 3)                    #               
# =========================================================================== #
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Define the layers
layers = [3, 4, 4, 1]

# Create a directed graph
G = nx.DiGraph()

# Add nodes with positions
pos = {
    0: (0, -0.5), 1: (0, -1.5), 2: (0, -2.5), 
    3: (1, 0), 4: (1, -1), 5: (1, -2), 6: (1, -3), 
    7: (2, 0), 8: (2, -1), 9: (2, -2), 10: (2, -3), 
    11: (3, -1.5)
}
G.add_nodes_from(pos.keys())

# Add edges between layers
edges = [(i, j) for i in range(3) for j in range(3, 7)] + \
        [(i, j) for i in range(3, 7) for j in range(7, 11)] + \
        [(i, j) for i in range(7, 11) for j in range(11, 12)]
G.add_edges_from(edges)

# Draw the network
fig, ax = plt.subplots(figsize=(10, 6))
nx.draw(G, pos, with_labels=False, node_size=1000, node_color='blue', font_size=10, ax=ax)

# Add rectangles around the layers
input_layer_rect = patches.Rectangle((-0.2, -2.8), 0.44, 2.6, linewidth=1, edgecolor='black', facecolor='none')
hidden_layer_1_rect = patches.Rectangle((0.75, -3.3), 0.5, 3.6, linewidth=1, edgecolor='black', facecolor='none')
hidden_layer_2_rect = patches.Rectangle((1.75, -3.3), 0.5, 3.6, linewidth=1, edgecolor='black', facecolor='none')
output_layer_rect = patches.Rectangle((2.78, -2.5), 0.44, 2.1, linewidth=1, edgecolor='black', facecolor='none')

ax.add_patch(input_layer_rect)
ax.add_patch(hidden_layer_1_rect)
ax.add_patch(hidden_layer_2_rect)
ax.add_patch(output_layer_rect)

# Add text labels
plt.text(0, -3, 'Input lag', fontsize=12, ha='center')
plt.text(1, -3.5, 'Skjulte lag', fontsize=12, ha='center')
plt.text(2, -3.5, 'Skjulte lag', fontsize=12, ha='center')
plt.text(3, -2.7, 'Output lag', fontsize=12, ha='center')

plt.title('Neuralt Netværksarkitektur', fontsize=14, fontweight='bold')
plt.axis('off')
plt.show()

# =========================================================================== #
#                  Make plot for activation function (Figure 4)               #               
# =========================================================================== #

# Define the Softplus and ReLU functions
def softplus(x):
    return np.log(1 + np.exp(x))

def relu(x):
    return np.maximum(0, x)

# Generate x values
x = np.linspace(-4, 4, 400)

# Compute y values for both functions
y_softplus = softplus(x)
y_relu = relu(x)

# Plot the functions
plt.figure(figsize=(8, 6))
plt.plot(x, y_softplus, label='Softplus', color='green')
plt.plot(x, y_relu, label='Rectifier', color='blue')

# Add labels and title
plt.xlabel('$x$')
plt.ylabel('$\sigma(x)$')
plt.title('Softplus vs ReLU Aktiveringsfunktion')
plt.axvline(x=0, color='black', linestyle='--')

# Add legend
plt.legend()

# Show the plot
plt.grid(True)
plt.show()

# =========================================================================== #
#                        ML for Primal and Dual (Table 5)                     #   
#              This code needs to be ran seperatly from the others            #
#                      The code is both for GBM and Heston                    # 
#       The code needs to be executed for all combinations of parameters      #
# =========================================================================== #

# parameters for American Option
torch.manual_seed(92)
generator = torch.Generator().manual_seed(torch.seed())
use_cuda = torch.cuda.is_available()
my_device = torch.device("cuda:0" if use_cuda else "cpu")

# American put option using GBM
S0 = 36  # S0 = 100 - Heston
sigma = 0.4 # V0 = 0.01
T = 1 # T = 1
r = 0.06 

# Option specifikationer
my_option = {
    'N_step': 50*T,  
    'strike': 40, 
    'option_type': "put",  
    'option_name': 'Put_option' 
}

# Heston
#my_option = {'N_step': 50, 
#             'strike': 100, 
#             'r': torch.tensor(0.1, device = my_device), 
#             'sigma': 0.2, 'kappa': 2, 
#             'theta': 0.01, 
#             'rho1': torch.tensor(-0.3, device = my_device), 
#             'option_type': "put", 
#              'option_name': 'Heston'}

my_training = {'N_path': int(10**5), 
               'N_test': int(10**5), 
               'batch_size': int(10**3), 
               'N_neuron_1': [20, 20], 
               'N_neuron_2': [20, 20], 
               'val': 0.1,
               'patience': 5, 
               'max_epoch':100} 

# Heston
#my_training = {'N_path': int(10**5), 
#               'N_test': int(10**5), 
#               'batch_size': int(10**3), 
#               'N_neuron_1': [40, 40], 
#               'N_neuron_2': [40, 40], 
#               'val': 0.1,
#               'patience': 5, 
#               'max_epoch':100} 


for sub_steps in [1]:
    my_training['sub_step'] = sub_steps
    
    # Heston
    # my_option, my_training = prep_sim().prep_kwargs_Heston(S0, V0, T, my_device,  my_option, my_training)
    my_option, my_training = prep_sim().prep_kwargs(S0, r, sigma, T, my_device, my_option, my_training)


    lrs = [0.005]
    loss_f = nn.MSELoss() 
    for lr in lrs:
        my_training['lr'] = lr
        #if my_training['TwoNN'] == True:
        path_root = ''
        path = "{}_{}{}_{}p_{}sub".format(
            lr, my_training['N_neuron_1'], my_training['N_neuron_2'], 
            my_training['patience'], 
            my_training['sub_step']).replace(",", "")
            
            
        # Heston
        #N_varibels = NN_decision().num_free_variable(2,  my_training['N_neuron_1'], 1) \
        #    + num_free_variable(2,  my_training['N_neuron_2'], 4)
        N_varibels = NN_decision().num_free_variable(1,  my_training['N_neuron_1'], 1) \
            + NN_decision().num_free_variable( 1,  my_training['N_neuron_2'], 2)
        try:
            os.mkdir(path_root+path)
        except OSError:
            print("Creation of directory %s failed" % path)
        else:
            print("Successfully created directory %s " % path)
    
        Epochs = np.zeros((1, my_training['total_step']))
        LB, UB, Train_time, Test_time = (np.zeros((1,)) for _ in range(4))
        for ite in range(1):
            model_loc = path+'/Trial_'+str(ite+1)
            
            # Heston
            #Stock, Vol, dW_S, dW_V, Exe, S_mean, S_std, V_mean, V_std = simulation_NN().paths_Heston(
            #    my_device, True, my_training['total_step'], my_training['N_path'], 
            #    my_training['S0_train'], my_training['V0_train'], None, None, 
            #    None, None, **my_option, **my_training)
            Stock, dW, Exe, S_mean, S_std = simulation_NN().paths(
                my_device, True, my_training['total_step'], my_training['N_path'], 
                my_training['S0_train'], None, None, **my_option, **my_training)

            start1 = time.time()
            
            # Heston 
            #model_conti = Network(2, my_training['N_neuron_1'], 1)
            #model_mg = Network(2, my_training['N_neuron_2'], 4)
            
            model_conti = Network(1, my_training['N_neuron_1'], 1)
            model_mg = Network(1, my_training['N_neuron_2'], 2)
            if use_cuda:
                model_conti.cuda()
                model_mg.cuda()
            opt = torch.optim.Adam(list(model_conti.parameters()) + 
                            list(model_mg.parameters()), lr)               
            # Heston
            #Epochs[ite, :] = NN(model_conti, model_mg,generator, opt,loss_f).Training(Stock, Vol, dW_S, dW_V, Exe, path_root+model_loc, **my_option, **my_training)

            Epochs[ite, :] = NN(model_conti, model_mg,generator, opt,loss_f).Training(Stock, dW, Exe, path_root+model_loc, **my_option, **my_training)
                            
            end1 = time.time()
            
            LB[ite], UB[ite] = NN(model_conti, model_mg,generator, opt,loss_f).Testing(path_root+model_loc, my_device, S_mean, S_std
                                                                                       , **my_option, **my_training)
            Train_time[ite] = end1-start1           
            Test_time[ite] = time.time()-end1  




# =========================================================================== #
#                       Plot volatility of SP500 (Figure 5)                   #               
# =========================================================================== #

multi_data = yf.download(["^SPX"], start="2014-04-04", end="2024-04-04")


# Calculating rolling 30-day volatility 
multi_data['Returns'] = multi_data['Adj Close'].pct_change()
multi_data['Volatility'] = multi_data['Returns'].rolling(window=30).std() * np.sqrt(252) 

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Plotting the volatility
plt.figure(figsize=(14, 7))
plt.plot(multi_data.index, multi_data['Volatility'], color='blue')
plt.title('Volatiliteten af SP500', fontsize=16, fontweight='bold')
plt.xlabel('Dato')
plt.ylabel('Volatilitet')
plt.grid(True)
plt.show()


# =========================================================================== #
#                              Jarque Bera test                               #               
# =========================================================================== #
# Downloading data
multi_data = yf.download(["^SPX"], start="2014-04-04", end="2024-04-04")

# Calculating daily log returns
multi_data['Log Returns'] = np.log(multi_data['Adj Close'] / multi_data['Adj Close'].shift(1))

# Dropping NA values
log_returns_data = multi_data['Log Returns'].dropna()

# Number of observations
n = len(log_returns_data)

# Mean and standard deviation of log returns
mean_log_returns = np.mean(log_returns_data)
std_log_returns = np.std(log_returns_data)

# Calculating skewness
skewness = np.mean(((log_returns_data - mean_log_returns) / std_log_returns) ** 3)

# Calculating kurtosis
kurtosis = np.mean(((log_returns_data - mean_log_returns) / std_log_returns) ** 4)

# Adjusting kurtosis to match the formula for Jarque-Bera test (subtract 3)
adjusted_kurtosis = kurtosis - 3

# Jarque-Bera statistic
jb_stat = (n / 6) * (skewness**2 + (adjusted_kurtosis**2) / 4)

# p-value (approximated using chi-squared distribution with 2 degrees of freedom)
jb_pvalue = 1 - chi2.cdf(jb_stat, df=2)

# Outputting the results with higher precision
print(f"Number of observations: {n}")
print(f"Mean of log returns: {mean_log_returns}")
print(f"Standard deviation of log returns: {std_log_returns}")
print(f"Skewness: {skewness}")
print(f"Kurtosis: {kurtosis}")  # Report the original kurtosis value
print(f"Adjusted Kurtosis: {adjusted_kurtosis}")
print(f"Jarque-Bera Statistic: {jb_stat:.10f}")
print(f"p-value: {jb_pvalue:.2e}")

            
# =========================================================================== #
#   Plot distribution log returns against normal distribution (Figure 6)      #               
# =========================================================================== #

# Downloading data
multi_data = yf.download(["^SPX"], start="2014-04-04", end="2024-04-04")

# Calculating daily log returns
multi_data['Log Returns'] = np.log(multi_data['Adj Close'] / multi_data['Adj Close'].shift(1))

# Drop NaN values
log_returns = multi_data['Log Returns'].dropna()

# Plotting the KDE of log returns
plt.figure(figsize=(9, 6))
sns.kdeplot(log_returns, color='blue', label='Log Afkast', fill=True)

# Setting mean and standard deviation
mean = np.mean(log_returns)
std_dev = np.std(log_returns)

# Generating normally distributed data with the same mean and standard deviation
normal_data = np.random.normal(mean, std_dev, size=100000)

# Plotting the KDE of the normal distribution
sns.kdeplot(normal_data, color='red', label='Normalfordelingen', fill=True)


# Adding titles and labels
plt.title('Sammenligning af Empirisk afkast og Normal-fordelingen', fontweight='bold')
plt.xlabel('Log Afkast')
plt.ylabel('Tæthed')
plt.legend()

# Show plot
plt.show()

# =========================================================================== #
#           Plot stock prices and volatility Heston (Figure 7)                #               
# =========================================================================== #

# Parameters for Heston model
s0 = 100        # initial stock price
v0 = 0.01       # initial variance
kappa = 2.0     # rate of mean reversion
theta = 0.01    # long-run variance
sigma = 0.2     # volatility of volatility
rho = -0.3     # correlation between the two Brownian motions
r = 0.1        # risk-free rate
T = 1        # time to maturity
n = 50         # number of time steps
N = 100         # number of paths

S,V = GBM().heston(s0, v0, kappa, theta, sigma, rho, r, T, n, N)


plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Plot the results with modified x-axis
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Heston Modellen', fontsize=16, fontweight='bold')

# Generate time steps for the x-axis
time_steps = np.linspace(0, T, n + 1)

# Plot stock prices
for i in range(N):
    ax1.plot(time_steps, S[:, i])
ax1.set_title('Aktie Priser')
ax1.set_xlabel('T')
ax1.set_ylabel('Pris')
ax1.set_xticks(np.linspace(0, T, 11))
ax1.set_xticklabels(np.round(np.linspace(0, T, 11), 1))

# Plot variance process
for i in range(N):
    ax2.plot(time_steps, np.sqrt(V[:, i]))
ax2.set_title('Volatilitet')
ax2.set_xlabel('T')
ax2.set_ylabel('Volatilitet')
ax2.set_xticks(np.linspace(0, T, 11))
ax2.set_xticklabels(np.round(np.linspace(0, T, 11), 1))

plt.tight_layout()
plt.show()

# =========================================================================== #
#                LSM and Delta LSM under Heston model (Table 6)               #               
# =========================================================================== #

# Define constant parameters
r = 0.1  # risk-free rate
kappa = 2  # mean reversion speed
theta = 0.1**2  # long-run variance
sigma = 0.2  # implied volatility
rho = -0.3  # correlation between the stock price and volatility
K = 100  # strike price
N = 100000  # number of simulation paths
N_dual = 2000
nested_sim_n = 500

# Define parameter sets to iterate over
S0_values = [100, 95]
v0_values = [0.01, 0.04]
T_values = [1, 2]

# List to store results
results = []

# Iterate through each combination of parameters
for s0 in S0_values:
    for v0 in v0_values:
        for T in T_values:
            n = int(T * 50)  
            
            start_time = time.time()  # Start timer

            # Simulate Heston paths
            S, V = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, rho, r, T, n, N)
            
            # Perform LSM on the simulated paths
            price_LSM, reg_coef_LSM = Heston_model(kappa, theta, rho).Heston_LSM(S, V, r, T, K, n)
            
            # Simulate Heston paths for dual LSM
            S_dual, V_dual = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, rho, r, T, n, N_dual)
            
            # Perform Dual LSM on the simulated paths
            dual_price = Heston_model(kappa, theta, rho).Dual_LSM(S_dual, V_dual, K, r, sigma, "Put", nested_sim_n, T, n, reg_coef_LSM)
            
            overall_time = time.time() - start_time  # End timer

            # Append results to the list
            results.append({
                'S0': s0,
                'v0': v0,
                'T': T,
                'LSM Price': price_LSM,
                'Dual LSM Price': dual_price,
                'Time': overall_time
            })

# Create a DataFrame from the results
df_results = pd.DataFrame(results)

# List to store results
results_delta = []

# Iterate through each combination of parameters
for s0 in S0_values:
    for v0 in v0_values:
        for T in T_values:
            n = int(T * 50)  
            start_time = time.time()  # Start timer
            
            # Simulate Heston paths
            S, V = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, rho, r, T, n, N)
            
            # Perform LSM on the simulated paths
            price_LSM, reg_coef_LSM = Heston_model(kappa, theta, rho).Heston_Delta_LSM(S, V, r, T, K, n)
            
            # Simulate Heston paths for dual LSM
            S_dual, V_dual = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, rho, r, T, n, N_dual)
            
            # Perform Dual LSM on the simulated paths
            dual_price = Heston_model(kappa, theta, rho).Dual_LSM(S_dual, V_dual, K, r, sigma, "Put", nested_sim_n, T, n, reg_coef_LSM)
            
            overall_time = time.time() - start_time  # End timer
            # Append results to the list
            results_delta.append({
                'S0': s0,
                'v0': v0,
                'T': T,
                'Delta LSM Price': price_LSM,
                'Dual Delta LSM Price': dual_price,
                'Time': overall_time
            })

# Create a DataFrame from the results
df_results_delta = pd.DataFrame(results_delta)

# =========================================================================== #
#                           Sobol sequence comparison                         #               
# =========================================================================== #

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 12

# Generate Sobol sequence
sobol_sampler = Sobol(d=2)
sobol_samples = sobol_sampler.random(n=512)

# Generate Uniform samples
uniform_samples = np.random.rand(512, 2)

# Plotting
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Sobol sequence plot
axes[0].scatter(sobol_samples[:, 0], sobol_samples[:, 1])
axes[0].set_title('2-dimensionale Sobol-sekvenser', fontweight='bold')
axes[0].set_xlim(0, 1)
axes[0].set_ylim(0, 1)

# Uniform samples plot
axes[1].scatter(uniform_samples[:, 0], uniform_samples[:, 1])
axes[1].set_title('2-dimensionale pseudo-tilfældige tal', fontweight='bold')
axes[1].set_xlim(0, 1)
axes[1].set_ylim(0, 1)

plt.tight_layout()
plt.show()

# =========================================================================== #
#                                Heston plots                                 #               
# =========================================================================== #

# Parameters for Heston model
s0 = 100.0           
T = 1.0               
r = 0.02             
M = 50               
N = 10000             
kappa = 2             
theta = 0.2**2      
v0 = 0.25**2         
sigma = 0.6           

rho_p = 0.98
rho_n = -0.98

S_p,v_p = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, rho_p, r, T, n, N)
S_n,v_n = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, rho_n, r, T, n, N)
S_t,v_t = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, theta, sigma, 0, r, T, n, N)



fig, ax = plt.subplots()
ax = sns.kdeplot(S_p[-1], label=r"$\rho= 0.98$", ax=ax)
ax = sns.kdeplot(S_n[-1], label=r"$\rho= -0.98$", ax=ax)
ax = sns.kdeplot(S_t[-1], label=r"$\rho= 0$", ax=ax)

#ax = sns.kdeplot(gbm, label="GBM", ax=ax)
plt.title(r'Asset Price Density under Heston Model')
plt.xlim([20, 180])
plt.xlabel('$S_T$')
plt.ylabel('Density')
plt.legend()
plt.show()

S_pp,v_pp = Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, 0.2**2, 0, 0, r, T, n, N)
S_nn,v_nn =Heston_model(kappa, theta, rho).simulate_heston_paths(s0, v0, kappa, 0.4**2, 0, 0, r, T, n, N)

#gbm = S0*np.exp( (r - #**2/2)*T + np.sqrt(theta)*np.sqrt(T)*np.random.normal(0,1,M) )

fig, ax = plt.subplots()
ax = sns.kdeplot(S_pp[-1], label=r"$v0 = 0.2^2, \theta = 0.2^2$", ax=ax)
ax = sns.kdeplot(S_nn[-1], label=r"$v0 = 0.4^2, \theta = 0.4^2$", ax=ax)
#ax = sns.kdeplot(gbm, label="GBM", ax=ax)
plt.title(r'Asset Price Density under Heston Model')
plt.xlim([20, 180])
plt.xlabel('$S_T$')
plt.ylabel('Density')
plt.legend()
plt.show()
