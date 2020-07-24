####################################
########### Module Load ############
####################################
# Basic module import
import sys
sys.path.append("../src/")
import numpy as np
import math
import pandas as pd

# PYTORCH packages
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from torch.nn.parameter import Parameter

# Customized 
from utils import linear_generator, data_load_l, measure, mse
from models import Net_linear
from dfs import DFS_epoch, training_l

#######################
### Data Generation ###
#######################
K = 100
p = 1000
N = 1000
s = 100

# Generate K datasets 
np.random.seed(1)
#linear_generator(p, N, s, K)

### Function for finding best s using BIC criteria
def optimal_s(k, Ss = list(range(90, 115)), optimal_c=1):
    #Ss = np.array([ 50, 70, 90, 91, 92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 120, 140, 160, 180, 200]) # Candidates s
    #Ss = np.arange(90, 115)
    BIC = [] # list to store bic values for candidates s
    # Load Data
    dirc = "../data/linear/p_1000_N_1000_s_100/"
    X, Y, X_test, Y_test, supp_true = data_load_l(k, directory=dirc)
    for i, s in enumerate(Ss):
        c = optimal_c # tuning parameter for lambda, i.e. the ratio of lambda_1 and lambda_2.
        # Training dataset k with given s
        model, supp, bic, _, [err_train, err_test] = training_l(X, Y, X_test, Y_test, supp_true, c, s, epochs=15, C=3.)
        # Store bic values
        BIC.append(bic)
        # if current bic is the smallest, save the trained model, support and other metric
        if bic == min(BIC):
            best_model = model
            best_supp = supp
            best_err_train, best_err_test = err_train, err_test # one step model training and testing error
    
    idx = np.argmin(BIC) # index for best s
    # Generate automatic for best s of dataset k
    report = "For datasets "+str(k)+":\n"
    report += "    Optimal s: "+str(Ss[idx])+"\n"
    res_dict = {"s": Ss[idx],
                "BICs": BIC,
                "model": best_model,
                "support": best_supp,
                "true": supp_true,
                "report": report,
                "train_mse": best_err_train,
                "test_mse": best_err_test
               }
    return res_dict

##################################
### Multiple datasets training ###
##################################
Report = "" # Text report for linear regression methods

# List to save results for different datasets
BICs = [] # BICs over different s for each datasets, for the purpose of plotting 
SUPPs = [] # best support selected for each datasets
TRUEs = [] # true support for each datasets
#ERR_train_1 = [] # one-step training error
#ERR_test_1 = [] # one-step testing error
ERR_train = [] # two-step training error
ERR_test = [] # two-step testing error

### Training Over K datasets (K is set to 10 for shorter training time)
K = 10
for k in range(K):
    #X, Y, X_test, Y_test, supp_true = data_load(k) # data load for two-step mse calculation
    res = optimal_s(k) # single datasets training
    #mse_train = mse(model, X, Y) # two-step training mse
    #mse_test = mse(model, X_test, Y_test) # two-step testing mse
    ### automatic text report
    Report += res["report"]
    Report += "    Training MSE: "+str(res["train_mse"])+"\n"
    Report += "    Testing MSE: "+str(res["test_mse"])+"\n"
    # Results saving
    BICs.append(res["BICs"])
    SUPPs.append(res["support"])
    TRUEs.append(res["true"])
    ERR_train.append(res["train_mse"])
    ERR_test.append(res["test_mse"])

############################################
### Metric calculation and result saving ###
############################################

BICs = np.array(BICs)
DIR_res = "../outputs/reports/"
#np.savetxt(DIR_res+"linear_BICs.txt", BICs)
#np.savetxt(DIR_res+"linear_train_2step.txt", np.array(ERR_train))
#np.savetxt(DIR_res+"linear_test_2step.txt", np.array(ERR_test))
#supp_file = open(DIR_res+"linear_supp.txt", "a")
#for supp in SUPPs:
#    supp_file.write(str(supp))
#    supp_file.write("\n")
#supp_file.close()

fsr, nsr = measure(TRUEs, SUPPs)
final_report = "For " + str(K) + " datasets:\n"
final_report += "  False Selection Rate: " + str(fsr) + "\n"
final_report += "  Negative Selection Rate: " + str(nsr) + "\n"
final_report += "  Training error: " + str(np.mean(ERR_train)) + "(" + str(np.std(ERR_train)) + ")\n"
final_report += "  Testing error: " + str(np.mean(ERR_test)) + "(" + str(np.std(ERR_test)) + ")\n"

Report = final_report + Report
report_file = open(DIR_res+"DFS_linear.txt", "a")
report_file.write(Report)
report_file.close()