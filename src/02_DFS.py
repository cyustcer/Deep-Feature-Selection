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
from utils import nonlinear_generator, data_load_n, measure, accuracy
from models import Net_nonlinear
from dfs import DFS_epoch, training_n

#######################
### Data Generation ###
#######################
K = 5
N = 600
p = 500
s = 4
# Generate K datasets, you only need to do this once
np.random.seed(1)
# nonlinear_generator(N, s, p, K) # Generate K datasets in DATA/nonlinear/p_500_N_600_s_4/

### Function for finding best s using BIC criteria
def optimal_s(k, Ss = list(range(1, 11)), optimal_c=1, n_hidden1=50, n_hidden2=10, p=p):
    #Ss = range(1, 11) # Candidates s
    BIC = [] # list to store bic values for candidates s
    dirc = "../data/nonlinear/p_500_N_600_s_4/"
    X, Y, X_test, Y_test = data_load_n(k, directory=dirc)
    # reserve model for the save of best model
    best_model = Net_nonlinear(n_feature=p, n_hidden1=n_hidden1, n_hidden2=n_hidden2, n_output=2)
    for i, s in enumerate(Ss):
        c = optimal_c # tuning parameter for lambda, i.e. the ratio of lambda_1 and lambda_2.
        # Training dataset k with given s
        model, supp, bic, _, [err_train, err_test] = training_n(X, Y, X_test, Y_test, c, s, epochs=10)
        # Store bic values
        BIC.append(bic)
        # if current bic is the smallest, save the trained model, support and other metric
        if bic == min(BIC):
            best_model.load_state_dict(model.state_dict())
            best_supp = supp
            best_err_train, best_err_test = err_train, err_test # one step model training and testing error
            
    
    idx = np.argmin(BIC) # index for best s
    # Generate automatic for best s of dataset k
    accu_train = accuracy(best_model, X, Y)
    accu_test = accuracy(best_model, X_test, Y_test)
    report = "For datasets "+str(k)+":\n"
    report += "    Optimal s: "+str(Ss[idx])+"\n"
    res_dict = {"s": Ss[idx],
                "BICs": BIC,
                "model": best_model,
                "support": best_supp,
                "report": report,
                "train_accu": accu_train,
                "test_accu": accu_test
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
ERR_train = [] # two-step training error
ERR_test = [] # two-step testing error

### Training Over K datasets (K is set to 5 for shorter training time)
K = 3
for k in range(K):
    res = optimal_s(k) # single datasets training
    ### automatic text report
    Report += res["report"]
    Report += "    Selected Variables: "+str(res["support"])+"\n"
    Report += "    Training Accuracy: "+str(res["train_accu"])+"\n"
    Report += "    Testing Accuray: "+str(res["test_accu"])+"\n"
    # Results saving
    BICs.append(res["BICs"])
    SUPPs.append(res["support"])
    TRUEs.append(set([0, 1, 2, 3]))
    ERR_train.append(1-res["train_accu"])
    ERR_test.append(1-res["test_accu"])

############################################
### Metric calculation and result saving ###
############################################
BICs = np.array(BICs)
DIR_res = "../outputs/reports/"

fsr, nsr = measure(TRUEs, SUPPs)
final_report = "For " + str(K) + " datasets:\n"
final_report += "  False Selection Rate: " + str(fsr) + "\n"
final_report += "  Negative Selection Rate: " + str(nsr) + "\n"
final_report += "  Training error: " + str(np.mean(ERR_train)) + "(" + str(np.std(ERR_train)) + ")\n"
final_report += "  Testing error: " + str(np.mean(ERR_test)) + "(" + str(np.std(ERR_test)) + ")\n"

Report = final_report + Report
report_file = open(DIR_res+"DFS_nonlinear.txt", "a")
report_file.write(Report)
report_file.close()
