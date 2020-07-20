####################################
########### Module Load ############
####################################
# Basic module import
import numpy as np
import math
import pandas as pd

# PYTORCH packages
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.autograd import grad
from torch.nn.parameter import Parameter

# Other methods
from sklearn import linear_model as lm

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
def optimal_s(k, optimal_c=1):
    #Ss = np.array([ 50, 70, 90, 91, 92,  93,  94,  95,  96,  97,  98,  99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 120, 140, 160, 180, 200]) # Candidates s
    Ss = np.arange(90, 115)
    BIC = [] # list to store bic values for candidates s
    # Load Data
    dirc = "../data/linear/p_1000_N_1000_s_100/"
    X, Y, X_test, Y_test, supp = data_load_l(k)
    for i, s in enumerate(Ss):
        c = optimal_c # tuning parameter for lambda, i.e. the ratio of lambda_1 and lambda_2.
        # Training dataset k with given s
        model, supp, bics, [err_train, err_test], _, _, _ = training_l(X, Y, X_test, Y_test, c, s, epochs=15, C=3.)
        # Store bic values
        BIC.append(bics[1])
        # if current bic is the smallest, save the trained model, support and other metric
        if bics[1] == min(BIC):
            best_model = model
            best_supp = supp
            best_err_train, best_err_test = err_train, err_test # one step model training and testing error
    
    idx = np.argmin(BIC) # index for best s
    # Generate automatic for best s of dataset k
    report = "For datasets "+str(k)+":\n"
    report += "    Optimal s: "+str(Ss[idx])+"\n"
    return Ss[idx], BIC, best_model, best_supp, report, [best_err_train, best_err_test]

##################################
### Multiple datasets training ###
##################################
Report = "" # Text report for linear regression methods

# List to save results for different datasets
BICs = [] # BICs over different s for each datasets, for the purpose of plotting 
SUPPs = [] # best support selected for each datasets
TRUEs = [] # true support for each datasets
ERR_train_1 = [] # one-step training error
ERR_test_1 = [] # one-step testing error
ERR_train = [] # two-step training error
ERR_test = [] # two-step testing error

### Training Over K datasets (K is set to 10 for shorter training time)
K = 10
for k in range(K):
    X, Y, X_test, Y_test, supp_true = data_load(k) # data load for two-step mse calculation
    s, BIC, model, supp, report, errs = optimal_s(k) # single datasets training
    mse_train = mse(model, X, Y) # two-step training mse
    mse_test = mse(model, X_test, Y_test) # two-step testing mse
    ### automatic text report
    Report += report
    Report += "    Training MSE: "+str(mse_train)+"\n"
    Report += "    Testing MSE: "+str(mse_test)+"\n"
    # Results saving
    BICs.append(BIC)
    SUPPs.append(supp)
    TRUEs.append(supp_true)
    ERR_train_1.append(errs[0])
    ERR_test_1.append(errs[1])
    ERR_train.append(mse_train)
    ERR_test.append(mse_test)

############################################
### Metric calculation and result saving ###
############################################

BICs = np.array(BICs)
DIR_res = "./output/"
np.savetxt(DIR_res+"linear_BICs.txt", BICs)
np.savetxt(DIR_res+"linear_train_1step.txt", np.array(ERR_train_1))
np.savetxt(DIR_res+"linear_test_1step.txt", np.array(ERR_test_1))
np.savetxt(DIR_res+"linear_train_2step.txt", np.array(ERR_train))
np.savetxt(DIR_res+"linear_test_2step.txt", np.array(ERR_test))
supp_file = open(DIR_res+"linear_supp.txt", "a")
for supp in SUPPs:
    supp_file.write(str(supp))
    supp_file.write("\n")
supp_file.close()

fsr, nsr = measure(TRUEs, SUPPs)
final_report = "For " + str(K) + " datasets:\n"
final_report += "  False Selection Rate: " + str(fsr) + "\n"
final_report += "  Negative Selection Rate: " + str(nsr) + "\n"
final_report += "  Training error: " + str(np.mean(ERR_train)) + "(" + str(np.std(ERR_train)) + ")\n"
final_report += "  Testing error: " + str(np.mean(ERR_test)) + "(" + str(np.std(ERR_test)) + ")\n"

Report = final_report + Report
report_file = open(DIR_res+"linear_report.txt", "a")
report_file.write(Report)
report_file.close()