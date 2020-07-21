# Bayesian Additive Regression Trees
options(java.parameters="-Xmx5g")
library(bartMachine)
set_bart_machine_num_cores(4)
source("../src/utils.R")

SUPPs_BART = list() # to save support from best BART model 
TRUEs_BART = list() # to save support from underlying model
ERR_train_BART = c() # to save the training MSE using the best BART model
ERR_test_BART = c() # to save the testing MSE using the best BART model

for (k in 1:30) {
	data_dir = "../data/nonlinear/p_500_N_600_s_4/"
	X = read.table(paste(data_dir, 'X_', toString(k-1), '.txt', sep=""))
	y = read.table(paste(data_dir, 'y_', toString(k-1), '.txt', sep=""))
	train_pos_idx = which(y == 1)[1:150]
	train_neg_idx = which(y == 0)[1:150]
	test_pos_idx = which(y == 1)[151:300]
	test_neg_idx = which(y == 0)[151:300]
	train_idx = sort(cbind(train_pos_idx, train_neg_idx))
	test_idx = sort(cbind(test_pos_idx, test_neg_idx))
	X_train = X[train_idx,]
	y_train = y[train_idx,]
	X_test = X[test_idx,]
	y_test = y[test_idx,]

	bart_vs = bartMachine(X=X_train, y=y_train, num_trees=75)
	var_sel = var_selection_by_permute_cv(bart_vs)
    supp_name = var_sel$importance_vars_cv
    for(i in 1:length(supp_name)) {
        supp_x = c(supp_x, strtoi(substr(supp_name[i], 2, nchar(supp_name[i]))))
    }
    SUPPs_BART[[k]] = supp_x
    TRUEs_BART[[k]] = c(1, 2, 3, 4)
    
    bart = bartMachine(X=X_train, y=factor(y_train), num_trees=75)
    fit_bart = predict(bart, X_train, type="class")
    pred_bart = predict(bart, X_test, type="class")
    ERR_train_BART = c(ERR_train_BART,
                       1 - (sum(fit_bart == y_train)/300.))
    ERR_test_BART = c(ERR_test_BART,
                      1 - (sum(pred_bart == y_test)/300.))
}


measurements = measure(TRUEs_BART, SUPPs_BART)
lens = c()
for (k in 1:30) {
    lens = c(lens, length(SUPPs_BART[[k]]))
}

# False Selection Rate
(fsr_BART = measurements$fsr)
# Negative Selection Rate
(nsr_BART = measurements$nsr)
# Average length for selected supports
(len_BART = c(mean(lens), sd(lens)))
# Training MSE
(train_BART = c(mean(ERR_train_BART), sd(ERR_train_BART)))
# Testing MSE
(test_BART = c(mean(ERR_test_BART), sd(ERR_test_BART)))

