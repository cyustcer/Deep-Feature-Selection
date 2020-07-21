library(h2o)
source("../src/utils.R")

SUPPs_RF = list() # to save support from best RF model 
TRUEs_RF = list() # to save support from underlying model
ERR_train_RF = c() # to save the training MSE using the best RF model
ERR_test_RF = c() # to save the testing MSE using the best RF model

h2o.init()
for(k in 1:30) {
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
    data_train = as.h2o(cbind(X_train, y_train))
    data_train["y_train"] = as.factor(data_train["y_train"])
    data_test = as.h2o(cbind(X_test, y_test))
    data_test["y_test"] = as.factor(data_test["y_test"])
    
    res = h2o.randomForest(y="y_train", training_frame=data_train)
    supp_x = c(1:500)[h2o.varimp(res)$percentage>0.0125]
    SUPPs_RF[[k]] = supp_x
    TRUEs_RF[[k]] = c(1:4)
    ERR_train_RF = c(ERR_train_RF, 
                     1-(sum((predict(res, as.h2o(X_train))$predict == as.h2o(y_train))/300.)))
    ERR_test_RF = c(ERR_test_RF, 
                    1-(sum((predict(res, as.h2o(X_test))$predict == as.h2o(y_test))/300.)))
}

measurements = measure(TRUEs_RF, SUPPs_RF)
lens = c()
for (k in 1:30) {
    lens = c(lens, length(SUPPs_RF[[k]]))
}

# False Selection Rate
(fsr_RF = measurements$fsr)
# Negative Selection Rate
(nsr_RF = measurements$nsr)
# Average length for selected supports
(len_RF = c(mean(lens), sd(lens)))
# Training MSE
(train_RF = c(mean(ERR_train_RF), sd(ERR_train_RF)))
# Testing MSE
(test_RF = c(mean(ERR_test_RF), sd(ERR_test_RF)))
