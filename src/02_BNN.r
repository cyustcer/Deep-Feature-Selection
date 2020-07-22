library(BNN)
source("../src/utils.R")
set.seed(1)

k = as.integer(commandArgs(TRUE))
X <- read.table(paste('../data/nonlinear/p_500_N_600_s_4/X_', toString(k-1), '.txt', sep=""))
y <- read.table(paste('../data/nonlinear/p_500_N_600_s_4/y_', toString(k-1), '.txt', sep=""))
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
X_data = rbind(X_train, X_test)
y_data = as.factor(c(y_train, y_test))
lbds = c(0.025, 0.05, 0.1, 0.2, 0.3)
for (lbd in lbds) {
    bnn = BNNsel(X_data, y_data, train_num=300, hid_num=3, lambda=lbd,
                 total_iteration=1000000, popN=20, nCPUs=10)
    save(bnn, file=paste("Results/BNN/bnn_", toString(k-1), "_", toString(lbd), ".RData", sep=""))
}
