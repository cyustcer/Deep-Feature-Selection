library(glmnet) # Load the package for LASSO method
source("../src/utils.R")
set.seed(1)

SUPPs_LASSO = list() # to save support from best RF model 
TRUEs_LASSO = list() # to save support from underlying model
ERR_train_LASSO = c() # to save the training MSE using the best RF model
ERR_test_LASSO = c() # to save the testing MSE using the best RF model


for (k in 1:100) {
  dirc = "../data/linear/p_1000_N_1000_s_100/"
  X <- read.table(paste(dirc, 'X_', toString(k-1), '.txt', sep=""))
  y <- read.table(paste(dirc, 'y_', toString(k-1), '.txt', sep=""))
  beta <- read.table(paste(dirc, "beta_", toString(k-1), '.txt', sep=""))
  supp = which(beta!=0)
  X_train = as.matrix(X[1:500,])
  y_train = y[1:500,]
  X_test = as.matrix(X[501:1000,])
  y_test = y[501:1000,]
  N = dim(X_train)[1]
  p = dim(X_train)[2]
  
  LAMBDAs = exp(seq(log(0.05), log(5), length.out=100))
  lasso = glmnet(as.matrix(X_train), as.matrix(y_train), 
                 lambda=LAMBDAs, alpha=1, seed=1)
  
  Ss = colSums(lasso$beta!=0)
  Y_Fits = predict(lasso, X_train)
  Y_Preds = predict(lasso, X_test)
  EBICs = EBICseq(Y_Fits, y_train, Ss, N)
  best_idx = which.min(EBICs)
  
  supp_lasso = c(1:1000)[lasso$beta[, best_idx]!=0]
  train_mse_lasso = mean((Y_Fits[, best_idx]-y_train)^2)
  test_mse_lasso = mean((Y_Preds[, best_idx]-y_test)^2)
  
  SUPPs_LASSO[[k]] = supp_lasso
  TRUEs_LASSO[[k]] = supp
  ERR_train_LASSO = c(ERR_train_LASSO, train_mse_lasso)
  ERR_test_LASSO = c(ERR_test_LASSO, test_mse_lasso)
}

### Metrics for SCAD models
measurements = measure(TRUEs_LASSO, SUPPs_LASSO)
lens = c()
for (k in 1:100) {
  lens = c(lens, length(SUPPs_LASSO[[k]]))
}

# False Selection Rate
(fsr_LASSO = measurements$fsr)
# Negative Selection Rate
(nsr_LASSO = measurements$nsr)
# Average length for selected supports
(len_LASSO = c(mean(lens), sd(lens)))
# Training MSE
(train_LASSO = c(mean(ERR_train_LASSO), sd(ERR_train_LASSO)))
# Testing MSE
(test_LASSO = c(mean(ERR_test_LASSO), sd(ERR_test_LASSO)))

fileConn <- file("../outputs/reports/lasso.txt")
content <- c("Results of LASSO",
             "The false selection rate is:",
             toString(fsr_LASSO),
             "The negative selection rate is:",
             toString(nsr_LASSO),
             "The average s is:",
             toString(len_LASSO),
             "The average training mse is:",
             toString(train_LASSO),
             "The average test mse is:",
             toString(test_LASSO))

writeLines(content, fileConn)
close(fileConn)
