library(ncvreg)
source("../src/utils.R")
set.seed(1)

SUPPs_SCAD = list() # to save support from best SCAD model 
TRUEs_SCAD = list() # to save support from underlying model
ERR_train_SCAD = c() # to save the training MSE using the best SCAD model
ERR_test_SCAD = c() # to save the testing MSE using the best SCAD model

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
  N = dim(X_train)[1]
  
  LAMBDAs = exp(seq(log(0.15), log(0.022), length.out=500))
  scad = ncvreg(as.matrix(X_train), y_train, family="binomial", penalty="SCAD", 
                lambda = LAMBDAs, seed=1)
  Ss = predict(scad, as.matrix(X_train), type="nvars")
  Y_Fits = predict(scad, as.matrix(X_train), type="response")
  LOSSes = apply(Y_Fits, 2, cross_entropy, y_true=y_train)
  BICs = BIC(LOSSes, Ss, N)
  best_idx = which.min(BICs)
  supp_scad = c(1:500)[scad$beta[2:501, best_idx] != 0]
  SUPPs_SCAD[[k]] = supp_scad
  TRUEs_SCAD[[k]] = c(1, 2, 3, 4)
  fit_scad = predict(scad, as.matrix(X_train), type="class")[, best_idx]
  pred_scad = predict(scad, as.matrix(X_test), type="class")[, best_idx]
  ERR_train_SCAD = c(ERR_train_SCAD, 1-sum(fit_scad==y_train)/300.)
  ERR_test_SCAD = c(ERR_test_SCAD, 1-sum(pred_scad==y_test)/300.)
}


measurements = measure(TRUEs_SCAD, SUPPs_SCAD)
lens = c()
for (k in 1:30) {
  lens = c(lens, length(SUPPs_SCAD[[k]]))
}

# False Selection Rate
(fsr_SCAD = measurements$fsr)
# Negative Selection Rate
(nsr_SCAD = measurements$nsr)
# Average length for selected supports
(len_SCAD = c(mean(lens), sd(lens)))
# Training MSE
(train_SCAD = c(mean(ERR_train_SCAD), sd(ERR_train_SCAD)))
# Testing MSE
(test_SCAD = c(mean(ERR_test_SCAD), sd(ERR_test_SCAD)))

fileConn <- file("../outputs/reports/scad_nonlinear.txt")
content <- c("Results of Nonlinear SCAD",
             "The false selection rate is:",
             toString(fsr_SCAD),
             "The negative selection rate is:",
             toString(nsr_SCAD),
             "The average s is:",
             toString(len_SCAD),
             "The average training error is:",
             toString(train_SCAD),
             "The average test error is:",
             toString(test_SCAD))

writeLines(content, fileConn)
close(fileConn)
