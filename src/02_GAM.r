library(gamsel)
source("../src/utils.R")
set.seed(1)

SUPPs_GAM = list() # to save support from best GAM model 
TRUEs_GAM = list() # to save support from underlying model
ERR_train_GAM = c() # to save the training error using the best GAM model
ERR_test_GAM = c() # to save the testing error using the best GAM model

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
  p = dim(X_train)[2]

	gam = gamsel(X_train, y_train, family="binomial")
  SUPPs = getActive(gam, c(1:50))
  Ss = as.numeric(lapply(SUPPs, length))
  Y_Preds = predict(gam, X_train, type="response")
  LOSSes = apply(Y_Preds, 2, cross_entropy, y_true=y_train)
  BICs = BIC(LOSSes, Ss, N)
  best_idx = which.min(BICs)
  supp_x = getActive(gam, index=c(best_idx), type="nonlinear")[[1]]
  train_err = 1 - (sum((Y_Preds[, best_idx]>=0.5)*1==y_train)/300.)
  Y_Preds_t = predict(gam, X_test, type="response")
  test_err = 1 - (sum((Y_Preds_t[, best_idx]>=0.5)*1==y_test)/300.)
  
  SUPPs_GAM[[k]] = supp_x
  TRUEs_GAM[[k]] = c(1:4)
  ERR_train_GAM = c(ERR_train_GAM, train_err)
  ERR_test_GAM = c(ERR_test_GAM, test_err)
}

measurements = measure(TRUEs_GAM, SUPPs_GAM)
lens = c()
for (k in 1:30) {
  lens = c(lens, length(SUPPs_GAM[[k]]))
}

# False Selection Rate
(fsr_GAM = measurements$fsr)
# Negative Selection Rate
(nsr_GAM = measurements$nsr)
# Average length for selected supports
(len_GAM = c(mean(lens), sd(lens)))
# Training MSE
(train_GAM = c(mean(ERR_train_GAM), sd(ERR_train_GAM)))
# Testing MSE
(test_GAM = c(mean(ERR_test_GAM), sd(ERR_test_GAM)))

fileConn <- file("../outputs/reports/gam.txt")
content <- c("Results of GAM",
             "The false selection rate is:",
             toString(fsr_GAM),
             "The negative selection rate is:",
             toString(nsr_GAM),
             "The average s is:",
             toString(len_GAM),
             "The average training error is:",
             toString(train_GAM),
             "The average test error is:",
             toString(test_GAM))

writeLines(content, fileConn)
close(fileConn)
