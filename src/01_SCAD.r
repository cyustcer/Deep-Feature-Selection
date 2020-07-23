library(ncvreg) # Load the package for SCAD method
source("../src/utils.R")
set.seed(1)

SUPPs_SCAD = list() # to save support from best RF model 
TRUEs_SCAD = list() # to save support from underlying model
ERR_train_SCAD = c() # to save the training MSE using the best RF model
ERR_test_SCAD = c() # to save the testing MSE using the best RF model


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
    
    LAMBDAs = exp(seq(log(1), log(0.01), length.out=100))
    scad = ncvreg(X_train, y_train, penalty="SCAD", 
                  lambda = LAMBDAs, seed=1)
    
    Ss = predict(scad, X_train, type="nvars")
    Y_Fits = predict(scad, X_train)
    Y_Preds = predict(scad, X_test)
    EBICs = EBICseq(Y_Fits, y_train, Ss, N)
    best_idx = which.min(EBICs)
    
    supp_scad = c(1:1000)[scad$beta[, best_idx]!=0]
    train_mse_scad = mean((Y_Fits[, best_idx]-y_train)^2)
    test_mse_scad = mean((Y_Preds[, best_idx]-y_test)^2)
    
    SUPPs_SCAD[[k]] = supp_scad
    TRUEs_SCAD[[k]] = supp
    ERR_train_SCAD = c(ERR_train_SCAD, train_mse_scad)
    ERR_test_SCAD = c(ERR_test_SCAD, test_mse_scad)
}

### Metrics for SCAD models
measurements = measure(TRUEs_SCAD, SUPPs_SCAD)
lens = c()
for (k in 1:100) {
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

fileConn <- file("../outputs/reports/scad_linear.txt")
content <- c("Results of linear SCAD",
             "The false selection rate is:",
             toString(fsr_SCAD),
             "The negative selection rate is:",
             toString(nsr_SCAD),
             "The average s is:",
             toString(len_SCAD),
             "The average training mse is:",
             toString(train_SCAD),
             "The average test mse is:",
             toString(test_SCAD))

writeLines(content, fileConn)
close(fileConn)
