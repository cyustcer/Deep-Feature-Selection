library(glmnet) # Load the package for Elastic Net method
source("../src/utils.R")
set.seed(1)

SUPPs_ELASTIC = list() # to save support from best RF model 
TRUEs_ELASTIC = list() # to save support from underlying model
ERR_train_ELASTIC = c() # to save the training MSE using the best RF model
ERR_test_ELASTIC = c() # to save the testing MSE using the best RF model


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
  
  LAMBDAs = exp(seq(log(0.001), log(10), length.out=100))
  ALPHAs = seq(0., 0.5, length.out=20)
  EBICs_elastic = c()
  for (alpha in ALPHAs) {
    elastic = glmnet(as.matrix(X_train), as.matrix(y_train), 
                     alpha=alpha, lambda=LAMBDAs, seed=1)
    
    Ss = colSums(elastic$beta!=0)
    Y_Fits = predict(elastic, X_train)
    Y_Preds = predict(elastic, X_test)
    EBICs = EBICseq(Y_Fits, y_train, Ss, N)
    best_idx = which.min(EBICs)
    EBICs_elastic = c(EBICs_elastic, min(EBICs))
    if (min(EBICs) == min(EBICs_elastic)) {
      supp_elastic = c(1:1000)[elastic$beta[, best_idx]!=0]
      train_mse_elastic = mean((Y_Fits[, best_idx]-y_train)^2)
      test_mse_elastic = mean((Y_Preds[, best_idx]-y_test)^2)
    }
  }
  
  SUPPs_ELASTIC[[k]] = supp_elastic
  TRUEs_ELASTIC[[k]] = supp
  ERR_train_ELASTIC = c(ERR_train_ELASTIC, train_mse_elastic)
  ERR_test_ELASTIC = c(ERR_test_ELASTIC, test_mse_elastic)
}

### Metrics for SCAD models
measurements = measure(TRUEs_ELASTIC, SUPPs_ELASTIC)
lens = c()
for (k in 1:100) {
  lens = c(lens, length(SUPPs_ELASTIC[[k]]))
}

# False Selection Rate
(fsr_ELASTIC = measurements$fsr)
# Negative Selection Rate
(nsr_ELASTIC = measurements$nsr)
# Average length for selected supports
(len_ELASTIC = c(mean(lens), sd(lens)))
# Training MSE
(train_ELASTIC = c(mean(ERR_train_ELASTIC), sd(ERR_train_ELASTIC)))
# Testing MSE
(test_ELASTIC = c(mean(ERR_test_ELASTIC), sd(ERR_test_ELASTIC)))

fileConn <- file("../outputs/reports/elastic.txt")
content <- c("Results of Elastic Net",
             "The false selection rate is:",
             toString(fsr_ELASTIC),
             "The negative selection rate is:",
             toString(nsr_ELASTIC),
             "The average s is:",
             toString(len_ELASTIC),
             "The average training mse is:",
             toString(train_ELASTIC),
             "The average test mse is:",
             toString(test_ELASTIC))

writeLines(content, fileConn)
close(fileConn)
