library(ncvreg) # Load the package for SCAD method
set.seed(1)

### Function to calculate the fsr and nsr
measure <- function(TRUEs, ESTs) {
    K = length(TRUEs)
    fs = 0
    fs_d = 0
    ns = 0
    ns_d = 0
    for (i in 1:K) {
        fs = fs + length(setdiff(ESTs[[i]], TRUEs[[i]]))
        fs_d = fs_d + length(ESTs[[i]])
        ns = ns + length(setdiff(TRUEs[[i]], ESTs[[i]]))
        ns_d = ns_d + length(TRUEs[[i]])
    }
    fsr = fs/fs_d
    nsr = ns/ns_d
    return(list('fsr'=fsr, 'nsr'=nsr))
}

### Load data with index of datasets, from 0 to 99
read_data <- function(k) {
    # Directory for the datasets
    dir = 'data/linear/p_1000_N_1000_s_100/'
    x = read.table(paste(dir, 'X_', k, '.txt', sep=''))
    y = read.table(paste(dir, 'y_', k, '.txt', sep=''))
    beta = read.table(paste(dir, 'beta_', k, '.txt', sep=''))
    supp = which(beta != 0) # True support
    # Take last 500 samples as testing set
    x_test = x[501:1000,]
    y_test = y[501:1000,]
    # Take first 500 samples as training set
    x = x[1:500,]
    y = y[1:500,]
    return(list('x'=x, 'y'=y, 'x_test'=x_test, 'y_test'=y_test, 'supp'=supp))
}

### Function for training model for dataset k with optimal lambda using BIC criteria, C is the constant for BIC criteria.
# SCAD model fails when p > 1300, when C=1.0
train_SCAD <- function(k, C=1.0) {
    data = read_data(k)
    N = dim(data$x)[1]
    p = dim(data$x)[2]
    LAMBDAs = seq(0.05, 0.5, 0.05) # Candidate lambdas
    BIC = c()
    SUPP = list()
    idx = 1
    # Train through all lambdas
    for (lambda in LAMBDAs) {
        scad = ncvreg(data$x, data$y, penalty='SCAD', lambda=lambda)
        supp_x = which(scad$beta[2:1001] != 0)
        s = length(supp_x)
        loss = mean((predict(scad, as.matrix(data$x))-data$y)^2)
        bic = N*log(loss) + C*s*log(N)
        BIC = c(BIC, bic)
        SUPP[[idx]] = supp_x
        idx = idx + 1
    }
    # Find the best lambda and train the model
    best_lambda = LAMBDAs[which.min(BIC)]
    best_supp = SUPP[[which.min(BIC)]]
    best_scad = ncvreg(data$x, data$y, penalty='SCAD', lambda=best_lambda)
    mse_train = mean((predict(scad, as.matrix(data$x)) - data$y)^2)
    mse_test = mean((predict(scad, as.matrix(data$x_test)) - data$y_test)^2)
    return(list('best_lambda'=best_lambda, 'best_supp'=best_supp, 'true_supp'=data$supp, 'mse_train'=mse_train, 'mse_test'=mse_test))
}

# list and sequence to store the results
SUPPs_SCAD = list() # to save support from best SCAD model 
TRUEs_SCAD = list() # to save support from underlying model
ERR_train_SCAD = c() # to save the training MSE using the best SCAD model
ERR_test_SCAD = c() # to save the testing MSE using the best SCAD model
LAMBDAs_SCAD = c() # to save the chosen lambda for the best SCAD model

### Training process for 100 datasets
for (k in 1:20) {
    res = train_SCAD(k-1, C=1.)
    SUPPs_SCAD[[k]] = res$best_supp
    TRUEs_SCAD[[k]] = res$true_supp
    ERR_train_SCAD = c(ERR_train_SCAD, res$mse_train)
    ERR_test_SCAD = c(ERR_test_SCAD, res$mse_test)
    LAMBDAs_SCAD = c(LAMBDAs_SCAD, res$best_lambda)
}

### Metrics for SCAD models
measurements = measure(TRUEs_SCAD, SUPPs_SCAD)
lens = c()
for (k in 1:20) {
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
