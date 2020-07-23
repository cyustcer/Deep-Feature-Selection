### Load data with index of datasets, from 0 to 99
read_data <- function(k,
                      dir = 'data/linear/p_1000_N_1000_s_100/') {
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

cross_entropy <- function(y_pred, y_true) {
  
  y_pred = (y_pred>=1)*0.9999 + (y_pred<1&y_pred>0)*y_pred + (y_pred<0)*0.0001
  loss = -mean((y_true*log(y_pred))+(1-y_true)*log(1-y_pred))
  return(loss)
}

BIC <- function(loss, s, n) {
  bic = 2*n*loss + s*log(n)
  return(bic)
}

EBIC <- function(fit, true, s, n, c=100) {
  sigma2 = mean((fit - true)^2)
  ebic = n*log(sigma2) + c*s*log(n)
  return(ebic)
}

EBICseq <- function(y_fits, true, Ss, n, c=3) {
  EBICs = c()
  for (i in 1:ncol(Y_Fits)) {
    ebic = n*log(mean((y_fits[, i]-true)^2)) + c*Ss[i]*log(n)
    EBICs = c(EBICs, ebic)
  }
  return(EBICs)
}

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

