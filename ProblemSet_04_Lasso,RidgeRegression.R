#####################################################################
###   ProblemSet 04: Lasso and Ridge Regression
#####################################################################

### Preface
# clearing workspace and set wd
rm(list=ls())
cat("\014")
dev.off()

#library(plm)       # Panel Methods - regression models
library(mvtnorm)   # random draws from a multivariate normal distr.
library(MASS)       # to fit LDA and QDA analysis
library(glmnet)     # Lasso and Ridge regression
# Note: packages above require the data to be saved as a data frame

#####################################################################
###   Exercise 1

# data config
N = 300
p = 50
mu = 0

## DGP
# Variance-Covariance Matrix, beta vector
set.seed(123)
temp=runif(p, min = 1, max = 2)
covmat=diag(temp)
beta=runif(p, min = 0.1, max = 0.5)

# sampling data and generate y's
# X = replicate(p , rmvnorm(n = N, mu = mu, Sigma = covmat))
X = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
eps = rnorm(N, 0, 1^0.5)
Y = X %*% beta + eps
#Y = sin(X %*% beta) + eps
rm(temp, eps)

## a)
# Ridge
grid <- seq(0.5, 0, by=-.001)
#grid <- 10^seq(p, -2, by = -.1)
ridge <- glmnet(X, Y, alpha = 0, lambda = grid)
summary(ridge)

# Lasso
# Note alpha=1 for lasso only and can blend with ridge penalty down to
# alpha=0 ridge only.
lasso <- glmnet(X, Y, alpha=1, lambda = grid)
summary(lasso)

## b)
# generating test data
set.seed(321)
X_test = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
eps = rnorm(N, 0, 1^0.5)
Y_test = X_test %*% beta + eps
#Y_test = sin(X_test %*% beta) + eps
rm(eps)

rfit <- predict(ridge, newx=X_test)
lfit <- predict(lasso, newx=X_test)

#lerr <- apply(lfit, 2, mean((Y_test-lfit)^2))
#rerr <- apply(rfit, 2, mean((Y_test-rfit)^2))
lerr <- c(rep(NaN, ncol(lfit)))
rerr <- c(rep(NaN, ncol(rfit)))
for (i in 1:ncol(lfit)){
  lerr[i] <- mean((Y_test-lfit[,i])^2)
  rerr[i] <- mean((Y_test-rfit[,i])^2)
  }

par(mfrow=c(3,2))
plot(lasso, main="Lasso")
plot(ridge, main="Ridge")
plot(lasso$lambda, main="Lasso Lambda")
plot(ridge$lambda, main="Ridge Lambda")
plot(lerr, main="Lasso MSE")
plot(rerr, main="Ridge MSE")


## c)
# set s = optimal lambda to optimize cross-validation
cvridge <- cv.glmnet(X, Y, alpha = 0, lambda = grid)
cvlasso <- cv.glmnet(X, Y, alpha=1, lambda = grid)

opt_rlambda <- cvridge$lambda.min
opt_llambda <- cvlasso$lambda.min
cvrfit <- predict(cvridge, s = opt_rlambda, newx = X_test)
cvlfit <- predict(cvlasso, s = opt_llambda, newx = X_test)

cvlerr <- mean((Y_test-cvlfit)^2)
cvrerr <- mean((Y_test-cvrfit)^2)

paste("CV - MSE Ridge Regression is:", format(cvrerr, digit=4))
paste("CV - MSE Lasso Regression is:", format(cvlerr, digit=4))
paste("Difference between Ridge and Lasso is:", format((cvrerr-cvlerr), digit=4))
# Note: since our optimal lambda is zero in the Lasso, we compute simply OLS

#####################################################################
###   Exercise 2 - Simulation Study

## a)
rm(list=ls())
cat("\014")

MCN=100
cvlerr <- c(rep(NaN, MCN))
cvrerr <- c(rep(NaN, MCN))
set.seed(123)

 for (i in 2:MCN){
   # data config
   N = 300
   p = i
   mu = 0

   ## DGP
   # Variance-Covariance Matrix, beta vector
   temp=runif(p, min = 1, max = 2)
   covmat=diag(temp)
   beta=runif(p, min = 0.1, max = 0.5)

   # sampling data and generate y's
   # X = replicate(p , rmvnorm(n = N, mu = mu, Sigma = covmat))
   X = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
   eps = rnorm(N, 0, 1^0.5)
   Y = X %*% beta + eps
   rm(temp, eps)
   cvridge <- cv.glmnet(X, Y, alpha = 0)
   cvlasso <- cv.glmnet(X, Y, alpha=1)

   X_test = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
   eps = rnorm(N, 0, 1^0.5)
   Y_test = X_test %*% beta + eps
   rm(eps)

   opt_rlambda <- cvridge$lambda.min
   opt_llambda <- cvlasso$lambda.min
   cvrfit <- predict(cvridge, s = opt_rlambda, newx = X_test)
   cvlfit <- predict(cvlasso, s = opt_llambda, newx = X_test)

   cvlerr[i] <- mean((Y_test-cvlfit)^2)
   cvrerr[i] <- mean((Y_test-cvrfit)^2)
 }

par(mfrow=c(1,2))
plot(cvlerr, type="l", main="Lasso Error dep. number of Regressors")
plot(cvrerr, type="l", main="Ridge Error dep. number of Regressors")

### Note: Since we increase the number of regressors in the true model, we loose many degrees of freedom.
#         Hence, our estimation becomes less precise!

## b) sparsity of beta - means that the higher the sparsity the more zeros are in our beta vector
rm(list=ls())
cat("\014")

MCN=100
cvlerr <- c(rep(NaN, MCN))
cvrerr <- c(rep(NaN, MCN))
set.seed(123)

for (i in 2:MCN){
  # data config
  N = 300
  p = 50
  mu = 0

  ## DGP
  # Variance-Covariance Matrix, beta vector
  temp=runif(p, min = 1, max = 2)
  covmat=diag(temp)
  beta=runif(p, min = 0.1, max = 0.5)
  prob=i/(i^2)
  binvec=rbinom(p,1,prob)
  beta=beta*binvec

  # sampling data and generate y's
  # X = replicate(p , rmvnorm(n = N, mu = mu, Sigma = covmat))
  X = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
  eps = rnorm(N, 0, 1^0.5)
  Y = X %*% beta + eps
  rm(temp, eps)
  cvridge <- cv.glmnet(X, Y, alpha = 0)
  cvlasso <- cv.glmnet(X, Y, alpha=1)

  X_test = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
  eps = rnorm(N, 0, 1^0.5)
  Y_test = X_test %*% beta + eps
  rm(eps)

  opt_rlambda <- cvridge$lambda.min
  opt_llambda <- cvlasso$lambda.min
  cvrfit <- predict(cvridge, s = opt_rlambda, newx = X_test)
  cvlfit <- predict(cvlasso, s = opt_llambda, newx = X_test)

  cvlerr[i] <- mean((Y_test-cvlfit)^2)
  cvrerr[i] <- mean((Y_test-cvrfit)^2)
}

par(mfrow=c(1,2))
plot(cvlerr, type="l", main="Lasso Error dep. sparsity of Regressors")
plot(cvrerr, type="l", main="Ridge Error dep. sparsity of Regressors")

### Note: since we increase the sparsity of beta, our true model contains more coefficients which are truely zero.
#         Therefore Lasso performs better, since lasso allows estimates to be zero!
