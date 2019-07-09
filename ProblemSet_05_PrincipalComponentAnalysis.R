#####################################################################
###   ProblemSet 05: Principal Component Analysis
#####################################################################

### Preface
# clearing workspace and set wd
rm(list=ls())
cat("\014")
dev.off()

#library(plm)       # Panel Methods - regression models
library(mvtnorm)   # random draws from a multivariate normal distr.
#library(MASS)       # to fit LDA and QDA analysis
library(glmnet)     # Lasso and Ridge regression
# Note: packages above require the data to be saved as a data frame

# data config
N = 300
p = 2
mu = 0

## DGP
# Variance-Covariance Matrix, beta vector
set.seed(123)
temp=abs(rnorm(p, 0, 1^0.5))
covmat=diag(temp)
beta=runif(p, min = 0.1, max = 0.5)

# sampling data and generate y's
X = rmvnorm(N, mean = rep(mu, p), sigma = covmat)
eps = rnorm(N, 0, 1^0.5)
Y = X %*% beta + eps
rm(temp, eps)

#####################################################################
###   Exercise 1

## a)
### how to implement the eigenvector calculation for PCA
# 1. calculate cov(x), x is demeaned
# 2. calculate Eigenvectors of cov(x)=phi
# 3. transformed Z=t(phi) times t(x)
###

X_m  = (X-colMeans(X))
covx = cov(X_m)
phi  = eigen(covx)
ev   = phi$vectors
ev1  = phi$vectors[,1]    # first eigenvector for PCR-comp 1
ev2  = phi$vectors[,2]    # second eigenvector for PCR-comp 2


plot(X[,1],X[,2])
par(new=TRUE)
plot(ev1, type="l", col="red", axes=FALSE, ylab="", xlab="")
par(new=TRUE)
plot(ev2, type="l", col="blue", axes=FALSE, ylab="", xlab="")
# Why are these eigenvectors not orthogonal?
Z   = X%*%ev
Z1  = Z[,1]
Z2  = Z[,2]

# PCA package in R
pca = prcomp(X)$rotation
ZT  = X%*%pca

# compare eigenvector vs package
plot(Z,ZT, xlab="eigenvector method", ylab="PCR Package")
error = sum(abs(ZT-Z)>0.01)
print(error)


## b)
# draw test data
X_test= rmvnorm(N, mean = rep(mu, p), sigma = covmat)
eps_test= eps=rnorm(N,0,1)
Y_test= X_test %*% beta + eps_test
X_test_d=data.frame(X_test) #predict functions need dataframe, but the one
Y_test_d=data.frame(Y_test) #for ridge does not

# perform PCR on Z1 and then Z1+Z2
PCR1=lm(Y ~ Z1)
PCR1_predict=data.frame(predict(PCR1, newdata = X_test_d))
Test_error_PCR1=(1/N)*sum((Y_test-PCR1_predict)^2)

PCR2=lm(Y ~ Z1 + Z2)
PCR2_predict=data.frame(predict(PCR2, newdata = X_test_d))
Test_error_PCR2=(1/N)*sum((Y_test-PCR2_predict)^2)

#Ridge
library(glmnet)
grid = 10^seq(10,-2, length =100)
ridge.mod = glmnet(X,Y,alpha=0, lambda=grid)

set.seed(1)
cv.out=cv.glmnet(X,Y,alpha=0)
plot(cv.out)
bestlam =cv.out$lambda.min
ridge.pred=predict(ridge.mod ,s=bestlam ,newx=X_test)
Test_error_ridge=(1/N)*sum((ridge.pred - Y_test)^2)

#OLS
OLS=lm(Y ~ X)
OLS_predict=data.frame(predict(OLS, newdata = X_test_d))
Test_error_OLS=(1/N)*sum((Y_test-OLS_predict)^2)

paste("OLS Test-error:", format(Test_error_OLS, digit=4))
paste("PCR one direction Test-error:", format(Test_error_PCR1, digit=4))
paste("PCR both directions / true model Test-error:", format(Test_error_PCR2, digit=4))
paste("Ridge Test-error:", format(Test_error_ridge, digit=4))
paste("Ridge regression performs the best!")

#####################################################################
###   Exercise 2 - Discussion and Simulation Study

# first set p=50
# 1. DGP most suited for PCA
#    -> zentralized data (standard normal N(0,1) )
# 2. DGP most suited for Ridge
#    -> beta is not equal to zero
# 3. DGP most suited for OLS (BLUE - estimation)
#    -> linear model with - uncorrelated, homoskedastic epsilons with mean zero
#    -> first we need regressors which is driving our DGP with high variances
#    -> the higher N the more precise is our estimation

### Simulation Study to check different methods given a certain DGP process
# clear space
# clearing workspace and set wd
rm(list=ls())
cat("\014")
dev.off()
library(mvtnorm)   # random draws from a multivariate normal distr.
library(glmnet)     # Lasso and Ridge regression

######################################################################
n=100
p=50
mu=0
beta=runif(p, min=0.01, max=0.99)

MCN=100
MSE=matrix(NaN,4,MCN)

# Note: the to check different specifications of the DGP check out
#       the "covmat" comments in the MCN Loop
######################################################################


for (i in 1:MCN){

  #generate training set

  ######################################################################
  beta=runif(p, min=0.1, max=0.5)
  beta[2:p]=0
  #a=runif(1,min=-0.0,max=0.0)
  #covmat=matrix(c(1,a,a,2),2,2)
  temp=runif(p, min = 0.5, max = 1.5)
  covmat=diag(temp)
  #covmat[1:p,1:p]=1
  #covmat=matrix(runif(p*p,min=0.5,max=0.9),p)
  #covmat=forceSymmetric(covmat)
  #covmat=matrix(covmat@x,p)
  #covmat[1,1]=10

  ######################################################################
  eps=rnorm(n,0,1)
  X = rmvnorm(n, mean = rep(mu, p), sigma = covmat)
  Y = X %*% beta + eps

  #generate test set
  X_test= rmvnorm(n, mean = rep(mu, p), sigma = covmat)
  eps_test= eps=rnorm(n,0,1)
  Y_test= X_test %*% beta + eps_test
  X_test_d=data.frame(X_test)
  Y_test_d=data.frame(Y_test)

  #calculate Z using PCA
  q=prcomp(X)$rotation
  Z=X%*%q
  Z1=Z[,1]
  Z2=Z[,2]

  #do PCR
  PCR1=lm(Y ~ Z1)
  PCR1_predict=data.frame(predict(PCR1, newdata = X_test_d))
  Test_error_PCR1=(1/n)*sum((Y_test-PCR1_predict)^2)

  PCR2=lm(Y ~ Z1 + Z2)
  PCR2_predict=data.frame(predict(PCR2, newdata = X_test_d))
  Test_error_PCR2=(1/n)*sum((Y_test-PCR2_predict)^2)

  #Ridge
  ridge.mod = glmnet(X,Y,alpha=0)
  cv.out=cv.glmnet(X,Y,alpha=0)
  bestlam =cv.out$lambda.min
  ridge.pred=predict(ridge.mod ,s=bestlam ,newx=X_test)
  Test_error_ridge=(1/n)*sum((ridge.pred - Y_test)^2)

  #OLS
  OLS=lm(Y ~ X)
  OLS_predict=data.frame(predict(OLS, newdata = X_test_d))
  Test_error_OLS=(1/n)*sum((Y_test-OLS_predict)^2)

  #save results for each run
  MSE[1,i]=Test_error_OLS
  MSE[2,i]=Test_error_PCR1
  MSE[3,i]=Test_error_PCR2
  MSE[4,i]=Test_error_ridge

}
### calculating Training Errors
MSE_ave=matrix(NaN,4,MCN)
for (l in 1:MCN){
  MSE_ave[1,l]=mean(MSE[1,1:(l)])
  MSE_ave[2,l]=mean(MSE[2,1:(l)])
  MSE_ave[3,l]=mean(MSE[3,1:(l)])
  MSE_ave[4,l]=mean(MSE[4,1:(l)])
}
### plotting results from simulation study
plot(MSE_ave[2,],type="l",col="green",ylim=c(min(MSE_ave),max(MSE_ave)),xlim=c(0,100),axis=FALSE,
     main="PCR 1, PCR 2, Ridge",ylab="average MSE", xlab="number of runs")
par(new=TRUE)
plot(MSE_ave[3,],type="l",col="blue",ylim=c(min(MSE_ave),max(MSE_ave)),xlim=c(0,100),axis=FALSE,
     ylab="average MSE", xlab="number of runs")
par(new=TRUE)
plot(MSE_ave[4,],type="l",col="red",ylim=c(min(MSE_ave),max(MSE_ave)),xlim=c(0,100),axis=FALSE,
     ylab="average MSE", xlab="number of runs")
par(new=TRUE)
plot(MSE_ave[1,],type="l",col="black",ylim=c(min(MSE_ave),max(MSE_ave)),xlim=c(0,100),axis=FALSE,
     ylab="average MSE", xlab="number of runs")
legend(40, max(MSE_ave), legend=c("PCR 1", "PCR 2","Ridge", "OLS"),
       col=c("green", "blue","red","black"), lty=1, cex=0.8)
