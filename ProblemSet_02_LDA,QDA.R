#####################################################################
###   ProblemSet 02: LDA and QDA
#####################################################################

### Preface
# clearing workspace and set wd
rm(list=ls())
cat("\014")
dev.off()

#library(plm)       # Panel Methods - regression models
#library(mvtnorm)   # random draws from a multivariate normal distr.
library(MASS)       # to fit LDA and QDA analysis
# Note: packages above require the data to be saved as a data frame

#####################################################################
###   Exercise 1

# class 1
n1 = 300
mu1= c(-3,3)
# class 2
n2 = 500
mu2= c(5,5)
# Variance-Covariance Matrix
covmat=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
#total number of observations
N=n1+n2

## a)
# DGP - for both classes
set.seed(123)
X1=mvrnorm(n = n1, mu = mu1, Sigma = covmat)
X2=mvrnorm(n = n2, mu = mu2, Sigma = covmat)
df1 <- data.frame(X1)
df1['class'] = 1
df2 <- data.frame(X2)
df2['class'] = 2
df  <- merge(df1,df2, all=TRUE)
rm(df1, df2, X1, X2)

## b)
mod_lda  =  lda(class ~ X1 + X2, data=df)
summary(mod_lda)
class_lfit  <- as.numeric(predict(mod_lda)$class)
mod_qda  =  qda(class ~ X1 + X2 , data=df)
summary(mod_qda)
class_qfit  <- as.numeric(predict(mod_qda)$class)

## c)
# Note: since classes are ordinal scale we can not use MSE, due to the fact
#       that the distance between class 1 and 3 are the same as between 1 and 2
#       furthermore it is not appropriarte to use OLS
MTE_LDA=sum(class_lfit!=df$class)/N
MTE_QDA=sum(class_qfit!=df$class)/N
print(MTE_LDA)
print(MTE_QDA)
print(MTE_LDA-MTE_QDA)

#####################################################################
###   Exercise 2 - Simulation Study

# a)
rm(list=ls())
cat("\014")

MCN=100
MSE=matrix(NaN,MCN,2)


n1 = 300
mu1= c(-3,3)
n2 = 500
mu2= c(5,5)
covmat=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
N=n1+n2

set.seed(123)

for (i in 1:MCN){
  X1=mvrnorm(n = n1, mu = mu1, Sigma = covmat)
  X2=mvrnorm(n = n2, mu = mu2, Sigma = covmat)
  df1 <- data.frame(X1)
  df1['class'] = 1
  df2 <- data.frame(X2)
  df2['class'] = 2
  df  <- merge(df1,df2, all=TRUE)

  mod_lda  =  lda(class ~ X1 + X2, data=df)
  class_lfit  <- as.numeric(predict(mod_lda)$class)
  mod_qda  =  qda(class ~ X1 + X2 , data=df)
  class_qfit  <- as.numeric(predict(mod_qda)$class)

  MSE[i,1]=sum(class_lfit!=df$class)/N
  MSE[i,2]=sum(class_qfit!=df$class)/N
}

avg_MSE_LDA=mean(MSE[,1])
avg_MSE_QDA=mean(MSE[,2])

#par(mfrow=c(1,2))
#plot(MSE[,1], ylab="LDA")
#abline(h=avg_MSE_LDA, col="red")
#plot(MSE[,2], ylab="QDA")
#abline(h=avg_MSE_QDA, col="red")

summary(avg_MSE_LDA-avg_MSE_QDA)

# b)
# Note: since we have different covariate matrixes sigma 1 and sigma 2
#       QDA is more precise than LDA
#       From a theoretical perspective:
#       * if LDAs assumption that the K classes share a common covariance matrix
#          is badly off, then LDA --> high bias
#       * LDA is a much less flexible classifier than QDA --> lower variance
# Hence: Try with different covariance matrices

rm(list=ls())
cat("\014")

MCN=100
MSE=matrix(NaN,MCN,2)


n1 = 300
mu1= c(-3,3)
n2 = 500
mu2= c(5,5)
covmat_1=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
covmat_2=matrix(c(10,-2,-2,5), nrow=2, ncol=2)
N=n1+n2

set.seed(123)

for (i in 1:MCN){
  X1=mvrnorm(n = n1, mu = mu1, Sigma = covmat_1)
  X2=mvrnorm(n = n2, mu = mu2, Sigma = covmat_2)
  df1 <- data.frame(X1)
  df1['class'] = 1
  df2 <- data.frame(X2)
  df2['class'] = 2
  df  <- merge(df1,df2, all=TRUE)

  mod_lda  =  lda(class ~ X1 + X2, data=df)
  class_lfit  <- as.numeric(predict(mod_lda)$class)
  mod_qda  =  qda(class ~ X1 + X2 , data=df)
  class_qfit  <- as.numeric(predict(mod_qda)$class)

  MSE[i,1]=sum(class_lfit!=df$class)/N
  MSE[i,2]=sum(class_qfit!=df$class)/N
}

avg_MSE_LDA=mean(MSE[,1])
avg_MSE_QDA=mean(MSE[,2])

#par(mfrow=c(1,2))
#plot(MSE[,1], ylab="LDA")
#abline(h=avg_MSE_LDA, col="red")
#plot(MSE[,2], ylab="QDA")
#abline(h=avg_MSE_QDA, col="red")

summary(avg_MSE_LDA-avg_MSE_QDA)
