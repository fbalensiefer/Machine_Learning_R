#####################################################################
###   ProblemSet - 1
#####################################################################

### Preface
# clearing workspace and set wd
rm(list=ls())
cat("\014")
dev.off()
#library(plm)

#####################################################################
###   Exercise 1

N=1000
sig_x= 1.5
mu_x=  0
sig_u= 10

beta1= 5
beta2= -0.5
x_1= c(rep(1,N))      # is a constant
set.seed(123)         # to generate always the same random numbers

#a)
x_2_train=rnorm(N,mu_x,sig_x^0.5)
res_train=rnorm(N,0,sig_u^0.5)
y_train=beta1*x_1+beta2*x_2_train+res_train

#b)
x_2_test=rnorm(N,mu_x,sig_x^0.5)
res_test=rnorm(N,0,sig_u^0.5)
y_test=beta1*x_1+beta2*x_2_test+res_test

#c)
model=lm(y_train ~ x_2_train)
summary(model)
beta1_hat=unname(model$coefficients[1])
beta2_hat=unname(model$coefficients[2])



#d)
yfit_train=predict(model)
MSE=1/N*sum((y_train-yfit_train)^2)

yfit_test=beta1_hat+beta2_hat*x_2_test
AVE=1/N*sum((y_test-yfit_test)^2)     # Ave=MSE but only for compairing training and test sample


# e)
#defining additional X_2T values with increasing polynomial
x_2_train.poly=matrix(NaN,N,5) #empty matrix to be filled with polynomials of x_2T
for (i in 0:4){
  x_2_train.poly[,i+1]=x_2_train^i #column 1-5 filled with x_2T to the power of 0-4
}

#defining coefficients for each regression on additional polynomial of x_2T
OLS.poly=matrix(0,5,5) #empty matrix to be filled with coefficients of regressions
yfit_train.poly=matrix(NaN,N,5) #empty matrix to be filled with fitted values of regressions
for (i in 0:4){
  OLS.poly[1:(i+1),(i+1)]=lm(y_train~x_2_train.poly[,1:(i+1)]-1)$coefficients
  yfit_train.poly[,(i+1)]=lm(y_train~x_2_train.poly[,1:(i+1)]-1)$fitted.values
}

#calculating MSEs with increasing polynomial
MSE.poly=numeric()
for (i in 1:5){
  MSE.poly[i]=(1/N)*sum((y_train-yfit_train.poly[,i])^2)
}
MSE.poly

#calculating predited values for test sample
yfit_test.poly=matrix(NaN,N,5)
for (i in 1:5){
  yfit_test.poly[,i]=OLS.poly[1,i]*x_1+OLS.poly[2,i]*x_2_test+OLS.poly[3,i]*(x_2_test^2)+
    OLS.poly[4,i]*(x_2_test^3)+OLS.poly[5,i]*(x_2_test^4)
}


#calculating AVEs with increasing polynomial
AVE.poly=numeric()
for (i in 1:5){
  AVE.poly[i]=(1/N)*sum((y_test-yfit_test.poly[,i])^2)
}
AVE.poly

par(mfrow=c(1,2))
plot(MSE.poly, type='l', col="red")
plot(AVE.poly, type='l', col="red")

### Note: Our average prediction error (AVE) increases after adding the
#         thrid polynominal (x^3) - Reason: the true DGP is
#         y = c + b*x + res

#####################################################################
###   Exercise 2

rm(list=ls())
#cat("\014")

N=1000
sig_x= 1.5
mu_x=  0
sig_u= 10

beta1= 5
beta2= -0.5
x_1= c(rep(1,N))      # is a constant

# a) and b)
MCN=1000
MSE_Ex2=matrix(NaN,MCN,1)
AVE_Ex2=matrix(NaN,MCN,1)
set.seed(100)

### Note: For efficieny it is better to generatetest sample once
#         before the MC-Simulation is started.
#         Reason: training and test sample are drawn from a random distribution

# test Data
x_2_test=rnorm(N,mu_x,sig_x^0.5)
res_test=rnorm(N,0,sig_u^0.5)
y_test=beta1*x_1+beta2*x_2_test+res_test


for (i in 1:MCN){
  # training Data
  x_2_train=rnorm(N,mu_x,sig_x^0.5)
  res_train=rnorm(N,0,sig_u^0.5)
  y_train=beta1*x_1+beta2*x_2_train+res_train

  # setting up regression model
  model=lm(y_train ~ x_2_train)
  beta1_hat=unname(model$coefficients[1])
  beta2_hat=unname(model$coefficients[2])

  # calculation of MSE
  yfit_train=predict(model)
  MSE_Ex2[i]=1/N*sum((y_train-yfit_train)^2)

  # calculation of AVE
  yfit_test=beta1_hat+beta2_hat*x_2_test
  AVE_Ex2[i]=1/N*sum((y_test-yfit_test)^2)
}

# b)
avg_MSE=mean(MSE_Ex2)
avg_AVE=mean(AVE_Ex2)

# c)
par(mfrow=c(1,2))
plot(MSE_Ex2, type="l", col="blue")
abline(h=avg_MSE, col="black")
plot(AVE_Ex2, type="l", col="red")
abline(h=avg_AVE, col="black")

# d) Along which margins could you vary parameters of the initial simulation set-up and what would
#    be your intuition based on the theoretical properties of the considered objects of interest?
#
#    Idea:  the higher the number of iterations in our Monte-Carlo simulation, the more precise will be our estimation
#           of avg. MSE and avg. AVE to the true DGP value sigma_u
#    For example: MCN=10 vs. MCN=1000
