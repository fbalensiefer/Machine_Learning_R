#####################################################################
###   ProblemSet 03: Cross Validation
#####################################################################

# Clear workspace
rm(list=ls())

# Load packages that will be needed
library(MASS)


###################### Exercise 1 ##############################
#Set-up:
n1 = 300
mu1= c(-3,3)
n2 = 500
mu2= c(5,5)
covmat=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
N=n1+n2

#Creating training data
set.seed(100)
X1=mvrnorm(n = n1, mu = mu1, Sigma = covmat)
X2=mvrnorm(n = n2, mu = mu2, Sigma = covmat)
df1 = data.frame(cbind(X1,rep(1,length(X1[,1]))))
df2 = data.frame(cbind(X2,rep(2,length(X2[,1]))))
df = merge(df1,df2, all=TRUE)
colnames(df)=c("X1","X2","class")
rm(df1, df2, X1, X2)

#Creating test data
set.seed(200)
X1_test=mvrnorm(n = n1, mu = mu1, Sigma = covmat)
X2_test=mvrnorm(n = n2, mu = mu2, Sigma = covmat)
df1_test = data.frame(cbind(X1_test,rep(1,length(X1_test[,1]))))
df2_test = data.frame(cbind(X2_test,rep(2,length(X2_test[,1]))))
df_test = merge(df1_test,df2_test, all=TRUE)
colnames(df_test)=c("X1","X2","class")
rm(df1_test, df2_test, X1_test, X2_test)

# LDA analysis
LDA = lda(class ~ X1 + X2, data=df)
class_lfit = as.numeric(predict(LDA)$class)
class_lfit_test = as.numeric(predict(LDA,df_test)$class)

# QDA analysis
QDA = qda(class ~ X1 + X2 , data=df)
class_qfit = as.numeric(predict(QDA)$class)
class_qfit_test = as.numeric(predict(QDA,df_test)$class)

#### a)

#Comparison LDA and QDA for training data
table(class_lfit,df[,3])
table(class_qfit,df[,3])
mean_error_lda=(1/N)*sum(class_lfit!=df[,3])
mean_error_qda=(1/N)*sum(class_qfit!=df[,3])

#Comparison LDA and QDA for test data
table(class_lfit_test,df_test[,3])
table(class_qfit_test,df_test[,3])
mean_error_lda_test=(1/N)*sum(class_lfit_test!=df_test[,3])
mean_error_qda_test=(1/N)*sum(class_qfit_test!=df_test[,3])

#### b)
rm(list=ls())

n1=500
n2=500
N=n1+n2
mu1=c(-7,8)
mu2=c(3,8)
covmat1=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
covmat2=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
c=0.5

#create function that uses defined variables to create data set and
#randomly splits it in samples with fraction of training data equal to c
validation_set = function(n1=n1,n2=n2,mu1=mu1,mu2=mu2,covmat1=covmat1,
                          covmat2=covmat2,c=c){
  #create data set
  X1=mvrnorm(n = n1, mu = mu1, Sigma = covmat1)
  X2=mvrnorm(n = n2, mu = mu2, Sigma = covmat2)
  df1 = data.frame(cbind(X1,rep(1,length(X1[,1]))))
  df2 = data.frame(cbind(X2,rep(2,length(X2[,1]))))
  df = merge(df1,df2, all=TRUE)
  colnames(df)=c("X1","X2","class")

  #randomly split data in fraction c and 1-c
  sample = sample.int(n=nrow(df), size=floor(c*nrow(df)), replace=F)
  training_sample = df[sample,]
  test_sample = df[-sample,]

  return(list(training_sample=training_sample,test_sample=test_sample))
}

validation_samples=validation_set(n1,n2,mu1,mu2,covmat1,covmat2,c)

#### c)
#define variables that should be used to create sample
n1=500
n2=500
N=n1+n2
mu1=c(-7,8)
mu2=c(3,8)
covmat1=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
covmat2=matrix(c(16,-2,-2,9), nrow=2, ncol=2)
k=5  #number of folds

k_fold = function(n1=n1,n2=n2,mu1=mu1,mu2=mu2,covmat1=covmat1,
                  covmat2=covmat2,k=k){
  #create data set
  X1=mvrnorm(n = n1, mu = mu1, Sigma = covmat1)
  X2=mvrnorm(n = n2, mu = mu2, Sigma = covmat2)
  df1 = data.frame(cbind(X1,rep(1,length(X1[,1]))))
  df2 = data.frame(cbind(X2,rep(2,length(X2[,1]))))
  df = merge(df1,df2,all=TRUE)
  colnames(df)=c("X1","X2","class")
  rm(X1,X2,df1,df2)

  #randomly split data in k folds
  fold_list=list()
  naming=c(1:k)
  for (i in 1:k){
    fold = sample.int(n=nrow(df), size=(N/k), replace=F)
    fold_list[[i]]= data.frame(df[fold,])
    names(fold_list)[[i]]=naming[i]
    df=df[-fold,]}

  return(fold_list)
}

folds=k_fold(n1,n2,mu1,mu2,covmat1,covmat2,k)

########################## Exercise 2 ############################
### a)
n=100
c=0.5

Mean_error=matrix(NaN,n,2)
Mean_error_test=matrix(NaN,n,2)
for(i in 1:n){
  df=validation_set(n1,n2,mu1,mu2,covmat1,covmat2,c)$training_sample
  df_test=validation_set(n1,n2,mu1,mu2,covmat1,covmat2,c)$test_sample

  LDA = lda(class ~ X1 + X2, data=df)
  class_lfit = as.numeric(predict(LDA)$class)
  class_lfit_test = as.numeric(predict(LDA,df_test)$class)

  QDA = qda(class ~ X1 + X2 , data=df)
  class_qfit = as.numeric(predict(QDA)$class)
  class_qfit_test = as.numeric(predict(QDA,df_test)$class)

  Mean_error[i,1] = (1/N)*sum(class_lfit!=df[,3])
  Mean_error[i,2] = (1/N)*sum(class_qfit!=df[,3])

  Mean_error_test[i,1] = (1/N)*sum(class_lfit_test!=df_test[,3])
  Mean_error_test[i,2] = (1/N)*sum(class_qfit_test!=df_test[,3])
}

#difference in average error of LDA compared to QDA
#training data:
avg_lda=mean(Mean_error[,1])
avg_qda=mean(Mean_error[,2])
#test data:
avg_lda_test=mean(Mean_error_test[,1])
avg_qda_test=mean(Mean_error_test[,2])


### b)
k=5

Mean_error_lda=matrix(NaN,n,k)
Mean_error_qda=matrix(NaN,n,k)
Mean_error_test_lda=matrix(NaN,n,k)
Mean_error_test_qda=matrix(NaN,n,k)
for(i in 1:n){
  for (l in 1:k){
    total_sample=k_fold(n1,n2,mu1,mu2,covmat1,covmat2,k)
    pre_sample=total_sample[-l]
    sample=do.call(rbind, pre_sample)
    names(sample)=c("X1","X2","class")
    test_sample=data.frame(total_sample[l])
    names(test_sample)=c("X1","X2","class")

    LDA = lda(class ~ X1 + X2, data=sample)
    class_lfit = as.numeric(predict(LDA)$class)
    class_lfit_test = as.numeric(predict(LDA,test_sample)$class)

    QDA = qda(class ~ X1 + X2 , data=sample)
    class_qfit = as.numeric(predict(QDA)$class)
    class_qfit_test = as.numeric(predict(QDA,test_sample)$class)

    num=nrow(sample)
    num1=nrow(test_sample)

    Mean_error_lda[i,l] = (1/num)*sum(class_lfit!=sample$class)
    Mean_error_qda[i,l] = (1/num)*sum(class_qfit!=sample$class)

    Mean_error_test_lda[i,l] = (1/num1)*sum(class_lfit_test!=test_sample$class)
    Mean_error_test_qda[i,l] = (1/num1)*sum(class_qfit_test!=test_sample$class)
  }}


#Creating a matrix to compare average mean error of LDA and QDA
#First column average mean error of the 5 folds for n runs of LDA
#Second column average mean error of the 5 folds for n runs of QDA
Mean_error=cbind(rowMeans(Mean_error_lda),rowMeans(Mean_error_qda))


#Same thing for the test samples
Mean_error_test=cbind(rowMeans(Mean_error_test_lda),rowMeans(Mean_error_test_qda))


#Comparing LDA and QDA of evaluation samples
mean(Mean_error_lda[,1])-mean(Mean_error_qda[,1])

#Comparing LDA and QDA of test samples
mean(Mean_error_test_lda[,1])-mean(Mean_error_test_qda[,1])

#Comparing average mean error of LDA for evaluation and test samples
mean(Mean_error_lda[,1])-mean(Mean_error_test_lda[,1])

#Comparing average mean error of QDA for evaluation and test samples
mean(Mean_error_qda[,1])-mean(Mean_error_test_qda[,1])


#### c)
"enter k=10 in line 167 and run b) again"
