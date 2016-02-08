tab=read.table('spam.txt',header=T,sep=';')
namespam=row.names(read.table('spaminfo.txt',sep=':',comment.char="|"))
names(tab)<-c(namespam,'spam')

set.seed(1)

# PRELILINARY ANALYSIS

# 4601 observations of 58 variables

ind=read.table('indtrain.txt',header=T,sep='\n')
ind=as.matrix(ind)

tabTrain=tab[ind,]
tabTest=tab[-ind,]

# CLASSIFICATION TREES

library(rpart)

tree=rpart('spam~.',tabTrain,method='class')

#quartz()
#x11(width=10,height=10)
plot(tree,cex=0.8)
text(tree) # A faire après le plot

summary(tree)

# Most influent variable : char_freq_$
# Number of variables in the tree : only 5

##### TRAINING SET #####

Y=as.numeric(tabTrain$spam)-1
Yp=predict(tree,tabTrain)
Yp=as.numeric(Yp[,2]>0.5)

err=mean(Yp!=Y) # 10% error on training set
table(Y,Yp) # confusion matrix

# False positive rate : FP/(FP+VN) = 98/(98+2001) = 0.0467
# False negative rate : 244/(244+2001) = 0.180

##### TEST SET #####

Y=as.numeric(tabTest$spam)-1
Yp=predict(tree,tabTest)
Yp=as.numeric(Yp[,2]>0.5)

err=mean(Yp!=Y) # 11% error on test set
table(Y,Yp) # confusion matrix

# FP rate: FP/(FP+VN) = 30/(30+659) = 0.0435
# FN rate: 97/(97+365) = 0.210

# Conclusion: reasonably no overfitting, the classification tree favours missed detection 
# to false alarms which is good : it is easier to deal with a spam than to recuperate a discarded message.

##### BAGGING #####

library(ipred)
bag=bagging(spam~.,tabTrain) # default nbagg = 25

##### TRAIN #####

Y=as.numeric(tabTrain$spam)-1
YpBag = predict(bag, tabTrain, type="class")
YpBag = as.numeric(YpBag)-1
errBag=mean(YpBag!=Y) # 0,3188 % error
table(Y,YpBag)
# FP rate : FP/(FP+VN) = 5/(5+2094) = 0.00238
# FN rate : FN/(FN+VP) = 6/(6+1345) = 0.00444

##### TEST #####

Y=as.numeric(tabTest$spam)-1
YpBag = predict(bag, tabTest, type="class")
YpBag = as.numeric(YpBag)-1
errBag=mean(YpBag!=Y) # 5,994 % d'erreur de classification sur la base de test

table(Y,YpBag)

# FP rate : FP/(FP+VN) = 22/(22+667) = 0.0319
# FN rate : FN/(FN+VP) = 47/(47+415) = 0.102

# Conclusion
# Base de test : error cut in half compared to one tree.
# Remark : overfitting. No surprise, since the classifier is more complex than previously.



# To push further the analysis of different models :

### ------ COMPARISON OF CLASSIFICATION MODELS ----- ###

library(MASS)
library(rpart.plot)
library(e1071)
library(randomForest)
library(ipred)

# initialise error containers
TrainingErrorCART=c() 
TestErrorCART=c()
TrainingErrorBag=c()
TestErrorBag=c()
TrainingErrorForest=c()
TestErrorForest=c()
TrainingErrorLogit=c()
TestErrorLogit=c()
TrainingErrorLin=c()
TestErrorLin=c()
TrainingErrorSVM=c()
TestErrorSVM=c()


for (k in 1:10)
  
{
    print(k)
  
  # create random training and test sets : 75-25
  shuffle=sample(1:nrow(tab))
  ntrain=round(nrow(tab)*0.75) 
  ntest=nrow(tab)-round(nrow(tab)*0.75)
  tabTrain=tab[shuffle[seq(1,ntrain)],]
  tabTest=tab[shuffle[seq(1+ntrain,nrow(tab))],]

  
  #CART
  CART = rpart(spam~.,data=tabTrain)
  #-- train
  Y = as.numeric(tabTrain$spam)-1
  Yp = predict(CART, tabTrain, type="class")
  Yp = as.numeric(Yp)-1
  TrainingErrorCART=c(TrainingErrorCART,mean(Yp!=Y)) 
  #--- test
  Y = as.numeric(tabTest$spam)-1
  Yp = predict(CART, tabTest, type="class")
  Yp = as.numeric(Yp)-1
  TestErrorCART=c(TestErrorCART,mean(Yp!=Y))  
  
  #Bagging
  BAG = bagging(spam~.,data=tabTrain) # vote pour 25 arbres
  #--- train
  Y = as.numeric(tabTrain$spam)-1
  Yp = predict(BAG, tabTrain, type="class")
  Yp = as.numeric(Yp)-1
  TrainingErrorBag=c(TrainingErrorBag,mean(Yp!=Y)) 
  #--- test
  Y = as.numeric(tabTest$spam)-1
  Yp = predict(BAG, tabTest, type="class")
  Yp = as.numeric(Yp)-1
  TestErrorBag=c(TestErrorBag,mean(Yp!=Y))
  
  #Random Forest
  Xtrain=as.matrix(tabTrain[,1:57])
  Ytrain=tabTrain$spam
  randF = randomForest(Xtrain, Ytrain)
  #--- train
  Y=as.numeric(tabTrain$spam)
  YpRF = predict(randF, tabTrain, type="class")
  YpRF = as.numeric(YpRF)
  TrainingErrorForest =c(TrainingErrorForest, mean(YpRF!=Y))
  #--- test
  Y = as.numeric(tabTest$spam)
  YpRF = predict(randF, tabTest, type="class")
  YpRF = as.numeric(YpRF)
  TestErrorForest=c(TestErrorForest,mean(YpRF!=Y) )
  

  #Logistic regression (Scoring)
  RegLog=glm(spam~.,data=tabTrain, family = binomial)
  #--- train
  Y=as.numeric(tabTrain$spam)-1
  YpLog = predict(RegLog, tabTrain, type="response")
  YpLog = as.numeric(YpLog)>0.5
  TrainingErrorLogit=c(TrainingErrorLogit,mean(YpLog !=Y)) 
  #--- test
  Y=as.numeric(tabTest$spam)-1
  YpLog = predict(RegLog, tabTest,type="response")
  YpLog = as.numeric(YpLog)>0.5
  TestErrorLogit=c(TestErrorLogit,mean(YpLog !=Y)) 
  
  
  #Linear Discriminant Analysis
  Lin = lda(spam~., family=binomial(logit), data=tabTrain)
  #--- train
  Y=as.numeric(tabTrain$spam)-1
  Yp = predict(Lin, tabTrain, type="class")
  Yp = Yp$posterior>0.5
  TrainingErrorLin=c(TrainingErrorLin,mean(Yp !=Y)) 
  #--- test
  Y=as.numeric(tabTest$spam)-1
  Yp = predict(Lin, tabTest, type="class")
  Yp = Yp$posterior>0.5
  TestErrorLin=c(TestErrorLin,mean(Yp !=Y))
  
  #SVM
  SVM = svm(spam~.,type = "C-classification", kernel = "radial",cost=1.0, data=tabTrain)
  #--- train
  Y=as.numeric(tabTrain$spam)-1
  Yp = predict(SVM, tabTrain, type="class")
  Yp = as.numeric(Yp)>0.5
  TrainingErrorSVM=c(TrainingErrorSVM,mean(Yp !=Y)) 
  #--- test
  Y=as.numeric(tabTest$spam)-1
  Yp = predict(SVM, tabTest, type="class")
  Yp = as.numeric(Yp)>0.5
  TestErrorSVM=c(TestErrorSVM,mean(Yp !=Y)) 
  
}

testerror=list(TestErrorCART,TestErrorLin,TestErrorLogit,TestErrorSVM,TestErrorBag,TestErrorForest)
trainingerror=list(TrainingErrorCART,TrainingErrorLin,TrainingErrorLogit,TrainingErrorSVM,TrainingErrorBag,TrainingErrorForest)
names=c("CART","LDA","Logit","SVM","Bagging","RF")

# Show results
boxplot(trainingerror,names=names,main="Spam Train")
boxplot(testerror,names=names,main="Spam Test")

# Without any fine tuning, random forests perform the best (median of the error : arround 5%)
# However, Bagging in second position computes much less tress (default : 25 against 500)
# (Increase k number of iterations to be more precise...)


