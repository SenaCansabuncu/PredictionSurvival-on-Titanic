
#Loading the libraries
library(dplyr)
library(ggplot2)
library(e1071)
library(caret)
library(randomForest)
library(nnet)
library(NeuralNetTools)
library(neuralnet)
library(arules)
library(class)
library(caTools)
library(DMwR)
library(caret)
library(mlbench)
library(caret)
library(rpart)
library(caret)
library(DMwR)


#Read Training and testing Data Set

FileName_Func <- function(fileName){
  set_data <- read.table(file=fileName, header=TRUE, sep=",")
  return (set_data)
}

Train_set=FileName_Func("train.csv")
#head(Train_set)

Test_set=FileName_Func("test.csv")
head(Train_set)


# A bit positively skewed.
ggplot(Train_set, aes(x = Age))+geom_histogram(bins=30)


#Converting numeric and categorical variables in factors
Convert_DataType_Func <-function(Data_set,numeric_columns){
  head(Train_set)
  for(i in 1:ncol(Data_set)){
    if(i %in% numeric_columns)
      Data_set[,i] <- as.numeric(Data_set[,i])
    else
      Data_set[,i] <- as.factor(Data_set[,i])
  }
  return (Data_set)
}

Train_numeric_columns <- c(6,7,8,10)
Train_set=Convert_DataType_Func(Train_set,Train_numeric_columns)
head(Train_set)
Test_numeric_columns <- c(5,6,7,9)
Test_set=Convert_DataType_Func(Test_set,Test_numeric_columns)
head(Test_set)


# To categorize, Receiving only first character for cabin 
Train_set$Cabin=substring(Train_set$Cabin, 1, 1)
table(substring(Train_set$Cabin, 1, 1))
head(Train_set)
#Filling null cabin with Unknown
levels(Train_set$Cabin) <- c(levels(Train_set$Cabin), "unknown") 
Train_set$Cabin[Train_set$Cabin==""]="Unknown"
levels(Test_set$Cabin) <- c(levels(Test_set$Cabin), "unknown") 
Test_set[is.na(Test_set$Cabin),"Cabin"]="Unknown"

#Filling KNN IMPUTATION for Missing Data 
#Function that fills in all NA values using the k Nearest Neighbours of each case with NA values

KNN_IMPUTATION_FUNC <-function(Data_set,vrb){
  Train_Numeric_Part=Data_set[,vrb ]
  
  clean_Numeric_Train_Part <- knnImputation(Train_Numeric_Part, k = 10, scale = T, meth = "weighAvg",
                                            distData = NULL)
  Categoric_train_part=Data_set[,c("PassengerId","Pclass","Sex","Cabin","Embarked") ]
  Train_set=cbind(Categoric_train_part, clean_Numeric_Train_Part)
  return(Train_set)
}

vr=c("Survived","Age","SibSp","Parch","Fare")
Train_set=KNN_IMPUTATION_FUNC(Train_set,vr)
head(Train_set)

vr_test=c("Age","SibSp","Parch","Fare")
Test_set=KNN_IMPUTATION_FUNC(Test_set,vr_test)
head(Test_set)

# Summary of the data after imputation
summary(Train_set)

#Exploratory Data analysis

options(repr.plot.width = 20, repr.plot.height = 12)
#Distribution Age& gender: Similar distribution for both genders
ggplot(Train_set, aes(x = Age,fill = Sex))+geom_histogram()+theme(text = element_text(size = 10))

table(Train_set$Age)
#Fare distribution : Extremely postively skewed.
# distribution of Fare & Count of gender
ggplot(Train_set, aes(x = Fare,fill = Sex))+geom_histogram()+theme(text = element_text(size = 10))

# distribution of Fare & Count of gender
#median fare for people who survived was higher than people who did not
#outliers fare for people who survived was higher than people who did not
ggplot(Train_set, aes(x = Fare, fill = Survived))+geom_boxplot()+theme(text = element_text(size = 10))

#The most median surviving embarkation port was C. Fare in Embarked C's median almost 90.
#Embarked Q has people that received lowest fare.
ggplot(Train_set, aes(x = Fare, fill= Survived,y = Embarked)) +geom_boxplot()+theme(text = element_text(size = 10))
table(Train_set$Embarked)

#Fare distribution vs age & survival 
family_size <-Train_set$SibSp+Train_set$Parch
ggplot(Train_set, aes(x = Fare,y = Age, color = Survived, size = family_size))+scale_size(range = c(0.5,15))+geom_point(alpha = 0.5)+theme(text = element_text(size = 10))


#According To PClass & Survival
ggplot(Train_set,aes(x=Pclass))+
  geom_bar(aes(fill=Survived),position="dodge")

ggplot(Train_set,aes(x=Sex))+
  geom_bar(aes(fill=Survived),position="dodge")

#scatterplot func. ggplot
#It appears from the scatterplot plot that the probability of survival is less as age increases.
ggplot(Train_set,
       aes(y = Survived,
           x = Age, colour = Age)) +
  geom_point(alpha = 0.6)


#Dummy Coding For Embarked Column
Train_set$Embarked=as.factor(Train_set$Embarked)
table(Train_set$Embarked)
dummy_Embarked <- as.data.frame(model.matrix(~ Embarked, data = Train_set))
head(dummy_Embarked[,2:4])
new_train_Set=cbind(Train_set, dummy_Embarked[,2:4])


#Dummy Coding For Gender Column

table(Train_set$Sex)
dummy_Gender <- as.data.frame(model.matrix(~ Sex + 0, data = Train_set))
#head(dummy_Gender)
new_train_Set=cbind(new_train_Set, dummy_Gender[,1:2])
new_train_Set= new_train_Set[-c(2)]

#Dummy Coding For Pclass Column

table(Train_set$Pclass)
dummy_Pclass <- as.data.frame(model.matrix(~ Pclass+ 0 , data = Train_set))
new_train_Set=cbind(new_train_Set, dummy_Pclass[,])
#new_train_Set= new_train_Set[-c(2)]
head(new_train_Set)

#new_train_Set = new_train_Set[-c(12)]

new_train_Set$EmbarkedS<-as.factor(new_train_Set$EmbarkedS)
new_train_Set$EmbarkedC<-as.factor(new_train_Set$EmbarkedC)
new_train_Set$EmbarkedQ<-as.factor(new_train_Set$EmbarkedQ)



#Test Train Split
#In order to choose the best model, the train data was divided into test and train datasets to create the model and to assess the performance of the the models.

set.seed(1)
my_indexes <- caret::createDataPartition(y = new_train_Set$Survived, times = 1, p = .75, list = F)
train1 <- as.data.frame(new_train_Set[my_indexes,])
test1 <- as.data.frame(new_train_Set[-my_indexes,])
summary(new_train_Set)


#Note:
#Cabin feauture has a lot of unknown values. So i didnt use this feature in models.

#A       B       C       D       E       F       G       T Unknown 
#15      47      59      33      32      13       4       1     687 


####### Applying CART Algorithm  ################################################

CART_model <- rpart(Survived ~ ., method = "class", data = new_train_Set[,-c(1,2,3,4)])
#Data that used in models.
head(new_train_Set[,-c(1,2,3,4)])
plot(CART_model, uniform=TRUE,
     main="Titanic Survivoral chart")
text(CART_model, use.n=TRUE, all=TRUE, cex=.8)

# Finding Predictions of The Model
CART_predictions <- predict(CART_model, newdata = test1[,-5], type = "class")


# Performance Evaluation
confusionMatrix(data = CART_predictions, reference = test1[,5], dnn = c("Predictions", "Actual/Reference"), mode = "everything")
#Accuracy : 0.8514
#Sensitivity : 0.9343          
#Specificity : 0.7176


#Support Vector Machine###############
model_svm<-svm(Survived~
               +Sexfemale 
               +Sexmale
               #+Cabin
               +Age
               +SibSp
               +Parch
               +Fare
               +EmbarkedC
               +EmbarkedQ
               +EmbarkedS
               +Pclass1 
               +Pclass2
               +Pclass3
               ,data = train1, kernel = 'radial')
y_pred<-predict(model_svm,newdata = test1) 
confusionMatrix(test1$Survived,y_pred)

summary (y_pred)
#Accuracy : 0.8288 
#Sensitivity : 0.8462          
#Specificity : 0.7975  


#Random Forest Model##############

head(train1)
set.seed(101)
model.rf<-randomForest(Survived~
                         +Sexfemale 
                       +Sexmale
                       #+Cabin
                       +Age
                       +SibSp
                       +Parch
                       +Fare
                       +EmbarkedC
                       +EmbarkedQ
                       +EmbarkedS
                       +Pclass1 
                       +Pclass2
                       +Pclass3,
                       data = train1,ntrees=2000,importance=TRUE,proximity=TRUE, mtry=2)
#colnames(importance(model.rf))
caret::varImp(model.rf)
#Accuracy : 0.8468          
#Sensitivity : 0.8411          
#Specificity : 0.8592
#Importance of Columns
# 0         1
# Pclass    18.928215 18.928215
# Sex       47.038763 47.038763
# Age       22.511265 22.511265
# # SibSp     10.197929 10.197929
# Parch      9.708086  9.708086
# Fare      18.165886 18.165886
# EmbarkedC  2.390244  2.390244
# EmbarkedQ  4.025041  4.025041
# EmbarkedS  6.453431  6.453431

varImpPlot(model.rf)
plot(model.rf)
y_pred.rf<-predict(model.rf,newdata = test1,type='class')
confusionMatrix(test1$Survived,as.factor(y_pred.rf))



##Neural Network#######

set.seed(5)
model.nnet<-nnet(Survived~
                   +Sexfemale 
                 +Sexmale
                 #+Cabin
                 +Age
                 +SibSp
                 +Parch
                 +Fare
                 +EmbarkedC
                 +EmbarkedQ
                 +EmbarkedS
                 +Pclass1 
                 +Pclass2
                 +Pclass3,
                 data = train1, size =3, maxit = 100000, decay = 5e-4)

y_nnet<-predict(model.nnet,newdata = test1,type = 'class')
confusionMatrix(as.factor(y_nnet),test1$Survived)#86.52
plotnet(model.nnet)

#Accuracy : 0.8423  
#Sensitivity : 0.9124          
#Specificity : 0.7294



#final submission

#Test Dummy Coding For Embarked Column
Test_set$Embarked=as.factor(Test_set$Embarked)
table(Test_set$Embarked)
test_Embarked <- as.data.frame(model.matrix(~ Embarked+0, data = Test_set))
head(test_Embarked)
new_Test_Set=cbind(Test_set, test_Embarked[,1:3])

head(new_Test_Set)

new_Test_Set$EmbarkedS<-as.factor(new_Test_Set$EmbarkedS)
new_Test_Set$EmbarkedC<-as.factor(new_Test_Set$EmbarkedC)
new_Test_Set$EmbarkedQ<-as.factor(new_Test_Set$EmbarkedQ)

#Dummy Coding For Gender Column
test_Gender <- as.data.frame(model.matrix(~ Sex + 0, data = Test_set))
head(new_Test_Set)
new_Test_Set=cbind(new_Test_Set, test_Gender[,1:2])

#Dummy Coding For Pclass Column

table(Train_set$Pclass)
Test_Pclass <- as.data.frame(model.matrix(~ Pclass+ 0 , data = Test_set))
new_Test_Set=cbind(new_Test_Set, Test_Pclass[,1:3])
head(new_Test_Set)

new_Test_Set= new_Test_Set[-c(2,3,4,5)]



#Final Prediction using Ensemble stacked
#Stacked

df<-data.frame(rf=predict(
  model.rf,new_train_Set,type = 'prob'),
  svm=predict(model_svm,new_train_Set, type = 'raw'),
  nnet=predict(model.nnet,new_train_Set,type = 'raw'),
  Survived=new_train_Set$Survived)
model.stacked<-nnet(Survived~.,data=df,size =3, maxit = 10000, decay = 5e-4)


head(new_Test_Set)
#applying on the test data
nnet<-predict(model.nnet,new_Test_Set, type = 'raw')
rf<-predict(model.rf,new_Test_Set, type = 'prob')
svm<-predict(model_svm,new_Test_Set)


newdata = data.frame(nnet=nnet,rf=rf,svm=svm)


final<-as.factor(predict(model.stacked,newdata = newdata,type='class'))


head(new_Test_Set)

#writing the submission file
# this submission file can check with gender submission file
sub<-data.frame(PassengerId=new_Test_Set$PassengerId,Survived=final)

write.csv(sub,'submission.csv',row.names=FALSE)