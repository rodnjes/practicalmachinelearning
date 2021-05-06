---
title: "Can Fitness Devices Determine if Exercise is Done Correctly"
author: "Jessica Hyde"
date: "06/05/2021"
output: 
  html_document: 
    keep_md: yes
---

# Summary

Peolpe often wear exercise gadgets to determine how much exercise they do, but not if they are exercising correctly.  This analysis will see if fitness trackers can determine if barbell lifts are performed correctly or incorrectly.  The model that was selected was a random forest one, which was able to achieve high accuracy on the training and cross-validation set. 

## Different classifications trackers are tested on

* A: exactly according to the specification

* B: throwing the elbows to the front

* C: lifting the dumbbell only halfway

* D: lowering the dumbbell only halfway

* E: throwing the hips to the front




## libaries used


```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

# preparing the data for modeling

## data 

First I downloaded the data from the websites below and then make them available to R


```r
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv","pml-training.csv")
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv","pml-testing.csv")

trainingRaw <- read.csv("pml-training.csv")
testingRaw <- read.csv("pml-testing.csv")
```

## clean the data

For this analysis I will be using the numeric data points that are not missing data.  I will also remove columns which do not
make sence in affecting the measurments of exercise correctness, such as date, index, and window variables


```r
#remove NA's

training <- trainingRaw[, colSums(is.na(trainingRaw)) == 0]
testing <- testingRaw[, colSums(is.na(testingRaw)) == 0]

# remove non numeric data but keel the training classifications
training <- cbind(training[, sapply(training, is.numeric)],factor(training$classe))
colnames(training)[57] <- "classe"
testing <- testing[, sapply(testing, is.numeric)]

# remove index, timestaps and window columns as they should not affect analysis
training <- training[, !grepl("^X|timestamp|window", names(training))]
testing <- testing[, !grepl("^X|timestamp|window", names(testing))]

# enusre no data is near zero value
nearZeroVar(testing)
```

```
## integer(0)
```

```r
nearZeroVar(training)
```

```
## integer(0)
```
## Slice the training data into 2 sets for testing 

I will slice the training data into 2 sets, 75%  for training models and 25% for cross validation of the model.  


```r
set.seed(13)
inTrain <- caret::createDataPartition(training$classe,p=3/4,list=FALSE)

trainData <- training[inTrain,]
validData <- training[-inTrain,]
```

# Modeling the Data
I am going to try 3 different decision tree methods to find the model that fits the data best. (Please note it may take some time to run all 3)

* Classification and Regression Tree (CART)
* Gradient Boosting Machine Tree (CBM)
* Random Forest Tree (RF)


```r
varNames <-names(trainData)[1:length(trainData)-1]

CARTmodel <- train(
  classe ~ ., 
  data=trainData[, c('classe', varNames)],
  method='rpart'
)

GBMmodel <- train(
  classe ~ ., 
  data=trainData[, c('classe',varNames )],
  method='gbm'
)

RFmodel <- train(
  classe ~ ., 
  data=trainData[, c('classe', varNames)],
  method='rf',
)

```

## Evaluate the accuracy of the models on the training and the crossvalidation data


```r
trainingPredCART <- predict(CARTmodel, trainData)
trainConfusionMatrixCART<-confusionMatrix(trainingPredCART, trainData$classe)
trainingPredGBM <- predict(GBMmodel, trainData)
trainConfusionMatrixGBM<-confusionMatrix(trainingPredGBM, trainData$classe)
trainingPredRF <- predict(RFmodel, trainData)
trainConfusionMatrixRF<-confusionMatrix(trainingPredRF, trainData$classe)

validPredCART <- predict(CARTmodel, validData)
validConfusionMatrixCART<-confusionMatrix(validPredCART, validData$classe)
validPredGBM <- predict(GBMmodel, validData)
validConfusionMatrixGBM<-confusionMatrix(validPredGBM, validData$classe)
validPredRF <- predict(RFmodel, validData)
validConfusionMatrixRF<-confusionMatrix(validPredRF, validData$classe)

AccuracyResults <- data.frame(
  Model = c('training CART', 'training GBM', 'training RF','cross-validate CART', 'cross-validate GBM', 'cross-validate RF'),
  Accuracy = rbind(trainConfusionMatrixCART$overall[1], trainConfusionMatrixGBM$overall[1], trainConfusionMatrixRF$overall[1], validConfusionMatrixCART$overall[1], validConfusionMatrixGBM$overall[1], validConfusionMatrixRF$overall[1])
)
print(AccuracyResults)
```

```
##                 Model  Accuracy
## 1       training CART 0.5770485
## 2        training GBM 0.9733659
## 3         training RF 1.0000000
## 4 cross-validate CART 0.5676998
## 5  cross-validate GBM 0.9596248
## 6   cross-validate RF 0.9938825
```
Looking at the different models, the most accurate model when it comes to the cross-validated data is the random forest tree model.



```r
RFmodel$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.65%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4178    4    1    0    2 0.001672640
## B   20 2822    5    1    0 0.009129213
## C    0    8 2552    7    0 0.005843397
## D    0    0   31 2379    2 0.013681592
## E    0    1    5    8 2692 0.005173688
```
# Prediction

```r
prediction <-predict(RFmodel,testing)
results <- data.frame(id = testing$problem_id,predicted = prediction)

print(results)
```

```
##    id predicted
## 1   1         B
## 2   2         A
## 3   3         B
## 4   4         A
## 5   5         A
## 6   6         E
## 7   7         D
## 8   8         B
## 9   9         A
## 10 10         A
## 11 11         B
## 12 12         C
## 13 13         B
## 14 14         A
## 15 15         E
## 16 16         E
## 17 17         A
## 18 18         B
## 19 19         B
## 20 20         B
```

# Conclusion
By training a random forest model to a high accuracy level, predictions can by looking at exercise gadget data as to whether one is exercising correctly or not.

