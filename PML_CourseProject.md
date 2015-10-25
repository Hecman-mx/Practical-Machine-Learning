#   Practical Machine Learning: Course Project
##  Created By: hmcbmx

## Introduction
For this project, we are given data from accelerometers on the belt, forearm, arm, and dumbell of 6 research study participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: [GroupWare](http://groupware.les.inf.puc-rio.br/har) (see the section on the Weight Lifting Exercise Dataset). 

Our training data consists of accelerometer data and a label identifying the quality of the activity the participant was doing. Our testing data consists of accelerometer data without the identifying label. Our goal is to predict the labels for the test set observations.


## Data Preparation
Loading the required packages, and read in the training and testing data:


```r
# Installing all the packages needed
install.packages("caret")
install.packages("lattice")
install.packages("ggplot2")
install.packages("randomForest")
install.packages("rattle")
install.packages("rpart.plot")
install.packages("AppliedPredictiveModeling")

# Loading all the packages needed
library(caret, quietly=TRUE)
library(lattice, quietly=TRUE)
library(ggplot2, quietly=TRUE)
library(randomForest, quietly=TRUE)
library(rattle, quietly=TRUE)
library(rpart, quietly=TRUE)
library(AppliedPredictiveModeling, quietly=TRUE)

# Loading training and testing data
pmlTrain <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv")
pmlTest <- read.csv("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv")
```

## Splitting Data into Testing and Cross-Validation
To find an optimal model, with the best performance both in Accuracy as well as minimizing Out of Sample Error, the full testing data is split randomly with a set seed with 80% of the data into the training sample and 20% of the data used as cross-validation. When the samples are created, they are sliced by column against the feature set so only the variables of interest are fed into the final model.



```r
set.seed(100)
idxTrain <- createDataPartition(y=pmlTrain$classe, p=0.8, list=F)
pmlTrain1 <- pmlTrain[idxTrain, ]
pmlTrain2 <- pmlTrain[-idxTrain, ]
```


Now, we will attempt to reduce the number of features by removing variables with nearly zero variance, variables that are almost always NA, and variables that don't make intuitive sense for prediction. Note that I decide which ones to remove by analyzing pmlTrain1, and perform the identical removals on pmlTrain2:



```r
# remove variables with nearly zero variance
nzv <- nearZeroVar(pmlTrain1)
pmlTrain1 <- pmlTrain1[, -nzv]
pmlTrain2 <- pmlTrain2[, -nzv]

# remove variables that are almost always NA
mostlyNA <- sapply(pmlTrain1, function(x) mean(is.na(x))) > 0.95
pmlTrain1 <- pmlTrain1[, mostlyNA==F]
pmlTrain2 <- pmlTrain2[, mostlyNA==F]

# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables
pmlTrain1 <- pmlTrain1[, -(1:5)]
pmlTrain2 <- pmlTrain2[, -(1:5)]
```


## Model Building

Starting with Random Forest model, to see if it would have acceptable performance. I fit the model on pmlTrain1, and instruct the "train" function to use 3-fold cross-validation to select optimal tuning parameters for the model.



```r
# instruct train to use 3-fold CV to select optimal tuning parameters
fitControl <- trainControl(method="cv", number=3, verboseIter=F)

# fit model on pmlTrain1
fit <- train(classe ~ ., data=pmlTrain1, method="rf", trControl=fitControl)

# print final model to see tuning parameters it chose
fit$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.19%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4462    1    0    0    1 0.0004480287
## B    6 3030    2    0    0 0.0026333114
## C    0    5 2733    0    0 0.0018261505
## D    0    0    9 2563    1 0.0038865138
## E    0    1    0    4 2881 0.0017325017
```

The Random Forest model decided to use 500 trees and try 27 variables at each split.


## Model Evaluation and Selection

Using the fitted model to predict the label ("classe") in pmlTrain2, and show the confusion matrix to compare the predicted versus the actual labels:



```r
# use model to predict classe in validation set (pmlTrain2)
preds <- predict(fit, newdata=pmlTrain2)

# show confusion matrix to get estimate of out-of-sample error
confusionMatrix(pmlTrain2$classe, preds)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1116    0    0    0    0
##          B    2  755    2    0    0
##          C    0    1  680    3    0
##          D    0    0    0  643    0
##          E    0    0    0    0  721
## 
## Overall Statistics
##                                          
##                Accuracy : 0.998          
##                  95% CI : (0.996, 0.9991)
##     No Information Rate : 0.285          
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9974         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9982   0.9987   0.9971   0.9954   1.0000
## Specificity            1.0000   0.9987   0.9988   1.0000   1.0000
## Pos Pred Value         1.0000   0.9947   0.9942   1.0000   1.0000
## Neg Pred Value         0.9993   0.9997   0.9994   0.9991   1.0000
## Prevalence             0.2850   0.1927   0.1738   0.1647   0.1838
## Detection Rate         0.2845   0.1925   0.1733   0.1639   0.1838
## Detection Prevalence   0.2845   0.1935   0.1744   0.1639   0.1838
## Balanced Accuracy      0.9991   0.9987   0.9979   0.9977   1.0000
```

The accuracy is 99.8%, an the accuracy for the out-of-sample error is 0.2%.

This is an excellent result, so rather than trying additional algorithms, we will use Random Forests model to predict on the test set.

## Re-training the Selected Model
The first model we will run is a linear regression model against mpg for each variable. This gives us insight into variables with coefficient significance as well as an initial attempt at explaining mpg. Additionally, we will also look at the correlation of variables with mpg to help us choose an appropriate model.

Before predicting on the test set, it is important to train the model on the full training set (pmlTrain), rather than using a model trained on a reduced training set (pmlTrain1), in order to produce the most accurate predictions. Therefore, I now repeat everything I did above on pmlTrain and pmlTest:


```r
# remove variables with nearly zero variance
nzv <- nearZeroVar(pmlTrain)
pmlTrain <- pmlTrain[, -nzv]
pmlTest <- pmlTest[, -nzv]

# remove variables that are almost always NA
mostlyNA <- sapply(pmlTrain, function(x) mean(is.na(x))) > 0.95
pmlTrain <- pmlTrain[, mostlyNA==F]
pmlTest <- pmlTest[, mostlyNA==F]

# remove variables that don't make intuitive sense for prediction (X, user_name, raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp), which happen to be the first five variables
pmlTrain <- pmlTrain[, -(1:5)]
pmlTest <- pmlTest[, -(1:5)]

# re-fit model using full training set (pmlTrain)
fitControl <- trainControl(method="cv", number=3, verboseIter=F)
fit <- train(classe ~ ., data=pmlTrain, method="rf", trControl=fitControl)
```

## Making Test Set Predictions

Now, we will use the model fit on pmlTrain to predict the label for the observations in pmlTest, and write those predictions to individual files:


```r
# predict on test set
preds <- predict(fit, newdata=pmlTest)

# convert predictions to character vector
preds <- as.character(preds)

# create function to write predictions to files
pml_write_files <- function(x) {
    n <- length(x)
    for(i in 1:n) {
        filename <- paste0("problem_id_", i, ".txt")
        write.table(x[i], file=filename, quote=F, row.names=F, col.names=F)
    }
}

# create prediction files to submit
pml_write_files(preds)
```
