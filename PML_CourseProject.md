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
fit <- train(classe ~ ., data=pmlTrain1, method="rpart", trControl=fitControl)

# print final model to see tuning parameters it chose
fit$finalModel
```

```
## n= 15699 
## 
## node), split, n, loss, yval, (yprob)
##       * denotes terminal node
## 
##  1) root 15699 11235 A (0.28 0.19 0.17 0.16 0.18)  
##    2) roll_belt< 129.5 14308  9895 A (0.31 0.21 0.19 0.18 0.11)  
##      4) pitch_forearm< -33.95 1245     8 A (0.99 0.0064 0 0 0) *
##      5) pitch_forearm>=-33.95 13063  9887 A (0.24 0.23 0.21 0.2 0.12)  
##       10) magnet_dumbbell_y< 437.5 11038  7933 A (0.28 0.18 0.24 0.19 0.11)  
##         20) roll_forearm< 123.5 6886  4110 A (0.4 0.19 0.19 0.17 0.056) *
##         21) roll_forearm>=123.5 4152  2784 C (0.079 0.17 0.33 0.23 0.19) *
##       11) magnet_dumbbell_y>=437.5 2025   989 B (0.035 0.51 0.044 0.22 0.18) *
##    3) roll_belt>=129.5 1391    51 E (0.037 0 0 0 0.96) *
```

```r
# Plotting our classification tree
fancyRpartPlot(fit$finalModel)
```

![plot of chunk unnamed-chunk-3](figure/unnamed-chunk-3-1.png) 

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
##          A 1001   19   73    0   23
##          B  291  260  208    0    0
##          C  306   19  359    0    0
##          D  271  124  248    0    0
##          E  105   98  159    0  359
## 
## Overall Statistics
##                                           
##                Accuracy : 0.5045          
##                  95% CI : (0.4887, 0.5202)
##     No Information Rate : 0.5032          
##     P-Value [Acc > NIR] : 0.4429          
##                                           
##                   Kappa : 0.3537          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.5071  0.50000  0.34288       NA  0.93979
## Specificity            0.9410  0.85336  0.88700   0.8361  0.89777
## Pos Pred Value         0.8970  0.34256  0.52485       NA  0.49792
## Neg Pred Value         0.6534  0.91783  0.78759       NA  0.99282
## Prevalence             0.5032  0.13255  0.26689   0.0000  0.09737
## Detection Rate         0.2552  0.06628  0.09151   0.0000  0.09151
## Detection Prevalence   0.2845  0.19347  0.17436   0.1639  0.18379
## Balanced Accuracy      0.7240  0.67668  0.61494       NA  0.91878
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
fit <- train(classe ~ ., data=pmlTrain, method="rpart", trControl=fitControl)

# Plotting our classification tree
fancyRpartPlot(fit$finalModel)
```

![plot of chunk unnamed-chunk-5](figure/unnamed-chunk-5-1.png) 

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
