# Course Project Writeup

### Introduction

The purpose of this project is to analyse a dataset from the [*Quantified Self Movement*](http://quantifiedself.com/),
a group of enthusiasts who regularly take measurements about themselves to improve 
their health, to find patterns in their behavior, or because they are tech geeks. 
Measures may come from sensor in devices that can be worn, like accelerometers
in the belt or bracelet. In this dataset we analyse measurements related to
weight lifting exercises performed by six subjects, in order to predict the manner
in which they did them. More information available on http://groupware.les.inf.puc-rio.br/har.

This is an assignment for the 3rd week of the Practical Machine Learning course by the Johns Hopkins University on [Coursera](http://www.coursera.org/).

### Raw data source

The raw data consist of training and test datasets for the study.

The training dataset is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test dataset is available here:
https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

### Loading the data
Let's load the data paying attention at the NA values; 
the first 6 variables identify user and time of the observation and thus
are not observation values; the last variable is the outcome that has to be
predicted and will be treated as a factor; the variables used as
predictors will be selected from the ones with less than 5% of NAs.

```r
trainDS<-read.csv('pml-training.csv',header=TRUE,sep=',',quote='"',na.strings=c('#DIV/0!','NA'),row.names=1)
testDS <-read.csv('pml-testing.csv', header=TRUE,sep=',',quote='"',na.strings=c('#DIV/0!','NA'),row.names=1)
dim(trainDS)    # focus only on trainDS now
```

```
## [1] 19622   159
```

```r
th <- round(dim(trainDS)[1] * .05) # threshold for NA: 5% of the rows
completeColumns   <- sapply(trainDS[,7:158], function(x) sum(is.na(x)) < th)
completeVariables <- names(completeColumns[completeColumns==TRUE]); completeVariables
```

```
##  [1] "roll_belt"            "pitch_belt"           "yaw_belt"            
##  [4] "total_accel_belt"     "gyros_belt_x"         "gyros_belt_y"        
##  [7] "gyros_belt_z"         "accel_belt_x"         "accel_belt_y"        
## [10] "accel_belt_z"         "magnet_belt_x"        "magnet_belt_y"       
## [13] "magnet_belt_z"        "roll_arm"             "pitch_arm"           
## [16] "yaw_arm"              "total_accel_arm"      "gyros_arm_x"         
## [19] "gyros_arm_y"          "gyros_arm_z"          "accel_arm_x"         
## [22] "accel_arm_y"          "accel_arm_z"          "magnet_arm_x"        
## [25] "magnet_arm_y"         "magnet_arm_z"         "roll_dumbbell"       
## [28] "pitch_dumbbell"       "yaw_dumbbell"         "total_accel_dumbbell"
## [31] "gyros_dumbbell_x"     "gyros_dumbbell_y"     "gyros_dumbbell_z"    
## [34] "accel_dumbbell_x"     "accel_dumbbell_y"     "accel_dumbbell_z"    
## [37] "magnet_dumbbell_x"    "magnet_dumbbell_y"    "magnet_dumbbell_z"   
## [40] "roll_forearm"         "pitch_forearm"        "yaw_forearm"         
## [43] "total_accel_forearm"  "gyros_forearm_x"      "gyros_forearm_y"     
## [46] "gyros_forearm_z"      "accel_forearm_x"      "accel_forearm_y"     
## [49] "accel_forearm_z"      "magnet_forearm_x"     "magnet_forearm_y"    
## [52] "magnet_forearm_z"
```

```r
#trainSubset <- trainDS[c("user_name", completeVariables, "classe")] # actual training subset
trainSubset <- trainDS[c(completeVariables, "classe")] # actual training subset
trainSubset$classe <- factor(trainSubset$classe)
```

$52$ variables have less than $981$ NAs (5% of $19622$)
and will be the starting point to find predictors.

### Exploratory Analysis

Let's now get a subset of the data to perform a quick exploratory analysis, to get
an idea of the distribution of the outcome variable.

```r
library(dplyr); explDS <- trainDS %>% select(user_name, time=cvtd_timestamp, classe) %>%
    group_by(user_name, time, classe) %>% summarise(num_obs=length(classe))
library(ggplot2); ggplot(explDS, aes(x=time, y=num_obs, fill=classe)) + facet_grid(user_name ~ .) +
    geom_bar(stat='identity',position='dodge') + theme(axis.text.x=element_text(angle=90,vjust=0.5))
```

<img src="Writeup_files/figure-html/unnamed-chunk-2-1.png" title="" alt="" style="display: block; margin: auto;" />

The outcome variable does not seem to follow a linear trend, therefore linear
predictors are unlikely to work well: we'll then try first with a random forest approach.

**NOTE:** It might be unclear if the variable $user\_name$ is to be considered as 
a predictor. The assigment explicitly asks to *predict activity quality from activity monitors*,
therefore here it is excluded.

### Partitioning and Training

Let's now create the actual training and test dataset out of the "trainSubset",
splitting them in the usual 60% - 40% partitions.
These will be used for the in-sampling assessment and validation:

```r
rm(trainDS, explDS) # save some memory
set.seed (123456)   # for reproducibility
library(caret); indexTraining <- createDataPartition(trainSubset$classe, p=0.6, list=TRUE)[[1]]
trainTraining <- trainSubset[indexTraining,]; trainTesting <- trainSubset[-indexTraining,]
```

The *training* with random forests method is computationally intensive and can 
be speeded up with parallel processing. Now check the accuracy of the model:

```r
library(doParallel); cluster<-makeCluster(detectCores()-1); registerDoParallel(cluster)
model=train(classe~., data=trainTraining, method='rf',trControl=trainControl('oob',seeds=list(123)),ntree=200)
stopCluster(cluster); registerDoSEQ()  # switch back to single-core processing
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, ntree = 200, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 200
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.82%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 3341    3    3    0    1 0.002090800
## B   22 2248    8    1    0 0.013602457
## C    0   13 2031   10    0 0.011197663
## D    0    1   24 1904    1 0.013471503
## E    0    0    4    6 2155 0.004618938
```

```r
model$results[model$results$mtry == model$finalModel$mtry,]
```

```
##    Accuracy     Kappa mtry
## 2 0.9915082 0.9892575   27
```

The model accuracy is $0.9915082$,
therefore the in-sample is 
$1-0.9915082 = 0.0084918$. The out-of-sample error should be close to this value (usually
a bit bigger): if not, the model could be overfitting the training dataset.

### Cross-Validation

And now, cross-validate with the testing subset:

```r
Q <- confusionMatrix(predict(model,trainTesting),trainTesting$classe)
Q; Q$overall["Accuracy"]
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 2231   12    0    0    0
##          B    1 1502    6    0    1
##          C    0    4 1358   14    3
##          D    0    0    4 1272    2
##          E    0    0    0    0 1436
## 
## Overall Statistics
##                                          
##                Accuracy : 0.994          
##                  95% CI : (0.992, 0.9956)
##     No Information Rate : 0.2845         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.9924         
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9996   0.9895   0.9927   0.9891   0.9958
## Specificity            0.9979   0.9987   0.9968   0.9991   1.0000
## Pos Pred Value         0.9947   0.9947   0.9848   0.9953   1.0000
## Neg Pred Value         0.9998   0.9975   0.9985   0.9979   0.9991
## Prevalence             0.2845   0.1935   0.1744   0.1639   0.1838
## Detection Rate         0.2843   0.1914   0.1731   0.1621   0.1830
## Detection Prevalence   0.2859   0.1925   0.1758   0.1629   0.1830
## Balanced Accuracy      0.9987   0.9941   0.9947   0.9941   0.9979
```

```
##  Accuracy 
## 0.9940097
```

The out-of-sample error is $1-0.9940097=0.0059903$, therefore
the model provides a pretty good estimate. 

### Prediction

On this basis, the prediction on the test dataset is:

```r
answers <- predict(model,testDS); answers
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```

Finally, here is the code to generate the files for the project submission:

```r
pml_write_files = function(x) {
  n = length(x)
  for(i in 1:n){
    filename = paste0("problem_id_",i,".txt")
    write.table(x[i],file=filename,quote=FALSE,row.names=FALSE,col.names=FALSE)
  }
}
pml_write_files (answers)
```
