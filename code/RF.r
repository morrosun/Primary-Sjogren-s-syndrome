setwd("~")
library(caret)
library(ranger)
library(glmnet)
library(pROC)
library(ggplot2)
library(Metrics)
library(ggfortify)

df <- read.csv(file = "~", head = 1, check.names=FALSE, stringsAsFactors = TRUE)
column = ncol(df)
metabolite <- as.matrix(df[,1:(column-1)])
group <- df[,column]
type <- as.matrix(df[,column])

set.seed(71)
train_group <- createDataPartition(df$group,
                                  p = 0.7,   
                                  list = FALSE)
train_set <- metabolite[train_group, ]
test_set  <- metabolite[-train_group, ]
train_set_group <- type[train_group, ]
test_set_group <- type[-train_group, ]

set.seed(71)
tr = trainControl(
  method = "repeatedcv", 
  number = 5,　
  repeats = 5, 
  summaryFunction = twoClassSummary, 
  classProbs = TRUE)  
train_grid_lasso = expand.grid(alpha = 1, lambda = 10 ^ (0:10 * -1)) 

#RF

library("ranger")
train_grid_rf = expand.grid(mtry = 1:10, 
                            splitrule = "gini",
                            min.node.size = 5) 
ptm <- proc.time() 
set.seed(71) 
rf_fit = train(train_set,  
                   train_set_group,　   
                   method = "ranger",　　 
                   tuneGrid = train_grid_rf,    
                   trControl = tr,              
                   preProc = c("center", "scale"), 
                   metric = "ROC",                 
                   importance = "impurity_corrected") 
rf_fit
for (i in 1:10000) x <- rnorm(1000) 
proc.time() - ptm 

pred_train_rf <- predict(rf_fit, train_set)  
sprintf("Accuracy of RF on the train set: %f",accuracy(train_set_group,pred_train_rf))
roc_rf <- roc(train_set_group, as.numeric(pred_train_rf))
roc_rf$auc
roc_rf$sensitivities
roc_rf$specificities
write.table(cbind(pred_train_rf,train_set_group),"rf_ROC_train.csv",row.names=FALSE,col.names=TRUE,sep=",")

pred_test_rf <- predict(rf_fit, test_set)  
sprintf("Accuracy of RF on the test set: %f",accuracy(test_set_group,pred_test_rf))
roc_rf <- roc(test_set_group, as.numeric(pred_test_rf))
roc_rf$auc
roc_rf$sensitivities
roc_rf$specificities
#write.table(cbind(pred_test_rf,test_set_group),"rf_ROC_test.csv",row.names=FALSE,col.names=TRUE,sep=",")

important_variables(rf_fit, scale = 1)