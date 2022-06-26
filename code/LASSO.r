setwd("~")
library(caret)
library(ranger)
library(glmnet)
library(pROC)
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

#lasso
ptm <- proc.time() 
set.seed(71) 
lasso_fit = train(train_set,  
                        train_set_group,　　
                        method = "glmnet",　　
                        tuneGrid = train_grid_lasso, 
                        trControl=tr,               
                        preProc = c("center", "scale"), 
                        metric = "ROC")　　　
lasso_fit
for (i in 1:10000) x <- rnorm(1000) 
proc.time() - ptm 
autoplot(lasso_fit$finalModel, xvar = "lambda") + theme(legend.position = 'none')

pred_train_lasso <- predict(lasso_fit, train_set) 
sprintf("Accuracy of LASSO on the train set: %f",accuracy(train_set_group,pred_train_lasso))
roc_lasso_train <- roc(train_set_group, as.numeric(pred_train_lasso)) # ROC, AUC
roc_lasso_train$auc  
roc_lasso_train$sensitivities 
roc_lasso_train$specificities 
#write.table(cbind(pred_train_lasso,train_set_group),"lasso_ROC_train.csv",row.names=FALSE,col.names=TRUE,sep=",")

pred_test_lasso <- predict(lasso_fit, test_set) 
sprintf("Accuracy of LASSO on the test set: %f",accuracy(test_set_group,pred_test_lasso))
roc_lasso <- roc(test_set_group, as.numeric(pred_test_lasso)) 
roc_lasso$auc  
roc_lasso$sensitivities 
roc_lasso$specificities 
#write.table(cbind(pred_test_lasso,test_set_group),"lasso_ROC_test.csv",row.names=FALSE,col.names=TRUE,sep=",")

important_variables(lasso_fit, scale = 1)