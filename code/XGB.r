setwd("~")
library(xgboost)
library(ggplot2)
library(caret)
library(pROC)
library(rBayesianOptimization)
df <- read.csv(file = "~", head = 1, check.names=FALSE, stringsAsFactors = TRUE)
column = ncol(df)

#xgboost
set.seed(71) 
df <- na.omit(df)
train_group <- createDataPartition(df$group, 
                                  p = 0.7,           
                                  list = FALSE)
train_set <- df[train_group,]
test_set  <- df[-train_group,]
train_dummy <- dummyVars(~., data=train_set)
test_dummy <- dummyVars(~., data=test_set)
train_set_dummy <- data.frame(predict(train_dummy, train_set))
test_set_dummy <- data.frame(predict(test_dummy, test_set))
train_x <- data.matrix(train_set_dummy[, 1:(column-1)])
train_y <- data.matrix(train_set_dummy[, column])
test_x <- data.matrix(test_set_dummy[, 1:(column-1)])
test_y <- data.matrix(test_set_dummy[, column])
train <- xgb.DMatrix(train_x, label = train_y)

ptm <- proc.time() 
cv_folds <- KFold(train_y, 
                  nfolds = 5,
                  stratified = TRUE, 
                  seed = 71)
xgb_cv_bayesopt <- function(max_depth, subsample, lambda, alpha) {
  cv <- xgb.cv(params = list(booster = "gbtree", 
                             eta = 0.2,
                             max_depth = max_depth,
                             subsample = subsample, 
                             lambda = lambda, 
                             alpha = alpha,
                             colsample_bytree = 0.7,
                             objective = "binary:logistic",
                             eval_metric = "auc"), #"auc"."logloss"
               data = train, 
               folds = cv_folds, 
               nround = 100,
               early_stopping_rounds = 20, 
               maximize = TRUE, 
               verbose = 1)
  list(Score = cv$evaluation_log$test_auc_mean[cv$best_iteration],
       Pred = cv$pred)
}
set.seed(71) 
Opt_res <- BayesianOptimization(xgb_cv_bayesopt,
                                bounds = list(max_depth = c(3L, 7L),
                                              subsample = c(0.7, 1.0),
                                              lambda = c(0.5, 1), 
                                              alpha = c(0.0, 0.5)), 
                                init_points = 20, 
                                n_iter = 30,
                                acq = "ucb", 
                                kappa = 5, 
                                verbose = 1)
params <- list(
  "objective"           = "binary:logistic",
  "eval_metric"         = "auc",
  "eta"                 = 0.2,
  "max_depth"           = 7,
  "subsample"           = 0.7,
  "alpha"               = 0.5,
  "lambda"              = 0.5
)
set.seed(71)
cv_nround = 100
cv_test <- xgb.cv(params = params, data = train, nfold = 5, nrounds = cv_nround, 
                 early_stopping_rounds = 20, maximize = TRUE, verbose = 1)

cv_nround <- cv_test$best_iteration

model <- xgboost(data = train, 
                 params = params, 
                 nrounds = cv_nround, 
                 verbose = FALSE)
pred <- predict(model, test_x)
for (i in 1:10000) x <- rnorm(1000) 
proc.time() - ptm 

# Defining performance
loglikelihood <- function(y, py) {
  pysmooth <- ifelse(py==0, 1e-12,
                     ifelse(py==1, 1-1e-12, py))
  sum(y * log(pysmooth) + (1-y)*log(1 - pysmooth))
}

accuracy <- function(pred, truth, name="model") {
  dev.norm <- -2*loglikelihood(as.numeric(truth), pred)/length(pred)
  ctable <- table(truth=truth,
                  pred=(pred>0.5))
  accuracy <- sum(diag(ctable))/sum(ctable)
  precision <- ctable[2,2]/sum(ctable[,2])
  recall <- ctable[2,2]/sum(ctable[2,])
  f1 <- precision*recall
  data.frame(model=name, accuracy=accuracy, f1=f1, dev.norm)
}
accuracy(pred, as.numeric(test_y))

for(i in 1:length(pred)){
  if(pred[i]<0.5) {pred[i]="0"}
  else if(pred[i]>0.5) {pred[i]="1"}
}
roc_xgb <- roc(pred, as.numeric(test_y)) 
roc_xgb$auc  
roc_xgb$sensitivities 
roc_xgb$specificities

write.table(cbind(test_y,pred),"XGB_ROC_test.csv",row.names=FALSE,col.names=TRUE,sep=",")

evalframe <- as.data.frame(cv_test$evaluation_log)
ggplot(evalframe, aes(x = iter, y = test_auc_mean)) +
  geom_line() +
  geom_vline(xintercept = cv_nround, color = "darkred", linetype = 2) +
  ggtitle("Cross-validated auc")
ggplot(evalframe, aes(x = iter, y = train_auc_mean)) +
  geom_line() +
  geom_vline(xintercept = cv_nround, color = "darkred", linetype = 2) +
  ggtitle("Cross-validated auc")  

