setwd("~")
library(ropls)
library(GGally)

df <- read.csv(file = "~", head = 1, check.names=FALSE, stringsAsFactors = TRUE)
column = ncol(df)
metabolite <- as.matrix(df[,1:(column-1)])
group <- df[,column]

test <- apply(df[, 1:(column-1)], 2, 
                    function(x) t.test(x ~ df$group)$p.value) 
bonferoni <- p.adjust(test, 
                          method = "bonferroni", 
                          n = ncol(df[, 1:(column-1)])) 
bonferoni <- data.frame(bonferoni) 
subset(bonferoni, bonferoni < 0.05) 

FDR <- p.adjust(test, 
                     method = "fdr", 
                     n = ncol(df[, 1:(column-1)])) 
FDR <- data.frame(FDR) 
subset(FDR, FDR < 0.05) 

set.seed(71)
oplsda <- opls(metabolite, 
                 group, 
                 predI = 1,   
                 orthoI = NA, # NA=OPPLS-DA，0=PLS-DA
                 permI = 500, 
                 crossvalI = 7, 
                 scaleC = "standard", 
                 printL = FALSE, 
                 plotL = FALSE)  

oplsda

layout(matrix(1:4, nrow = 2, byrow = TRUE)) 
for(typeC in c("x-score", "overview", "permutation", "outlier"))
  
plot(oplsda, 
     typeVc = typeC,     
     parDevNewL = FALSE  
)
subset(oplsda@vipVn, oplsda@vipVn > 1.0)
write.table(cbind(FDR,oplsda@vipVn),"OPLSDA.csv",row.names=TRUE,col.names=TRUE,sep=",")
