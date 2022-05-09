

library(e1071)
library(tictoc)
library(kernlab)
library(matlib)
x <- read.csv("mydata3.csv")
x <- x[1:200,]
y <- read.csv("y3.csv")[1:200,]
data.lhsk  <- cbind(x, y)
names(data.lhsk)[ncol(data.lhsk)] <- "y"

tic("total")
seed <- 0
no.col <- 3 #the dimension of subspace
k <- 5 #5-fold CV
all.testerr <- c()

crit.col <- rep(list(c()), 5)

# random sample the dataset
set.seed(seed);id.trts <- sample(nrow(data.lhsk))
seed <- seed+1
folds.train <- cut(seq(1,nrow(data.lhsk)),breaks=5,labels=FALSE)

# hyperparameter selction
degree=c(1)
epsilon =c(0.01, 0.1, 0.5)
cost=c(1, 2, 5)
#gamma = c(1/no.col)
par.test = expand.grid(degree, epsilon, cost)


for (m in 1:5) {
  print(paste0("Iteration: ",m,"******************************"))
  
  #running time for each train/test split
  tic("each dataset")
  
  #split the train and test data
  testIndexes <- id.trts[which(folds.train==m,arr.ind=TRUE)]
  train.set4 <- data.lhsk[-testIndexes, ]
  test.set4 <- data.lhsk[testIndexes, ]
  
  ### DATA NORMALIZATION ########################################################
  x.mean <- apply(train.set4[,-ncol(train.set4)], 2, mean)
  x.sd <- apply(train.set4[,-ncol(train.set4)], 2, sd)
  train.set4[,-ncol(train.set4)] <- t(apply(train.set4[,-ncol(train.set4)], 1, function(x) (x-x.mean)/x.sd))
  test.set4[,-ncol(test.set4)] <- t(apply(test.set4[,-ncol(test.set4)], 1, function(x) (x-x.mean)/x.sd))
  ###############################################################################

  new.form <- c()
  
  # random sample the train  dataset
  set.seed(seed);id.crit <- sample(nrow(train.set4))
  seed <- seed+1
  
  #split the train to HOULDOUT and TUNE
  folds <- cut(seq(1,nrow(train.set4)),breaks=k,labels=FALSE)
  
  
  gcv_store = c()
  store_testerror <- c()
  opt_col =rep(list(c()), nrow(par.test))
  for (w in 1: nrow(par.test)) {
    print(paste0("Parameter Iteration: ",w,"********"))
    par = as.numeric(par.test[w,])
    # Find the baseline values (prediction error/RMSE)
    ave0.rmse <- c()
    resid.tune <- rep(list(c()), k)
    resid.hold <- rep(list(c()), k)
    fitted.tune <- rep(list(c()), k)
    fitted.hold <- rep(list(c()), k)
    for(e in 1:k){
      #Segement your data by fold using the which() function 
      holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
      TUNE <- train.set4[-holdoutIndexes, ]
      HOLDOUT <- train.set4[holdoutIndexes, ]
      
      mod0 <- lm(y~1, data = TUNE)
      pred0 <- predict(mod0, HOLDOUT)
      
      ave0.rmse <- c(ave0.rmse, sqrt(mean((HOLDOUT$y - pred0)^2)))
      
      resid.tune[[e]] <- TUNE$y
      resid.hold[[e]] <- HOLDOUT$y
    }
    rmse0 <- mean(ave0.rmse)
    
    # Iterative subspace search
    iter <- 0
    all.col <- c()
    repeat {
      iter <- iter + 1
      # randomly generate a subspace with three variables
      set.seed(seed);three.col <- sample(1:ncol(train.set4[,-c(42)]),no.col)
      seed <- seed+1
      
      var.tmp <- names(train.set4[, three.col])
      tmp.form <- as.formula(paste0("res~",paste(var.tmp, collapse = "+")))
      
      #calculate the prediction error (RMSE) of 5- fold TUNE/HOLDOUT
      rmse <- c()
      for(e in 1:k){
        res <- resid.tune[[e]]
        
        holdoutIndexes <- id.crit[which(folds==e,arr.ind=TRUE)]
        TUNE <- train.set4[-holdoutIndexes, ]
        HOLDOUT <- train.set4[holdoutIndexes, ]
        # SVM Model
        mod.sel <- svm(tmp.form, data = TUNE, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
        fitted.tune[[e]] <- predict(mod.sel, TUNE)
        fitted.hold[[e]] <- predict(mod.sel, HOLDOUT)
        rmse <- c(rmse, sqrt(mean((resid.hold[[e]] - fitted.hold[[e]])^2)))
      }
      avg.rmse <- mean(rmse)
      
      #selection criterion
      if(avg.rmse <= rmse0 |iter > 10000){
        #termination criterion (iter > 10000 is the hard threshold)
        if(((rmse0-avg.rmse)/rmse0 < .00001) | (iter > 10000)){
          if (iter > 10000){print("iteration is larger than 10000")}
          resid.tr <- train.set4$y
          resid.ts <- test.set4$y
          s_store = c()
          # build models with train data (add one subspace at a time and update the response to the residual, boosting)
          for(i in 1:round(length(all.col)/no.col)){
            form <- as.formula(paste0("resid.tr~",paste(all.col[((i-1)*no.col+1):(i*no.col)], collapse = "+")))
            mod <- svm(form, data = train.set4, kernel = "polynomial", degree = par[1], epsilon= par[2], cost= par[3])
            
            ##### calculate trace(s),inner product
            polykernel <- polydot(degree = par[1], scale = 1/3, offset = 0)
            TRAIN <- train.set4[,all.col[((i-1)*no.col+1):(i*no.col)]]
            
            matrixx <- matrix(as.numeric(unlist(TRAIN)),nrow=nrow(TRAIN), ncol= ncol(TRAIN))
            inner_product <- kernelMatrix(polykernel, matrixx)
            alpha_y <- rep(0, length(resid.tr))
            alpha_y[mod$index] <- as.numeric(mod$coefs)
            s <- inner_product %*% diag(alpha_y/resid.tr)
            trace_s <- tr(s)
            
            # store the trace of each matrix
            s_store <- c(s_store, trace_s)
            
            #update the response to residual, boosting
            resid.tr <- resid.tr - predict(mod, train.set4)
            #calculate test error
            resid.ts <- resid.ts - predict(mod, test.set4)
          }
          test.error <- sqrt(mean((resid.ts)^2))
          s_sum <- sum(s_store)
          
          # GCV calculation
          gcv_cal <- mean((resid.tr/(1-s_sum/nrow(train.set4)))**2)
          gcv_store <- c(gcv_store, gcv_cal)
          
          store_testerror = c(store_testerror, test.error)

          #store the critical subspaces (columns)
          opt_col[[w]] <- all.col
          
          #print(iter);print(rmse0);print(avg.rmse);print(test.error);print(gcv_cal)#;print(opt_col)
          break
        } else if ((rmse0-avg.rmse)/rmse0 > .01) {
          ###### store the critical subspaces; update the current prediction error and the response of TUNE/HOLDOUT##########
          all.col <- c(all.col, var.tmp)
          rmse0 <- avg.rmse
          resid.tune <- lapply(1:k, function(id) resid.tune[[id]] - fitted.tune[[id]])
          resid.hold <- lapply(1:k, function(id) resid.hold[[id]] - fitted.hold[[id]])
          #print(iter)#;print(all.col);print(rmse0)
        }
      }
    }
    
  }
  # find the optimal hyperparameters by minimum GCV and the corresponding critical subspace and test error
  mini = which(gcv_store == min(gcv_store))
  print(mini);print(gcv_store[mini])
  print(par.test[mini,])
  onetesterror = store_testerror[mini]
  all.testerr <- c(all.testerr, onetesterror)
  crit.col[[m]] <- opt_col[[mini]]
  
  toc()
  
}
# calculate the average prediction error and standard deviation
avg.testerr <- mean(all.testerr)
print(all.testerr)
print(avg.testerr)
print(sd(all.testerr))
print(crit.col)
toc()