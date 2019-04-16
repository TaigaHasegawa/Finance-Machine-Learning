# Differen networks for predicting price movement based on limit order book
# Larryhua.com/teaching

rm(list = setdiff(ls(), lsf.str()))
library(data.table)
library(keras)

work_folder <- "~/Dropbox/stat430-dl"
# load limit order book
dat <- fread("gunzip -c ~/Dropbox/stat430-dl/datasets/MSFT.zip")
dat <- apply(dat, 2, as.numeric)
for(i in seq(1,40, 2)) dat[,i] <- dat[,i] / 10000
dat <- data.frame(dat)
names(dat) <- c("AP1", "AV1", "BP1", "BV1", "AP2", "AV2", "BP2", "BV2", "AP3", "AV3", "BP3", "BV3", "AP4", "AV4", "BP4", "BV4", "AP5", "AV5", "BP5", "BV5", 
                "AP6", "AV6", "BP6", "BV6", "AP7", "AV7", "BP7", "BV7", "AP8", "AV8", "BP8", "BV8", "AP9", "AV9", "BP9", "BV9", "AP10", "AV10", "BP10", "BV10")

head(dat)
dim(dat)

########################
# label price movement #
########################
## mid price
dat$mPrice <- (dat$AP1 + dat$BP1)/2

## use previous w mPrice for previous price level and use future w mPrice for future price level to detect price movements
w <- 100
avgMprice <- c(rep(NA, w-1), zoo::rollmean(dat$mPrice, k=w, align="left"))
dat1 <- dat[-c(((nrow(dat)-w+1):nrow(dat))), ] # remove last w observations
dat1$preMP <- avgMprice[1:nrow(dat1)]
dat1$postMP <- avgMprice[(w+1):(nrow(dat1)+w)]
dat1 <- dat1[-(1:(w-1)),] # remove first (w-1) observations
head(dat1)

## a: threshold of price change percentages for labeling the direction
a <- 0.00005
chg <- dat1$postMP / dat1$preMP - 1

## direction of price movement
dat1$direction <- -1 # stable, excluded label
dat1$direction[chg > a] <- 1 # increase
dat1$direction[chg < -a] <- 0 # decrease
table(dat1$direction)

head(dat1) # note that the first (w-1) observations do not have labels

## select label and volumes as features
col_used <- c("direction", 
              "AV10", "AV9", "AV8", "AV7", "AV6", "AV5", "AV4", "AV3", "AV2", "AV1",
              "BV1", "BV2", "BV3", "BV4", "BV5", "BV6", "BV7", "BV8", "BV9", "BV10")
dat1 <- dat1[, names(dat1) %in% col_used]
dat1 <- dat1[, sapply(col_used, function(x){which(x==names(dat1))})]
head(dat1)

# train | validation | test splits: 3|1|1
data_train <- dat1[(1:floor(nrow(dat1)/5*3)),]
data_val <- dat1[((floor(nrow(dat1)/5*3)+1):floor(nrow(dat1)/5*4)),]
data_test <- dat1[((floor(nrow(dat1)/5*4)+1):nrow(dat1)),]
dim(data_train); dim(data_val); dim(data_test); dim(dat1)
table(data_train$direction)
table(data_val$direction)
table(data_test$direction)

col_volume <- (2:21)
up_down_train <- (data_train$direction != -1)
me_volume_train <- mean(as.matrix(data_train[up_down_train, col_volume]))
sd_volume_train <- sd(as.matrix(data_train[up_down_train, col_volume]))

# rescale train data
for(i in col_volume) data_train[,i] <- scale(data_train[,i], center = me_volume_train, scale = sd_volume_train)
X_data_train <- data_train[, col_volume]
Y_data_train <- data_train$direction

# rescale validation data (using train mean and sd)
for(j in col_volume) data_val[,j] <- scale(data_val[,j], center = me_volume_train, scale = sd_volume_train)
X_data_val <- data_val[, col_volume]
Y_data_val <- data_val$direction

# rescale test data (using train mean and sd)
for(k in col_volume) data_test[,k] <- scale(data_test[,k], center = me_volume_train, scale = sd_volume_train)
X_data_test <- data_test[, col_volume]
Y_data_test <- data_test$direction

###########
# Model A #
###########
if(1==2)
{
  checkPoint <- callback_model_checkpoint(filepath = file.path(work_folder, "lob_A.h5"),
                                          monitor = "val_acc", save_best_only = TRUE)
  
  reduceLr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.1, patience = 3)
  logger <- callback_csv_logger(file.path(work_folder, "logger-lob_A.csv"))

  sampling_generator <- function(X_data, Y_data, batch_size, w)
  {
    function()
    {
      rows_with_up_down <- w:nrow(X_data)
      rows_with_up_down <- intersect(rows_with_up_down, which( Y_data %in% c(0,1)))  # only use labels 0 and 1
      
      rows <- sample( rows_with_up_down, batch_size, replace = TRUE )
      
      Y <- X <- NULL
      Xlist <- list()
      for(i in rows)
      {
        Xlist[[i]] <- X_data[(i-w+1):i,] 
        Y <- c(Y, Y_data[i])
      }
      X <- array(abind::abind(Xlist, along = 0), c(batch_size, w, ncol(X_data), 1)) # add one axis of dimension of 1
      list(X, Y)
    }
  }
  
  batch_size = 24
  epochs = 20
  
  k_clear_session()

  model_A <- keras_model_sequential() %>% 
    layer_conv_2d(filters = 6, kernel_size = c(3, 3), activation = "relu", input_shape = c(100, 20, 1)) %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>% 
    layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
    layer_flatten() %>% 
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l1(0.001)) %>% 
    layer_dropout(rate = 0.5) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  summary(model_A)
  
  model_A %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c("accuracy") 
  )

  his_A <- model_A %>% fit_generator(sampling_generator(X_data_train, Y_data_train, 
                                                        batch_size = batch_size, w=w),
                                 steps_per_epoch = 100, epochs = epochs,
                                 callbacks = list(logger, checkPoint, reduceLr),
                                 validation_data = sampling_generator(X_data_val, 
                                                                      Y_data_val, batch_size = batch_size, w=w),
                                 validation_steps = 100)
  plot(his_A)
  
  fitted <- load_model_hdf5(file.path(work_folder, "lob_A.h5"))
  
  results <- fitted %>% evaluate_generator(sampling_generator(X_data_test, Y_data_test, 
                                                              batch_size = batch_size, w=w), 
                                           steps = 1000)
  results
  
  fmlr::acc_lucky( table(Y_data_train)[c(2,3)], table(Y_data_test)[c(2,3)], results$acc)
  
}


###########
# Model B #
###########
if(1==2)
{
  checkPoint <- callback_model_checkpoint(filepath = file.path(work_folder, "lob_B.h5"),
                                          monitor = "val_acc", save_best_only = TRUE)
  
  reduceLr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.1, patience = 3)
  logger <- callback_csv_logger(file.path(work_folder, "logger-lob_B.csv"))
  
  sampling_generator <- function(X_data, Y_data, batch_size, w)
  {
    function()
    {
      rows_with_up_down <- w:nrow(X_data)
      rows_with_up_down <- intersect(rows_with_up_down, which( Y_data %in% c(0,1)))  # only use labels 0 and 1
      
      rows <- sample( rows_with_up_down, batch_size, replace = TRUE )
      
      Y <- X <- NULL
      Xlist <- list()
      for(i in rows)
      {
        Xlist[[i]] <- X_data[(i-w+1):i,] 
        Y <- c(Y, Y_data[i])
      }
      X <- array(abind::abind(Xlist, along = 0), c(batch_size, w, ncol(X_data)))
      list(X, Y)
    }
  }
  
  k_clear_session()
  model_B <- keras_model_sequential() %>%
    layer_cudnn_gru(unit=8, input_shape = list(NULL, 20), return_sequences = TRUE) %>%
    layer_cudnn_gru(unit=16) %>%
    layer_dense(units = 1, activation = "sigmoid")
  summary(model_B)

  model_B %>% compile(
      loss = "binary_crossentropy",
      optimizer = optimizer_rmsprop(lr = 1e-4),
      metrics = c('accuracy') 
  )
  
  #############
  # run model #
  #############
  batch_size = 24
  epochs = 20
  
  his_B <- model_B %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
                                 steps_per_epoch = 100, epochs = epochs,
                                 callbacks = list(checkPoint, reduceLr, logger),
                                 validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
                                 validation_steps = 100)
  plot(his_B)
  fitted_B <- load_model_hdf5(file.path(work_folder, "lob_B.h5"))
  
  results_B <- fitted_B %>% evaluate_generator(sampling_generator(X_data_test, Y_data_test, batch_size = batch_size, w=w), 
                                               steps = 1000)
  results_B
  fmlr::acc_lucky( table(Y_data_train)[c(2,3)], table(Y_data_test)[c(2,3)], results_B$acc)
  
}

###########
# Model C #
###########
if(1==2)
{
  checkPoint <- callback_model_checkpoint(filepath = file.path(work_folder, "lob_C.h5"),
                                          monitor = "val_acc", save_best_only = TRUE)
  
  reduceLr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.1, patience = 3)
  logger <- callback_csv_logger(file.path(work_folder, "logger-lob_C.csv"))
  
  sampling_generator <- function(X_data, Y_data, batch_size, w)
  {
    function()
    {
      rows_with_up_down <- w:nrow(X_data)
      rows_with_up_down <- intersect(rows_with_up_down, which( Y_data %in% c(0,1)))  # only use labels 0 and 1
      
      rows <- sample( rows_with_up_down, batch_size, replace = TRUE )
      
      Y <- X <- NULL
      Xlist <- list()
      for(i in rows)
      {
        Xlist[[i]] <- X_data[(i-w+1):i,] 
        Y <- c(Y, Y_data[i])
      }
      X <- array(abind::abind(Xlist, along = 0), c(batch_size, w, ncol(X_data)))
      list(X, Y)
    }
  }
  
  k_clear_session()
  model_C <- keras_model_sequential() %>%
    layer_conv_1d(filters = 8, kernel_size = 5, activation = "relu",
                  input_shape = list(NULL, 20)) %>% 
    layer_cudnn_gru(unit=8, return_sequences = TRUE) %>%
    layer_cudnn_gru(unit=16) %>%
    layer_dense(units = 1, activation = "sigmoid")
  
  model_C %>% compile(
    loss = "binary_crossentropy",
    optimizer = optimizer_rmsprop(lr = 1e-4),
    metrics = c('accuracy') 
  )
  
  #############
  # run model #
  #############
  batch_size <- 24
  his_C <- model_C %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
                                 steps_per_epoch = 100, epochs = 20,
                                 callbacks = list(checkPoint, reduceLr, logger),
                                 validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
                                 validation_steps = 100)
  plot(his_C)
  
  fitted_C <- load_model_hdf5(file.path(work_folder, "lob_C.h5"))
  
  results_C <- fitted_C %>% evaluate_generator(sampling_generator(X_data_test, Y_data_test, batch_size = batch_size, w=w), 
                                               steps = 1000)
  results_C
  fmlr::acc_lucky( table(Y_data_train)[c(2,3)], table(Y_data_test)[c(2,3)], results_C$acc)
}


