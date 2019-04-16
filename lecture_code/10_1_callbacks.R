# Using CNN to predict price movement based on limit order book
# We are interested in whether volumes at different depths can be used to predict price movements 
# Larryhua.com/teaching

rm(list = setdiff(ls(), lsf.str()))
library(data.table)
library(keras)

work_folder <- "~/Dropbox/stat430-dl"
# load limit order book
dat <- fread("gunzip -c ~/Dropbox/stat430-dl/datasets/MSFT_2012-06-21_34200000_57600000_orderbook_10.zip")
dat <- apply(dat, 2, as.numeric)
for(i in seq(1,40, 2)) dat[,i] <- dat[,i] / 10000
dat <- data.frame(dat)
names(dat) <- c("AP1", "AV1", "BP1", "BV1", "AP2", "AV2", "BP2", "BV2", "AP3", "AV3", "BP3", "BV3", "AP4", "AV4", "BP4", "BV4", "AP5", "AV5", "BP5", "BV5", 
                "AP6", "AV6", "BP6", "BV6", "AP7", "AV7", "BP7", "BV7", "AP8", "AV8", "BP8", "BV8", "AP9", "AV9", "BP9", "BV9", "AP10", "AV10", "BP10", "BV10")

head(dat)

########################
# label price movement #
########################
## mid price
dat$mPrice <- (dat$AP1 + dat$BP1)/2

## use previous w mPrice for previous price level and use future w mPrice for future price level to detect price movements
w <- 100
library(zoo)
?rollmean
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
me_volume_train <- mean(as.matrix(data_train[up_down_train, col_volume]))  # <----------------- try log
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

#################
# build convnet #
#################
# use w = 100 most recent states of limit order book and 20 volume data (10 levels for each bid/ask)

k_clear_session()

model <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 6, kernel_size = c(3, 3), activation = "relu", input_shape = c(100, 20, 1)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 8, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l1(0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy") 
)

#########################
# create data generator #
#########################
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


#######################
# Do not run in class #
#######################
w = 100
batch_size = 24
epochs = 40
rows_with_up_down_train <- w:nrow(X_data_train)
rows_with_up_down_train <- intersect(rows_with_up_down_train, which( Y_data_train %in% c(0,1)))  # only use labels 0 and 1
sample_size_up_down_train <- length(rows_with_up_down_train)
rows_with_up_down_val <- w:nrow(X_data_val)
rows_with_up_down_val <- intersect(rows_with_up_down_val, which( Y_data_val %in% c(0,1)))  # only use labels 0 and 1
sample_size_up_down_val <- length(rows_with_up_down_val)

# his <- model %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
#                                steps_per_epoch = floor(sample_size_up_down_train / w), epochs = epochs,
#                                validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
#                                validation_steps = floor(sample_size_up_down_val / batch_size))

his <- model %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
                               steps_per_epoch = 100, epochs = epochs,
                               validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
                               validation_steps = 100)
plot(his)

results <- model %>% evaluate_generator(sampling_generator(X_data_test, Y_data_test, batch_size = batch_size, w=w), 
                                        steps = 1000)
results

############################
# Keras callback functions #
############################
# Interrupts training when validation accuracy has stopped improving for more than 4 epoch
earlyStop <- callback_early_stopping(monitor = "val_acc", patience = 4)

# do not overwrite the model file unless val_loss has improved
checkPoint <- callback_model_checkpoint(filepath = file.path(work_folder, "LOB_CNN.h5"),
                                        monitor = "val_acc", save_best_only = TRUE)

# The callback is triggered after the val_acc has stopped improving for 3 epochs
# Then learning rate is reduced to lr*0.1
reduceLr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.1, patience = 3)

# learning rate scheduler
schedule <- function(epoch,lr) (lr)*(0.75^(floor(epoch/2)))
schedulLr <- callback_learning_rate_scheduler(schedule)

# runtime csv loggers
logger <- callback_csv_logger(file.path(work_folder, "logger_lobCNN_callback.csv"))


###################################
# Fit model adding Keras callback #
###################################
his <- model %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
                               steps_per_epoch = 100, epochs = epochs,
                               callbacks = list(logger, earlyStop, checkPoint, reduceLr),
                               validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
                               validation_steps = 100)

str(his)
fitted <- load_model_hdf5(file.path(work_folder, "LOB_CNN.h5"))

results <- fitted %>% evaluate_generator(sampling_generator(X_data_test, Y_data_test, batch_size = batch_size, w=w), 
                                         steps = 1000)
results







#### other models ####
earlyStop <- callback_early_stopping(monitor = "val_acc", patience = 4)
checkPoint <- callback_model_checkpoint(filepath = file.path(work_folder, "LOB_CNN_a.h5"),
                                        monitor = "val_acc", save_best_only = TRUE)
reduceLr <- callback_reduce_lr_on_plateau(monitor = "val_acc", factor = 0.1, patience = 3)
schedule <- function(epoch,lr) (lr)*(0.75^(floor(epoch/2)))
schedulLr <- callback_learning_rate_scheduler(schedule)
logger <- callback_csv_logger(file.path(work_folder, "logger_lobCNN_callback_a.csv"))


model1 <- keras_model_sequential() %>% 
  layer_conv_2d(filters = 12, kernel_size = c(4, 20), activation = "relu", input_shape = c(100, 20, 1)) %>% 
  layer_conv_2d(filters = 12, kernel_size = c(1, 1), activation = "relu") %>% 
  layer_conv_2d(filters = 12, kernel_size = c(4, 1), activation = "relu") %>% 
  layer_flatten() %>% 
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l1(0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

summary(model1)

model1 %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4),
  metrics = c("accuracy") 
)

his1 <- model1 %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
                               steps_per_epoch = 100, epochs = 5,
                               callbacks = list(logger, earlyStop, checkPoint, reduceLr),
                               validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
                               validation_steps = 100)

plot(his1)
fitted1 <- load_model_hdf5(file.path(work_folder, "LOB_CNN_a.h5"))

results1 <- fitted1 %>% evaluate_generator(sampling_generator(X_data_test, Y_data_test, batch_size = batch_size, w=w), 
                                         steps = 1000)
results1



if(1==2)
{
  #######################
  # Customized callback #
  #######################
  library(R6)
  afterBatch <- R6Class("LossHistory", inherit = KerasCallback,
                        public = list(losses = NULL,
                                      on_batch_end = function(batch, logs = list())
                                      {
                                        self$losses <- c(self$losses, logs[["loss"]])
                                      }
                        )
  )
  
  hisAfterBatch <- afterBatch$new()
  
  ########################################
  # Fit model adding customized callback #
  ########################################
  his <- model %>% fit_generator(sampling_generator(X_data_train, Y_data_train, batch_size = batch_size, w=w),
                                 steps_per_epoch = 100, epochs = 5,
                                 callbacks = list(hisAfterBatch),
                                 validation_data = sampling_generator(X_data_val, Y_data_val, batch_size = batch_size, w=w),
                                 validation_steps = 100)
  str(hisAfterBatch)
}
