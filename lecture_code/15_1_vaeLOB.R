# Different networks for predicting price movement based on limit order book
# Larryhua.com/teaching

# Model A: 2D CNN (feature dimension = 1)
# Model B: RNN (feature dimension = 20)
# Model C: 1D CNN + RNN (feature dimension = 20)
# Model D: HRNN (feature dimension = 1)
# Model E: Depthwise Separable Convolution (1D) + RNN (feature dimension = 20)
# Model F: VAE + RNN (feature dimension = 20)

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
dat1$direction <- 1 # stable, excluded label
dat1$direction[chg > a] <- 2 # increase
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
# Model F # 
###########
# For Model F, we consider 3 different classes to simplify the data generator

sampling_generator <- function(X_data, batch_size, w) # <------------ we do not need Y_data for VAE!
{
  function()
  {
    rows <- sample(1:(nrow(X_data)-w+1), batch_size, replace = TRUE)
    X <- NULL
    X_list <- list()
    j <- 1
    for(i in rows)
    {
      X_list[[j]] <- X_data[(i:(i+w-1)),]
      j <- j+1
    }
    X <- abind::abind(X_list, along = 0)  # along = 0 to add the sample axis
    Y <- rep(1,batch_size) # arbitrary
    list(X, Y)
  }
}

# callbacks
# note that VAE does not have val_acc # <--------------------

checkPoint <- callback_model_checkpoint(filepath = file.path(work_folder, "lob_F.h5"), monitor = "val_loss", save_best_only = TRUE, save_weights_only = T) 
# due to some issues with Keras, when layer_lambda() is used, keras cannot save models using save_model_hdf5() but the weights can be saved
# see here: https://github.com/rstudio/keras/issues/90

earlyStop <- callback_early_stopping(monitor = "val_loss", patience = 4)
reduceLr <- callback_reduce_lr_on_plateau(monitor = "val_loss", factor = 0.1, patience = 3)
logger <- callback_csv_logger(file.path(work_folder, "logger-lob_F.csv"))

# Parameters --------------------------------------------------------------
batch_size <- 24
epochs <- 40
original_dim <- 20
latent_dim <- 2
lr <- 1e-4

# flags used to manage different runs ----------------------------------------
# this is just a simple example, a lot more parameters can be tuned
FLAGS <- flags(
  flag_integer("intermediate_dim", 16), # default values, can be passed through 
  flag_numeric("epsilon_std", 1.0)
)

# Model definition --------------------------------------------------------
x <- layer_input(shape = list(NULL, original_dim))
h <- layer_cudnn_gru(x, units=FLAGS$intermediate_dim)
z_mean <- layer_dense(h, latent_dim) # no activation, use linear transformations
z_log_var <- layer_dense(h, latent_dim)

sampling <- function(arg){
  z_mean <- arg[, 1:(latent_dim)]
  z_log_var <- arg[, (latent_dim + 1):(2 * latent_dim)]
  
  epsilon <- k_random_normal(
    shape = c(k_shape(z_mean)[[1]]), 
    mean=0.,
    stddev=FLAGS$epsilon_std
  )
  z_mean + k_exp(z_log_var/2)*epsilon
}

z <- layer_concatenate(list(z_mean, z_log_var)) %>% 
  layer_lambda(sampling)

# we instantiate these layers separately so as to reuse them later
decoder_h <- layer_dense(units = FLAGS$intermediate_dim, activation = "relu")
decoder_mean <- layer_cudnn_gru(units = original_dim, input_shape = c(NULL,FLAGS$intermediate_dim))

wz <- function(z){k_repeat(z,w)} # note that we need k_repeat() to create w identical z tensors

layer_wz <- layer_lambda(z,wz) # then we need to define it to be layer 
# in order to be put into decoder_h() in the following

h_decoded <- decoder_h(layer_wz) 
x_decoded_mean <- decoder_mean(h_decoded)

# end-to-end autoencoder (sequence-to-sequence)
vae <- keras_model(x, x_decoded_mean) # <-------------------------------------- vae

# encoder, from inputs to latent space
encoder <- keras_model(x, z_mean) # <------------------------------------------- encoder

# generator, from latent space to reconstructed inputs
decoder_input <- layer_input(shape = list(w,latent_dim))
h_decoded_2 <- decoder_h(decoder_input)

# x_decoded_mean_2 <- decoder_mean(layer_wh)
x_decoded_mean_2 <- decoder_mean(h_decoded_2)
generator <- keras_model(decoder_input, x_decoded_mean_2) # <-------------------- decoder

vae_loss <- function(x, x_decoded_mean){
  xent_loss <- (original_dim/1.0)*loss_binary_crossentropy(x, x_decoded_mean)
  kl_loss <- -0.5*k_mean(1 + z_log_var - k_square(z_mean) - k_exp(z_log_var), axis = -1L)
  xent_loss + kl_loss
}

vae %>% compile(
  optimizer = optimizer_rmsprop(lr = lr),
  loss = vae_loss)

# Model training ----------------------------------------------------------
vae %>% fit_generator(sampling_generator(X_data_train, batch_size = batch_size, w=w),
                      steps_per_epoch = 100, epochs = epochs,
                      callbacks = list(checkPoint, earlyStop, reduceLr, logger),
                      validation_data = sampling_generator(X_data_val, batch_size = batch_size, w=w),
                      validation_steps = 100)

# Visualizations ----------------------------------------------------------
library(ggplot2)
library(dplyr)

# predict_generator needs another sampling_generator to go through all the test data. <----------------
sampling_generator_test <- function(X_data, batch_size, w)
{
  i <- 1
  function()
  {
    max_index <- nrow(X_data) - w + 1
    
    if(i > max_index) i <<- 1
    
    rows <- c(i:min(i+batch_size-1, max_index))
    i <<- i + length(rows)
    
    X <- NULL
    X_list <- list()
    
    j <- 1
    for(k in rows)
    {
      X_list[[j]] <- X_data[(k:(k+w-1)),]
      j <- j+1
    }
    X <- abind::abind(X_list, along = 0)  # along = 0 to add the sample axis
    list(X)
  }
}

max_steps <- floor((nrow(X_data_test)-w+1) / batch_size)

# load_model_weights_hdf5(vae, file.path(work_folder, "lob_F.h5"), by_name = T)
load_model_weights_hdf5(encoder, file.path(work_folder, "lob_F.h5"), by_name = T)

x_test_encoded <- predict_generator(encoder, sampling_generator_test(X_data_test, batch_size, w=w),
                                    steps = max_steps)

x_test_encoded %>%
  as_data_frame() %>% 
  mutate(class = as.factor(Y_data_test[1:(max_steps*batch_size)])) %>%
  ggplot(aes(x = V1, y = V2, colour = class)) + geom_point()

