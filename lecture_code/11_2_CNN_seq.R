#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: 1D CNN for sequence data
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

work_folder <- "~/Dropbox/stat430-dl"

#####################################################################
## a simple two-layer 1D convnet for IMDB sentiment-classification ##
#####################################################################
library(keras)
max_features <- 10000
max_len <- 500

cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb 
cat(length(x_train), "train sequences\n")
cat(length(x_test), "test sequences")

cat("Pad sequences (samples x time)\n")
x_train <- pad_sequences(x_train, maxlen = max_len)
x_test <- pad_sequences(x_test, maxlen = max_len)
cat("x_train shape:", dim(x_train), "\n")
cat("x_test shape:", dim(x_test), "\n")

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 128,
                  input_length = max_len) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 5) %>% 
  layer_conv_1d(filters = 32, kernel_size = 7, activation = "relu") %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 1)

summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

plot(history)

############################
## CNN for long sequences ##
############################
library(tibble)
library(readr)

data_dir <- file.path(work_folder, "jena_climate") # temparature data
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
data <- data.matrix(data[,-1])

train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

generator <- function(data, lookback, delay, min_index, max_index,
                      shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {                         # keep the sequential order
      if (i + batch_size >= max_index)
        i <<- min_index + lookback   # superassignment to store the status of i
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), lookback/step, dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples, targets)
  }
}

lookback <- 1440 # observations will go back 10 days (6*24*10)
step <- 6 # observations will be sampled at one data point per hour
delay <- 144 # targets will be 24 hours in the future
batch_size <- 128

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen = generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step,
  batch_size = batch_size
)

# This is how many steps to draw from `val_gen`
# in order to see the whole validation set:
val_steps <- (300000 - 200001 - lookback) / batch_size

# This is how many steps to draw from `test_gen`
# in order to see the whole test set:
# test_steps <- (nrow(data) - 300001 - lookback) / batch_size

model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
  layer_global_max_pooling_1d() %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)
# ~ val MAE 0.40s: not even beat the common-sense baseline using the small convnet

#######################################################
## Combining CNNs and RNNs to process long sequences ##
#######################################################
# This was previously set to 6 (one point per hour), now 3 (one point per 30 min)
# So twice the length now (with the same lookback)
step <- 3 
lookback <- 720  # Unchanged
delay <- 144  # Unchanged

train_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step
)

val_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step
)

test_gen <- generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 300001,
  max_index = NULL,
  step = step
)

val_steps <- (300000 - 200001 - lookback) / 128
# test_steps <- (nrow(data) - 300001 - lookback) / 128

# This is the model, starting with two `layer_conv_1d()` and following up with a `layer_gru()`:
model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu") %>% 
  layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)
# ~ val MAE 0.27

# Judging from the validation loss, this setup is not quite as good as the regularized GRU alone, but it's significantly faster. It is looking at twice more data, which in this case doesn't appear to be hugely helpful, but may be important for other datasets.

###########################################################################
## Combining CNNs (+ dilated kernals) and RNNs to process long sequences ##
###########################################################################
# One useful and important concept that we won't cover in these pages is that of 1D convolution with dilated kernels.
model <- keras_model_sequential() %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                input_shape = list(NULL, dim(data)[[-1]]),
                dilation_rate = 2) %>%  # used dilated kernels, default = 1 
  layer_max_pooling_1d(pool_size = 3) %>% 
  layer_conv_1d(filters = 32, kernel_size = 5, activation = "relu",
                dilation_rate = 2) %>%  # used dilated kernels
  layer_gru(units = 32, dropout = 0.1, recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

summary(model)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)
# val MAE slightly < 0.27
