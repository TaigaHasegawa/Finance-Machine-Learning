#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: advanced use of RNN
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

work_folder <- "~/Dropbox/stat430-dl/"
if(1==2) # data is already downloaded
{
  dir.create(file.path(work_folder, "jena_climate"), recursive = TRUE)
  download.file(
    "https://s3.amazonaws.com/keras-datasets/jena_climate_2009_2016.csv.zip",
    file.path(work_folder, "jena_climate/jena_climate_2009_2016.csv.zip")
  )
  unzip(file.path(work_folder, "jena_climate/jena_climate_2009_2016.csv.zip"),exdir = file.path(work_folder, "jena_climate"))
}
library(tibble)
library(readr)

data_dir <- file.path(work_folder, "jena_climate")
fname <- file.path(data_dir, "jena_climate_2009_2016.csv")
data <- read_csv(fname)
glimpse(data)

library(ggplot2)
ggplot(data, aes(x = 1:nrow(data), y = `T (degC)`)) + geom_line()

# 144 data points per day:
# daily periodicity
ggplot(data[1:1440,], aes(x = 1:1440, y = `T (degC)`)) + geom_line()

####################################################
# is this timeseries predictable at a daily scale? #
####################################################

## Preparing the data

# each timeseries in the data is on a different scale 
# (e.g. temperature is typically between -20 and +30, 
# but pressure, measured in mbar, is around 1000). 
# So we will normalize each timeseries independently so 
# that they all take small values on a similar scale.

data <- data.matrix(data[,-1])
train_data <- data[1:200000,]
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
data <- scale(data, center = mean, scale = std)

generator <- function(data, lookback, delay, min_index, max_index, 
                      shuffle = FALSE, batch_size = 128, step = 6) 
{
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size-1, max_index))
      # The `i` variable contains the state that tracks next window of data to return, so it is updated using superassignment (e.g. `i <<- i + length(rows)`)
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]] - 1, 
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

val_gen <- generator(
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
test_steps <- (nrow(data) - 300001 - lookback) / batch_size

##################################################
## A common sense, non-machine learning baseline #
##################################################

# use sample mean as a baseline
batch_maes <- c()
for (step in 1:val_steps) {
  c(samples, targets) %<-% val_gen()
  preds <- samples[,dim(samples)[[2]],2]
  mae <- mean(abs(preds - targets))
  batch_maes <- c(batch_maes, mae)
}
print(mean(batch_maes))
# ~ 0.279 

######################################
## A basic machine learning approach #
######################################

# try simple, cheap machine-learning models (such as small, densely connected networks) before looking into complicated and computationally expensive models such as RNNs. This is the best way to make sure any further complexity you throw at the problem is legitimate and delivers real benefits.
# use MAE as the loss, the exact same data and with the exact same metric with the common-sense approach, the results will be directly comparable.
step <- 6
model <- keras_model_sequential() %>% 
  layer_flatten(input_shape = c(lookback / step, dim(data)[-1])) %>% 
  layer_dense(units = 32, activation = "relu") %>% 
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

# When looking for a solution with a space of complicated models, the simple 
# well-performing baseline might be unlearnable, even if it's technically 
# part of the hypothesis space. That is a pretty significant limitation of 
# machine learning in general: unless the learning algorithm is hard-coded to 
# look for a specific kind of simple model, parameter learning can sometimes 
# fail to find a simple solution to a simple problem.
                                                                                                     
################################
## A first recurrent baseline ##
################################
                                                                                                     
# Fully-connected approach removed the notion of time from the input data
# Let us instead look at our data as what it is: a sequence, where causality and order matter
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
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

# Our new validation MAE of ~0.265 (before we start significantly overfitting) translates to a mean absolute error of 2.35˚C after de-normalization. That's a solid gain on our initial error of 2.57˚C, but we probably still have a bit of margin for improvement.

################################################## 
## Using recurrent dropout to fight overfitting ##
##################################################
                                                                                                     
# It has long been known that applying dropout before a recurrent layer hinders learning rather than helping with regularization.
# The proper way to use dropout with a recurrent network: the same dropout mask (the same pattern of dropped units) should be applied at every timestep, instead of a dropout mask that would vary randomly from timestep to timestep.
# in order to regularize the representations formed by the recurrent gates of layers such as GRU and LSTM, a temporally constant dropout mask should be applied to the inner recurrent activations of the layer
# using the same dropout mask at every timestep allows the network to properly propagate its learning error through time; a temporally random dropout mask would instead disrupt this error signal and be harmful to the learning process
# every recurrent layer in Keras has two dropout-related arguments: `dropout`, a float specifying the dropout rate for input units of the layer, and `recurrent_dropout`, specifying the dropout rate of the recurrent units
# Because networks being regularized with dropout always take longer to fully converge, we train our network for twice as many epochs.
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, dropout = 0.2, recurrent_dropout = 0.2,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)

###############################
## Stacking recurrent layers ##
###############################
                                                                                                     
# Since we are no longer overfitting yet we seem to have hit a performance bottleneck, we should start considering increasing the capacity of our network. 
# it is a generally a good idea to increase the capacity of your network until overfitting becomes your primary obstacle
# Recurrent layer stacking is a classic way to build more powerful recurrent networks: for instance, what currently powers the Google translate algorithm is a stack of seven large LSTM layers -- that's huge
model <- keras_model_sequential() %>% 
  layer_gru(units = 32, 
            dropout = 0.1, 
            recurrent_dropout = 0.5,
            return_sequences = TRUE,
            input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_gru(units = 64, activation = "relu",
            dropout = 0.1,
            recurrent_dropout = 0.5) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)
plot(history)
             
# Since we are still not overfitting too badly, we could safely increase the size of our layers, in quest for a bit of validation loss improvement. This does have a non-negligible computational cost, though. 
# Since adding a layer did not help us by a significant factor, we may be seeing diminishing returns to increasing network capacity at this point
                                                                                                     
#############################
## Using bidirectional RNNs #
#############################
# A bidirectional RNN is common RNN variant which can offer higher performance than a regular RNN on certain tasks
# It is frequently used in natural language processing -- you could call it the Swiss army knife of deep learning for NLP
                                                                                                     
# A bidirectional RNN exploits the order-sensitivity of RNNs: it simply consists of two regular RNNs, such as the GRU or LSTM layers that you are already familiar with, each processing input sequence in one direction (chronologically and antichronologically), then merging their representations
# By processing a sequence both way, a bidirectional RNN is able to catch patterns that may have been overlooked by a one-direction RNN
# All you need to do is write a variant of the data generator where the input sequences are reverted along the time dimension (replace the last line with `list(samples[,ncol(samples):1,], targets)`). 
                                                                                                     
reverse_order_generator <- function( data, lookback, delay, min_index, max_index,
                                     shuffle = FALSE, batch_size = 128, step = 6) {
  if (is.null(max_index))
    max_index <- nrow(data) - delay - 1
  i <- min_index + lookback
  function() {
    if (shuffle) {
      rows <- sample(c((min_index+lookback):max_index), size = batch_size)
    } else {
      if (i + batch_size >= max_index)
        i <<- min_index + lookback
      rows <- c(i:min(i+batch_size, max_index))
      i <<- i + length(rows)
    }
    
    samples <- array(0, dim = c(length(rows), 
                                lookback / step,
                                dim(data)[[-1]]))
    targets <- array(0, dim = c(length(rows)))
    
    for (j in 1:length(rows)) {
      indices <- seq(rows[[j]] - lookback, rows[[j]], 
                     length.out = dim(samples)[[2]])
      samples[j,,] <- data[indices,]
      targets[[j]] <- data[rows[[j]] + delay,2]
    }            
    
    list(samples[,ncol(samples):1,], targets)
  }
}

train_gen_reverse <- reverse_order_generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 1,
  max_index = 200000,
  shuffle = TRUE,
  step = step, 
  batch_size = batch_size
)

val_gen_reverse = reverse_order_generator(
  data,
  lookback = lookback,
  delay = delay,
  min_index = 200001,
  max_index = 300000,
  step = step,
  batch_size = batch_size
)

model <- keras_model_sequential() %>% 
  layer_gru(units = 32, input_shape = list(NULL, dim(data)[[-1]])) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen_reverse,
  steps_per_epoch = 500,
  epochs = 20,
  validation_data = val_gen_reverse,
  validation_steps = val_steps
)
plot(history)
             
# So the reversed-order GRU strongly underperforms even the common-sense baseline, indicating that the in our case chronological processing is very important to the success of our approach. 
# the underlying GRU layer will typically be better at remembering the recent past than the distant past, and naturally the more recent weather data points are more predictive than older data points in our problem


#############################
# bidirectional RNNs - IMDB #
#############################

max_features <- 10000  # Number of words to consider as features
maxlen <- 500          # Cut texts after this number of words 
# (among top max_features most common words)

# Load data
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

# Reverse sequences
x_train <- lapply(x_train, rev) 
x_test <- lapply(x_test, rev) 

# Pad sequences
x_train <- pad_sequences(x_train, maxlen = maxlen)
x_test <- pad_sequences(x_test, maxlen = maxlen)

model <- keras_model_sequential() %>% 
layer_embedding(input_dim = max_features, output_dim = 128) %>% 
layer_lstm(units = 32) %>% 
layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
optimizer = "rmsprop",
loss = "binary_crossentropy",
metrics = c("acc")
)

history <- model %>% fit(
x_train, y_train,
epochs = 10,
batch_size = 128,
validation_split = 0.2
)

# We get near-identical performance as the chronological-order LSTM we tried in the previous section.

# In machine learning, representations that are *different* yet *useful* are always worth exploiting, and the more they differ the better: they offer a new angle from which to look at your data, capturing aspects of the data that were missed by other approaches, and thus they can allow to boost performance on a task. This is the intuition behind "ensembling", a concept that we will introduce in the next chapter.

k_clear_session()
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  bidirectional(
    layer_lstm(units = 32)
  ) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)

model <- keras_model_sequential() %>% 
  bidirectional(
    layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
  ) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)

model <- keras_model_sequential() %>% 
  bidirectional(
    layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
  ) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)


# try the same approach on the weather prediction task:

model <- keras_model_sequential() %>% 
  bidirectional(
    layer_gru(units = 32), input_shape = list(NULL, dim(data)[[-1]])
  ) %>% 
  layer_dense(units = 1)

model %>% compile(
  optimizer = optimizer_rmsprop(),
  loss = "mae"
)

history <- model %>% fit_generator(
  train_gen,
  steps_per_epoch = 500,
  epochs = 40,
  validation_data = val_gen,
  validation_steps = val_steps
)
