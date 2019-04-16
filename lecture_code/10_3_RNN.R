#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: RNN
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

work_folder <- "~/Dropbox/stat430-dl"

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32)

summary(model)

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE)

summary(model)

# all intermediate layers need to return full sequences:
model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = 10000, output_dim = 32) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>% 
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32, return_sequences = TRUE) %>%
  layer_simple_rnn(units = 32)  # This last layer only returns the last outputs.
summary(model)

#######################
# Simple RNN for IMDB #
#######################
max_features <- 10000  # Number of words to consider as features
maxlen <- 500  # Cuts off texts after this many words (among the max_features most common words)
batch_size <- 32

cat("Loading data...\n")
imdb <- dataset_imdb(num_words = max_features)
c(c(input_train, y_train), c(input_test, y_test)) %<-% imdb 
cat(length(input_train), "train sequences\n")
cat(length(input_test), "test sequences")

cat("Pad sequences (samples x time)\n")
input_train <- pad_sequences(input_train, maxlen = maxlen, truncating = "post")
input_test <- pad_sequences(input_test, maxlen = maxlen, truncating = "post")
cat("input_train shape:", dim(input_train), "\n")
cat("input_test shape:", dim(input_test), "\n")

k_clear_session()

# Let's train a simple recurrent network using a `layer_embedding()` and `layer_simple_rnn()`.
model <- keras_model_sequential() %>%
  layer_embedding(input_dim = max_features, output_dim = 32) %>%
  layer_simple_rnn(units = 32) %>%
  layer_dense(units = 1, activation = "sigmoid")
summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2
)
plot(history)
# ~ 0.84 

# with a dense layer, we got a test accuracy of 88%

####################################
# A concrete LSTM example in Keras # ----- DO NOT RUN, TOO SLOW
####################################

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy", 
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 5,
  batch_size = 128,
  validation_split = 0.2
)
plot(history)

############################
# fast LSTM, cuDNN version #
############################

k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_cudnn_lstm(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy", 
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)
plot(history)

####################################
# A concrete GRU example in Keras #
####################################
k_clear_session()

model <- keras_model_sequential() %>% 
  layer_embedding(input_dim = max_features, output_dim = 32) %>% 
  layer_cudnn_gru(units = 32) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop", 
  loss = "binary_crossentropy", 
  metrics = c("acc")
)

history <- model %>% fit(
  input_train, y_train,
  epochs = 10,
  batch_size = 128,
  validation_split = 0.2
)
plot(history)

