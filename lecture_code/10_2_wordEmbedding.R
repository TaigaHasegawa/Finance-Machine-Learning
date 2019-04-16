#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: words embedding
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

work_folder <- "~/STAT430"

#####################################################
## Learning word embeddings with an embedding layer #
#####################################################

# The embedding layer takes at least two arguments:
# the number of possible tokens, here 1000 (1 + maximum word index),
# and the dimensionality of the embeddings, here 64.
embedding_layer <- layer_embedding(input_dim = 1000, output_dim = 64) 

# Number of words to consider as features
max_features <- 10000

# Cut texts after this number of words (among top max_features most common words)
maxlen <- 20

# Load the data as lists of integers.
imdb <- dataset_imdb(num_words = max_features)
c(c(x_train, y_train), c(x_test, y_test)) %<-% imdb

# This turns our lists of integers into a 2D integer tensor of shape `(samples, maxlen)`
x_train <- pad_sequences(x_train, maxlen = maxlen, truncating = "post") # truncating default is "pre"
x_test <- pad_sequences(x_test, maxlen = maxlen, truncating = "post")

model <- keras_model_sequential() %>% 
  # We specify the maximum input length to our Embedding layer
  # so we can later flatten the embedded inputs
  layer_embedding(input_dim = 10000, output_dim = 8, input_length = maxlen) %>% 
  # We flatten the 3D tensor of embeddings 
  # into a 2D tensor of shape `(samples, maxlen * 8)`
  layer_flatten() %>% 
  # We add the classifier on top
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy",
  metrics = c("acc")
)

history <- model %>% fit(
  x_train, y_train,
  epochs = 10,
  batch_size = 32,
  validation_split = 0.2
)
# val acc ~ 75%
# the  model treats each word in the input sequence separately, without considering inter-word relationships and structure sentence
