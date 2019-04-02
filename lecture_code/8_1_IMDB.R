#-------------------------------------------------------------------------------------------------
# A simple example of using Keras to binary classification for IMDB reviews
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

# IMDB #
# Dataset of 25,000 movies reviews from IMDB, labeled by sentiment (positive/negative). 
# Reviews have been preprocessed, and each review is encoded as a sequence of word 
# indexes (integers). For convenience, words are indexed by overall frequency in the dataset, 
# so that for instance the integer "3" encodes the 3rd most frequent word in the data. 
# This allows for quick filtering operations such as: "only consider the top 10,000 most 
# common words, but eliminate the top 20 most common words".

#############
# Read data #
#############
# only keep the top 10,000 most frequently occurring words in the training data. 
# Rare words will be discarded (denoted as 2:unknown)
imdb <- dataset_imdb(num_words = 10000)

imdb$train$x[[1]]
# note that: 
# In Keras, 0, 1, and 2 are reserved for padding, start of sequence, and unknown

## Using the multi-assignment (%<-%) operator
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% imdb

## train_data is a list, and each element is a sequence of numbers for the words in the review
str(train_data[[1]])
str(train_data[[2]])

# Because you’re restricting yourself to the top 10,000 most frequent words,
# no word index will exceed 10,000
max(sapply(train_data, max))

# word_index is a dictionary mapping words to an integer index
word_index <- dataset_imdb_word_index()
str(word_index)

# We reverse it, mapping integer indices to words
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index

# more frequently used words have smaller numbers
reverse_word_index[which(names(reverse_word_index) %in% c("1","2","3","4","5","6") )]
word_index$'i'

# We decode the review; note that our indices were offset by 3
# because 0, 1 and 2 are reserved indices for "padding", "start of sequence", and "unknown".
decoded_review <- sapply(train_data[[1]], function(index) {
  word <- if (index >= 3) reverse_word_index[[as.character(index - 3)]]
  if (!is.null(word)) word else "?"
})

# look at the first review
train_labels[[1]] # it's a positive review
cat(decoded_review) # the review

# Note that this is NOT one-hot encoding!!
vectorize_sequences <- function(sequences, dimension = 10000) {
  # Create an all-zero matrix of shape (len(sequences), dimension)
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    # Sets specific indices of results[i] to 1s
    results[i, sequences[[i]]] <- 1 # each column corresponds to a unique word, and if the word is present the value of the column is 1
                                    # note that here we don't care the order of the words any more!
  results
}

# Our vectorized training data
x_train <- vectorize_sequences(train_data)

# The following is one-hot encoding
# tmp <- to_categorical(train_data[[1]], num_classes=10000)

# Our vectorized test data
x_test <- vectorize_sequences(test_data)
str(x_train[1,])

# converting integers to numeric
y_train <- as.numeric(train_labels)
y_test <- as.numeric(test_labels)

###################
# construct model #
###################
k_clear_session()

model <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
  
#########################################
# compile the model with appropriate    #
# loss function, optimizer, and metrics #
#########################################
model %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) 

val_indices <- 1:10000

x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- y_train[val_indices]
partial_y_train <- y_train[-val_indices]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
plot(history)

results <- model %>% evaluate(x_test, y_test)
results

########################
# Lower capacity model #
########################
k_clear_session()

model_lowerCapacity <- keras_model_sequential() %>% 
  layer_dense(units = 4, activation = "relu", input_shape = c(10000)) %>% 
  layer_dense(units = 4, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")

model_lowerCapacity %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) 

history_lowerCapacity <- model_lowerCapacity %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
plot(history_lowerCapacity)

history$metrics$val_loss
history_lowerCapacity$metrics$val_loss
his <- data.frame(val_loss=history$metrics$val_loss,
                  val_loss_low=history_lowerCapacity$metrics$val_loss,
                  epoch=1:20)


# the smaller network starts overfitting later than the reference network, and its
# performance degrades more slowly once it begins to overfit
library(ggplot2)
ggplot(his, aes(epoch)) + geom_line(aes(y = val_loss, colour = "val_loss"))+ geom_line(aes(y = val_loss_low, colour = "val_loss_low"))

#########################
# weight regularization #
#########################
k_clear_session()

model_weightRegulation <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l2(0.001), input_shape = c(10000)) %>% 
  layer_dense(units = 16, activation = "relu", kernel_regularizer = regularizer_l2(0.001)) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model_weightRegulation %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) 

history_weightRegulation <- model_weightRegulation %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)
plot(history_weightRegulation)

history$metrics$val_loss
history_weightRegulation$metrics$val_loss
his <- data.frame(val_loss=history$metrics$val_loss,
                  val_loss_regulation=history_weightRegulation$metrics$val_loss,
                  epoch=1:20)

# the model with L2 regularization has become much more resistant to overfitting than the
# reference model, even though both models have the same number of parameters
ggplot(his, aes(epoch)) + geom_line(aes(y = val_loss, colour = "val_loss"))+ geom_line(aes(y = val_loss_regulation, colour = "val_loss_regulation"))

##################
# Adding dropout #
##################
k_clear_session()

model_dropout <- keras_model_sequential() %>% 
  layer_dense(units = 16, activation = "relu", input_shape = c(10000)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 16, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

model_dropout %>% compile(
  optimizer = optimizer_rmsprop(lr=0.001),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
) 

history_dropout <- model_dropout %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

history$metrics$val_loss
history_dropout$metrics$val_loss
his <- data.frame(val_loss=history$metrics$val_loss,
                  val_loss_dropout=history_dropout$metrics$val_loss,
                  epoch=1:20)


# the model with dropout become more resistant to overfitting than the reference model
ggplot(his, aes(epoch)) + geom_line(aes(y = val_loss, colour = "val_loss"))+ geom_line(aes(y = val_loss_dropout, colour = "val_loss_dropout"))

