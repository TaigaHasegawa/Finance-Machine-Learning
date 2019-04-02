#-------------------------------------------------------------------------------------------------
# A simple example of using Keras to multiple classification for Reuters dataset
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

# Reuters dataset is a set of short newswires and their topics, published by 
# Reuters in 1986. It's a very simple, widely used toy dataset for text classification. 
# There are 46 different topics; some topics are more represented than others, 
# but each topic has at least 10 examples in the training set.

reuters <- dataset_reuters(num_words = 10000)
c(c(train_data, train_labels), c(test_data, test_labels)) %<-% reuters
length(train_data)
length(test_data)
train_data[[1]]


# Decoding newswires back to text
word_index <- dataset_reuters_word_index()
reverse_word_index <- names(word_index)
names(reverse_word_index) <- word_index
train_labels[[1]]

vectorize_sequences <- function(sequences, dimension = 10000) {
  results <- matrix(0, nrow = length(sequences), ncol = dimension)
  for (i in 1:length(sequences))
    results[i, sequences[[i]]] <- 1
  results
}

x_train <- vectorize_sequences(train_data)
x_test <- vectorize_sequences(test_data)

one_hot_train_labels <- to_categorical(train_labels)
one_hot_test_labels <- to_categorical(test_labels)


################################
# multiple class, single label #
################################
k_clear_session()

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "softmax") # 46 topics

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

val_indices <- 1:1000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]
y_val <- one_hot_train_labels[val_indices,]
partial_y_train = one_hot_train_labels[-val_indices,]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

plot(history)

##############################################
# Multiple class, multiple label             #
# [do not run: need multi-hot-encoded labels #
##############################################
if(1==2)
{
  
k_clear_session()

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = c(10000)) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 46, activation = "sigmoid") # <---------------------------- 46 non-exclusive topics

model %>% compile(
  optimizer = "rmsprop",
  loss = "binary_crossentropy", # <-------------------------- calculate loss independently for each label
  metrics = c("accuracy")
)

val_indices <- 1:1000
x_val <- x_train[val_indices,]
partial_x_train <- x_train[-val_indices,]

y_val <- multi_hot_train_labels[val_indices,] # <--------------------  multi-hot-encoded labels are needed
partial_y_train = multi_hot_train_labels[-val_indices,]

history <- model %>% fit(
  partial_x_train,
  partial_y_train,
  epochs = 20,
  batch_size = 512,
  validation_data = list(x_val, y_val)
)

plot(history)
} # do not run







