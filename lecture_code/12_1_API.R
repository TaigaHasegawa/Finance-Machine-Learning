#-------------------------------------------------------------------------------------------------
# Examples for Keras API
#-------------------------------------------------------------------------------------------------
# MIT license: # https://github.com/rstudio/keras
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)
work_folder <- "~/Dropbox/stat430-dl"

##############################################
# A simple example: API for sequential model #
##############################################

# input layer
inputs <- layer_input(shape = c(784))

# outputs compose input + dense layers
predictions <- inputs %>%
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 10, activation = 'softmax')

# create and compile model
model <- keras_model(inputs = inputs, outputs = predictions)
summary(model)

###############################################################
# With the functional API, it is easy to reuse trained models #
###############################################################

# you can treat any model as if it were a layer. Note that you aren't just reusing the architecture of the model, you are also reusing its weights.
x <- layer_input(shape = c(784))
y <- x %>% model # returns the 10-way softmax we defined above.


######################################################
# create models that can process sequences of inputs #
######################################################
# You could turn an image classification model into a video classification model, in just one line:

# Input tensor for sequences of 20 timesteps,
# each containing a 784-dimensional vector
input_sequences <- layer_input(shape = c(20, 784))
# This applies our previous model to the input sequence
processed_sequences <- input_sequences %>% time_distributed(model)

#-------------------------------------#
# MNIST Example with Hierarchical RNN #
#-------------------------------------#

# Training parameters.
batch_size <- 32
num_classes <- 10
epochs <- 5

# Embedding dimensions.
row_hidden <- 128
col_hidden <- 128

# The data, shuffled and split between train and test sets
mnist <- dataset_mnist()
x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# Reshapes data to 4D for Hierarchical RNN.
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))
x_train <- x_train / 255
x_test <- x_test / 255

dim_x_train <- dim(x_train)
cat('x_train_shape:', dim_x_train)
cat(nrow(x_train), 'train samples')
cat(nrow(x_test), 'test samples')

# Converts class vectors to binary class matrices
y_train <- to_categorical(y_train, num_classes)
y_test <- to_categorical(y_test, num_classes)

# Define input dimensions
row <- dim_x_train[[2]]
col <- dim_x_train[[3]]
pixel <- dim_x_train[[4]]

# Model input (4D)
input <- layer_input(shape = c(row, col, pixel))

# Encodes a row of pixels using TimeDistributed Wrapper
# the axis right after the sample axis will be considered to be the temporal axis
encoded_rows <- input %>% time_distributed(layer_lstm(units = row_hidden))

# Encodes columns of encoded rows
encoded_columns <- encoded_rows %>% layer_lstm(units = col_hidden)

# Model output
prediction <- encoded_columns %>%
  layer_dense(units = num_classes, activation = 'softmax')

# Define Model ------------------------------------------------------------------------
model <- keras_model(input, prediction)
summary(model)

model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = 'rmsprop',
  metrics = c('accuracy')
)

# Training
model %>% fit(
  x_train, y_train,
  batch_size = batch_size,
  epochs = epochs,
  verbose = 1,
  validation_data = list(x_test, y_test)
)

# Evaluation
scores <- model %>% evaluate(x_test, y_test, verbose = 0)
cat('Test loss:', scores[[1]], '\n')
# Test loss: 0.05627204 
cat('Test accuracy:', scores[[2]], '\n')
# Test accuracy: 0.9838 

#######################################
# Multi-input and multi-output models #
#######################################
# Predict how many retweets and likes a news headline will receive on Twitter
# The main input to the model will be the headline itself, as a sequence of words
# our model will also have an auxiliary input, receiving extra data such as the time of day when the headline was posted
# The model will also be supervised via two loss functions


# The integers will be between 1 and 10,000 (a vocabulary of 10,000 words) and the sequences will be 100 words long.
main_input <- layer_input(shape = c(100), dtype = 'int32', name = 'main_input')
lstm_out <- main_input %>%
  layer_embedding(input_dim = 10000, output_dim = 512, input_length = 100) %>%
  layer_lstm(units = 32)

# Here we insert the auxiliary loss, allowing the LSTM and Embedding layer to be trained smoothly even though the main loss will be much higher in the model:

auxiliary_output <- lstm_out %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'aux_output')

# At this point, we feed into the model our auxiliary input data by concatenating it with the LSTM output, stacking a deep densely-connected network on top and adding the main logistic regression layer

auxiliary_input <- layer_input(shape = c(5), name = 'aux_input')
main_output <- layer_concatenate(c(lstm_out, auxiliary_input)) %>%  
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 64, activation = 'relu') %>% 
  layer_dense(units = 1, activation = 'sigmoid', name = 'main_output')


# This defines a model with two inputs and two outputs:
model <- keras_model(
  inputs = c(main_input, auxiliary_input), 
  outputs = c(main_output, auxiliary_output)
)
summary(model)

# We compile the model and assign a weight of 0.2 to the auxiliary loss.
# To specify different `loss_weights` or `loss` for each different output, you can use a list or a dictionary.
# Here we pass a single loss as the `loss` argument, so the same loss will be used on all outputs.

model %>% compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  loss_weights = c(1.0, 0.2) # the order is based on outputs specified in keras_model()
)

# We can train the model by passing it lists of input arrays and target arrays
# here the data is not avaliable, and we use this as psuedo codes to show the idea
model %>% fit(
  x = list(headline_data, additional_data),
  y = list(labels, labels),
  epochs = 50,
  batch_size = 32
)

# Since our inputs and outputs are named (we passed them a "name" argument),
# We could also have compiled the model via:

model %>% compile(
  optimizer = 'rmsprop',
  loss = list(main_output = 'binary_crossentropy', aux_output = 'binary_crossentropy'),
  loss_weights = list(main_output = 1.0, aux_output = 0.2)
)

# And trained it via:
model %>% fit(
  x = list(main_input = headline_data, aux_input = additional_data),
  y = list(main_output = labels, aux_output = labels),
  epochs = 50,
  batch_size = 32
)

###################
## Shared layers ##
###################

# Let's consider a dataset of tweets. We want to build a model that can tell whether two tweets are from the same person or not (this can allow us to compare users by the similarity of their tweets, for instance).

# One way to achieve this is to build a model that encodes two tweets into two vectors, concatenates the vectors and then adds a logistic regression; this outputs a probability that the two tweets share the same author. The model would then be trained on positive tweet pairs and negative tweet pairs.

# Because the problem is symmetric, the mechanism that encodes the first tweet should be reused (weights and all) to encode the second tweet. Here we use a shared LSTM layer to encode the tweets.

# Let's build this with the functional API. We will take as input for a tweet a binary matrix of shape `(280, 256)`, i.e. a sequence of 280 vectors of size 256, where each dimension in the 256-dimensional vector encodes the presence/absence of a character (out of an alphabet of 256 frequent characters).

tweet_a <- layer_input(shape = c(280, 256))
tweet_b <- layer_input(shape = c(280, 256))

# To share a layer across different inputs, simply instantiate the layer once, then call it on as many inputs as you want:

# This layer can take as input a matrix and will return a vector of size 64
shared_lstm <- layer_lstm(units = 64)

# When we reuse the same layer instance multiple times, the weights of the layer are also
# being reused (it is effectively *the same* layer)
encoded_a <- tweet_a %>% shared_lstm
encoded_b <- tweet_b %>% shared_lstm

# We can then concatenate the two vectors and add a logistic regression on top
predictions <- layer_concatenate(c(encoded_a, encoded_b)) %>%
  layer_dense(units = 1, activation = 'sigmoid')

# We define a trainable model linking the tweet inputs to the predictions
model <- keras_model(inputs = c(tweet_a, tweet_b), outputs = predictions)
model %>% compile(
  optimizer = 'rmsprop',
  loss = 'binary_crossentropy',
  metrics = c('accuracy')
)

model %>% fit(list(data_a, data_b), labels, epochs = 10)

################################
## The concept of layer "node" #
################################

# Whenever you are calling a layer on some input, you are creating a new tensor (the output of the layer), and you are adding a "node" to the layer, linking the input tensor to the output tensor. When you are calling the same layer multiple times, that layer owns multiple nodes indexed as 1, 2, 3 ...

# You can obtain the output tensor of a layer via `layer$output`, or its output shape via `layer$output_shape`. But what if a layer is connected to multiple inputs?

# As long as a layer is only connected to one input, there is no confusion, and `$output` will return the one output of the layer:

a <- layer_input(shape = c(280, 256))
lstm <- layer_lstm(units = 32)
encoded_a <- a %>% lstm
lstm$output

# Not so if the layer has multiple inputs:

a <- layer_input(shape = c(280, 256))
b <- layer_input(shape = c(280, 256))
lstm <- layer_lstm(units = 32)
encoded_a <- a %>% lstm
encoded_b <- b %>% lstm
lstm$output
# AttributeError: Layer lstm_4 has multiple inbound nodes, hence the notion of "layer output" is ill-defined. 

# Use `get_output_at(node_index)` instead.
get_output_at(lstm, 1)
get_output_at(lstm, 2)

# The same is true for the properties `input_shape` and `output_shape`: as long as the layer has only one node, or as long as all nodes have the same input/output shape, then the notion of "layer output/input shape" is well defined, and that one shape will be returned by `layer$output_shape`/`layer$input_shape`. But if, for instance, you apply the same `layer_conv_2d()` layer to an input of shape `(32, 32, 3)`, and then to an input of shape `(64, 64, 3)`, the layer will have multiple input/output shapes, and you will have to fetch them by specifying the index of the node they belong to:

a <- layer_input(shape = c(32, 32, 3))
b <- layer_input(shape = c(64, 64, 3))
conv <- layer_conv_2d(filters = 16, kernel_size = c(3,3), padding = 'same')
conved_a <- a %>% conv

# only one input so far, the following will work
conv$input_shape
conved_b <- b %>% conv

# now the `$input_shape` property wouldn't work
conv$input_shape

# but this does
get_input_shape_at(conv, 1)
get_input_shape_at(conv, 2) 

####################
# Inception module #
####################
# For more information about the Inception architecture, see [Going Deeper with Convolutions](http://arxiv.org/abs/1409.4842).

## Figure 7.8
input <- layer_input(shape = c(28, 28, 3))
branch_a <- input %>% layer_conv_2d(filters = 128, kernel_size = 1, activation = "relu", strides = 4)
branch_b <- input %>% layer_conv_2d(filters = 128, kernel_size = 1, activation = "relu") %>% 
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu", strides = 2, padding = "same")
branch_c <- input %>% layer_average_pooling_2d(pool_size = 3, strides = 2, padding = "same") %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu", padding = "same")
branch_d <- input %>% layer_conv_2d(filters = 128, kernel_size = 1,activation = "relu") %>%
  layer_conv_2d(filters = 128, kernel_size = 3, activation = "relu", padding = "same") %>%
  layer_conv_2d(filters = 64, kernel_size = 3, activation = "relu", strides = 2, padding = "same")
output <- layer_concatenate(list(branch_a, branch_b, branch_c, branch_d))
# note that the number of channels is now 128*4 = 512
# Tensor("concatenate_5/concat:0", shape=(?, 14, 14, 512), dtype=float32)

##############################################
# Residual connection on a convolution layer #
##############################################
# For more information about residual networks, see [Deep Residual Learning for Image Recognition](http://arxiv.org/abs/1512.03385).

# input tensor for a 3-channel 256x256 image
x <- layer_input(shape = c(256, 256, 3))
# 3x3 conv with 3 output channels (same as input channels)
y <- x %>% layer_conv_2d(filters = 3, kernel_size =c(3, 3), padding = "same")
# this returns x + y.
z <- layer_add(c(x, y))

#################################################
# The following are more models using Keras API #
#################################################

#######################
# Shared vision model #
#######################
# This model reuses the same image-processing module on two inputs, to classify whether two MNIST digits are the same digit or different digits.

# First, define the vision model
digit_input <- layer_input(shape = c(27, 27, 1))
out <- digit_input %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten()
vision_model <- keras_model(digit_input, out)

# Then define the tell-digits-apart model
digit_a <- layer_input(shape = c(27, 27, 1))
digit_b <- layer_input(shape = c(27, 27, 1))

# The vision model will be shared, weights and all
out_a <- digit_a %>% vision_model
out_b <- digit_b %>% vision_model
out <- layer_concatenate(c(out_a, out_b)) %>% 
  layer_dense(units = 1, activation = 'sigmoid')
classification_model <- keras_model(inputs = c(digit_a, digit_b), out)

###################################
# Visual question answering model #
###################################
# This model can select the correct one-word answer when asked a natural-language question about a picture.
# It works by encoding the question into a vector, encoding the image into a vector, concatenating the two, and training on top a logistic regression over some vocabulary of potential answers.

# First, let's define a vision model using a Sequential model.
# This model will encode an image into a vector.
vision_model <- keras_model_sequential() 
vision_model %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu', padding = 'same',
                input_shape = c(224, 224, 3)) %>% 
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu', padding = 'same') %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_conv_2d(filters = 256, kernel_size = c(3, 3), activation = 'relu') %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>% 
  layer_flatten()

# Now let's get a tensor with the output of our vision model:
image_input <- layer_input(shape = c(224, 224, 3))
encoded_image <- image_input %>% vision_model

# Next, let's define a language model to encode the question into a vector.
# Each question will be at most 100 word long,
# and we will index words as integers from 1 to 9999.
question_input <- layer_input(shape = c(100), dtype = 'int32')
encoded_question <- question_input %>% 
  layer_embedding(input_dim = 10000, output_dim = 256, input_length = 100) %>% 
  layer_lstm(units = 256)

# Let's concatenate the question vector and the image vector then
# train a logistic regression over 1000 words on top
output <- layer_concatenate(c(encoded_question, encoded_image)) %>% 
  layer_dense(units = 1000, activation='softmax')
# This is our final model:
vqa_model <- keras_model(inputs = c(image_input, question_input), outputs = output)

##################################
# Video question answering model #
##################################
# Now that we have trained our image QA model, we can quickly turn it into a video QA model. With appropriate training, you will be able to show it a short video (e.g. 100-frame human action) and ask a natural language question about the video (e.g. "what sport is the boy playing?" -> "football").

video_input <- layer_input(shape = c(100, 224, 224, 3))
# This is our video encoded via the previously trained vision_model (weights are reused)

encoded_video <- video_input %>% 
  time_distributed(vision_model) %>% 
  layer_lstm(units = 256)

# This is a model-level representation of the question encoder, reusing the same weights as before:
question_encoder <- keras_model(inputs = question_input, outputs = encoded_question)

# Let's use it to encode the question:
video_question_input <- layer_input(shape = c(100), dtype = 'int32')
encoded_video_question <- video_question_input %>% question_encoder

# And this is our video question answering model:
output <- layer_concatenate(c(encoded_video, encoded_video_question)) %>% 
  layer_dense(units = 1000, activation = 'softmax')
video_qa_model <- keras_model(inputs= c(video_input, video_question_input), outputs = output)
