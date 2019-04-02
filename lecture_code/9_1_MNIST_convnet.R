#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: MNIST data using convnet
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

# Before discussing convnet, let's revist the MNIST example, for which we used dense layers

rm(list = setdiff(ls(), lsf.str()))
library(keras)
use_condaenv("r-tensorflow")

## For comparison, the following was the model based on dense layers
# model <- keras_model_sequential() %>% 
#   layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
#   layer_dropout(rate = 0.4) %>% 
#   layer_dense(units = 128, activation = 'relu') %>%
#   layer_dropout(rate = 0.3) %>%
#   layer_dense(units = 10, activation = 'softmax')

model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>%
  layer_flatten() %>% # flatten 3D to 1D
  layer_dense(units = 64, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

summary(model)

mnist <- dataset_mnist()
c(c(train_images, train_labels), c(test_images, test_labels)) %<-% mnist
train_images <- array_reshape(train_images, c(60000, 28, 28, 1))
train_images <- train_images / 255

test_images <- array_reshape(test_images, c(10000, 28, 28, 1))
test_images <- test_images / 255
train_labels <- to_categorical(train_labels)
test_labels <- to_categorical(test_labels)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)

model %>% fit(
  train_images, train_labels,
  epochs = 5, batch_size=64
)

results <- model %>% evaluate(test_images, test_labels)
results
