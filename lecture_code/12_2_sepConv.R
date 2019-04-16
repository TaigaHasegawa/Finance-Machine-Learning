#-------------------------------------------------------------------------------------------------
# Examples for sepConv
#-------------------------------------------------------------------------------------------------
# MIT license: # https://github.com/rstudio/keras
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

height <- 64
width <- 64
channels <- 3
num_classes <- 10

model <- keras_model_sequential() %>%
  layer_separable_conv_2d(filters = 32, kernel_size = 3,
                          activation = "relu",
                          input_shape = c(height, width, channels))
summary(model)  
# 155 parameters

model1 <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = 3,
                activation = "relu",
                input_shape = c(height, width, channels))
summary(model1)
# 896 parameters

model %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_max_pooling_2d(pool_size = 2) %>%
  layer_separable_conv_2d(filters = 64, kernel_size = 3,
                          activation = "relu") %>%
  layer_separable_conv_2d(filters = 128, kernel_size = 3,
                          activation = "relu") %>%
  layer_global_average_pooling_2d() %>%
  layer_dense(units = 32, activation = "relu") %>%
  layer_dense(units = num_classes, activation = "softmax")
summary(model)

model %>% compile(
  optimizer = "rmsprop",
  loss = "categorical_crossentropy"
)



