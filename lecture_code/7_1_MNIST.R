#-------------------------------------------------------------------------------------------------
# A simple example of using Keras to recognize handwritten digits from the MNIST dataset
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

#############
# Read data #
#############
mnist <- dataset_mnist()

# check data types
class(mnist)
names(mnist)
class(mnist$train$y)
class(mnist$train$x)

x_train <- mnist$train$x
y_train <- mnist$train$y
x_test <- mnist$test$x
y_test <- mnist$test$y

# draw some numbers
digit <- x_train[1,,]
plot(as.raster(digit, max = 255))

# check which is which
head(y_train,4)
par(mfrow=c(1,4))
for(i in 1:4)
{
  tmp <- cbind(rep(1:28,each=28), rep(28:1,28), as.vector(x_train[i,,]/255))
  plot(tmp[,2]~tmp[,1], col=grey(tmp[,3]),pch=15)
}

####################
# reshape the data #
####################
str(x_train)
x_train <- array_reshape(x_train, c(nrow(x_train), 784))
str(x_train)
x_test <- array_reshape(x_test, c(nrow(x_test), 784))

# rescale
x_train <- x_train / 255
x_test <- x_test / 255

##################
# prepare labels #
##################
y_train <- to_categorical(y_train, 10)
y_test <- to_categorical(y_test, 10)

###################
# construct model #
###################
model <- keras_model_sequential() 
model %>% 
  layer_dense(units = 256, activation = 'relu', input_shape = c(784)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 10, activation = 'softmax')

summary(model)

#########################################
# compile the model with appropriate    #
# loss function, optimizer, and metrics #
#########################################
model %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_rmsprop(),
  metrics = c('accuracy')
)

#############
# fit model #
#############
# Use the fit() function to train the model for 30 epochs using batches of 128 images:
history <- model %>% fit(
  x_train, y_train, 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


########################
# evaluate and predict #
########################

# The history object returned by fit() includes loss and accuracy metrics which we can plot:
plot(history)

# Evaluate the model's performance on the test data:
model %>% evaluate(x_test, y_test)

# Generate predictions on new data:
model %>% predict_classes(x_test)
