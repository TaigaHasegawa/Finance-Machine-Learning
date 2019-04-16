#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: use pretrained model for classifying dog and cat
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)


conv_base <- application_vgg16(
  weights = "imagenet", # specify which weight checkpoint to initialize the model from
  include_top = FALSE, # refers to including or not the densely-connected classifier on top of the network
  input_shape = c(150, 150, 3) # the shape of the image tensors that we will feed to the network. If we don't pass it, then the network will be able to process inputs of any size.
)

# The architecture of the VGG16 convolutional base
# The final feature map has shape `(4, 4, 512)`. That's the feature on top of which we will stick a densely-connected classifier.
summary(conv_base)

########################
# Two ways to proceed: #
########################

##################
# First approach #
##################
work_folder <- "~/Dropbox/stat430-dl"
data_dir <- file.path(work_folder, "datasets/cats_and_dogs_small")
train_dir <- file.path(data_dir, "train")
validation_dir <- file.path(data_dir, "validation")
test_dir <- file.path(data_dir, "test")

datagen <- image_data_generator(rescale = 1/255)
batch_size <- 20

?flow_images_from_directory
extract_features <- function(directory, sample_count) {
  features <- array(0, dim = c(sample_count, 4, 4, 512))  
  labels <- array(0, dim = c(sample_count))
  
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = c(150, 150), # conv_base used the same size
    batch_size = batch_size,
    class_mode = "binary"
  )
  
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    
    ###########################
    # key step: use conv_base #
    ###########################
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch # shape of output features maps: (sample_count, 4, 4, 512)
    labels[index_range] <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      # Note that because generators yield data indefinitely in a loop, 
      # you must break after every image has been seen once.
      break
  }
  
  list(
    features = features, 
    labels = labels
  )
}

train <- extract_features(train_dir, 2000)
validation <- extract_features(validation_dir, 1000)
test <- extract_features(test_dir, 1000)

# The extracted features are currently of shape `(samples, 4, 4, 512)`
# We will feed them to a densely-connected classifier, so first we must flatten them to `(samples, 8192)`:

reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
train$features <- reshape_features(train$features)
validation$features <- reshape_features(validation$features)
test$features <- reshape_features(test$features)

# Define our densely-connected classifier
model <- keras_model_sequential() %>% 
  layer_dense(units = 256, activation = "relu", 
              input_shape = 4 * 4 * 512) %>% 
  layer_dropout(rate = 0.5) %>% 
  layer_dense(units = 1, activation = "sigmoid")

model %>% compile(
  optimizer = optimizer_rmsprop(lr = 2e-5),
  loss = "binary_crossentropy",
  metrics = c("accuracy")
)

history <- model %>% fit(
  train$features, train$labels,
  epochs = 30,
  batch_size = 20,
  validation_data = list(validation$features, validation$labels)
)

plot(history)
# validation accuracy ~ 90%




#########################################
# Second approach - Do not run in class #
#########################################
# much slower and more expensive, but allows data augmentation during training

# Because models behave just like layers, you can add a model (like `conv_base`) to a sequential model just like you would add a layer

model <- keras_model_sequential() %>% 
  
  #############
  #  key step #
  #############
  
  conv_base %>% 
  layer_flatten() %>% 
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = 1, activation = "sigmoid")
summary(model)


############
# key step #
############

# Before you compile and train the model, it's very important to freeze the convolutional base
# _Freezing_ a layer or set of layers means preventing their weights from being updated during training. If you don't do this, then the representations that were previously learned by the convolutional base will be modified during training

length(conv_base$trainable_weights)
freeze_weights(conv_base)
length(conv_base$trainable_weights)

# With this setup, only the weights from the two dense layers that you added will be trained
# In order for these changes to take effect, you must first compile the model
# If you ever modify weight trainability after compilation, you should then recompile the model, or these changes will be ignored

# Start training your model, with the same data-augmentation configuration
train_datagen <- image_data_generator(
  rescale = 1/255,
  rotation_range = 40,
  width_shift_range = 0.2,
  height_shift_range = 0.2,
  shear_range = 0.2,
  zoom_range = 0.2,
  horizontal_flip = TRUE,
  fill_mode = "nearest"
)

test_datagen <- image_data_generator(rescale = 1/255)

train_generator <- flow_images_from_directory(
  train_dir,
  train_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

validation_generator <- flow_images_from_directory(
  validation_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 2e-5),
  metrics = c("accuracy")
)
?fit_generator
#######################
# Do not run in class #
#######################
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 30,
  validation_data = validation_generator,
  validation_steps = 50,
  workers = 8
)

save_model_hdf5(model, file.path(work_folder, "cats_and_dogs_small_3.h5"))
# validation accuracy ~ 90%

#---------------------------------------------------------------------------------------------------------

################
## Fine-tuning #
################

# We have already completed the first 3 steps when doing feature extraction. Let's proceed with the 4th step: we will unfreeze our `conv_base`, and then freeze individual layers inside of it.

summary(conv_base)

unfreeze_weights(conv_base, from = "block3_conv1")

# Using a very low learning rate: limit the magnitude of the modifications we make to the representations of the layers that we are fine-tuning. Updates that are too large may harm these representations

model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-5),
  metrics = c("accuracy")
)


#######################
# Do not run in class #
#######################
history <- model %>% fit_generator(
  train_generator,
  steps_per_epoch = 100,
  epochs = 100,
  validation_data = validation_generator,
  validation_steps = 50
)
save_model_hdf5(model, "cats_and_dogs_small_4.h5")
plot(history)
# validation accuracy ~ 96%

# We can now finally evaluate this model on the test data:
                                                          
test_generator <- flow_images_from_directory(
  test_dir,
  test_datagen,
  target_size = c(150, 150),
  batch_size = 20,
  class_mode = "binary"
)

model %>% evaluate_generator(test_generator, steps = 50)

## takeaways
# Convnets are the best type of machine learning models for computer vision tasks. It is possible to train one from scratch even on a very small dataset, with decent results.
# On a small dataset, overfitting will be the main issue. Data augmentation is a powerful way to fight overfitting when working with image data.
# It is easy to reuse an existing convnet on a new dataset, via feature extraction. This is a very valuable technique for working with small image datasets.
# As a complement to feature extraction, one may use fine-tuning, which adapts to a new problem some of the representations previously learned by an existing model. This pushes performance a bit further.
