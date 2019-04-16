#-------------------------------------------------------------------------------------------------
# Examples for convnet using keras: visualize what convnet has learned
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

work_folder <- "/home/rstudio-user/STAT430"

model <- load_model_hdf5(file.path(work_folder, "cats_and_dogs_small_2.h5"))
summary(model)

img_path <- file.path(work_folder, "/datasets/cats_and_dogs_small/test/cats/cat.1700.jpg")

# We preprocess the image into a 4D tensor
img <- image_load(img_path, target_size = c(150, 150))
img_tensor <- image_to_array(img)
dim(img_tensor)
img_tensor <- array_reshape(img_tensor, c(1, 150, 150, 3)) 

# Remember that the model was trained on inputs that were preprocessed in the following way:
img_tensor <- img_tensor / 255
plot(as.raster(img_tensor[1,,,]))

####################################
# visualizing intermediate outputs #
####################################

# Extracts the outputs of the top 8 layers:
layer_outputs <- lapply(model$layers[1:8], function(layer) layer$output)

# Creates a model that will return these outputs, given the model input:
activation_model <- keras_model(inputs = model$input, outputs = layer_outputs)
# This one has one input and 8 outputs, one output per layer activation.

# Returns a list of 8 arrays: one array per layer activation
activations <- activation_model %>% predict(img_tensor)

# the activation of the first convolution layer for our cat image input
first_layer_activation <- activations[[1]]
dim(first_layer_activation)

# Let's visualize some of them. First we define an R function that will plot a channel:
plot_channel <- function(channel) {
  rotate <- function(x) t(apply(x, 2, rev)) # rev: reverse orders
  image(rotate(channel), axes = FALSE, asp = 1, col = terrain.colors(12))
}

# Let's try visualizing the 5th and 7th channels:
plot_channel(first_layer_activation[1,,,2])
plot_channel(first_layer_activation[1,,,32])

# Extract and plot every channel in each of our 8 activation maps, and we will stack the results in one big image tensor, with channels stacked side by side.
dir.create(file.path(work_folder, "cat_activations"))
image_size <- 58
images_per_row <- 16

for (i in 1:8) {
  layer_activation <- activations[[i]]
  layer_name <- model$layers[[i]]$name
  
  n_features <- dim(layer_activation)[[4]]
  n_cols <- n_features %/% images_per_row
  
  png(paste0(file.path(work_folder, "cat_activations/"), i, "_", layer_name, ".png"), 
      width = image_size * images_per_row, 
      height = image_size * n_cols)
  op <- par(mfrow = c(n_cols, images_per_row), mai = rep_len(0.02, 4))
  
  for (col in 0:(n_cols-1)) {
    for (row in 0:(images_per_row-1)) {
      channel_image <- layer_activation[1,,,(col*images_per_row) + row + 1]
      plot_channel(channel_image)
    }
  }
  par(op)
  dev.off()
}

################################
## Visualizing convnet filters #
################################

model <- application_vgg16(weights = "imagenet", include_top = FALSE)

layer_name <- "block3_conv1"
filter_index <- 1

layer_output <- get_layer(model, layer_name)$output
summary(layer_output)

# `k_xxxx()` is part of a set of Keras backend functions that enable lower level access to the 
# core operations of the backend tensor engine (e.g. TensorFlow, CNTK, Theano, etc.)
loss <- k_mean(layer_output[,,,filter_index])

# To implement gradient ascent, we will need the gradient of this loss with respect to the model's input. 
# To do this, we will use the `k_gradients` Keras backend function:

# The call to `gradients` returns a list of tensors (of size 1 in this case)
# hence we only keep the first element -- which is a tensor.
grads <- k_gradients(loss, model$input)[[1]] 

# A non-obvious trick to use for the gradient descent process to go smoothly is to normalize the gradient tensor, by dividing it by its L2 norm (the square root of the average of the square of the values in the tensor). 
# This ensures that the magnitude of the updates done to the input image is always within a same range.
# We add 1e-5 before dividing so as to avoid accidentally dividing by 0.
grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)

# Now you need a way to compute the value of the loss tensor and the gradient tensor, given an input image. 
# You can define a Keras backend function to do this: 
# `iterate` is a function that takes a tensor (as a list of tensors of size 1) and returns a list of two tensors: 
# the loss value and the gradient value

iterate <- k_function(list(model$input), list(loss, grads))

# Let's test it
c(loss_value, grads_value) %<-% iterate(list(array(0, dim = c(1, 150, 150, 3))))

# At this point we can define an R loop to do stochastic gradient descent:

# We start from a gray image with some noise
input_img_data <- array(runif(150 * 150 * 3), dim = c(1, 150, 150, 3)) * 20 + 128

step <- 1  # this is the magnitude of each gradient update
################################
# note that loss is increasing #
################################
for (i in 1:40) { 
  # Compute the loss value and gradient value
  c(loss_value, grads_value) %<-% iterate(list(input_img_data))
  cat(i, "loss value: ", loss_value, "\n")
  # Here we adjust the input image in the direction that maximizes the loss
  input_img_data <- input_img_data + (grads_value * step)
}

# The resulting image tensor is a floating-point tensor of shape `(1, 150, 150, 3)`, 
# with values that may not be integers within [0, 255]. 
# Hence you need to post-process this tensor to turn it into a displayable image. 
# You do so with the following straightforward utility function

deprocess_image <- function(x) {
  dms <- dim(x)
  
  # normalize tensor: center on 0., ensure std is 0.1
  x <- x - mean(x) 
  x <- x / (sd(x) + 1e-5)
  x <- x * 0.1 
  
  # clip to [0, 1]
  x <- x + 0.5 
  x <- pmax(0, pmin(x, 1))
  
  # Reshape to original image dimensions
  array(x, dim = dms)
}

# Now you have all the pieces. Let's put them together into an R function 
# that takes as input a layer name and a filter index, and returns a valid 
# image tensor representing the pattern that maximizes the activation of the specified filter.

generate_pattern <- function(layer_name, filter_index, size = 150) {
  
  # Build a loss function that maximizes the activation
  # of the nth filter of the layer considered.
  layer_output <- model$get_layer(layer_name)$output
  loss <- k_mean(layer_output[,,,filter_index]) 
  
  # Compute the gradient of the input picture wrt this loss
  grads <- k_gradients(loss, model$input)[[1]]
  
  # Normalization trick: we normalize the gradient
  grads <- grads / (k_sqrt(k_mean(k_square(grads))) + 1e-5)
  
  # Define a function that returns the loss and grads given the input picture
  iterate <- k_function(input=list(model$input), output=list(loss, grads))
  
  # We start from a gray image with some noise
  input_img_data <- array(runif(size * size * 3), dim = c(1, size, size, 3)) * 20 + 128
  
  # Run gradient ascent for 40 steps
  step <- 1
  for (i in 1:40) {
    c(loss_value, grads_value) %<-% iterate(list(input_img_data))
    input_img_data <- input_img_data + (grads_value * step) 
  }
  
  img <- input_img_data[1,,,]
  deprocess_image(img)
}

# Let's try this:
library(grid)
grid.raster(generate_pattern("block3_conv1", 1))

# It seems that filter 1 in layer `block3_conv1` is responsive to a polka dot pattern.

# Now the fun part: we can start visualising every single filter in every layer. 
# For simplicity, we will only look at the first 64 filters in each layer, 
# and will only look at the first layer of each convolution block 
# (block1_conv1, block2_conv1, block3_conv1, block4_conv1, block5_conv1). 
# We will arrange the outputs on a 8x8 grid of filter patterns.

############################
# See slides for the plots #
############################
library(gridExtra)
dir.create("~/Dropbox/Teaching/STAT430/R/vgg_filters")
for (layer_name in c("block1_conv1", "block2_conv1", "block3_conv1", "block4_conv1"))
{
  size <- 140
  png(paste0("~/Dropbox/Teaching/STAT430/R/vgg_filters/", layer_name, ".png"), width = 8 * size, height = 8 * size)
  grobs <- list()
  for (i in 0:7) {
    for (j in 0:7) {
      pattern <- generate_pattern(layer_name, i + (j*8) + 1, size = size)
      grob <- rasterGrob(pattern,
                         width = unit(0.9, "npc"), 
                         height = unit(0.9, "npc"))
      grobs[[length(grobs)+1]] <- grob
    }  
  }
  grid.arrange(grobs = grobs, ncol = 8)
  dev.off()
}

##############################################
## Visualizing heatmaps of class activation ##
##############################################
# We'll demonstrate this technique using the pretrained VGG16 network again.

# Clear out the session
k_clear_session()

# Note that we are including the densely-connected classifier on top;
# all previous times, we were discarding it.
model <- application_vgg16(weights = "imagenet")

# Let's consider the following image of two African elephants, possible a mother and its cub, 
# strolling in the savanna (under a Creative Commons license):

# ![elephants](https://s3.amazonaws.com/book.keras.io/img/ch5/creative_commons_elephant.jpg)

# Let's convert this image into something the VGG16 model can read: 
# the model was trained on images of size 224 × 244, preprocessed according to a few rules 
# that are packaged in the utility function `imagenet_preprocess_input()`. 
# So you need to load the image, resize it to 224 × 224, convert it to an array, and apply these preprocessing rules.

# The local path to our target image
img_path <- file.path(work_folder, "creative_commons_elephant.jpg")

# Start witih image of size 224 × 224
img <- image_load(img_path, target_size = c(224, 224)) %>% 
  # Array of shape (224, 224, 3)
  image_to_array() %>% 
  # Adds a dimension to transform the array into a batch of size (1, 224, 224, 3)
  array_reshape(dim = c(1, 224, 224, 3)) %>% 
  # Preprocesses the batch (this does channel-wise color normalization)
  imagenet_preprocess_input()

# You can now run the pretrained network on the image and decode its prediction vector back to a human-readable format:

preds <- model %>% predict(img)
imagenet_decode_predictions(preds, top = 3)[[1]]

# The top-3 classes predicted for this image are:

#* African elephant (with 90.9% probability)
#* Tusker (with 8.6% probability)
#* Indian elephant (with 0.4% probability)

# Thus our network has recognized our image as containing an undetermined quantity of African elephants. The entry in the prediction vector 
# that was maximally activated is the one corresponding to the "African elephant" class, at index 387:

which.max(preds[1,])

# To visualize which parts of our image were the most "African elephant"-like, let's set up the Grad-CAM process:

# This is the "african elephant" entry in the prediction vector
african_elephant_output <- model$output[, 387]

# The is the output feature map of the `block5_conv3` layer,
# the last convolutional layer in VGG16
last_conv_layer <- model %>% get_layer("block5_conv3")

# This is the gradient of the "african elephant" class with regard to
# the output feature map of `block5_conv3`
grads <- k_gradients(african_elephant_output, last_conv_layer$output)[[1]]

# This is a vector of shape (512,), where each entry
# is the mean intensity of the gradient over a specific feature map channel
pooled_grads <- k_mean(grads, axis = c(1, 2, 3))

# This function allows us to access the values of the quantities we just defined:
# `pooled_grads` and the output feature map of `block5_conv3`,
# given a sample image
iterate <- k_function(list(model$input),list(pooled_grads, last_conv_layer$output[1,,,])) 

# These are the values of these two quantities, as arrays,
# given our sample image of two elephants
c(pooled_grads_value, conv_layer_output_value) %<-% iterate(list(img))

# We multiply each channel in the feature map array
# by "how important this channel is" with regard to the elephant class
for (i in 1:512) {
  conv_layer_output_value[,,i] <- 
    conv_layer_output_value[,,i] * pooled_grads_value[[i]] 
}

# The channel-wise mean of the resulting feature map
# is our heatmap of class activation
heatmap <- apply(conv_layer_output_value, c(1,2), mean)

# For visualization purposes, you'll also normalize the heatmap between 0 and 1.
heatmap <- pmax(heatmap, 0) 
heatmap <- heatmap / max(heatmap)

write_heatmap <- function(heatmap, filename, width = 224, height = 224, bg = "white", col = terrain.colors(12))
{
  png(filename, width = width, height = height, bg = bg)
  op = par(mar = c(0,0,0,0))
  on.exit({par(op); dev.off()}, add = TRUE)
  rotate <- function(x) t(apply(x, 2, rev))
  image(rotate(heatmap), axes = FALSE, asp = 1, col = col)
}

write_heatmap(heatmap, file.path(work_folder,"elephant_heatmap.png"))

# Finally, we will use the *magick* package to generate an image that superimposes the original image with the heatmap we just obtained:
library(magick) 
library(viridis) 

# Read the original elephant image and it's geometry
image <- image_read(img_path)
info <- image_info(image) 
geometry <- sprintf("%dx%d!", info$width, info$height) 

# Create a blended / transparent version of the heatmap image
pal <- col2rgb(viridis(20), alpha = TRUE)
alpha <- floor(seq(0, 255, length = ncol(pal)))
pal_col <- rgb(t(pal), alpha = alpha, maxColorValue = 255)
write_heatmap(heatmap, "~/Dropbox/Teaching/STAT430/R/images/elephant_overlay.png", 
              width = 14, height = 14, bg = NA, col = pal_col) 

# Overlay the heatmap
image_read("~/Dropbox/Teaching/STAT430/R/images/elephant_overlay.png") %>% 
  image_resize(geometry, filter = "quadratic") %>% 
  image_composite(image, operator = "blend", compose_args = "20") %>%
  plot()

# This visualisation technique answers two important questions:
# Why did the network think this image contained an African elephant?
# Where is the African elephant located in the picture?

# In particular, it is interesting to note that the ears of the elephant cub are strongly activated: this is probably how the network can tell the difference between African and Indian elephants.
