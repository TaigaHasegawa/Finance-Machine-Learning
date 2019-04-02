#-------------------------------------------------------------------------------------------------
# A simple example of using Keras to scalar regression for house prices
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

dataset <- dataset_boston_housing()
c(c(train_data, train_targets), c(test_data, test_targets)) %<-% dataset
str(train_data)
str(test_data)
str(train_targets)

###############
## covariates #
###############
# 1. Per capita crime rate.
# 2. Proportion of residential land zoned for lots over 25,000 square feet.
# 3. Proportion of non-retail business acres per town.
# 4. Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 5. Nitric oxides concentration (parts per 10 million).
# 6. Average number of rooms per dwelling.
# 7. Proportion of owner-occupied units built prior to 1940.
# 8. Weighted distances to five Boston employment centres.
# 9. Index of accessibility to radial highways.
# 10. Full-value property-tax rate per $10,000.
# 11. Pupil-teacher ratio by town.
# 12. 1000 * (Bk - 0.63) ** 2 where Bk is the proportion of Black people by town.
# 13. % lower status of the population.

###########
# rescale #
###########
mean <- apply(train_data, 2, mean)
std <- apply(train_data, 2, sd)
train_data <- scale(train_data, center = mean, scale = std)

test_data <- scale(test_data, center = mean, scale = std)

##########
# models #
##########
# Because we will need to instantiate the same model multiple times,
# we use a function to construct it.
build_model <- function() {
  model <- keras_model_sequential() %>% 
    layer_dense(units = 64, activation = "relu", 
                input_shape = dim(train_data)[[2]]) %>% 
    layer_dense(units = 64, activation = "relu") %>% 
    layer_dense(units = 1)  # <--------------------------------------- Note: no activation
  
  model %>% compile(
    optimizer = "rmsprop", 
    loss = "mse", 
    metrics = c("mae")
  )
}


#############
# K-fold CV #
#############
k <- 4
indices <- sample(1:nrow(train_data))
folds <- cut(1:length(indices), breaks = k, labels = FALSE)

num_epochs <- 50
all_scores <- c()
for (i in 1:k) {
  cat("processing fold #", i, "\n")
  # Prepare the validation data: data from partition # k
  val_indices <- which(folds == i, arr.ind = TRUE) 
  val_data <- train_data[val_indices,]
  val_targets <- train_targets[val_indices]
  
  # Prepare the training data: data from all other partitions
  partial_train_data <- train_data[-val_indices,]
  partial_train_targets <- train_targets[-val_indices]
  
  # Build the Keras model (already compiled)
  model <- build_model()
  
  # Train the model (in silent mode, verbose=0)
  model %>% fit(partial_train_data, partial_train_targets,
                epochs = num_epochs, batch_size = 40, verbose = 1)
  
  # Evaluate the model on the validation data
  results <- model %>% evaluate(val_data, val_targets, verbose = 0)
  all_scores <- c(all_scores, results$mean_absolute_error)
}

all_scores
mean(all_scores)

##################
# record history #
##################
if(1==2) # do not run in class
{
  # Some memory clean-up
  k_clear_session()
  
  num_epochs <- 500
  all_mae_histories <- NULL
  for (i in 1:k) {
    cat("processing fold #", i, "\n")
    
    # Prepare the validation data: data from partition # k
    val_indices <- which(folds == i, arr.ind = TRUE)
    val_data <- train_data[val_indices,]
    val_targets <- train_targets[val_indices]
    
    # Prepare the training data: data from all other partitions
    partial_train_data <- train_data[-val_indices,]
    partial_train_targets <- train_targets[-val_indices]
    
    # Build the Keras model (already compiled)
    model <- build_model()
    
    # Train the model (in silent mode, verbose=0)
    history <- model %>% fit(
      partial_train_data, partial_train_targets,
      validation_data = list(val_data, val_targets),
      epochs = num_epochs, batch_size = 1, verbose = 0
    )
    mae_history <- history$metrics$val_mean_absolute_error
    all_mae_histories <- rbind(all_mae_histories, mae_history)
  }
  write.table(all_mae_histories, "~/Dropbox/stat430-dl/all_mae_histories.csv", sep=",", row.names = F, col.names = F)
} # do not run in class

all_mae_histories <- read.csv("~/Dropbox/stat430-dl/all_mae_histories.csv", header = F)

average_mae_history <- data.frame(
  epoch = seq(1:ncol(all_mae_histories)),
  validation_mae = apply(all_mae_histories, 2, mean)
)

library(ggplot2)
ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_line()

ggplot(average_mae_history, aes(x = epoch, y = validation_mae)) + geom_smooth()

# Get a fresh, compiled model.
model <- build_model()

# Train it on the entirety of the data.
model %>% fit(train_data, train_targets,
              epochs = 80, batch_size = 16, verbose = 1)

result <- model %>% evaluate(test_data, test_targets)

result
