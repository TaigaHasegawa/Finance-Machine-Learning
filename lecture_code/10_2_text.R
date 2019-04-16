#-------------------------------------------------------------------------------------------------
# Examples for one-hot encoding
#-------------------------------------------------------------------------------------------------
# Codes are based on https://github.com/jjallaire/deep-learning-with-r-notebooks             
# MIT license: # https://github.com/jjallaire/deep-learning-with-r-notebooks/blob/master/LICENSE
#-------------------------------------------------------------------------------------------------

rm(list = setdiff(ls(), lsf.str()))
library(keras)

#############################
# toy examples - word level #
#############################

# This is our initial data; one entry per "sample"
# (in this toy example, a "sample" is just a sentence, but
# it could be an entire document).
samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# First, build an index of all tokens in the data.
token_index <- list()
for (sample in samples)
  # Tokenizes the samples via the strsplit function. In real life, you'd also
  # strip punctuation and special characters from the samples.
  for (word in strsplit(sample, " ")[[1]])
    if (!word %in% names(token_index))
      # Assigns a unique index to each unique word. Note that you don't
      # attribute index 1 to anything.
      token_index[[word]] <- length(token_index) + 2 

# Vectorizes the samples. You'll only consider the first max_length 
# words in each sample.
max_length <- 10

# This is where you store the results.
results <- array(0, dim = c(length(samples), 
                            max_length, 
                            max(as.integer(token_index))))
dim(results)

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    index <- token_index[[words[[j]]]]
    results[[i, j, index]] <- 1
  }
}
results[1,,]

####################################
# toy examples - charachater level #
####################################

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# raw: hexadecimal (base 16)
ascii_tokens <- c("", sapply(as.raw(c(32:126)), rawToChar))
token_index <- c(1:(length(ascii_tokens)))
names(token_index) <- ascii_tokens

max_length <- 50

results <- array(0, dim = c(length(samples), max_length, length(token_index)))

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  characters <- strsplit(sample, "")[[1]]
  for (j in 1:length(characters)) {
    character <- characters[[j]]
    results[i, j, token_index[[character]]] <- 1
  }
}
results[1,,]

###################
# Keras functions #
###################

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# Creates a tokenizer, configured to only take into account the 1,000 
# most common words, then builds the word index.
tokenizer <- text_tokenizer(num_words = 1000) %>%
  fit_text_tokenizer(samples)

# Turns strings into lists of integer indices
sequences <- texts_to_sequences(tokenizer, samples)

# You could also directly get the one-hot binary representations. Vectorization 
# modes other than one-hot encoding are supported by this tokenizer.
one_hot_results <- texts_to_matrix(tokenizer, samples, mode = "binary")
# NOTICE !!!!!!!!!!!!!! NOT one-hot

# How you can recover the word index that was computed
word_index <- tokenizer$word_index

cat("Found", length(word_index), "unique tokens.\n")


#######################################
# one-hot encoding with hashing trick #
#######################################
library(hashFunction)

samples <- c("The cat sat on the mat.", "The dog ate my homework.")

# We will store our words as vectors of size 1000.
# Note that if you have close to 1000 words (or more)
# you will start seeing many hash collisions, which
# will decrease the accuracy of this encoding method.
dimensionality <- 1000
max_length <- 10

results <- array(0, dim = c(length(samples), max_length, dimensionality))

for (i in 1:length(samples)) {
  sample <- samples[[i]]
  words <- head(strsplit(sample, " ")[[1]], n = max_length)
  for (j in 1:length(words)) {
    # Hash the word into a "random" integer index
    # that is between 0 and 1,000
    index <- abs(spooky.32(words[[i]])) %% dimensionality # %%: modulus
    results[[i, j, index]] <- 1
  }
}

