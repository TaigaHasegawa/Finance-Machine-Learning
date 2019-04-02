# Homework 4

### Description:

In this homework, you will gain firsthand experience of analyzing unstructured high frequency financial market data using (shallow) machine learning methods. Consider yourself as a quantitative analyst, and you are analyzing a futures dataset provided directly by a data vendor, which is AlgoSeek. Your goal is to provide models and features that are potentially useful for quantitative strategists / traders.

### Dataset: 

Please download the tick data from Compass 2g, and you will be using E-mini SP500 data from the folders 201603 and 201604. Please note that the license of the data only allows you to do homework and projects for this course, and you should delete the data if you do not take the course.

### Practices:

1. Preprocess the unstructured data so that you will have a labeled dataset of a sufficiently large sample size. You can label the price movement direction based on the volume weighted average prices (vwap) of each minute, and take advantage of the 10 levels limit order books to construct any potentially useful features. Some R functions from the fmlr package can be used, such as, fmlr::read_algoseek_futures_fullDepth(), fmlr::istar_CUSUM(), and fmlr::label_meta().

2. This is a relatively open assignment, and there are some necessary sub-tasks, including, but not limited to the following:

• futures rollovers

• randomforest models (sequential bootstrap is optional)

• grid search and parameter tuning with purged k-fold cross validation with embargo
