# Homework 2

### Description: 

In this homework, you will practice implementing algorithms using R, conducting standard triple-barrier and meta labeling, and evaluating the performance of models you choose for the labeling techniques.

### Dataset: 

Please download the tick data of E-Mini SP500 futures from here (yes, click on it). Please read the readme file, and the following questions will be based on the tick data for symbol “ESU13” of “ES_Trades.csv”.

### Practices:

1. Form dollar bars from the ‘Unfiltered Price’ of the dataset; choose an appropriate threshold so that you have at least thousands of bars to work with.

2. Apply a symmetric CUSUM filter and set up a reasonable threshold so that you have at least hundreds of feature bars.

3. On those sampled features, apply the standard triple-barrier method, where ptSl = [1,1] and t1 can be set up as a reasonable value based on your preference.

4. On those sampled features, apply the meta labeling method and only consider the labels associated with the upper barriers, where ptSl = [1,0] and t1 can be set up as a reasonable value based on your preference. After you get the labels and the feature bars, try using some potential features from the feature bars as the predictors, and use the labels (either 1 or 0) from the meta labeling method to conduct a classification. Your report for the analysis should include at least how you evaluate the in-sample performance of the models based on the confusion matrix, ROC curves, F1 scores, etc.
Your grade for this homework will be based on how much you follow the homework policy, the completeness of the practices, and the degree of maturity and sophistication of your statistical arguments, analyses, and interpretations.
