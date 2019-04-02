# Homework 3

### Description: 

In this homework, you will practice implementing algorithms using R, generating fractional differentiated features while considering the trade off between memory and stationarity of time series, and applying bagged classification trees.

### Dataset: 

Please download the unit bar data of bitcoin futures XBTUSD from Compass 2g, and the data has been preprocessed from tick data of the same ticker traded during the first 8 months of 2018. Use the first 2/3 as the training set and the remaining 1/3 as the test set.

### Practices:

1. Implement the function fracDiff for fractionally differentiated features. Apply the function on the closed prices of the provided unit bars (training and test sets together), choose d = 0.5 and τ = 0.001, and then print out the two correlation coefficients: one for the closed prices of the derived series and the original series, and the other for the first order difference of the closed prices and the original series. Which is larger? Why?

2. Based on the training set, let τ = 0.0001, and choose appropriate values of 1 > d > 0 for the closed prices and the volumes respectively, to pass some unit root tests and the stationarity test (KPSS), while keeping as much memory as possible.

3. Apply fracDiff with the selected d’s for the closed prices and the volume, respectively, based on all the data sets. Apply a symmetric CUSUM filter on the raw closed prices series and set up the threshold as h. For those sampled features bars, assign labels (0 for lower, and 1 for upper) by applying the triple barrier labeling method on the raw closed prices series, where ptSl = [1,1], target returns is trgt, and t1 can be set up large enough to generate enough labels (for example, vertical barriers can be 200 bars away). You can choose to use fmlr::label_meta() for labeling.

4. Apply bagged classification trees on the obtained features bars and labels from the training set, including at least the fractionally differentiated closed prices and the fractionally differentiated volumes as predictors, while treating h and trgt as the two tuning parameters. Choose appropriate values of h and trgt so that the following goals are achieved: (1) There are at least 10 labels of 1 for the features bars obtained from the test set; (2) the AUC for the test set predicted by the bagged trees is at least 0.6. Finaly, use fmlr::acc_lucky() to check whether your candidate model with the tuned parameters can overperform various guesses; using the cutoff of 0.5 for calculating the accuracy based on the candidate model.

Your grade for this homework will be based on how much you follow the homework policy, the completeness of the practices, the accuracy of Questions 1 and 2, and whether appropriate models and analyses have been selected and conducted for Questions 3 and 4.
