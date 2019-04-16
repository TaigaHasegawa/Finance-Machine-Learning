# An example of using tfruns to do hyper-parameter tuning
# for the example of using VAE for limit order book analysis
# Larryhua.com/teaching

library(tfruns)
work_folder <- "~/Dropbox/stat430-dl"

runs <- tuning_run(file.path(work_folder,"15_1_vaeLOB.R"),
                   flags = list(intermediate_dim = c(12, 16),
                                epsilon_std = c(1.0, 2.0))
)

runs[order(runs$metric_val_loss, decreasing = TRUE), ]

ls_runs(which.min(metric_val_loss))
ls_runs(order=metric_val_loss)

compare_runs(c("~/runs/2018-12-04T02-29-00Z", "~/runs/2018-12-04T01-43-31Z"))


