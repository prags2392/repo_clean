# repo_clean
different architectures for small scale image classification

Each script contains a different architecture that was tried to run a classifier for 5-classes with very limited training data and high variation in testing data.

Data_prepare tries different augmentation techniques based on the dataset available.

RESNET seemed to be the most stable, and learnt the most about intricate features. However, overfitting was the norm even with different regularization techniques.
