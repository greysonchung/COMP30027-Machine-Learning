Code and function explanation for assignment 1
Haonan Zhong 867492

Gaussian Naive Bayes Classifier
This classifier preprocess, train and predict pose label for the test dataset provided based on trained model based on the train dataset

Library used for Gaussian Naive Bayes Classifier
import numpy as np
import pandas as pd
import math

How to generate results?
Simply run each cell of the Jupyter Notebook in order.

# Functions

normal_pdf(x, mu, std):

This function calculate and returns the normal density with the given x, mean and standard deviation.

log_gaussian(x, mu, std):

This function does smoothing on the missing values in the test dataset while returning the log gaussian density with the given x, mean and standard deviation.

preprocess(train_file, test_file):

This function takes two file name string and convert the csv file into pandas DataFrame.
For the train file, instances that contains only missing keypoints are removed, and the other missing values are being filled with mean of its attribute column by class.
For the train file, nothing is modified.
Returns two cleaned file, train_clean and test_clean.

train(train_df):

This function takes in a preprocessed train DataFrame and calculates the prior, mean and standard deviation of each class, and return a dictionary which contains Class name as they key and prior, mean and standard deviation as item.

predict(test, model):

This function takes in the test clean file preprocessed in the preprocess function and the train model dictionary generated in the train function.
Returns a list of predicted label of each instance in the test dataset.

Evaluation(actual_lbl, predicted_lbl):

This function takes in two class label array, one is the actual label, the other one is predicted label and compares it.
Returns the accuracy of the prediction.


# Question 1

confusion_matrix(actual_lbl, predicted_lbl, lbl_names):

This function computes the confusion matrix with the given arrays of labels, which helps to evaluate the prediction

class_report(actual_lbl, predicted_lbl, class_name):

This function calculate the macro averaged and micro averaged metric of the prediction.

# Question 3

kde_preprocess(train_file, test_file):

This function takes in the train file and test file, then preprocess
Returns a clean train DataFrame.

kde_train(train):

Instead of storing mean and standard deviation, this function stores the transpose of the data frame of each class as numpy array.
Returns a dictionary with class as key and it dataframe as item

kde_gaussian(x, train_column, bandwidth)

This function calculates the kernel density estimate of each test keypoints using Gaussian distribution
Returns the log of that kernel density estimate

kde_predict(test, model):

This function takes in the test clean file preprocessed in the preprocess function and the train model dictionary generated in the train function.
Returns a list of predicted label of each instance in the test dataset.

Plots:
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import norm

Uses library functions such as hist(), kdeplot() and distplot() to generate plots of histogram, kernel density estimate and fitted normal distribution of the given attribute