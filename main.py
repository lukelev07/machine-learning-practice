__author__ = 'lukelev07'


import csv
import sys
from numpy import *
import numpy as np


def classify(x, t_tupes):
    min_dist = sys.maxint
    min_feature = sys.maxint
    x_vectorized = np.array(x)
    for label, value in t_tupes:
        t_vectorized = np.array(value)
        result = np.linalg.norm(x_vectorized - t_vectorized)
        if result < min_dist:
            min_feature = label
    return min_feature


# open training data
training_set_file = open('hw12data/digitsDataset/trainFeatures.csv','r')
training_set_iter = csv.reader(training_set_file)
training_set = [item for item in training_set_iter]

# open labels
training_label_file = open('hw12data/digitsDataset/trainLabels.csv','r')
training_label_iter = csv.reader(training_label_file)
training_label = [item for item in training_label_iter]

# pair label with training data
t_set = []
for index, val in enumerate(training_label):
    t_set.append((val, training_set[index]))

# open vals
val_file = open('hw12data/digitsDataset/valFeatures.csv','r')
val_iter = csv.reader(val_file)
val_set = [item for item in val_iter]

for image in val_set:
    print(classify(image, t_set))

