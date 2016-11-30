#!/usr/bin/env python3
'''
preformance mesures computed:
accuracy
ROC
F-mesure
Precision
Recal
confusion matrix

'''
print('Loading modules...  ',end='',flush=True)
from sys import argv
import numpy as np
import sklearn.preprocessing as pp
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, PassiveAggressiveClassifier, RidgeClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import sklearn.metrics as metrics


data = np.loadtxt(argv[1],delimiter=',', skiprows=1,
#                      usecols=(1,2,3,4,5,6,7,8,9,10,11,13),
                      converters={13: lambda s:float(s == b'"High"'),
                                  0:  lambda s:float(s.strip(b'"'))})



