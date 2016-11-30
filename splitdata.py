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
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
print('Done!')

if len(argv) < 2:
    print('Error no file given')
    exit(1)

print('Loading data from %s...  ' % argv[1],end='',flush=True)
data = np.loadtxt(argv[1],delimiter=',', skiprows=1,
                  converters={13: lambda s:float(s == b'"High"'),
                              0:  lambda s:float(s.strip(b'"'))})

print('Done!')

train,test =  train_test_split(data)
fmt = '%.18g'
np.savez_compressed('train_test',train=train,test=test)
