import csv
import sklearn
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier as Knn
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

class dataSet:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        
def min_max_norm(matrix,col,small=None,big=None):
    if not small:
        small = np.amin(matrix[:,col])
    if not big:
        big = np.amax(matrix[:,col])
    matrix[:,col] = (matrix[:,col]-small)/(big-small)    
    return small,big
    
def z_score_norm(matrix,col,mean=None,std=None):
    if not mean:
        mean = np.mean(matrix[:,col])
    if not std:
        std = np.std(matrix[:,col])
    matrix[:,col] = (matrix[:,col]-mean)/(std)
    return mean,std
    
def readData(filename):
    m = np.loadtxt(filename,delimiter=';',skiprows=1)
    with open(filename,'r') as f:
        reader = csv.reader(f,delimiter=';')
        h=reader.next()
    return h,m

def testClassifier(c,train,test,reg=False):
    if reg:
        c.fit(train.X,train.Y>5)
        return c.score(test.X,test.Y>5)
    else:
        c.fit(train.X,train.Y)
        return c.score(train.X,train.Y)

def testClassifier(c,data,itr=4,reg=False):
    scores = []
    for j in range(itr):
        train,test=splitData(data)
        if not reg:
            train.Y = train.Y>5
            test.Y = test.Y>5
        for i in range(train.X.shape[1]):
            small,big = min_max_norm(train.X,i)
            min_max_norm(test.X,i,small,big)
        c.fit(train.X,train.Y)
        scores.append(c.score(test.X,test.Y))
    return sum(scores)/float(itr)
    
def splitData(data,ratio=.8):
    split_idx = int(data.shape[0] * ratio)
    perm  = np.random.permutation(data)
    train = dataSet(perm[:split_idx,0:-1],perm[:split_idx,-1])
    test  = dataSet(perm[split_idx:,0:-1],perm[split_idx:,-1])    
    return train,test

class majority:
    def fit(self,x,y):
        if sum(y) > sum(y==False):
            self.r = True
        else:
            self.r = False
    def score(self,x,y):
        return sum(y==self.r)/float(y.shape[0])

class linreg:
    def fit(self,x,y):
        X = np.matrix(np.hstack(( x, np.ones((x.shape[0], 1)) )))
        Y = np.matrix(y).T
        self.B = (X.T*X).I*X.T*Y
    def score(self,x,y):
        X = np.matrix(np.hstack(( x, np.ones((x.shape[0], 1)) )))
        Y = np.matrix(y).T
        return np.sum((X*self.B > 5) == (Y>5))/float(Y.shape[0])

class bayes:
    def __init__(self,s):
        self.s=s
    def fit(self,x,y):
        X = x[:,self.s]
        self.c = GaussianNB()
        self.c.fit(X,y)
    def score(self,x,y):
        X = x[:,self.s]
        return self.c.score(X,y)
        
#################
################


headers,data = readData(argv[1])
print 'majority class: ' +str(testClassifier(majority(),data))
print 'Gaussian naive bayes: ' +str(testClassifier(GaussianNB(),data))
print 'Bernoulli Naive bayes: ' +str(testClassifier(BernoulliNB(),data))
s1 = (0,1,2,3,4,5,6,7,8,9,10,0,1,7,6,9)
print 'not so naive Gaussian bayes: ' +str(testClassifier(bayes(s1),data))
print 'linerar regression: '+str(testClassifier(linreg(),data,reg=True))
print 'logistic regression: '+str(testClassifier(LogisticRegression(), data))
print 'logistic regression: '+str(testClassifier(LogisticRegression(C=.3,), data))
exit(0)
for p in range(2,5):
    for k in range(1,50,10):
        c1= Knn(n_neighbors=k, p=p)
        score1 = testClassifier(c1,data)
        c2= Knn(n_neighbors=k, p=p,weights='distance')
        score2 = testClassifier(c2,data)
        print 'knn          k={0:3} p={1:3}: {2}'.format(k,p,score1)
        print 'knn distance k={0:3} p={1:3}: {2}'.format(k,p,score2)


