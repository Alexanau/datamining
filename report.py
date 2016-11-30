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
from sklearn.model_selection import cross_val_score as cvs
from sklearn.utils import shuffle
from sklearn.neural_network import MLPClassifier,MLPRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegressionCV, PassiveAggressiveClassifier, RidgeClassifierCV
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
import sklearn.metrics as metrics
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from scipy.stats import beta,betaprime,expon,norm,uniform,poisson,randint,geom


class Pair(object):
    def __init__(self,x,y):
        self.x = x
        self.y = y

print('Done!')

if len(argv) < 2:
    print('Error no file given')
    exit(1)
        
print('Loading data from %s...  ' % argv[1],end='',flush=True)
data = np.load(argv[1])
print('Done!\n')
train = Pair(data['train'][:,1:12], data['train'][:,13])
test = Pair(data['test'][:,1:12], data['test'][:,13])

# pre-proccessing 
scaler = pp.MinMaxScaler()
train.x = scaler.fit_transform(train.x)
test.x = scaler.transform(test.x)


adaBoostParams = {'n_estimators':[50,100],
                  'algorithm':['SAMME','SAMME.R']}

mlpParams = {'hidden_layer_sizes':[(100,),(200,200),(500,500)],
             'activation':['identity', 'logistic', 'tanh', 'relu'],
             'solver':['lbfgs', 'sgd', 'adam'],
             'alpha':expon(scale=.0005,loc=.0001),
             'learning_rate' : ['constant', 'invscaling', 'adaptive'],
             'power_t':beta(a=2,b=2),
             'momentum':uniform(),
             'early_stopping':[True],
             'beta_1':beta(a=2,b=.5),
             'beta_2':beta(a=2,b=.5),
             'max_iter':[500]}

passiveAggParams = {'C':betaprime(a=5,b=5), #list(range(0.1,3,0.3)),
                    'n_iter':list(range(5,10)),
                    'loss':['hinge','squared_hinge']}

sgdParams = {'loss':['hinge', 'log', 'modified_huber', 'squared_hinge',
                     'perceptron','squared_loss', 'huber',
                     'epsilon_insensitive','squared_epsilon_insensitive']}

baggingParams = {'n_estimators':[5,10,20],
                 'base_estimator':[None], ####################
                 'max_samples':[1.0,2,.5,.8],
                 'max_features':[1,1.0],
                 'n_jobs':[4]}

extraTreesParams = {'n_estimators':poisson(20),
                    'criterion':['gini','entropy'],
                    'max_features':['sqrt','log2',None],
                    'n_jobs':[2]}

gbcParams = {'loss':['deviance','exponential'],
             'criterion':['friedman_mse','mse','mae'],
             'max_features':['sqrt','log2',None]}

randomForestParams = {'n_estimators':list(range(5,31,5)),
                      'criterion':['gini','entropy'],
                      'max_features':['sqrt','log2',None],
                      'n_jobs':[3]}
knnParams = {'n_neighbors':geom(p=.2,loc=1),
             'weights':['uniform','distance'],
             'p':[1,2,4,8],
             'n_jobs':[-1]}
nearestCentroidParams = {'shrink_threshold':[None] + list(map(lambda x:x/10.,range(1,10)))}

svcParams = {'C':uniform(scale=10),
             'kernel':['linear','poly','rbf','sigmoid'],
             'degree':geom(p=.5)}
dtParams = {'criterion':['gini','entropy'],
            'splitter':['best','random'],
            'max_features':['sqrt','log2',None]}
            

models1 = [GaussianNB(),
           LogisticRegressionCV(),
           RidgeClassifierCV()]

n_iter = 25
print('Making models...',end='',flush=True)
models2 = [GridSearchCV(AdaBoostClassifier(),adaBoostParams),
           RandomizedSearchCV(MLPClassifier(),mlpParams,n_iter=n_iter,n_jobs=2),
           RandomizedSearchCV(PassiveAggressiveClassifier(),passiveAggParams,n_iter=n_iter,n_jobs=2),
           GridSearchCV(SGDClassifier(),sgdParams),
           GridSearchCV(BaggingClassifier(),baggingParams),
           RandomizedSearchCV(ExtraTreesClassifier(),extraTreesParams,n_iter=n_iter,n_jobs=2),
           RandomizedSearchCV(GradientBoostingClassifier(),gbcParams,n_iter=n_iter,n_jobs=2),
           GridSearchCV(RandomForestClassifier(),randomForestParams),
           RandomizedSearchCV(KNeighborsClassifier(),knnParams,n_iter=n_iter,n_jobs=2),
           RandomizedSearchCV(NearestCentroid(),nearestCentroidParams,n_iter=n_iter,n_jobs=2),
           RandomizedSearchCV(SVC(),svcParams,n_iter=n_iter,n_jobs=2),
           GridSearchCV(DecisionTreeClassifier(),dtParams),
           GridSearchCV(ExtraTreeClassifier(),dtParams)]
          
print('Done!')
'''
models2[0].fit(train.x,train.y)
print(models2[0].score(train.x,train.y))
exit(0)
'''

'''
precision
recall
roc
confustion matrix



'''



for i,classifier in enumerate(models2):

    print('classifier: ' + type(classifier.estimator).__name__)
    classifier.fit(train.x,train.y)
    print('params: ' + str(classifier.estimator.get_params()))
    print('test set score: ' + str(classifier.score(test.x,test.y)))
    print('#########')
    '''
    pred = classifer.predict(test.x)
    print(type(classifer).__name__ + ':')
    print(classifer.get_params())
    print(metrics.classification_report(test.y,pred))
    print('#'*52)
'''    
exit(0)


