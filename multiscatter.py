import csv
import numpy as np
import matplotlib.pyplot as plt
from sys import argv

def class_split(matrix,col=None,gt=5):
    if not col:
        col = matrix.shape[1] -1
    good = matrix[matrix[:,col] > gt]
    bad = matrix[matrix[:,col] <= gt]
    return good,bad
    
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
    

x_idx = 0
y_idx = 0
h,m = readData(argv[1])
for c in range(11):
    min_max_norm(m,c)
good,bad = class_split(m)

# fig = plt.figure()
# fig.canvas.mpl_connect('key_press_event', key_event)
# ax = fig.add_subplot(111)
# txt = fig.suptitle(h[x_idx] +' vs '+h[y_idx])


# 
# 
X,Y = 11,11
f, ax = plt.subplots(nrows=X,ncols=Y,sharex=True,sharey=True,figsize=(X,Y))

for x,row in enumerate(ax):
    
    for y,col in enumerate(row):
        ax[y,x].scatter(bad[:,x], bad[:,y], marker='x', color='red',alpha=.6)
        ax[y,x].scatter(good[:,x], good[:,y], marker='o', color='blue',alpha=.1)
        col.set_xlim([0,1])
        col.set_ylim([0,1])
        

        if x == len(ax) -1:
            col.set_xlabel(h[y],rotation=35)
        if y == 0:
            col.set_ylabel(h[x],rotation=0,horizontalalignment='right')
            #l.set_rotation(0)

        
plt.gca().axes.get_xaxis().set_ticks([])
plt.gca().axes.get_yaxis().set_ticks([])
plt.show()



