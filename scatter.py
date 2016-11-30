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
def key_event(e):
    global x_idx
    global y_idx
    if e.key == "right":
        x_idx += 1
    elif e.key == "left":
        x_idx -= 1
    elif e.key == "up":
        y_idx += 1
    elif e.key == "down":
        y_idx -= 1
    elif e.key == "q":
        exit(0)
    else:
        return
    x_idx = x_idx % good.shape[1]
    y_idx = y_idx % good.shape[1]

    ax.cla()
    ax.set_ylabel(h[y_idx])
    ax.set_xlabel(h[x_idx])
    ax.scatter(bad[:,x_idx], bad[:,y_idx], marker='x', color='red', alpha=.6)
    ax.scatter(good[:,x_idx], good[:,y_idx], marker='o', color='blue',alpha=.1)

    txt.set_text('x: ' + h[x_idx] + ' vs ' + 'y: ' + h[y_idx])
    fig.canvas.draw()



h,m = readData(argv[1])
for c in range(11):
    z_score_norm(m,c)
good,bad = class_split(m)


fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
ax.set_ylabel(h[y_idx])
ax.set_xlabel(h[x_idx])

txt = fig.suptitle(h[x_idx] +' vs '+h[y_idx])


ax.scatter(bad[:,x_idx], bad[:,y_idx], marker='x', color='red',alpha=.6)
ax.scatter(good[:,x_idx], good[:,y_idx], marker='o', color='blue',alpha=.1)
plt.show()


