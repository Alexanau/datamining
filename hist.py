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
    
h,m = readData(argv[1])

for c in range(11):
    pass #z_score_norm(m,c)
good,bad = class_split(m)

curr_pos = 0
bin_ct = 20
def key_event(e):
    global curr_pos
    global bin_ct
    if e.key == "right":
        curr_pos = curr_pos + 1
    elif e.key == "left":
        curr_pos = curr_pos - 1
    elif e.key == "up":
        bin_ct+=1
    elif e.key == "down":
        bin_ct -= 1
    elif e.key == "q":
        exit(0)
    else:
        return
    curr_pos = curr_pos % good.shape[1]
    ax.cla()
    ax.hist(good[:,curr_pos],bins=bin_ct,label='Good')
    ax.hist(bad[:,curr_pos],bins=bin_ct,label='Bad',alpha=0.5,color='red')
    txt.set_text(h[curr_pos] + ' bins = ' + str(bin_ct))    
    fig.canvas.draw()

fig = plt.figure()
fig.canvas.mpl_connect('key_press_event', key_event)
ax = fig.add_subplot(111)
txt = fig.suptitle(h[0])
ax.hist(good[:,0], bins=bin_ct,label='Good')
plt.show()
