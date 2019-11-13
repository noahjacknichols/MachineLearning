from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt
import scipy.stats
import math

names = ['AGE', 'EDUCATION', 'MARITAL STATUS', 'OCCUPATION', 'ANNUAL INCOME']
dataset = pd.read_csv('q2.txt', names = names)
def gini(labels):
    value, counts = np.unique(labels, return_counts=True)
    s = 0
    for count in counts:
        s +=count
    val = 0
    for count in counts:
        val += (count/s) * (count/s)
    
    return 1 - val

def ent(labels, base=None):
    print("LABELS:",labels)
    value, counts = np.unique(labels, return_counts=True)
    print(value,counts)
    return scipy.stats.entropy(counts, base=base)

def IG(labels, incLabels, setEnt):
    incValue, incCounts = np.unique(labels, return_counts=True)
    for label in incValue:
        

    return

def ent_IGR(labels, base=None):
    value, counts = np.unique(labels, return_counts=True)
    s = sum(count for count in counts)
    val = 0
    for count in counts:
        val += count/s * math.log(count/s,2)
    return -val


data_Entropy = ent(pd.DataFrame(dataset, columns=[names[1]]),2)
gini = (gini(pd.DataFrame(dataset, columns=[names[4]])))
print(data_Entropy)
print(gini)
IGR = {}
for name in names:
    IGR[name] = []
    IGR[name].append(ent(pd.DataFrame(dataset, columns=[name]),2))
    IGR[name].append(IG(pd.DataFrame(dataset, columns =[name]),pd.DataFrame(dataset, columns =[names[4]]),data_Entropy))

