import sklearn
import pandas
import numpy as np

from sklearn.neighbors import KDTree
names = ['ID', 'SIZE', 'RENT', 'PRICE']

data = pandas.read_csv('data.csv', names=names)

print(data)

tree = KDTree(data)
# k = 3
dist, ind = tree.query([[8., 1000., 2200., -1.]], k=3)

# print(dist)
print(ind)