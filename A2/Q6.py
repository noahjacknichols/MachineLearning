import sklearn
import pandas
import numpy as np

from sklearn.neighbors import KDTree
names = ['ID', 'SIZE', 'RENT', 'PRICE']

data = pandas.read_csv('data.csv', names=names)

print(data)

tree = KDTree(data)

#we then query the KDTree for the k-nn of the new node with values size=1000, rent=2200
#for k=1
dist, ind = tree.query([[8., 1000., 2200., -1.]], k=1)
#we add 1 to index to get the ID of the closest property

print(ind+1) 
