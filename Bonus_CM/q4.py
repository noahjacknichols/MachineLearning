from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt
import sklearn
lab_enc = preprocessing.LabelEncoder()
russiaQuery = [[67.62,31.68,10.00,3.87,12.90,7]]
names = ['LIFE EXP.', 'TOP-10 INCOME', 'INFANT MORT.', 'MIL SPEND', 'SCHOOL YEARS', 'CPI']
dataset = pd.read_csv('q4.txt', names=names)
d = pd.DataFrame(dataset, columns = ['LIFE EXP.', 'TOP-10 INCOME', 'INFANT MORT.', 'MIL SPEND', 'SCHOOL YEARS'])
t = pd.DataFrame(dataset, columns = ['CPI'])

lab_enc.fit_transform(t)
neigh = KNeighborsClassifier(n_neighbors=3)
print(d)
print(t)
print(np.ravel(t))
neigh.fit(d,np.ravel(t))
print(neigh.predict(russiaQuery))