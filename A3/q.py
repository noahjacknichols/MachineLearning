
import pandas as pd
import numpy as np

from sklearn.svm import SVC


clf = SVC(gamma='auto', degree=2)
names = ['Dose1', 'Dose2', 'Class']
data = pd.read_csv('data.txt', names = names)
weights = [7.1655,6.9060,2.0033,6.1144,5.9538]
X = pd.DataFrame(data, columns = names[:2])
Y = pd.DataFrame(data, columns = [names[2]])
clf.fit(X,np.ravel(Y), weights)

clf2 = SVC(gamma='auto', kernel='poly', degree=2)
clf2.fit(X,np.ravel(Y), weights)
print("prediction:", clf.predict([[0.90, -0.90]])[0])
print("prediction:", clf2.predict([[0.22, 0.16]])[0])

print("prediction:", clf2.predict([(0.90, -0.90)])[0])
print("prediction:", clf.predict([(0.22, 0.16)])[0])
