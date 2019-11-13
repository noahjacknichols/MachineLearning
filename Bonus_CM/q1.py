from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt

names = ['good behavior', 'age<30', 'drugdependent', 'recidivist']
targetNames = ['False','True']
#Import the dataset 
dataset = pd.read_csv('q1.txt', names = names)


# train_features = dataset.iloc[:80,:-1]
# test_features = dataset
# train_targets = dataset.iloc[:80,-1]
# test_targets = dataset.iloc[80:,-1]

# print(dataset)
d = pd.DataFrame(dataset, columns = ['good behavior', 'age<30', 'drugdependent'])
t = pd.DataFrame(dataset, columns = ['recidivist'])
plot = DecisionTreeClassifier(criterion = 'entropy').fit(d,t)
ree = tree.plot_tree(plot, filled = True, feature_names=names, class_names = targetNames)
plt.show()
print("after tree")


#Predict subquestion B
print("Prediction for " + names[0] + ": False, " + names[1] + ": False, " + names[2]+ ": True:")
ans = plot.predict([[False, False, True]])
print(ans[0])

#predict subquestion C
print("Prediction for " + names[0] + ": True, " + names[1] + ": True, " + names[2]+ ": False:")
ans = plot.predict([[True, True, False]])
print(ans[0])




# print("The prediction accuracy is: ",tree.score(test_features,test_targets)*100,"%")