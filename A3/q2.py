from keras.models import Sequential
from keras.layers import Dense, Activation

import pandas as pd
import numpy as np


names = ['Dose1', 'Dose2', 'Class']
data = pd.read_csv('data.txt', names = names)
X = pd.DataFrame(data, columns = names[:2])
Y = pd.DataFrame(data, columns = [names[2]])
model = Sequential()

model.add(Dense(4, Activation = 'tanh'))

model.compile(optimizer='rmsprop', loss='mse')


score = model.evalute(X, Y, batch_size=1)