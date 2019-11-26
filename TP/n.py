import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from pmdarima.arima import auto_arima, ARIMA
names = ['Date','Open','High','Low','Close','Volume','OpenInt']
data = pd.read_csv('aapl.us.txt', names = names)
print(data)
data.plot()
plt.show()
#divide into train and validation set
# train = data[:int(0.7*(len(data)))]
# valid = data[int(0.7*(len(data))):]
x = pd.DataFrame(data, columns = ['Open'])
x = x[len(x)-90:]
valid = x[len(x)-30:]
train = x[:len(x)-30]
#preprocessing (since arima takes univariate series as input)

# train.drop('Month',axis=1,inplace=True)
# valid.drop('Month',axis=1,inplace=True)

#plotting the data


# x.plot()
model = auto_arima(train, seasonal=True, start_p=1, start_q=1, max_p=3, max_q=3, error_action = 'ignore', suppress_warnings=True)

fitted = model.fit(train)
print(fitted.summary())
forecast = fitted.predict(n_periods=30)
# print(forecast)
# forecast = pd.DataFrame(forecast,index = valid.index, columns = ['Open'])
# d1 = train+valid
# ax = plt.gca()
# d1.plot(ax=ax)
# d2 = train + forecast
# d2.plot(ax=ax)
# print(d1)
# print(d2)
# plt.plot(train+valid,train+forecast,x)
# plt.plot(train+valid, label='Train')
# # plt.plot(valid, label='Valid')
# plt.plot(train+forecast, label='Prediction')
# plt.plot(x, label='Real Data')
model.plot_diagnostics(figsize=(7,5))
plt.show()
print(forecast)
print(valid)
rms = sqrt(mean_squared_error(valid,forecast))
print(rms)