import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from fbprophet import Prophet
from sklearn.metrics import mean_squared_error
import pmdarima as pm
from pmdarima.arima import auto_arima, ARIMA
import plotly.offline as py 
from fbprophet.plot import plot_plotly
names = ['Date','Open','High','Low','Close','Volume','OpenInt']
data = pd.read_csv('amd.us.txt', names = names)
print(data)
data.describe()

# data.plot()
# plt.show()
#divide into train and validation set
# train = data[:int(0.7*(len(data)))]
# valid = data[int(0.7*(len(data))):]
x = pd.DataFrame(data, columns = ['Open'])
x.plot()
plt.show()
x = x[len(x)-90:]
valid = x[len(x)-30:]
train = x[:len(x)-30]
#preprocessing (since arima takes univariate series as input)
proph_data = pd.read_csv('abc.us.txt', names = ['ds', 'y', 'a','b', 'c', 'd','e'])
proph_train = pd.DataFrame(proph_data, columns = ['ds', 'y'])
proph_train['ds'] = pd.to_datetime(proph_train['ds'])
proph_train['y'] = proph_train['y'].astype(float)
proph_train.dropna(axis=0, how='any', thresh=None, subset=None, inplace=True)
proph_train = proph_train[len(proph_train)-720:len(proph_train)-360]
print("len of proph train:",len(proph_train))
# train.drop('Month',axis=1,inplace=True)
# valid.drop('Month',axis=1,inplace=True)

#plotting the data


# x.plot()
proph_model = Prophet()
proph_model.fit(proph_train)
proph_forecast = proph_model.make_future_dataframe(periods=360, freq="D")
proph_forecast = proph_model.predict(proph_forecast)
print("PROPHET FORECAST:")
# print(proph_forecast)
print("--------------------------------")

py.init_notebook_mode()
fig2 = proph_model.plot(proph_forecast)
fig2 = proph_model.plot_components(proph_forecast)
# fig = plot_plotly(proph_model,proph_forecast)
# py.iplot(fig)
# proph_forecast.plot()
# plt.show()

print(proph_forecast.tail())


model = auto_arima(train, seasonal=True, start_p=1, start_q=1, max_p=3, max_q=3, error_action = 'ignore', suppress_warnings=True)



fitted = model.fit(train)
print(fitted.summary())
forecast = fitted.predict(n_periods=30)
model.plot_diagnostics(figsize=(7,5))
plt.show()
print(forecast)
print(valid)
rms = sqrt(mean_squared_error(valid,forecast))
print(rms)