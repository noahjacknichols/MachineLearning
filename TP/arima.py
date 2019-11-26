import pandas as pd
from pandas import datetime
import pmdarima as pm
from pmdarima.arima import auto_arima, ARIMA
import numpy
import matplotlib.pyplot as plt

from statsmodels.tsa.ar_model import AR
from sklearn.metrics import mean_squared_error


def parser(x):
    return datetime.strptime(x, "%Y-%m-%d")


names = ['Date','Open','High','Low','Close','Volume','OpenInt']

data = pd.read_csv('aapl.us.txt', index_col=0, parse_dates=[0], date_parser=parser,names=names)
# data['Date'] = pd.to_datetime(data['Date'], infer_datetime_format=True)
data.plot()
plt.show()

X = data.values
print(X)


# stepwise_fit = pm.auto_arima(data, start_p=1, start_q=1, max_p=3, max_q=3, trace=True, error_action='ignore', suppress_warnings=True, stepwise=True)

# stepwise_fit.summary()