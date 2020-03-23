#=============================================
#utf-8   2020-03-10 16:16:19
#Finding the optimal parameters by minimizing AIC

import warnings
import itertools
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
from pandas import read_excel
plt.style.use('fivethirtyeight')

series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 
df = series['K54D']

# Define the p, d and q parameters to take any value between 0 and 2
p = d = q = range(0, 2)
 
# Generate all different combinations of p, q and q triplets
pdq = list(itertools.product(p, d, q))
 
# Generate all different combinations of seasonal p, q and q triplets
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


warnings.filterwarnings("ignore") # specify to ignore warning messages
 
for param in pdq:
    for param_seasonal in seasonal_pdq:
        try:
            mod = sm.tsa.statespace.SARIMAX(df,
                                            order=param,
                                            seasonal_order=param_seasonal,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
 
            results = mod.fit()
 
            print('ARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal, results.aic))
        except:
            continue

#=============================================
#ARIMA

from pandas import read_excel
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm  
plt.style.use('fivethirtyeight')

###############################
series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 
df = series['K54D']

# ARIMA model with (p, d, q)=(1, 1, 1)
mod = sm.tsa.statespace.SARIMAX(df, order=(1,1,1), 
                                seasonal_order=(0,1,1,12))
results = mod.fit(disp=False)
print(results.summary())

# graphical statistics of model (correlogram = ACF plot)
results.plot_diagnostics(figsize=(15, 12))
plt.show()

#============================================
# this code requires the fitted forecasts (for accuracy evaluation) to start 01 Jan 1979.
pred = results.get_prediction(start=pd.to_datetime('2019-01-01'), dynamic=False)
pred_ci = pred.conf_int()

# this code requires the whole plot to start in 1956 (start year of data)
ax = df['2000':].plot(label='Original data')
pred.predicted_mean.plot(ax=ax, label='One-step ahead Forecast', alpha=.7)

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)
plt.legend()
plt.show()

#=============================================
# MSE evaluation
y_forecasted = pred.predicted_mean
y_truth = df['2005-01-01':]
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE for ARIMA is', mse)
#============================================
#Dynamic Forecasting
pred = results.get_prediction(start=pd.to_datetime('2005-01-01'), dynamic=True, full_results=True)
pred_ci = pred.conf_int()

# this code requires the whole plot to start in 1956 (start year of data)
ax = df['2000':].plot(label='Original data')
pred.predicted_mean.plot(ax=ax, label='Dynamic Forecast')

ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')

plt.legend()
plt.show()

#=============================================
# MSE evaluation
y_forecasted = pred.predicted_mean
y_truth = df['2005-01-01':]
# Compute the mean square error
mse = ((y_forecasted - y_truth) ** 2).mean()
print('MSE for Dynamic Forecasting is', mse)

#=============================================
# get forecast one year ahead in future
pred_uc = results.get_forecast(steps=12)
# Get confidence intervals of forecasts
pred_ci = pred_uc.conf_int()

print(pred_uc.predicted_mean)

# plotting forecasts ahead
ax = df.plot(label='Original data')
pred_uc.predicted_mean.plot(ax=ax, label='Forecast values', title='Forecast plot with confidence interval')
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.25)
plt.legend()
plt.show()

#=============================================
