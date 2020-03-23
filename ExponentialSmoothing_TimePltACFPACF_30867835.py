#=======================================
#Preliminary Analysis
'''
Excel took the data of four variables (K54D, EAFV, K226 and JQ2J) 
from January 2000 to December 2019,
 next, reduced the value of JQ2J by 100 times 
 so that it is on the same scale as the other three variables.
'''
# ---Time Plot---
from pandas import read_excel
from matplotlib import pyplot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 
series.plot()
pyplot.show()

# ---ACF & PACF---
# ACF plot on 50 time lags
plot_acf(series['K54D'], title='ACF of K54D', lags=50)
plot_acf(series['EAFV'], title='ACF of EAFV', lags=50)
plot_acf(series['K226'], title='ACF of K226', lags=50)
plot_acf(series['JQ2J(*.01)'], title='ACF of JQ2J(*.01)', lags=50)

# PACF plot on 50 time lags
plot_pacf(series['K54D'], title='PACF of K54D', lags=50)
plot_pacf(series['EAFV'], title='PACF of EAFV', lags=50)
plot_pacf(series['K226'], title='PACF of K226', lags=50)
plot_pacf(series['JQ2J(*.01)'], title='PACF of JQ2J(*.01)', lags=50)

pyplot.show()

#=======================================
#utf-8  2020-03-09 16:47:04
#Exponential Smoothing for EAFV
from pandas import read_excel
from statsmodels.tsa.api import ExponentialSmoothing
from matplotlib import pyplot
series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 

# Holt method
fit3 = ExponentialSmoothing(series['EAFV'], seasonal='add').fit(use_boxcox=True)
Forecasting = fit3.forecast(12).rename("Forecasting")
Forecasting.plot( color='orange', legend=True)
series['EAFV'].plot(title='Exponential Smoothing for EAFV',color='blue', legend=True)
fit3.fittedvalues.plot( color='orange')
pyplot.show()
print(Forecasting)

#Calculate MSE
from sklearn.metrics import mean_squared_error 
MSE=mean_squared_error(fit3.fittedvalues, series['EAFV'])
print(MSE)


#Seperate EAFV to Testing dataset and training dataset
train= series['EAFV'][0:228]
test = series['EAFV'][228:241]

#Using previous data forecast data in 2019 and compare with observations in 2019
fittset = ExponentialSmoothing(train, seasonal='add').fit(use_boxcox=True)
fcasttest = fittset.forecast(12).rename("Forecasting")
fcasttest.plot(color='orange', legend=True)
test.plot(title='Testing for EAFV',color='blUE', legend=True)
pyplot.show()

MSE=mean_squared_error(fcasttest, test)
print('Testing MSE for EAFV =', MSE)
#=======================================

#utf-8  2020-03-09 16:47:04
#Exponential Smoothing for JQ2J
from pandas import read_excel
from statsmodels.tsa.api import ExponentialSmoothing
from matplotlib import pyplot
series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 

# Holt method
fit3 = ExponentialSmoothing(series['JQ2J(*.01)'], trend='add', seasonal='mul').fit(use_boxcox=True)
Forecasting = fit3.forecast(12).rename("Forecasting")
Forecasting.plot( color='orange', legend=True)
series['JQ2J(*.01)'].plot(title='Exponential Smoothing for JQ2J(*.01)',color='blue', legend=True)
fit3.fittedvalues.plot( color='orange')
pyplot.show()
print(Forecasting)

#Calculate MSE
from sklearn.metrics import mean_squared_error 
MSE=mean_squared_error(fit3.fittedvalues, series['JQ2J(*.01)'])
print(MSE)

#Seperate JQ2J(*.01) to Testing dataset and training dataset
train= series['JQ2J(*.01)'][0:228]
test = series['JQ2J(*.01)'][228:241]

#Using previous data forecast data in 2019 and compare with observations in 2019
fittset = ExponentialSmoothing(train, trend='add', seasonal='mul').fit(use_boxcox=True)
fcasttest = fittset.forecast(12).rename("Forecasting")
fcasttest.plot(color='orange', legend=True)
test.plot(title='Testing for JQ2J(*.01)',color='blUE', legend=True)
pyplot.show()

MSE=mean_squared_error(fcasttest, test)
print('Testing MSE for JQ2J(*.01) =', MSE)
#=======================================

#utf-8  2020-03-09 16:47:04
#Exponential Smoothing for K54D
from pandas import read_excel
from statsmodels.tsa.api import ExponentialSmoothing
from matplotlib import pyplot
series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 

# Holt method
fit3 = ExponentialSmoothing(series['K54D'], trend='add', seasonal='add').fit(use_boxcox=True)
Forecasting = fit3.forecast(12).rename("Forecasting")
Forecasting.plot(color='orange', legend=True)
series['K54D'].plot(title='Exponential Smoothing for K54D', color='blue', legend=True)
fit3.fittedvalues.plot( color='orange')
pyplot.show()
print(Forecasting)

#Calculate MSE
from sklearn.metrics import mean_squared_error 
MSE=mean_squared_error(fit3.fittedvalues, series['K54D'])
print(MSE)

#Seperate K54D to Testing dataset and training dataset
train= series['K54D'][0:228]
test = series['K54D'][228:241]

#Using previous data forecast data in 2019 and compare with observations in 2019
fittset = ExponentialSmoothing(train, trend='add', seasonal='add').fit(use_boxcox=True)
fcasttest = fittset.forecast(12).rename("Forecasting")
fcasttest.plot(color='orange', legend=True)
test.plot(title='Testing for K54D',color='blUE', legend=True)
pyplot.show()

MSE=mean_squared_error(fcasttest, test)
print('Testing MSE for K54D =', MSE)
#=======================================

#utf-8  2020-03-09 16:28:08
#Exponential Smoothing for K226
from pandas import read_excel
from statsmodels.tsa.api import Holt
from matplotlib import pyplot
series = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True) 

# Holt model
fit4 = Holt(series['K226']).fit(optimized=True)
fcast4 = fit4.forecast(12).rename("Forecasting")
fcast4.plot(color='orange', legend=True)
series['K226'].plot(title='Exponential Smoothing for K226',color='blUE', legend=True)
fit4.fittedvalues.plot( color='orange')
pyplot.show()
print(fcast4)

#Calculate MSE
from sklearn.metrics import mean_squared_error 
MSE=mean_squared_error(fit4.fittedvalues, series['K226'])
print(MSE)

#Seperate K226 to Testing dataset and training dataset
train= series['K226'][0:228]
test = series['K226'][228:241]

#Using previous data forecast data in 2019 and compare with observations in 2019
fittset = Holt(train).fit(optimized=True)
fcasttest = fittset.forecast(12).rename("Forecasting")
fcasttest.plot(color='orange', legend=True)
test.plot(title='Testing for K226',color='blUE', legend=True)
pyplot.show()

MSE=mean_squared_error(fcasttest, test)
print('Testing MSE for K226 =', MSE)
#=======================================