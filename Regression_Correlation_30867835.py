#utf-8  Multi Linear Regration Model
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import read_excel
import numpy as np

new_adv_data = read_excel('forecasting.xls', index_col=0, parse_dates=True, squeeze=True)
#Get the dataset we need and look at the first few columns and data shapes
print('head:',new_adv_data.head(),'\nShape:',new_adv_data.shape)

#data description
print(new_adv_data.describe())
#Missing value test
print(new_adv_data[new_adv_data.isnull()==True].count())
 
new_adv_data.boxplot()
plt.savefig("boxplot.jpg")
plt.show()

#correlation matrix
print(new_adv_data.corr())
 
#Create a scatter chart to see the data distribution in the dataset
sns.pairplot(new_adv_data, x_vars=['K54D','EAFV','K226','JQ2J'], y_vars='FTSE', size=7, aspect=0.8,kind = 'reg')
plt.savefig("pairplot.jpg")
plt.show()
 
#Use the package in sklearn to partition the data set, so as to create training set and test set
X_train,X_test,Y_train,Y_test = train_test_split(new_adv_data[['K54D','K226','JQ2J']],new_adv_data['FTSE'],train_size=.8)
 
print("Original data characteristics:",new_adv_data.shape,
      ",Training data characteristics:",X_train.shape,
      ",Test data characteristics:",X_test.shape)
 
print("Original data label:",new_adv_data['FTSE'].shape,
      ",Training data label:",Y_train.shape,
      ",Test data label:",Y_test.shape)

#Descriptive statistics points the existence of influential points; therefore, 
#Fourier function is used to converge data moreover improve linear regression coefficient.
#The value of parameters are cauculated by cftool of Matlab
a0 =   6285 
a1 =   376.7 
b1 =  -888.7  
w =   0.0002479  
new_adv_data['JQ2J']= a0 + a1*np.cos(new_adv_data['JQ2J']*w) + b1*np.sin(new_adv_data['JQ2J']*w)

a0 =    5744  
a1 =   -616.6 
b1 =    656.1  
w =     0.026  
new_adv_data['K226']= a0 + a1*np.cos(new_adv_data['K226']*w) + b1*np.sin(new_adv_data['K226']*w)

a0 =    6105  
a1 =    -1120 
b1 =    -51.58 
w =     0.0181 
new_adv_data['K54D']= a0 + a1*np.cos(new_adv_data['K54D']*w) + b1*np.sin(new_adv_data['K54D']*w)

#create training set and test set
X_train,X_test,Y_train,Y_test = train_test_split(new_adv_data[['K54D','K226','JQ2J']],new_adv_data['FTSE'],train_size=.95)
print('X_train',X_train)

#build the model
model = LinearRegression()
model.fit(X_train,Y_train)
a  = model.intercept_#intercept
b = model.coef_#regression coefficient
print("Best Fitting Curve:Intercept",a,",regression coefficient：",b)
 
#predict using linear regression
Y_pred = model.predict(X_test)
print(Y_pred)

#calculate MSE
from sklearn.metrics import mean_squared_error 
MSE=mean_squared_error(Y_pred, Y_test)
print('MSE of Linear Regression is' ,MSE)

#print testing curve
plt.figure()
plt.plot(range(len(Y_pred)),Y_pred,'b',label="predict")
plt.plot(range(len(Y_pred)),Y_test,'r',label="test dataset")
plt.legend(loc="upper right") #Show labels in diagram
plt.xlabel("tag")
plt.ylabel('value of FTSE')
plt.show()

#predict value of FTSE in 2020 using linear regression
X_2020 = read_excel('ForecastingTable.xls', index_col=0, parse_dates=True, squeeze=True)
Y_2020 = model.predict(X_2020)
print(Y_2020)

#Moving average (n=4) are calculated using Excel
from matplotlib import pyplot
FTSE_2020 = read_excel('fore_y.xls', index_col=0, parse_dates=True, squeeze=True)
FTSE_2020.plot(title='Predict for FTSE after smothing',color='orange', legend=True)
new_adv_data['FTSE'].plot(color='blue', legend=True)
pyplot.show()