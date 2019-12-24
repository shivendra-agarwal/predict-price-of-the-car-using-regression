# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import os
#to work with dataframes
import pandas as pd

#to perfom numerical operations
import numpy as np

#to visualize data
import seaborn as sns

#setting dimenaions for the plots
sns.set(rc={'figure.figsize':(11.7,8.27)})

#import data from csv
cars_data=pd.read_csv("cars_sampled.csv")
 
#copy of data
cars=cars_data.copy()

#structure of the data
cars.info()

#summerizing data
cars.describe()
#change decimal places 
pd.set_option('display.float_format', lambda x: '%.3f' % x)
#change number of columns displayed in console
pd.set_option('display.max_columns',100)
#summarizing data again
cars.describe()

#dropping unwanted columns
col=['name','dateCrawled','dateCreated','postalCode','lastSeen']
cars=cars.drop(columns=col,axis=1)

#removing duplicate records
cars.drop_duplicates(keep='first',inplace=True)
#470 records deleted

#data cleaning

#No. of missing values
cars.isnull().sum()

#variable yearOfRegistration
yearwise_count=cars['yearOfRegistration'].value_counts().sort_index()
#group yearOfRegistration by frequency
#the yearOfRegistration are more than 2019, which is impossible as per now
sum(cars['yearOfRegistration'] > 2019)
#24
sum(cars['yearOfRegistration'] < 1950)
#38
sns.regplot(x='yearOfRegistration', y='price', data=cars, scatter=True, fit_reg=False)
#we set our workign range from 1950 - 2019

#varaible price
price_count=cars['price'].value_counts().sort_index()
sns.distplot(cars['price'])
cars['price'].describe()
sns.boxplot(y=cars['price'])
sum(cars['price'] > 150000)
sum(cars['price'] < 100)
#working range set 100 - 150000

#variable powerPS
power_count=cars['powerPS'].value_counts().sort_index()
sns.distplot(cars['powerPS'])
cars['powerPS'].describe()
sns.regplot(x='powerPS', y='price', scatter=True, fit_reg=False, data=cars)

sum(cars['powerPS'] > 500)
sum(cars['powerPS'] < 10)
#working range 10 - 500

#working range of data

cars = cars[
        (cars['yearOfRegistration'] <= 2018)
      & (cars['yearOfRegistration'] >= 1950)
      & (cars['price'] >= 100)
      & (cars['price'] <= 150000)
      & (cars['powerPS'] >= 10)
      & (cars['powerPS'] <= 500)]

#roughly around 67000 records dropped

#combining yearOfRegistration and monthOfRegistration as Age 
cars['monthOfRegistration']/=12

#creating Age variable
cars['Age']=(2019-cars['yearOfRegistration'])+cars['monthOfRegistration']
cars['Age']=round(cars['Age'],2)
cars['Age'].describe()

#dropping yearOfRegistration and monthOfRegistration
cars=cars.drop(columns=['yearOfRegistration','monthOfRegistration'],axis=1)


#visualize parameters

#Age
sns.distplot(cars['Age'])
sns.boxplot(y=cars['Age'])

#price
sns.distplot(cars['price'])
sns.boxplot(y=cars['price'])

#powerPS
sns.distplot(cars['powerPS'])
sns.boxplot(y=cars['powerPS'])

#visualize now with new data

#Age vs price
sns.regplot(x='Age', y='price', data=cars, scatter=True, fit_reg=False)
#cars priced higher are newer
#with increase in age, price decreases
#however, some cars are priced higher with the age increased, we can treat them as vintage

#powerPS vs price
sns.regplot(x='powerPS', y='price', data=cars, scatter=True, fit_reg=False)

#variable seller
cars['seller'].value_counts()
pd.crosstab(cars['seller'], columns='count', normalize=True)
sns.countplot(x='seller', data=cars)
#fewer cars have 'commercial' => insignificant

#variable offerType
cars['offerType'].value_counts()
sns.countplot(x = 'offerType', data=cars)
#All cars have 'offerType' => insignificant

#variable abtest
cars['abtest'].value_counts()
pd.crosstab(cars['abtest'], columns='count', normalize=True)
sns.countplot(x='abtest', data=cars)
#equally distributed
sns.boxplot(x='abtest', y='price', data=cars)
#for every price value there is almost 50-50 distribution
#does not affect price => insignificant

#variable vehicleType
cars['vehicleType'].value_counts()
pd.crosstab(cars['vehicleType'], columns='count', normalize=True)
sns.countplot(x='vehicleType', data=cars)
sns.boxplot(x='vehicleType', y='price', data=cars)
# 8 types 
#vehicle type affects price

#variable gearbox
cars['gearbox'].value_counts()
pd.crosstab(cars['gearbox'], columns='count', normalize=True)
sns.countplot(x='gearbox', data=cars)
sns.boxplot(x='gearbox', y='price', data=cars)
#gearbox affects price


#variable model
cars['model'].value_counts()
pd.crosstab(cars['model'], columns='count', normalize=True)
sns.countplot(x='model', data=cars)
#cars are distributed over many models
#considered in modelling

#variable kilometer
cars['kilometer'].value_counts().sort_index()
pd.crosstab(cars['kilometer'], columns='counts', normalize=True)
sns.boxplot(x='kilometer', y='price', data=cars)
cars['kilometer'].describe()

sns.distplot(cars['kilometer'], bins=8, kde=False)
sns.regplot(x='kilometer', y='price', scatter=True, fit_reg=False, data=cars)
#considered in modelling

#variable fuelType
cars['fuelType'].value_counts()
pd.crosstab(cars['fuelType'], columns ='count', normalize=True)
sns.countplot(x='fuelType', data=cars)
sns.boxplot(x='fuelType', y='price', data=cars)
#fuelType affects price

#variable brand
cars['brand'].value_counts()
pd.crosstab(cars['brand'], columns='count', normalize=True)
sns.countplot(x='brand',data=cars)
sns.boxplot(x='brand', y='price', data=cars)
#cars are distributed over many brands
#brand affects price

#varaible notRepairedDamage
#yes - car is damanged but not rectified
#no - car is damaged but not rectified
cars['notRepairedDamage'].value_counts()
pd.crosstab(cars['notRepairedDamage'], columns='count', normalize=True)
sns.countplot(x='notRepairedDamage', data=cars)
sns.boxplot(x='notRepairedDamage', y='price', data=cars)
#as expected the cars that reqiure the damages to be repaired


#removing insignificant variables
col=['seller', 'offerType', 'abtest']
cars=cars.drop(columns=col, axis=1)
cars_copy=cars.copy()

#correlation

cars_select1=cars.select_dtypes(exclude=[object])
correlation=cars_select1.corr()
round(correlation, 3)
cars_select1.corr().loc[:,'price'].abs().sort_values(ascending=False)[1:]


"""
linear regression and random forest on two dataset
1. data obtained by omitting rows with missing value
2. data obtained by inputting the missing values

"""

#omitting missing values

cars_omits=cars.dropna(axis=0)

#converting categorical variables to dummy varaibles
cars_omit=pd.get_dummies(cars_omits, drop_first=True)

#importing necessary libraries

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

#model building

#seperating input and output features
y1 = cars_omit['price']
x1 = cars_omit.drop(['price'], axis='columns', inplace=False)


#plotting the varaible price
prices = pd.DataFrame({"1. Before":y1,"2. After":np.log(y1)})
prices.hist()

#transforming price as a logarithmic value
y1 = np.log(y1)

#splitting data into test and train
X_train, X_test, y_train, y_test = train_test_split(x1, y1, test_size=0.3, random_state=3)
print(X_train.shape, X_test.shape, y_train.shape,y_test.shape)


#baseline model for omitted data

"""
set a benchmark and to compare with our regression model
"""
#finding the mean for test data value

base_pred=np.mean(y_test)
print(base_pred)

#repeating same value till length of test data
base_pred=np.repeat(base_pred, len(y_test))

#finding the RMSE
base_root_mean_squared_error=np.sqrt(mean_squared_error(y_test, base_pred))

#print
print(base_root_mean_squared_error)

#linear regression with omitted data

#setting intercept as true
lgr=LinearRegression(fit_intercept=True)

#model
model_lin1=lgr.fit(X_train,y_train)

#predicting model on test set
cars_predictions_lin1=lgr.predict(X_test)

#computing MSE and RMSE
lin_mse1=mean_squared_error(y_test, cars_predictions_lin1)
lin_rmse1=np.sqrt(lin_mse1)
print(lin_rmse1)

#R Squared value
r2_lin_test1=model_lin1.score(X_test, y_test)
r2_lin_train1=model_lin1.score(X_train, y_train)
print(r2_lin_test1, r2_lin_train1)

#regession diagnostics - Residual plot analysis
residuals1=y_test-cars_predictions_lin1
sns.regplot(x=cars_predictions_lin1, y=residuals1, scatter=True, fit_reg=False)
residuals1.describe()

#random forest with omitted data

#model parameters
rf = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, min_samples_split=10, min_samples_leaf=4, random_state=1)


#model
model_rf1=rf.fit(X_train, y_train)

#predicting model on test set
cars_prediction_rf1 = rf.predict(X_test)

#computing model on test set
cars_prediction_rf1=rf.predict(X_test)

#computing MSE and RMSE
rf_mse1 = mean_squared_error(y_test, cars_prediction_rf1)
rf_rmse1 = np.sqrt(rf_mse1)
print(rf_rmse1)
#R squared value
r2_rf_test1 = model_rf1.score(X_test, y_test)
r2_rf_train1 = model_rf1.score(X_train, y_train)
print(r2_rf_test1, r2_rf_train1)

#model building with inputed data

cars_imputed = cars.apply(lambda x:x.fillna(x.median()) \
                         if x.dtype=='float' else \
                         x.fillna(x.value_counts().index[0]))
cars_imputed.isnull().sum()

#converting categorical variables to dummy variables
cars_imputed = pd.get_dummies(cars_imputed, drop_first=True)

#seperating input and output feature
x2=cars_imputed.drop(['price'], axis='columns', inplace=False)
y2 = cars_imputed['price']

#plotting the variable price
prices = pd.DataFrame({"1. Before":y2, "2. After":np.log(y2)})
prices.hist()

#transforming price as a logarithmic value
y2=np.log(y2)

#splitting data into test and train
X_train1, X_test1, y_train1, y_test1 = train_test_split(x2, y2, test_size=0.3, random_state = 3)
print(X_train1.shape, X_test1.shape, y_train1.shape, y_test1.shape)


#baseline model for imputed data
#creating benchmark and to compare our regression model

#finding the mean for test data value
base_pred = np.mean(y_test1)
print(base_pred)

#repeating same value till length of test data
base_pred = np.repeat(base_pred, len(y_test1))

#finding RMSE
base_root_mean_square_error_imputed = np.sqrt(mean_squared_error(y_test1, base_pred))

print(base_root_mean_square_error_imputed)

#linear regression with imputed data

#setting intercept as true
lgr2 = LinearRegression(fit_intercept=True)

#model
model_lin2 = lgr2.fit(X_train1, y_train1)

#predicting model on test set
cars_predictions_lin2 = lgr2.predict(X_test1)

#computing MSE and RMSE
lin_mse2 = mean_squared_error(y_test1, cars_predictions_lin2)
lin_rmse2 = np.sqrt(lin_mse2)
print(lin_rmse2)

#R squared value
r2_lin_train2 = model_lin2.score(X_test1, y_test1)
r2_lin_test2 = model_lin2.score(X_train1, y_train1)
print(r2_lin_test2, r2_lin_train2)

#random forest with imputed data

rf2 = RandomForestRegressor(n_estimators=100, max_features='auto', max_depth=100, min_samples_leaf=4, min_samples_split=10, random_state=1)

#model
model_rf2 = rf2.fit(X_train1, y_train1)

#predicting model on test data
cars_prediction_rf2=rf2.predict(X_test1)

#computing MSE and RMSE
rf_mse2 = mean_squared_error(y_test1, cars_prediction_rf2)
rf_rmse2=np.sqrt(rf_mse2)
print(rf_rmse2)

#R squared value
rf2_rf_test2 = model_rf2.score(X_test1, y_test1)
rf2_rf_train2 = model_rf2.score(X_train1, y_train1)
print(rf2_rf_test2, rf2_rf_train2)

print("Metrics for models built from data where missing values were omitted")
print("R squared value for train from linear regression = %s"%r2_lin_train1)
print("R squared value for test from linear regression = %s"%r2_lin_test1)
print("R squared value for train from random forest = %s"%r2_rf_train1)
print("R squared value for test from random forest = %s"%r2_rf_test1)

print("Base RMSE of model built from data where missing values were omitted = %s"%base_root_mean_squared_error)
print("RMSE value for test from Linear Regression = %s"% lin_rmse1)
print("RMSE value for test from Random Forest = %s"% rf_rmse1)
print("\n\n")

print("Metrics for models built from data where missing values were imputed")
print("R squared value for train from linear regression = %s"%r2_lin_train2)
print("R squared value for test from linear regression = %s"%r2_lin_test2)
print("R squared value for train from random forest = %s"%rf2_rf_train2)
print("R squared value for test from random forest = %s"%rf2_rf_test2)

print("Base RMSE of model built from data where missing values were imputed = %s"%base_root_mean_square_error_imputed)
print("RMSE value for test from Linear Regression = %s"% lin_rmse2)
print("RMSE value for test from Random Forest = %s"% rf_rmse2)


#end

















































