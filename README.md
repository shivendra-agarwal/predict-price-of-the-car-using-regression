# predict-price-of-the-car-using-regression
In this case study, we will predict price of the used cars using different regression models
### Problem
Predict price of the cars based on various attributes

Total size of data = 50000 rows and 19 columns
### Approach
- Identify the data is clean
- look for missing values
- identify variable influencing price and look for relationships among variables
- identify outliers
- filter data based on logical check
- reduce number of data
#### Linear regression Vs. Random forest
#### output
Metrics for models built from data where missing values were omitted
- R squared value for train from linear regression = 0.7800936978183916
- R squared value for test from linear regression = 0.7658615091649247
- R squared value for train from random forest = 0.9202494705146291
- R squared value for test from random forest = 0.8504017440877062
- Base RMSE of model built from data where missing values were omitted = 1.1274483657478247
- RMSE value for test from Linear Regression = 0.5455481266513836
- RMSE value for test from Random Forest = 0.436073731962631

Metrics for models built from data where missing values were imputed
- R squared value for train from linear regression = 0.7023339008631184
- R squared value for test from linear regression = 0.7071658736894363
- R squared value for train from random forest = 0.9024289705830797
- R squared value for test from random forest = 0.8269944666838189
- Base RMSE of model built from data where missing values were imputed = 1.1884349112889792
- RMSE value for test from Linear Regression = 0.6483956449231296
- RMSE value for test from Random Forest = 0.494316830858078

#### Random forest perfoms better in the 2nd case. Linear performs better in the 1st case.
#### Thank you NPTEL and Prof. Rengaswamy and teaching assistants of course Python for Data Science. 
