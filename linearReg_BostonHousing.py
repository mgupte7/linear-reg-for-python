# Code source: Kaggle Tutorial
# License: Apache 2.0 open source
# Notes: Mitali Gupte
# Date: 10/26/2021

# -------------------- columns in data ----------------------
# 'crim': per capita crime rate by town.
# 'zn': proportion of residential land zoned for lots over 25,000 sq.ft.
# 'indus': proportion of non-retail business acres per town.
# 'chas':Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).
# 'nox': nitrogen oxides concentration (parts per 10 million).
# 'rm': average number of rooms per dwelling.
# 'age': proportion of owner-occupied units built prior to 1940.
# 'dis': weighted mean of distances to five Boston employment centres.
# 'rad': index of accessibility to radial highways.
# 'tax': full-value property-tax rate per $10,000.
# 'ptratio': pupil-teacher ratio by town
# 'black': 1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.
# 'lstat': lower status of the population (percent).
# 'medv': median value of owner-occupied homes in $$1000s --> response variable!

# -------------------- library imports ----------------------
import inline
import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# %matplotlib inline // this might only work for Jupyter IPython notebook

# -------------------- looking at data ----------------------

# Importing DataSet and take a look at Data
BostonTrain = pd.read_csv(
    "/Users/mitaligupte/PycharmProjects/pythonProject1/boston_train.csv")

# Here we can look at the BostonTrain data as a dataframe
BostonTrain.head()

# printing out more information about entries (null or not null)
BostonTrain.info()
BostonTrain.describe()

# -------------------- Finding relevant columns ----------------------

# ID columns does not relevant for our analysis.
BostonTrain.drop('ID', axis=1, inplace=True)  # "no documentation" for axis?

BostonTrain.plot.scatter('rm', 'medv')
# output: matplotlib.axes._subplots.AxesSubplot at 0x7fbe883a8080>

# ----------------- how the variables relate to each other ------------
plt.subplots(figsize=(12, 8))
sns.heatmap(BostonTrain.corr(), cmap='RdGy') # Compute pairwise correlation of columns, excluding NA/null values.
#output: <matplotlib.axes._subplots.AxesSubplot at 0x7fbe883530b8>

# heatmap explained
# y= medv (median value, on last coulmn)
# Red/Orange: the more red the color is on X axis, smaller the medv. Negative correlation
# light colors: those variables at axis x and y, they dont have any relation. Zero correlation
# Gray/Black : the more black the color is on X axis, more higher the value med is. Positive correlation

# ---------- plot the paiplot, for all different correlations ------------

# The pairplot function creates a grid of Axes such that each variable in data
# Will by shared in the y-axis across a single row and in the x-axis across a single column
# Main idea is to find where a postive corelation exists (where x and y go together)

sns.pairplot(BostonTrain, vars=['lstat', 'ptratio', 'indus', 'tax', 'crim', 'nox', 'rad', 'age', 'medv'])
# output: <seaborn.axisgrid.PairGrid at 0x7fbe88285c50>


sns.pairplot(BostonTrain, vars = ['rm', 'zn', 'black', 'dis', 'chas','medv'])
# output: <seaborn.axisgrid.PairGrid at 0x7fbdf20f9d30>


# -------------------- Training Linear Regression Model ----------------------

X = BostonTrain[['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax',
       'ptratio', 'black', 'lstat']]
y = BostonTrain['medv']

# Import sklearn librarys:

from sklearn.model_selection import train_test_split # to split our data in two DF, one for build a model and other to validate
from sklearn.linear_model import LinearRegression # # LinearRegression, to apply the linear regression.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4)

lm = LinearRegression()
lm.fit(X_train,y_train)

# output: LinearRegression(copy_X=True, fit_intercept=True, n_jobs=1, normalize=False)

predictions = lm.predict(X_test)

plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')
# output: Text(0,0.5,'Predicted Y')

from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions)) # MAE: 3.53544941908
print('MSE:', metrics.mean_squared_error(y_test, predictions)) # MSE: 20.8892997114
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions))) # RMSE: 4.57048134351

sns.distplot((y_test-predictions),bins=50);

# As more normal distribution, better it is.
coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['coefficients']
coefficients

# for one unit that nox increase, the house value decrease 'nox'*1000 (Negative correlation) money unit.
# for one unit that rm increase, the house value increase 'rm'*1000 (Positive correlation) money unit.



