
from sklearn.linear_model import LinearRegression
import pandas as pd
from sklearn.datasets import make_sparse_coded_signal

# Import and display first five rows of advertising dataset
# advert = pd.read_csv('advertising.csv')
advert = pd.read_csv('http://www-bcf.usc.edu/~gareth/ISL/Advertising.csv', index_col=0)
advert.head()

# Build linear regression model using TV and Radio as predictors
# Split data into predictors X and output Y
predictors = ['TV', 'Radio']
X = advert[predictors]
y = advert['Sales']

# Initialise and fit model
lm = LinearRegression()
model = lm.fit(X, y)

# Calculate the values for alpha and betas:
print(model.intercept_) # alpha
print(model.coef_) # beta

# Predict values:
model.predict(X)

# Predict sales from any combination of TV and Radio advertising costs
new_X = [[300, 200]]
print(model.predict(new_X))