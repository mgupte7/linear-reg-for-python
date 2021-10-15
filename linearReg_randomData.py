import math

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

# Generate 'random' data
from sklearn.metrics import mean_squared_error

np.random.seed(0)
X = 2.5 * np.random.randn(100) + 1.5   # Array of 100 values with mean = 1.5, stddev = 2.5
res = 0.5 * np.random.randn(100)       # Generate 100 residual terms
y = 2 + 0.3 * X + res                  # Actual values of Y

# Create pandas dataframe to store our X and y values
df = pd.DataFrame(
    {'X': X,
     'y': y}
)

# Show the first five rows (default) of our dataframe
df.head(10)

# Calculate the mean of X and y
xmean = np.mean(X)
ymean = np.mean(y)
# Calculate the terms needed for the numerator and denominator of beta
df['xycov'] = (df['X'] - xmean) * (df['y'] - ymean)
df['xvar'] = (df['X'] - xmean)**2


# Calculate beta and alpha
beta = df['xycov'].sum() / df['xvar'].sum()
alpha = ymean - (beta * xmean)
#print(f'alpha = {alpha}')
#print(f'beta = {beta}')
print(alpha)
print(beta)

ypred = alpha + beta * X
MSE = (1/len(y)) * (sum(y - ypred))**2
print("version 1")
print(MSE)
print("\n")
MSE2 = np.square(np.subtract(y,ypred)).mean()
print("version 2")
print(MSE2)
print("\n")
MSE3 = mean_squared_error(y,ypred)
print("version 3")
print(MSE3)
print("\n")

# Plot regression against actual data
plt.figure(figsize=(12, 6))
plt.plot(X, ypred)     # regression line
plt.plot(X, y, 'ro')   # scatter plot showing actual data
plt.title('Actual vs Predicted')
plt.xlabel('X')
plt.ylabel('y')

plt.show()