# Linear regression project
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
customers = pd.read_csv("/Users/janekom/Desktop/Refactored_Py_DS_ML_Bootcamp-master/11-Linear-Regression/Ecommerce Customers")
customers.head()
customers.describe()
customers.info()

#%%
# EDA
# Use seaborn to create a jointplot to compare the
# Time on Website and Yearly Amount Spent columns to check does the correlation make sense.
sns.set_palette("GnBu_d")
sns.set_style('whitegrid')

# More time on site, more money spent
sns.jointplot(x='Time on Website',y='Yearly Amount Spent',data=customers)

# Do the same but with the Time on App column instead
sns.jointplot(x='Time on App',y='Yearly Amount Spent',data=customers)

#Use jointplot to create a 2D hex bin plot comparing Time on App and Length of Membership
sns.jointplot(x='Time on App',y='Length of Membership',kind='hex',data=customers)

#%%
# explore these types of relationships across the entire data set
sns.pairplot(customers)
# the most correlated feature with Yearly Amount Spent is length of membership
#%%
# Create a linear model plot (using seaborn's lmplot) of Yearly Amount Spent vs. Length of Membership
sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)

#%%
# Training and testing data
y = customers['Yearly Amount Spent']
X = customers[['Avg. Session Length', 'Time on App','Time on Website', 'Length of Membership']]

#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
#%%
# Training the model
from sklearn.linear_model import LinearRegression
# Create an instance of a LinearRegression() model named lm
lm = LinearRegression()
# Train/fit lm on the training data
lm.fit(X_train,y_train)
# Print out the coefficients of the model
print('Coefficients: \n', lm.coef_)

#%%
# Predicting test data
predictions = lm.predict( X_test)

# Create a scatterplot of the real test values versus the predicted values
plt.scatter(y_test,predictions)
plt.xlabel('Y Test')
plt.ylabel('Predicted Y')

#%%
# Evaluating the model
from sklearn import metrics

print('MAE:', metrics.mean_absolute_error(y_test, predictions))
print('MSE:', metrics.mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

#%%
# Conslusion

coefficients = pd.DataFrame(lm.coef_,X.columns)
coefficients.columns = ['Coefficient']
coefficients

# Interpreting the coefficients:
# Holding all other features fixed, a 1 unit increase in Avg. Session Length is associated with an increase of 25.98 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on App is associated with an increase of 38.59 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Time on Website is associated with an increase of 0.19 total dollars spent.
# Holding all other features fixed, a 1 unit increase in Length of Membership is associated with an increase of 61.27 total dollars spent.


# Do you think the company should focus more on their mobile app or on their website?
# This is tricky, there are two ways to think about this: Develop the Website to catch up to the performance of the mobile app,
# or develop the app more since that is what is working better. This sort of answer really depends on the other factors
# going on at the company, and it would probably to explore the relationship between Length of Membership
# and the App or the Website before coming to a conclusion.


