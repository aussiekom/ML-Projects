# fake advertising project
# import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#%%
# get the data
ad_data = pd.read_csv('/Users/janekom/Desktop/Logistic-Regression/advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()

#%%
# Exploratory data analysis
# create a histogram of the Age
sns.set_style('whitegrid')
ad_data['Age'].hist(bins=30)
plt.xlabel('Age')

# Create a jointplot showing Area Income versus Age
sns.jointplot(x='Age',y='Area Income',data=ad_data)

# Create a jointplot showing the kde distributions of Daily Time spent on site vs. Age
sns.jointplot(x='Age',y='Daily Time Spent on Site',data=ad_data,color='red',kind='kde')

# Create a jointplot of 'Daily Time Spent on Site' vs. 'Daily Internet Usage'
sns.jointplot(x='Daily Time Spent on Site',y='Daily Internet Usage',data=ad_data,color='green')

# Create a pairplot with the hue defined by the 'Clicked on Ad' column feature.
sns.pairplot(ad_data,hue='Clicked on Ad',palette='bwr')

#%%
# logistic regression
# do a train test split, and train our model
# Split the data into training set and testing set using train_test_split
from sklearn.model_selection import train_test_split
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income','Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# Train and fit a logistic regression model on the training set
from sklearn.linear_model import LogisticRegression
logmodel = LogisticRegression()
logmodel.fit(X_train,y_train)


#%%
# prediction and evaluation
# predict values for the testing data
predictions = logmodel.predict(X_test)

# create a classification report for the model
from sklearn.metrics import classification_report
print(classification_report(y_test,predictions))

