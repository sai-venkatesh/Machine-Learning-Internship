# Task-1--sparks-foundation

Data Science & Business Analytics Internship at The Sparks Foundation
TASK 1 - Prediction using Supervised Machine Learning

The aim of this task is to predict the percentage of a student based on the nnumber of study hours using the Linear Regression supervised machine learning algorithm.

Steps to be followed:

Step 1 - Importing the libraries
Step 2 - Importing the dataset
Step 3 - Visualizing the dataset
Step 4 - Data preparation
Step 5 - Training the algorithm
Step 6- Visualizing the model
Step 7- Making predcitions
Step 8- Evaluating the model


# Step 1 - Importing the dataset
*In this step, we will import the necessary libraries *

import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
%matplotlib inline
import seaborn as sns


# Step 2 - Importing the dataset

*Importing/Reading the data*

url = "http://bit.ly/w-data"
data = pd.read_csv(url)

*Displaying head of the data*
data.head()

*Displaying tail of the data*
data.tail()

*Finding the data type of the data*
data.dtypes

*Describing the data*
data.describe()

# Step 3 - Visualizing the dataset
*We will plot the dataset and check if there is any relation between the variables*

   # Countplot for "Hours" 
   sns.countplot(x="Hours",data=data)
   
   # Countplot for "Scores" 
   sns.countplot(x="Scores",data=data)
   
   # Plotting the distribution of scores
   data.plot(x='Hours', y='Scores', style='*', color='red', markersize=13)  
   plt.title('Hours vs Percentage')  
   plt.xlabel('Hours Studied by students')  
   plt.ylabel('Percentage Scored')  
   plt.grid()
   plt.show()

# STEP 4 - Data preparation
In this step we will divide the data into inputs and outputs. And then we will divide the whole dataset into 2 parts - testing data and training data.

   # spliting the dataset into dependent and independent values by using  "iloc" Function* 
   X = data.iloc[:, :-1].values  
   Y = data.iloc[:, 1].values
   
   # training and testing the dataset using "train-test-split" function.
   from sklearn.model_selection import train_test_split  
   X_train, X_test, Y_train, Y_test = train_test_split(X, Y,test_size=0.2, random_state=0)
   
# Step 5 - Training the algorithm

*We have split our data into training and testing sets, and now is finally the time to train our algorithm.*

   from sklearn.linear_model import LinearRegression
   model = LinearRegression() 
   
   # Fitting the model
   model.fit(X_train, Y_train)
   LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None, normalize=False)
   
# STEP 6 - Visualizing the model
Visualizing the model after training it
     
   # Plotting the regression line
   line = model.coef_*X + model.intercept_
   
   # Plotting for the train data
   plt.scatter(X_train, Y_train, color='red')
   plt.plot(X, line, color='blue');
   plt.xlabel('Hours Studied')  
   plt.ylabel('Percentage Score') 
   plt.grid()
   plt.show()
   
   # Plotting for the test data
   plt.scatter(X_test, Y_test, color='brown')
   plt.plot(X, line, color='pink');
   plt.xlabel('Hours Studied')  
   plt.ylabel('Percentage Score') 
   plt.grid()
   plt.show()
   
# STEP 7 - Making Predictions
After training the algorithm, it's time to make some predictions.

   # Predicting the model
   Y_predicted = model.predict(X_test)
  
   # Comparision of Real and Predicted Class values 
   df = pd.DataFrame({'Actual score': Y_test, 'Predicted score': Y_predicted})  
   df
  
   # Finding the Percentage   
   hrs = 9.25
   Score_prediction = model.predict([[hrs]])
   print("The predicted score, if a person studies for",hrs,"hours is",Score_prediction[0])
  
   # Plot for Percentage Scored
   sns.distplot(own_prediction[0],color='brown',bins=20,kde=False)
   
# Step 8- Evaluating the model
*Final step ,we are going to evaluate our trained model by calculating mean absolute error*

from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(Y_test, Y_predicted))

