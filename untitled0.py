# -*- coding: utf-8 -*-
"""
Created on Sun Jul 16 11:21:25 2023

@author: HP
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Step 1: Load the dataset
dataset = pd.read_csv('df_traffic_csv.csv')

# Step 2: Splitting the data into features (X) and labels (y)
X = dataset.iloc[:, :-1]  # Assuming features are in all columns except the last one
y = dataset.iloc[:, 6]   # Assuming the last column contains the labels/targets
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Feature scaling using StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
from sklearn.linear_model import LinearRegression

# Assuming you have X_train as the input features and y_train as the corresponding targets

# Step 1: Create an instance of the LinearRegression model
model = LinearRegression()

# Step 2: Fit the model to the training data
model.fit(X_train, y_train)

# Step 3: Predict using the trained model
y_pred = model.predict(X_train)

import matplotlib.pyplot as plt

# Assuming you have X_train as the input features, y_train as the corresponding targets,
# and y_pred as the predicted values

# Plotting the actual values
plt.scatter(X_train, y_train, color='blue', label='Actual Data')

# Plotting the predicted values
plt.plot(X_train, y_pred, color='red', label='Predicted Data')

# Adding labels and title
plt.xlabel('X')
plt.ylabel('y')
plt.title('Linear Regression')

# Adding legend
plt.legend()

# Displaying the plot
plt.show()
