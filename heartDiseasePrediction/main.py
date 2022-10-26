# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

import numpy as np # useful for large array and matrices
import pandas as pd # useful for creating data frames / table structures
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load csv into a pandas dataframe
heart_data_csv_path = "./heart_disease_data.csv"
heart_data = pd.read_csv(heart_data_csv_path)


# print first 5 rows of the dataset
#print(heart_data.head())

# print last 5 rows of the dataset
#print(heart_data.tail())

# number of rows and columns in the dataset
print("Number of rows and columns: ")
print(heart_data.shape)
print("\n")

# getting more info about the data
print("Dataset Information")
print(heart_data.info())
print("\n")

# check if any null values in the data
# if there is any we can use imputation to handle the missing values
print("Check for null in Dataset")
print(heart_data.isnull().sum())
print("\n")

# get statistical measures of the data (Mean,std,min,percentile values,max)
print("Statistical measures of dataset")
print(heart_data.describe())
print("\n")

# check the distribution of target variable in the dataset
# 1--> Defective heart
# 0 --> Healthy heart
print("Distribution of target variable")
print(heart_data["target"].value_counts())
print("\n")

# Splitting the Features and Target
# The Features predicts / determines the Target
data_features = heart_data.drop(columns='target', axis=1)
data_target = heart_data['target']

print("Features Columns & Rows")
print(data_features)
print("\n")

print("Target Columns & Rows")
print(data_target)
print("\n")

# Splitting the data into Training data & Test data
    # test_size refers to % of the data to test, so 0.2 means 20% of the data for testing
    # stratify means it will distribute the target evenly otherwise all the values in the test data may be of target 0 (defective heart)
    # random_state refers to the shuffling of the data
data_features_train, data_features_test, data_target_train, data_target_test = train_test_split(data_features,
                                                                                                data_target,
                                                                                                test_size=0.2,
                                                                                                stratify=data_target,
                                                                                                random_state=2)

print("[Features] Original Data vs Test/Training Data Shape")
print(data_features.shape, data_features_train.shape, data_features_test.shape)

print("[Target] Original Data vs Test/Training Data Shape")
print(data_target.shape, data_target_train.shape, data_target_test.shape)


# Model Training starts here
# We will use and load Logistic Regression algorithm into the model
model = LogisticRegression()

# Training the LogisticRegression model with Training Data
# It will find the corresponding relationship between the features and target
model.fit(data_features_train,data_target_train)

# Model Evaluation starts here
# We will use accuracy score as our measure metric to evaluate the model

# Check accuracy on the training data
data_features_train_prediction = model.predict(data_features_train)
training_data_accuracy = accuracy_score(data_features_train_prediction, data_target_train)
print('Training Data accuracy: ', training_data_accuracy)

# Check accuracy on the training data
data_features_test_prediction = model.predict(data_features_test)
testing_data_accurancy = accuracy_score(data_features_test_prediction, data_target_test)
print('Testing Data accuracy: ', testing_data_accurancy)

# Generally, the test and train data accuracy should be similar
# If the model is 'trained' too much on the test data, it will return a much higher value which
# is not desired as it can throw the model off and this is known as 'overfitting' the model




# Building a Predictive System
user_input_data = (54,1,0,124,266,0,0,109,1,2.2,1,1,3)

# we need to changy the input data to a numpy array to be data reshaping / processing
# it is easier to reshape an array than a tuple
user_input_data_as_numpy_array = np.asarray(user_input_data)

# reshape the numpy array as we are predicting for only one instance
user_input_data_reshaped = user_input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(user_input_data_reshaped)
if (prediction[0] == 1):
    print('The person has a heart disease')
else:
    print('The person has no heart disease')