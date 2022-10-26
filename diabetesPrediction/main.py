import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm # support vector machine algorithm
from sklearn.metrics import accuracy_score


# loading the diabetes dataset
diabetes_dataset_path = './diabetes.csv';
diabetes_dataset = pd.read_csv(diabetes_dataset_path)

# print the first 5 rows
print("First 5 rows")
print(diabetes_dataset.head())
print("\n")

# print the last 5 rows
print("Last 5 rows")
print(diabetes_dataset.tail())
print("\n")

# number of rows n columns
print("Number of rows and columns")
print(diabetes_dataset.shape)
print("\n")

# dataset statistical measures
print("Dataset Statistical Measures")
print (diabetes_dataset.describe())
print("\n")

# info on the dataset -> null values, data type, etc
print("Dataset Info")
print(diabetes_dataset.info())
print("\n")

# check how many outcomes are there (target value)
print("Dataset target variable/outcomes info")
print(diabetes_dataset['Outcome'].value_counts())
print("\n")

# get the mean value for the outcome grouped by 0 and 1
print("Mean value of the target variable/outcome")
print(diabetes_dataset.groupby('Outcome').mean())
print("\n")

# Separating the data and labels
features_data = diabetes_dataset.drop(columns='Outcome',axis=1)
outcome_data = diabetes_dataset['Outcome']


# Data standardisation of the Features
    # Helps to standardise the data to a command format like instead of
    # a mixture of whole numbers and decimals across different columns and rows,
    # we standardise all to decimals
scaler = StandardScaler()
features_data_standardised = scaler.fit(features_data).transform(features_data)
features_data = features_data_standardised # Pass it back to the original variable

# Split into Training Data & Test Data
    # test_size refers to % of the data to test, so 0.2 means 20% of the data for testing
    # stratify means it will distribute the target evenly otherwise all the values in the test data may be of target 0 (defective heart)
    # random_state refers to the shuffling of the data
# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
features_data_train,features_data_test,outcome_data_train,outcome_data_test = train_test_split(features_data,
                                                                                               outcome_data,
                                                                                               test_size=0.2,
                                                                                               stratify=outcome_data,
                                                                                               random_state=2)

# Compare the Rows & Columns of Original Data vs Training Data vs Test Data
print(features_data.shape, features_data_train.shape, features_data_test.shape)


# Training the Model begins here
# SVC is the Support Vector Classification (SVC), it is a form of supervised learning to train the model
# We are using a linear model for the SVC
classifier = svm.SVC(kernel='linear')

# Training the Support Vector Machine Classifier (SVM.SVC)
classifier.fit(features_data_train,outcome_data_train)


# Model Evaluation begins here
# We gather the accuracy score on the training data
features_data_train_prediction = classifier.predict(features_data_train)
training_data_accuracy = accuracy_score(features_data_train_prediction, outcome_data_train)
print("Training data accuracy: ", training_data_accuracy)

# We gather the accuracy score on the test data
features_data_test_prediction = classifier.predict(features_data_test)
test_data_accuracy = accuracy_score(features_data_test_prediction, outcome_data_test)
print("Test data accuracy: ", test_data_accuracy)


# Make a Predictive System
input_data = (1,85,66,29,0,26.6,0.351,31)
# change the input data to numpy array for easier data reshaping / processing
input_data_as_numpy_array = np.asarray(input_data)
# reshape the array as we are predicting for one instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)
# we standardise the reshaped numpy array, no need to .fit() again
input_data_standardised = scaler.transform(input_data_reshaped)
# Predict the data
prediction = classifier.predict(input_data_standardised)

if (prediction[0] == 1):
    print("Person has diabetes")
else:
    print("Person has no diabetes")