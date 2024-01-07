"""
CS 4104 - Assignment 02
Name        : D. R. R. Wijewardene
Index No.   : S14245
"""

import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score

# Load dataset
with pd.ExcelFile('SCS4204_IS4103_CS4104 _dataset.xlsx') as dataset:
    training_dataset = pd.read_excel(dataset, sheet_name='Training Dataset')
    testing_dataset = pd.read_excel(dataset, sheet_name='Testing Dataset')

# Replace the '?' of missing values with the column mean
# Remove the rows with missing values for Gender or Class attributes
for column in training_dataset.columns:
    if column not in ("Gender", "Class"):
        # training data
        training_dataset[column] = training_dataset[column].replace("?", np.NaN)
        train_mean = int(training_dataset[column].mean(skipna=True))
        training_dataset[column] = training_dataset[column].replace(np.NaN, train_mean)
        # testing data
        testing_dataset[column] = testing_dataset[column].replace("?", np.NaN)
        test_mean = int(testing_dataset[column].mean(skipna=True))
        testing_dataset[column] = testing_dataset[column].replace(np.NaN, test_mean)
    else:
        training_dataset[column] = training_dataset[column].replace("?", np.NaN)
        training_dataset = training_dataset.dropna()
        testing_dataset[column] = testing_dataset[column].replace("?", np.NaN)
        testing_dataset = testing_dataset.dropna()


# Convert nominal attributes to numerical attributes
def convert_to_numeric(x):
    if x == 'Male' or x == 'Yes':
        return 1
    if x == 'Female' or x == 'No':
        return 0


# Convert the gender attribute to numerical values
training_dataset['Gender'] = training_dataset['Gender'].apply(convert_to_numeric)
testing_dataset['Gender'] = testing_dataset['Gender'].apply(convert_to_numeric)

# Convert the class attribute to numerical values
training_dataset['Class'] = training_dataset['Class'].apply(convert_to_numeric)
testing_dataset['Class'] = testing_dataset['Class'].apply(convert_to_numeric)

# Split and extract training data and labels
x_train = training_dataset.iloc[:, 1:-1]
y_train = training_dataset.iloc[:, 11]

# Split and extract testing data and labels
x_test = testing_dataset.iloc[:, 1:-1]
y_test = testing_dataset.iloc[:, 11]

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Hyper-parameters
leaf_size = list(range(1, 50))
n_neighbors = list(range(1, 30))
p = [1, 2]
hyperparameters = dict(leaf_size=leaf_size, n_neighbors=n_neighbors, p=p)

# Define model
knn = KNeighborsClassifier()

# Use gridsearch
classifier = GridSearchCV(knn, hyperparameters, cv=10)

# Fit the model
best_model = classifier.fit(x_train, y_train)

# Best hyper-parameters
print('Best leaf_size :', best_model.best_estimator_.get_params()['leaf_size'])
print('Best p :', best_model.best_estimator_.get_params()['p'])
print('Best n_neighbors (k) :', best_model.best_estimator_.get_params()['n_neighbors'])

# Predict the test set results
y_pred = classifier.predict(x_test)

# Evaluate model

cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)
error_rate = 1 - accuracy

print("\nConfusion matrix : \n", cm)
print("\nAccuracy\t: %0.2f " % (accuracy * 100), "%")
print("Precision\t: %0.2f " % (precision * 100), "%")
print("Sensitivity\t: %0.2f " % (sensitivity * 100), "%")
print("Specificity\t: %0.2f " % (specificity * 100), "%")
print("Error Rate\t: %0.2f " % (error_rate * 100), "%")
