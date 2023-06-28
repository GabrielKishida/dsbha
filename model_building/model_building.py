# Imports
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import RandomOverSampler
from imblearn.over_sampling import SMOTE

# Reading data and preparing arrays for model building
fruits = pd.read_csv('fruits.csv')
feature_names = ['mass', 'width', 'height', 'color_score']
X = fruits[feature_names]
Y = fruits['fruit_name']

# Definition of function that builds models with different resampling methods and test sizes
def test_model(resampling, test_size):
  print("Current resampling: " + resampling + " | Current test size: " + str(test_size))
  x_prepared = X.copy()
  y_prepared = Y.copy()
  if resampling == 'RUS':
    rus = RandomUnderSampler()
    x_prepared, y_prepared =  rus.fit_resample(x_prepared, y_prepared)
  elif resampling == 'ROS':
    ros = RandomOverSampler()
    x_prepared, y_prepared = ros.fit_resample(x_prepared, y_prepared)
  elif resampling == 'SMOTE':
    sm = SMOTE(k_neighbors=4)
    x_prepared, y_prepared = sm.fit_resample(x_prepared, y_prepared)

  X_train, X_test, Y_train, Y_test = train_test_split(x_prepared, y_prepared,
                                                      random_state=0, test_size=test_size)
  scaler = MinMaxScaler()
  X_train = scaler.fit_transform(X_train)
  X_test = scaler.transform(X_test)
  svm = SVC()
  svm.fit(X_train, Y_train)
  pred = svm.predict(X_test)
  print('Accuracy of classifier on training set: {:.2f}'.format(svm.score(X_train, Y_train)))
  print('Accuracy of classifier on test set: {:.2f}'.format(svm.score(X_test, Y_test)))
  print(classification_report(Y_test, pred))
  print("")

# Begin building models
resampling_methods = {"RUS", "ROS", "SMOTE"}
test_sample_sizes = {0.4, 0.1}
for current_resampling_method in resampling_methods:
  for current_test_size in test_sample_sizes:
    test_model(current_resampling_method, current_test_size)