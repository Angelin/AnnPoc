
import pandas as pd
import keras
from keras.models import load_model
import numpy as np
import sys
import pickle

# Importing the dataset
dataset = pd.read_csv('assets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

#load the saved model
classifier_model = load_model('ann_poc_model.h5')

'''
# @todo Importing the test dataset
X_test = pickle.load(open('assets/x_testdata.txt', 'rb'))
y_test = pickle.load(open('assets/y_testdata.txt', 'rb'))

# load scaler & Feature Scaling
scaler = pickle.load(open('scaler.sav', 'rb'))
X_test = scaler.transform(x_testdatast)
y_test = scaler.transform(y_test)
'''

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# Importing the dataset
dataset = pd.read_csv('assets/Churn_Modelling.csv')
X = dataset.iloc[:, 3:13].values
y = dataset.iloc[:, 13].values

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder(categorical_features = [1])
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#part 3: evaluating the model
scores = classifier_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (classifier_model.metrics_names[1], scores[1]*100))
print(scores)
classifier_model.summary();

y_pred = classifier_model.predict(X_test, batch_size=None, verbose=1, steps=None)
y_pred = (y_pred > 0.5)

# error: ValueError: Classification metrics can't handle a mix of continuous-multioutput and continuous targets
# # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("confusion_matrix: " ,cm, " 0: " , cm[0][0] , " 1: " ,cm[1][1])

# calculation the accuracy
# accuracy = sum(right predictions)/ total 
sum_confusion_matrix = np.sum(cm)
accuracy = (cm[0][0] + cm[1][1])/ np.sum(sum_confusion_matrix)
print("accuracy: ", accuracy)
