import pandas as pd
import keras
from keras.models import load_model
import numpy as np
import sys

# 0- france  1- germany 2-spain
# 0-female 1-male
# X_new = [0,0,600,1,40, 3, 60000, 2, 1, 1,50000] - final array after transformation

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

X_new = np.array([[619,'France','Female',42,2,0,1,1,1,101348.88]])

labelencoder_X_1 = LabelEncoder()
X_new[:, 1] = labelencoder_X_1.fit_transform(X_new[:, 1])
labelencoder_X_2 = LabelEncoder()
X_new[:, 2] = labelencoder_X_2.fit_transform(X_new[:, 2])


# onehotencoder = OneHotEncoder(categories='auto')
# onehotencoder = OneHotEncoder(categorical_features = [1])
# X_new = onehotencoder.fit_transform(X_new).toarray() # onehotencoder.fit_transform() ---returns---> coo matrix 
# X_new = X_new[:, 1:]

# @todo generate the the below transformed array
X_new = np.array([[0,0,600,1,40, 3, 60000, 2, 1, 1,50000]]) 
# sys.exit()

#part 2: prediction
#load the saved model
classifier_model = load_model('ann_poc_model.h5')

# loading the scaler
import pickle
scalerfile = 'scaler.sav'
scaler     = pickle.load(open(scalerfile, 'rb'))
X_new      = scaler.transform(X_new)

# Y_new = classifier_model.predict(X_new_scaled_set, batch_size=None, verbose=1, steps=None)
Y_new = classifier_model.predict(X_new, batch_size=None, verbose=1, steps=None)
# Y_new = (Y_new > 0.5)

# show the inputs and predicted outputs
for i in range(len(X_new)):
	print("X=%s, Predicted=%s" % (X_new[i], Y_new[i]))

#part 3: evaluating the model
scores = classifier_model.evaluate(X_new, Y_new, verbose=0)
print("%s: %.2f%%" % (classifier_model.metrics_names[1], scores[1]*100))
print(scores)
classifier_model.summary();


# error: ValueError: Classification metrics can't handle a mix of continuous-multioutput and continuous targets
# # Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(X_new, Y_new)
print(cm)

