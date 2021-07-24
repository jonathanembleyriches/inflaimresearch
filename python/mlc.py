# mlp for multi-label classification
import codecs
import tensorflow as tf
import arff
import numpy
import pandas as pd
from numpy import mean, asarray
from numpy import std
from sklearn import preprocessing
from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import RepeatedKFold
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import accuracy_score
from skmultilearn.dataset import load_dataset_dump, save_dataset_dump, load_from_arff


def get_dataset():
	X, y = make_multilabel_classification(n_samples=1000, n_features=10, n_classes=3, n_labels=2, random_state=1)
	return X, y

# get the model
# get the model
def get_model(n_inputs, n_outputs):
	# create model
	model = Sequential()
	model.add(Dense((n_inputs*2/3)+n_outputs, input_dim=n_inputs, kernel_initializer='he_uniform',activation='relu'))
	model.add(Dense((n_inputs*2/3)+n_outputs,activation='relu'))

	model.add(Dense(n_outputs, activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

	return model

# evaluate a model using repeated k-fold cross-validation
def evaluate_model(X, y):
	results = list()
	n_inputs, n_outputs = X.shape[1], y.shape[1]
	# define evaluation procedure
	cv = RepeatedKFold(n_splits=10, n_repeats=3, random_state=1)
	# enumerate folds
	for train_ix, test_ix in cv.split(X):
		# prepare data
		X_train, X_test = X[train_ix], X[test_ix]
		y_train, y_test = y[train_ix], y[test_ix]
		# define model
		model = get_model(n_inputs, n_outputs)
		# fit model
		model.fit(X_train, y_train, verbose=0, epochs=10)
		# make a prediction on the test set
		yhat = model.predict(X_test)
		# round probabilities to class labels
		yhat = yhat.round()
		# calculate accuracy
		acc = accuracy_score(y_test, yhat)
		# store result
		print('>%.3f' % acc)
		results.append(acc)
	return results



"""xTrain, yTrain, feature_names, label_names = load_from_arff(
    "test_data/delicious-train.arff",
    label_count=983,
    label_location="end",
    load_sparse=False,
    return_attribute_definitions=True)

xTest, yTest, feature_names2, label_names2 = load_from_arff(
    "test_data/delicious-test.arff",
    label_count=983,
    label_location="end",
    load_sparse=False,
    return_attribute_definitions=True
)
n_inputs, n_outputs = len(feature_names), len(label_names)
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(numpy.array(xTrain.toarray()), numpy.array(yTrain.toarray()), verbose=0, epochs=100)
# make a prediction for new data
yhat = model.predict(numpy.array(xTest.toarray()))


acc = accuracy_score(numpy.array(yTest.toarray()), yhat.round())
print(acc)"""

df = pd.read_csv("test_data/train.csv")
df = df.drop(['ID'],axis=1)
print(df)
train_x = df.iloc[:,:2]
train_y = df.iloc[:,2:]

dataset = tf.data.Dataset.from_tensor_slices((train_x.values, train_y.values))
print(dataset)
print(train_y)
n_inputs, n_outputs = len(train_x.columns), len(train_y.columns)
model = get_model(n_inputs, n_outputs)
model.fit(train_x, train_y, verbose=0, epochs=100)
yhat = model.predict(train_x)

acc = accuracy_score(train_y, yhat.round())
"""df = pd.read_csv("test_data/train.csv",header=None)
df2 = pd.read_csv("test_data/test.csv",header=None)
print("test")
train_x = df.iloc[:,:103]
train_y = df.iloc[:,103:]
test_x = df2.iloc[:,:103]
test_y = df2.iloc[:,103:]


n_inputs, n_outputs = len(train_x.columns), len(train_y.columns)
# get model
model = get_model(n_inputs, n_outputs)
# fit the model on all data
model.fit(train_x, train_y, verbose=0, epochs=100)
# make a prediction for new data
yhat = model.predict(test_x)

print(test_y.iloc[0])
print(yhat.round(0))

acc = accuracy_score(test_y, yhat.round())
for n in range(len(train_y)):
	print("======================")
	print(test_y.iloc[0])
	print(yhat[0].round(0))

print(acc)"""

"""# load dataset
X, y = get_dataset()
# evaluate model
results = evaluate_model(X, y)
# summarize performance
print('Accuracy: %.3f (%.3f)' % (mean(results), std(results)))"""