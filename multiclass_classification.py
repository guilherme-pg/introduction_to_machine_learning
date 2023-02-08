
#  ~~~~~~~~~~~~~~~~~~~~~~~~~ IMPORTS  ~~~~~~~~~~~~~~~~~~~~~~~~~
from sklearn.datasets import fetch_openml
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ LOAD DATASET  ~~~~~~~~~~~~~~~~~~~~~~~~~
mnist = fetch_openml('mnist_784', version=1)

mnist.keys()

X, y = mnist['data'], mnist['target']

print(X.shape)
print(y.shape)


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ SPLIT TRAIN AND TEST  ~~~~~~~~~~~~~~~~~~~~~~~~~
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# Training the SVM classifier
svm_clf = SVC(decision_function_shape='ovo')

svm_clf.fit(X_train, y_train)

# SVC(decision_function_shape='ovo')


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICTION IN TRAINING  ~~~~~~~~~~~~~~~~~~~~~~~~~
training_prediction = svm_clf.predict(X_train[:5])
print("Prediction in training: ", training_prediction)
print("Actual values: ", y_train[:5])

#  ~~~~~~~~~~~~~~~~~~~~~~~~~ PREDICTION IN TESTING  ~~~~~~~~~~~~~~~~~~~~~~~~~
testing_prediction = svm_clf.predict(X_test[:5])
print("Prediction in testing:", testing_prediction)
print("Actual values:", y_test[:5])


#  ~~~~~~~~~~~~~~~~~~~~~~~~~ EVALUATION  ~~~~~~~~~~~~~~~~~~~~~~~~~
y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

y_train_predict = cross_val_predict(svm_clf, X_train, y_train_5, cv=3)

confusion_matrix_array = confusion_matrix(y_train_5, y_train_predict)

print("confusion_matrix_array: ", confusion_matrix_array)

precision = precision_score(y_train_5, y_train_predict)
recall = recall_score(y_train_5, y_train_predict)
f1score = f1_score(y_train_5, y_train_predict)

print("Precision: ", precision)
print("Recall: ", recall)
print("f1_score: ", f1score)
