import pandas as pd
import sklearn
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import mean
from numpy import std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.model_selection import cross_val_score


#import the file
telstra_final = pd.read_csv('telstra_traindata.csv')
X=telstra_final[['location','severity_type','resource_type', 'log_feature', 'volume', 'event_type']]
#print(X.head(10))

y=telstra_final.fault_severity
y.head()

#splitting Without stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
X_train, X_val, y_train, y_val  = train_test_split(X_train, y_train, test_size=0.25, random_state=1)
#print(X_train.shape)
#print(X_val.shape)
#print(X_test.shape)

#comparing models
logr=linear_model.LogisticRegression(solver= 'liblinear').fit(X_train, y_train)
logrPred= logr.predict(X_val)
print("LR :", accuracy_score(y_val, logrPred))

result =[]
result.append(logrPred)


knn = KNeighborsClassifier().fit(X_train, y_train)
knnPred= knn.predict(X_val)
print("knn :", accuracy_score(y_val, knnPred))
result.append(knnPred)


gus = GaussianNB().fit(X_train, y_train)
gusPred= gus.predict(X_val)
print("Gaussian :", accuracy_score(y_val, gusPred))
result.append(gusPred)

svm = SVC().fit(X_train, y_train)
svmPred = svm.predict(X_val)
print("SVC :", accuracy_score(y_val, svmPred))
result.append(svmPred)


#Decisiontreeclassifier is most preferable for prediction
model = DecisionTreeClassifier().fit(X_train, y_train)
cart_pred= model.predict(X_val)
print("Cart : ", accuracy_score(y_val, cart_pred))
result.append(cart_pred)
 
print("Confusion Matrix")
print(confusion_matrix(y_val, cart_pred))

def get_models():
    models = dict()
    models['lr'] = LogisticRegression()
    models['knn'] = KNeighborsClassifier()
    models['cart'] = DecisionTreeClassifier()
    models['svm'] = SVC()
    models['bayes'] = GaussianNB()
    return models

pyplot.boxplot(result, labels=get_models(), showmeans=True)
print(pyplot.show())




import pickle 
# save the model to disk
filename = 'telstrafinalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
