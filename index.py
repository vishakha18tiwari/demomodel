from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import pickle
import requests, json
iris=datasets.load_iris()
X=iris.data
y=iris.target
X_train,X_test, y_train,y_test=train_test_split(X,y)
rfc= RandomForestClassifier(n_estimators=100,n_jobs=2)
rfc.fit(X_train,y_train)
pickle.dump(rfc,open("iris_rfc.pkl","wb"))
my_random_forest=pickle.load(open("iris_rfc.pkl","rb"))
my_random_forest
classification_report(y_test, my_random_forest.predict(X_test))

