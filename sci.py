#Imports required
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn import metrics

#Load Data sheet, (Needs to be placed in programming folder)
#Most CSV files use ; but since I played with this data its ,
car = pd.read_csv('~/PycharmProjects/First/opel_corsa_gerry.csv',sep=',')

#Prints out data from the CSV file and verables we have to work with
print(car.head())
print(car.info())
#This tell us how many null values in each coloum
print(car.isnull().sum())

#Preprocessing data between (Good and bad data for this example)
bins = (2, 14.5, 30)
group_names = ['Bad', 'Good']
car['FuelConsumptionAverage'] = pd.cut(car['FuelConsumptionAverage'], bins = bins, labels = group_names)
car['FuelConsumptionAverage'].unique()

#labels data bad = 0 good = 1
label_quality = LabelEncoder()
car['FuelConsumptionAverage'] = label_quality.fit_transform(car['FuelConsumptionAverage'])
print(car.head(10))
#Tell us the good fuel consumption vs bad
print(car['FuelConsumptionAverage'].value_counts())

#Plot out graph
sns.countplot(car['FuelConsumptionAverage'])

#Seperate the dataset as reponse variable and feature variables
X = car.drop('FuelConsumptionAverage', axis = 1)
y = car['FuelConsumptionAverage']

#Train and Test splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 42)

#Applying a stander scaler to optimise the results
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test  = sc.transform(X_test)
X_train[:10]

#Random Forest Classifier
rfc = RandomForestClassifier(n_estimators=200)
rfc.fit(X_train, y_train)
pred_rfc = rfc.predict(X_test)
#How the Forest Classifier preforms
print(classification_report(y_test, pred_rfc))
print(confusion_matrix(y_test, pred_rfc))


##SVM Classifier
clf=svm.SVC()
clf.fit(X_train, y_train)
pred_clf = clf.predict(X_test)
#How the CLF model preformes
print(classification_report(y_test, pred_clf))
print(confusion_matrix(y_test, pred_clf))

##Neural Network
#hidden layers is the nodes in the NN
mlpc=MLPClassifier(hidden_layer_sizes=(11,11,11),max_iter=500)
mlpc.fit(X_train, y_train)
pred_mlpc = mlpc.predict(X_test)
#How the NN model preformes
print(classification_report(y_test, pred_mlpc))
print(confusion_matrix(y_test, pred_mlpc))

#Score the AI
from sklearn.metrics import accuracy_score
bn = accuracy_score(y_test, pred_rfc)
print('Forest score')
print("Accuracy:", bn)
print("Precision:",metrics.precision_score(y_test, pred_rfc))
print("Recall:",metrics.recall_score(y_test, pred_rfc))
dm = accuracy_score(y_test, pred_clf)
print('\nSVM Classifier score')
print("Accuracy:", dm)
print("Precision:",metrics.precision_score(y_test, pred_clf))
print("Recall:",metrics.recall_score(y_test, pred_clf))

cm = accuracy_score(y_test, pred_mlpc)
print("\nNeural Network score")
print("Accuracy:", cm)
print("Precision:",metrics.precision_score(y_test, pred_mlpc))
print("Recall:",metrics.recall_score(y_test, pred_mlpc))
