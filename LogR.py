import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn import metrics

# This function reads comma-separated values (csv) file into a DataFrame.
# As my data is stored in a folder inside the project one folder up form the source I use ..\ to navigate out od source
# then \data\ to get inside the data folder. It has a sub folder kaggle_input. The csv files use I have used either use
# , or a ; as a delimiter
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html
data = pd.read_csv("~/PycharmProjects/First/combined_kaggle.csv", sep=',')

# This function returns the first n rows for the object. The default for n is 5 so this line prints the first 5 rows
# of the csv file contents which have been loaded into data
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.head.html
print(data.head)

# Purely integer-location based indexing for selection by position.
# From importing the files provided Sean into excel and the data provided in the pdf I saw that each file has 18
# columns which are integer indexed from 0 - 17. The first column whose index is 0 is not data from a sensor it
# is row count left over form the original conversion. I skipped this by starting to index form 1. The last 3 columns
# contain the classes so the features end at column 14. Having looked at the data I opted to go with the drivingStyle
# as it had only two classes EvenPaceStyle and AggressiveStyle each of the others had three classes.
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.iloc.html
# X contains all of the feature columns
X = data.iloc[:, 1:14]
# y contains the labels for the drivingStyle class it has either
# if you get the error IndexError: single positional indexer is out-of-bounds on this line check the delimiter
y = data.iloc[:, 17]

# Create a series of boolean values which are true for all locations in y that have the label EvenPaceStyle and false
# otherwise
positive = y == 'EvenPaceStyle'
# Create a series of boolean values which are true for all locations in y that have the label AggressiveStyle and false
# otherwise
negative = y == 'AggressiveStyle'

# Create a new array of given shape and type, filled with zeros.
# Create a 1D array the same length as yfilled with zeros named yinput
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.zeros.html
yinput = np.zeros((len(y),1)).ravel()

# This will set the value of yinput to 1 for the indices of positive which where had true written to them at line 36
yinput[positive] = 1
# This is unneeded as yinput was initialised with 0s but I include it for completeness. It will set the value of yinput
# to 0 for the indices of negative that were set to true at line 39
yinput[negative] = 0

# Split arrays or matrices into random train and test subsets
# Split the data set using the value which test_size is set to determine the proportion of the split. The data is split
# randomly using .4 means that 40% of the data goes to test 60% goes to train
#https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
X_train, X_test, y_train, y_test = train_test_split(X, yinput, test_size=.4)

# Not going to use right now but these functions scale the data sets
#https://scikit-learn.org/stable/modules/preprocessing.html#
#X_train = RobustScaler().fit_transform(X_train)
#X_test = RobustScaler().fit_transform(X_test)

#min_max_scaler = MinMaxScaler()
#X_train = min_max_scaler.fit_transform(X_train)
#X_train = StandardScaler().fit_transform(X_train)
#X_test = min_max_scaler.fit_transform(X_test)
#X_test = StandardScaler().fit_transform(X_test)

# Logistic Regression classifier. This class implements regularized logistic regression using the ‘liblinear’ library,
# ‘newton-cg’, ‘sag’, ‘saga’ and ‘lbfgs’ solvers. Note that regularization is applied by default. Based on the
# documentation which recommended liblinear for small data sets, it is the default setting for the solver
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
logreg = LogisticRegression()

# Fit the model according to the given training data
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.fit
logreg.fit(X_train, y_train)

# Predict class labels for samples in the test data X_test
# https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict
ypred_logreg = logreg.predict(X_test)

cnf_matrix = metrics.confusion_matrix(y_test, ypred_logreg)
print(cnf_matrix)

class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.show()

print("Accuracy Logistic:", metrics.accuracy_score(y_test, ypred_logreg))
print("Precision Logistic:", metrics.precision_score(y_test, ypred_logreg))
print("Recall Logistic:", metrics.recall_score(y_test, ypred_logreg))

y_pred_proba = logreg.predict_proba(X_test)[::,1]
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_proba)
auc = metrics.roc_auc_score(y_test, y_pred_proba)
plt.plot(fpr,tpr,label="data 1, auc="+str(auc))
plt.legend(loc=4)
plt.show()


