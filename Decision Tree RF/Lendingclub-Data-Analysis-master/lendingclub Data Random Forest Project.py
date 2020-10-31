#Importing the important libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().magic('matplotlib inline')

#Useing pandas to read loan_data.csv as a dataframe called loans.
loans = pd.read_csv('loan_data.csv')


#Checking out the info(), head(), and describe() methods on loans.
loans.info()

loans.describe()

loans.head()


#Exploratory Data Analysis
#Creating a histogram of two FICO distributions on top of each other, one for each credit.policy outcome.

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(loans[loans['credit.policy']==1]['fico'],kde=False, label='Credit Policy=1',color='blue')
sns.distplot(loans[loans['credit.policy']==0]['fico'],kde=False, label='Credit Policy=2',color='red')
plt.legend()
plt.xlabel('FICO')


#Creating a similar figure, except this time select by the not.fully.paid column.

plt.figure(figsize=(10,6))
sns.set_style('darkgrid')
sns.distplot(loans[loans['not.fully.paid']==1]['fico'],kde=False, label='Credit Policy=1',color='blue',bins=30)
sns.distplot(loans[loans['not.fully.paid']==0]['fico'],kde=False, label='Credit Policy=2',color='red',bins=30)
plt.legend()
plt.xlabel('FICO')

#Creating a countplot using seaborn showing the counts of loans by purpose, with the color hue defined by not.fully.paid.
plt.figure(figsize=(11,7))
sns.countplot(x='purpose',data=loans,hue='not.fully.paid')


#Creating jointplots.
sns.jointplot(x='fico',y='int.rate',data=loans)

sns.lmplot(x='fico',y='int.rate',data=loans,hue='credit.policy',col='not.fully.paid',markers='.', palette='Set1')


#Setting up the Data for Random Forest Classification Model.
loans.info()


"""
Categorical Features
Since 'Purpose' column is catagorical, i will be transforming using pd.get_dummies(loans,columns=cat_feats,drop_first=True)
to create a fixed larger dataframe that has new feature columns with dummy variables.
"""
cat_feats = ['purpose']

final_data= pd.get_dummies(loans,columns=cat_feats,drop_first=True)

final_data.info()


#Splitting the Data using Train Test Split
from sklearn.model_selection import train_test_split
X = final_data.drop('not.fully.paid',axis=1)
y = final_data['not.fully.paid']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)


#Training a Decision Tree Model
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()

dtree.fit(X_train,y_train)


#Predictions and Evaluation of Decision Tree
prediction = dtree.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(classification_report(y_test,prediction))

print(confusion_matrix(y_test,prediction))


#Training the Random Forest model
from sklearn.ensemble import RandomForestClassifier

rdtree = RandomForestClassifier(n_estimators=600)

rdtree.fit(X_train,y_train)


#Predictions and Evaluation
prediction1 = rdtree.predict(X_test)

print(classification_report(y_test,prediction1))

print(confusion_matrix(y_test,prediction1))

#Based on the result from the above code, we can determine whihc model performed better as well as analyse the data.
