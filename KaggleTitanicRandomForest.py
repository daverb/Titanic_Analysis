"""Submission to Kaggle - Titanic Competition
Author: David Burson
Date: 12/11/2014
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sklearn

# Prepare the data for our model
df = pd.read_csv('data/train.csv', header=0)
df = df.drop(['Name','Cabin','Ticket'], axis=1)

# Interpolate present age data to fill in the missing age data
df['Age']=df['Age'].interpolate()

# Convert Pclass, Sex and Embarked to numeric columns
dummies=[]
cols=['Pclass','Sex','Embarked']
for cols in cols:
  dummies.append(pd.get_dummies(df[cols]))

titanic_dummies=pd.concat(dummies, axis=1)

# Now concat the dummies to the original data and drop the original columns
df =pd.concat((df,titanic_dummies),axis=1)
df=df.drop(['Sex','Pclass','Embarked'],axis=1)
df.info()

###### PREPARE TEST DATA

tdf = pd.read_csv('data/test.csv', header=0)
tdf = tdf.drop(['Name','Cabin','Ticket'], axis=1)

# Interpolate present age data to fill in the missing age data
tdf['Age']=tdf['Age'].interpolate()
tdf['Fare']=tdf['Fare'].interpolate()

# Convert Pclass, Sex and Embarked to numeric columns
dummies=[]
cols=['Pclass','Sex','Embarked']
for cols in cols:
  dummies.append(pd.get_dummies(tdf[cols]))

titanic_dummies=pd.concat(dummies, axis=1)

# Now concat the dummies to the original data and drop the original columns
tdf =pd.concat((tdf,titanic_dummies),axis=1)
tdf=tdf.drop(['Sex','Pclass','Embarked'],axis=1)
tdf.info()
##### END OF TEST DATA PREPARATION

## Machine Learning

X=df.values
X=np.delete(X,1,axis=1)
y=df['Survived'].values

X_results=tdf.values

from sklearn import cross_validation
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X,y,test_size=0.3,random_state=0)


from sklearn import ensemble
clf = ensemble.RandomForestClassifier(n_estimators=100)
clf.fit(X,y)

clf=ensemble.GradientBoostingClassifier()
clf.fit (X, y)

clf = ensemble.GradientBoostingClassifier(n_estimators=50)
clf.fit(X,y)

y_results = clf.predict(X_results)

output = np.column_stack((X_results[:,0],y_results))
df_results = pd.DataFrame(output.astype('int'),columns=['PassengerID','Survived'])
df_results.to_csv('results/titanic_results.csv',index=False)
