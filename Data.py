# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 12:14:00 2019

@author: Rohan
"""
#importing the libraties
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random

#importing the test data and train data
df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')

#looking into data before processing 
df_train.info()
df_test.info()      
df_train.Survived.value_counts(normalize = True).plot(kind='bar', alpha = 0.7)
df_train.Pclass[df_train['Survived'] == 1].value_counts(normalize = True).plot(kind='bar', alpha = 0.7)
df_train.boxplot(column='Age', by='Pclass')

#Evaluating missing values of column 'Age' 
def impute_age(cols):
  Age = cols[0]
  Pclass = cols[1]
  if pd.isnull(Age):
      if Pclass == 1:
          return random.randint(35,42)
      elif Pclass == 2:
          return random.randint(27,29)
      else:
          return random.randint(21,24)
  else:
      return Age
  
data = [df_train, df_test]
for dataframe in data:
    dataframe['Age'] = dataframe[['Age', 'Pclass']].apply(impute_age, axis = 1)
    dataframe['Age'] = dataframe['Age'].astype('int64')

#Encoding sex column 
sex = {'male': 1, 'female': 0}
data = [df_train, df_test]
for dataframe in data:
    dataframe['Sex'] = dataframe['Sex'].map(sex)
    
#Filling Missing values of column 'Embarked'
data = [df_train, df_test]
for dataframe in data:
    dataframe['Embarked'] = dataframe['Embarked'].fillna('S')
    dataframe['Embarked'] = dataframe['Embarked'].astype('str')

#Encoding Embarked column 
embarked = {'S': 0, 'C': 1, 'Q': 2}  
data = [df_train, df_test]     
for dataframe in data:
    dataframe['Embarked'] = dataframe['Embarked'].map(embarked)    
    
df_train = df_train.drop(['Ticket'], axis=1)
df_test = df_test.drop(['Ticket'], axis=1)
#Evaluating missing 
data = [df_train, df_test]
c = {'A': 0, 'B': 1, 'C': '2', 'D': '3', 'E': '4','F': '5', 'G': '6'}
for dataframe in data:
    dataframe['Cabin_Up'] = [x[0] if isinstance(x, str) else np.nan for x in dataframe['Cabin']]
    dataframe['Cabin_Up'] = dataframe['Cabin_Up'].map(c)

df_train.boxplot(column='Fare', by='Cabin_Up')

data = [df_train, df_test]
for dataset in data:
    dataset['Fare'] = dataset['Fare'].fillna(0)
    dataset['Fare'] = dataset['Fare'].astype('int64')
    
data = [df_train, df_test]
for dataframe in data:
    dataframe.loc[ dataframe['Fare'] <= 10, 'Cabin_Up'] = 5
    dataframe.loc[(dataframe['Fare'] > 10) & (dataframe['Fare'] <= 30), 'Cabin_Up'] = 0
    dataframe.loc[(dataframe['Fare'] > 30) & (dataframe['Fare'] <= 45), 'Cabin_Up']   = 4
    dataframe.loc[(dataframe['Fare'] > 45) & (dataframe['Fare'] <= 55), 'Cabin_Up']   = 3
    dataframe.loc[(dataframe['Fare'] > 55) & (dataframe['Fare'] <= 75), 'Cabin_Up']   = 1
    dataframe.loc[ dataframe['Fare'] > 75, 'Cabin_Up'] = 2
    dataset['Cabin_Up'] = dataset['Cabin_Up'].astype('int64')
df_train = df_train.drop(['Cabin'], axis=1)
df_test = df_test.drop(['Cabin'], axis=1)
df_train = df_train.drop(['Name'], axis=1)
df_test = df_test.drop(['Name'], axis=1)

X_train = df_train.drop("Survived", axis=1)
Y_train = df_train["Survived"]
X_test  = df_test


from sklearn.ensemble import RandomForestClassifier
random_forest = RandomForestClassifier(n_estimators=50)
random_forest.fit(X_train, Y_train)
Y_prediction = random_forest.predict(X_test)
random_forest.score(X_train, Y_train)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors = 1)
knn.fit(X_train, Y_train)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)

pass_id = df_test["PassengerId"]

final = { 'p': pass_id,'s': Y_prediction
        }
df = pd.DataFrame(final, columns = ['p', 's'])
df.to_csv('gender_submission_predicted.csv')






