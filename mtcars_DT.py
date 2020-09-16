# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 21:24:02 2020

@author: kjaya
"""

#python : Topic :Decision Tree using mtcars

#standard libaries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pydataset import data
import seaborn as sns
df = data('mtcars')
df.head()
#from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.model_selection import train_test_split
from sklearn import metrics, tree
df['am'].value_counts()

#classification
#predict if transmission of car is 0 or 1 on basis of mpg, hp, wt
#0 - automatic, 1 - manual
X1 = df[['mpg','hp','wt']]
X1
Y1 = df['am']
Y1.value_counts()

#Data splitting for better prediction
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=.20)
X1_train.shape
X1_test.shape

#classification Tree
from sklearn.tree import DecisionTreeRegressor #note this
clsModel = DecisionTreeRegressor()  #model with parameter
clsModel.fit(X1_train, Y1_train)

#predict
Ypred1 = clsModel.predict(X1_test) 
Ypred1
len(Ypred1)
Y1_test

#confusion matrix
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
classification_report(y_true=Y1_test, y_pred= Ypred1)
confusion_matrix(y_true=Y1_test, y_pred=Ypred1)
accuracy_score(y_true=Y1_test, y_pred=Ypred1)

np.array(Y1_test)
np.array(Ypred1).reshape((-1,1))
df1 = Y1_test
df1['Ypred1'] = Ypred1
df1

#new data
newData = X1.sample(4)
newData
clsModel.predict(newData)
#automatic car = low mileage, heavier

#visualise 
#pip install graphviz
from graphviz import Source
from sklearn import tree
tree.plot_tree(decision_tree=clsModel)
tree.plot_tree(decision_tree=clsModel, max_depth=2, feature_names=['mpg', 'hp', ' wt'], class_names=['Org','Fake'], fontsize=12)

Source(tree.export_graphviz(clsModel))
Source(tree.export_graphviz(clsModel, max_depth=3))
dot_data1 = tree.export_graphviz(clsModel, max_depth=3, out_file=None, filled=True, rounded=True,  special_characters=True, feature_names=['mpg', 'hp', ' wt'], class_names=['Org','Fake'])  

import os
os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz/bin/'
import graphviz 
from subprocess import call
call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])
graph1 = graphviz.Source(dot_data1)  
graph1 

df.head()
df.columns

#regression
#predict if mpg (numerical value) on basis of am, hp, wt
X2 = df[['am','hp','wt']]
Y2 = df[['mpg']]
np.mean(Y2)
X2.shape
X2_train, X2_test, Y2_train, y2_test = train_test_split(X2, y2, test_size=.20, random_state=123 )
X2_train.shape
X2_test.shape