import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import *
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
from sklearn import tree


data = pd.read_excel(r'C:\Users\Acer\OneDrive\Desktop\Labs\Project 5\finaldata.csv')

age_array = data[['Age']].to_numpy()
continue_array = data[['How likely are you to continue using your current bank?']].to_numpy()

X = age_array
y = continue_array

X_train, X_test, y_train, y_test = train_test_split(X,y, train_size=0.70, test_size=0.30, random_state=14)

scaler = MinMaxScaler()
scaled_train = scaler.fit_transform(X_train)
scaled_test = scaler.transform(X_test)

draw_tree = DecisionTreeClassifier()

draw_tree.fit(scaled_train,y_train)

pred = draw_tree.predict(scaled_test)

accuracy_score(y_test,pred)

param_grid = {'criterion': ['gini', 'entropy'],'splitter': ['best', 'random'],'max_depth': [2,4,6,8,10,None],'min_samples_split': [2,5,10,.03,.05],'min_samples_leaf': [1,5,10,.03,.05],'max_features': [None, 'auto'], 'random_state': [0]}

tune_model = GridSearchCV(DecisionTreeClassifier(), param_grid=param_grid, scoring = 'roc_auc', cv = 5, verbose=0)

tune_model.fit(scaled_train, y_train)

print('\033[1m'+'Decision Tree Parameters:{} '.format(tune_model.best_params_))

dt_tuned =  DecisionTreeClassifier(criterion='gini',min_samples_split=0.03, max_depth=None, max_features = None,min_samples_leaf=10,random_state = 0, splitter='random')

dt_tuned.fit(scaled_train,y_train)

pred = dt_tuned.predict(scaled_test)

accuracy_score(y_test,pred)

print(classification_report(y_test,pred))

fig, ax = plt.subplots(figsize=(10, 10))

plot_confusion_matrix(dt_tuned,scaled_test,y_test,ax=ax)

feature_names = age_array

#fig = plt.figure(figsize=(150,150))
#_ = tree.plot_tree(dt_tuned, feature_names=feature_names,class_names= ['yes','no'],filled=True)