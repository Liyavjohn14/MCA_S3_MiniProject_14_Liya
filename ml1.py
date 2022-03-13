# -*- coding: utf-8 -*-
"""ml1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/168_S8lcEv6-mszryW2J9crh4we_HQByg
"""

from flask import Flask, render_template, request
import time
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
import numpy as np
from imblearn.combine import SMOTEENN
from imblearn.combine import SMOTETomek
import sklearn
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,roc_auc_score,ConfusionMatrixDisplay,precision_score,recall_score,f1_score,classification_report,roc_curve,plot_roc_curve,auc,precision_recall_curve,plot_precision_recall_curve,average_precision_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split

def fitModel():
    df = pd.read_csv("train_strokes.csv")
    df = df.drop(['id'], axis=1)

    df = df.drop(df[df['gender'] == 'Other'].index)
    oldX, y = df.drop('stroke', axis=1).values, df['stroke'].values
    encoder = OneHotEncoder(handle_unknown='ignore')
    ct = ColumnTransformer([('encoder', encoder, [0, 4, 5, 6, 9])], remainder='passthrough')
    X = ct.fit_transform(oldX)
    smote_enn = SMOTEENN(random_state=0)
    X_res, y_res = smote_enn.fit_resample(X, y)

    smote_tomek = SMOTETomek(random_state=0)
    X_res, y_res = smote_tomek.fit_resample(X, y)
    scaler = StandardScaler()
    X = scaler.fit_transform(X) 
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res)

  

    """MODEL TUNING"""

    grid_models = [(XGBClassifier(), [{'learning_rate': [0.01, 0.05, 0.1], 'eval_metric': ['error']}]),
                    (KNeighborsClassifier(),[{'n_neighbors':[5,7,8,10], 'metric': ['euclidean', 'manhattan', 'chebyshev', 'minkowski']}]),  
                    (RandomForestClassifier(),[{'n_estimators':[100,150,200],'criterion':['gini','entropy'],'random_state':[0]}])]

    for i,j in grid_models:
        grid = GridSearchCV(estimator=i,param_grid = j, scoring = 'accuracy',cv = 10)
        grid.fit(X_train,y_train)
        best_accuracy = grid.best_score_
        best_param = grid.best_params_
        print(' {}: \n Best Accuracy: {:.2f} %'.format(i,best_accuracy*100))
        print('')
        print('-'*25)
        print('')

    from sklearn.ensemble import VotingClassifier
    clf1 = XGBClassifier(learning_rate=0.1,objective='binary:logistic',random_state=0,eval_metric='mlogloss')
    clf2 = KNeighborsClassifier()
    clf3 = RandomForestClassifier()
    model = VotingClassifier(estimators=[('XGB', clf1), ('KNN', clf2), ('RF', clf3)], voting='soft')

    pipe = model.fit(X_train, y_train)
    return ct, pipe