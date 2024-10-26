from flask import Flask, render_template, request, jsonify, redirect
import pandas as pd
import numpy as np
import requests
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords, words
from nltk.tokenize import word_tokenize
from collections import defaultdict
from nltk.util import ngrams
from textblob import TextBlob
from pyswip import Prolog  
from datetime import datetime
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from flask import Flask, render_template, request, jsonify
import random
import json
import torch
from model import Chat
from nltk_utils import words, tokenize
from nltk.corpus import words

word_list = words.words()

app = Flask(_name_)

course_data = pd.read_csv("complete_course_data.csv")

def preprocess_data(data):

    data['level'] = data['level'].str.strip()  

    return data

le_platform = LabelEncoder()
le_level = LabelEncoder()
le_certification = LabelEncoder()
scaler = StandardScaler()

def preprocess_data(data):
    data['platform_encoded'] = le_platform.fit_transform(data['platform'])
    data['level_encoded'] = le_level.fit_transform(data['level'])
    data['certification_encoded'] = le_certification.fit_transform(data['certification'])
    
    data[['platform_encoded', 'level_encoded', 'certification_encoded']] = scaler.fit_transform(
        data[['platform_encoded', 'level_encoded', 'certification_encoded']])
    
    return data

course_data = preprocess_data(course_data)

X = course_data[['platform_encoded', 'level_encoded', 'certification_encoded']]
y = course_data['organization']  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

models = {
    'KNN': KNeighborsClassifier(),
    'Random Forest': RandomForestClassifier(n_estimators=50, random_state=0),
    'SVM': SVC(),
    'Naive Bayes': GaussianNB(),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Logistic Regression': LogisticRegression()
}

def train_and_evaluate(models, X_train, y_train, X_test, y_test):
    best_model = None
    best_score = 0
    for name, model in models.items():

        score = cross_val_score(model, X_train, y_train, cv=5).mean()
        print(f'{name} Cross-Validation Score: {score}')
        
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        conf_matrix = confusion_matrix(y_test, y_pred)
        class_report = classification_report(y_test, y_pred)
        
        print(f'Confusion Matrix for {name}:\n{conf_matrix}')
        print(f'Classification Report for {name}:\n{class_report}')
        
        if score > best_score:
            best_score = score
            best_model = model
    
    print(f'Best Model: {best_model.__class__.__name__} with score: {best_score}')
    return best_model

best_model = train_and_evaluate(models, X_train, y_train, X_test, y_test)


