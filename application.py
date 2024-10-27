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

corpus = " ".join(course_data['course_title'].astype(str))

tokens = nltk.word_tokenize(corpus.lower())
bigrams = ngrams(tokens, 2)
trigrams = ngrams(tokens, 3)

ngram_model = defaultdict(list)
for w1, w2 in bigrams:
    ngram_model[(w1,)].append(w2)
for w1, w2, w3 in trigrams:
    ngram_model[(w1, w2)].append(w3)

def predict_next_word(input_text):

    corrected_text = str(TextBlob(input_text).correct())
    words_input = nltk.word_tokenize(corrected_text.lower())
    
    if len(words_input) >= 2:
        last_two_words = tuple(words_input[-2:])
        next_words_trigram = ngram_model.get(last_two_words, [])
        if next_words_trigram:
            valid_words = [word for word in next_words_trigram if word in word_list]
            if valid_words:
                return valid_words[:3] 
    
    if len(words_input) >= 1:
        last_word = tuple(words_input[-1:])
        next_words_bigram = ngram_model.get(last_word, [])
        if next_words_bigram:
            valid_words = [word for word in next_words_bigram if word in word_list]
            if valid_words:
                return valid_words[:3]  
    
    if len(words_input) > 0:
        partial_word = words_input[-1]
        suggestions = [word for word in word_list if word.startswith(partial_word)]
        if suggestions:
            return suggestions[:3]  
    
    return ["development", "programming", "design", "software"]

@app.route('/predict_next_word', methods=['POST'])
def predict_next():
    data = request.get_json()
    next_words = predict_next_word(data['text'])
    return jsonify({'next_words': next_words})
    
@app.route('/recommend', methods=['POST'])
def recommend():
    preferences = request.form['preferences']
    keywords = extract_keywords(preferences)
    recommended_courses = find_matching_courses(keywords)

    user_skill_level = request.form['skill_level']  
    filtered_courses = filter_courses_by_skill_level(recommended_courses, user_skill_level)

    if filtered_courses:  
        filtered_courses_df = course_data[course_data['course_title'].isin(filtered_courses)]
    else:
        filtered_courses_df = pd.DataFrame()  # Create an empty DataFrame

    return render_template('result.html', courses=filtered_courses_df)
    
complete_course_data = pd.read_csv('complete_course_data.csv')

popular_courses = pd.DataFrame({
    'index': complete_course_data['index'],
    'course_title': complete_course_data['course_title'],
    'last_viewed': pd.NA  
})

@app.route('/course/<int:course_id>')
def course_view(course_id):
    course_data = pd.read_csv('complete_course_data.csv')
    popular_courses = pd.read_csv('popular_courses.csv', sep=';')

    if popular_courses['index'].duplicated().sum() > 0:
        print("Duplicate indices found in popular courses. Resetting indices to make them unique.")
        popular_courses = popular_courses.drop_duplicates(subset=['index'], keep='first')
        popular_courses.to_csv('popular_courses.csv', sep=';', index=False) 

    if course_id not in course_data['index'].values:
        return "Course not found", 404

    course_url = course_data.loc[course_data['index'] == course_id, 'url'].values[0]

    if not is_valid_url(course_url):
        return "Course link is invalid or no longer offered", 404


    if course_id in popular_courses['index'].values:
        popular_courses.loc[popular_courses['index'] == course_id, 'views'] += 1
        popular_courses.loc[popular_courses['index'] == course_id, 'last_viewed'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    else:
   
        new_entry = pd.DataFrame({
            'index': [course_id],
            'views': [1],
            'last_viewed': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
        })
        popular_courses = pd.concat([popular_courses, new_entry], ignore_index=True)

    popular_courses.to_csv('popular_courses.csv', sep=';', index=False)

    return redirect(course_url)

def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False
