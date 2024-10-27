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
course_data = preprocess_data(course_data)

prolog = Prolog()

def add_courses_to_prolog(data):
    for index, row in data.iterrows():
    
        course_title = row['course_title'].replace("'", "\\'")
        platform = row['platform'].replace("'", "\\'")
        level = row['level'].replace("'", "\\'").capitalize()  
        certification = row['certification'].replace("'", "\\'")
        organization = row['organization'].replace("'", "\\'")
        
        course_fact = f"course('{course_title}', '{platform}', '{level}', '{certification}', '{organization}')"
              
        try:
            prolog.assertz(course_fact)
        except Exception as e:
            print(f"Error asserting fact: {e}")

add_courses_to_prolog(course_data)
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

@app.route('/popular-courses')
def popular_courses_page():
    popular_courses = pd.read_csv('popular_courses.csv', sep=';')

    if popular_courses['index'].duplicated().sum() > 0:
        print("Duplicate indices found in popular courses. Resetting indices.")
        popular_courses = popular_courses.drop_duplicates(subset=['index'], keep='first')
        popular_courses.to_csv('popular_courses.csv', sep=';', index=False)  # Save back after fixing

    viewed_courses = popular_courses[popular_courses['views'] > 0]

    viewed_courses_sorted = viewed_courses.sort_values(by='views', ascending=False)

    return render_template('popular_courses.html', courses=viewed_courses_sorted)

def load_popular_courses():
    return pd.read_csv('popular_courses.csv', sep=';')

@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/result')
def result():
    return render_template('result.html')

@app.route('/')
def home():
    return render_template('home.html')

def trending_courses_analysis():
    popular_courses = load_popular_courses()
    
    popular_courses = popular_courses[popular_courses['views'] > 0]
    
    if popular_courses.empty:
        return "<p>No data available for trending courses.</p>"
    
    course_ids = popular_courses['index'].unique()
    results = []

    for course_id in course_ids:
       
        course_data = popular_courses[popular_courses['index'] == course_id]
        
        dates = pd.date_range(end=datetime.now(), periods=10).to_list()
        views = np.random.poisson(lam=course_data['views'].values[0], size=10)
        ts_data = pd.Series(views, index=dates)

        model = ARIMA(ts_data, order=(1, 1, 1))
        model_fit = model.fit()

        forecast = model_fit.forecast(steps=5)
        total_forecast = forecast.sum()

        results.append((course_id, total_forecast))

    results.sort(key=lambda x: x[1], reverse=True)

    trending_html = "<ul>"
    for course_id, predicted_views in results:

        trending_html += f'<li><a href="/course/{course_id}">Course ID: {course_id}</a> - Predicted Views: {predicted_views:.2f}</li>'
    trending_html += "</ul>"

    return trending_html

@app.route('/trending-courses')
def trending_courses():
    trending_html = trending_courses_analysis()
    return render_template('trending_courses.html', trending_html=trending_html)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as b:
    intents = json.load(b)

FILE = "data.pth"
data = torch.load(FILE, map_location=device, weights_only=True)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = Chat(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "ZEN"

@app.route('/bot')
def chatb():
    return render_template('chatbot.html')
    

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    
    sentence = tokenize(user_input)
    x = words(sentence, all_words)
    x = torch.from_numpy(x).reshape(1, x.shape[0]).to(device)
    
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]

    if prob.item() > 0.75:
        for intent in intents['intents']:
            if tag == intent['tag']:
                response = random.choice(intent['responses'])
                return jsonify({"response": response})
    else:
        return jsonify({"response": f"{bot_name}: I do not understand..."})


if name == 'main':
    app.run(debug=True)

