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

#Apply preprocessing to the course data initially
def preprocess_data(data):
    data['level'] = data['level'].str.strip()  
    return data
    
course_data = preprocess_data(course_data)
prolog = Prolog()
#Populating a Prolog database with course information
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
    
    # Applies standard scaling to the encoded columns to normalize them
    
    data[['platform_encoded', 'level_encoded', 'certification_encoded']] = scaler.fit_transform(
        data[['platform_encoded', 'level_encoded', 'certification_encoded']])
    
      # Return the processed DataFrame
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

#trains and evaluates multiple machine learning models, and identifies the best-performing model based on cross-validation scores

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

# Initialize N-gram model
ngram_model = defaultdict(Counter)

# Populate N-gram model from corpus data
corpus = " ".join(course_data['course_title'].astype(str))
tokens = nltk.word_tokenize(corpus.lower())

# Build the bigrams and trigrams
bigrams = ngrams(tokens, 2)
trigrams = ngrams(tokens, 3)

for w1, w2 in bigrams:
    ngram_model[(w1,)][w2] += 1
for w1, w2, w3 in trigrams:
    ngram_model[(w1, w2)][w3] += 1

# Define function to predict next word
def predict_next_word(ngram_model, prev_words, n=1):
    prev_words = tuple(prev_words.lower().split())
    if prev_words in ngram_model:
        predicted_words = ngram_model[prev_words].most_common(n)
        return [word for word, _ in predicted_words]
    elif (prev_words[-1],) in ngram_model:
        predicted_words = ngram_model[(prev_words[-1],)].most_common(n)
        return [word for word, _ in predicted_words]
    else:
        return ["No prediction available"]

# API route to handle word prediction requests
@app.route('/predict_next_word', methods=['POST'])
def predict_next():
    data = request.get_json()
    input_text = data.get('text', '')
    suggestions = predict_next_word(ngram_model, input_text)
    return jsonify({'next_words': suggestions})

    #defining a Flask route for recommending courses based on user preferences and skill level.

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
    
#analyzing a given text string and return a list of meaningful keywords while filtering out common, unimportant words

def extract_keywords(text):
    stop_words = set(stopwords.words('english'))
    words = word_tokenize(text.lower())
    return [word for word in words if word.isalpha() and word not in stop_words]
    
#search for courses in a dataset that match a list of keywords, either in the course titles or the organizations offering those courses

def find_matching_courses(keywords):
    matching_courses = course_data[
        course_data['course_title'].str.contains('|'.join(keywords), case=False, na=False) |
        course_data['organization'].str.contains('|'.join(keywords), case=False, na=False)
    ]
    return matching_courses

#filtering a list of courses based on the specified skill level provided by the user

def filter_courses_by_skill_level(courses, skill_level):
    print(f"Skill level from form: {skill_level}")

    skill_level = skill_level.capitalize()
    
    print(f"Capitalized skill level: {skill_level}")

    prolog = Prolog()

    prolog.assertz("filter_course('Beginner', Course) :- course(Course, _, 'Beginner', _, _)")
    prolog.assertz("filter_course('Intermediate', Course) :- course(Course, _, 'Intermediate', _, _)")
    prolog.assertz("filter_course('Expert', Course) :- course(Course, _, 'Expert', _, _)")

    if skill_level == 'All':
        return courses['course_title'].tolist()

    filtered_courses = []
    for course in courses['course_title']:
        prolog_query = f"filter_course('{skill_level}', '{course}')"
             
        try:
            result = list(prolog.query(prolog_query))
            print(f"Query result for {course}: {result}") 
            if result:  
                filtered_courses.append(course)
        except Exception as e:
            print(f"Error executing query: {e}")

    return filtered_courses

complete_course_data = pd.read_csv('complete_course_data.csv')

popular_courses = pd.DataFrame({
    'index': complete_course_data['index'],
    'course_title': complete_course_data['course_title'],
    'last_viewed': pd.NA  
})

#defining a Flask route and associated function for viewing details about a specific course based on its ID and also also handling loading course data, checking for duplicates in a popular courses dataset, and validating whether the requested course exists

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

#updates the "popular courses" dataset based on user interactions with specific courses. It tracks views and the last viewed timestamp for each course and handles adding new courses if they haven't been previously recorded
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
    
#check the validity of a given URL by sending an HTTP HEAD request to that URL

def is_valid_url(url):
    try:
        response = requests.head(url, allow_redirects=True)
        return response.status_code == 200
    except requests.RequestException:
        return False
        
#Flask route that serves a webpage displaying popular courses

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

#reads and return the contents of a CSV file containing information about popular courses.

def load_popular_courses():
    return pd.read_csv('popular_courses.csv', sep=';')

#handler that serves as the entry point for the application, rendering the main index page

@app.route('/index')
def index():
    return render_template('index.html')

#renders a results page

@app.route('/result')
def result():
    return render_template('result.html')

#handler that serves the results page to the user

@app.route('/')
def home():
    return render_template('home.html')
    
#analyze and return information about trending courses based on their view counts

def trending_courses_analysis():
    popular_courses = load_popular_courses()
    
    popular_courses = popular_courses[popular_courses['views'] > 0]
    
    if popular_courses.empty:
        return "<p>No data available for trending courses.</p>"
    
    course_ids = popular_courses['index'].unique()
    results = []

    #predicts the future views of courses using time series analysis
    
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

#displays trending courses

@app.route('/trending-courses')
def trending_courses():
    trending_html = trending_courses_analysis()
    return render_template('trending_courses.html', trending_html=trending_html)

#integrates a chatbot using PyTorch

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
    
#handles chatbot interactions.

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
