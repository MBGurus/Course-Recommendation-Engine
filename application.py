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