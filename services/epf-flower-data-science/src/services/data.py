import os
from kaggle.api.kaggle_api_extended import KaggleApi
import opendatasets as od
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import joblib
import json
import pickle
from sklearn.linear_model import LogisticRegression
from flask import Flask, request, jsonify
import logging

def get_kaggle_data():
    api = KaggleApi()
    api.authenticate()
    api.dataset_download_files('uciml/iris', path='src/data/', unzip=True)

    return {"status": "Dataset downloaded successfully"}

def load_kaggle_data_json():
    file_path = 'src/data/Iris.csv'
    try:
        df = pd.read_csv(file_path)
        return df.to_dict(orient='records')
    except FileNotFoundError:
        return {"error": "Dataset file not found."}

def process_species_data():
    data = load_kaggle_data_json()
    for record in data:
        if 'Species' in record and record['Species'].startswith('Iris-'):
            record['Species'] = record['Species'].replace('Iris-', '')
    return data

def split_dataset():
    dataset = process_species_data() # (extract features and labels)
    features = [list(record.values())[1:-1] for record in dataset] # ID field excluded
    labels = [record['Species'] for record in dataset]
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test 

def train_and_save_model():

    X_train, X_test, y_train, y_test = split_dataset()

    # Load the parameter of the classifier
    with open('src/config/model_parameters.json') as f:
        parameters = json.load(f)

    n_neighbors = int(parameters['n_neighbors']) 

    # Train the KNeighbors model
    KNN_model = KNeighborsClassifier(n_neighbors=n_neighbors)
    KNN_model.fit(X_train, y_train)

    if not os.path.exists('src/models'):
        os.makedirs('src/models')

    # pickle.dump(KNN_model, open('KNN_model.sav', 'wb'))
    joblib.dump(KNN_model, 'src/models/KNN_model.pkl')

def make_prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm):
    model = joblib.load('src/models/KNN_model.pkl')
    prediction = model.predict([[SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm]])

    return prediction.tolist()
