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
import firebase_admin
from firebase_admin import firestore
from firebase_admin import credentials, firestore


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

# def get_firestore_parameters():
#     """
#     Retrieves the parameters from the Firestore database.

#     Returns:
#         dict: A dictionary containing the parameters retrieved from the database.
#     """
#     if not firebase_admin._apps:
#         cred = credentials.Certificate('api-tp2-firebase-adminsdk-z9nj2-cd56cc936d.json')
#         firebase_admin.initialize_app(cred)

#     db = firestore.client()
#     collection_reference = db.collection("parameters").document("parameters")
#     parameters = collection_reference.get().to_dict()

#     return parameters

# def update_firestore_parameters(parameter_value):
#     """
#     Updates the value of the 'n_estimators' parameter in the Firestore database.

#     Parameters:
#         parameter_value (int): The new value for the 'n_estimators' parameter.

#     Returns:
#         dict: A dictionary representation of the updated document in the 'parameters' collection.
#     """
#     if not firebase_admin._apps:
#         cred = credentials.Certificate('api-tp2-firebase-adminsdk-z9nj2-cd56cc936d.json')
#         firebase_admin.initialize_app(cred)

#     db = firestore.client()
#     collection_reference = db.collection("parameters").document("parameters")
#     # update n_estimators parameter
#     collection_reference.update({"n_estimators": parameter_value})

#     return collection_reference.get().to_dict()

# def create_new_firestore_parameters(parameter_name, parameter_value):
#     """
#     Creates new Firestore parameters.

#     Parameters:
#         parameter_name (str): The name of the parameter to create.
#         param parameter_value (int): The value of the parameter.
    
#     Returns:
#         dict: A dictionary containing the parameters (and so the new one)and their value.
#     """
#     if not firebase_admin._apps:
#         cred = credentials.Certificate('api-tp2-firebase-adminsdk-z9nj2-cd56cc936d.json')
#         firebase_admin.initialize_app(cred)
        
#     db = firestore.client()
#     collection_reference = db.collection("parameters").document("parameters")
#     collection_reference.update({parameter_name: parameter_value})

#     return collection_reference.get().to_dict()