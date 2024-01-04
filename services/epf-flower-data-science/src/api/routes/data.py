from src.services.data import get_kaggle_data, load_kaggle_data_json, process_species_data, split_dataset, train_and_save_model, make_prediction, get_firestore_parameters, update_firestore_parameters, create_new_firestore_parameters
from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/data")
def get_data():
    try:
        get_kaggle_data()
    except:
        return "Error: Couldn't download data."

    return 'Ok'

@router.get("/data/dowload")
def load_data():
    try:
        dataset = load_kaggle_data_json()
    except:
        return "Error: Couldn't download data in JSON format."

    return dataset

@router.get("/data/process")
def process_data():
    try:
        dataset = process_species_data()
    except:
        return "Error: Couldn't process the data."

    return dataset

@router.get("/data/split")
def split_data():
    try:
        X_train, X_test, y_train, y_test = split_dataset()
    except:
        return "Error: Couldn't split the data."

    return X_train, X_test, y_train, y_test

@router.get("/data/train")
def train_model():
    try:
        result = train_and_save_model()
    except:
        return "Error: Couldn't train the model."

    return "Model trained and saved"

@router.get("/data/prediction")
def predict(SepalLengthCm: float, SepalWidthCm: float, PetalLengthCm: float, PetalWidthCm: float):
    try: 
        prediction = make_prediction(SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
    except Exception as e:
        return f"Error: {str(e)}"
    
    return JSONResponse(content=prediction, status_code=200)

@router.get("/data/get_firestore_collection_parameters")
def create_collection():
    try:
        parametres = get_firestore_parameters()
    except:
        return "Error: couldn't get Firestone collection parameters."

    return parametres

@router.get("/data/update_firestore_collection_parameters")
def update_parameters(n_estimators: int):
    try:
        parameters = update_firestore_parameters(n_estimators)
    except:
        return "Error: couldn't update Firestone collection parameters."

    return parameters

@router.get("/data/new_firestore_collection_parameters")
def update_parameters(parameter_name: str, parameter_value: int):
    try:
        parameters = create_new_firestore_parameters(parameter_name, parameter_value)
    except:
        return "Error: couldn't create new Firestone collection parameters."

    return parameters