from fastapi import APIRouter
import requests
import os
from src.services.data import download_dataset

router = APIRouter()

data_directory = "src/data"
file_path = os.path.join(data_directory, "iris_dataset.csv")

os.makedirs(data_directory, exist_ok=True)

@router.get("/api/routes/data")
def save_dataset():
    url = "https://www.kaggle.com/datasets/uciml/iris"
    response = requests.get(url)
    
    with open("src/data/iris_dataset.csv", "wb") as f:
        f.write(response.content)
    
    return {"message": "Dataset downloaded and saved successfully"}