import pandas as pd
import opendatasets as od

def download_dataset():

    od.download("https://www.kaggle.com/datasets/uciml/iris")
    
    return "OK"
