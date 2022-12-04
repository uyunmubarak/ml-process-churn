from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import utils
from utils import load_config, pickle_dump, pickle_load
import data_collection as data_collection
import preprocessing as preprocessing
import uvicorn

config = utils.load_config()
ohe_categorical = utils.pickle_load(config["ohe_categorical_path"])
std_scaler = utils.pickle_load(config["scaler_path"])
model_data = utils.pickle_load(config["production_model_path"])

class churn_data(BaseModel):
    age : int
    days_since_last_login : float
    points_in_wallet : float
    gender : str
    region_category : str
    membership_category : str
    joined_through_referral : str
    preferred_offer_types : str
    medium_of_operation : str
    internet_option : str
    used_special_discount : str
    offer_application_preference : str
    past_complaint : str
    complaint_status : str
    feedback : str

app = FastAPI()

@app.get("/")
def home():
    return "Hello, FastAPI up!"

@app.post("/predict/")
def predict(data: churn_data):    
    # Convert data api to dataframe
    data = pd.DataFrame(data).set_index(0).T.reset_index(drop = True)
    data.column = config["predictors"]

     # Convert dtype
    data = pd.concat(
        [
            data[config["predictors"][1:3]].astype(np.float64),  # type: ignore
            data[config["predictors"][0]].astype(np.int64),  # type: ignore
            data[config["predictors"][3:]]
        ],
        axis = 1
    )

    # ohe
    data = preprocessing.ohe_transform(data, ohe_categorical)

    # scaling
    data = preprocessing.std_scaler_transform(data, std_scaler)

    # Predict data
    y_pred = str(model_data.predict(data))[1]

    return {"prediction" : y_pred}

if __name__ == "__main__":
    uvicorn.run("api:app", host = "0.0.0.0", port = 8080)
