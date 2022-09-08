import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

MODEL_PATH = "trained_model"

scaler = StandardScaler()
x = pd.read_csv("./datasets/breast-cancer-wisconsin.csv")[["area_mean", "area_se", "texture_mean", "concavity_worst", "concavity_mean"]]
scaler.fit(x)

mdl = tf.keras.models.load_model(MODEL_PATH)

class RequestModel(BaseModel):
    area_mean: float
    area_se: float
    texture_mean: float
    concavity_worst: float
    concavity_mean: float

class ResponseModel(BaseModel):
    prediction: str

app = FastAPI()

@app.get("/")
def read_root():
    return {"check": "success"}

@app.post("/predict", response_model=ResponseModel)
def read_item(patient_data: RequestModel):
    inps = pd.DataFrame([dict(
            area_mean=patient_data.area_mean,
            area_se=patient_data.area_se,
            texture_mean=patient_data.texture_mean,
            concavity_worst=patient_data.concavity_worst,
            concavity_mean=patient_data.concavity_mean,
        )
    ])
    norm_inps = scaler.transform(inps)
    pred = mdl.predict(norm_inps)[0]
    prediction = "malignant" if pred[0] > pred[1] else "benign"
    return dict(prediction=prediction)