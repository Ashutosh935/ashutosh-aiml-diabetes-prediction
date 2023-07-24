from pydantic import BaseModel
from typing import Optional
import numpy as np
import uvicorn as uvicorn
from fastapi import FastAPI
from utils import *
import pandas as pd


app = FastAPI()


class DiabetesData(BaseModel):
    Age: int
    Gender: str
    Polyuria: str
    Polydipsia: str
    sudden_weight_loss: str
    weakness: str
    Polyphagia: str
    Genital_thrush: str
    visual_blurring: str
    Itching: str
    Irritability: str
    delayed_healing: str
    partial_paresis: str
    muscle_stiffness: str
    Alopecia: str
    Obesity: str
    
    
    
@app.get('/train/')
async def Train():
    
    result = train()
    return result

@app.post('/predict/')
async def Predict(item:DiabetesData):
    global x
    x=predict(pd.DataFrame([dict(item)]))

    return x

if __name__ == '__main__':
    uvicorn.run(app, port=8080, host='0.0.0.0')
