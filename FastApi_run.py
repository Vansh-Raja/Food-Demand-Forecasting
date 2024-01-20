from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

import mlflow.pyfunc
import mlflow

app=FastAPI()

mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
      
class Item(BaseModel):
    name: str
    version: str
    modelInp: List[float]

@app.post("/process_data")
async def process_data(item: Item):
    
    name = item.name
    version = item.version
    modelInp = item.modelInp

    model = mlflow.sklearn.load_model(model_uri=f"models:/{name}/{version}")
    prediction = model.predict([modelInp])

    return {"prediction": prediction[0]}
