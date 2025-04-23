from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import numpy as np

app = FastAPI()

# Trenujemy prosty model ML
X = np.array([[0], [1], [2], [3]])
y = np.array([0, 0, 1, 1])
model = LogisticRegression()
model.fit(X, y)

# Klasa danych wejściowych
class InputData(BaseModel):
    value: float

@app.get("/")
def read_root():
    return {"message": "Witaj w API ML!"}

@app.post("/predict")
def predict(data: InputData):
    prediction = model.predict([[data.value]])
    return {"prediction": int(prediction[0])}

@app.get("/info")
def model_info():
    return {"model": "LogisticRegression", "features": 1}

@app.get("/health")
def health_check():
    return {"status": "ok"}
