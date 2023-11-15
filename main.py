from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modele import predict






# *********************************** Chargement de l'Api *********************************** #

app = FastAPI()






# *********************************** Objets manipulés par l'Api *********************************** #


# Objet en entrée :
class StockIn(BaseModel):
        sepal_length: float
        sepal_width: float
        petal_length: float
        petal_width: float


# Objet en sortie :
class StockOut(StockIn):
    forecast: dict






# *********************************** Routes *********************************** #


# Route de test :
@app.get("/ping")
async def pong():
    return {"ping": "pong!"}


# Route de l'Api :
@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):

    prediction_list = predict(payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width)
    print("log :", prediction_list)
    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    response_object = StockOut(
        sepal_length=payload.sepal_length,
        sepal_width=payload.sepal_width,
        petal_length=payload.petal_length,
        petal_width=payload.petal_width,
        forecast={'response': prediction_list[0]}
    )
    return response_object



