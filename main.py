from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.ensemble import RandomForestClassifier
import joblib
from modele import predict, initializeModel
from bo import IrisData






# *********************************** Commande pour démarrer l'application *********************************** #
# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008






# *********************************** Chargement de l'Api *********************************** #

app = FastAPI()






# *********************************** Api de test *********************************** #


# Route de test :
@app.get("/ping")
async def pong():
    return {"ping": "pong!"}






# *********************************** Route de l'Api qui initialise le modèle *********************************** #


class StockOutInitialize(BaseModel):
    succes: str


# Route de l'Api qui initialise le modèle :
@app.get("/initialize-model", response_model=StockOutInitialize, status_code=200)
async def initialize():

    # Ré-initialisation du modèle :
    message = initializeModel()

    return StockOutInitialize(succes=message)






# *********************************** Route de l'Api qui appelle le modèle *********************************** #


# Objet en entrée :
class StockIn(BaseModel):
        sepal_length: float
        sepal_width: float
        petal_length: float
        petal_width: float


# Objet en sortie :
class StockOut(StockIn):
    forecast: dict


# Route de l'Api qui appelle le modèle :
@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):

    # Exécution du modèle :
    prediction_list = predict(payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width)
    print("log :", prediction_list)

    # Gestion des erreurs :
    if not prediction_list:
        raise HTTPException(status_code=400, detail="Model not found.")

    # Renvoie du résultat :
    response_object = StockOut(
        sepal_length=payload.sepal_length,
        sepal_width=payload.sepal_width,
        petal_length=payload.petal_length,
        petal_width=payload.petal_width,
        forecast={'response': prediction_list[0]}
    )
    return response_object






# *********************************** Route de l'Api qui entraine le modèle avec les prédictions qu'il produit *********************************** #


# Objet en entrée :
class StockUserIn(BaseModel):
    data_lines: List[IrisData]


# Route de l'Api qui entraine le modèle avec les prédictions qu'il a produite :
@app.post("/load-predict-in-model", status_code=200)
def load_model(payload: StockUserIn):
    # 1- Chargement des données dans une structure de données :
    userDataset = {'data': [], 'target': []}
    for line in payload.data_lines:
        selected_sepal_length = line.sepalLength
        selected_sepal_width = line.sepalWidth
        selected_petal_length = line.petalLength
        selected_petal_width = line.petalWidth
        generated_prediction = line.prediction
        print(" ************** TEST ************** ")
        print(line.sepalLength)
        print(line.sepalWidth)
        print(line.petalLength)
        print(line.petalWidth)
        print(line.prediction)
        print(" ************** TEST ************** ")

    # 2- Entrainement du modèle :
    # Chargement du set de données :
    userDataset['data'].append([selected_sepal_length, selected_sepal_width, selected_petal_length,
                        selected_petal_width])
    userDataset['target'].append(generated_prediction)

    # Entrainement du modèle :
    modele = RandomForestClassifier()
    modele.fit(userDataset['data'], userDataset['target'])
    joblib.dump(modele, 'modele.joblib')


