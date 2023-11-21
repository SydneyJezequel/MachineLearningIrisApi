from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
from sklearn.ensemble import RandomForestClassifier
import joblib
from modele import predict, initializeModel
from bo import IrisData, initializeDataSet






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
        forecast={'response': prediction_list}
    )
    return response_object






# *********************************** Route de l'Api qui entraine le modèle avec les prédictions qu'il produit *********************************** #

# Objet en entrée :
class StockUserIn(BaseModel):
    data_lines: List[IrisData]


# Route de l'Api qui entraine le modèle avec les prédictions qu'il a produite :
@app.post("/load-predict-in-model", status_code=200)
def load_model(payload: StockUserIn):
    # 1- Initialisation de la structure de données :
    bo_iris = initializeDataSet()

    # Dictionnaire pour stocker la correspondance entre les prédictions et les cibles
    target_names_numbers = {}

    # 2- Chargement des données dans une structure de données :
    for line in payload.data_lines:
        print(" ************** TEST ************** ")
        print('sepal length ', line.sepalLength)
        print('sepal width ', line.sepalWidth)
        print('petal length ', line.petalLength)
        print('petal width ', line.petalWidth)
        print('prediction ', line.prediction)
        print(" ************** TEST ************** ")
        # Intégration des données dans le dataset :
        bo_iris['data'].append([line.sepalLength, line.sepalWidth, line.petalLength, line.petalWidth])

    #############################################################################
    # TRAITEMENT POUR DEFINIR LES TARGET (nombre d'étiquettes) ET TARGET_NAMES (étiquettes) :
    #############################################################################
        if line.prediction not in bo_iris['target_names']:
            # Ajoute la nouvelle prédiction à target_names :
            bo_iris['target_names'].append(line.prediction)
            # Trouve le chiffre le plus élevé dans target et ajoute targetMax + 1 :
            max_target = max(bo_iris['target'], default=0)
            bo_iris['target'].append(max_target + 1)
            # Actualiser le dictionnaire :
            target_names_numbers[line.prediction] = max_target + 1
        else:
            # On récupère la target correspondant au target_name via le dictionnaire :
            target = target_names_numbers[line.prediction]
            # On actualise la target :
            bo_iris['target'].append(target)
    # A la fin de la boucle : On trie le tableau en ordre croissant :
    bo_iris['target'].sort()
    #############################################################################
    # TRAITEMENT POUR
    #############################################################################
    print(" ************** TEST ************** ")
    print('bo_iris[data]', bo_iris['data'])
    print('bo_iris[target] ', bo_iris['target'])
    print('bo_iris[target_names]', bo_iris['target_names'])
    print(" ************** TEST ************** ")

    # 3- Entrainement du modèle :
    if bo_iris['data'] and bo_iris['target']:
        modele = RandomForestClassifier()
        modele.fit(bo_iris['data'], bo_iris['target'])
        joblib.dump(modele, 'modele.joblib')
    else:
        print("Aucune donnée disponible pour l'entraînement du modèle.")

