from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from modele import predict







# *********************************** Chargement de l'Api *********************************** #

app = FastAPI()







# *********************************** Api de test *********************************** #



# Route de test :
@app.get("/ping")
async def pong():
    return {"ping": "pong!"}







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
        selected_sepal_length: float
        selected_sepal_width: float
        selected_petal_length: float
        selected_petal_width: float
        generated_prediction: str













# ******************************************* TEST ******************************************* #
# ******************************************* TEST ******************************************* #
# ******************************************* TEST ******************************************* #


# Route de l'Api qui entraine le modèle avec les prédictions qu'il a produite :
@app.post("/load-predict-in-model", status_code=200)
def load_model(payload : StockUserIn):
    # 1- Charger les données Java dans une structure de données pandas ou numpy
        # Assurez-vous d'avoir les données d'entraînement et les étiquettes correspondantes
        # par exemple, X_train, y_train = load_data_from_java()
    for ligne in payload:
        selected_sepal_length = ligne.sepal_length
        selected_sepal_width = ligne.sepal_width
        selected_petal_length = ligne.petal_length
        selected_petal_width = ligne.petal_width
        generated_prediction = ligne.prediction

    # 2- Entrainement du modèle :
    # Chargement du set de données :
    userDataset.append([selected_sepal_length, selected_sepal_width, selected_petal_length,
                        selected_petal_width, generated_prediction])

    # Entrainement du modèle :
    IrisModelTrainByUser = RandomForestClassifier()
    IrisModelTrainByUser.fit(userDataset.data, userDataset.target)
    joblib.dump(irisModel, 'IrisModelTrainByUser.joblib')




# ******************************************* TEST ******************************************* #
# ******************************************* TEST ******************************************* #
# ******************************************* TEST ******************************************* #







