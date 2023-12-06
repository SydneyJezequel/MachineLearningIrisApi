from fastapi import FastAPI, HTTPException
from IrisModeleService import predict, initializeModel, load_new_data_set
from IrisBo import StockOutInitialize, StockIn, StockOut, StockUserIn








# ****************************************************************************************************************************** #
# ******************************************** Commande pour démarrer l'application ******************************************** #
# ****************************************************************************************************************************** #

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn Iriscontroller:app --reload --workers 1 --host 0.0.0.0 --port 8008








# ****************************************************************************************************************************** #
# **************************************************** Chargement de l'Api ***************************************************** #
# ****************************************************************************************************************************** #

app = FastAPI()








# ****************************************************************************************************************************** #
# ******************************************************** Api de test ********************************************************* #
# ****************************************************************************************************************************** #

@app.get("/ping")
async def pong():
    return {"ping": "pong!"}








# ****************************************************************************************************************************** #
# ******************************************* Route de l'Api qui initialise le modèle ****************************************** #
# ****************************************************************************************************************************** #

@app.get("/initialize-model", response_model=StockOutInitialize, status_code=200)
async def initialize():

    # Ré-initialisation du modèle :
    message = initializeModel()

    return StockOutInitialize(succes=message)








# ****************************************************************************************************************************** #
# ********************************************* Route de l'Api qui appelle le modèle ******************************************* #
# ****************************************************************************************************************************** #

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








# ****************************************************************************************************************************** #
# ************************* Route de l'Api qui entraine le modèle avec les prédictions qu'il produit *************************** #
# ****************************************************************************************************************************** #

@app.post("/load-predict-in-model", status_code=200)
def load_model(payload: StockUserIn):
    load_new_data_set(payload)

