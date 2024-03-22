from fastapi import FastAPI, HTTPException
from IrisModelService import IrisModelService
from IrisBo import StockOutInitialize, StockIn, StockOut, StockUserIn, StockOutIrisDataSet






""" ********************************** Commande pour démarrer l'application ********************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn IrisController:app --reload --workers 1 --host 0.0.0.0 --port 8008






""" ********************************** Chargement de l'Api ********************************** """

app = FastAPI()






""" ********************************** Méthodes ********************************** """

@app.get("/ping")
async def pong():
    """ Api de test """
    return {"ping": "pong!"}




@app.get("/initialize-model", response_model=StockOutInitialize, status_code=200)
async def initialize():
    """ Controller qui initialise le modèle """
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Ré-initialisation du modèle :
    message = iris_model_service_instance.initializeModel()
    return StockOutInitialize(succes=message)




@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):
    """ Controller qui gère les prédictions """
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Exécution du modèle :
    prediction_list = iris_model_service_instance.predict(payload.sepal_length, payload.sepal_width, payload.petal_length, payload.petal_width)
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




@app.post("/load-predict-in-model", status_code=200)
def load_model(payload: StockUserIn):
    """ Controller qui entraine le modèle """
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Chargement des données et entrainement du modèle :
    iris_model_service_instance.load_new_data_set(payload)




@app.get("/get-iris-dataset", response_model=StockOutIrisDataSet, status_code=200)
def send_iris_data_set():
    """ Controller qui initialise le modèle """
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Initialisation du modèle :
    iris_data_set = iris_model_service_instance.get_iris_data_set()
    return iris_data_set

