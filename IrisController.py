from fastapi import FastAPI, HTTPException
from IrisModelService import IrisModelService
from IrisBo import StockOutInitialize, StockIn, StockOut, StockUserIn, StockOutIrisDataSet






""" ********************************** Commande pour démarrer l'application ********************************** """

# uvicorn main:app --reload --workers 1 --host 0.0.0.0 --port 8008
# uvicorn IrisController:app --reload --workers 1 --host 0.0.0.0 --port 8008






""" ********************************** Méthodes ********************************** """


""" Chargement de l'Api """
app = FastAPI()




""" Api de test """
@app.get("/ping")
async def pong():
    return {"ping": "pong!"}




""" Controller qui initialise le modèle """
@app.get("/initialize-model", response_model=StockOutInitialize, status_code=200)
async def initialize():
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Ré-initialisation du modèle :
    message = iris_model_service_instance.initializeModel()
    return StockOutInitialize(succes=message)




""" Controller qui gère les prédictions """
@app.post("/predict", response_model=StockOut, status_code=200)
def get_prediction(payload: StockIn):

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




""" Controller qui entraine le modèle """
@app.post("/load-predict-in-model", status_code=200)
def load_model(payload: StockUserIn):
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Chargement des données et entrainement du modèle :
    iris_model_service_instance.load_new_data_set(payload)




""" Controller qui initialise le modèle """
@app.get("/get-iris-dataset", response_model=StockOutIrisDataSet, status_code=200)
def send_iris_data_set():
    # Instanciation du service :
    iris_model_service_instance = IrisModelService()
    # Initialisation du modèle :
    iris_data_set = iris_model_service_instance.get_iris_data_set()
    return iris_data_set


