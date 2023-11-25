from pydantic import BaseModel
from typing import List








# ************************************************************************************************************** #
# ************************** Structure et Initialisation du DataSet (Variable globale) ************************* #
# ************************************************************************************************************** #

iris_data = {
    'data': [],
    'target': [],
    'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    'target_names': ['setosa', 'versicolor', 'virginica'],
    'DESCR': 'Iris DataSet'
}








# ******************************************************************************************************************** #
# ********************************************** Structure Dataset Excel/Csv ***************************************** #
# ******************************************************************************************************************** #

class IrisData(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float
    prediction: str








# ************************************************************************************************************** #
# ********************************************* Paramètres des API ********************************************* #
# ************************************************************************************************************** #




# ********************** Api "/initialize-model"     |     Méthode : initialize() ********************
class StockOutInitialize(BaseModel):
    succes: str




# ***************** Api "/predict"    |     Méthode : get_prediction(payload: StockIn) *******************

# Objet en entrée :
class StockIn(BaseModel):
        sepal_length: float
        sepal_width: float
        petal_length: float
        petal_width: float

# Objet en sortie :
class StockOut(StockIn):
    forecast: dict




# *********************  Api "/predict"    |     Méthode : get_prediction(payload: StockIn) *********************
class StockUserIn(BaseModel):
    data_lines: List[IrisData]


