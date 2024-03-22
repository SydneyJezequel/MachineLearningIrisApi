from pydantic import BaseModel
from typing import List






iris_data = {
    """ Structure du DataSet (Variable globale) """
    'data': [],
    'target': [],
    'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    'target_names': ['setosa', 'versicolor', 'virginica'],
    'DESCR': 'Iris DataSet'
}




class IrisData(BaseModel):
    """ Structure du Dataset au format Excel/Csv """
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float
    prediction: str






""" *********************************** Paramètres des API *********************************** """


class StockOutInitialize(BaseModel):
    """  Controller : initialize() """
    succes: str




class StockIn(BaseModel):
    """ Controller : get_prediction(payload: StockIn) | Objet en entrée """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float




class StockOut(StockIn):
    """ Controller : get_prediction(payload: StockIn) | Objet en sortie """
    forecast: dict




class StockUserIn(BaseModel):
    """ Controller : get_prediction(payload: StockIn) | Objet en entrée"""
    data_lines: List[IrisData]




class StockOutIrisDataSet(BaseModel):
    """ Controller send_iris_data_set() | Objet en sortie """
    data_lines: List[IrisData]

