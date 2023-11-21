from pydantic import BaseModel






# *********************************** Ligne du DataSet d'entrainement Excel ou Csv *********************************** #
class IrisData(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float
    prediction: str






# *********************************** Structure et Initialisation du DataSet *********************************** #

# Variable globale : Structure et Initialisation du DataSet :
iris_data = {
    'data': [],
    'target': [],
    'feature_names': ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
    'target_names': ['setosa', 'versicolor', 'virginica'],
    'DESCR': 'Iris DataSet'
}


# Méthode qui permet de l'initialiser :
def initializeDataSet():
    # Ré-initialisation du dataSet :
    iris_data['data'] = []
    iris_data['target'] = []
    iris_data['target_names'] = []
    return iris_data


