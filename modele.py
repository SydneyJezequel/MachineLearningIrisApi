from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import pandas as pd
from bo import iris_data as bo_iris, initializeDataSet






# ******************** Entrainement du modèle de Machine Learning ******************** #

def initializeModel():

    # Purge de la structure du dataset :
    bo_iris = initializeDataSet()

    # Chargement du set de données de base :
    iris_dataset = datasets.load_iris()

    # Chargement des 'data' et 'target' dans la structure 'bo_iris' :
    bo_iris['data'] = iris_dataset.data.tolist()
    bo_iris['target'] = iris_dataset.target.tolist()
    bo_iris['target_names'] = iris_dataset.target_names.tolist()

    # Entrainement du modèle :
    irisModel = RandomForestClassifier()
    irisModel.fit(bo_iris['data'], bo_iris['target'])
    joblib.dump(irisModel, 'modele.joblib')
    return "modele re-initialise"






# *********************************** Méthode de calcul des prédictions *********************************** #

def predict(sepal_length, sepal_width, petal_length, petal_width):

    print("****************** TEST ******************** ")
    print("Data : ", bo_iris['data'])
    print("Target : ", bo_iris['target'])
    print("Description du dataset : ", bo_iris['DESCR'])
    print("Variables indépendantes (features) : ",  bo_iris['feature_names'])
    print("Noms des prédictions: ", bo_iris['target_names'])
    print("****************** TEST ******************** ")

    # Path du fichier ou est enregistré le modèle :
    model_file = Path(__file__).resolve(strict=True).parent.joinpath("modele.joblib")
    # Si le modèle n'est pas sauvegardé :
    if not model_file.exists():
        return False
    # Charger le modèle :
    model = joblib.load("modele.joblib")
    # Encapsulation des paramètres dans un dictionnaire :
    parametres_iris = input(sepal_length, sepal_width, petal_length, petal_width)
    # Exécution du modèle :
    forecast = model.predict(parametres_iris)
    # On récupère le forecast :
    forecast = int(forecast[0])
    # Renvoi des prédiction :
    return bo_iris['target_names'][forecast]






# *********************************** Méthode qui encapsule les valeurs dans un dictionnaire *********************************** #

def input(sepal_length, sepal_width, petal_length, petal_width):
    # Encapsulation des paramètres dans un dictionnaire :
    data = {
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    # Intégration dans le Dataframe :
    parametres = pd.DataFrame(data, index=[0])
    # Renvoie des paramètres :
    return parametres













    """
    Il y a 2 éléments à charger : Le Modèle et le DataSet.
    DANS CETTE METHODE JE VAIS DEVOIR CHARGER UN SET DE DONNEES :

     - Le code envoyé par Linkedin me donne une façon de structurer le nouveau jeu de données
     en dataSet interprétable par mon modèle. Je dois l'intégrer dans la méthode load_model() /
     @app.post("/load-predict-in-model", status_code=200).

     - Pour charger l'IrisDataSet ou le jeu de données EXCEL/CSV, je dois charger ces 2 fichiers dans
     une variable datasetGlobaal auquel les méthodes predict() (pour faire ces prédictions),
     load_model() (pour charger un nouveau jeu de données), initializeModel() (pour ré-initialiser l'IrisDataSet)
     doivent accéder.
    """