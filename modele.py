from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import pandas as pd






# ******************** Entrainement du modèle de Machine Learning ******************** #


def initializeModel():
    # Chargement du set de données :
    iris = datasets.load_iris()

    # Entrainement du modèle :
    irisModel = RandomForestClassifier()
    irisModel.fit(iris.data, iris.target)
    joblib.dump(irisModel, 'modele.joblib')
    return "modele re-initialise"






# *********************************** Méthode de calcul des prédictions *********************************** #


def predict(sepal_length, sepal_width, petal_length, petal_width):
    # Initialisation du set de données :
    iris = datasets.load_iris()
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
    # Renvoi des prédiction :
    return iris.target_names[forecast]






# *********************************** Méthode qui encapsule les valeurs dans un dictionnaire *********************************** #


def input(sepal_length, sepal_width, petal_length, petal_width):
    # Encapsulation des paramètres dans un dictionnaire :
    data={
        'sepal_length': sepal_length,
        'sepal_width': sepal_width,
        'petal_length': petal_length,
        'petal_width': petal_width
    }
    # Intégration dans le Dataframe :
    parametres = pd.DataFrame(data, index=[0])
    # Renvoie des paramètres :
    return parametres


