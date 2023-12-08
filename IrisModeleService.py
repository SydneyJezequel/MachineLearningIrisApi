from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
import joblib
from pathlib import Path
import pandas as pd
from IrisBo import iris_data as bo_iris, StockUserIn, IrisData, StockOutIrisDataSet








# ****************************************************************************************************************************** #
# *************************************** Initialisation du modèle avec le DataSet Iris **************************************** #
# ****************************************************************************************************************************** #

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








# ****************************************************************************************************************************** #
# ********************************************** Méthode de calcul des prédictions ********************************************* #
# ****************************************************************************************************************************** #

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








# ****************************************************************************************************************************** #
# ********************************* Méthode qui encapsule les paramètres dans un dictionnaire ********************************** #
# ****************************************************************************************************************************** #

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








# ****************************************************************************************************************************** #
# ************************************** Méthode qui initialise les valeurs DataSet à 0 **************************************** #
# ****************************************************************************************************************************** #

def initializeDataSet():
    # Ré-initialisation du dataSet :
    bo_iris['data'] = []
    bo_iris['target'] = []
    bo_iris['target_names'] = []
    return bo_iris








# ****************************************************************************************************************************** #
# ******************************* Méthode qui charge et entraine un nouveau jeu de données ************************************* #
# ****************************************************************************************************************************** #

def load_new_data_set(payload: StockUserIn):


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

        # Traitement pour définir les Targets (nombre d'étiquettes) et les Targets_names (étiquettes) :
        if line.prediction not in bo_iris['target_names']:
            # Ajoute la nouvelle prédiction à target_names :
            bo_iris['target_names'].append(line.prediction)
            # Initialisation avec la première target à 0 :
            if not bo_iris['target']:
                bo_iris['target'].append(0)
                target_names_numbers[line.prediction] = 0
            # Initialisation avec les targets suivantes :
            else:
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








# ****************************************************************************************************************************** #
# ******************************* Méthode renvoie le dataset de classification des Iris ************************************* #
# ****************************************************************************************************************************** #

def get_iris_data_set():

    # Chargement du set de données de base :
    iris_dataset = datasets.load_iris()

    # Récupération des caractéristiques (data) et des étiquettes (target)
    data = iris_dataset.data
    target = iris_dataset.target

    # Création de la liste d'IrisData
    iris_data_list = []
    for i in range(len(data)):
        iris_data = IrisData(
            sepalLength=data[i, 0],
            sepalWidth=data[i, 1],
            petalLength=data[i, 2],
            petalWidth=data[i, 3],
            prediction=iris_dataset.target_names[target[i]]
        )
        iris_data_list.append(iris_data)

    # Création de l'objet StockOutIrisDataSet
    stock_out_iris_data_set = StockOutIrisDataSet(data_lines=iris_data_list)

    # log :
    print(stock_out_iris_data_set)

    return stock_out_iris_data_set

