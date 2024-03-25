import os
import joblib
import pandas as pd
import config
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from BO.IrisDataStructure import iris_data_structure
from BO.TrainDataset import TrainDataset
from BO.IrisData import IrisData
from BO.IrisDataSetLines import IrisDataSetLines






class IrisModelService:
    """ Service qui manipule le modèle Random Forest """





    """ ************************ Constructeur ************************ """
    _instance = None

    def __new__(cls):
        """ Constructeur """
        if cls._instance is None:
            cls._instance = super(IrisModelService, cls).__new__(cls)
        return cls._instance





    """ ************************ Méthodes ************************ """

    def initializeModel(self):
        """ Méthode qui initialise le modèle """
        # Purge de la structure du dataset :
        iris_data_structure = self.initializeDataSet()
        # Chargement du set de données de base :
        iris_dataset = datasets.load_iris()
        # Chargement des 'data' et 'target' dans la structure 'iris_data_structure' :
        iris_data_structure['data'] = iris_dataset.data.tolist()
        iris_data_structure['target'] = iris_dataset.target.tolist()
        iris_data_structure['target_names'] = iris_dataset.target_names.tolist()
        # Entrainement du modèle :
        irisModel = RandomForestClassifier()
        irisModel.fit(iris_data_structure['data'], iris_data_structure['target'])
        # Sauvegarde du modèle :
        model_file = config.MODEL_PATH
        joblib.dump(irisModel, model_file)
        # Message de fin :
        return "modele re-initialise"



    def predict(self, sepal_length, sepal_width, petal_length, petal_width):
        """ Méthode qui calcule les prédictions """
        print("Data : ", iris_data_structure['data'])
        print("Target : ", iris_data_structure['target'])
        print("Description du dataset : ", iris_data_structure['DESCR'])
        print("Variables indépendantes (features) : ",  iris_data_structure['feature_names'])
        print("Noms des prédictions: ", iris_data_structure['target_names'])
        # Chargement du modèle :
        model_file = config.MODEL_PATH
        # Si le modèle n'est pas sauvegardé :
        if not os.path.exists(model_file):
            return False
        model = joblib.load(model_file)
        # Encapsulation des paramètres dans un dictionnaire :
        parametres_iris = self.input(sepal_length, sepal_width, petal_length, petal_width)
        # Exécution du modèle :
        prediction = model.predict(parametres_iris)
        print("PREDICTION : ", prediction)
        # On récupère le forecast :
        prediction = int(prediction[0])
        # Renvoi des prédiction :
        return iris_data_structure['target_names'][prediction]



    def input(self, sepal_length, sepal_width, petal_length, petal_width):
        """ Méthode qui encapsule les paramètres dans un dictionnaire """
        # Encapsulation des paramètres dans un dictionnaire :
        data = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width
        }
        # Intégration dans le Dataframe :
        parameters = pd.DataFrame(data, index=[0])
        # Renvoie des paramètres :
        return parameters



    def initializeDataSet(self):
        """ Méthode qui ré-initialise les valeurs DataSet à 0 """
        # Ré-initialisation du dataSet :
        iris_data_structure['data'] = []
        iris_data_structure['target'] = []
        iris_data_structure['target_names'] = []
        return iris_data_structure



    def load_new_data_set(self, payload: TrainDataset):
        """ Méthode qui charge et entraine un nouveau jeu de données """
        # 1- Initialisation de la structure de données :
        iris_data_structure = self.initializeDataSet()
        # Dictionnaire pour stocker la correspondance entre les prédictions et les cibles
        target_names_numbers = {}
        # 2- Chargement des données dans une structure de données :
        for line in payload.data_lines:
            print('sepal length ', line.sepalLength)
            print('sepal width ', line.sepalWidth)
            print('petal length ', line.petalLength)
            print('petal width ', line.petalWidth)
            print('prediction ', line.prediction)
            # Intégration des données dans le dataset :
            iris_data_structure['data'].append([line.sepalLength, line.sepalWidth, line.petalLength, line.petalWidth])
            # Traitement pour définir les Targets (nombre d'étiquettes) et les Targets_names (étiquettes) :
            if line.prediction not in iris_data_structure['target_names']:
                # Ajoute la nouvelle prédiction à target_names :
                iris_data_structure['target_names'].append(line.prediction)
                # Initialisation avec la première target à 0 :
                if not iris_data_structure['target']:
                    iris_data_structure['target'].append(0)
                    target_names_numbers[line.prediction] = 0
                # Initialisation avec les targets suivantes :
                else:
                    max_target = max(iris_data_structure['target'], default=0)
                    iris_data_structure['target'].append(max_target + 1)
                    # Actualiser le dictionnaire :
                    target_names_numbers[line.prediction] = max_target + 1
            else:
                # On récupère la target correspondant au target_name via le dictionnaire :
                target = target_names_numbers[line.prediction]
                # On actualise la target :
                iris_data_structure['target'].append(target)
        # A la fin de la boucle : On trie le tableau en ordre croissant :
        iris_data_structure['target'].sort()
        print('iris_data_structure[data]', iris_data_structure['data'])
        print('iris_data_structure[target] ', iris_data_structure['target'])
        print('iris_data_structure[target_names]', iris_data_structure['target_names'])
        # 3- Entrainement du modèle :
        if iris_data_structure['data'] and iris_data_structure['target']:
            modele = RandomForestClassifier()
            modele.fit(iris_data_structure['data'], iris_data_structure['target'])
            joblib.dump(modele, config.MODEL_PATH)
        else:
            print("Aucune donnée disponible pour l'entraînement du modèle.")



    def get_iris_data_set(self):
        """ Méthode qui renvoie le dataset de classification des Iris"""
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
        stock_out_iris_data_set = IrisDataSetLines(data_lines=iris_data_list)
        # log :
        print(stock_out_iris_data_set)
        return stock_out_iris_data_set

