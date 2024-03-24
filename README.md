L'objectif de ce projet est de manipuler un modèle de type Random Forest.

Le dataset fournit par défaut est le dataset des Iris.
Il s'agit d'un dataset souvent utilisé pour se familiariser avec le modèle Random Forest.

Ce projet rassemble les fonctionnalités suivantes :
* Initialiser le modèle avec le dataset des Iris (dataset par défaut).
* Initialiser / Entrainer le modèle avec un autre dataset.
* Générer des prédictions.
* Récupérer le dataset des Iris en local (dataset par défaut).

Ces fonctionnalités sont mises à disposition via Fast API.

Le Front et le Backend pour manipuler ce modèle sont mis à disposition dans les projets suivants :
* https://github.com/SydneyJezequel/applicationIABackend
* https://github.com/SydneyJezequel/applicationIAFrontend

La commande pour lancer ce projet est :
uvicorn IrisController:app --reload --workers 1 --host 0.0.0.0 --port 8008

