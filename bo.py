from pydantic import BaseModel






# *********************************** Ligne du DataSet d'entrainement Excel ou Csv *********************************** #
class IrisData(BaseModel):
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float
    prediction: str


