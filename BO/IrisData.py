from pydantic import BaseModel






class IrisData(BaseModel):
    """ Structure du Dataset au format Excel/Csv """
    sepalLength: float
    sepalWidth: float
    petalLength: float
    petalWidth: float
    prediction: str

