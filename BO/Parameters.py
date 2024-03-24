from pydantic import BaseModel






class Parameters(BaseModel):
    """ Controller : get_prediction(payload: StockIn) | Objet en entrée """
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

