from pydantic import BaseModel
from BO.IrisData import IrisData
from typing import List






class TrainDataset(BaseModel):
    """ Controller : get_prediction(payload: StockIn) | Objet en entrée"""
    data_lines: List[IrisData]

