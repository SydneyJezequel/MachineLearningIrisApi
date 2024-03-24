from pydantic import BaseModel
from BO.IrisData import IrisData
from typing import List






class IrisDataSetLines(BaseModel):
    """ Controller send_iris_data_set() | Objet en sortie """
    data_lines: List[IrisData]

