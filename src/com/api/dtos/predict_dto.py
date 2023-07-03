from pydantic import BaseModel


class PredictDto(BaseModel):
    grid_name: str
    lat: float
    long: float
