from fastapi import APIRouter, status
from fastapi.responses import Response

from com.api.dtos.predict_dto import PredictDto

router = APIRouter()


@router.post("/linear_regression")
async def predict(request_body: PredictDto) -> Response:
    # await predict_model(request_body)
    return Response(content="Hello world", status_code=status.HTTP_200_OK)
