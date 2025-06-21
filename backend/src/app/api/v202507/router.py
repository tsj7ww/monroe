from fastapi import APIRouter

from app.api.v202507.data import router as data

router = APIRouter()

router.include_router(data, prefix="/data", tags=["data"])
# router.include_router(auth, prefix="/auth", tags=["authentication"])
# router.include_router(models, prefix="/models", tags=["ml-models"])
# router.include_router(predictions, prefix="/predictions", tags=["predictions"])