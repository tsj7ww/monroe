from fastapi import APIRouter

from app.api.v202507.router import router as v202507_router

router = APIRouter()
router.include_router(v202507_router, prefix="/v202507", tags=["v202507"])