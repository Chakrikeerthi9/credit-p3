from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.config import settings
from app.database import get_client

@asynccontextmanager
async def lifespan(app: FastAPI):
    get_client()
    print("Supabase client initialized")
    yield
    print("Shutting down")

app = FastAPI(
    title="Credit Risk ML API",
    version=settings.MODEL_VERSION,
    lifespan=lifespan
)

from app.routes import health, predict, audit
app.include_router(health.router)
app.include_router(predict.router)
app.include_router(audit.router)
