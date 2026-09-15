import os

from dotenv import load_dotenv
load_dotenv()  # Load .env before any adapter reads env vars

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.api.analyze import router as analyze_router


app = FastAPI(title="BHUVISION API", version="0.1.0")


# Frontend URLs allowed to communicate with the API
frontend_url = os.getenv("FRONTEND_URL")

allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

if frontend_url:
    allowed_origins.append(frontend_url)


app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "project": "BHUVISION"
    }


app.include_router(analyze_router)