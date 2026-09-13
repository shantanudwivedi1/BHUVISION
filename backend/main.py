from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from backend.api.analyze import router as analyze_router

app = FastAPI(title="BHUVISION API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health_check():
    return {"status": "ok", "project": "BHUVISION"}


app.include_router(analyze_router)
