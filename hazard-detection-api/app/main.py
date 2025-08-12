from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1 import endpoints

app = FastAPI(title="Hazard Detection API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(endpoints.router, prefix="/api/v1", tags=["Detection"])


@app.get("/", tags=["Root"])
def read_root():
    return {"status": "API is running"}
