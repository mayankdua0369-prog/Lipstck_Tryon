from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from .lipstick_engine import get_shade_catalog, process_try_on
from .schemas import ShadeCatalogResponse, TryOnResponse


app = FastAPI(title="Lipstick Try-On API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/shades", response_model=ShadeCatalogResponse)
def shades():
    return {"shades": get_shade_catalog()}


@app.post("/api/try-on", response_model=TryOnResponse)
async def try_on(
    file: UploadFile = File(...),
    shade_name: str | None = Form(default=None),
    custom_hex: str | None = Form(default=None),
    opacity: float = Form(default=0.72),
    finish: str = Form(default="Matte"),
):
    image_bytes = await file.read()
    result = process_try_on(
        image_bytes=image_bytes,
        shade_name=shade_name,
        custom_hex=custom_hex,
        opacity=opacity,
        finish=finish,
    )
    return result
