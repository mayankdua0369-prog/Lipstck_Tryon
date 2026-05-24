from pydantic import BaseModel


class ToneProfile(BaseModel):
    undertone: str
    depth: str
    skin_hex: str | None = None


class ShadeRecommendation(BaseModel):
    family: str
    subcategory: str
    name: str
    hex: str
    undertone: str
    depth: str


class TryOnResponse(BaseModel):
    detected: bool
    image_base64: str | None = None
    tuned_hex: str | None = None
    tone_profile: ToneProfile | None = None
    recommendations: list[ShadeRecommendation] = []
    message: str | None = None


class ShadeOption(BaseModel):
    name: str
    hex: str
    undertone: str
    depth: str


class ShadeCatalogResponse(BaseModel):
    shades: dict[str, dict[str, list[ShadeOption]]]
