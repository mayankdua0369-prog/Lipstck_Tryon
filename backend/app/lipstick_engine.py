import base64
import io
import os
from functools import lru_cache

import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)


MODEL_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    "face_landmarker.task",
)
MAX_DETECT_SIZE = 640

OUTER_LIP = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409,
    291, 375, 321, 405, 314, 17, 84, 181, 91, 146,
]
INNER_LIP = [
    78, 191, 80, 81, 82, 13, 312, 311, 310, 415,
    308, 324, 318, 402, 317, 14, 87, 178, 88, 95,
]
UPPER_OUTER_ARC = [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291]
UPPER_INNER_ARC = [308, 415, 310, 311, 312, 13, 82, 81, 80, 191, 78]
LOWER_OUTER_ARC = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291]
LOWER_INNER_ARC = [308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78]


SHADE_LIBRARY = {
    "Reds": {
        "Classic Reds": [
            {"name": "Ruby Red", "bgr": (35, 50, 192), "undertone": "neutral", "depth": "medium"},
            {"name": "Classic Red", "bgr": (32, 32, 255), "undertone": "neutral", "depth": "medium"},
            {"name": "Scarlet Pop", "bgr": (40, 58, 238), "undertone": "warm", "depth": "medium"},
            {"name": "Crimson Velvet", "bgr": (28, 28, 180), "undertone": "cool", "depth": "deep"},
            {"name": "Brick Rose", "bgr": (48, 70, 156), "undertone": "warm", "depth": "deep"},
        ],
        "Deep Reds": [
            {"name": "Bordeaux Red", "bgr": (36, 24, 120), "undertone": "cool", "depth": "deep"},
            {"name": "Wine Red", "bgr": (42, 30, 136), "undertone": "cool", "depth": "deep"},
            {"name": "Ox Blood", "bgr": (28, 22, 96), "undertone": "neutral", "depth": "deep"},
        ],
        "Orange Reds": [
            {"name": "Poppy Red", "bgr": (28, 74, 244), "undertone": "warm", "depth": "medium"},
            {"name": "Tomato Red", "bgr": (30, 86, 230), "undertone": "warm", "depth": "medium"},
            {"name": "Chili Red", "bgr": (34, 64, 206), "undertone": "warm", "depth": "deep"},
        ],
    },
    "Pinks": {
        "Soft Pinks": [
            {"name": "Baby Pink", "bgr": (182, 175, 244), "undertone": "cool", "depth": "fair"},
            {"name": "Ballet Pink", "bgr": (167, 156, 221), "undertone": "neutral", "depth": "fair"},
            {"name": "Blush Pink", "bgr": (156, 142, 214), "undertone": "neutral", "depth": "medium"},
            {"name": "Rosy Nude Pink", "bgr": (138, 126, 196), "undertone": "warm", "depth": "medium"},
            {"name": "Cloud Pink", "bgr": (188, 182, 246), "undertone": "cool", "depth": "fair"},
            {"name": "Powder Pink", "bgr": (176, 166, 232), "undertone": "neutral", "depth": "fair"},
            {"name": "Silk Petal", "bgr": (168, 154, 224), "undertone": "neutral", "depth": "medium"},
            {"name": "Peony Veil", "bgr": (160, 146, 216), "undertone": "cool", "depth": "medium"},
        ],
        "Cool Pinks": [
            {"name": "Rose Pink", "bgr": (138, 116, 232), "undertone": "cool", "depth": "medium"},
            {"name": "Lotus Pink", "bgr": (132, 94, 210), "undertone": "cool", "depth": "medium"},
            {"name": "Bubblegum Pink", "bgr": (164, 90, 248), "undertone": "cool", "depth": "fair"},
            {"name": "Fuchsia Bloom", "bgr": (126, 34, 214), "undertone": "cool", "depth": "deep"},
            {"name": "Rose Quartz", "bgr": (146, 118, 222), "undertone": "cool", "depth": "medium"},
            {"name": "Icy Pink", "bgr": (178, 130, 244), "undertone": "cool", "depth": "fair"},
            {"name": "Pink Sapphire", "bgr": (136, 62, 226), "undertone": "cool", "depth": "deep"},
            {"name": "Cool Bloom", "bgr": (152, 104, 234), "undertone": "cool", "depth": "medium"},
        ],
        "Warm Pinks": [
            {"name": "Peach Pink", "bgr": (112, 138, 228), "undertone": "warm", "depth": "fair"},
            {"name": "Strawberry Pink", "bgr": (120, 86, 224), "undertone": "warm", "depth": "medium"},
            {"name": "Hot Pink", "bgr": (146, 30, 233), "undertone": "warm", "depth": "medium"},
            {"name": "Candy Pink", "bgr": (154, 78, 240), "undertone": "warm", "depth": "medium"},
            {"name": "Watermelon Pink", "bgr": (116, 96, 232), "undertone": "warm", "depth": "medium"},
            {"name": "Guava Pink", "bgr": (124, 112, 226), "undertone": "warm", "depth": "fair"},
            {"name": "Sun Kiss Pink", "bgr": (132, 102, 218), "undertone": "warm", "depth": "medium"},
            {"name": "Coral Rose Pink", "bgr": (118, 120, 210), "undertone": "warm", "depth": "medium"},
        ],
        "Nude Pinks": [
            {"name": "Petal Nude", "bgr": (150, 144, 205), "undertone": "neutral", "depth": "fair"},
            {"name": "Muted Rose", "bgr": (126, 118, 184), "undertone": "neutral", "depth": "medium"},
            {"name": "Dusty Pink Nude", "bgr": (136, 124, 176), "undertone": "warm", "depth": "deep"},
            {"name": "Rosewood Pink", "bgr": (98, 92, 152), "undertone": "neutral", "depth": "deep"},
            {"name": "Bare Rose", "bgr": (140, 132, 190), "undertone": "neutral", "depth": "fair"},
            {"name": "Pink Sand", "bgr": (146, 138, 194), "undertone": "warm", "depth": "fair"},
            {"name": "Taupe Pink", "bgr": (118, 112, 170), "undertone": "cool", "depth": "medium"},
            {"name": "Mink Pink", "bgr": (108, 104, 160), "undertone": "neutral", "depth": "deep"},
        ],
        "Bright Pinks": [
            {"name": "Neon Pink", "bgr": (158, 40, 245), "undertone": "cool", "depth": "medium"},
            {"name": "Electric Pink", "bgr": (170, 52, 255), "undertone": "cool", "depth": "medium"},
            {"name": "Pop Fuchsia", "bgr": (132, 20, 226), "undertone": "cool", "depth": "deep"},
            {"name": "Party Pink", "bgr": (168, 64, 244), "undertone": "warm", "depth": "medium"},
            {"name": "Flash Pink", "bgr": (180, 46, 252), "undertone": "cool", "depth": "medium"},
            {"name": "Candy Flare", "bgr": (172, 72, 246), "undertone": "warm", "depth": "medium"},
            {"name": "Runway Pink", "bgr": (150, 36, 232), "undertone": "cool", "depth": "deep"},
            {"name": "Doll Pink", "bgr": (176, 86, 250), "undertone": "cool", "depth": "fair"},
        ],
        "Mauve Pinks": [
            {"name": "Mauve Whisper", "bgr": (138, 126, 190), "undertone": "cool", "depth": "medium"},
            {"name": "Vintage Mauve", "bgr": (126, 114, 176), "undertone": "neutral", "depth": "medium"},
            {"name": "Soft Plum Pink", "bgr": (118, 102, 166), "undertone": "cool", "depth": "deep"},
            {"name": "Muted Mauve Rose", "bgr": (144, 130, 196), "undertone": "neutral", "depth": "fair"},
            {"name": "Misty Mauve", "bgr": (150, 136, 202), "undertone": "cool", "depth": "fair"},
            {"name": "Dust Rose Mauve", "bgr": (132, 118, 182), "undertone": "neutral", "depth": "medium"},
        ],
        "Rose Pinks": [
            {"name": "English Rose", "bgr": (132, 106, 204), "undertone": "cool", "depth": "medium"},
            {"name": "Garden Rose", "bgr": (124, 92, 198), "undertone": "neutral", "depth": "medium"},
            {"name": "Tea Rose Pink", "bgr": (154, 130, 214), "undertone": "warm", "depth": "fair"},
            {"name": "Velvet Rose", "bgr": (110, 78, 176), "undertone": "cool", "depth": "deep"},
            {"name": "Rosette Pink", "bgr": (146, 116, 208), "undertone": "neutral", "depth": "medium"},
            {"name": "Rose Bloom", "bgr": (136, 98, 206), "undertone": "cool", "depth": "medium"},
        ],
        "Berry Pinks": [
            {"name": "Raspberry Pink", "bgr": (118, 54, 190), "undertone": "cool", "depth": "deep"},
            {"name": "Berry Blush", "bgr": (126, 70, 198), "undertone": "cool", "depth": "medium"},
            {"name": "Cranberry Pink", "bgr": (112, 64, 180), "undertone": "neutral", "depth": "deep"},
            {"name": "Jam Pink", "bgr": (104, 58, 166), "undertone": "cool", "depth": "deep"},
            {"name": "Cherry Blossom Jam", "bgr": (132, 84, 198), "undertone": "warm", "depth": "medium"},
            {"name": "Mulberry Pink", "bgr": (98, 52, 158), "undertone": "cool", "depth": "deep"},
        ],
        "Pastel Pinks": [
            {"name": "Marshmallow Pink", "bgr": (194, 188, 248), "undertone": "cool", "depth": "fair"},
            {"name": "Sorbet Pink", "bgr": (186, 176, 242), "undertone": "warm", "depth": "fair"},
            {"name": "Fairy Pink", "bgr": (190, 178, 246), "undertone": "neutral", "depth": "fair"},
            {"name": "Angel Pink", "bgr": (182, 170, 238), "undertone": "cool", "depth": "fair"},
            {"name": "Cream Pink", "bgr": (174, 162, 230), "undertone": "warm", "depth": "fair"},
            {"name": "Cotton Petal", "bgr": (188, 174, 240), "undertone": "neutral", "depth": "fair"},
        ],
        "Deep Rose Pinks": [
            {"name": "Burgundy Rose", "bgr": (92, 54, 142), "undertone": "cool", "depth": "deep"},
            {"name": "Garnet Pink", "bgr": (96, 58, 152), "undertone": "neutral", "depth": "deep"},
            {"name": "Velour Pink", "bgr": (100, 62, 160), "undertone": "cool", "depth": "deep"},
            {"name": "Noir Rose", "bgr": (82, 48, 132), "undertone": "cool", "depth": "deep"},
            {"name": "Dark Rosewood", "bgr": (90, 60, 144), "undertone": "neutral", "depth": "deep"},
            {"name": "Cabernet Pink", "bgr": (86, 50, 138), "undertone": "cool", "depth": "deep"},
        ],
    },
    "Corals": {
        "Soft Corals": [
            {"name": "Coral Crush", "bgr": (60, 96, 232), "undertone": "warm", "depth": "medium"},
            {"name": "Sunset Coral", "bgr": (70, 112, 240), "undertone": "warm", "depth": "fair"},
            {"name": "Peach Coral", "bgr": (86, 132, 228), "undertone": "warm", "depth": "fair"},
            {"name": "Tropical Coral", "bgr": (78, 102, 220), "undertone": "warm", "depth": "medium"},
        ],
        "Bright Corals": [
            {"name": "Mango Coral", "bgr": (54, 122, 246), "undertone": "warm", "depth": "medium"},
            {"name": "Flame Coral", "bgr": (42, 100, 236), "undertone": "warm", "depth": "deep"},
            {"name": "Papaya Coral", "bgr": (72, 128, 242), "undertone": "warm", "depth": "fair"},
        ],
    },
    "Nudes": {
        "Warm Nudes": [
            {"name": "Nude Beige", "bgr": (110, 144, 196), "undertone": "warm", "depth": "fair"},
            {"name": "Caramel Nude", "bgr": (90, 116, 168), "undertone": "warm", "depth": "medium"},
            {"name": "Latte Nude", "bgr": (104, 128, 176), "undertone": "neutral", "depth": "medium"},
            {"name": "Sand Beige", "bgr": (118, 152, 194), "undertone": "warm", "depth": "fair"},
        ],
        "Pink Nudes": [
            {"name": "Rose Beige", "bgr": (122, 134, 188), "undertone": "neutral", "depth": "fair"},
            {"name": "Soft Taupe Nude", "bgr": (112, 122, 166), "undertone": "cool", "depth": "medium"},
            {"name": "Toffee Rose", "bgr": (96, 106, 154), "undertone": "warm", "depth": "deep"},
        ],
        "Deep Nudes": [
            {"name": "Mocha Brown", "bgr": (82, 94, 139), "undertone": "warm", "depth": "deep"},
            {"name": "Cocoa Nude", "bgr": (68, 82, 126), "undertone": "neutral", "depth": "deep"},
            {"name": "Chestnut Nude", "bgr": (74, 92, 148), "undertone": "warm", "depth": "deep"},
        ],
    },
    "Berry & Plum": {
        "Berry": [
            {"name": "Berry Wine", "bgr": (104, 46, 125), "undertone": "cool", "depth": "deep"},
            {"name": "Mulberry Kiss", "bgr": (86, 44, 112), "undertone": "cool", "depth": "deep"},
            {"name": "Blackberry", "bgr": (58, 20, 82), "undertone": "cool", "depth": "deep"},
        ],
        "Plums": [
            {"name": "Plum Dark", "bgr": (64, 16, 74), "undertone": "cool", "depth": "deep"},
            {"name": "Deep Plum", "bgr": (74, 24, 92), "undertone": "cool", "depth": "deep"},
            {"name": "Dusty Mauve", "bgr": (128, 128, 192), "undertone": "neutral", "depth": "medium"},
        ],
    },
    "Browns": {
        "Soft Browns": [
            {"name": "Cinnamon Brown", "bgr": (82, 100, 156), "undertone": "warm", "depth": "medium"},
            {"name": "Maple Brown", "bgr": (76, 94, 146), "undertone": "warm", "depth": "medium"},
            {"name": "Hazelnut Brown", "bgr": (88, 106, 162), "undertone": "neutral", "depth": "fair"},
        ],
        "Deep Browns": [
            {"name": "Espresso Brown", "bgr": (50, 60, 98), "undertone": "neutral", "depth": "deep"},
            {"name": "Mahogany Brown", "bgr": (52, 54, 118), "undertone": "warm", "depth": "deep"},
            {"name": "Walnut Brown", "bgr": (58, 72, 116), "undertone": "neutral", "depth": "deep"},
        ],
    },
    "Purples": {
        "Lavenders": [
            {"name": "Lavender Rose", "bgr": (154, 132, 208), "undertone": "cool", "depth": "fair"},
            {"name": "Lilac Bloom", "bgr": (146, 118, 214), "undertone": "cool", "depth": "medium"},
            {"name": "Orchid Mist", "bgr": (136, 102, 196), "undertone": "cool", "depth": "medium"},
        ],
        "Violets": [
            {"name": "Violet Kiss", "bgr": (112, 58, 170), "undertone": "cool", "depth": "deep"},
            {"name": "Royal Orchid", "bgr": (104, 44, 156), "undertone": "cool", "depth": "deep"},
            {"name": "Purple Berry", "bgr": (94, 52, 144), "undertone": "neutral", "depth": "deep"},
        ],
    },
}


def flatten_shades():
    shades = []
    for family, subcategories in SHADE_LIBRARY.items():
        for subcategory, entries in subcategories.items():
            for entry in entries:
                shades.append({"family": family, "subcategory": subcategory, **entry})
    return shades


ALL_SHADES = flatten_shades()
SHADE_BY_NAME = {shade["name"]: shade for shade in ALL_SHADES}


def bgr_to_hex(bgr):
    return "#{:02X}{:02X}{:02X}".format(int(bgr[2]), int(bgr[1]), int(bgr[0]))


def hex_to_bgr(value: str):
    value = value.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (b, g, r)


def get_shade_catalog():
    catalog = {}
    for family, subcategories in SHADE_LIBRARY.items():
        catalog[family] = {}
        for subcategory, entries in subcategories.items():
            catalog[family][subcategory] = [
                {
                    "name": entry["name"],
                    "hex": bgr_to_hex(entry["bgr"]),
                    "undertone": entry["undertone"],
                    "depth": entry["depth"],
                }
                for entry in entries
            ]
    return catalog


@lru_cache(maxsize=1)
def load_landmarker():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def image_bytes_to_bgr(image_bytes: bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def bgr_to_base64_png(image_bgr):
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb)
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def get_pts(landmarks, indices, width, height):
    return np.array(
        [[int(landmarks[i].x * width), int(landmarks[i].y * height)] for i in indices],
        dtype=np.int32,
    )


def prepare_detection_frame(image_bgr):
    height, width = image_bgr.shape[:2]
    longest_edge = max(height, width)
    if longest_edge <= MAX_DETECT_SIZE:
        return image_bgr
    scale = MAX_DETECT_SIZE / float(longest_edge)
    return cv2.resize(
        image_bgr,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def detect_landmarks(image_bgr):
    resized = prepare_detection_frame(image_bgr)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    )
    result = load_landmarker().detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def build_lip_geometry(landmarks, width, height):
    return {
        "outer": get_pts(landmarks, OUTER_LIP, width, height),
        "inner": get_pts(landmarks, INNER_LIP, width, height),
        "upper_polygon": get_pts(landmarks, UPPER_OUTER_ARC + UPPER_INNER_ARC, width, height),
        "lower_polygon": get_pts(landmarks, LOWER_OUTER_ARC + LOWER_INNER_ARC, width, height),
    }


def geometry_to_masks(geometry, width, height):
    full_mask = np.zeros((height, width), dtype=np.uint8)
    upper_mask = np.zeros((height, width), dtype=np.uint8)
    lower_mask = np.zeros((height, width), dtype=np.uint8)

    cv2.fillPoly(full_mask, [geometry["outer"].astype(np.int32)], 255)
    cv2.fillPoly(full_mask, [geometry["inner"].astype(np.int32)], 0)
    cv2.fillPoly(upper_mask, [geometry["upper_polygon"].astype(np.int32)], 255)
    cv2.fillPoly(lower_mask, [geometry["lower_polygon"].astype(np.int32)], 255)

    upper_mask = cv2.bitwise_and(upper_mask, full_mask)
    lower_mask = cv2.bitwise_and(lower_mask, full_mask)
    return full_mask, upper_mask, lower_mask


def sample_skin_tone(image_bgr, lip_mask):
    kernel_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    kernel_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    outer_ring = cv2.dilate(lip_mask, kernel_outer)
    inner_ring = cv2.dilate(lip_mask, kernel_inner)
    sample_mask = cv2.subtract(outer_ring, inner_ring)

    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    _, sat, val = cv2.split(hsv)
    valid = (sample_mask > 0) & (sat > 12) & (sat < 175) & (val > 45)
    if np.count_nonzero(valid) < 120:
        valid = sample_mask > 0
    if np.count_nonzero(valid) < 120:
        return None
    return image_bgr[valid].mean(axis=0)


def classify_tone_profile(skin_bgr):
    if skin_bgr is None:
        return {"undertone": "neutral", "depth": "medium", "skin_hex": None}

    skin_lab = cv2.cvtColor(
        np.uint8([[skin_bgr.astype(np.uint8).tolist()]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)

    warmth_score = skin_lab[2] - skin_lab[1]
    if warmth_score > 8:
        undertone = "warm"
    elif warmth_score < -6:
        undertone = "cool"
    else:
        undertone = "neutral"

    if skin_lab[0] > 175:
        depth = "fair"
    elif skin_lab[0] > 125:
        depth = "medium"
    else:
        depth = "deep"

    return {
        "undertone": undertone,
        "depth": depth,
        "skin_hex": bgr_to_hex(skin_bgr),
    }


def adapt_colour_to_skin(colour_bgr, skin_bgr):
    if skin_bgr is None:
        return colour_bgr

    colour_lab = cv2.cvtColor(
        np.uint8([[list(colour_bgr)]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)
    skin_lab = cv2.cvtColor(
        np.uint8([[skin_bgr.astype(np.uint8).tolist()]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)

    tone_lightness = np.clip(skin_lab[0] * 0.78 + 28, 55, 205)
    colour_lab[0] = np.clip(colour_lab[0] * 0.52 + tone_lightness * 0.48, 35, 225)
    colour_lab[1] = np.clip(colour_lab[1] * 0.84 + skin_lab[1] * 0.16, 116, 190)
    colour_lab[2] = np.clip(colour_lab[2] * 0.88 + skin_lab[2] * 0.12, 108, 188)

    adjusted = cv2.cvtColor(
        np.uint8([[colour_lab.astype(np.uint8)]]), cv2.COLOR_LAB2BGR
    )[0, 0]
    return tuple(int(channel) for channel in adjusted)


def recommend_shades(tone_profile, family="Pinks"):
    scored = []
    for shade in ALL_SHADES:
        score = 0
        if shade["family"] == family:
            score += 4
        if shade["undertone"] == tone_profile["undertone"]:
            score += 3
        elif shade["undertone"] == "neutral":
            score += 1
        if shade["depth"] == tone_profile["depth"]:
            score += 2
        elif tone_profile["depth"] == "medium" and shade["depth"] in {"fair", "deep"}:
            score += 1
        scored.append((score, shade))

    scored.sort(key=lambda item: (-item[0], item[1]["name"]))
    return [
        {
            "family": shade["family"],
            "subcategory": shade["subcategory"],
            "name": shade["name"],
            "hex": bgr_to_hex(shade["bgr"]),
            "undertone": shade["undertone"],
            "depth": shade["depth"],
        }
        for _, shade in scored[:4]
    ]


def create_highlight_map(full_mask, lower_mask, upper_mask, finish):
    if finish == "Matte":
        return np.zeros_like(full_mask, dtype=np.float32)

    height, width = full_mask.shape
    gradient = np.tile(
        np.linspace(0.15, 1.0, height, dtype=np.float32).reshape(-1, 1),
        (1, width),
    )
    lower_gloss = (lower_mask.astype(np.float32) / 255.0) * gradient
    upper_gloss = (upper_mask.astype(np.float32) / 255.0) * 0.35
    gloss = lower_gloss * 0.12 + upper_gloss * 0.04
    gloss *= 1.7 if finish == "Gloss" else 0.9
    return cv2.GaussianBlur(gloss, (11, 11), 0)


def apply_lab_tint(image_bgr, full_mask, upper_mask, lower_mask, tuned_colour_bgr, opacity, finish):
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    colour_lab = cv2.cvtColor(
        np.uint8([[list(tuned_colour_bgr)]]), cv2.COLOR_BGR2LAB
    )[0, 0].astype(np.float32)

    full_alpha = cv2.GaussianBlur(full_mask.astype(np.float32) / 255.0, (9, 9), 0)
    upper_alpha = cv2.GaussianBlur(upper_mask.astype(np.float32) / 255.0, (7, 7), 0) * opacity * 0.9
    lower_alpha = cv2.GaussianBlur(lower_mask.astype(np.float32) / 255.0, (7, 7), 0) * min(opacity * 1.05, 1.0)

    lip_pixels = full_mask > 0
    lip_mean_l = float(image_lab[:, :, 0][lip_pixels].mean()) if np.any(lip_pixels) else float(colour_lab[0])
    texture = image_lab[:, :, 0] - lip_mean_l

    target_l = np.full_like(image_lab[:, :, 0], colour_lab[0], dtype=np.float32)
    target_l += lower_alpha * 10.0
    target_l -= upper_alpha * 6.0

    image_lab[:, :, 0] = np.where(
        full_alpha > 0,
        np.clip(
            image_lab[:, :, 0] * (1.0 - full_alpha * 0.22)
            + target_l * (full_alpha * 0.22)
            + texture * 0.38 * full_alpha,
            0,
            255,
        ),
        image_lab[:, :, 0],
    )
    image_lab[:, :, 1] = np.clip(
        image_lab[:, :, 1] * (1.0 - full_alpha * opacity * 0.95)
        + colour_lab[1] * (full_alpha * opacity * 0.95),
        0,
        255,
    )
    image_lab[:, :, 2] = np.clip(
        image_lab[:, :, 2] * (1.0 - full_alpha * opacity * 0.98)
        + colour_lab[2] * (full_alpha * opacity * 0.98),
        0,
        255,
    )

    output_bgr = cv2.cvtColor(image_lab.astype(np.uint8), cv2.COLOR_LAB2BGR).astype(np.float32)
    highlight = create_highlight_map(full_mask, lower_mask, upper_mask, finish)
    output_bgr = np.clip(output_bgr + np.dstack([highlight * 255.0] * 3), 0, 255)
    return output_bgr.astype(np.uint8)


def resolve_colour(shade_name: str | None, custom_hex: str | None):
    if custom_hex:
        return hex_to_bgr(custom_hex), "Custom"
    if shade_name and shade_name in SHADE_BY_NAME:
        shade = SHADE_BY_NAME[shade_name]
        return shade["bgr"], shade["name"]
    default_shade = SHADE_BY_NAME["Blush Pink"]
    return default_shade["bgr"], default_shade["name"]


def process_try_on(image_bytes: bytes, shade_name: str | None, custom_hex: str | None, opacity: float, finish: str):
    image_bgr = image_bytes_to_bgr(image_bytes)
    landmarks = detect_landmarks(image_bgr)
    if landmarks is None:
        return {
            "detected": False,
            "message": "No face detected. Use a front-facing image with visible lips.",
        }

    geometry = build_lip_geometry(landmarks, image_bgr.shape[1], image_bgr.shape[0])
    full_mask, upper_mask, lower_mask = geometry_to_masks(geometry, image_bgr.shape[1], image_bgr.shape[0])
    skin_bgr = sample_skin_tone(image_bgr, full_mask)
    tone_profile = classify_tone_profile(skin_bgr)

    selected_colour, _ = resolve_colour(shade_name, custom_hex)
    tuned_colour = adapt_colour_to_skin(selected_colour, skin_bgr)
    rendered = apply_lab_tint(
        image_bgr,
        full_mask,
        upper_mask,
        lower_mask,
        tuned_colour,
        opacity,
        finish,
    )

    lip_width = max(int(geometry["outer"][:, 0].max() - geometry["outer"][:, 0].min()), 1)
    edge_blur = max(5, (lip_width // 14) * 2 + 1)
    feather = cv2.GaussianBlur(full_mask.astype(np.float32) / 255.0, (edge_blur, edge_blur), 0)
    feather3 = np.dstack([feather] * 3)
    output = (feather3 * rendered + (1.0 - feather3) * image_bgr).astype(np.uint8)

    return {
        "detected": True,
        "image_base64": bgr_to_base64_png(output),
        "tuned_hex": bgr_to_hex(tuned_colour),
        "tone_profile": tone_profile,
        "recommendations": recommend_shades(tone_profile, "Pinks"),
    }
