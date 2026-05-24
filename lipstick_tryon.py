import io
import os
import time
import urllib.request

import av
import cv2
import mediapipe as mp
import numpy as np
import streamlit as st
from PIL import Image
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    FaceLandmarker,
    FaceLandmarkerOptions,
    RunningMode,
)

try:
    from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer
except ImportError:
    VideoProcessorBase = object
    WebRtcMode = None
    webrtc_streamer = None


st.set_page_config(
    page_title="Lipstick Try-On",
    page_icon="💄",
    layout="wide",
)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/1/face_landmarker.task"
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
            {"name": "💄 Ruby Red", "bgr": (35, 50, 192), "undertone": "neutral", "depth": "medium"},
            {"name": "🍒 Classic Red", "bgr": (32, 32, 255), "undertone": "neutral", "depth": "medium"},
            {"name": "🔥 Scarlet Pop", "bgr": (40, 58, 238), "undertone": "warm", "depth": "medium"},
            {"name": "❤️ Crimson Velvet", "bgr": (28, 28, 180), "undertone": "cool", "depth": "deep"},
            {"name": "🥀 Brick Rose", "bgr": (48, 70, 156), "undertone": "warm", "depth": "deep"},
        ],
    },
    "Pinks": {
        "Soft Pinks": [
            {"name": "🎀 Baby Pink", "bgr": (182, 175, 244), "undertone": "cool", "depth": "fair"},
            {"name": "🩰 Ballet Pink", "bgr": (167, 156, 221), "undertone": "neutral", "depth": "fair"},
            {"name": "🌷 Blush Pink", "bgr": (156, 142, 214), "undertone": "neutral", "depth": "medium"},
            {"name": "🌹 Rosy Nude Pink", "bgr": (138, 126, 196), "undertone": "warm", "depth": "medium"},
        ],
        "Cool Pinks": [
            {"name": "🌸 Rose Pink", "bgr": (138, 116, 232), "undertone": "cool", "depth": "medium"},
            {"name": "🪷 Lotus Pink", "bgr": (132, 94, 210), "undertone": "cool", "depth": "medium"},
            {"name": "💞 Bubblegum Pink", "bgr": (164, 90, 248), "undertone": "cool", "depth": "fair"},
            {"name": "🌺 Fuchsia Bloom", "bgr": (126, 34, 214), "undertone": "cool", "depth": "deep"},
        ],
        "Warm Pinks": [
            {"name": "🌼 Peach Pink", "bgr": (112, 138, 228), "undertone": "warm", "depth": "fair"},
            {"name": "🍓 Strawberry Pink", "bgr": (120, 86, 224), "undertone": "warm", "depth": "medium"},
            {"name": "🌺 Hot Pink", "bgr": (146, 30, 233), "undertone": "warm", "depth": "medium"},
            {"name": "💗 Candy Pink", "bgr": (154, 78, 240), "undertone": "warm", "depth": "medium"},
        ],
        "Nude Pinks": [
            {"name": "🌸 Petal Nude", "bgr": (150, 144, 205), "undertone": "neutral", "depth": "fair"},
            {"name": "🪞Muted Rose", "bgr": (126, 118, 184), "undertone": "neutral", "depth": "medium"},
            {"name": "🌷 Dusty Pink Nude", "bgr": (136, 124, 176), "undertone": "warm", "depth": "deep"},
            {"name": "🥀 Rosewood Pink", "bgr": (98, 92, 152), "undertone": "neutral", "depth": "deep"},
        ],
        "Bright Pinks": [
            {"name": "💓 Neon Pink", "bgr": (158, 40, 245), "undertone": "cool", "depth": "medium"},
            {"name": "🩷 Electric Pink", "bgr": (170, 52, 255), "undertone": "cool", "depth": "medium"},
            {"name": "🌺 Pop Fuchsia", "bgr": (132, 20, 226), "undertone": "cool", "depth": "deep"},
            {"name": "🎉 Party Pink", "bgr": (168, 64, 244), "undertone": "warm", "depth": "medium"},
        ],
    },
    "Corals": {
        "Soft Corals": [
            {"name": "🍑 Coral Crush", "bgr": (60, 96, 232), "undertone": "warm", "depth": "medium"},
            {"name": "🌅 Sunset Coral", "bgr": (70, 112, 240), "undertone": "warm", "depth": "fair"},
            {"name": "🧡 Peach Coral", "bgr": (86, 132, 228), "undertone": "warm", "depth": "fair"},
            {"name": "🪸 Tropical Coral", "bgr": (78, 102, 220), "undertone": "warm", "depth": "medium"},
        ],
    },
    "Nudes": {
        "Warm Nudes": [
            {"name": "🏖️ Nude Beige", "bgr": (110, 144, 196), "undertone": "warm", "depth": "fair"},
            {"name": "🤎 Caramel Nude", "bgr": (90, 116, 168), "undertone": "warm", "depth": "medium"},
            {"name": "🧋 Latte Nude", "bgr": (104, 128, 176), "undertone": "neutral", "depth": "medium"},
            {"name": "🥥 Sand Beige", "bgr": (118, 152, 194), "undertone": "warm", "depth": "fair"},
        ],
        "Deep Nudes": [
            {"name": "☕ Mocha Brown", "bgr": (82, 94, 139), "undertone": "warm", "depth": "deep"},
            {"name": "🍫 Cocoa Nude", "bgr": (68, 82, 126), "undertone": "neutral", "depth": "deep"},
            {"name": "🟤 Chestnut Nude", "bgr": (74, 92, 148), "undertone": "warm", "depth": "deep"},
        ],
    },
    "Berry & Plum": {
        "Berry": [
            {"name": "🍷 Berry Wine", "bgr": (104, 46, 125), "undertone": "cool", "depth": "deep"},
            {"name": "🍇 Mulberry Kiss", "bgr": (86, 44, 112), "undertone": "cool", "depth": "deep"},
            {"name": "🌌 Blackberry", "bgr": (58, 20, 82), "undertone": "cool", "depth": "deep"},
        ],
        "Plums": [
            {"name": "🫐 Plum Dark", "bgr": (64, 16, 74), "undertone": "cool", "depth": "deep"},
            {"name": "🫐 Deep Plum", "bgr": (74, 24, 92), "undertone": "cool", "depth": "deep"},
            {"name": "🌷 Dusty Mauve", "bgr": (128, 128, 192), "undertone": "neutral", "depth": "medium"},
        ],
    },
}

DEFAULT_FAMILY = "Pinks"
DEFAULT_SUBCATEGORY = "Soft Pinks"
DEFAULT_SHADE_NAME = "🌷 Blush Pink"


def flatten_shades():
    shades = []
    for family, subcategories in SHADE_LIBRARY.items():
        for subcategory, entries in subcategories.items():
            for entry in entries:
                shades.append(
                    {
                        "family": family,
                        "subcategory": subcategory,
                        **entry,
                    }
                )
    return shades


ALL_SHADES = flatten_shades()
SHADE_BY_NAME = {shade["name"]: shade for shade in ALL_SHADES}


@st.cache_resource(show_spinner=False)
def download_model():
    if not os.path.exists(MODEL_PATH):
        with st.spinner("Downloading face landmark model..."):
            urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return MODEL_PATH


def hex_to_bgr(value: str):
    value = value.lstrip("#")
    r, g, b = int(value[0:2], 16), int(value[2:4], 16), int(value[4:6], 16)
    return (b, g, r)


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


@st.cache_resource(show_spinner=False)
def load_landmarker_image(model_path):
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.IMAGE,
        num_faces=1,
        min_face_detection_confidence=0.5,
        min_face_presence_confidence=0.5,
        min_tracking_confidence=0.5,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def create_video_landmarker():
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=download_model()),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.45,
        min_face_presence_confidence=0.45,
        min_tracking_confidence=0.4,
        output_face_blendshapes=False,
        output_facial_transformation_matrixes=False,
    )
    return FaceLandmarker.create_from_options(options)


def detect_landmarks_image(image_bgr):
    resized = prepare_detection_frame(image_bgr)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    )
    result = load_landmarker_image(download_model()).detect(mp_image)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def detect_landmarks_video(detector, image_bgr, timestamp_ms):
    resized = prepare_detection_frame(image_bgr)
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=cv2.cvtColor(resized, cv2.COLOR_BGR2RGB),
    )
    result = detector.detect_for_video(mp_image, timestamp_ms)
    if not result.face_landmarks:
        return None
    return result.face_landmarks[0]


def build_lip_geometry(landmarks, width, height):
    geometry = {
        "outer": get_pts(landmarks, OUTER_LIP, width, height),
        "inner": get_pts(landmarks, INNER_LIP, width, height),
        "upper_polygon": get_pts(landmarks, UPPER_OUTER_ARC + UPPER_INNER_ARC, width, height),
        "lower_polygon": get_pts(landmarks, LOWER_OUTER_ARC + LOWER_INNER_ARC, width, height),
    }
    return geometry


def smooth_points(current_pts, previous_pts, motion_px):
    if previous_pts is None or previous_pts.shape != current_pts.shape:
        return current_pts.astype(np.float32)
    if motion_px < 2:
        alpha = 0.75
    elif motion_px < 8:
        alpha = 0.58
    else:
        alpha = 0.35
    return previous_pts * alpha + current_pts.astype(np.float32) * (1.0 - alpha)


def smooth_geometry(current_geometry, previous_geometry):
    if previous_geometry is None:
        return {key: value.astype(np.float32) for key, value in current_geometry.items()}

    current_center = current_geometry["outer"].mean(axis=0)
    previous_center = previous_geometry["outer"].mean(axis=0)
    motion_px = float(np.linalg.norm(current_center - previous_center))

    smoothed = {}
    for key in current_geometry:
        smoothed[key] = smooth_points(current_geometry[key], previous_geometry.get(key), motion_px)
    return smoothed


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
        return {
            "undertone": "neutral",
            "depth": "medium",
            "skin_hex": None,
        }

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

    skin_hex = "#{:02X}{:02X}{:02X}".format(
        int(skin_bgr[2]), int(skin_bgr[1]), int(skin_bgr[0])
    )
    return {
        "undertone": undertone,
        "depth": depth,
        "skin_hex": skin_hex,
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


def recommend_shades(tone_profile, family):
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
        elif (
            tone_profile["depth"] == "medium"
            and shade["depth"] in {"fair", "deep"}
        ):
            score += 1
        scored.append((score, shade))

    scored.sort(key=lambda item: (-item[0], item[1]["name"]))
    return [shade for _, shade in scored[:4]]


def create_highlight_map(full_mask, lower_mask, upper_mask, finish):
    if finish == "Matte":
        return np.zeros_like(full_mask, dtype=np.float32)

    height, width = full_mask.shape
    gradient = np.tile(np.linspace(0.15, 1.0, height, dtype=np.float32).reshape(-1, 1), (1, width))
    lower_gloss = (lower_mask.astype(np.float32) / 255.0) * gradient
    upper_gloss = (upper_mask.astype(np.float32) / 255.0) * 0.35
    gloss = lower_gloss * 0.12 + upper_gloss * 0.04

    if finish == "Gloss":
        gloss *= 1.7
    else:
        gloss *= 0.9

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


def render_lipstick(image_bgr, geometry, colour_bgr, opacity, finish):
    height, width = image_bgr.shape[:2]
    full_mask, upper_mask, lower_mask = geometry_to_masks(geometry, width, height)
    skin_bgr = sample_skin_tone(image_bgr, full_mask)
    tone_profile = classify_tone_profile(skin_bgr)
    tuned_colour = adapt_colour_to_skin(colour_bgr, skin_bgr)

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

    analysis = {
        "tuned_colour": tuned_colour,
        "tone_profile": tone_profile,
        "recommendations": recommend_shades(tone_profile, "Pinks"),
    }
    return output, analysis


def apply_lipstick_image(image_bgr, colour_bgr, opacity, finish):
    landmarks = detect_landmarks_image(image_bgr)
    if landmarks is None:
        return image_bgr.copy(), False, None
    geometry = build_lip_geometry(landmarks, image_bgr.shape[1], image_bgr.shape[0])
    result, analysis = render_lipstick(image_bgr, geometry, colour_bgr, opacity, finish)
    return result, True, analysis


class LipstickVideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.detector = create_video_landmarker()
        self.previous_geometry = None
        self.colour_bgr = SHADE_BY_NAME[DEFAULT_SHADE_NAME]["bgr"]
        self.opacity = 0.7
        self.finish = "Matte"

    def recv(self, frame):
        image_bgr = frame.to_ndarray(format="bgr24")
        timestamp_ms = int(time.time() * 1000)
        landmarks = detect_landmarks_video(self.detector, image_bgr, timestamp_ms)
        if landmarks is not None:
            geometry = build_lip_geometry(landmarks, image_bgr.shape[1], image_bgr.shape[0])
            geometry = smooth_geometry(geometry, self.previous_geometry)
            self.previous_geometry = geometry
            image_bgr, _ = render_lipstick(
                image_bgr,
                geometry,
                self.colour_bgr,
                self.opacity,
                self.finish,
            )
        else:
            self.previous_geometry = None
        return av.VideoFrame.from_ndarray(image_bgr, format="bgr24")


with st.sidebar:
    st.header("Customise")

    selected_family = st.selectbox(
        "Shade family",
        list(SHADE_LIBRARY.keys()),
        index=list(SHADE_LIBRARY.keys()).index(DEFAULT_FAMILY),
    )
    selected_subcategory = st.selectbox(
        "Subcategory",
        list(SHADE_LIBRARY[selected_family].keys()),
        index=list(SHADE_LIBRARY[selected_family].keys()).index(
            DEFAULT_SUBCATEGORY if selected_family == DEFAULT_FAMILY else list(SHADE_LIBRARY[selected_family].keys())[0]
        ),
    )

    shade_names = [entry["name"] for entry in SHADE_LIBRARY[selected_family][selected_subcategory]]
    default_index = shade_names.index(DEFAULT_SHADE_NAME) if DEFAULT_SHADE_NAME in shade_names else 0
    selected_name = st.radio("Lipstick shade", shade_names, index=default_index)
    colour_bgr = SHADE_BY_NAME[selected_name]["bgr"]

    if st.checkbox("Pick a custom colour"):
        custom_hex = st.color_picker("Choose colour", "#C96A78")
        colour_bgr = hex_to_bgr(custom_hex)
        selected_name = "Custom"

    opacity = st.slider("Opacity / Coverage", 0.1, 1.0, 0.72, 0.05)
    finish = st.radio("Finish", ["Matte", "Satin", "Gloss"])


st.title("Virtual Lipstick Try-On")
st.caption(
    "MediaPipe lip tracking with LAB blending, upper/lower lip shading, tone-aware recommendations, and smoother live preview."
)


def show_recommendations(analysis):
    if analysis is None:
        return

    tone_profile = analysis["tone_profile"]
    undertone = tone_profile["undertone"].title()
    depth = tone_profile["depth"].title()
    st.markdown(f"**Detected tone:** `{undertone}` undertone · `{depth}` depth")
    if tone_profile["skin_hex"]:
        st.markdown(f"**Sampled skin tone:** `{tone_profile['skin_hex']}`")

    recommendations = analysis["recommendations"]
    if recommendations:
        recommendation_text = " · ".join(
            f"{shade['name']} ({shade['subcategory']})" for shade in recommendations
        )
        st.markdown(f"**Recommended pinks:** {recommendation_text}")


def show_result(pil_image):
    image_bgr = cv2.cvtColor(np.array(pil_image.convert("RGB")), cv2.COLOR_RGB2BGR)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Original")
        st.image(pil_image, use_container_width=True)

    with st.spinner("Applying lipstick..."):
        result_bgr, detected, analysis = apply_lipstick_image(
            image_bgr, colour_bgr, opacity, finish
        )

    result_rgb = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
    with col2:
        st.subheader(f"With {selected_name}")
        st.image(result_rgb, use_container_width=True)

    if not detected:
        st.warning("No face detected. Use a front-facing image with visible lips.")
        return

    tuned_colour = analysis["tuned_colour"]
    tuned_hex = "#{:02X}{:02X}{:02X}".format(
        tuned_colour[2], tuned_colour[1], tuned_colour[0]
    )
    st.caption(f"Tone-matched shade used: `{tuned_hex}`")
    show_recommendations(analysis)

    buf = io.BytesIO()
    Image.fromarray(result_rgb).save(buf, format="PNG")
    st.download_button(
        "Download result",
        buf.getvalue(),
        "lipstick_tryon.png",
        "image/png",
    )


tab_live, tab_upload, tab_camera = st.tabs(
    ["Live Camera", "Upload Photo", "Take Photo"]
)

with tab_live:
    st.subheader("Real-time preview")
    st.caption("The live preview smooths lip landmarks across frames to reduce flicker and edge jitter.")
    if webrtc_streamer is None:
        st.info("Install `streamlit-webrtc` to enable the live webcam preview.")
    else:
        ctx = webrtc_streamer(
            key="lipstick-live",
            mode=WebRtcMode.SENDRECV,
            media_stream_constraints={"video": True, "audio": False},
            video_processor_factory=LipstickVideoProcessor,
            async_processing=True,
        )
        if ctx.video_processor:
            ctx.video_processor.colour_bgr = colour_bgr
            ctx.video_processor.opacity = opacity
            ctx.video_processor.finish = finish

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload a front-facing photo",
        type=["jpg", "jpeg", "png", "webp"],
    )
    if uploaded_file:
        show_result(Image.open(uploaded_file))
    else:
        st.info("Upload a photo to test the improved lipstick rendering.")

with tab_camera:
    camera_photo = st.camera_input("Take a selfie")
    if camera_photo:
        show_result(Image.open(camera_photo))
    else:
        st.info("Take a single photo here if you do not need the live preview.")


st.markdown("---")
st.markdown(
    "Quality upgrades in this build: smoother upper-lip coverage, LAB-space tinting to preserve lip texture, "
    "separate upper/lower lip shading, tone-based pink recommendations, and temporal smoothing for webcam preview."
)
