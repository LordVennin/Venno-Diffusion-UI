"""Object detection for the detailer: YOLO (ultralytics, boxes or segmentation masks) and mediapipe faces."""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import numpy as np
from PIL import Image, ImageDraw

EXTENSIONS = (".pt", ".tflite", ".task")
_lock = threading.Lock()
_yolo: dict = {}  # path → YOLO model (kept: small, and reloading is slow)


@dataclass
class Detection:
    box: tuple[int, int, int, int]  # x1, y1, x2, y2
    mask: Image.Image  # "L", image size, white = object
    score: float


def folder() -> str:
    from modules import paths  # Phase 4: Venno settings

    return os.path.join(paths.models_path, "adetailer")


def models() -> dict[str, str]:
    root = folder()
    try:
        return {f: os.path.join(root, f) for f in sorted(os.listdir(root)) if f.lower().endswith(EXTENSIONS) and f != "face_landmarker.task"} | (
            {"mediapipe_face_mesh": os.path.join(root, "face_landmarker.task")} if os.path.isfile(os.path.join(root, "face_landmarker.task")) else {}
        )
    except OSError:
        return {}


def detect(name: str, image: Image.Image, confidence: float = 0.3) -> list[Detection]:
    path = models().get(name)
    if path is None:
        raise ValueError(f'Detection model "{name}" not found in {folder()}')
    image = image.convert("RGB")
    if path.endswith(".pt"):
        found = _yolo_detect(path, image, confidence)
    elif path.endswith(".task"):
        found = _mediapipe_mesh(path, image, confidence)
    else:
        found = _mediapipe_faces(path, image, confidence)
    return sorted(found, key=lambda d: (d.box[2] - d.box[0]) * (d.box[3] - d.box[1]), reverse=True)


def _box_mask(box, size) -> Image.Image:
    m = Image.new("L", size, 0)
    ImageDraw.Draw(m).rectangle([round(v) for v in box], fill=255)
    return m


@contextmanager
def _no_ultralytics_dirs():
    """ultralytics creates run folders next to the working directory on predict; nothing is saved, so skip that."""
    makedirs, mkdir = os.makedirs, os.mkdir
    os.makedirs = lambda *a, **k: None
    os.mkdir = lambda *a, **k: None
    try:
        yield
    finally:
        os.makedirs, os.mkdir = makedirs, mkdir


def _yolo_detect(path: str, image: Image.Image, confidence: float) -> list[Detection]:
    from ultralytics import YOLO

    with _lock:
        if path not in _yolo:
            _yolo[path] = YOLO(path)
        model = _yolo[path]
        # bigger input for big images: small faces in full-body shots are missed at 640
        size = 1024 if max(image.size) >= 1024 else 640
        with _no_ultralytics_dirs():
            pred = model(image, conf=confidence, imgsz=size, verbose=False)[0]
    boxes = pred.boxes.xyxy.cpu().numpy().tolist()
    scores = pred.boxes.conf.cpu().numpy().tolist()
    masks = []
    if pred.masks is not None:
        for m in pred.masks.data.float().cpu().numpy():
            masks.append(Image.fromarray((m * 255).astype(np.uint8)).resize(image.size, Image.BILINEAR))
    else:
        masks = [_box_mask(b, image.size) for b in boxes]
    return [Detection(tuple(round(v) for v in b), m, s) for b, m, s in zip(boxes, masks, scores)]


def _mediapipe_faces(path: str, image: Image.Image, confidence: float) -> list[Detection]:
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    options = vision.FaceDetectorOptions(base_options=python.BaseOptions(model_asset_path=path), min_detection_confidence=confidence)
    with vision.FaceDetector.create_from_options(options) as detector:
        pred = detector.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
    out = []
    for d in pred.detections or []:
        b = d.bounding_box
        box = (b.origin_x, b.origin_y, b.origin_x + b.width, b.origin_y + b.height)
        out.append(Detection(box, _box_mask(box, image.size), d.categories[0].score))
    return out


def _mediapipe_mesh(path: str, image: Image.Image, confidence: float) -> list[Detection]:
    import cv2
    import mediapipe as mp
    from mediapipe.tasks import python
    from mediapipe.tasks.python import vision

    w, h = image.size
    options = vision.FaceLandmarkerOptions(base_options=python.BaseOptions(model_asset_path=path), num_faces=20, min_face_detection_confidence=confidence)
    with vision.FaceLandmarker.create_from_options(options) as landmarker:
        pred = landmarker.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(image)))
    out = []
    for landmarks in pred.face_landmarks or []:
        pts = np.array([[int(p.x * w), int(p.y * h)] for p in landmarks])
        mask = Image.new("L", image.size, 0)
        ImageDraw.Draw(mask).polygon(cv2.convexHull(pts).reshape(-1).tolist(), fill=255)
        out.append(Detection(mask.getbbox(), mask, 1.0))
    return out
