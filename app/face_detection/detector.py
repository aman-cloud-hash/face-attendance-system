"""
Face Detector Module using InsightFace (RetinaFace backbone).

Key AI/CV Capabilities:
- Single & Multi-face detection in real-time video streams.
- RetinaFace Feature Pyramid Network (FPN) architecture for scale invariance.
- Keypoint regression (5 canonical facial landmarks: eyes, nose, mouth corners).
- Optimized with (320, 320) detection input resolution for high FPS.
- Model initialized ONCE with global singleton caching, GPU acceleration, and automatic CPU fallback.
"""

import warnings
from typing import Optional, List
import numpy as np
import cv2

# Suppress scikit-image / insightface future warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import insightface
from insightface.app import FaceAnalysis
import app.config as config

# Global singleton model cache to prevent redundant re-initializations
_GLOBAL_INSIGHTFACE_APP: Optional[FaceAnalysis] = None


class FaceDetector:
    """
    High-performance Face Detector wrapping InsightFace's RetinaFace model.
    Handles model initialization, inference, bounding box extraction, and visualization.
    """

    def __init__(self, name: str = config.INSIGHTFACE_MODEL_NAME, 
                 det_size: tuple = config.DETECTION_SIZE,
                 det_thresh: float = config.DETECTION_THRESHOLD):
        """
        Initialize the RetinaFace detector with singleton caching and optimized detection resolution.
        """
        self.name = name
        self.det_size = det_size
        self.det_thresh = det_thresh
        self.app = self._get_or_create_model()

    def _get_or_create_model(self) -> FaceAnalysis:
        """
        Retrieves existing global model instance or loads it once.
        """
        global _GLOBAL_INSIGHTFACE_APP
        if _GLOBAL_INSIGHTFACE_APP is not None:
            return _GLOBAL_INSIGHTFACE_APP

        print(f"[FaceDetector] Loading face recognition model '{self.name}' (Detection Size: {self.det_size})...")
        
        providers_to_try = [
            ['CUDAExecutionProvider', 'CPUExecutionProvider'],
            ['CPUExecutionProvider']
        ]

        last_error = None
        for providers in providers_to_try:
            try:
                app = FaceAnalysis(
                    name=self.name,
                    providers=providers,
                    allowed_modules=['detection', 'recognition', 'landmark_2d_106']
                )
                app.prepare(ctx_id=0, det_size=self.det_size, det_thresh=self.det_thresh)
                _GLOBAL_INSIGHTFACE_APP = app
                print(f"[FaceDetector] Model initialized successfully with detection size {self.det_size}.")
                return app
            except Exception as e:
                last_error = e

        raise RuntimeError(f"[FaceDetector] Critical: Failed to initialize InsightFace model: {last_error}")

    def detect_faces(self, frame: np.ndarray) -> list:
        """
        Run RetinaFace inference on a BGR video frame.
        """
        if frame is None or frame.size == 0:
            return []

        faces = self.app.get(frame)
        return faces

    def draw_faces(self, frame: np.ndarray, faces: list, 
                   draw_landmarks: bool = True, 
                   color: tuple = config.COLOR_RECOGNIZED) -> np.ndarray:
        """
        Draw clean bounding boxes, confidence badges, and landmarks on frame.
        """
        output_frame = frame.copy()

        for face in faces:
            bbox = face.bbox.astype(int)
            x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
            
            h, w = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)

            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

            score = float(face.det_score) if hasattr(face, 'det_score') else 0.0
            label = f"Face: {score:.2f}"
            
            (lw, lh), _ = cv2.getTextSize(label, config.FONT, 0.5, 1)
            cv2.rectangle(output_frame, (x1, max(0, y1 - lh - 8)), (x1 + lw + 6, y1), color, -1)
            cv2.putText(output_frame, label, (x1 + 3, max(12, y1 - 4)), 
                        config.FONT, 0.5, config.COLOR_WHITE, 1, cv2.LINE_AA)

            if draw_landmarks and hasattr(face, 'kps') and face.kps is not None:
                for pt in face.kps:
                    px, py = int(pt[0]), int(pt[1])
                    cv2.circle(output_frame, (px, py), 2, (0, 255, 255), -1)

        return output_frame
