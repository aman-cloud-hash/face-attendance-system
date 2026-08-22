"""
Centralized Configuration Module for Face Attendance AI System.

Handles:
- File system paths for data, models, attendance logs, and embeddings
- Camera settings (DirectShow, CAMERA_INDEX=0 for Irium/Windows default)
- AI inference interval optimizations (Detection Interval, Recognition Interval)
- Model configurations for RetinaFace (det_size=(320, 320)), ArcFace, and Landmarks
- Calibrated Eye Aspect Ratio (EAR) parameters for Active Liveness (Blink Detection)
- OpenCV UI Color palettes and display styling
"""

import os
import warnings
from pathlib import Path

# Silence verbose C++ video capture probe warnings on Windows
os.environ["OPENCV_LOG_LEVEL"] = "ERROR"
os.environ["OPENCV_VIDEOIO_DEBUG"] = "0"

# Silence Scikit-Image / InsightFace FutureWarnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# ==========================================
# BASE DIRECTORIES & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
FACES_DIR = DATA_DIR / "faces"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
ATTENDANCE_DIR = DATA_DIR / "attendance"
MODELS_DIR = BASE_DIR / "models"
SCRIPTS_DIR = BASE_DIR / "scripts"
EVALUATION_DIR = BASE_DIR / "evaluation"

# Storage Files
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "face_embeddings.pkl"
STUDENT_META_FILE = EMBEDDINGS_DIR / "students_meta.json"
ATTENDANCE_CSV = ATTENDANCE_DIR / "attendance.csv"
EVALUATION_RESULTS_CSV = EVALUATION_DIR / "results.csv"

# Ensure all critical directories exist
for directory in [FACES_DIR, EMBEDDINGS_DIR, ATTENDANCE_DIR, MODELS_DIR, SCRIPTS_DIR, EVALUATION_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==========================================
# WEBCAM & VIDEO STREAM CONFIGURATION
# ==========================================
# Primary Camera Index:
# 0 = Irium Webcam / Default Windows Camera (Directly configured)
CAMERA_INDEX = 0

# Windows DirectShow provides near-instant webcam startup
USE_DIRECTSHOW = True

# Fallback to active camera if configured CAMERA_INDEX is offline
AUTO_DETECT_CAMERA = True

# Webcam Stream Resolution
CAMERA_WIDTH = 640
CAMERA_HEIGHT = 480
FRAME_WIDTH = CAMERA_WIDTH
FRAME_HEIGHT = CAMERA_HEIGHT
TARGET_FPS = 30

# ==========================================
# AI INFERENCE PIPELINE OPTIMIZATIONS
# ==========================================
# Detection Size: (320, 320) reduces inference latency by ~4.5x
DETECTION_SIZE = (320, 320)
DETECTION_THRESHOLD = 0.50

# Frame Skipping / Intervals:
DETECTION_INTERVAL = 2
RECOGNITION_INTERVAL = 5

# InsightFace Model & Execution Provider Config
INSIGHTFACE_MODEL_NAME = "buffalo_l"
PREF_PROVIDERS = ['CUDAExecutionProvider', 'CPUExecutionProvider']

# ==========================================
# FACE RECOGNITION CONFIGURATION
# ==========================================
RECOGNITION_THRESHOLD = 0.45
EMBEDDING_DIM = 512

# Registration Configuration
SAMPLES_PER_STUDENT = 5
SAMPLE_ANGLES = ["Front", "Slight Left", "Slight Right", "Slight Up", "Slight Down"]

# ==========================================
# LIVENESS & BLINK DETECTION (EAR) CONFIG
# ==========================================
# Eye Aspect Ratio (EAR) Calibration:
# Measured with glasses & webcam:
# Open Eyes EAR: ~0.45 - 0.48
# Closed Eyes EAR dip with glasses: ~0.38 - 0.41
# Calibrated Baseline Decision Threshold = 0.41
EAR_THRESHOLD = 0.41
BLINK_MIN_FRAMES = 1
BLINK_COOLDOWN = 0.5
LIVENESS_TIMEOUT_SECONDS = 6.0

# ==========================================
# ATTENDANCE CSV & GOOGLE SHEETS CONFIG
# ==========================================
ATTENDANCE_CSV_COLUMNS = [
    "student_id",
    "roll_number",
    "name",
    "date",
    "check_in_time",
    "status",
    "recognition_confidence",
    "liveness_status"
]

# Google Sheets Realtime Cloud Sync
from dotenv import load_dotenv
load_dotenv(BASE_DIR / ".env")

GOOGLE_SERVICE_ACCOUNT_FILE = BASE_DIR / os.getenv("GOOGLE_SERVICE_ACCOUNT_FILE", "credentials/google_service_account.json")
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID", "1Ad9OOjlWBP0fYaBrY0dZbE63PC_AtvFzZHf97pxJ1zw")
GOOGLE_SHEETS_HEADERS = [
    "Student ID",
    "Roll Number",
    "Student Name",
    "Date",
    "Time",
    "Status"
]

# ==========================================
# UI & VISUALIZATION STYLING (BGR Colors)
# ==========================================
COLOR_RECOGNIZED = (0, 200, 0)       # Green: Verified & Present
COLOR_UNKNOWN = (0, 165, 255)        # Orange: Unknown identity
COLOR_FAKE = (0, 0, 230)             # Red: Spoof / Liveness Failed
COLOR_CHALLENGE = (255, 191, 0)      # Cyan / Deep Sky: Active Blink Challenge
COLOR_ALREADY_MARKED = (255, 200, 0) # Cyan/Yellow: Already Marked Today
COLOR_INFO_BOX = (30, 30, 30)        # Dark Gray background for text badges
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)

FONT = 0  # cv2.FONT_HERSHEY_SIMPLEX
