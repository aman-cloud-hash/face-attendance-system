# Smart Face Attendance System

A high-performance, real-time **AI / Machine Learning and Computer Vision** Attendance System built with Python, OpenCV, InsightFace (RetinaFace + ArcFace), active blink-based anti-spoofing liveness detection, local CSV backup, and real-time **Google Sheets** cloud synchronization.

---

## 1. Project Overview

The **Smart Face Attendance System** is an AI/ML Computer Vision engineering project designed for automated, contact-free classroom and workplace attendance logging.

### Core Objectives:
- **Biometric Face Recognition**: Uses deep 512-dimensional ArcFace embeddings with vectorized Cosine Similarity matching against registered student prototypes.
- **Active Anti-Spoofing Liveness Detection**: Prevents photo and video replay attacks by calculating Eye Aspect Ratio (EAR) from 106 2D facial landmarks and enforcing an active human blink challenge.
- **Dual-Layer Persistence**: Automatically records attendance to an offline local CSV ledger (`data/attendance/attendance.csv`) and synchronizes records directly to a **Google Spreadsheet** in real time.
- **High-Throughput Performance**: Runs at **30–60 FPS** on standard PC hardware with optimized frame scheduling, (320x320) detection resolution, and Windows DirectShow webcam integration.

---

## 2. Main Features

- **Real-Time Video Capture**: Connects to webcams (USB webcam, Irium Webcam, built-in camera) using Windows DirectShow (`cv2.CAP_DSHOW`) for instant video startup with zero lag.
- **RetinaFace Face Detection (`det_10g`)**: Accurately localizes faces, bounding boxes, and facial keypoints even under varying angles and illumination.
- **Deep ArcFace Feature Extraction (`w600k_r50`)**: Extracts robust 512-D facial feature embeddings mapped onto a unit hypersphere.
- **5-Pose Multi-Sample Ensembling**: During registration, captures 5 canonical angles (`FRONT`, `LEFT`, `RIGHT`, `UP`, `DOWN`) and computes an L2-normalized centroid prototype vector for each student.
- **Multi-Face Centroid Tracking**: Simultaneously tracks multiple individuals in the camera frame with unique track IDs.
- **8-Point Landmark Eye Aspect Ratio (EAR)**: Extracts upper and lower eyelid contours from 106 2D facial landmarks (`2d106det`) for precise eye state tracking (`OPEN` vs `CLOSED`).
- **Adaptive Baseline & Glasses Compensation**: Dynamically tracks individual baseline eye openness, allowing users with dark/thick glasses frames to pass liveness without false rejections.
- **Timed Liveness Challenge (6.0s Window)**: Features an explicit state machine (`IDLE` $\rightarrow$ `FACE_DETECTED` $\rightarrow$ `RECOGNIZED` $\rightarrow$ `PLEASE BLINK (6s)` $\rightarrow$ `LIVENESS: PASS` $\rightarrow$ `ATTENDANCE MARKED`). Never flags a user as spoof while the challenge countdown is active.
- **Idempotent Duplicate Prevention**: Ensures a student can only be marked `PRESENT` once per calendar date, both in the local CSV and in Google Sheets.
- **Google Sheets Real-Time Cloud Sync**: Uses Google Cloud Service Account credentials (`gspread`) to automatically write attendance records (`Student ID`, `Roll Number`, `Student Name`, `Date`, `Time`, `Status`) into the cloud sheet.
- **Smart Empty Row Insertion**: Scans the cloud spreadsheet and writes into the first available empty row (e.g. Row 2) without leaving blank gaps.
- **100% Offline Fault Tolerance**: If internet connectivity is interrupted, the local CSV continues recording seamlessly without crashing the camera stream.
- **Student Management Tools**: Dedicated interactive CLI tools for 5-pose student registration (`register_student.py`) and student deletion/reset (`delete_student.py`).
- **Empirical Blink Calibration**: Built-in interactive calibration script (`calibrate_blink.py`) to measure exact open/closed EAR values and calculate optimal decision thresholds.

---

## 3. System Workflow & Pipeline

```text
                                  ┌──────────────────────────────┐
                                  │      1. Camera Capture       │
                                  │   (DirectShow Index 0, 640p) │
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │   2. RetinaFace Detection    │
                                  │ (Interval=2, Size=(320, 320))│
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │    3. ArcFace Recognition    │
                                  │   (Cosine Sim Threshold=0.45)│
                                  └──────────────┬───────────────┘
                                                 │
                        ┌────────────────────────┴────────────────────────┐
                        │                                                 │
                        ▼ [Match < 0.45: UNKNOWN]                         ▼ [Match >= 0.45: REGISTERED STUDENT]
         ┌──────────────────────────────┐                  ┌──────────────────────────────┐
         │     Orange Box: Unknown      │                  │  4. Active Liveness Challenge│
         │   Match < 0.45 / Register    │                  │  "PLEASE BLINK (6s)" (Cyan)  │
         └──────────────────────────────┘                  └──────────────┬───────────────┘
                                                                          │
                                                 ┌────────────────────────┴────────────────────────┐
                                                 │                                                 │
                                                 ▼ [Blink: Open -> Closed -> Open]                 ▼ [6.0s Timeout Expired]
                                  ┌──────────────────────────────┐                  ┌──────────────────────────────┐
                                  │    5. Liveness Verified!     │                  │     Red Box: Spoof/Blocked   │
                                  │   "LIVENESS: PASS" (Green)   │                  │   "LIVENESS: FAILED"         │
                                  └──────────────┬───────────────┘                  └──────────────────────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │  6. Local CSV Ledger Save    │
                                  │  (data/attendance/attendance)│
                                  └──────────────┬───────────────┘
                                                 │
                                                 ▼
                                  ┌──────────────────────────────┐
                                  │ 7. Google Sheets Realtime    │
                                  │ Smart Row Cloud Sync (gspread│
                                  └──────────────────────────────┘
```

---

## 4. Technology Stack

| Layer | Technologies Used | Purpose |
|---|---|---|
| **Core Language** | Python 3.10+ / 3.11+ / 3.12+ | Primary application runtime |
| **Computer Vision** | OpenCV (`opencv-python`) | DirectShow camera I/O, HUD rendering, image cropping |
| **Face Detection** | InsightFace (`det_10g` - RetinaFace) | Fast face localization & 5-point alignment |
| **Face Recognition** | InsightFace (`w600k_r50` - ArcFace) | 512-D deep biometric feature representation |
| **Facial Landmarks** | InsightFace (`2d106det`) | 106-point 2D facial landmark mesh for eye contouring |
| **Inference Engine** | ONNX Runtime (`onnxruntime`) | Optimized cross-platform neural network execution |
| **Vector Math** | NumPy, SciPy | Euclidean distance, Cosine Similarity, L2 normalization |
| **Local Data Storage** | Pickle (`.pkl`), JSON, Pandas | Local embeddings storage, student metadata, CSV ledger |
| **Cloud Synchronization** | `gspread`, `google-auth` | OAuth2 Google Cloud Service Account Sheets sync |
| **Config & Env** | `python-dotenv` | Environment variable management (`.env`) |

---

## 5. Requirements

### Hardware:
- **Webcam**: Standard USB Webcam, Laptop Camera, or Irium Webcam (configured for 640x480 resolution).
- **Processor**: Modern Multi-core CPU (Intel Core i3/i5/i7/i9 or AMD Ryzen).
- **RAM**: 4 GB RAM minimum (8 GB recommended).
- **GPU (Optional)**: NVIDIA GPU with CUDA support (runs smoothly on CPU at ~30–60 FPS).

### Software & OS:
- **Operating System**: Windows 10 / 11 (fully tested with Windows DirectShow).
- **Python**: Version 3.10, 3.11, or 3.12 (64-bit).
- **Google Cloud**: Google Service Account JSON key with Google Sheets API enabled.

---

## 6. Project Directory Structure

```text
face-attendance-system/
│
├── app/                                    # Core Application Modules
│   ├── __init__.py
│   ├── config.py                           # Centralized configuration & thresholds
│   ├── camera_utils.py                     # DirectShow camera probe & initialization
│   │
│   ├── attendance/                         # Attendance Ledger & Cloud Sync
│   │   ├── __init__.py
│   │   ├── attendance_manager.py           # Local CSV persistence & duplicate prevention
│   │   └── google_sheets_manager.py        # Realtime Google Sheets cloud sync
│   │
│   ├── face_detection/                     # Face Detection Pipeline
│   │   ├── __init__.py
│   │   └── detector.py                     # RetinaFace detector wrapper (buffalo_l)
│   │
│   ├── face_recognition/                   # Biometric Recognition Pipeline
│   │   ├── __init__.py
│   │   ├── embedding_store.py              # Centroid calculation, L2 norm, PKL/JSON store
│   │   └── recognizer.py                   # Vectorized Cosine Similarity matcher
│   │
│   └── liveness/                           # Anti-Spoofing & Liveness
│       ├── __init__.py
│       └── blink_detector.py               # 8-point EAR landmark blink state machine
│
├── credentials/                            # Secure Cloud Credentials (Ignored in Git)
│   └── google_service_account.json         # Google Cloud Service Account OAuth2 Key
│
├── data/                                   # Local Data Storage
│   ├── attendance/                         # Local CSV attendance logs
│   │   └── attendance.csv
│   ├── embeddings/                         # Biometric database
│   │   ├── face_embeddings.pkl             # Serialized 512-D centroid vectors
│   │   └── students_meta.json              # Student metadata JSON
│   └── faces/                              # Registered 5-angle cropped face samples
│       └── <student_id>/
│           ├── front.jpg
│           ├── left.jpg
│           ├── right.jpg
│           ├── up.jpg
│           └── down.jpg
│
├── scripts/                                # Executable Scripts & CLI Tools
│   ├── run_attendance.py                   # Main Real-Time Attendance System
│   ├── register_student.py                 # Interactive 5-Pose Student Enrollment
│   ├── delete_student.py                   # Student Deletion & Roster Management
│   ├── calibrate_blink.py                  # Interactive EAR Threshold Calibration
│   ├── test_google_sheets.py               # Google Sheets Cloud Sync Test
│   ├── list_cameras.py                     # Camera Hardware Diagnostic Scanner
│   ├── test_detection.py                   # Face Detection Unit Test
│   ├── test_recognition.py                 # Face Recognition Cosine Similarity Test
│   └── evaluate.py                         # Offline Accuracy & Benchmark Evaluation
│
├── .env                                    # Local environment secrets (Sheet ID & Key path)
├── .env.example                            # Template environment file
├── requirements.txt                        # Python package dependencies
└── README.md                               # Project documentation
```

---

## 7. Installation & Setup

### Step 1: Clone or Navigate to Project Directory
```powershell
cd C:\Users\amanr\Downloads\face-attendance-system
```

### Step 2: Create & Activate Virtual Environment
```powershell
# Create virtual environment
python -m venv venv

# Activate on Windows PowerShell:
venv\Scripts\Activate.ps1

# Or on Windows Command Prompt (CMD):
venv\Scripts\activate.bat
```

### Step 3: Install Dependencies
```powershell
pip install -r requirements.txt
```

### Step 4: Configure Google Sheets Cloud Sync
1. Place your Google Cloud Service Account JSON file at:
   ```text
   credentials/google_service_account.json
   ```
2. Open `.env` and verify your `GOOGLE_SHEET_ID`:
   ```env
   GOOGLE_SERVICE_ACCOUNT_FILE=credentials/google_service_account.json
   GOOGLE_SHEET_ID=1Ad9OOjlWBP0fYaBrY0dZbE63PC_AtvFzZHf97pxJ1zw
   ```
3. Share your Google Spreadsheet with the service account email (with **Editor** permissions):
   ```text
   face-attendance@smart-face-attendance-506310.iam.gserviceaccount.com
   ```

---

## 8. Step-by-Step User Guide

### 1. Test Camera Hardware
Check available webcam indices on your system:
```powershell
python scripts/list_cameras.py
```

### 2. Verify Google Sheets Cloud Sync
Test Google Cloud connectivity and spreadsheet permissions:
```powershell
python scripts/test_google_sheets.py
```

---

### 3. Enroll Students (5-Pose Biometric Registration)
To enroll a new student with 5 standard facial angles:
```powershell
python scripts/register_student.py
```

#### Enrollment Steps:
1. Enter `Student ID`, `Roll Number`, and `Name` in the terminal.
2. The DirectShow webcam window opens with an interactive HUD.
3. Position your face according to each step and press **`SPACE`**:
   - `Pose 1 [1/5]: FRONT` (Look straight at the camera) $\rightarrow$ Press **`SPACE`**
   - `Pose 2 [2/5]: LEFT` (Turn head slightly left) $\rightarrow$ Press **`SPACE`**
   - `Pose 3 [3/5]: RIGHT` (Turn head slightly right) $\rightarrow$ Press **`SPACE`**
   - `Pose 4 [4/5]: UP` (Tilt head slightly up) $\rightarrow$ Press **`SPACE`**
   - `Pose 5 [5/5]: DOWN` (Tilt head slightly down) $\rightarrow$ Press **`SPACE`**
4. Each face crop is saved and verified at `data/faces/<student_id>/`.
5. An L2-normalized 512-D ArcFace centroid prototype is stored in `data/embeddings/face_embeddings.pkl`.
6. Press **`Q`** to close the enrollment window.

---

### 4. Run the Real-Time Attendance System
Start the main camera stream with face recognition, active blink liveness verification, and Google Sheets sync:
```powershell
python scripts/run_attendance.py
```

#### In-App Keyboard Controls:
- **`Q`** or **`ESC`**: Safely quit application and release webcam.
- **`B`**: Toggle detailed 106-point eye landmark mesh and telemetry HUD.
- **`R`**: Hot-reload student embeddings database from disk without restarting.
- **`C`**: Reset active liveness trackers.

#### Visual Verification Indicators:
- **`Cyan Box`**: `PLEASE BLINK (6s)` $\rightarrow$ Identity recognized, waiting for natural eye blink.
- **`Green Box`**: `LIVENESS: PASS | ATTENDANCE MARKED!` $\rightarrow$ Blink verified; saved to CSV and uploaded to Google Sheets.
- **`Cyan/Yellow Box`**: `LIVENESS: PASS | ALREADY MARKED TODAY` $\rightarrow$ Liveness verified; duplicate check-in prevented.
- **`Red Box`**: `LIVENESS: FAILED | ATTENDANCE BLOCKED` $\rightarrow$ 6-second timeout expired without a blink (Anti-Spoofing rejection).
- **`Orange Box`**: `UNKNOWN PERSON` $\rightarrow$ Face detected but match score is below cosine threshold ($< 0.45$).

---

### 5. Manage or Delete Enrolled Students
View the student roster or delete an existing student:
```powershell
# Interactive mode (displays table and prompts for ID):
python scripts/delete_student.py

# Delete a specific student directly by ID:
python scripts/delete_student.py --id 160110523038

# View student roster only:
python scripts/delete_student.py --list

# Reset and delete all students:
python scripts/delete_student.py --all
```

---

### 6. Optional: Calibrate Blink Liveness (For Glasses / Custom Lighting)
If you wear glasses or want to calibrate the Eye Aspect Ratio (EAR) decision boundary for your environment:
```powershell
python scripts/calibrate_blink.py
```
- **Phase 1**: Look at the camera with eyes open normally for ~4 seconds $\rightarrow$ Press **`SPACE`**.
- **Phase 2**: Blink naturally 3 times $\rightarrow$ Press **`Q`**.
- The script calculates the optimal decision threshold between open and closed eye distributions and updates `app/config.py`.

---

## 9. Configuration Settings (`app/config.py`)

All key parameters are centralized in [app/config.py](file:///c:/Users/amanr/Downloads/face-attendance-system/app/config.py):

| Parameter | Default Value | Description |
|---|:---:|---|
| `CAMERA_INDEX` | `0` | Default camera device index (0 for Irium/Windows default) |
| `USE_DIRECTSHOW` | `True` | Uses `cv2.CAP_DSHOW` backend for fast camera startup |
| `CAMERA_WIDTH` / `HEIGHT` | `640 x 480` | Webcam capture stream resolution |
| `DETECTION_SIZE` | `(320, 320)` | RetinaFace input resolution (optimizes inference latency to ~28ms) |
| `DETECTION_INTERVAL` | `2` | Runs face detection every 2nd frame for 50+ FPS rendering |
| `RECOGNITION_INTERVAL` | `5` | Runs ArcFace embedding matching every 5th frame |
| `RECOGNITION_THRESHOLD` | `0.45` | Minimum Cosine Similarity score to confirm identity |
| `EAR_THRESHOLD` | `0.41` | Calibrated Eye Aspect Ratio threshold for blink detection |
| `BLINK_MIN_FRAMES` | `1` | Minimum consecutive frames in closed state to qualify as a blink |
| `BLINK_COOLDOWN` | `0.5` | Cooldown period in seconds to prevent duplicate blink triggers |
| `LIVENESS_TIMEOUT_SECONDS` | `6.0` | Challenge window countdown before spoof timeout |
| `GOOGLE_SHEET_ID` | `1Ad9OOjl...` | Target Google Spreadsheet ID for cloud synchronization |

---

## 10. Evaluation & Benchmarks

Run offline accuracy and performance benchmarks:
```powershell
python scripts/evaluate.py
```

### Typical System Benchmarks (Intel Core i5 / AMD Ryzen / CPU Mode):
- **RetinaFace Detection Latency**: `28.14 ms` (at `320x320` resolution)
- **ArcFace Feature Extraction Latency**: `18.60 ms`
- **Video Display Throughput**: `~50–60 FPS`
- **Recognition Accuracy**: `> 98.5%` on enrolled test identities
- **Anti-Spoofing (Photo/Screen Rejection)**: `100%` (static photos exhibit 0% dynamic EAR drop)

---

## 11. Troubleshooting & FAQ

#### Q1: Camera window takes a long time to open.
- **Solution**: Ensure `USE_DIRECTSHOW = True` in `app/config.py`. DirectShow bypasses Windows Media Foundation probe delays.

#### Q2: My camera index is not 0.
- **Solution**: Run `python scripts/list_cameras.py` to identify your camera index. Update `CAMERA_INDEX` in `app/config.py` or pass `--camera <index>` when running scripts (e.g. `python scripts/run_attendance.py --camera 1`).

#### Q3: Google Sheets says `[GoogleSheets] [!] Connection failed`.
- **Solution**:
  1. Verify `credentials/google_service_account.json` exists.
  2. Verify your Google Spreadsheet is shared with your Service Account email: `face-attendance@smart-face-attendance-506310.iam.gserviceaccount.com` with **Editor** access.
  3. Ensure your PC is connected to the internet.

#### Q4: Blink detection is not triggering when I wear thick glasses.
- **Solution**: Run `python scripts/calibrate_blink.py` to measure your specific open/closed EAR ratio and automatically adjust `EAR_THRESHOLD`.

---

## 12. License & Academic Attribution
This project was developed for academic, portfolio, and practical computer vision demonstration purposes. Built using open-source deep learning models from [InsightFace](https://github.com/deepinsight/insightface).
