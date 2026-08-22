"""
Camera Management and Video Capture Utility for Windows (DirectShow & MSMF).

Features:
- Fast camera startup using Windows DirectShow (cv2.CAP_DSHOW).
- Automatic detection of external webcams (e.g., Irium Webcam, DroidCam, USB Cams).
- Safe fallback if selected camera index is unavailable.
- Low-latency buffer configuration (cv2.CAP_PROP_BUFFERSIZE = 1).
- Camera hardware discovery and status reporting tool.
"""

import sys
from typing import List, Dict, Optional, Tuple, Any
import cv2

# Suppress verbose OpenCV DirectShow probe warnings
try:
    cv2.utils.logging.setLogLevel(cv2.utils.logging.LOG_LEVEL_ERROR)
except Exception:
    pass

import app.config as config


def get_available_cameras(max_to_test: int = 6) -> List[Dict[str, Any]]:
    """
    Scans Windows video capture devices and returns status and capabilities for each.

    Args:
        max_to_test: Number of indices to probe (0 to max_to_test - 1)

    Returns:
        List of dicts: [{'index': 0, 'width': 640, 'height': 480, 'fps': 30, 'backend': 'DSHOW'}, ...]
    """
    available = []

    for index in range(max_to_test):
        # 1. Try DirectShow backend (Recommended for Windows / Irium)
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        backend_name = "DirectShow"
        
        if not cap.isOpened():
            # 2. Fallback to default backend
            cap.release()
            cap = cv2.VideoCapture(index)
            backend_name = "Default/MSMF"

        if cap.isOpened():
            # Check frame read
            ret, frame = cap.read()
            if ret and frame is not None:
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                available.append({
                    "index": index,
                    "width": w,
                    "height": h,
                    "fps": fps if fps > 0 else 30,
                    "backend": backend_name,
                    "status": "Active & Ready"
                })
            else:
                available.append({
                    "index": index,
                    "width": 0,
                    "height": 0,
                    "fps": 0,
                    "backend": backend_name,
                    "status": "Device Busy / In Use"
                })
            cap.release()

    return available


def open_camera(camera_index: Optional[int] = None,
                width: int = config.FRAME_WIDTH,
                height: int = config.FRAME_HEIGHT,
                use_dshow: bool = config.USE_DIRECTSHOW) -> Tuple[Optional[cv2.VideoCapture], int]:
    """
    Opens the requested camera with DirectShow optimization and safe index fallback.

    Args:
        camera_index: Target camera index (defaults to config.CAMERA_INDEX)
        width: Desired width resolution
        height: Desired height resolution
        use_dshow: Whether to force cv2.CAP_DSHOW

    Returns:
        Tuple: (cv2.VideoCapture or None, active_camera_index)
    """
    target_index = config.CAMERA_INDEX if camera_index is None else camera_index
    backend = cv2.CAP_DSHOW if use_dshow else cv2.CAP_ANY

    print(f"[*] Connecting to Camera Index {target_index} (Backend: {'DirectShow' if use_dshow else 'Default'})...")

    # 1. Try Target Index
    cap = cv2.VideoCapture(target_index, backend)
    if cap.isOpened():
        _configure_capture(cap, width, height)
        # Test read
        ret, frame = cap.read()
        if ret and frame is not None:
            print(f"[+] Camera Index {target_index} opened successfully! (Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))})")
            return cap, target_index
        cap.release()

    # If DirectShow failed on target, try default backend on target
    if use_dshow:
        cap = cv2.VideoCapture(target_index)
        if cap.isOpened():
            _configure_capture(cap, width, height)
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"[+] Camera Index {target_index} opened with default backend.")
                return cap, target_index
            cap.release()

    print(f"[!] Warning: Camera Index {target_index} could not be opened.")

    # 2. Auto-Detect Fallback if enabled
    if config.AUTO_DETECT_CAMERA:
        print("[*] Probing available system cameras for automatic fallback...")
        available = get_available_cameras()
        if available:
            for cam in available:
                if cam["status"] == "Active & Ready" and cam["index"] != target_index:
                    fallback_idx = cam["index"]
                    print(f"[*] Attempting fallback to Camera Index {fallback_idx} ({cam['backend']})...")
                    fallback_cap = cv2.VideoCapture(fallback_idx, backend)
                    if not fallback_cap.isOpened():
                        fallback_cap = cv2.VideoCapture(fallback_idx)
                    
                    if fallback_cap.isOpened():
                        _configure_capture(fallback_cap, width, height)
                        print(f"[+] Successfully connected to fallback Camera Index {fallback_idx}!")
                        return fallback_cap, fallback_idx

    # If all failed, print diagnostic list
    print("\n" + "!" * 65)
    print(f"[ERROR] Could not connect to Camera Index {target_index}.")
    print("Available Cameras Detected on System:")
    cams = get_available_cameras()
    if not cams:
        print("  - No active webcams detected! Make sure Irium Webcam app is running on PC and phone.")
    else:
        for c in cams:
            print(f"  - Camera Index {c['index']}: {c['status']} ({c['width']}x{c['height']}, {c['backend']})")
    print("!" * 65 + "\n")
    return None, target_index


def _configure_capture(cap: cv2.VideoCapture, width: int, height: int):
    """Configures resolution and disables frame buffer lag."""
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    # Set internal buffer to 1 frame to eliminate lag in real-time recognition
    try:
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    except Exception:
        pass
