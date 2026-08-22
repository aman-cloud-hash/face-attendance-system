"""
Script: calibrate_blink.py
Purpose: Empirical Eye Aspect Ratio (EAR) Calibration & Diagnostic Tool.

Why Calibration is Needed:
- Facial geometry, webcam resolution, distance, and glasses frames affect raw EAR values.
- For users with glasses, open-eye EAR is typically ~0.40 - 0.48, and closed-eye dips reach ~0.26 - 0.33.
- This script empirically measures YOUR specific Open-Eye and Closed-Eye EAR values,
  calculates the optimal decision threshold, and updates app/config.py.
"""

import sys
import time
import argparse
from pathlib import Path
from typing import List, Tuple
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.camera_utils import open_camera
from app.face_detection.detector import FaceDetector
from app.liveness.blink_detector import BlinkLivenessDetector


def draw_eye_geometry(frame: np.ndarray, lm106: np.ndarray) -> np.ndarray:
    """
    Renders eye contour landmarks and vertical/horizontal measurement lines on frame.
    """
    out = frame.copy()
    if lm106 is None or len(lm106) < 106:
        return out

    # Left Eye: 35 (outer), 36, 37, 38 (top), 39 (inner), 40, 41, 42 (bottom)
    # Right Eye: 89 (inner), 90, 91, 92 (top), 93 (outer), 94, 95, 96 (bottom)
    left_pts = [35, 36, 37, 38, 39, 40, 41, 42]
    right_pts = [89, 90, 91, 92, 93, 94, 95, 96]

    # Draw points
    for idx in left_pts + right_pts:
        pt = lm106[idx]
        cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

    # Draw measurement lines (Left Eye)
    cv2.line(out, (int(lm106[35][0]), int(lm106[35][1])), (int(lm106[39][0]), int(lm106[39][1])), (0, 255, 0), 1)
    cv2.line(out, (int(lm106[36][0]), int(lm106[36][1])), (int(lm106[42][0]), int(lm106[42][1])), (255, 0, 0), 1)
    cv2.line(out, (int(lm106[37][0]), int(lm106[37][1])), (int(lm106[41][0]), int(lm106[41][1])), (255, 0, 0), 1)
    cv2.line(out, (int(lm106[38][0]), int(lm106[38][1])), (int(lm106[40][0]), int(lm106[40][1])), (255, 0, 0), 1)

    # Draw measurement lines (Right Eye)
    cv2.line(out, (int(lm106[89][0]), int(lm106[89][1])), (int(lm106[93][0]), int(lm106[93][1])), (0, 255, 0), 1)
    cv2.line(out, (int(lm106[90][0]), int(lm106[90][1])), (int(lm106[96][0]), int(lm106[96][1])), (255, 0, 0), 1)
    cv2.line(out, (int(lm106[91][0]), int(lm106[91][1])), (int(lm106[95][0]), int(lm106[95][1])), (255, 0, 0), 1)
    cv2.line(out, (int(lm106[92][0]), int(lm106[92][1])), (int(lm106[94][0]), int(lm106[94][1])), (255, 0, 0), 1)

    return out


def run_calibration(camera_index: int = config.CAMERA_INDEX):
    """
    Interactive 2-phase calibration workflow.
    """
    print("\n" + "=" * 70)
    print("       EYE ASPECT RATIO (EAR) EMPIRICAL CALIBRATION")
    print("=" * 70)
    print("[*] Opening Webcam (DirectShow)...")
    
    cap, active_cam_idx = open_camera(
        camera_index=camera_index,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        use_dshow=config.USE_DIRECTSHOW
    )
    if cap is None:
        print("[!] Error: Could not connect to webcam.")
        return

    print("[*] Loading InsightFace Landmark Detector...")
    detector = FaceDetector(det_size=config.DETECTION_SIZE)
    liveness = BlinkLivenessDetector()
    print("[+] System Ready!\n")

    print("=" * 70)
    print("CALIBRATION INSTRUCTIONS:")
    print("  1. PHASE 1: Keep eyes OPEN normally for ~4 seconds.")
    print("  2. PHASE 2: BLINK naturally 3 times (close for ~1s, then open).")
    print("  3. Press [SPACE] when you are ready to switch phases.")
    print("  4. Press [Q] to calculate results and finish.")
    print("=" * 70 + "\n")

    current_phase = 1  # 1 = OPEN EYES BASELINE, 2 = BLINK DIP CAPTURE
    open_ear_samples: List[float] = []
    closed_ear_samples: List[float] = []
    all_ears_history: List[float] = []

    lowest_ear_seen = 1.0
    highest_ear_seen = 0.0

    fps_smooth = 0.0
    prev_time = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = (0.92 * fps_smooth) + (0.08 * fps) if fps_smooth > 0 else fps

            h, w = frame.shape[:2]
            faces = detector.detect_faces(frame)

            display_frame = frame.copy()
            active_face = faces[0] if faces else None

            curr_ear, left_ear, right_ear = 0.0, 0.0, 0.0

            if active_face is not None:
                # Extract EAR
                left_ear, right_ear, curr_ear = liveness.extract_ear_from_face(active_face)
                all_ears_history.append(curr_ear)

                if curr_ear < lowest_ear_seen:
                    lowest_ear_seen = curr_ear
                if curr_ear > highest_ear_seen:
                    highest_ear_seen = curr_ear

                # Record samples based on current phase
                if current_phase == 1:
                    open_ear_samples.append(curr_ear)
                elif current_phase == 2:
                    # In phase 2, we collect all samples during the blink action
                    closed_ear_samples.append(curr_ear)

                # Draw landmark mesh on eye
                if hasattr(active_face, 'landmark_2d_106') and active_face.landmark_2d_106 is not None:
                    display_frame = draw_eye_geometry(display_frame, active_face.landmark_2d_106)

                # Bounding box
                bbox = active_face.bbox.astype(int)
                cv2.rectangle(display_frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)

            # TOP HUD
            cv2.rectangle(display_frame, (0, 0), (w, 105), (20, 20, 20), -1)
            cv2.line(display_frame, (0, 105), (w, 105), (60, 60, 60), 2)

            if current_phase == 1:
                phase_title = "PHASE 1: KEEP EYES OPEN NATURALLY"
                phase_color = (0, 255, 255)
                phase_sub = f"Recorded {len(open_ear_samples)} Open Frames | Press [SPACE] to go to Phase 2"
            else:
                phase_title = "PHASE 2: BLINK NATURALLY 3 TIMES"
                phase_color = (0, 255, 0)
                phase_sub = f"Blink 3 times (close ~1s & open) | Press [Q] to Finish & Save"

            cv2.putText(display_frame, phase_title, (15, 26), config.FONT, 0.60, phase_color, 2, cv2.LINE_AA)
            cv2.putText(display_frame, phase_sub, (15, 52), config.FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

            # Metrics Line
            metrics_str = f"Live EAR: {curr_ear:.3f} (L: {left_ear:.3f}, R: {right_ear:.3f})  |  Lowest: {lowest_ear_seen:.3f}  |  Highest: {highest_ear_seen:.3f}"
            cv2.putText(display_frame, metrics_str, (15, 82), config.FONT, 0.50, (0, 255, 0) if curr_ear > 0 else (0, 0, 255), 1, cv2.LINE_AA)

            cv2.putText(display_frame, f"FPS: {fps_smooth:.1f}", (w - 110, 26), config.FONT, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

            # BOTTOM HUD
            cv2.rectangle(display_frame, (0, h - 35), (w, h), (20, 20, 20), -1)
            cv2.putText(display_frame, "[SPACE] Switch Phase  |  [Q] Compute Threshold & Finish", (15, h - 12), 
                        config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

            cv2.imshow("Blink EAR Calibration - AI / CV", display_frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 32:  # Spacebar
                if current_phase == 1:
                    current_phase = 2
                    print(f"\n[+] Phase 1 Complete! Collected {len(open_ear_samples)} Open-Eye EAR samples.")
                    print("[*] Entering Phase 2: Please perform 3 natural blinks now. Press 'Q' when done.\n")
            elif key in [ord('q'), ord('Q'), 27]:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

    # ==========================================
    # CALCULATE CALIBRATION RESULTS
    # ==========================================
    print("\n" + "=" * 65)
    print("           BLINK CALIBRATION RESULT REPORT")
    print("=" * 65)

    if not open_ear_samples:
        print("[!] No open-eye samples collected. Using fallback defaults.")
        return

    open_min = float(np.min(open_ear_samples))
    open_max = float(np.max(open_ear_samples))
    open_avg = float(np.mean(open_ear_samples))

    # Closed eye estimation: the lowest 15% quantile of samples during phase 2 (or overall lowest)
    if closed_ear_samples:
        closed_sorted = sorted(closed_ear_samples)
        # Take the lowest 15% readings representing the closed eye duration
        low_count = max(3, int(len(closed_sorted) * 0.15))
        closed_subset = closed_sorted[:low_count]
        closed_min = float(np.min(closed_subset))
        closed_max = float(np.max(closed_subset))
        closed_avg = float(np.mean(closed_subset))
    else:
        closed_min = lowest_ear_seen
        closed_max = lowest_ear_seen + 0.05
        closed_avg = lowest_ear_seen + 0.02

    # Calculate optimal decision boundary (midpoint)
    # Recommended Threshold = (Open_Avg + Closed_Avg) / 2
    recommended_thresh = round(float((open_avg + closed_avg) / 2.0), 3)

    # Safety bounds check
    if recommended_thresh < 0.20:
        recommended_thresh = 0.23
    elif recommended_thresh > 0.40:
        recommended_thresh = 0.35

    print(f"Open Eyes Distribution (N={len(open_ear_samples)}):")
    print(f"  - Minimum EAR : {open_min:.3f}")
    print(f"  - Maximum EAR : {open_max:.3f}")
    print(f"  - Average EAR : {open_avg:.3f}")
    print("\nClosed Eyes Distribution / Lowest Dips:")
    print(f"  - Minimum EAR : {closed_min:.3f}")
    print(f"  - Maximum EAR : {closed_max:.3f}")
    print(f"  - Average EAR : {closed_avg:.3f}")
    print("-" * 65)
    print(f"RECOMMENDED OPTIMAL EAR THRESHOLD: >>> {recommended_thresh:.3f} <<<")
    print("=" * 65)

    # Automatically update app/config.py
    update_config_threshold(recommended_thresh)


def update_config_threshold(new_threshold: float):
    """Updates EAR_THRESHOLD in app/config.py."""
    config_file = config.BASE_DIR / "app" / "config.py"
    if not config_file.exists():
        return

    try:
        content = config_file.read_text(encoding="utf-8")
        import re
        # Replace EAR_THRESHOLD = X.XX
        new_content = re.sub(
            r"EAR_THRESHOLD\s*=\s*[0-9.]+",
            f"EAR_THRESHOLD = {new_threshold:.2f}",
            content
        )
        config_file.write_text(new_content, encoding="utf-8")
        print(f"[SUCCESS] Updated 'EAR_THRESHOLD = {new_threshold:.2f}' in app/config.py!")
    except Exception as e:
        print(f"[!] Note: Could not auto-update config.py ({e}). Please set EAR_THRESHOLD = {new_threshold:.2f} manually.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate Eye Aspect Ratio (EAR) for Blink Detection")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    args = parser.parse_args()

    run_calibration(camera_index=args.camera)
