"""
Script: register_student.py
Purpose: High-Performance, Interactive Face Enrollment and Biometric Registration System.

Key Features:
- Smooth DirectShow camera stream with zero preview lag.
- Strict capture validation: Exactly 1 face, minimum size, centering check.
- Clear step-by-step visual HUD with real-time feedback.
- Physical disk verification of saved samples (data/faces/<student_id>/<pose>.jpg).
- 5 Standard Biometric Poses: FRONT, LEFT, RIGHT, UP, DOWN.
- Automatic ArcFace centroid embedding calculation and database persistence.
"""

import sys
import os
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List
import cv2
import numpy as np

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.camera_utils import open_camera
from app.face_detection.detector import FaceDetector
from app.face_recognition.embedding_store import EmbeddingStore


POSES = ["FRONT", "LEFT", "RIGHT", "UP", "DOWN"]
POSE_FILES = ["front.jpg", "left.jpg", "right.jpg", "up.jpg", "down.jpg"]
POSE_INSTRUCTIONS = [
    "Look straight at the camera",
    "Turn head slightly to your LEFT",
    "Turn head slightly to your RIGHT",
    "Tilt head slightly UPWARDS",
    "Tilt head slightly DOWNWARDS"
]


def validate_face_quality(faces: list, frame_w: int, frame_h: int) -> Tuple[bool, Optional[Any], str, tuple]:
    """
    Validates face presence, count, minimum size, and frame centering.

    Returns:
        Tuple: (is_valid, face_object_or_none, status_message, bgr_color)
    """
    if len(faces) == 0:
        return False, None, "No face detected - position your face", (0, 0, 255)

    if len(faces) > 1:
        return False, None, f"Multiple faces ({len(faces)}) - only 1 allowed", (0, 0, 255)

    face = faces[0]
    bbox = face.bbox.astype(int)
    bw = bbox[2] - bbox[0]
    bh = bbox[3] - bbox[1]

    # 1. Size Validation (Must not be too far/small)
    min_size = int(min(frame_w, frame_h) * 0.22)  # ~105px for 480p
    if bw < min_size or bh < min_size:
        return False, face, "Move closer to the camera", (0, 165, 255)

    # 2. Centering & Margin Validation (Must not touch outer border)
    margin = 30
    if bbox[0] < margin or bbox[1] < margin or bbox[2] > (frame_w - margin) or bbox[3] > (frame_h - margin):
        return False, face, "Center your face in frame", (0, 165, 255)

    return True, face, "READY - Press SPACE to Capture", (0, 255, 0)


def draw_hud(frame: np.ndarray,
             student_id: str,
             student_name: str,
             current_pose_idx: int,
             status_msg: str,
             status_color: tuple,
             active_face: Optional[Any],
             feedback_state: Optional[Dict[str, Any]],
             is_complete: bool,
             active_cam_idx: int,
             fps: float) -> np.ndarray:
    """
    Renders a professional, readable OpenCV HUD for student registration.
    """
    out = frame.copy()
    h, w = out.shape[:2]

    # Draw Face Bounding Box with Corner Accents
    if active_face is not None:
        bbox = active_face.bbox.astype(int)
        x1, y1 = max(0, bbox[0]), max(0, bbox[1])
        x2, y2 = min(w - 1, bbox[2]), min(h - 1, bbox[3])

        cv2.rectangle(out, (x1, y1), (x2, y2), status_color, 2)
        corner_len = min(22, (x2 - x1) // 4)
        cv2.line(out, (x1, y1), (x1 + corner_len, y1), status_color, 4)
        cv2.line(out, (x1, y1), (x1, y1 + corner_len), status_color, 4)
        cv2.line(out, (x2, y1), (x2 - corner_len, y1), status_color, 4)
        cv2.line(out, (x2, y1), (x2, y1 + corner_len), status_color, 4)
        cv2.line(out, (x1, y2), (x1 + corner_len, y2), status_color, 4)
        cv2.line(out, (x1, y2), (x1, y2 - corner_len), status_color, 4)
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), status_color, 4)
        cv2.line(out, (x2, y2), (x2 - corner_len, y2), status_color, 4)

    # 1. TOP HEADER BANNER
    header_h = 100
    cv2.rectangle(out, (0, 0), (w, header_h), (20, 20, 20), -1)
    cv2.line(out, (0, header_h), (w, header_h), (60, 60, 60), 2)

    # Student Info
    cv2.putText(out, f"STUDENT ENROLLMENT | {student_name.upper()} (ID: {student_id})", 
                (15, 26), config.FONT, 0.58, (255, 255, 255), 2, cv2.LINE_AA)

    if not is_complete:
        curr_pose = POSES[current_pose_idx]
        curr_instr = POSE_INSTRUCTIONS[current_pose_idx]

        # Pose & Progress Bar
        pose_text = f"POSE [{current_pose_idx + 1}/5]: >>> {curr_pose} <<< ({curr_instr})"
        cv2.putText(out, pose_text, (15, 56), config.FONT, 0.60, (0, 255, 255), 2, cv2.LINE_AA)

        # Status badge
        cv2.putText(out, f"Status: {status_msg}", (15, 84), config.FONT, 0.50, status_color, 2, cv2.LINE_AA)
    else:
        cv2.putText(out, "✓ ALL 5 SAMPLES CAPTURED SUCCESSFULLY!", (15, 56), config.FONT, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(out, "Biometric centroid saved to database. Press Q to exit.", (15, 84), config.FONT, 0.50, (200, 200, 200), 1, cv2.LINE_AA)

    # FPS Indicator
    cv2.putText(out, f"FPS: {fps:.1f}", (w - 110, 26), config.FONT, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    # 2. BOTTOM CONTROL BAR
    footer_h = 36
    cv2.rectangle(out, (0, h - footer_h), (w, h), (20, 20, 20), -1)
    cv2.line(out, (0, h - footer_h), (w, h - footer_h), (60, 60, 60), 1)

    control_text = f"[SPACE] Capture Pose  |  [Q / ESC] Cancel  |  Cam: #{active_cam_idx} (Irium/DirectShow)"
    if is_complete:
        control_text = "[Q] or [ESC] Close Registration  |  Registration Complete"
    cv2.putText(out, control_text, (15, h - 12), config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

    # 3. ON-SCREEN FEEDBACK TOAST / POPUP
    curr_time = time.time()
    if feedback_state and curr_time < feedback_state["until"]:
        fb_col = feedback_state["color"]
        box_w = 420
        box_h = 75
        bx1 = (w - box_w) // 2
        by1 = header_h + 15
        cv2.rectangle(out, (bx1, by1), (bx1 + box_w, by1 + box_h), (25, 25, 25), -1)
        cv2.rectangle(out, (bx1, by1), (bx1 + box_w, by1 + box_h), fb_col, 2)

        cv2.putText(out, feedback_state["title"], (bx1 + 15, by1 + 28), 
                    config.FONT, 0.60, fb_col, 2, cv2.LINE_AA)
        cv2.putText(out, feedback_state["subtitle"], (bx1 + 15, by1 + 56), 
                    config.FONT, 0.48, (255, 255, 255), 1, cv2.LINE_AA)

    return out


def draw_completion_screen(student_id: str, student_name: str, saved_files: List[str]) -> np.ndarray:
    """
    Renders a dedicated summary card when registration finishes.
    """
    card = np.full((config.CAMERA_HEIGHT, config.CAMERA_WIDTH, 3), 25, dtype=np.uint8)
    
    # Title
    cv2.rectangle(card, (20, 20), (config.CAMERA_WIDTH - 20, 75), (40, 40, 40), -1)
    cv2.rectangle(card, (20, 20), (config.CAMERA_WIDTH - 20, 75), (0, 255, 0), 2)
    cv2.putText(card, "REGISTRATION COMPLETE", (40, 56), config.FONT, 0.75, (0, 255, 0), 2, cv2.LINE_AA)

    # Info
    cv2.putText(card, f"Student: {student_name}", (40, 115), config.FONT, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(card, f"Student ID: {student_id}", (40, 145), config.FONT, 0.58, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(card, "Samples Saved to Disk:", (40, 185), config.FONT, 0.58, (0, 255, 255), 1, cv2.LINE_AA)

    # Checklist
    for idx, (p, fn) in enumerate(zip(POSES, saved_files)):
        y_pos = 220 + (idx * 28)
        cv2.putText(card, f"  [x] Pose {idx + 1} ({p}): {fn}", (40, y_pos), 
                    config.FONT, 0.48, (0, 255, 0), 1, cv2.LINE_AA)

    cv2.putText(card, "Normalized ArcFace Biometric Centroid Stored in Database.", 
                (40, 380), config.FONT, 0.46, (255, 255, 255), 1, cv2.LINE_AA)

    cv2.rectangle(card, (20, config.CAMERA_HEIGHT - 55), (config.CAMERA_WIDTH - 20, config.CAMERA_HEIGHT - 20), (45, 45, 45), -1)
    cv2.putText(card, "Press 'Q' or 'ESC' to Close Window", (140, config.CAMERA_HEIGHT - 32), 
                config.FONT, 0.52, (0, 255, 255), 1, cv2.LINE_AA)

    return card


def register_student(student_id: Optional[str] = None,
                     roll_number: Optional[str] = None,
                     name: Optional[str] = None,
                     camera_index: Optional[int] = None,
                     auto_capture: bool = False):
    """
    Executes the interactive student enrollment process.
    """
    print("\n" + "=" * 70)
    print("      STUDENT BIOMETRIC ENROLLMENT / REGISTRATION")
    print("=" * 70)

    # 1. Collect Student Metadata via Terminal
    if not student_id:
        try:
            student_id = input("[?] Enter Student ID (e.g., 101): ").strip()
        except EOFError:
            student_id = "101"
    if not roll_number:
        try:
            roll_number = input("[?] Enter Roll Number (e.g., 23CS001): ").strip()
        except EOFError:
            roll_number = "23CS001"
    if not name:
        try:
            name = input("[?] Enter Full Name: ").strip()
        except EOFError:
            name = "Aman Rajbhar"

    student_id = str(student_id).strip()
    roll_number = str(roll_number).strip()
    name = str(name).strip()

    if not student_id or not name:
        print("[!] Error: Student ID and Name cannot be blank.")
        return False

    print(f"\n[*] Enrolling: {name} (ID: {student_id}, Roll: {roll_number})")

    # 2. Check if student already exists
    store = EmbeddingStore()
    existing = store.get_student(student_id)
    if existing:
        print(f"[!] Notice: Student ID {student_id} is already registered as '{existing['name']}'.")
        try:
            overwrite = input("[?] Overwrite existing biometrics for this ID? (y/N): ").strip().lower()
        except EOFError:
            overwrite = "y" if auto_capture else "n"
        if overwrite != "y":
            print("[*] Registration cancelled.")
            return False

    # 3. Open Webcam (DirectShow / Irium Webcam)
    cam_target = camera_index if camera_index is not None else config.CAMERA_INDEX
    print(f"\n[*] [1/2] Opening Camera Index {cam_target} (Irium / DirectShow)...")
    cap, active_cam_idx = open_camera(
        camera_index=camera_index,
        width=config.CAMERA_WIDTH,
        height=config.CAMERA_HEIGHT,
        use_dshow=config.USE_DIRECTSHOW
    )
    if cap is None:
        print(f"[!] Critical Error: Could not connect to camera.")
        return False

    # 4. Load AI Models ONCE
    print(f"[*] [2/2] Loading Face Detection & ArcFace Models (Det Size: {config.DETECTION_SIZE})...")
    detector = FaceDetector(det_size=config.DETECTION_SIZE)
    print("[+] System Ready! Webcam window opening...\n")

    student_img_dir = config.FACES_DIR / student_id
    student_img_dir.mkdir(parents=True, exist_ok=True)

    current_pose_idx = 0
    captured_embeddings = []
    saved_filenames = []
    
    feedback_state = None
    is_complete = False

    fps_smooth = 0.0
    prev_time = time.perf_counter()
    frame_idx = 0
    last_detected_faces = []

    print("=" * 70)
    print(f"[*] Required Poses: {len(POSES)}")
    for idx, (p, instr) in enumerate(zip(POSES, POSE_INSTRUCTIONS), 1):
        print(f"    {idx}. {p:<8} - {instr}")
    print("\n[*] Controls:")
    print("    [SPACE] - Capture current pose")
    print("    [Q]     - Exit / Cancel")
    print("=" * 70 + "\n")

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Failed to grab frame from camera.")
                break

            frame_idx += 1
            curr_time = time.perf_counter()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps_inst = 1.0 / dt
                fps_smooth = (0.92 * fps_smooth) + (0.08 * fps_inst) if fps_smooth > 0 else fps_inst

            h, w = frame.shape[:2]

            # If enrollment is complete, display completion screen
            if is_complete:
                comp_screen = draw_completion_screen(student_id, name, saved_filenames)
                cv2.imshow("Student Face Registration", comp_screen)
                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("[*] Registration window closed by user.")
                    break
                continue

            # Run detection at interval for high FPS
            if (frame_idx % config.DETECTION_INTERVAL == 0) or len(last_detected_faces) == 0:
                last_detected_faces = detector.detect_faces(frame)

            # Validate face quality
            is_valid, active_face, status_msg, status_col = validate_face_quality(
                last_detected_faces, w, h
            )

            # Render HUD
            display_frame = draw_hud(
                frame=frame,
                student_id=student_id,
                student_name=name,
                current_pose_idx=current_pose_idx,
                status_msg=status_msg,
                status_color=status_col,
                active_face=active_face,
                feedback_state=feedback_state,
                is_complete=is_complete,
                active_cam_idx=active_cam_idx,
                fps=fps_smooth
            )

            cv2.imshow("Student Face Registration", display_frame)

            key = cv2.waitKey(1) & 0xFF

            # SPACE KEY: TRIGGER CAPTURE
            if key == 32:  # Spacebar
                current_pose_name = POSES[current_pose_idx]
                current_filename = POSE_FILES[current_pose_idx]

                # Immediate validation check
                if not is_valid or active_face is None:
                    print(f"[!] Cannot capture: {status_msg}")
                    feedback_state = {
                        "color": (0, 0, 255),
                        "title": "✗ CAPTURE REJECTED",
                        "subtitle": status_msg,
                        "until": time.time() + 1.2
                    }
                    continue

                # Run exact ArcFace embedding extraction on high-res capture frame
                print(f"[*] Processing capture for pose '{current_pose_name}'...")
                precise_faces = detector.detect_faces(frame)
                if len(precise_faces) != 1 or not hasattr(precise_faces[0], 'embedding') or precise_faces[0].embedding is None:
                    print("[!] Warning: Could not extract deep embedding on current frame. Please hold steady and try again.")
                    feedback_state = {
                        "color": (0, 0, 255),
                        "title": "✗ EMBEDDING FAILED",
                        "subtitle": "Hold steady and press SPACE again",
                        "until": time.time() + 1.2
                    }
                    continue

                capture_face = precise_faces[0]
                save_path = student_img_dir / current_filename

                # Crop face with padding for clean storage
                bbox = capture_face.bbox.astype(int)
                pad_x = int((bbox[2] - bbox[0]) * 0.20)
                pad_y = int((bbox[3] - bbox[1]) * 0.20)
                crop_x1 = max(0, bbox[0] - pad_x)
                crop_y1 = max(0, bbox[1] - pad_y)
                crop_x2 = min(w, bbox[2] + pad_x)
                crop_y2 = min(h, bbox[3] + pad_y)
                face_crop = frame[crop_y1:crop_y2, crop_x1:crop_x2]

                # 1. Save Image to Disk
                cv2.imwrite(str(save_path), face_crop)

                # 2. Verify File Actually Exists on Disk
                if not save_path.exists() or save_path.stat().st_size == 0:
                    print(f"[!] Error: Physical disk save failed for {save_path.name}")
                    feedback_state = {
                        "color": (0, 0, 255),
                        "title": "✗ SAVE FAILED",
                        "subtitle": f"Could not write {current_filename}",
                        "until": time.time() + 1.5
                    }
                    continue

                # 3. Store Valid Embedding
                captured_embeddings.append(capture_face.embedding)
                saved_filenames.append(current_filename)
                current_pose_idx += 1

                print(f"[+] [SAVED] Sample {current_pose_idx}/5: '{current_pose_name}' -> {save_path}")

                # 4. Provide Visual Feedback
                next_pose_str = POSES[current_pose_idx] if current_pose_idx < len(POSES) else "DONE"
                feedback_state = {
                    "color": (0, 255, 0),
                    "title": f"✓ CAPTURE SUCCESSFUL ({current_pose_name})",
                    "subtitle": f"Saved {current_pose_idx}/5 | Next: {next_pose_str}",
                    "until": time.time() + 1.3
                }

                # Quick white capture flash
                flash = np.full_like(display_frame, 240)
                cv2.imshow("Student Face Registration", flash)
                cv2.waitKey(40)

                # 5. Check if all 5 poses completed
                if current_pose_idx == len(POSES):
                    print("\n" + "=" * 70)
                    print("[*] All 5 samples captured. Computing normalized ArcFace centroid...")
                    reg_time = datetime.now().isoformat()
                    store.add_student(
                        student_id=student_id,
                        roll_number=roll_number,
                        name=name,
                        sample_embeddings=captured_embeddings,
                        registered_at=reg_time
                    )
                    print(f"[SUCCESS] Student '{name}' (ID: {student_id}) Successfully Registered!")
                    print(f"[*] Images directory: {student_img_dir}")
                    print("=" * 70 + "\n")
                    is_complete = True

            elif key in [ord('q'), ord('Q'), 27]:
                print("[*] Registration cancelled by user.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released.")

    return is_complete


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enroll student with 5-angle face biometrics")
    parser.add_argument("--id", type=str, default=None, help="Student ID")
    parser.add_argument("--roll", type=str, default=None, help="Student Roll Number")
    parser.add_argument("--name", type=str, default=None, help="Student Full Name")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (e.g. 1 for Irium)")
    parser.add_argument("--auto", action="store_true", help="Auto-capture for testing")
    args = parser.parse_args()

    register_student(
        student_id=args.id,
        roll_number=args.roll,
        name=args.name,
        camera_index=args.camera,
        auto_capture=args.auto
    )
