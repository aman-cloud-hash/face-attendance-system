"""
Script: test_recognition.py
Purpose: Real-time Face Recognition Test on Webcam.

Demonstrates:
- ArcFace feature extraction
- Cosine similarity matching against registered students
- Decision threshold boundary evaluation (Default: 0.45)
- Real-time Multi-face recognition & HUD visualization
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.camera_utils import open_camera
from app.face_detection.detector import FaceDetector
from app.face_recognition.recognizer import FaceRecognizer
from app.face_recognition.embedding_store import EmbeddingStore


def run_recognition_test(camera_index: int = config.CAMERA_INDEX,
                         threshold: float = config.RECOGNITION_THRESHOLD,
                         test_frames: int = 0):
    """
    Runs real-time face recognition test.
    """
    print("=" * 65)
    print("      PHASE 5: ARCFACE FACE RECOGNITION TEST")
    print("=" * 65)
    print(f"[*] Similarity Decision Threshold: {threshold}")

    store = EmbeddingStore()
    students = store.list_students()
    print(f"[*] Registered identities loaded in database: {len(students)}")
    for s in students:
        print(f"    - ID: {s['student_id']} | Roll: {s['roll_number']} | Name: {s['name']}")

    if not students:
        print("\n[!] Notice: Database is currently empty.")
        print("[*] Faces detected will be labeled as 'Unknown'. Run register_student.py to enroll.")

    detector = FaceDetector()
    recognizer = FaceRecognizer(store=store, threshold=threshold)

    print(f"\n[*] Opening camera (Configured Index: {camera_index if camera_index is not None else config.CAMERA_INDEX})...")
    cap, active_cam_idx = open_camera(camera_index=camera_index)

    if cap is None:
        print(f"[!] Warning: Could not open camera.")
        print("[*] Running synthetic validation check...")
        dummy_emb = np.random.randn(512).astype(np.float32)
        res = recognizer.recognize_face(dummy_emb)
        print(f"[+] Recognition pipeline checked successfully on probe vector: Recognized={res['is_recognized']}, Score={res['confidence']:.3f}")
        return

    print(f"[+] Camera Index {active_cam_idx} running. Press 'Q' or 'ESC' to exit.\n")

    prev_time = time.time()
    fps_smooth = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            # FPS calculation
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                fps = 1.0 / dt
                fps_smooth = (0.9 * fps_smooth) + (0.1 * fps) if fps_smooth > 0 else fps

            # 1. Detect faces
            start_infer = time.time()
            faces = detector.detect_faces(frame)
            
            # 2. Recognize faces
            rec_results = recognizer.recognize_faces(faces)
            infer_time_ms = (time.time() - start_infer) * 1000

            display_frame = frame.copy()

            # Render annotations
            for rec in rec_results:
                face = rec["face"]
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox[0], bbox[1], bbox[2], bbox[3]
                
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w - 1, x2), min(h - 1, y2)

                is_rec = rec["is_recognized"]
                name = rec["name"]
                sid = rec["student_id"]
                conf = rec["confidence"]

                box_color = config.COLOR_RECOGNIZED if is_rec else config.COLOR_UNKNOWN

                # Bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                # Label text
                if is_rec:
                    line1 = f"{name.upper()}"
                    line2 = f"ID:{sid} | Match:{conf:.2f}"
                else:
                    line1 = "UNKNOWN"
                    line2 = f"Match:{conf:.2f}"

                # Draw text badge
                cv2.rectangle(display_frame, (x1, max(0, y1 - 42)), (x1 + 180, y1), box_color, -1)
                cv2.putText(display_frame, line1, (x1 + 5, max(15, y1 - 24)), 
                            config.FONT, 0.55, config.COLOR_WHITE, 2, cv2.LINE_AA)
                cv2.putText(display_frame, line2, (x1 + 5, max(30, y1 - 6)), 
                            config.FONT, 0.45, config.COLOR_WHITE, 1, cv2.LINE_AA)

            # HUD Banner
            cv2.putText(display_frame, f"FPS: {fps_smooth:.1f} | Inference: {infer_time_ms:.1f}ms", 
                        (15, 30), config.FONT, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(display_frame, f"Faces: {len(faces)} | Threshold: {threshold:.2f}", 
                        (15, 60), config.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            cv2.imshow("ArcFace Face Recognition - Test", display_frame)

            frame_count += 1
            if test_frames > 0 and frame_count >= test_frames:
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test ArcFace face recognition")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    parser.add_argument("--thresh", type=float, default=config.RECOGNITION_THRESHOLD, help="Cosine threshold")
    parser.add_argument("--test-frames", type=int, default=0, help="Exit after N frames")
    args = parser.parse_args()

    run_recognition_test(camera_index=args.camera, threshold=args.thresh, test_frames=args.test_frames)
