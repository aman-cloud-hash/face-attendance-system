"""
Script: test_detection.py
Purpose: Test real-time face detection using InsightFace (RetinaFace) with webcam.
Demonstrates:
- Multi-face detection
- RetinaFace landmark visualization
- Real-time FPS calculation
- Camera stream handling
"""

import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.camera_utils import open_camera
from app.face_detection.detector import FaceDetector


def run_detection_test(camera_index: int = config.CAMERA_INDEX, 
                       det_thresh: float = config.DETECTION_THRESHOLD,
                       test_frames: int = 0):
    """
    Runs real-time face detection on webcam feed.
    
    Args:
        camera_index: Index of the camera device (default 0)
        det_thresh: Confidence threshold for RetinaFace
        test_frames: If > 0, exits after processing N frames (useful for automated testing)
    """
    print("=" * 60)
    print("  PHASE 2: RETINAFACE DETECTION TEST")
    print("=" * 60)
    print(f"[*] Initializing Face Detector (Model: {config.INSIGHTFACE_MODEL_NAME})...")
    
    detector = FaceDetector(det_thresh=det_thresh)

    print(f"[*] Opening webcam (Configured Index: {camera_index if camera_index is not None else config.CAMERA_INDEX})...")
    cap, active_cam_idx = open_camera(camera_index=camera_index)

    if cap is None:
        print(f"[!] Warning: Could not open camera.")
        print("[*] Generating synthetic test canvas to verify detector pipeline...")
        test_img = np.zeros((config.FRAME_HEIGHT, config.FRAME_WIDTH, 3), dtype=np.uint8)
        cv2.ellipse(test_img, (320, 240), (100, 140), 0, 0, 360, (200, 180, 170), -1)
        faces = detector.detect_faces(test_img)
        print(f"[+] Pipeline check complete: Model executed successfully. Detected faces: {len(faces)}")
        return

    print(f"[+] Camera Index {active_cam_idx} opened successfully!")
    print("[*] Press 'Q' or 'ESC' to exit the test window.")

    prev_time = time.time()
    fps_smooth = 0.0
    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                print("[!] Error: Failed to capture image from camera.")
                break

            # Calculate FPS
            curr_time = time.time()
            dt = curr_time - prev_time
            prev_time = curr_time
            if dt > 0:
                current_fps = 1.0 / dt
                fps_smooth = (0.9 * fps_smooth) + (0.1 * current_fps) if fps_smooth > 0 else current_fps

            # Detect faces
            start_infer = time.time()
            faces = detector.detect_faces(frame)
            infer_ms = (time.time() - start_infer) * 1000

            # Draw detections
            annotated_frame = detector.draw_faces(frame, faces, draw_landmarks=True)

            # Draw HUD Overlay
            hud_bg = np.zeros((60, annotated_frame.shape[1], 3), dtype=np.uint8)
            cv2.putText(annotated_frame, f"FPS: {fps_smooth:.1f} | Inference: {infer_ms:.1f}ms", 
                        (15, 30), config.FONT, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(annotated_frame, f"Faces Detected: {len(faces)} | Press Q to Exit", 
                        (15, 60), config.FONT, 0.55, (255, 255, 255), 1, cv2.LINE_AA)

            # Show frame
            cv2.imshow("InsightFace RetinaFace Detector - Test", annotated_frame)

            frame_count += 1
            if test_frames > 0 and frame_count >= test_frames:
                print(f"[+] Automated test completed {test_frames} frames successfully.")
                break

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == ord('Q') or key == 27:
                print("[*] Exit key pressed. Closing test.")
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("[+] Camera released and windows destroyed.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test InsightFace RetinaFace face detector")
    parser.add_argument("--camera", type=int, default=config.CAMERA_INDEX, help="Camera index")
    parser.add_argument("--thresh", type=float, default=config.DETECTION_THRESHOLD, help="Detection threshold")
    parser.add_argument("--test-frames", type=int, default=0, help="Exit after N frames (for testing)")
    args = parser.parse_args()

    run_detection_test(camera_index=args.camera, det_thresh=args.thresh, test_frames=args.test_frames)
