"""
Main Application: Real-Time Face Recognition Smart Attendance System with Liveness Detection.

Key AI/CV Capabilities:
1. Camera Index 0 (Irium / DirectShow) instant startup.
2. RetinaFace detection at configurable intervals for 30+ FPS video smoothness.
3. ArcFace recognition against registered student biometric centroids.
4. Active Eye Blink Liveness with live EAR telemetry:
   - Shows EAR, Left/Right components, Min dip observed, Eye State (OPEN/CLOSED), and Blink Status.
5. Calibrated decision threshold (default: EAR_THRESHOLD = 0.35).
6. Press [B] to toggle detailed eye landmark geometry & contour mesh.
7. Clear distinction between 'LIVENESS: PASS' and 'ALREADY MARKED TODAY'.
8. Clean exit on [Q], [ESC], and Ctrl+C without tracebacks.
"""

import sys
import time
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import cv2
import numpy as np

# Suppress library warnings
warnings.filterwarnings("ignore")

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.camera_utils import open_camera
from app.face_detection.detector import FaceDetector
from app.face_recognition.recognizer import FaceRecognizer
from app.face_recognition.embedding_store import EmbeddingStore
from app.liveness.blink_detector import BlinkLivenessDetector, FaceLivenessTracker
from app.attendance.attendance_manager import AttendanceManager


class AttendanceApp:
    """
    Real-Time Attendance Engine with explicit Liveness Challenge-Response State Machine.
    """

    def __init__(self, 
                 camera_index: Optional[int] = None,
                 threshold: float = config.RECOGNITION_THRESHOLD,
                 ear_threshold: float = config.EAR_THRESHOLD,
                 det_interval: int = config.DETECTION_INTERVAL,
                 rec_interval: int = config.RECOGNITION_INTERVAL,
                 require_liveness: bool = True):
        self.camera_index = config.CAMERA_INDEX if camera_index is None else camera_index
        self.threshold = threshold
        self.ear_threshold = ear_threshold
        self.det_interval = max(1, det_interval)
        self.rec_interval = max(1, rec_interval)
        self.require_liveness = require_liveness
        self.show_debug_landmarks = False

        print("\n" + "=" * 70)
        print("   REAL-TIME FACE RECOGNITION SMART ATTENDANCE SYSTEM")
        print("=" * 70)

        # 1. Open Webcam First (DirectShow on Camera Index 0)
        print(f"[*] [1/3] Opening Camera Index {self.camera_index} (DirectShow)...")
        self.cap, self.active_cam_idx = open_camera(
            camera_index=self.camera_index,
            width=config.CAMERA_WIDTH,
            height=config.CAMERA_HEIGHT,
            use_dshow=config.USE_DIRECTSHOW
        )
        if self.cap is None:
            raise RuntimeError(f"Could not open camera at index {self.camera_index}. Run 'python scripts/list_cameras.py' to check.")

        # 2. Load Models (ONCE at startup)
        print(f"[*] [2/3] Loading AI Models (Detection: {config.DETECTION_SIZE}, EAR Threshold: {self.ear_threshold:.2f})...")
        self.detector = FaceDetector(det_size=config.DETECTION_SIZE)
        self.store = EmbeddingStore()
        self.recognizer = FaceRecognizer(store=self.store, threshold=self.threshold)
        self.liveness_detector = BlinkLivenessDetector(
            ear_thresh=self.ear_threshold,
            min_closed_frames=config.BLINK_MIN_FRAMES,
            timeout_sec=config.LIVENESS_TIMEOUT_SECONDS,
            cooldown_sec=config.BLINK_COOLDOWN
        )
        self.attendance_mgr = AttendanceManager()

        # 3. System Ready
        reg_count = len(self.store.list_students())
        print(f"[+] [3/3] System Ready! Loaded {reg_count} registered student identity/identities.\n")

        self.tracked_faces: Dict[int, Dict[str, Any]] = {}
        self.next_track_id = 0
        self.toasts: List[Tuple[str, tuple, float]] = []

    def _match_or_create_tracks(self, faces: list) -> List[int]:
        """Associates detected faces with historical track IDs."""
        assigned_ids = []
        curr_centroids = []
        for f in faces:
            b = f.bbox.astype(int)
            curr_centroids.append(((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0))

        if not self.tracked_faces:
            for idx, f in enumerate(faces):
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracked_faces[tid] = {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps if hasattr(f, 'kps') else None,
                    "lm106": f.landmark_2d_106 if hasattr(f, 'landmark_2d_106') else None,
                    "centroid": curr_centroids[idx],
                    "last_seen": time.time(),
                    "recognition": None,
                    "attendance_marked_this_session": False
                }
                assigned_ids.append(tid)
            return assigned_ids

        for idx, (f, c) in enumerate(zip(faces, curr_centroids)):
            best_id = None
            best_dist = 130.0

            for tid, tinfo in list(self.tracked_faces.items()):
                prev_c = tinfo["centroid"]
                dist = np.hypot(c[0] - prev_c[0], c[1] - prev_c[1])
                if dist < best_dist:
                    best_dist = dist
                    best_id = tid

            if best_id is not None:
                self.tracked_faces[best_id]["bbox"] = f.bbox.astype(int)
                self.tracked_faces[best_id]["kps"] = f.kps if hasattr(f, 'kps') else None
                self.tracked_faces[best_id]["lm106"] = f.landmark_2d_106 if hasattr(f, 'landmark_2d_106') else None
                self.tracked_faces[best_id]["centroid"] = c
                self.tracked_faces[best_id]["last_seen"] = time.time()
                assigned_ids.append(best_id)
            else:
                tid = self.next_track_id
                self.next_track_id += 1
                self.tracked_faces[tid] = {
                    "bbox": f.bbox.astype(int),
                    "kps": f.kps if hasattr(f, 'kps') else None,
                    "lm106": f.landmark_2d_106 if hasattr(f, 'landmark_2d_106') else None,
                    "centroid": c,
                    "last_seen": time.time(),
                    "recognition": None,
                    "attendance_marked_this_session": False
                }
                assigned_ids.append(tid)

        # Reset tracks when face leaves the camera (missing > 1.8s)
        curr_t = time.time()
        for tid in list(self.tracked_faces.keys()):
            if curr_t - self.tracked_faces[tid]["last_seen"] > 1.8:
                self.liveness_detector.reset_tracker(str(tid))
                del self.tracked_faces[tid]

        return assigned_ids

    def add_toast(self, message: str, color: tuple = (0, 255, 0), duration: float = 3.0):
        """Adds temporary on-screen toast banner."""
        self.toasts.append((message, color, time.time() + duration))

    def run(self, test_frames: int = 0):
        """
        Main execution loop.
        """
        print("[+] Video stream active.")
        print("[*] Keyboard Controls:")
        print("    [Q] or [ESC] - Quit Application")
        print("    [B]         - Toggle Eye Landmarks & Blink Debug Mesh")
        print("    [R]         - Reload Student Database")
        print("    [C]         - Clear Trackers & Reset Liveness")
        print("=" * 70 + "\n")

        fps_smooth = 0.0
        prev_time = time.perf_counter()
        frame_idx = 0
        latest_infer_ms = 0.0

        try:
            while True:
                ret, frame = self.cap.read()
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

                # ==========================================
                # INFERENCE PIPELINE
                # ==========================================
                should_run_detection = (frame_idx % self.det_interval == 0) or (len(self.tracked_faces) == 0)
                should_run_recognition = (frame_idx % self.rec_interval == 0)

                if should_run_detection:
                    t_infer_start = time.perf_counter()
                    faces = self.detector.detect_faces(frame)
                    assigned_track_ids = self._match_or_create_tracks(faces)

                    for face, tid in zip(faces, assigned_track_ids):
                        t_info = self.tracked_faces[tid]
                        track_key = str(tid)

                        # Step 1: ArcFace Recognition
                        if should_run_recognition or t_info["recognition"] is None:
                            if hasattr(face, 'embedding') and face.embedding is not None:
                                rec_res = self.recognizer.recognize_face(face.embedding)
                            else:
                                rec_res = {
                                    "is_recognized": False,
                                    "student_id": "Unknown",
                                    "name": "Unknown",
                                    "roll_number": "",
                                    "confidence": 0.0,
                                    "threshold": self.threshold
                                }
                            t_info["recognition"] = rec_res

                        rec = t_info["recognition"]
                        is_rec = rec["is_recognized"]
                        sid = rec["student_id"]
                        sname = rec["name"]
                        sroll = rec["roll_number"]
                        sconf = rec["confidence"]

                        # Step 2: State Machine Evaluation
                        if not is_rec:
                            t_info["color"] = config.COLOR_UNKNOWN
                            t_info["badge1"] = "UNKNOWN PERSON"
                            t_info["badge2"] = f"Match: {sconf:.2f} (< {self.threshold:.2f})"
                            t_info["badge3"] = ""
                        else:
                            # Recognized Student Identity
                            if self.require_liveness:
                                liveness_info = self.liveness_detector.process_liveness(face, track_key)
                                l_state = liveness_info["state"]
                                time_left = liveness_info["time_remaining"]
                                avg_ear = liveness_info["avg_ear"]
                                left_ear = liveness_info["left_ear"]
                                right_ear = liveness_info["right_ear"]
                                eye_state = liveness_info["eye_state"]
                                blink_status = liveness_info["blink_status"]
                                min_dip = liveness_info.get("min_ear_seen", 0.0)
                            else:
                                l_state = FaceLivenessTracker.STATE_LIVENESS_PASSED
                                time_left = 0.0
                                avg_ear, left_ear, right_ear = 0.35, 0.35, 0.35
                                eye_state, blink_status = "OPEN", "BYPASSED"
                                min_dip = 0.35

                            # Real-time EAR Telemetry Badge
                            ear_debug_str = f"EAR: {avg_ear:.2f} (L:{left_ear:.2f}, R:{right_ear:.2f} | Min:{min_dip:.2f}) | EYE: {eye_state}"
                            t_info["badge3"] = ear_debug_str

                            # --- STATE 3A: LIVENESS CHALLENGE ACTIVE ---
                            if l_state == FaceLivenessTracker.STATE_CHALLENGE_ACTIVE:
                                sec_display = max(1, int(np.ceil(time_left)))
                                t_info["color"] = config.COLOR_CHALLENGE
                                t_info["badge1"] = f"{sname.upper()} ({sid})"
                                t_info["badge2"] = f"PLEASE BLINK ({sec_display}s)"

                            # --- STATE 3B: LIVENESS FAILED (6s Timeout Expired) ---
                            elif l_state == FaceLivenessTracker.STATE_LIVENESS_FAILED:
                                t_info["color"] = config.COLOR_FAKE
                                t_info["badge1"] = f"{sname.upper()} ({sid})"
                                t_info["badge2"] = "LIVENESS: FAILED | ATTENDANCE BLOCKED"

                            # --- STATE 3C: LIVENESS PASSED (Blink Confirmed) ---
                            elif l_state == FaceLivenessTracker.STATE_LIVENESS_PASSED:
                                already_marked = self.attendance_mgr.is_already_marked(sid)
                                
                                if already_marked:
                                    t_info["color"] = config.COLOR_ALREADY_MARKED
                                    t_info["badge1"] = f"{sname.upper()} ({sid})"
                                    t_info["badge2"] = "LIVENESS: PASS | ALREADY MARKED TODAY"
                                else:
                                    # Mark attendance once
                                    if not t_info.get("attendance_marked_this_session", False):
                                        log_res = self.attendance_mgr.mark_attendance(
                                            student_id=sid,
                                            roll_number=sroll,
                                            name=sname,
                                            confidence=sconf,
                                            liveness_status="passed"
                                        )
                                        t_info["attendance_marked_this_session"] = True
                                        self.add_toast(f"PRESENT: {sname} (ID: {sid})", config.COLOR_RECOGNIZED)

                                    t_info["color"] = config.COLOR_RECOGNIZED
                                    t_info["badge1"] = f"{sname.upper()} ({sid})"
                                    t_info["badge2"] = "LIVENESS: PASS | ATTENDANCE MARKED!"

                    latest_infer_ms = (time.perf_counter() - t_infer_start) * 1000

                # ==========================================
                # RENDER OPENCV HUD
                # ==========================================
                curr_t = time.time()
                active_render_data = []
                for tid, t_info in self.tracked_faces.items():
                    if curr_t - t_info["last_seen"] < 1.0:
                        active_render_data.append(t_info)

                display_frame = self._render_hud(frame, active_render_data, fps_smooth, latest_infer_ms)

                cv2.imshow("Smart Attendance System - AI / CV", display_frame)

                if test_frames > 0 and frame_idx >= test_frames:
                    print(f"[+] Processed {test_frames} frames successfully.")
                    break

                key = cv2.waitKey(1) & 0xFF
                if key in [ord('q'), ord('Q'), 27]:
                    print("[*] Exit key pressed. Closing application.")
                    break
                elif key in [ord('b'), ord('B')]:
                    self.show_debug_landmarks = not self.show_debug_landmarks
                    state_str = "ENABLED" if self.show_debug_landmarks else "DISABLED"
                    print(f"[*] Detailed Eye Landmark Overlay: {state_str}")
                    self.add_toast(f"Eye Mesh Debug: {state_str}", (0, 255, 255), 2.0)
                elif key in [ord('r'), ord('R')]:
                    print("[*] Reloading student database from disk...")
                    self.recognizer.reload()
                    self.add_toast("Database Reloaded", (255, 255, 0), 2.0)
                elif key in [ord('c'), ord('C')]:
                    self.liveness_detector.trackers.clear()
                    self.tracked_faces.clear()
                    self.add_toast("Liveness Trackers Reset", (255, 255, 0), 2.0)

        except KeyboardInterrupt:
            print("\n[*] Application interrupted by user (Ctrl+C). Closing cleanly...")
        finally:
            self.cap.release()
            cv2.destroyAllWindows()
            print(f"[+] Camera released safely. Average throughput: {fps_smooth:.1f} FPS.")

    def _render_hud(self, 
                    frame: np.ndarray, 
                    render_data: list, 
                    fps: float, 
                    latency_ms: float) -> np.ndarray:
        """
        Renders bounding boxes, badges, landmarks, toast alerts, and status bar.
        """
        out = frame.copy()
        h, w = out.shape[:2]

        for item in render_data:
            bbox = item.get("bbox")
            if bbox is None:
                continue

            color = item.get("color", config.COLOR_UNKNOWN)
            b1 = item.get("badge1", "DETECTING...")
            b2 = item.get("badge2", "")
            b3 = item.get("badge3", "")

            x1, y1 = max(0, bbox[0]), max(0, bbox[1])
            x2, y2 = min(w - 1, bbox[2]), min(h - 1, bbox[3])

            # Bounding box with corner accents
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            corner_len = min(18, (x2 - x1) // 4)
            cv2.line(out, (x1, y1), (x1 + corner_len, y1), color, 3)
            cv2.line(out, (x1, y1), (x1, y1 + corner_len), color, 3)
            cv2.line(out, (x2, y1), (x2 - corner_len, y1), color, 3)
            cv2.line(out, (x2, y1), (x2, y1 + corner_len), color, 3)
            cv2.line(out, (x1, y2), (x1 + corner_len, y2), color, 3)
            cv2.line(out, (x1, y2), (x1, y2 - corner_len), color, 3)
            cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, 3)
            cv2.line(out, (x2, y2), (x2 - corner_len, y2), color, 3)

            # Three-tier Info Badge above box (Name, Challenge/Status, Live EAR Metrics)
            badge_h = 58 if b3 else 42
            badge_y1 = max(0, y1 - badge_h)
            max_len = max(len(b1), len(b2), len(b3))
            badge_w = max(230, max_len * 7 + 25)
            
            cv2.rectangle(out, (x1, badge_y1), (x1 + badge_w, y1), (20, 20, 20), -1)
            cv2.rectangle(out, (x1, badge_y1), (x1 + badge_w, y1), color, 1)

            cv2.putText(out, b1, (x1 + 6, badge_y1 + 17), config.FONT, 0.48, config.COLOR_WHITE, 2, cv2.LINE_AA)
            cv2.putText(out, b2, (x1 + 6, badge_y1 + 34), config.FONT, 0.40, color, 1, cv2.LINE_AA)
            if b3:
                cv2.putText(out, b3, (x1 + 6, badge_y1 + 50), config.FONT, 0.35, (0, 255, 255), 1, cv2.LINE_AA)

            # Draw Eye Geometry & Contour Mesh if enabled with [B]
            if self.show_debug_landmarks and item.get("lm106") is not None:
                lm106 = item["lm106"]
                # Left Eye (35..42)
                for pt_idx in [35, 36, 37, 38, 39, 40, 41, 42]:
                    pt = lm106[pt_idx]
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
                cv2.line(out, (int(lm106[35][0]), int(lm106[35][1])), (int(lm106[39][0]), int(lm106[39][1])), (0, 255, 0), 1)
                cv2.line(out, (int(lm106[37][0]), int(lm106[37][1])), (int(lm106[41][0]), int(lm106[41][1])), (255, 0, 0), 1)

                # Right Eye (89..96)
                for pt_idx in [89, 90, 91, 92, 93, 94, 95, 96]:
                    pt = lm106[pt_idx]
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)
                cv2.line(out, (int(lm106[89][0]), int(lm106[89][1])), (int(lm106[93][0]), int(lm106[93][1])), (0, 255, 0), 1)
                cv2.line(out, (int(lm106[91][0]), int(lm106[91][1])), (int(lm106[95][0]), int(lm106[95][1])), (255, 0, 0), 1)
            elif item.get("kps") is not None:
                for pt in item["kps"]:
                    cv2.circle(out, (int(pt[0]), int(pt[1])), 2, (0, 255, 255), -1)

        # Top Diagnostic Bar
        cv2.rectangle(out, (0, 0), (w, 36), (20, 20, 20), -1)
        cv2.line(out, (0, 36), (w, 36), (70, 70, 70), 1)
        
        cv2.putText(out, "AI ATTENDANCE SYSTEM", (15, 24), config.FONT, 0.58, (0, 255, 255), 2, cv2.LINE_AA)
        
        fps_color = (0, 255, 0) if fps >= 20 else ((0, 200, 255) if fps >= 12 else (0, 0, 255))
        stat_text = f"FPS: {fps:.1f}  |  Infer: {latency_ms:.1f}ms  |  Cam: #{self.active_cam_idx}"
        (tw, _), _ = cv2.getTextSize(stat_text, config.FONT, 0.48, 1)
        cv2.putText(out, stat_text, (w - tw - 15, 24), config.FONT, 0.48, fps_color, 1, cv2.LINE_AA)

        # Bottom Status Bar
        cv2.rectangle(out, (0, h - 32), (w, h), (20, 20, 20), -1)
        cv2.line(out, (0, h - 32), (w, h - 32), (70, 70, 70), 1)

        today_count = len(self.attendance_mgr.get_today_records())
        reg_count = len(self.store.list_students())
        footer_text = f"Today Present: {today_count}  |  Enrolled: {reg_count}  |  [Q] Quit  [B] Landmarks  [C] Reset"
        cv2.putText(out, footer_text, (15, h - 11), config.FONT, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

        # Render Active Toast Notifications
        curr_t = time.time()
        active_toasts = [t for t in self.toasts if t[2] > curr_t]
        self.toasts = active_toasts

        for idx, (tmsg, tcol, _) in enumerate(active_toasts[-2:]):
            toast_y = 52 + (idx * 38)
            (tw, th), _ = cv2.getTextSize(tmsg, config.FONT, 0.52, 2)
            toast_x = (w - tw) // 2
            cv2.rectangle(out, (toast_x - 12, toast_y - 18), (toast_x + tw + 12, toast_y + 8), (30, 30, 30), -1)
            cv2.rectangle(out, (toast_x - 12, toast_y - 18), (toast_x + tw + 12, toast_y + 8), tcol, 2)
            cv2.putText(out, tmsg, (toast_x, toast_y), config.FONT, 0.52, tcol, 2, cv2.LINE_AA)

        return out


def main():
    parser = argparse.ArgumentParser(description="Run Real-Time Face Attendance System with Liveness")
    parser.add_argument("--camera", type=int, default=None, help="Camera index (default: 0)")
    parser.add_argument("--thresh", type=float, default=config.RECOGNITION_THRESHOLD, help="Cosine threshold")
    parser.add_argument("--ear-thresh", type=float, default=config.EAR_THRESHOLD, help="EAR blink threshold (default: 0.35)")
    parser.add_argument("--det-interval", type=int, default=config.DETECTION_INTERVAL, help="Detection interval (default: 2)")
    parser.add_argument("--rec-interval", type=int, default=config.RECOGNITION_INTERVAL, help="Recognition interval (default: 5)")
    parser.add_argument("--no-liveness", action="store_true", help="Bypass liveness blink challenge")
    parser.add_argument("--test-frames", type=int, default=0, help="Run for N frames and exit (testing)")
    args = parser.parse_args()

    app = AttendanceApp(
        camera_index=args.camera,
        threshold=args.thresh,
        ear_threshold=args.ear_thresh,
        det_interval=args.det_interval,
        rec_interval=args.rec_interval,
        require_liveness=not args.no_liveness
    )
    app.run(test_frames=args.test_frames)


if __name__ == "__main__":
    main()
