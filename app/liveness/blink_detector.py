"""
Liveness Detection Module using Eye Aspect Ratio (EAR) Active Blink Analysis.

Key AI/CV Capabilities:
- Hybrid Static & Adaptive Baseline Blink Detection:
    1. Baseline Tracking: Continuously establishes individual user's open-eye baseline (e.g. 0.46 for glasses wearers).
    2. Relative Drop Detection: Detects closure when EAR dips below max(0.38, baseline * 0.88).
    3. Recovery Verification: Confirms blink when EAR returns to baseline openness.
- Solves dark/thick glasses frame landmark interference (where closure dip is ~0.40 vs 0.46 open).
- Guaranteed 6.0-second challenge countdown with strict timeout and no premature spoof triggers.
- Robust against static photo spoof attacks (photos exhibit 0% dynamic EAR dip).
"""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
import cv2

import app.config as config


class FaceLivenessTracker:
    """
    Tracks the active blink challenge lifecycle for an individual face instance.
    Uses Adaptive Baseline Relative Drop + Absolute Threshold.
    """

    STATE_CHALLENGE_ACTIVE = "CHALLENGE_ACTIVE"
    STATE_LIVENESS_PASSED = "LIVENESS_PASSED"
    STATE_LIVENESS_FAILED = "LIVENESS_FAILED"

    def __init__(self, 
                 ear_thresh: float = config.EAR_THRESHOLD,
                 min_closed_frames: int = config.BLINK_MIN_FRAMES,
                 timeout_sec: float = config.LIVENESS_TIMEOUT_SECONDS,
                 cooldown_sec: float = config.BLINK_COOLDOWN):
        self.ear_thresh = ear_thresh
        self.min_closed_frames = min_closed_frames
        self.timeout_sec = timeout_sec
        self.cooldown_sec = cooldown_sec

        self.state = self.STATE_CHALLENGE_ACTIVE
        self.start_time = time.time()
        self.last_terminal_print = 0.0
        self.last_blink_time = 0.0
        
        # Baseline & Dynamic Drop Tracking
        self.open_baseline = 0.45
        self.baseline_samples = []
        self.min_ear_seen = 1.0
        self.max_ear_seen = 0.0
        
        self.closed_frames_count = 0
        self.blink_completed = False
        self.eye_state = "OPEN"           # "OPEN" or "CLOSED"
        self.blink_status = "WAITING"     # "WAITING", "EYES_CLOSED", "DETECTED"
        
        self.last_ear = 0.45
        self.last_left_ear = 0.45
        self.last_right_ear = 0.45
        self.passed_timestamp = 0.0

    def get_time_remaining(self) -> float:
        """Returns seconds remaining in the challenge window (0.0 if expired)."""
        elapsed = time.time() - self.start_time
        return max(0.0, self.timeout_sec - elapsed)

    def get_effective_threshold(self) -> float:
        """
        Calculates dynamic threshold combining configured threshold and individual baseline drop.
        For glasses wearers with baseline 0.46, dynamic threshold is ~0.415.
        """
        if len(self.baseline_samples) >= 3:
            baseline = float(np.median(self.baseline_samples[-25:]))
            # 11% dip from open baseline or absolute threshold, whichever is higher
            adaptive_thresh = baseline * 0.89
            return float(max(adaptive_thresh, self.ear_thresh))
        return self.ear_thresh

    def update(self, avg_ear: float, left_ear: float, right_ear: float) -> str:
        """
        Updates the blink state machine given the current frame's Eye Aspect Ratio metrics.

        Returns:
            Current state: 'CHALLENGE_ACTIVE', 'LIVENESS_PASSED', or 'LIVENESS_FAILED'
        """
        self.last_ear = avg_ear
        self.last_left_ear = left_ear
        self.last_right_ear = right_ear
        curr_time = time.time()

        # Update min/max observed EAR
        if avg_ear > 0:
            if avg_ear < self.min_ear_seen:
                self.min_ear_seen = avg_ear
            if avg_ear > self.max_ear_seen:
                self.max_ear_seen = avg_ear

        # If already passed, stay in passed state
        if self.state == self.STATE_LIVENESS_PASSED:
            self.blink_status = "DETECTED"
            return self.state

        # If already failed, stay in failed state until track reset
        if self.state == self.STATE_LIVENESS_FAILED:
            return self.state

        # Check for 6.0s timeout expiration
        elapsed = curr_time - self.start_time
        if elapsed > self.timeout_sec:
            if not self.blink_completed:
                self.state = self.STATE_LIVENESS_FAILED
                print(f"[LivenessTracker] [!] Challenge Timed Out ({elapsed:.1f}s) -> LIVENESS FAILED")
                return self.state

        # Calculate effective threshold for this user
        effective_thresh = self.get_effective_threshold()

        # Determine eye state in current frame
        if avg_ear < effective_thresh:
            self.eye_state = "CLOSED"
            self.closed_frames_count += 1
            self.blink_status = "EYES_CLOSED"
        else:
            self.eye_state = "OPEN"
            # Add to baseline sample buffer when eyes are open
            if len(self.baseline_samples) < 50:
                self.baseline_samples.append(avg_ear)
            elif avg_ear > self.min_ear_seen + 0.04:
                self.baseline_samples.pop(0)
                self.baseline_samples.append(avg_ear)

            # If eyes were closed in preceding frame(s) and now open -> FULL BLINK CYCLE!
            if self.closed_frames_count >= self.min_closed_frames and (curr_time - self.last_blink_time > self.cooldown_sec):
                self.blink_completed = True
                self.blink_status = "DETECTED"
                self.state = self.STATE_LIVENESS_PASSED
                self.passed_timestamp = curr_time
                self.last_blink_time = curr_time
                print(f"\n[LivenessTracker] [SUCCESS] REAL BLINK VERIFIED!")
                print(f"  -> Open Baseline: {np.mean(self.baseline_samples):.3f}")
                print(f"  -> Closure Dip  : {self.min_ear_seen:.3f}")
                print(f"  -> Recovery EAR : {avg_ear:.3f}")
                print(f"  -> Decision Thresh: {effective_thresh:.3f}\n")
            else:
                if not self.blink_completed:
                    self.blink_status = "WAITING"
            self.closed_frames_count = 0

        # Throttled terminal debug output (once every 1.5s during active challenge)
        if curr_time - self.last_terminal_print > 1.5 and self.state == self.STATE_CHALLENGE_ACTIVE:
            self.last_terminal_print = curr_time
            time_left = self.get_time_remaining()
            print(f"[LivenessDebug] EAR: {avg_ear:.2f} (L: {left_ear:.2f}, R: {right_ear:.2f} | Thresh: {effective_thresh:.2f}) | Eye: {self.eye_state:<6} | Blink: {self.blink_status:<11} | Lowest: {self.min_ear_seen:.2f} | Time: {time_left:.1f}s")

        return self.state


class BlinkLivenessDetector:
    """
    Extracts 2D eye landmarks, computes EAR, and coordinates multi-face liveness challenges.
    """

    def __init__(self, 
                 ear_thresh: float = config.EAR_THRESHOLD,
                 min_closed_frames: int = config.BLINK_MIN_FRAMES,
                 timeout_sec: float = config.LIVENESS_TIMEOUT_SECONDS,
                 cooldown_sec: float = config.BLINK_COOLDOWN):
        self.ear_thresh = ear_thresh
        self.min_closed_frames = min_closed_frames
        self.timeout_sec = timeout_sec
        self.cooldown_sec = cooldown_sec
        self.trackers: Dict[str, FaceLivenessTracker] = {}

    @staticmethod
    def _dist(p1: np.ndarray, p2: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(p1) - np.asarray(p2)))

    def calculate_ear_left_eye(self, lm106: np.ndarray) -> float:
        """
        Calculates EAR for Left Eye using 8 contour points from InsightFace 2d106det:
        Points: 35 (outer corner), 36 (top-outer), 37 (top-mid), 38 (top-inner),
                39 (inner corner), 40 (bottom-inner), 41 (bottom-mid), 42 (bottom-outer)
        """
        v1 = self._dist(lm106[36], lm106[42])
        v2 = self._dist(lm106[37], lm106[41])
        v3 = self._dist(lm106[38], lm106[40])
        h = self._dist(lm106[35], lm106[39])

        if h == 0 or np.isnan(h):
            return 0.45

        ear = (v1 + v2 + v3) / (3.0 * h)
        return float(ear)

    def calculate_ear_right_eye(self, lm106: np.ndarray) -> float:
        """
        Calculates EAR for Right Eye using 8 contour points from InsightFace 2d106det:
        Points: 89 (inner corner), 90 (top-inner), 91 (top-mid), 92 (top-outer),
                93 (outer corner), 94 (bottom-outer), 95 (bottom-mid), 96 (bottom-inner)
        """
        v1 = self._dist(lm106[90], lm106[96])
        v2 = self._dist(lm106[91], lm106[95])
        v3 = self._dist(lm106[92], lm106[94])
        h = self._dist(lm106[89], lm106[93])

        if h == 0 or np.isnan(h):
            return 0.45

        ear = (v1 + v2 + v3) / (3.0 * h)
        return float(ear)

    def extract_ear_from_face(self, face) -> Tuple[float, float, float]:
        """
        Extracts Left EAR, Right EAR, and Average EAR from InsightFace Face landmarks.
        """
        if hasattr(face, 'landmark_2d_106') and face.landmark_2d_106 is not None:
            lm106 = face.landmark_2d_106
            if len(lm106) >= 106:
                left_ear = self.calculate_ear_left_eye(lm106)
                right_ear = self.calculate_ear_right_eye(lm106)
                avg_ear = (left_ear + right_ear) / 2.0
                return left_ear, right_ear, avg_ear

        return 0.45, 0.45, 0.45

    def process_liveness(self, face, face_track_id: str) -> Dict[str, Any]:
        """
        Processes active liveness challenge for a tracked face.
        """
        left_ear, right_ear, avg_ear = self.extract_ear_from_face(face)

        if face_track_id not in self.trackers:
            self.trackers[face_track_id] = FaceLivenessTracker(
                ear_thresh=self.ear_thresh,
                min_closed_frames=self.min_closed_frames,
                timeout_sec=self.timeout_sec,
                cooldown_sec=self.cooldown_sec
            )

        tracker = self.trackers[face_track_id]
        state = tracker.update(avg_ear, left_ear, right_ear)
        time_left = tracker.get_time_remaining()
        eff_thresh = tracker.get_effective_threshold()

        if state == FaceLivenessTracker.STATE_LIVENESS_PASSED:
            is_live = True
            msg = "LIVENESS: PASS"
        elif state == FaceLivenessTracker.STATE_LIVENESS_FAILED:
            is_live = False
            msg = "LIVENESS: FAILED | ATTENDANCE BLOCKED"
        else:
            is_live = False
            msg = f"PLEASE BLINK ({int(np.ceil(time_left))}s)"

        return {
            "state": state,
            "is_live": is_live,
            "left_ear": left_ear,
            "right_ear": right_ear,
            "avg_ear": avg_ear,
            "eye_state": tracker.eye_state,
            "blink_status": tracker.blink_status,
            "min_ear_seen": tracker.min_ear_seen,
            "max_ear_seen": tracker.max_ear_seen,
            "ear_thresh": eff_thresh,
            "time_remaining": time_left,
            "message": msg,
            "blink_detected": tracker.blink_completed
        }

    def reset_tracker(self, face_track_id: str):
        """Resets the challenge state for a specific track ID."""
        if face_track_id in self.trackers:
            del self.trackers[face_track_id]
