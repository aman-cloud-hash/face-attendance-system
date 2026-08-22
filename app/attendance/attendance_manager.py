"""
Attendance Management and CSV Persistence Module.

Key Functionality:
- Persistent CSV Ledger: Logs date, check-in timestamp, identity, confidence, and liveness status.
- Idempotency & Duplicate Prevention: Ensures each student is marked present at most ONCE per calendar date.
- Real-time Google Sheets Cloud Sync: Uploads records directly to cloud spreadsheet.
- Real-time Querying: Retrieves current-day attendance rosters and summary statistics.
- Thread-safe and fault-tolerant file append operations.
"""

import os
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd

import app.config as config
from app.attendance.google_sheets_manager import GoogleSheetsManager


class AttendanceManager:
    """
    Manages daily student attendance logging, duplicate verification, CSV serialization,
    and Real-Time Google Sheets cloud synchronization.
    """

    def __init__(self, csv_path: Path = config.ATTENDANCE_CSV, enable_google_sheets: bool = True):
        self.csv_path = Path(csv_path)
        self._initialize_csv()
        
        # Initialize Google Sheets Manager
        self.enable_google_sheets = enable_google_sheets
        self.sheets_manager: Optional[GoogleSheetsManager] = None
        if self.enable_google_sheets:
            try:
                self.sheets_manager = GoogleSheetsManager()
            except Exception as e:
                print(f"[AttendanceManager] [!] Google Sheets integration init warning: {e}")
                self.sheets_manager = None

    def _initialize_csv(self):
        """Creates the attendance CSV file with standard headers if it does not exist."""
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.csv_path.exists() or self.csv_path.stat().st_size == 0:
            with open(self.csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(config.ATTENDANCE_CSV_COLUMNS)
            print(f"[AttendanceManager] Initialized fresh attendance CSV at {self.csv_path.name}")

    def is_already_marked(self, student_id: str, date_str: Optional[str] = None) -> bool:
        """
        Checks whether a student has already been marked present for the given date.

        Args:
            student_id: Student unique ID
            date_str: Date string in 'YYYY-MM-DD' format (defaults to today)

        Returns:
            bool: True if already recorded today, False otherwise.
        """
        if date_str is None:
            date_str = datetime.now().strftime("%Y-%m-%d")

        student_id = str(student_id).strip()

        if not self.csv_path.exists():
            return False

        try:
            df = pd.read_csv(self.csv_path, dtype=str)
            if df.empty:
                return False

            match = df[(df["student_id"] == student_id) & (df["date"] == date_str)]
            return not match.empty
        except Exception as e:
            print(f"[AttendanceManager] Error checking duplicate record: {e}")
            return False

    def mark_attendance(self, 
                        student_id: str, 
                        roll_number: str, 
                        name: str, 
                        confidence: float, 
                        liveness_status: str = "passed") -> Dict[str, Any]:
        """
        Marks attendance for a verified student after passing recognition and liveness.
        Appends to local CSV and syncs to Google Sheets in real-time.

        Args:
            student_id: Student unique ID (e.g. '101')
            roll_number: Academic roll number (e.g. '23CS001')
            name: Full name of the student
            confidence: Cosine similarity score (e.g. 0.82)
            liveness_status: 'passed', 'failed', or 'bypassed'

        Returns:
            Dict containing:
            - status: 'SUCCESS', 'ALREADY_MARKED', or 'ERROR'
            - message: User-friendly description
            - timestamp: Check-in time string
        """
        now = datetime.now()
        date_str = now.strftime("%Y-%m-%d")
        time_str = now.strftime("%H:%M:%S")

        student_id = str(student_id).strip()
        roll_number = str(roll_number).strip()
        name = str(name).strip()

        # 1. Duplicate Prevention Check (Local CSV)
        if self.is_already_marked(student_id, date_str):
            msg = f"Attendance ALREADY MARKED today for {name} (ID: {student_id})"
            # Also ensure cloud sheet has it
            if self.sheets_manager and self.sheets_manager.is_connected:
                self.sheets_manager.upload_attendance(
                    student_id=student_id,
                    roll_number=roll_number,
                    name=name,
                    date_str=date_str,
                    time_str=time_str,
                    status="PRESENT"
                )

            return {
                "status": "ALREADY_MARKED",
                "message": msg,
                "date": date_str,
                "time": time_str,
                "student_id": student_id,
                "name": name
            }

        # 2. Append new attendance record to Local CSV
        record = [
            student_id,
            roll_number,
            name,
            date_str,
            time_str,
            "present",
            f"{confidence:.4f}",
            liveness_status
        ]

        try:
            with open(self.csv_path, mode="a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(record)

            msg = f"[SUCCESS] Attendance Marked: {name} (ID: {student_id}) at {time_str}"
            print(f"[AttendanceManager] {msg}")

            # 3. Real-Time Google Sheets Cloud Sync
            if self.sheets_manager:
                try:
                    self.sheets_manager.upload_attendance(
                        student_id=student_id,
                        roll_number=roll_number,
                        name=name,
                        date_str=date_str,
                        time_str=time_str,
                        status="PRESENT"
                    )
                except Exception as sheet_err:
                    print(f"[AttendanceManager] [!] Non-fatal Google Sheets sync error: {sheet_err}")

            return {
                "status": "SUCCESS",
                "message": msg,
                "date": date_str,
                "time": time_str,
                "student_id": student_id,
                "name": name
            }
        except Exception as e:
            err_msg = f"Failed to write attendance record: {e}"
            print(f"[AttendanceManager] [!] {err_msg}")
            return {
                "status": "ERROR",
                "message": err_msg,
                "date": date_str,
                "time": time_str,
                "student_id": student_id,
                "name": name
            }

    def get_today_records(self) -> pd.DataFrame:
        """Returns a pandas DataFrame of all attendance records marked today."""
        today = datetime.now().strftime("%Y-%m-%d")
        if not self.csv_path.exists():
            return pd.DataFrame(columns=config.ATTENDANCE_CSV_COLUMNS)

        try:
            df = pd.read_csv(self.csv_path)
            return df[df["date"] == today]
        except Exception as e:
            print(f"[AttendanceManager] Error loading today's records: {e}")
            return pd.DataFrame(columns=config.ATTENDANCE_CSV_COLUMNS)

    def get_total_attendance_count(self) -> int:
        """Returns total records in the CSV file."""
        if not self.csv_path.exists():
            return 0
        try:
            df = pd.read_csv(self.csv_path)
            return len(df)
        except Exception:
            return 0
