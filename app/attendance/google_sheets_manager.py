"""
Google Sheets Realtime Attendance Sync Manager.

Features:
- Connects securely using Google Cloud Service Account credentials.
- Auto-initializes worksheet header row if empty:
    ['Student ID', 'Roll Number', 'Student Name', 'Date', 'Time', 'Status']
- Smart Row Insertion: Finds the FIRST empty row (e.g. Row 2 if empty) instead of blindly appending at the bottom.
- Prevents duplicate attendance check-ins on the same calendar date.
- Fault-tolerant: API / Network errors are logged safely without interrupting real-time video stream.
"""

import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import gspread
from google.oauth2.service_account import Credentials

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config


class GoogleSheetsManager:
    """
    Manages direct cloud sync with Google Sheets via gspread API.
    """

    SCOPES = [
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive"
    ]

    def __init__(self, 
                 credentials_file: Optional[Path] = None, 
                 sheet_id: Optional[str] = None):
        self.credentials_file = Path(credentials_file or config.GOOGLE_SERVICE_ACCOUNT_FILE)
        self.sheet_id = str(sheet_id or config.GOOGLE_SHEET_ID).strip()
        self.client: Optional[gspread.Client] = None
        self.spreadsheet: Optional[gspread.Spreadsheet] = None
        self.worksheet: Optional[gspread.Worksheet] = None
        self.is_connected = False

        self._connect()

    def _connect(self) -> bool:
        """
        Authenticates with Google Sheets API and verifies worksheet headers.
        """
        if not self.credentials_file.exists():
            print(f"[GoogleSheets] [!] Service account key not found at: {self.credentials_file}")
            self.is_connected = False
            return False

        try:
            creds = Credentials.from_service_account_file(
                str(self.credentials_file),
                scopes=self.SCOPES
            )
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(self.sheet_id)
            self.worksheet = self.spreadsheet.sheet1
            self.is_connected = True

            # Check and auto-initialize header row
            self._ensure_headers()
            print(f"[GoogleSheets] [+] Connected to spreadsheet: '{self.spreadsheet.title}' (Sheet1)")
            return True
        except Exception as e:
            print(f"[GoogleSheets] [!] Connection failed: {e}")
            self.is_connected = False
            return False

    def _ensure_headers(self):
        """
        Ensures the first row contains the required column headers.
        """
        if not self.is_connected or self.worksheet is None:
            return

        try:
            rows = self.worksheet.get_all_values()
            if not rows or len(rows) == 0 or len(rows[0]) == 0:
                # Sheet is empty -> insert header
                self.worksheet.insert_row(config.GOOGLE_SHEETS_HEADERS, index=1)
                print(f"[GoogleSheets] [+] Initialized default headers: {config.GOOGLE_SHEETS_HEADERS}")
            else:
                first_row = [str(col).strip() for col in rows[0]]
                if first_row != config.GOOGLE_SHEETS_HEADERS:
                    if any(header not in first_row for header in ["Student ID", "Student Name", "Date"]):
                        self.worksheet.insert_row(config.GOOGLE_SHEETS_HEADERS, index=1)
                        print(f"[GoogleSheets] [+] Inserted missing header row at top.")
        except Exception as e:
            print(f"[GoogleSheets] [!] Warning checking headers: {e}")

    def is_already_marked(self, student_id: str, date_str: str) -> bool:
        """
        Checks if the student ID already has an entry for the given date in Google Sheets.
        """
        if not self.is_connected or self.worksheet is None:
            return False

        try:
            rows = self.worksheet.get_all_values()
            if len(rows) <= 1:
                return False

            headers = [h.strip() for h in rows[0]]
            try:
                id_idx = headers.index("Student ID")
                date_idx = headers.index("Date")
            except ValueError:
                id_idx, date_idx = 0, 3

            sid_str = str(student_id).strip().lower()
            d_str = str(date_str).strip()

            for row in rows[1:]:
                if len(row) > max(id_idx, date_idx):
                    row_sid = str(row[id_idx]).strip().lower()
                    row_date = str(row[date_idx]).strip()

                    if row_sid == sid_str and row_date == d_str:
                        return True
            return False
        except Exception as e:
            print(f"[GoogleSheets] [!] Duplicate check error: {e}")
            return False

    def find_first_available_row(self) -> int:
        """
        Finds the 1-based index of the first empty row after row 1.
        If row 2 is empty, returns 2.
        If rows 2..N are filled, returns N + 1.
        """
        if not self.is_connected or self.worksheet is None:
            return 2

        rows = self.worksheet.get_all_values()
        if not rows or len(rows) <= 1:
            return 2

        for idx, row in enumerate(rows[1:], start=2):
            # Check if all cells in this row are empty strings or whitespace
            if not any(str(cell).strip() for cell in row):
                return idx

        return len(rows) + 1

    def upload_attendance(self, 
                          student_id: str, 
                          roll_number: str, 
                          name: str, 
                          date_str: str, 
                          time_str: str, 
                          status: str = "PRESENT") -> bool:
        """
        Writes attendance record to the first empty row in Google Sheets.

        Returns:
            bool: True if uploaded or already present, False on failure.
        """
        if not self.is_connected:
            if not self._connect():
                return False

        try:
            # 1. Prevent duplicate entries for same student on same date
            if self.is_already_marked(student_id, date_str):
                print(f"[GoogleSheets] [*] Student {name} (ID: {student_id}) already marked in cloud sheet for {date_str}.")
                return True

            # 2. Prepare Row Data
            row_data = [
                str(student_id),
                str(roll_number),
                str(name),
                str(date_str),
                str(time_str),
                str(status).upper()
            ]

            # 3. Find the first empty row (e.g. Row 2)
            target_row = self.find_first_available_row()
            col_end = chr(ord('A') + len(row_data) - 1)
            range_name = f"A{target_row}:{col_end}{target_row}"

            # Write directly to the target empty row
            self.worksheet.update(range_name=range_name, values=[row_data])
            print(f"[GoogleSheets] Attendance uploaded successfully (Row {target_row})")
            return True

        except Exception as e:
            print(f"[GoogleSheets] [!] Upload error: {e}")
            return False
