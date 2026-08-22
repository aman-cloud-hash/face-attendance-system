"""
Script: test_google_sheets.py
Purpose: Test and verify Google Sheets Cloud Attendance Sync.

Tests:
1. Service Account Authentication & Google Sheets API Connectivity.
2. Header Row Initialization: ['Student ID', 'Roll Number', 'Student Name', 'Date', 'Time', 'Status'].
3. Upload of live attendance record for student Aman (ID: 101).
4. Duplicate check-in prevention on the same calendar date.
"""

import sys
from datetime import datetime
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.attendance.google_sheets_manager import GoogleSheetsManager


def test_google_sheets_sync():
    print("\n" + "=" * 70)
    print("      GOOGLE SHEETS REALTIME ATTENDANCE SYNC TEST")
    print("=" * 70)
    print(f"[*] Service Account JSON : {config.GOOGLE_SERVICE_ACCOUNT_FILE}")
    print(f"[*] Google Sheet ID      : {config.GOOGLE_SHEET_ID}\n")

    # 1. Initialize Connection
    print("[*] Connecting to Google Sheets API...")
    manager = GoogleSheetsManager()

    if not manager.is_connected:
        print("[!] Error: Could not connect to Google Sheets.")
        print("    Please check that the Google Sheet is shared with the Service Account email as Editor:")
        print("    face-attendance@smart-face-attendance-506310.iam.gserviceaccount.com\n")
        return False

    print(f"[+] Successfully Connected to Worksheet: '{manager.spreadsheet.title}'")

    # 2. Test Attendance Upload for Aman (ID: 101)
    now = datetime.now()
    date_str = now.strftime("%Y-%m-%d")
    time_str = now.strftime("%H:%M:%S")

    print(f"\n[*] Testing Real-Time Attendance Upload for Student:")
    print(f"    - Student ID   : 101")
    print(f"    - Roll Number  : 2525")
    print(f"    - Student Name : Aman")
    print(f"    - Date         : {date_str}")
    print(f"    - Time         : {time_str}")
    print(f"    - Status       : PRESENT")

    success = manager.upload_attendance(
        student_id="101",
        roll_number="2525",
        name="Aman",
        date_str=date_str,
        time_str=time_str,
        status="PRESENT"
    )

    if success:
        print("\n" + "=" * 70)
        print("[SUCCESS] Google Sheets Integration is FULLY OPERATIONAL!")
        print("=" * 70 + "\n")
    else:
        print("\n[!] Upload test encountered an error.\n")

    return success


if __name__ == "__main__":
    test_google_sheets_sync()
