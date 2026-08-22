"""
CLI Tool: list_cameras.py
Purpose: Scans and lists all connected cameras on Windows with DirectShow.
Helps identify the Irium Webcam index and test video stream availability.
"""

import sys
from pathlib import Path

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.camera_utils import get_available_cameras


def main():
    print("=" * 65)
    print("          WINDOWS CAMERA DIAGNOSTIC & DISCOVERY TOOL")
    print("=" * 65)
    print("[*] Probing Windows Video Devices (DirectShow)...\n")

    cameras = get_available_cameras(max_to_test=8)

    if not cameras:
        print("[!] No video capture devices found.")
        print("[*] Troubleshooting Tips for Irium Webcam:")
        print("    1. Ensure 'Irium Webcam' app is running on your Windows PC.")
        print("    2. Ensure 'Irium Webcam' app is open on your mobile phone.")
        print("    3. Ensure phone and PC are connected to the same Wi-Fi network (or USB debugging).")
        return

    print(f"[+] Found {len(cameras)} active camera device(s):\n")
    print(f"{'Index':^8} | {'Status':^18} | {'Resolution':^14} | {'Backend':^14}")
    print("-" * 65)

    for cam in cameras:
        idx = cam["index"]
        status = cam["status"]
        res = f"{cam['width']}x{cam['height']}"
        backend = cam["backend"]
        
        is_config_idx = " (Configured in config.py)" if idx == config.CAMERA_INDEX else ""
        print(f"{idx:^8} | {status:^18} | {res:^14} | {backend:^14}{is_config_idx}")

    print("-" * 65)
    print(f"\n[*] Current configured CAMERA_INDEX in app/config.py is: {config.CAMERA_INDEX}")
    print("[*] To change camera index, pass --camera <INDEX> to any script or edit app/config.py.")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
