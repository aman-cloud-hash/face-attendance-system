"""
Script: delete_student.py
Purpose: Manage and Delete Enrolled Student Biometrics and Metadata.

Features:
- Lists all registered students in a formatted table.
- Interactive and CLI deletion of individual students or complete database reset.
- Cleans up:
    1. Biometric feature embeddings from data/embeddings/face_embeddings.pkl
    2. Student metadata from data/embeddings/students_meta.json
    3. Facial capture images from data/faces/<student_id>/
"""

import sys
import shutil
import argparse
from pathlib import Path

# Ensure project root is in sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

import app.config as config
from app.face_recognition.embedding_store import EmbeddingStore


def list_and_display_students(store: EmbeddingStore) -> list:
    """Prints a formatted roster of all enrolled students."""
    students = store.list_students()
    print("\n" + "=" * 70)
    print("                 ENROLLED STUDENTS ROSTER")
    print("=" * 70)
    
    if not students:
        print("  [!] No students are currently registered in the database.")
        print("=" * 70 + "\n")
        return []

    print(f"  {'ID':<8} {'ROLL NO':<12} {'STUDENT NAME':<25} {'SAMPLES':<8} {'REGISTERED'}")
    print("  " + "-" * 66)

    for s in students:
        reg_time = s.get("registered_at", "")[:19] if s.get("registered_at") else "N/A"
        print(f"  {s['student_id']:<8} {s['roll_number']:<12} {s['name']:<25} {s['sample_count']:<8} {reg_time}")

    print("=" * 70 + "\n")
    return students


def delete_student_by_id(student_id: str, store: EmbeddingStore, force: bool = False) -> bool:
    """Deletes a student by ID from embeddings, metadata, and physical images."""
    student_id = str(student_id).strip()
    student = store.get_student(student_id)

    if not student:
        print(f"[!] Error: Student ID '{student_id}' not found in database.")
        return False

    name = student.get("name", "Unknown")
    roll = student.get("roll_number", "N/A")

    if not force:
        confirm = input(f"[?] Are you sure you want to delete '{name}' (ID: {student_id}, Roll: {roll})? (y/N): ").strip().lower()
        if confirm != "y":
            print("[*] Deletion cancelled.")
            return False

    # 1. Delete from EmbeddingStore (.pkl and .json)
    store.delete_student(student_id)

    # 2. Delete physical image samples directory
    student_faces_dir = config.FACES_DIR / student_id
    if student_faces_dir.exists():
        shutil.rmtree(student_faces_dir, ignore_errors=True)
        print(f"[+] Removed face images directory: data/faces/{student_id}/")

    print(f"\n[SUCCESS] Student '{name}' (ID: {student_id}) has been completely deleted!")
    return True


def delete_all_students(store: EmbeddingStore, force: bool = False):
    """Deletes all enrolled students."""
    students = store.list_students()
    if not students:
        print("[!] Database is already empty.")
        return

    if not force:
        confirm = input(f"[?] CAUTION: Are you sure you want to delete ALL {len(students)} student(s)? (y/N): ").strip().lower()
        if confirm != "y":
            print("[*] Database reset cancelled.")
            return

    for s in students:
        sid = s["student_id"]
        store.delete_student(sid)
        student_faces_dir = config.FACES_DIR / sid
        if student_faces_dir.exists():
            shutil.rmtree(student_faces_dir, ignore_errors=True)

    print(f"\n[SUCCESS] All {len(students)} students have been deleted from the database.")


def main():
    parser = argparse.ArgumentParser(description="Delete enrolled student biometrics and metadata")
    parser.add_argument("--id", type=str, default=None, help="Student ID to delete (e.g. --id 101)")
    parser.add_argument("--all", action="store_true", help="Delete all registered students")
    parser.add_argument("--list", action="store_true", help="List all registered students and exit")
    parser.add_argument("--force", action="store_true", help="Skip confirmation prompt")
    args = parser.parse_args()

    store = EmbeddingStore()

    if args.list:
        list_and_display_students(store)
        return

    if args.all:
        delete_all_students(store, force=args.force)
        return

    if args.id:
        delete_student_by_id(args.id, store, force=args.force)
        return

    # Interactive mode
    students = list_and_display_students(store)
    if not students:
        return

    try:
        target_id = input("[?] Enter Student ID to delete (or 'all' / 'q' to quit): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\n[*] Exiting.")
        return

    if target_id.lower() in ["q", "quit", "exit", ""]:
        print("[*] Exited.")
        return

    if target_id.lower() == "all":
        delete_all_students(store, force=args.force)
    else:
        delete_student_by_id(target_id, store, force=args.force)


if __name__ == "__main__":
    main()
