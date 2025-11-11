#!/usr/bin/env python3
"""
Cleanup Helper Script for Claude
Membantu Claude membersihkan file sementara dan merapikan struktur folder
"""

import os
import shutil
import glob
from pathlib import Path

def cleanup_temp_files():
    """Hapus file-file sementara dan hasil test"""
    print("Membersihkan file sementara...")

    # Pola file yang akan dihapus
    patterns_to_delete = [
        "debug_*.py",
        "test_*.py",
        "investigate_*.py",
        "temp_*.py",
        "*.tmp",
        "*.log",
        "*_backup.py",
        "*_old.py",
        "debug_output.txt",
        "test_results.txt",
        ".pytest_cache",
        "__pycache__",
        "*.pyc",
        "*.pyo"
    ]

    deleted_count = 0
    for pattern in patterns_to_delete:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                    print(f"  [OK] Dihapus: {file}")
                    deleted_count += 1
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                    print(f"  [OK] Dihapus folder: {file}")
                    deleted_count += 1
            except Exception as e:
                print(f"  [ERROR] Gagal menghapus {file}: {e}")

    print(f"\nTotal file/folder yang dihapus: {deleted_count}")
    return deleted_count

def organize_folders():
    """Rapikan struktur folder"""
    print("\nMerapikan struktur folder...")

    # Mapping file ke folder tujuan
    folder_mapping = {
        'src': ['main.py', '*.py', 'modules/', 'utils/'],
        'tests': ['test_*.py', '*_test.py'],
        'data': ['*.csv', '*.json', '*.xlsx', '*.parquet'],
        'docs': ['*.md', 'README*', 'CHANGELOG*'],
        'logs': ['*.log'],
        'temp': ['temp_*', '*_temp', 'debug_*']
    }

    # Buat folder jika belum ada
    for folder in folder_mapping.keys():
        os.makedirs(folder, exist_ok=True)
        print(f"  [OK] Folder siap: {folder}/")

    moved_count = 0

    # Pindahkan file ke folder yang sesuai
    current_dir = Path('.')
    for item in current_dir.iterdir():
        if item.name.startswith('.') or item.is_dir():
            continue

        item_path = str(item)
        item_lower = item.name.lower()

        # Tentukan folder tujuan
        target_folder = None
        for folder, patterns in folder_mapping.items():
            for pattern in patterns:
                if item.name == pattern or item_lower.endswith(pattern.replace('*', '').lower()):
                    target_folder = folder
                    break
            if target_folder:
                break

        # Pindahkan file jika ada folder tujuan
        if target_folder and not item.name.startswith('cleanup_'):
            try:
                target_path = Path(folder) / item.name
                if not target_path.exists():
                    shutil.move(item_path, folder)
                    print(f"  [MOVED] {item.name} -> {folder}/")
                    moved_count += 1
            except Exception as e:
                print(f"  [ERROR] Gagal memindahkan {item.name}: {e}")

    print(f"\nTotal file yang dipindahkan: {moved_count}")
    return moved_count

def show_directory_structure():
    """Tampilkan struktur folder saat ini"""
    print("\nStruktur folder saat ini:")

    def print_tree(directory, prefix="", max_depth=3, current_depth=0):
        if current_depth >= max_depth:
            return

        try:
            items = sorted(Path(directory).iterdir(), key=lambda x: (x.is_file(), x.name.lower()))
            dirs = [item for item in items if item.is_dir() and not item.name.startswith('.')]
            files = [item for item in items if item.is_file() and not item.name.startswith('.')]

            # Print directories first
            for i, item in enumerate(dirs):
                is_last_dir = (i == len(dirs) - 1) and len(files) == 0
                print(f"{prefix}{'L-- ' if is_last_dir else '|-- '}{item.name}/")
                next_prefix = prefix + ("    " if is_last_dir else "|   ")
                print_tree(item, next_prefix, max_depth, current_depth + 1)

            # Print files
            for i, item in enumerate(files):
                is_last = i == len(files) - 1
                print(f"{prefix}{'L-- ' if is_last else '|-- '}{item.name}")

        except PermissionError:
            print(f"{prefix}[Permission Denied]")

    print_tree(".", max_depth=3)

def main():
    """Main cleanup function"""
    print("=" * 50)
    print("CLEANUP HELPER FOR CLAUDE")
    print("=" * 50)

    # Tampilkan struktur awal
    show_directory_structure()

    # Cleanup file sementara
    deleted = cleanup_temp_files()

    # Rapikan folder
    moved = organize_folders()

    # Tampilkan struktur akhir
    show_directory_structure()

    print("\n" + "=" * 50)
    print("CLEANUP SELESAI")
    print(f"Total dihapus: {deleted} file/folder")
    print(f"Total dipindahkan: {moved} file")
    print("=" * 50)

if __name__ == "__main__":
    main()