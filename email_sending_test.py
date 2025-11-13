import os
import sys
import shutil
import tempfile
from pathlib import Path
from typing import List

# Ensure project root is on path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from main import send_email

# SES raw email limit (bytes)
SES_RAW_LIMIT = 10 * 1024 * 1024  # 10,485,760 in practice; use 10MB as safe bound

# Reserve some overhead for MIME headers and html body
MIME_OVERHEAD = 200_000

ALLOWED_EXT = {"jpg", "jpeg", "png"}


def list_subfolders(base: Path) -> List[Path]:
    if not base.exists():
        return []
    return [p for p in base.iterdir() if p.is_dir()]


def make_batches(file_paths: List[Path]) -> List[List[Path]]:
    """
    Batch files so that after base64 encoding (approx +33%) and MIME overhead
    the total raw message size stays under SES_RAW_LIMIT.
    """
    allowed_original = int((SES_RAW_LIMIT - MIME_OVERHEAD) * 3 / 4)

    batches = []
    current = []
    current_sum = 0

    for p in file_paths:
        size = p.stat().st_size
        # If single file larger than allowed_original, we'll still put it alone
        if current_sum + size <= allowed_original:
            current.append(p)
            current_sum += size
        else:
            if current:
                batches.append(current)
            current = [p]
            current_sum = size
            # If the single file already exceeds allowed_original, keep it as single batch
    if current:
        batches.append(current)

    return batches


def copy_batch_to_temp(batch: List[Path]) -> str:
    tmpdir = tempfile.mkdtemp(prefix="email_batch_")
    for p in batch:
        shutil.copy2(p, tmpdir)
    return tmpdir


def choose_folder(default_subfolder: str = None) -> Path:
    base = ROOT / 'pdf_image'
    subfolders = list_subfolders(base)
    if not subfolders:
        print(f"No subfolders found under {base}")
        return None

    if default_subfolder:
        candidate = base / default_subfolder
        if candidate.exists() and candidate.is_dir():
            return candidate
        else:
            print(f"Default subfolder not found: {default_subfolder}")

    print("Available subfolders:")
    for i, p in enumerate(subfolders, 1):
        print(f"{i}. {p.name}")

    try:
        choice = int(input(f"Select folder to send (1-{len(subfolders)}), or 0 to cancel: "))
    except Exception:
        print("Invalid input")
        return None

    if choice <= 0 or choice > len(subfolders):
        print("Cancelled")
        return None

    return subfolders[choice - 1]


def gather_image_files(folder: Path) -> List[Path]:
    files = [p for p in sorted(folder.iterdir()) if p.is_file() and p.suffix.lstrip('.').lower() in ALLOWED_EXT]
    return files


def main():
    folder = choose_folder()
    if not folder:
        return

    print(f"Sending images from folder: {folder}")

    files = gather_image_files(folder)
    if not files:
        print("No image files found in selected folder")
        return

    batches = make_batches(files)
    print(f"Split into {len(batches)} batch(es) to respect SES size limits")

    for i, batch in enumerate(batches, 1):
        print(f"Sending batch {i}/{len(batches)} with {len(batch)} file(s)")
        tmpdir = copy_batch_to_temp(batch)
        try:
            send_email(tmpdir)
        finally:
            try:
                shutil.rmtree(tmpdir)
            except Exception as e:
                print(f"Failed to remove tempdir {tmpdir}: {e}")


if __name__ == '__main__':
    main()
