import gzip
import shutil
import os
import sys
from pathlib import Path


def uncompress_folder(folder_path, extension=".gz"):
    """
    Recursively finds and uncompresses files in a directory.
    """
    root_dir = Path(folder_path)

    if not root_dir.exists():
        print(f"Error: Directory '{root_dir}' not found.")
        return

    print(f"Scanning {root_dir} for {extension} files...")

    count = 0
    # Walk through all subdirectories
    for file_path in root_dir.rglob(f"*{extension}"):
        # Define output filename (remove the .gz extension)
        output_path = file_path.with_suffix('')

        print(f"Extracting: {file_path.name} -> {output_path.name}")

        try:
            with gzip.open(file_path, 'rb') as f_in:
                with open(output_path, 'wb') as f_out:
                    # noinspection PyTypeChecker
                    shutil.copyfileobj(f_in, f_out)

            # Optional: Delete the original .gz file after extraction to save space
            os.remove(file_path)

            count += 1
        except Exception as e:
            print(f"Failed to extract {file_path}: {e}")

    print(f"Finished. Extracted {count} files.")


if __name__ == "__main__":
    # Usage: python unzip_all.py ./data/mimic4
    if len(sys.argv) > 1:
        target_folder = sys.argv[1]
    else:
        target_folder = input("Enter folder path to uncompress: ")

    uncompress_folder(target_folder)