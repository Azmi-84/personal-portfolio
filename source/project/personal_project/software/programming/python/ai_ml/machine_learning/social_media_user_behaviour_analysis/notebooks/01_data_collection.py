import re
import shutil
import kagglehub
from pathlib import Path

# Download latest version
path = kagglehub.dataset_download("rockyt07/social-media-user-analysis")

source_path = Path(path)
target_path = Path("data/raw/")

target_path.mkdir(parents=True, exist_ok=True)

for file_path in source_path.iterdir():
    if file_path.is_file():
        # Get the stem (filename without extension) and extension
        stem = file_path.stem
        suffix = file_path.suffix

        # Remove any non-alphanumeric characters and convert to lowercase
        clean_stem = re.sub(r'[^a-zA-Z0-9]', '_', stem).lower()

        # Split into parts and join with underscore
        parts = clean_stem.split('_')
        # Remove empty parts (from multiple underscores)
        parts = [p for p in parts if p]

        # Join with underscore
        new_stem = '_'.join(parts)
        new_name = new_stem + suffix

        dest_path = target_path / new_name
        shutil.copy2(file_path, dest_path)
        print(f"Copied: {file_path.name} -> {new_name}")
