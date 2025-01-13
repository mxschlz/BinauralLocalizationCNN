from pathlib import Path


def get_unique_folder_name(base_name):
    """
    Generate a unique folder name. Use the base name if it doesn't exist,
    otherwise append a numeric suffix to ensure uniqueness.
    """
    base_path = Path(base_name)
    if not base_path.exists():
        return base_path  # Return the base name directly if it doesn't exist

    counter = 1
    while (folder_path := base_path.with_name(f"{base_path.stem}_{counter}")).exists():
        counter += 1
    return folder_path