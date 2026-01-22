"""
File Utilities Module
Helper functions for file operations
"""

from pathlib import Path
from typing import Optional
import os


def get_file_size_mb(file_path: Path) -> float:
    """
    Get file size in megabytes
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in MB
    """
    return file_path.stat().st_size / (1024 * 1024)


def ensure_directory_exists(directory: Path):
    """
    Ensure a directory exists, create if it doesn't
    
    Args:
        directory: Path to directory
    """
    directory.mkdir(parents=True, exist_ok=True)


def get_safe_filename(filename: str) -> str:
    """
    Get a safe version of filename (remove invalid characters)
    
    Args:
        filename: Original filename
        
    Returns:
        Safe filename
    """
    # Remove or replace invalid characters
    invalid_chars = '<>:"/\\|?*'
    for char in invalid_chars:
        filename = filename.replace(char, '_')
    return filename


def get_unique_filepath(base_path: Path) -> Path:
    """
    Get a unique filepath by appending numbers if file exists
    
    Args:
        base_path: Base file path
        
    Returns:
        Unique file path
    """
    if not base_path.exists():
        return base_path
    
    stem = base_path.stem
    suffix = base_path.suffix
    parent = base_path.parent
    
    counter = 1
    while True:
        new_path = parent / f"{stem}_{counter}{suffix}"
        if not new_path.exists():
            return new_path
        counter += 1


def validate_csv_extension(file_path: Path) -> bool:
    """
    Check if file has .csv extension
    
    Args:
        file_path: Path to file
        
    Returns:
        True if file is CSV, False otherwise
    """
    return file_path.suffix.lower() == '.csv'
