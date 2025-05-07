import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Optional

def calculate_file_hash(file_path: str) -> str:
    """Calculate SHA-256 hash of a file."""
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
    return sha256_hash.hexdigest()

def load_hash_cache(cache_file: str) -> Dict[str, str]:
    """Load the hash cache from a JSON file."""
    if os.path.exists(cache_file):
        with open(cache_file, 'r') as f:
            return json.load(f)
    return {}

def save_hash_cache(cache_file: str, cache: Dict[str, str]) -> None:
    """Save the hash cache to a JSON file."""
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)

def get_cache_file_path(component: str) -> str:
    """Get the path for a component's hash cache file."""
    cache_dir = Path(__file__).parent / "cache"
    os.makedirs(cache_dir, exist_ok=True)
    return str(cache_dir / f"{component}_hashes.json")

def check_file_hash(file_path: str, component: str) -> bool:
    """Check if a file's hash has changed since last processing."""
    cache_file = get_cache_file_path(component)
    cache = load_hash_cache(cache_file)

    current_hash = calculate_file_hash(file_path)
    cached_hash = cache.get(file_path)

    if cached_hash == current_hash:
        return True

    # Update cache with new hash
    cache[file_path] = current_hash
    save_hash_cache(cache_file, cache)
    return False
