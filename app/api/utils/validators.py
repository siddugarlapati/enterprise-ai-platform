import re
from typing import List, Optional, Tuple
from fastapi import HTTPException


def validate_email(email: str) -> bool:
    """Validate email format."""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email))


def validate_password(password: str) -> Tuple[bool, Optional[str]]:
    """Validate password strength."""
    if len(password) < 8:
        return False, "Password must be at least 8 characters"
    if not re.search(r'[A-Z]', password):
        return False, "Password must contain at least one uppercase letter"
    if not re.search(r'[a-z]', password):
        return False, "Password must contain at least one lowercase letter"
    if not re.search(r'\d', password):
        return False, "Password must contain at least one digit"
    return True, None


def validate_username(username: str) -> Tuple[bool, Optional[str]]:
    """Validate username format."""
    if len(username) < 3:
        return False, "Username must be at least 3 characters"
    if len(username) > 50:
        return False, "Username must be at most 50 characters"
    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"
    return True, None


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """Sanitize text input."""
    # Remove null bytes
    text = text.replace('\x00', '')
    # Truncate to max length
    text = text[:max_length]
    # Strip leading/trailing whitespace
    text = text.strip()
    return text


def validate_labels(labels: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate classification labels."""
    if len(labels) < 2:
        return False, "At least 2 labels required"
    if len(labels) > 20:
        return False, "Maximum 20 labels allowed"
    for label in labels:
        if len(label) > 100:
            return False, "Label must be at most 100 characters"
    return True, None


def validate_pagination(page: int, page_size: int, max_page_size: int = 100) -> Tuple[int, int]:
    """Validate and normalize pagination parameters."""
    page = max(1, page)
    page_size = min(max(1, page_size), max_page_size)
    skip = (page - 1) * page_size
    return skip, page_size


def validate_file_extension(filename: str, allowed_extensions: List[str]) -> bool:
    """Validate file extension."""
    if not filename:
        return False
    ext = filename.rsplit('.', 1)[-1].lower()
    return ext in allowed_extensions


def validate_json_structure(data: dict, required_fields: List[str]) -> Tuple[bool, Optional[str]]:
    """Validate JSON structure has required fields."""
    missing = [f for f in required_fields if f not in data]
    if missing:
        return False, f"Missing required fields: {', '.join(missing)}"
    return True, None


def validate_model_name(name: str) -> Tuple[bool, Optional[str]]:
    """Validate model name format."""
    if len(name) < 2:
        return False, "Model name must be at least 2 characters"
    if len(name) > 100:
        return False, "Model name must be at most 100 characters"
    if not re.match(r'^[a-zA-Z][a-zA-Z0-9_-]*$', name):
        return False, "Model name must start with a letter and contain only letters, numbers, underscores, and hyphens"
    return True, None


def validate_api_key_format(api_key: str) -> bool:
    """Validate API key format."""
    return api_key.startswith('sk-') and len(api_key) >= 20
