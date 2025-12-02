from fastapi import Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from jose import jwt, JWTError
from datetime import datetime, timedelta
from passlib.context import CryptContext
from typing import Optional, Dict, Any
import hashlib
import secrets

from app.config.settings import settings
from app.config.constants import ROLE_PERMISSIONS

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def hash_password(password: str) -> str:
    return pwd_context.hash(password)

def verify_password(plain_password: str, hashed_password: str) -> bool:
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire, "type": "access"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def create_refresh_token(data: Dict[str, Any]) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    to_encode.update({"exp": expire, "type": "refresh"})
    return jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

def decode_token(token: str) -> Dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def generate_api_key() -> str:
    return f"sk-{secrets.token_urlsafe(32)}"

def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode()).hexdigest()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> Dict[str, Any]:
    token = credentials.credentials
    payload = decode_token(token)
    
    if payload.get("type") != "access":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    return {
        "user_id": payload.get("sub"),
        "role": payload.get("role", "user"),
        "email": payload.get("email")
    }

async def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(HTTPBearer(auto_error=False)),
    api_key: Optional[str] = Depends(api_key_header)
) -> Optional[Dict[str, Any]]:
    if credentials:
        try:
            return await get_current_user(credentials)
        except HTTPException:
            pass
    
    if api_key:
        # API key validation would check against database
        return {"user_id": None, "role": "api_user", "api_key": api_key}
    
    return None

def require_role(required_roles: list):
    async def role_checker(current_user: Dict[str, Any] = Depends(get_current_user)):
        user_role = current_user.get("role", "user")
        if user_role not in required_roles:
            raise HTTPException(status_code=403, detail="Insufficient permissions")
        return current_user
    return role_checker

def check_permission(user_role: str, permission: str) -> bool:
    permissions = ROLE_PERMISSIONS.get(user_role, [])
    return permission in permissions

class RoleChecker:
    def __init__(self, allowed_roles: list):
        self.allowed_roles = allowed_roles

    async def __call__(self, current_user: Dict[str, Any] = Depends(get_current_user)):
        if current_user.get("role") not in self.allowed_roles:
            raise HTTPException(status_code=403, detail="Access denied")
        return current_user

# Role dependency shortcuts
require_admin = RoleChecker(["admin"])
require_model_manager = RoleChecker(["admin", "model_manager"])
require_analyst = RoleChecker(["admin", "model_manager", "analyst"])
