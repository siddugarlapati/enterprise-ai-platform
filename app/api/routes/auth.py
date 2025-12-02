from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession
from datetime import datetime
import secrets

from app.db.connection import get_db
from app.db.crud import user_crud
from app.schemas.schemas import (
    UserCreate, UserLogin, UserResponse, Token, APIKeyCreate, APIKeyResponse
)
from app.api.middleware.auth import (
    hash_password, verify_password, create_access_token, create_refresh_token,
    decode_token, get_current_user, generate_api_key, hash_api_key, require_admin
)
from app.models.database import User, APIKey

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register", response_model=UserResponse, status_code=status.HTTP_201_CREATED)
async def register(user_data: UserCreate, db: AsyncSession = Depends(get_db)):
    """Register a new user."""
    # Check if user exists
    existing_user = await user_crud.get_by_email(db, user_data.email)
    if existing_user:
        raise HTTPException(status_code=400, detail="Email already registered")
    
    existing_username = await user_crud.get_by_username(db, user_data.username)
    if existing_username:
        raise HTTPException(status_code=400, detail="Username already taken")
    
    # Create user
    user_dict = {
        "username": user_data.username,
        "email": user_data.email,
        "hashed_password": hash_password(user_data.password),
        "full_name": user_data.full_name,
        "role": "user"
    }
    
    user = await user_crud.create(db, user_dict)
    await db.commit()
    
    return user

@router.post("/login", response_model=Token)
async def login(credentials: UserLogin, db: AsyncSession = Depends(get_db)):
    """Login and get access token."""
    user = await user_crud.get_by_email(db, credentials.email)
    
    if not user or not verify_password(credentials.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    
    # Update last login
    await user_crud.update(db, user.id, {"last_login": datetime.utcnow()})
    await db.commit()
    
    # Create tokens
    token_data = {"sub": user.id, "email": user.email, "role": user.role}
    access_token = create_access_token(token_data)
    refresh_token = create_refresh_token(token_data)
    
    return Token(access_token=access_token, refresh_token=refresh_token)

@router.post("/refresh", response_model=Token)
async def refresh_token(refresh_token: str, db: AsyncSession = Depends(get_db)):
    """Refresh access token."""
    payload = decode_token(refresh_token)
    
    if payload.get("type") != "refresh":
        raise HTTPException(status_code=401, detail="Invalid token type")
    
    user = await user_crud.get(db, payload.get("sub"))
    if not user or not user.is_active:
        raise HTTPException(status_code=401, detail="User not found or inactive")
    
    token_data = {"sub": user.id, "email": user.email, "role": user.role}
    new_access_token = create_access_token(token_data)
    new_refresh_token = create_refresh_token(token_data)
    
    return Token(access_token=new_access_token, refresh_token=new_refresh_token)

@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Get current user information."""
    user = await user_crud.get(db, current_user["user_id"])
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put("/me", response_model=UserResponse)
async def update_current_user(
    full_name: str = None,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Update current user information."""
    update_data = {}
    if full_name:
        update_data["full_name"] = full_name
    
    if update_data:
        await user_crud.update(db, current_user["user_id"], update_data)
        await db.commit()
    
    return await user_crud.get(db, current_user["user_id"])

@router.post("/api-keys", response_model=APIKeyResponse)
async def create_api_key(
    key_data: APIKeyCreate,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Create a new API key."""
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)
    
    from datetime import timedelta
    expires_at = None
    if key_data.expires_days:
        expires_at = datetime.utcnow() + timedelta(days=key_data.expires_days)
    
    api_key = APIKey(
        user_id=current_user["user_id"],
        key_hash=key_hash,
        name=key_data.name,
        scopes=key_data.scopes,
        expires_at=expires_at
    )
    
    db.add(api_key)
    await db.commit()
    await db.refresh(api_key)
    
    return APIKeyResponse(
        id=api_key.id,
        name=api_key.name,
        key=raw_key,  # Only returned once
        scopes=api_key.scopes,
        expires_at=api_key.expires_at,
        created_at=api_key.created_at
    )

@router.delete("/api-keys/{key_id}")
async def revoke_api_key(
    key_id: int,
    current_user: dict = Depends(get_current_user),
    db: AsyncSession = Depends(get_db)
):
    """Revoke an API key."""
    from sqlalchemy import select
    result = await db.execute(
        select(APIKey).where(APIKey.id == key_id, APIKey.user_id == current_user["user_id"])
    )
    api_key = result.scalar_one_or_none()
    
    if not api_key:
        raise HTTPException(status_code=404, detail="API key not found")
    
    await db.delete(api_key)
    await db.commit()
    
    return {"message": "API key revoked"}
