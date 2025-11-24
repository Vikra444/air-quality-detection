"""
Authentication and authorization utilities for AirGuard.
"""

from datetime import datetime, timedelta
from typing import Optional, Dict, Any
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import secrets
import re
from sqlalchemy import select
from ..config.settings import settings
from ..utils.logger import get_logger
from ..data.storage import storage, User as UserDB
from ..data.models import UserCreate, UserLogin, User

logger = get_logger("security.auth")

# Password hashing context with explicit bcrypt version handling
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12,
    bcrypt__ident="2b"  # Explicitly set bcrypt identifier
)


# JWT token creation and verification
oauth2_scheme = HTTPBearer(auto_error=False)


class TokenData:
    """Token data structure."""
    def __init__(self, username: str = None, scopes: list = None):
        self.username = username
        self.scopes = scopes or []


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a plain password against a hashed password."""
    # Truncate password to 72 bytes if needed for bcrypt compatibility
    if len(plain_password.encode('utf-8')) > 72:
        plain_password = plain_password[:72]
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password: str) -> str:
    """Hash a password."""
    # Debug logging
    print(f"get_password_hash called with password: {password}")
    print(f"Password length in bytes: {len(password.encode('utf-8'))}")
    print(f"Password type: {type(password)}")
    
    # Truncate password to 72 bytes if needed for bcrypt compatibility
    if len(password.encode('utf-8')) > 72:
        password = password[:72]
        print(f"Truncated password: {password}")
    return pwd_context.hash(password)


def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Create a JWT access token."""
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.jwt_secret_key, algorithm=settings.algorithm)
    return encoded_jwt


def verify_access_token(token: str) -> Optional[Dict[str, Any]]:
    """Verify a JWT access token and return payload."""
    try:
        payload = jwt.decode(token, settings.jwt_secret_key, algorithms=[settings.algorithm])
        return payload
    except JWTError as e:
        logger.warning(f"Invalid token: {e}")
        return None


async def get_current_user(credentials: Optional[HTTPAuthorizationCredentials] = Depends(oauth2_scheme)):
    """Get current user from JWT token."""
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    token = credentials.credentials
    payload = verify_access_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return payload


def generate_api_key() -> str:
    """Generate a secure API key."""
    return secrets.token_urlsafe(32)


class SecurityManager:
    """Manage security operations for the application."""
    
    def __init__(self):
        self.users = {}  # In production, this would be a database
        self.api_keys = {}  # In production, this would be a database
    
    async def create_user(self, user_data: UserCreate) -> User:
        """Create a new user in the database."""
        # Validate mobile uniqueness
        async with storage.async_session() as session:
            result = await session.execute(
                select(UserDB).where(UserDB.mobile == user_data.mobile)
            )
            existing_user = result.scalar_one_or_none()
            if existing_user:
                raise HTTPException(
                    status_code=400,
                    detail="Mobile number already registered"
                )
            
            # Validate email uniqueness if provided
            if user_data.email:
                result = await session.execute(
                    select(UserDB).where(UserDB.email == user_data.email)
                )
                existing_email = result.scalar_one_or_none()
                if existing_email:
                    raise HTTPException(
                        status_code=400,
                        detail="Email already registered"
                    )
            
            # Hash password
            hashed_password = get_password_hash(user_data.password)
            
            # Create user record
            user_record = UserDB(
                full_name=user_data.full_name,
                mobile=user_data.mobile,
                email=user_data.email,
                address=user_data.address,
                age=user_data.age,
                chronic_conditions=user_data.chronic_conditions,
                is_smoker=user_data.is_smoker or False,
                works_outdoors=user_data.works_outdoors or False,
                password_hash=hashed_password
            )
            
            session.add(user_record)
            await session.commit()
            await session.refresh(user_record)
            
            # Return user without password hash
            return User(
                id=user_record.id,
                full_name=user_record.full_name,
                mobile=user_record.mobile,
                email=user_record.email,
                address=user_record.address,
                age=user_record.age,
                chronic_conditions=user_record.chronic_conditions,
                is_smoker=user_record.is_smoker,
                works_outdoors=user_record.works_outdoors,
                created_at=user_record.created_at
            )
    
    async def authenticate_user(self, identifier: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate a user by mobile or email."""
        async with storage.async_session() as session:
            # Try to find user by mobile first
            result = await session.execute(
                select(UserDB).where(UserDB.mobile == identifier)
            )
            user = result.scalar_one_or_none()
            
            # If not found by mobile, try email
            if not user and '@' in identifier:
                result = await session.execute(
                    select(UserDB).where(UserDB.email == identifier)
                )
                user = result.scalar_one_or_none()
            
            if not user:
                return None
            
            # Truncate password to 72 bytes if needed for bcrypt compatibility
            if len(password.encode('utf-8')) > 72:
                password = password[:72]
                
            if verify_password(password, user.password_hash):
                return {
                    "id": user.id,
                    "username": user.mobile,  # Using mobile as username for JWT
                    "full_name": user.full_name,
                    "roles": ["user"],  # Default role
                    "is_active": True  # Default to active since there's no is_active field in DB
                }
        
        return None
    
    async def get_user_by_id(self, user_id: int) -> Optional[User]:
        """Get user by ID."""
        async with storage.async_session() as session:
            result = await session.execute(
                select(UserDB).where(UserDB.id == user_id)
            )
            user_record = result.scalar_one_or_none()
            
            if not user_record:
                return None
            
            return User(
                id=user_record.id,
                full_name=user_record.full_name,
                mobile=user_record.mobile,
                email=user_record.email,
                address=user_record.address,
                age=user_record.age,
                chronic_conditions=user_record.chronic_conditions,
                is_smoker=user_record.is_smoker,
                works_outdoors=user_record.works_outdoors,
                created_at=user_record.created_at
            )
    
    def create_access_token_for_user(self, user: Dict[str, Any]) -> str:
        """Create an access token for a user."""
        access_token_expires = timedelta(minutes=settings.access_token_expire_minutes)
        token_data = {
            "sub": str(user["id"]),  # Use user ID as subject
            "username": user["username"],
            "full_name": user["full_name"],
            "roles": user["roles"],
            "type": "access"
        }
        return create_access_token(token_data, expires_delta=access_token_expires)
    
    def create_api_key(self, username: str, name: str = "default") -> str:
        """Create an API key for a user."""
        api_key = generate_api_key()
        self.api_keys[api_key] = {
            "username": username,
            "name": name,
            "created_at": datetime.utcnow(),
            "last_used": None
        }
        logger.info(f"Created API key for user: {username}")
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[Dict[str, Any]]:
        """Verify an API key."""
        key_data = self.api_keys.get(api_key)
        if not key_data:
            return None
        
        # Update last used timestamp
        key_data["last_used"] = datetime.utcnow()
        return key_data
    
    def has_role(self, user: Dict[str, Any], required_role: str) -> bool:
        """Check if user has a required role."""
        return required_role in user.get("roles", [])
    
    def has_permission(self, user: Dict[str, Any], required_permission: str) -> bool:
        """Check if user has a required permission."""
        # For simplicity, we'll map roles to permissions
        role_permissions = {
            "admin": ["read", "write", "delete", "admin"],
            "user": ["read", "write"],
            "viewer": ["read"]
        }
        
        user_roles = user.get("roles", [])
        for role in user_roles:
            if required_permission in role_permissions.get(role, []):
                return True
        return False


# Global security manager instance
security_manager = SecurityManager()

# Create default admin user for demo purposes
try:
    # This will be handled by the API now, so we don't need to create a default user here
    pass
except ValueError:
    pass  # User already exists