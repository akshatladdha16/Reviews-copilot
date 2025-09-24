from fastapi import HTTPException, Depends, status
from fastapi.security import APIKeyHeader, HTTPBearer
from typing import Dict, Optional
import os
from enum import Enum

class UserRole(str, Enum):
    ADMIN = "admin"
    ANALYTICS = "analytics"
    PUBLIC = "public"

# API key headers
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)
http_bearer = HTTPBearer(auto_error=False)

# Valid API keys from environment
VALID_API_KEYS = {
    'admin123': UserRole.ADMIN,
    'analytics123': UserRole.ANALYTICS,
}

class AuthService:
    def __init__(self):
        self.valid_keys = VALID_API_KEYS
    
    async def get_current_user(self, api_key: str = Depends(api_key_header)) -> Dict[str, str]:
        """Validate API key and return user role"""
        if not api_key:
            return {"role": UserRole.PUBLIC}
        
        if api_key in self.valid_keys:
            return {"role": self.valid_keys[api_key], "api_key": api_key}
        
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    
    def require_role(self, required_role: UserRole):
        """Dependency to require specific role"""
        async def role_dependency(user: Dict = Depends(self.get_current_user)):
            if user["role"] == UserRole.ADMIN:
                return user
            
            if required_role == UserRole.ANALYTICS and user["role"] in [UserRole.ANALYTICS, UserRole.ADMIN]:
                return user
            
            if user["role"] != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required role: {required_role}"
                )
            return user
        return role_dependency

# Global auth service instance
auth_service = AuthService()

# Dependency shortcuts
require_admin = auth_service.require_role(UserRole.ADMIN)
require_analytics = auth_service.require_role(UserRole.ANALYTICS)
get_user = auth_service.get_current_user