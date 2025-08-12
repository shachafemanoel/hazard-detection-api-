"""
Authentication API endpoints
Handles login, registration, and authentication health checks with proper CORS and cookie support
"""

import logging
import secrets
from datetime import datetime, timedelta
from typing import Optional

from fastapi import APIRouter, HTTPException, Response, Request, Depends, status
from pydantic import BaseModel, EmailStr

from ..core.config import settings
from ..core.logging_config import get_logger

logger = get_logger("auth_api")

router = APIRouter(prefix="/auth", tags=["authentication"])


# Request/Response models
class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RegisterRequest(BaseModel):
    email: EmailStr
    username: str
    password: str


class AuthResponse(BaseModel):
    user: dict
    sessionId: str


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


# Simple in-memory user store for demonstration
# In production, this would be a proper database
USERS = {}
SESSIONS = {}


@router.options("/health", include_in_schema=False)
async def auth_health_options() -> Response:
    """Handle OPTIONS requests for auth health checks"""
    return Response(status_code=200)


@router.get("/health")
async def auth_health() -> HealthResponse:
    """
    Authentication service health check endpoint
    Used by client to verify auth service availability and detect CORS issues early
    """
    try:
        logger.info("Auth health check requested")
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            version=settings.app_version
        )
    except Exception as e:
        logger.error(f"Auth health check failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Authentication service health check failed"
        )


@router.options("/login", include_in_schema=False)
async def login_options() -> Response:
    """Handle OPTIONS requests for login endpoint"""
    return Response(status_code=200)


@router.post("/login")
async def login(request: LoginRequest, response: Response) -> AuthResponse:
    """
    User login endpoint with secure cookie handling
    Sets HttpOnly, Secure, SameSite=None cookies for cross-origin authentication
    """
    try:
        logger.info(f"Login attempt for email: {request.email}")
        
        # Check if user exists (simplified validation)
        user_key = request.email.lower()
        if user_key not in USERS:
            logger.warning(f"Login failed: user not found for {request.email}")
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        stored_user = USERS[user_key]
        
        # Simple password check (in production, use proper hashing)
        if stored_user["password"] != request.password:
            logger.warning(f"Login failed: invalid password for {request.email}")
            raise HTTPException(
                status_code=401,
                detail="Invalid email or password"
            )
        
        # Generate session
        session_id = secrets.token_urlsafe(32)
        session_data = {
            "user_id": user_key,
            "email": request.email,
            "username": stored_user["username"],
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=7)
        }
        
        SESSIONS[session_id] = session_data
        
        # Set secure cookie
        cookie_value = f"session_id={session_id}"
        response.set_cookie(
            key="hazard_session",
            value=session_id,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=True,  # HTTPS only
            samesite="none",  # Allow cross-origin
            path="/"
        )
        
        logger.info(f"Login successful for {request.email}, session: {session_id[:8]}...")
        
        return AuthResponse(
            user={
                "email": request.email,
                "username": stored_user["username"],
                "id": user_key
            },
            sessionId=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Login failed due to server error"
        )


@router.options("/register", include_in_schema=False)
async def register_options() -> Response:
    """Handle OPTIONS requests for register endpoint"""
    return Response(status_code=200)


@router.post("/register")
async def register(request: RegisterRequest, response: Response) -> AuthResponse:
    """
    User registration endpoint with automatic login
    Creates new user account and sets authentication cookie
    """
    try:
        logger.info(f"Registration attempt for email: {request.email}")
        
        # Check if user already exists
        user_key = request.email.lower()
        if user_key in USERS:
            logger.warning(f"Registration failed: user exists for {request.email}")
            raise HTTPException(
                status_code=400,
                detail="User already exists with this email"
            )
        
        # Validate password (basic validation)
        if len(request.password) < 8:
            raise HTTPException(
                status_code=400,
                detail="Password must be at least 8 characters long"
            )
        
        # Create user (in production, hash the password)
        user_data = {
            "email": request.email,
            "username": request.username,
            "password": request.password,  # In production: hash this!
            "created_at": datetime.now()
        }
        
        USERS[user_key] = user_data
        
        # Generate session for auto-login
        session_id = secrets.token_urlsafe(32)
        session_data = {
            "user_id": user_key,
            "email": request.email,
            "username": request.username,
            "created_at": datetime.now(),
            "expires_at": datetime.now() + timedelta(days=7)
        }
        
        SESSIONS[session_id] = session_data
        
        # Set secure cookie
        response.set_cookie(
            key="hazard_session",
            value=session_id,
            max_age=7 * 24 * 60 * 60,  # 7 days
            httponly=True,
            secure=True,  # HTTPS only
            samesite="none",  # Allow cross-origin
            path="/"
        )
        
        logger.info(f"Registration successful for {request.email}, session: {session_id[:8]}...")
        
        return AuthResponse(
            user={
                "email": request.email,
                "username": request.username,
                "id": user_key
            },
            sessionId=session_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Registration error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Registration failed due to server error"
        )


@router.options("/logout", include_in_schema=False)
async def logout_options() -> Response:
    """Handle OPTIONS requests for logout endpoint"""
    return Response(status_code=200)


@router.post("/logout")
async def logout(request: Request, response: Response):
    """
    User logout endpoint
    Invalidates session and clears authentication cookie
    """
    try:
        # Get session from cookie
        session_cookie = request.cookies.get("hazard_session")
        
        if session_cookie and session_cookie in SESSIONS:
            # Remove session
            session_data = SESSIONS.pop(session_cookie, None)
            if session_data:
                logger.info(f"Logout successful for user: {session_data.get('email', 'unknown')}")
        
        # Clear cookie
        response.delete_cookie(
            key="hazard_session",
            path="/",
            secure=True,
            samesite="none"
        )
        
        return {"message": "Logged out successfully"}
        
    except Exception as e:
        logger.error(f"Logout error: {e}")
        # Still clear the cookie even if there was an error
        response.delete_cookie(
            key="hazard_session",
            path="/",
            secure=True,
            samesite="none"
        )
        return {"message": "Logged out"}


@router.options("/forgot-password", include_in_schema=False)
async def forgot_password_options() -> Response:
    """Handle OPTIONS requests for forgot password endpoint"""
    return Response(status_code=200)


@router.post("/forgot-password")
async def forgot_password(request: dict):
    """
    Password reset request endpoint
    In production, this would send an email with reset link
    """
    try:
        email = request.get("email")
        if not email:
            raise HTTPException(
                status_code=400,
                detail="Email is required"
            )
        
        logger.info(f"Password reset requested for: {email}")
        
        # In production, check if user exists and send email
        # For now, just return success regardless
        
        return {"message": "If the email is registered, you will receive a password reset link shortly"}
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Forgot password error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Password reset request failed"
        )


def get_current_user(request: Request) -> Optional[dict]:
    """
    Dependency to get current authenticated user from session cookie
    """
    try:
        session_cookie = request.cookies.get("hazard_session")
        if not session_cookie or session_cookie not in SESSIONS:
            return None
        
        session_data = SESSIONS[session_cookie]
        
        # Check if session is expired
        if datetime.now() > session_data["expires_at"]:
            SESSIONS.pop(session_cookie, None)
            return None
        
        return {
            "email": session_data["email"],
            "username": session_data["username"],
            "user_id": session_data["user_id"]
        }
        
    except Exception as e:
        logger.error(f"Error getting current user: {e}")
        return None


@router.get("/me")
async def get_current_user_info(current_user: dict = Depends(get_current_user)):
    """
    Get current authenticated user information
    """
    if not current_user:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated"
        )
    
    return {"user": current_user}