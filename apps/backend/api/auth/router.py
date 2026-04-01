import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.dependencies import get_current_user

logger = logging.getLogger(__name__)
from api.auth.schemas import CreateOrgRequest, MeResponse, SignInRequest, SignUpRequest, TokenResponse, UserOrgOut
from api.auth.security import create_access_token
from api.auth.service import authenticate_user, create_organization, create_user_and_org, get_user_with_orgs
from api.db.session import get_session
from api.db.models import User

auth_router = APIRouter(prefix="/auth", tags=["auth"])


@auth_router.post("/signup", response_model=TokenResponse, status_code=status.HTTP_201_CREATED)
async def signup(req: SignUpRequest, db: AsyncSession = Depends(get_session)):
    try:
        user = await create_user_and_org(db, req)
    except IntegrityError as exc:
        orig = getattr(exc, "orig", None)
        msg = (str(orig) if orig else str(exc)).lower()
        logger.warning("signup integrity error: %s", orig or exc)
        if "email" in msg or "users_email" in msg or "ix_users_email" in msg:
            detail = "Email already registered"
        else:
            detail = "Could not complete signup (database conflict). Check server logs."
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=detail) from exc

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@auth_router.post("/signin", response_model=TokenResponse)
async def signin(req: SignInRequest, db: AsyncSession = Depends(get_session)):
    user = await authenticate_user(db, req.email, req.password)
    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
        )

    token = create_access_token({"sub": str(user.id)})
    return TokenResponse(access_token=token)


@auth_router.get("/me", response_model=MeResponse)
async def me(current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_session)):
    response = await get_user_with_orgs(db, current_user.id)
    if response is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="User not found")
    return response


orgs_router = APIRouter(prefix="/orgs", tags=["organizations"])


@orgs_router.post("", response_model=UserOrgOut, status_code=status.HTTP_201_CREATED)
async def create_org(
    req: CreateOrgRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_session),
):
    org = await create_organization(db, current_user.id, req.name)
    return UserOrgOut(id=str(org.id), name=org.name, role="owner")