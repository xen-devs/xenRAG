from typing import Optional

from pydantic import BaseModel, Field


class SignUpRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., min_length=6, description="Password (min 6 chars)")
    name: str = Field(..., min_length=1, description="User display name")
    org_name: Optional[str] = Field(None, min_length=1, description="Organization name (optional, can be created during onboarding)")


class CreateOrgRequest(BaseModel):
    name: str = Field(..., min_length=1, max_length=255, description="Organization name")
    product_name: Optional[str] = Field(None, max_length=255, description="Product name")
    description: Optional[str] = Field(None, max_length=1000, description="Product/org description for AI context")


class SignInRequest(BaseModel):
    email: str = Field(..., description="User email address")
    password: str = Field(..., description="User password")


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"


class UserOrgOut(BaseModel):
    id: str
    name: str
    role: str
    product_name: Optional[str] = None
    description: Optional[str] = None


class MeResponse(BaseModel):
    id: str
    email: str
    name: str
    orgs: list[UserOrgOut]