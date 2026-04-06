from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from api.auth.schemas import MeResponse, SignUpRequest, UserOrgOut
from api.auth.security import hash_password, verify_password
from api.db.models import Organization, User, UserOrganization


async def create_user_and_org(db: AsyncSession, req: SignUpRequest) -> User:
    user = User(
        email=req.email,
        name=req.name,
        password_hash=hash_password(req.password),
    )
    db.add(user)
    await db.flush()

    if req.org_name:
        org = Organization(name=req.org_name, created_by=user.id)
        db.add(org)
        await db.flush()

        membership = UserOrganization(user_id=user.id, org_id=org.id, role="owner")
        db.add(membership)

    await db.flush()
    await db.refresh(user)
    return user


async def create_organization(
    db: AsyncSession,
    user_id: str,
    name: str,
    product_name: str | None = None,
    description: str | None = None,
) -> Organization:
    org = Organization(name=name, product_name=product_name, description=description, created_by=user_id)
    db.add(org)
    await db.flush()

    membership = UserOrganization(user_id=user_id, org_id=org.id, role="owner")
    db.add(membership)

    await db.flush()
    await db.refresh(org)
    return org


async def authenticate_user(db: AsyncSession, email: str, password: str) -> User | None:
    result = await db.execute(select(User).where(User.email == email))
    user = result.scalar_one_or_none()
    if user is None or not verify_password(password, user.password_hash):
        return None
    return user


async def get_user_with_orgs(db: AsyncSession, user_id: str) -> MeResponse | None:
    result = await db.execute(select(User).where(User.id == user_id))
    user = result.scalar_one_or_none()
    if user is None:
        return None

    orgs = [
        UserOrgOut(
            id=str(m.org_id),
            name=m.organization.name,
            role=m.role,
            product_name=m.organization.product_name,
            description=m.organization.description,
        )
        for m in user.memberships
    ]

    return MeResponse(id=str(user.id), email=user.email, name=user.name, orgs=orgs)