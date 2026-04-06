"""add_product_name_and_description_to_organizations

Revision ID: 19eb5726aba2
Revises: cce4bec49b7d
Create Date: 2026-04-06 15:30:45.505132

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision: str = '19eb5726aba2'
down_revision: Union[str, Sequence[str], None] = 'cce4bec49b7d'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('organizations', sa.Column('product_name', sa.String(length=255), nullable=True))
    op.add_column('organizations', sa.Column('description', sa.String(length=1000), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('organizations', 'description')
    op.drop_column('organizations', 'product_name')
