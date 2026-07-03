"""add klist_schema_items (visitor-evolvable checklist sections/tiles)

Revision ID: e5f7a9b1c3d5
Revises: d4e6f8a0b2c4
Create Date: 2026-07-02
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "e5f7a9b1c3d5"
down_revision: Union[str, None] = "d4e6f8a0b2c4"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "klist_schema_items",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("section", sa.String(length=80), nullable=False),
        sa.Column("item", sa.String(length=120), nullable=False),
        sa.Column("ip", sa.String(length=64), nullable=True),
        sa.Column("created_at", sa.DateTime(), nullable=False),
    )


def downgrade() -> None:
    op.drop_table("klist_schema_items")
