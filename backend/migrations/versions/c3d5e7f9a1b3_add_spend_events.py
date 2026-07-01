"""add spend_events (durable daily LLM cost ceiling)

Revision ID: c3d5e7f9a1b3
Revises: a1b2c3d4e5f6
Create Date: 2026-07-01 22:00:00.000000

"""
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "c3d5e7f9a1b3"
down_revision: Union[str, None] = "a1b2c3d4e5f6"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "spend_events",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("ts", sa.Float(), nullable=False),
        sa.Column("usd", sa.Float(), nullable=False),
    )
    op.create_index("ix_spend_events_ts", "spend_events", ["ts"])


def downgrade() -> None:
    op.drop_index("ix_spend_events_ts", "spend_events")
    op.drop_table("spend_events")
