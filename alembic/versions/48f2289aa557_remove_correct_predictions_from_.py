"""remove correct_predictions from evaluations

Revision ID: 48f2289aa557
Revises: f36c3c97011d
Create Date: 2026-03-18 10:06:42.127642

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '48f2289aa557'
down_revision = 'f36c3c97011d'
branch_labels = None
depends_on = None


def upgrade():
    # No-op: correct_predictions column is still needed
    pass


def downgrade():
    pass
