"""update cols

Revision ID: 8bb576dae899
Revises: f36c3c97011d
Create Date: 2022-03-17 15:56:30.700399

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '8bb576dae899'
down_revision = 'f36c3c97011d'
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('teams', sa.Column('FirstD1Season', sa.Integer(), nullable=True))
    op.add_column('teams', sa.Column('LastD1Season', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('teams', 'LastD1Season')
    op.drop_column('teams', 'FirstD1Season')
    # ### end Alembic commands ###