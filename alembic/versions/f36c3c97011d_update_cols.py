"""update cols

Revision ID: f36c3c97011d
Revises: 
Create Date: 2022-03-16 15:56:51.807938

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = 'f36c3c97011d'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.add_column('matches', sa.Column('WFGP_avg', sa.Float(), nullable=True))
    op.add_column('matches', sa.Column('WFGP3_avg', sa.Float(), nullable=True))
    op.add_column('matches', sa.Column('WR_avg', sa.Integer(), nullable=True))
    op.add_column('matches', sa.Column('LFGP_avg', sa.Float(), nullable=True))
    op.add_column('matches', sa.Column('LFGP3_avg', sa.Float(), nullable=True))
    op.add_column('matches', sa.Column('LR_avg', sa.Float(), nullable=True))
    op.add_column('matches', sa.Column('Delta', sa.Integer(), nullable=True))
    # ### end Alembic commands ###


def downgrade():
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_column('matches', 'Delta')
    op.drop_column('matches', 'LR_avg')
    op.drop_column('matches', 'LFGP3_avg')
    op.drop_column('matches', 'LFGP_avg')
    op.drop_column('matches', 'WR_avg')
    op.drop_column('matches', 'WFGP3_avg')
    op.drop_column('matches', 'WFGP_avg')
    # ### end Alembic commands ###