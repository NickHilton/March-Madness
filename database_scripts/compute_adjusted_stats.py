"""
Pass 2: Compute opponent-adjusted FGP and FGP3 rolling averages.

Must run AFTER update_all_match_stats.py (Pass 1) which populates
the raw averages and "against" averages.

For each game, computes:
  adjusted_FGP = my_FGP_this_game - opponent_FGP_against_avg
  (how well I shot relative to what this opponent's defense typically allows)

Stores rolling averages of adjusted stats in WFGP_adj_avg / LFGP_adj_avg columns.
"""
from collections import defaultdict

from sqlalchemy.orm import sessionmaker

from models import engine, Match


def flush_batch(sess, batch):
    """Write adjusted averages to DB."""
    updates = defaultdict(dict)
    for match_id, prefix, adj_avg, adj3_avg in batch:
        updates[match_id][f"{prefix}FGP_adj_avg"] = adj_avg
        updates[match_id][f"{prefix}FGP3_adj_avg"] = adj3_avg

    for mid, vals in updates.items():
        sess.query(Match).filter(Match.id == mid).update(vals)
    sess.commit()


Session = sessionmaker(bind=engine)
session = Session()

# Get all matches with "against" data, ordered chronologically
matches = (
    session.query(
        Match.id,
        Match.Season,
        Match.WTeamID,
        Match.LTeamID,
        Match.WFGP,
        Match.LFGP,
        Match.WFGP3,
        Match.LFGP3,
        Match.WFGP_against_avg,
        Match.LFGP_against_avg,
        Match.WFGP3_against_avg,
        Match.LFGP3_against_avg,
    )
    .filter(Match.WFGP_against_avg.isnot(None))
    .filter(Match.LFGP_against_avg.isnot(None))
    .order_by(Match.id)
    .all()
)

print(f"Computing adjusted stats for {len(matches)} matches...")

# Track per-team adjusted stat rolling averages
team_state = defaultdict(lambda: {"FGP_adj_sum": 0, "FGP3_adj_sum": 0,
                                   "count": 0, "season": 0})

batch = []
batch_size = 5000

for i, m in enumerate(matches):
    match_id = m.id
    season = m.Season

    for prefix, team_id, my_fgp, my_fgp3, opp_against, opp_against3 in [
        ("W", m.WTeamID, m.WFGP, m.WFGP3, m.LFGP_against_avg, m.LFGP3_against_avg),
        ("L", m.LTeamID, m.LFGP, m.LFGP3, m.WFGP_against_avg, m.WFGP3_against_avg),
    ]:
        state = team_state[team_id]

        # Reset on new season
        if state["season"] != season:
            state["FGP_adj_sum"] = 0
            state["FGP3_adj_sum"] = 0
            state["count"] = 0
            state["season"] = season

        # Write current rolling average BEFORE updating (same convention as Pass 1)
        count = state["count"]
        if count > 0:
            adj_avg = state["FGP_adj_sum"] / count
            adj3_avg = state["FGP3_adj_sum"] / count
        else:
            adj_avg = 0.0
            adj3_avg = 0.0

        batch.append((match_id, prefix, adj_avg, adj3_avg))

        # Update rolling sum with this game's adjusted stat
        if my_fgp is not None and opp_against is not None and opp_against > 0:
            state["FGP_adj_sum"] += my_fgp - opp_against
        if my_fgp3 is not None and opp_against3 is not None and opp_against3 > 0:
            state["FGP3_adj_sum"] += my_fgp3 - opp_against3
        state["count"] += 1

    # Flush batch
    if len(batch) >= batch_size:
        flush_batch(session, batch)
        batch = []
        if (i + 1) % 20000 == 0:
            print(f"  {i+1}/{len(matches)} matches processed")

# Final flush
if batch:
    flush_batch(session, batch)

session.close()
print("Done.")
