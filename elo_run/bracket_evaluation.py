import os
import random

import numpy as np
import pandas as pd
from sqlalchemy import func, and_
from sqlalchemy.orm import sessionmaker

from models import Seed, engine, Team, Match, MatchPredictions

DATA_PATH = os.environ['DATA_PATH'] + "/" + os.environ["DATA_PREFIX"]
gender = "WOMENS" if os.environ["DATA_PREFIX"] == "W" else "MENS"


NUM_BRACKETS = 100_000
# Get the most recent stats before the tournament
def get_most_recent_stats(Season):
    """
    For a given Season, get the most recent stats available
    :param Season:
    :return:
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    q = session.query(Team.TeamID, func.max(Match.mdid).label('mdid')).join(
        Match.teams).filter(Match.Season == Season).group_by(Team.TeamID).subquery()

    winners = list(session.query(Match.WTeamID, Match.WFGP3_avg, Match.WFGP_avg, Match.WR_avg,
                                 MatchPredictions.WTeamRatingAfter).join(
        q, and_(Match.WTeamID == q.c.TeamID, Match.mdid == q.c.mdid)
    ).join(MatchPredictions).all())

    losers = list(session.query(Match.LTeamID, Match.LFGP3_avg, Match.LFGP_avg, Match.LR_avg,
                                MatchPredictions.LTeamRatingAfter).join(
        q, and_(Match.LTeamID == q.c.TeamID, Match.mdid == q.c.mdid)
    ).join(MatchPredictions).all())

    all_stats = winners + losers

    df = pd.DataFrame(all_stats, columns=['TeamID', 'FGP3', 'FGP', 'R', 'rating'])
    df.set_index('TeamID', inplace=True, drop=True)

    return df


def get_match(team_a, team_b):
    if team_a < team_b:
        return team_a * 100_000 + team_b
    return team_b * 100_000 + team_a


def generate_bracket(bracket_ix, seed_to_team_id, dancers_dicts, elo, team_id_to_seed, season):
    output_rows = []

    df_tourney_slots = pd.read_csv(f"{DATA_PATH}NCAATourneySlots.csv").query(
        f"Season == {season}").drop(columns=["Season"]).reset_index(drop=True)

    df_tourney_slots['round'] = df_tourney_slots["Slot"].apply(lambda x: int(x[1]) if x[0] == "R" else 0)

    df_tourney_slots = df_tourney_slots.query("round > 0")
    simulation_dicts = {k: {k1: v1 for k1, v1 in v.items()} for k, v in dancers_dicts.items()}

    for _, row in df_tourney_slots.iterrows():
        slot = row['Slot']

        team_1 = seed_to_team_id[row['StrongSeed']]

        team_2 = seed_to_team_id[row['WeakSeed']]

        team_1_stats = {**simulation_dicts[team_1]}
        team_2_stats = {**simulation_dicts[team_2]}

        prediction = elo.predict(team_1_stats, team_2_stats)

        result = random.random()

        if result < prediction:
            seed_to_team_id[slot] = team_1
            point_diff = 8
            result_likelihood = max(elo.response(point_diff, 'N'), prediction + 0.02)
            team_1_new = elo.update(prediction, result_likelihood, team_1_stats['rating'], elo.K)
            simulation_dicts[team_1]['rating'] = team_1_new
            winner = team_1

        else:
            seed_to_team_id[slot] = team_2
            point_diff = 8
            result_likelihood = max(elo.response(point_diff, 'N'), 1 - prediction + 0.02)
            team_2_new = elo.update(1 - prediction, result_likelihood, team_2_stats['rating'], elo.K)
            simulation_dicts[team_2]['rating'] = team_2_new
            winner = team_2

        # ['Tournament', 'Bracket', 'Slot', 'Team']
        output_row = [os.environ["DATA_PREFIX"], bracket_ix, slot, team_id_to_seed[winner]]

        output_rows.append(output_row)
    return output_rows


def get_actual_bracket(seed_to_team_id, team_id_to_seed, tourney_match_to_winner, season):
    df_tourney_slots = pd.read_csv(f"{DATA_PATH}NCAATourneySlots.csv").query(
        f"Season == {season}").drop(columns=["Season"]).reset_index(drop=True)

    output_rows = []

    df_tourney_slots['round'] = df_tourney_slots["Slot"].apply(lambda x: int(x[1]) if x[0] == "R" else 0)

    df_tourney_slots = df_tourney_slots.query("round > 0")

    for _, row in df_tourney_slots.iterrows():
        slot = row['Slot']

        team_1 = seed_to_team_id[row['StrongSeed']]
        team_2 = seed_to_team_id[row['WeakSeed']]

        match = get_match(team_1, team_2)
        if match == 133201433 and season == 2021:
            match_winner = 1332
        else:
            match_winner = tourney_match_to_winner[match]

        seed_to_team_id[slot] = match_winner

        output_row = [os.environ["DATA_PREFIX"], 1, slot, team_id_to_seed[match_winner]]

        output_rows.append(output_row)
    return output_rows


def make_implied_probability_table(df_sub):
    # pandas gibberish that gets you the proportion of times Team wins
    # a particular slot in the tournament
    tmp = df_sub[['Tournament', 'Slot', 'Team']] \
        .groupby(['Tournament', 'Slot']) \
        .agg('value_counts', normalize=True)

    # more pandas gibberish to get it in the format we want.
    # eventually want the columns to be named after rounds and have
    # rows correspond to tournament and team
    tmp = tmp.to_frame()
    tmp.reset_index(inplace=True)
    tmp['Round'] = tmp['Slot'].str[0:2]
    tmp.drop(columns='Slot', inplace=True)
    tmp.set_index(['Tournament', 'Team', 'Round'], inplace=True)
    tmp = tmp.stack().unstack(level=2).fillna(0.0)
    tmp.reset_index(inplace=True)

    # cleanup
    tmp.columns.name = None
    tmp.drop(columns='level_2', inplace=True)

    # now need to add in missing teams, if any
    # some teams may never appear in the bracket.  This means they
    # should have implied probabilities of 0 for all rounds
    df_missing = []
    seeds = [f'{region}{num:02d}' for region in list('WXYZ') \
             for num in range(1, 17)]
    for t, sdf in tmp.groupby('Tournament'):
        missing_seeds = np.setdiff1d(seeds, sdf['Team'])
        df_missing.append(pd.DataFrame({'Tournament': t, 'Team': missing_seeds}))
    df_missing = pd.concat(df_missing)

    tmp = pd.concat([tmp, df_missing])
    tmp.fillna(0.0, inplace=True)
    tmp.sort_values(['Tournament', 'Team'], inplace=True)
    tmp.reset_index(inplace=True, drop=True)

    return tmp


def make_evaluation_df(df_sub, df_truth):
    # makes a dataframe which will be used for computing the score
    proc_sub = make_implied_probability_table(df_sub)
    proc_truth = make_implied_probability_table(df_truth)
    tmp = proc_sub.merge(proc_truth, on=['Tournament', 'Team'], how='inner', suffixes=('_sub', '_truth'))

    for col in tmp.columns[tmp.columns.str.endswith('_truth')]:
        r = col.split('_')[0]
        tmp[r + '_brier'] = (tmp[r + '_sub'] - tmp[r + '_truth']) ** 2

    return tmp


def calc_evaluation_score(df_sub, df_truth):
    evaluation = make_evaluation_df(df_sub, df_truth)
    brier_cols = evaluation.columns[(evaluation.columns.str.endswith('_brier'))]
    score = evaluation.groupby('Tournament')[brier_cols].mean().mean(axis=1).mean()
    return score


def evaluate_season(elo, season):
    if season == 2020:
        return {}
    Session = sessionmaker(bind=engine)
    session = Session()

    dancer_to_stats = get_most_recent_stats(season)

    # Get the team id to name
    team_id_to_name = {x: y for x, y in session.query(Team.TeamID, Team.TeamName).all()}

    # Get all teams in the tournament for predicting
    dancers = pd.DataFrame(session.query(Seed.TeamID, Seed.Seed.label('seed'), Seed.SeedSlot.label('seedSlot')
                                         ).filter(Seed.Season == season).all())

    dancers_df = dancers.sort_values(by='TeamID').set_index('TeamID', drop=True)
    dancers_df = dancers_df.join(pd.DataFrame.from_dict(team_id_to_name, orient='index',
                                                        columns=['TeamName']), how='left')

    full_df = dancers_df.merge(dancer_to_stats, left_index=True, right_index=True)

    dancers_dicts = full_df.to_dict(orient='index')

    seed_to_team_id = full_df.reset_index(drop=False).set_index("seedSlot").to_dict()['TeamID']

    first_four_slots = [x for x in seed_to_team_id if 'a' in x]

    tourney_match_to_winner = pd.DataFrame(session.query(Match.MatchID, Match.WTeamID, Match.LTeamID
                                                         ).filter(Match.Season == season).filter(
        Match.stage == 'T').all()).set_index(
        'MatchID').to_dict()['WTeamID']

    # First Four results -> TO BE UPDATED
    for i in first_four_slots:
        other = i.replace("a", "b")
        team_a = seed_to_team_id[i]
        team_b = seed_to_team_id[other]
        match = get_match(team_a, team_b)
        match_winner = tourney_match_to_winner.get(match, team_a)
        seed = i[:3]
        seed_to_team_id[seed] = match_winner
        del seed_to_team_id[i]
        del seed_to_team_id[other]

    team_id_to_seed = {v: k for k, v in seed_to_team_id.items()}

    df_actual = pd.DataFrame(get_actual_bracket(seed_to_team_id, team_id_to_seed, tourney_match_to_winner, season),
                             columns=['Tournament', 'Bracket', 'Slot', 'Team'])

    full_sim = []
    for bracket_ix in range(1, NUM_BRACKETS):
        sim_rows = generate_bracket(bracket_ix, seed_to_team_id, dancers_dicts, elo, team_id_to_seed, season)
        full_sim.extend(sim_rows)

    df_full_sim = pd.DataFrame(full_sim, columns=['Tournament', 'Bracket', 'Slot', 'Team'])

    num_brackets_to_score = {}
    for i in [1000, 10_000, 20_000, 40_000, 50_000, 70_000, 80_000, 100_000]:
        score = calc_evaluation_score(df_full_sim.query(f"Bracket <= {i}"), df_actual)
        num_brackets_to_score[i] = score

    return num_brackets_to_score
