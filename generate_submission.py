import argparse
import datetime
import json
import os
import sys
from collections import defaultdict
from typing import Dict, Set

import pandas as pd
from sqlalchemy import func, and_
from sqlalchemy.orm import sessionmaker

from models import Seed, engine, Team, Match, MatchPredictions, EvaluationRecord, SEASON
from elo_run.param_tuning import run_system, set_up_elo_model, link_function_list


def load_params_from_eval(eval_id):
    """
    Load model params from an evaluation record in the DB.
    Matches by exact ID or by prefix (takes the best-performing match).
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    # Try exact match first
    record = session.query(EvaluationRecord).filter(EvaluationRecord.id == eval_id).first()

    if not record:
        # Try prefix match - pick the one with lowest tournament loss
        records = (
            session.query(EvaluationRecord)
            .filter(EvaluationRecord.id.like(f"{eval_id}%"))
            .filter(EvaluationRecord.tournament_loss.isnot(None))
            .order_by(EvaluationRecord.tournament_loss.asc())
            .all()
        )
        if not records:
            session.close()
            raise ValueError(f"No evaluation record found matching '{eval_id}'")
        record = records[0]
        print(f"Matched {len(records)} records, using best (loss={record.tournament_loss:.4f}): {record.id}")

    params = {
        "k": int(record.k),
        "seed": float(record.seed),
        "link": record.link,
        "FGP": float(record.FGP),
        "R": float(record.R),
        "FGP3": float(record.FGP3),
        "rating": float(record.rating),
    }

    session.close()
    return params


def get_most_recent_stats(season):
    """
    For a given season, get the most recent stats available for each team.
    """
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get last match where each team won, and last match where each team lost
    w_q = (
        session.query(Match.WTeamID.label("TeamID"), func.max(Match.mdid).label("mdid"))
        .filter(Match.Season == season)
        .group_by(Match.WTeamID)
        .subquery()
    )
    l_q = (
        session.query(Match.LTeamID.label("TeamID"), func.max(Match.mdid).label("mdid"))
        .filter(Match.Season == season)
        .group_by(Match.LTeamID)
        .subquery()
    )

    winners = list(
        session.query(
            Match.WTeamID,
            Match.WFGP3_adj_avg,
            Match.WFGP_adj_avg,
            Match.WR_avg,
            MatchPredictions.WTeamRatingAfter,
            Match.mdid,
            Match.WTO_margin_avg,
            Match.WOR_avg,
            Match.WDR_avg,
        )
        .join(w_q, and_(Match.WTeamID == w_q.c.TeamID, Match.mdid == w_q.c.mdid))
        .join(MatchPredictions)
        .all()
    )

    losers = list(
        session.query(
            Match.LTeamID,
            Match.LFGP3_adj_avg,
            Match.LFGP_adj_avg,
            Match.LR_avg,
            MatchPredictions.LTeamRatingAfter,
            Match.mdid,
            Match.LTO_margin_avg,
            Match.LOR_avg,
            Match.LDR_avg,
        )
        .join(l_q, and_(Match.LTeamID == l_q.c.TeamID, Match.mdid == l_q.c.mdid))
        .join(MatchPredictions)
        .all()
    )

    session.close()

    all_stats = winners + losers
    df = pd.DataFrame(all_stats, columns=["TeamID", "FGP3", "FGP", "R", "rating", "mdid", "TO_margin", "off_reb_rate", "def_reb_rate"])
    # Keep the most recent entry per team (last win or last loss, whichever is later)
    df = df.sort_values("mdid").drop_duplicates(subset="TeamID", keep="last").drop(columns="mdid")
    df.set_index("TeamID", inplace=True, drop=True)
    return df


def build_round_opponent_map(df_slots, remaining_first_four_seeds):
    """
    Build mapping of round -> team seed -> set of possible opponent seeds.
    """
    slot_to_teams: Dict[str, Set[str]] = dict()
    round_to_team_to_opponents: Dict[int, Dict[str, Set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )

    for _, row in df_slots.sort_values(by=["round"]).iterrows():
        slot = row["Slot"]
        if "R" in slot or slot in remaining_first_four_seeds:
            rd = row["round"]

            strong = row["StrongSeed"]
            weak_opponents = slot_to_teams.get(strong, {strong})

            weak = row["WeakSeed"]
            strong_opponents = slot_to_teams.get(weak, {weak})

            for wk in weak_opponents:
                for st in strong_opponents:
                    round_to_team_to_opponents[rd][st].add(wk)
                    round_to_team_to_opponents[rd][wk].add(st)

            teams = strong_opponents.union(weak_opponents)
            slot_to_teams[slot] = teams

    return round_to_team_to_opponents


def generate_predictions(
    elo,
    season,
    data_path,
    dancers_dicts,
    seed_to_team_id,
    losers,
    remaining_first_four_seeds,
    remaining_first_four_teams,
    gamble_team=None,
    gamble_round=0,
):
    """
    Generate pairwise predictions for all possible tournament matchups.
    Returns list of (matchup_id, prediction) and (team1_name, team2_name, prediction).
    """
    # Read bracket structure
    df_slots = pd.read_csv(f"{data_path}NCAATourneySlots.csv").query(
        f"Season == {season}"
    ).drop(columns=["Season"]).reset_index(drop=True)
    df_slots["round"] = df_slots["Slot"].apply(
        lambda x: int(x[1]) if x[0] == "R" else 0
    )

    round_to_team_to_opponents = build_round_opponent_map(
        df_slots, remaining_first_four_seeds
    )

    predictions = []
    predictions_named = []

    # Set up team ratings per round
    round_to_team_id_to_rating: Dict[int, Dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for team_id, dd in dancers_dicts.items():
        if team_id in remaining_first_four_teams:
            r = 0
        else:
            r = 1
        round_to_team_id_to_rating[r][team_id] = dd["rating"]

    # Set up record of probability of a team reaching round N
    round_to_team_id_to_prob: Dict[int, Dict[int, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for team_id, dd in dancers_dicts.items():
        if team_id in remaining_first_four_teams:
            p = 1
            r = 0
        elif team_id in losers:
            p = 0
            r = 1
        else:
            p = 1
            r = 1
        round_to_team_id_to_prob[r][team_id] = p

    # All 6 championship rounds (+ first four if applicable)
    for rd in range(0, 7):
        rd_matches = round_to_team_to_opponents[rd]

        for team, opponents in rd_matches.items():
            for opponent in opponents:
                # Only do once per matchup
                if team < opponent:
                    team_id = seed_to_team_id[team]
                    opponent_id = seed_to_team_id[opponent]

                    # Set team 1 and team 2 by using min team id as team 1
                    if team_id < opponent_id:
                        team_1 = team_id
                        team_2 = opponent_id
                    else:
                        team_2 = team_id
                        team_1 = opponent_id

                    # Get current ratings
                    team_1_rating = round_to_team_id_to_rating[rd][team_1]
                    team_2_rating = round_to_team_id_to_rating[rd][team_2]

                    # Get stats and latest rating
                    team_1_stats = {
                        **dancers_dicts[team_1],
                        **{"rating": team_1_rating},
                    }
                    team_2_stats = {
                        **dancers_dicts[team_2],
                        **{"rating": team_2_rating},
                    }

                    # Predict matchup
                    prediction = elo.predict(team_1_stats, team_2_stats)

                    # Apply gamble override
                    if gamble_team and rd <= gamble_round:
                        if team_1 == gamble_team:
                            prediction = 0.98
                        if team_2 == gamble_team:
                            prediction = 0.02

                    # Save prediction
                    matchup_id = f"{season}_{team_1}_{team_2}"

                    # Update ratings probabilistically
                    if rd < 6:
                        point_diff = 8
                        result_likelihood = max(
                            elo.response(point_diff, "N"), prediction + 0.02
                        )
                        team_1_new = elo.update(
                            prediction, result_likelihood, team_1_rating, elo.K
                        )

                        prob_playing_opponent = round_to_team_id_to_prob[rd][team_2]

                        round_to_team_id_to_rating[rd + 1][team_1] += (
                            team_1_new * prob_playing_opponent
                        )
                        round_to_team_id_to_prob[rd + 1][team_1] += (
                            prediction * prob_playing_opponent
                        ) * round_to_team_id_to_prob[rd][team_1]

                        # Repeat for other team
                        result_likelihood = max(
                            elo.response(point_diff, "N"), 1 - prediction + 0.02
                        )
                        team_2_new = elo.update(
                            1 - prediction, result_likelihood, team_2_rating, elo.K
                        )

                        prob_playing_opponent = round_to_team_id_to_prob[rd][team_1]

                        round_to_team_id_to_rating[rd + 1][team_2] += (
                            team_2_new * prob_playing_opponent
                        )
                        round_to_team_id_to_prob[rd + 1][team_2] += (
                            (1 - prediction) * prob_playing_opponent
                        ) * round_to_team_id_to_prob[rd][team_2]

                    predictions.append((matchup_id, prediction))
                    predictions_named.append(
                        (
                            team_1_stats["TeamName"],
                            team_2_stats["TeamName"],
                            prediction,
                        )
                    )

    # Build lookup of matchup predictions for simulation
    # key: (lower_id, higher_id) -> P(lower_id wins)
    matchup_to_pred = {}
    for matchup_id, pred in predictions:
        parts = matchup_id.split("_")
        t1, t2 = int(parts[1]), int(parts[2])
        matchup_to_pred[(t1, t2)] = pred

    # Invert seed_to_team_id for display
    team_id_to_seed = {}
    for s, tid in seed_to_team_id.items():
        team_id_to_seed[tid] = s

    _print_tournament_summary(
        round_to_team_to_opponents,
        round_to_team_id_to_prob,
        seed_to_team_id,
        team_id_to_seed,
        dancers_dicts,
        matchup_to_pred,
        remaining_first_four_teams,
    )

    return predictions, predictions_named


def _print_tournament_summary(
    round_to_team_to_opponents,
    round_to_team_id_to_prob,
    seed_to_team_id,
    team_id_to_seed,
    dancers_dicts,
    matchup_to_pred,
    remaining_first_four_teams,
):
    """Print first-round matchups and Monte Carlo tournament simulation results."""
    import random

    round_names = {
        0: "First Four",
        1: "R64",
        2: "R32",
        3: "Sweet 16",
        4: "Elite 8",
        5: "Final 4",
        6: "Championship",
    }

    # --- First round matchups ---
    print(f"\n  {'='*60}")
    print(f"  FIRST ROUND MATCHUP PREDICTIONS")
    print(f"  {'='*60}")

    rd1_matches = round_to_team_to_opponents.get(1, {})
    printed = set()
    matchups = []
    for seed_str, opponents in rd1_matches.items():
        for opp in opponents:
            pair = tuple(sorted([seed_str, opp]))
            if pair in printed:
                continue
            printed.add(pair)

            tid1 = seed_to_team_id.get(pair[0])
            tid2 = seed_to_team_id.get(pair[1])
            if tid1 is None or tid2 is None:
                continue

            low, high = min(tid1, tid2), max(tid1, tid2)
            pred = matchup_to_pred.get((low, high))
            if pred is None:
                continue

            # pred is P(lower_id wins)
            if tid1 == low:
                p1 = pred
            else:
                p1 = 1 - pred

            name1 = dancers_dicts[tid1]["TeamName"]
            name2 = dancers_dicts[tid2]["TeamName"]
            matchups.append((pair[0], name1, p1, pair[1], name2))

    # Sort by region then seed number
    matchups.sort(key=lambda x: (x[0][0], int(x[0][1:])))
    for s1, name1, p1, s2, name2 in matchups:
        winner = name1 if p1 > 0.5 else name2
        conf = max(p1, 1 - p1)
        print(f"  ({s1:>3}) {name1:<20} {p1:>5.1%}  vs  {1-p1:>5.1%} {name2:<20} ({s2:>3})  -> {winner} ({conf:.0%})")

    # --- Monte Carlo simulation using slot-based bracket ---
    print(f"\n  {'='*60}")
    print(f"  MONTE CARLO SIMULATION (100 tournaments)")
    print(f"  {'='*60}")

    # All tournament team IDs
    all_team_ids = set()
    for rd_probs in round_to_team_id_to_prob.values():
        all_team_ids.update(rd_probs.keys())

    # Track how far each team gets across simulations
    team_max_round = defaultdict(list)

    # Get slot structure sorted by round
    from functools import lru_cache
    slots_by_round = defaultdict(list)
    for _, row in round_to_team_to_opponents.items():
        pass  # we'll use the raw slots instead

    # We need the actual slots DataFrame - reconstruct from the data path
    # The caller already read this; pass it through. For now, re-read it.
    # (This is in the same function scope where data_path isn't available,
    # so we'll extract it from the predictions matchup IDs)
    # Actually, let's just use the round_to_team_to_opponents differently:
    # In each round, iterate the possible matchups. For round > 1, a matchup
    # only happens if both teams are still alive. Since round_to_team_to_opponents
    # lists ALL possible opponents, we filter to the one that's alive.

    for sim in range(100):
        alive = set()
        team_round = {}

        for tid in all_team_ids:
            if tid in remaining_first_four_teams:
                team_round[tid] = 0
                alive.add(tid)
            elif round_to_team_id_to_prob[1].get(tid, 0) > 0:
                team_round[tid] = 1
                alive.add(tid)

        for rd in range(0, 7):
            rd_matches = round_to_team_to_opponents.get(rd, {})
            matched_this_round = set()

            for seed_str in sorted(rd_matches.keys()):
                tid1 = seed_to_team_id.get(seed_str)
                if tid1 is None or tid1 not in alive or tid1 in matched_this_round:
                    continue

                # Find the opponent that's alive
                opponent_tid = None
                for opp_seed in rd_matches[seed_str]:
                    opp_tid = seed_to_team_id.get(opp_seed)
                    if opp_tid and opp_tid in alive and opp_tid not in matched_this_round:
                        opponent_tid = opp_tid
                        break

                if opponent_tid is None:
                    continue

                matched_this_round.add(tid1)
                matched_this_round.add(opponent_tid)

                low, high = min(tid1, opponent_tid), max(tid1, opponent_tid)
                pred = matchup_to_pred.get((low, high))
                if pred is None:
                    pred = 0.5

                if random.random() < pred:
                    winner, loser = low, high
                else:
                    winner, loser = high, low

                team_round[winner] = rd + 1
                alive.discard(loser)

        for tid, max_rd in team_round.items():
            team_max_round[tid].append(max_rd)

    # Build summary table
    rows = []
    for tid in all_team_ids:
        rounds = team_max_round.get(tid, [0])
        seed_str = team_id_to_seed.get(tid, "?")
        name = dancers_dicts.get(tid, {}).get("TeamName", "?")
        avg_rd = sum(rounds) / len(rounds)

        rd_counts = defaultdict(int)
        for r in rounds:
            rd_counts[r] += 1

        rows.append((seed_str, name, avg_rd, rd_counts))

    rows.sort(key=lambda x: -x[2])

    print(f"\n  {'Seed':<5} {'Team':<22} {'Avg Rd':>6}  {'R32':>4} {'S16':>4} {'E8':>5} {'F4':>5} {'Chmp':>5} {'Win':>5}")
    print(f"  {'-'*70}")

    for seed_str, name, avg_rd, rd_counts in rows[:32]:
        # Times reached each round (out of 100 sims)
        r32  = sum(rd_counts.get(r, 0) for r in range(2, 8))
        s16  = sum(rd_counts.get(r, 0) for r in range(3, 8))
        e8   = sum(rd_counts.get(r, 0) for r in range(4, 8))
        f4   = sum(rd_counts.get(r, 0) for r in range(5, 8))
        chmp = sum(rd_counts.get(r, 0) for r in range(6, 8))
        win  = rd_counts.get(7, 0)
        print(f"  {seed_str:<5} {name:<22} {avg_rd:>6.2f}  {r32:>4} {s16:>4} {e8:>4}  {f4:>4}  {chmp:>4}  {win:>4}")

    print(f"\n  (Top 32 teams shown, out of 100 simulated tournaments)")


def main():
    parser = argparse.ArgumentParser(description="Generate March Madness submission")
    parser.add_argument(
        "--gender",
        type=str,
        required=True,
        choices=["M", "W"],
        help="M for men's, W for women's",
    )
    parser.add_argument(
        "--gamble-team",
        type=int,
        default=None,
        help="Team ID to override predictions for",
    )
    parser.add_argument(
        "--gamble-round",
        type=int,
        default=3,
        help="Override predictions through this round (default: 3)",
    )
    parser.add_argument(
        "--eval-id",
        type=str,
        default=None,
        help="Evaluation record ID (or prefix) to pull params from the DB",
    )
    parser.add_argument(
        "--candidate",
        type=str,
        default=None,
        help="Path to candidate_params JSON file (e.g. candidate_params/260318T172856_optuna_best_mens.json)",
    )
    parser.add_argument(
        "--params",
        type=str,
        default=None,
        help="Tab-separated model params: k seed link FGP R FGP3 rating",
    )
    parser.add_argument(
        "--description",
        type=str,
        default="high_rating_low_seed",
        help="Description for output filename",
    )
    parser.add_argument(
        "--first-four",
        type=str,
        default="first_four_results.json",
        help="Path to first four results JSON (default: first_four_results.json)",
    )
    args = parser.parse_args()

    data_path = os.environ["DATA_PATH"] + "/" + os.environ["DATA_PREFIX"]
    gender = "WOMENS" if os.environ["DATA_PREFIX"] == "W" else "MENS"
    gender_key = "womens" if gender == "WOMENS" else "mens"
    season = SEASON

    # Load first four results
    args.first_four_results = {}
    if args.first_four and os.path.exists(args.first_four):
        with open(args.first_four) as f:
            ff_data = json.load(f)
        if gender_key in ff_data:
            for slot, info in ff_data[gender_key].items():
                if info.get("winner"):
                    winner_key = info["winner"]  # "a" or "b"
                    args.first_four_results[slot] = info[winner_key]["id"]

    # All model params with defaults
    PARAM_DEFAULTS = {
        "k": 25, "seed": -35.0, "link": "N", "fgp": 1200.0, "fgp3": 0.0,
        "reb": 20.0, "rating": 5.0, "d": 600.0, "alpha": 0.0,
        "to_margin": 0.0, "off_reb_rate": 0.0, "def_reb_rate": 0.0,
        "massey_rank": 0.0, "decay": 1.0,
    }

    # Parse model params from candidate file, eval ID, --params, or default_params.json
    source = "defaults"
    dp = dict(PARAM_DEFAULTS)

    if args.candidate:
        source = args.candidate
        with open(args.candidate) as f:
            candidate_data = json.load(f)
        # Support both {params: {...}} and flat {...} formats
        file_params = candidate_data.get("params", candidate_data)
        dp.update(file_params)
        # Handle old key names
        if "off_reb" in dp and "off_reb_rate" not in file_params:
            dp["off_reb_rate"] = dp.pop("off_reb", 0.0)
        if "def_reb" in dp and "def_reb_rate" not in file_params:
            dp["def_reb_rate"] = dp.pop("def_reb", 0.0)
    elif args.eval_id:
        source = f"eval:{args.eval_id}"
        print(f"  WARNING: --eval-id only loads 7 basic params from DB (k, seed, link, FGP, R, FGP3, rating).")
        print(f"  Use --candidate with a JSON file to load all params including d, alpha, decay, massey, etc.")
        params = load_params_from_eval(args.eval_id)
        dp.update({"k": params["k"], "seed": params["seed"], "link": params["link"],
                    "fgp": params["FGP"], "reb": params["R"], "fgp3": params["FGP3"],
                    "rating": params["rating"]})
    elif args.params:
        source = "--params"
        vals = args.params.split("\t")
        dp.update({"k": int(vals[0]), "seed": float(vals[1]), "link": vals[2],
                    "fgp": float(vals[3]), "reb": float(vals[4]), "fgp3": float(vals[5]),
                    "rating": float(vals[6])})
    else:
        source = "default_params.json"
        with open("default_params.json") as f:
            all_default = json.load(f)
        dp.update(all_default.get(gender_key, {}))

    # Print param report
    print(f"\n  Gender: {gender}")
    print(f"  Season: {season}")
    print(f"  Source: {source}")
    if args.candidate and "backtest" in candidate_data:
        bt = candidate_data["backtest"]
        print(f"  Backtest: Brier={bt.get('mean_brier_loss')}  Correct={bt.get('mean_correct_pct')}")
        if "meta" in candidate_data:
            print(f"  Git: {candidate_data['meta'].get('git_hash', '?')[:7]}  Dirty: {candidate_data['meta'].get('git_dirty')}")
    print()
    print(f"  {'Param':<15} {'Value':>15} {'Default':>15} {'Source':>10}")
    print(f"  {'-'*58}")
    for param, default in PARAM_DEFAULTS.items():
        val = dp[param]
        is_default = (val == default)
        src = "default" if is_default else "file"
        val_str = f"{val}" if isinstance(val, str) else f"{val:.6g}"
        def_str = f"{default}" if isinstance(default, str) else f"{default:.6g}"
        print(f"  {param:<15} {val_str:>15} {def_str:>15} {src:>10}")
    print()

    if args.gamble_team:
        print(f"  Gamble: team {args.gamble_team} through round {args.gamble_round}")

    link_function = link_function_list[dp["link"]]

    elo = set_up_elo_model(
        k=dp["k"], seed=dp["seed"], link_function=link_function,
        fgp=dp["fgp"], fgp3=dp["fgp3"], r=dp["reb"], rating=dp["rating"],
        d=dp["d"], alpha=dp["alpha"],
        to_margin=dp["to_margin"], off_reb_rate=dp["off_reb_rate"],
        def_reb_rate=dp["def_reb_rate"], massey_rank=dp["massey_rank"],
        decay=dp["decay"],
    )

    # Run model to get match predictions
    print(f"\nRunning ELO model through season {season}...")
    match_predictions = run_system(elo, season)
    conn = engine.raw_connection()
    match_predictions.to_sql(
        con=conn, index=False, name=MatchPredictions.__tablename__, if_exists="replace"
    )
    conn.close()
    print(f"Saved {len(match_predictions)} match predictions to DB")

    # Set up session
    Session = sessionmaker(bind=engine)
    session = Session()

    # Get team info
    team_id_to_name = {x: y for x, y in session.query(Team.TeamID, Team.TeamName).all()}

    # Get tournament teams
    dancers = pd.DataFrame(
        session.query(Seed.TeamID, Seed.Seed.label("seed")).filter(Seed.Season == season).all()
    )
    dancers_df = dancers.sort_values(by="TeamID").set_index("TeamID", drop=True)
    dancers_df = dancers_df.join(
        pd.DataFrame.from_dict(team_id_to_name, orient="index", columns=["TeamName"]),
        how="left",
    )

    # Get most recent stats
    dancer_to_stats = get_most_recent_stats(season)
    full_df = dancers_df.merge(dancer_to_stats, left_index=True, right_index=True)

    # Add Massey composite ranks
    from elo_run.massey import get_massey_ranks
    massey = get_massey_ranks(season)
    full_df["massey_rank"] = full_df.index.map(lambda tid: massey.get(tid, None))

    dancers_dicts = full_df.to_dict(orient="index")

    # Fill None/NaN defaults for features the model needs
    for tid, dd in dancers_dicts.items():
        for key in ["TO_margin", "off_reb_rate", "def_reb_rate", "FGP", "FGP3", "R"]:
            if dd.get(key) is None or (isinstance(dd.get(key), float) and pd.isna(dd[key])):
                dd[key] = 0.0

    # Get seeds and resolve first four
    df_seeds = pd.read_csv(f"{data_path}NCAATourneySeeds.csv").query(
        f"Season == {season}"
    ).drop(columns=["Season"]).reset_index(drop=True)
    seed_to_team_id = df_seeds.set_index("Seed").to_dict()["TeamID"]

    first_four_slots = [x for x in seed_to_team_id if "a" in x]

    print(f"\nFirst four matchups:")
    for i in first_four_slots:
        other = i.replace("a", "b")
        print(
            f"  {i}: {seed_to_team_id[i]}:{team_id_to_name[seed_to_team_id[i]]} vs "
            f"{seed_to_team_id[other]}:{team_id_to_name[seed_to_team_id[other]]}"
        )

    # Resolve first four
    # If --first-four-results provided, use those. Otherwise predict probabilistically.
    losers = []
    remaining_first_four_seeds = []
    remaining_first_four_teams = []

    for i in first_four_slots:
        other = i.replace("a", "b")
        s = i[:3]  # e.g. "X16"
        team_a = seed_to_team_id[i]
        team_b = seed_to_team_id[other]

        if args.first_four_results and s in args.first_four_results:
            # Known result: use the specified winner
            winner_id = args.first_four_results[s]
            loser_id = team_b if winner_id == team_a else team_a
            seed_to_team_id[s] = winner_id
            losers.append(loser_id)
            del seed_to_team_id[i]
            del seed_to_team_id[other]
            print(f"  {s}: {winner_id}:{team_id_to_name[winner_id]} WINS (known result)")
        else:
            # Unresolved: keep both teams with a/b keys for round 0
            remaining_first_four_seeds.append(s)
            remaining_first_four_teams.append(team_a)
            remaining_first_four_teams.append(team_b)
            # Keys already exist as i (e.g. X16a) and other (e.g. X16b) — keep them
            print(f"  {s}: predicting probabilistically")

    # Generate predictions
    print(f"\nGenerating tournament predictions...")
    predictions, predictions_named = generate_predictions(
        elo=elo,
        season=season,
        data_path=data_path,
        dancers_dicts=dancers_dicts,
        seed_to_team_id=seed_to_team_id,
        losers=losers,
        remaining_first_four_seeds=remaining_first_four_seeds,
        remaining_first_four_teams=remaining_first_four_teams,
        gamble_team=args.gamble_team,
        gamble_round=args.gamble_round,
    )

    # Build submission with non-tournament matchups filled from sample
    sample_df = pd.read_csv(f"{data_path[:-2]}/SampleSubmissionStage2.csv")

    if gender == "WOMENS":
        sample_submission_rows = sample_df.query(f"ID > '{season}_3'")
    else:
        sample_submission_rows = sample_df.query(f"ID < '{season}_3'")

    prediction_df = pd.DataFrame(predictions, columns=["ID", "Pred"])
    prediction_ids = set(prediction_df["ID"])

    sample_submission_rows = sample_submission_rows[
        ~sample_submission_rows["ID"].isin(prediction_ids)
    ]

    print(f"\nTournament predictions: {len(prediction_df)}")
    print(f"Remaining from sample: {len(sample_submission_rows)}")

    # Save
    final_out_df = pd.concat([prediction_df, sample_submission_rows], ignore_index=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%dT%H:%M")
    description = args.description

    ids_path = f"submissions/ids/{gender[0]}_{description}_{timestamp}.csv"
    names_path = f"submissions/names/{gender[0]}_{description}_{timestamp}.csv"

    final_out_df.to_csv(ids_path, index=False)

    prediction_named_df = pd.DataFrame(
        predictions_named, columns=["Team1", "Team2", "Pred"]
    )
    prediction_named_df.to_csv(names_path, index=False)

    print(f"\nSaved {len(final_out_df)} predictions to {ids_path}")
    print(f"Saved {len(prediction_named_df)} named predictions to {names_path}")

    session.close()


if __name__ == "__main__":
    main()
