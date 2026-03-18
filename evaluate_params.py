"""
Evaluate model params and print a performance report.

Usage:
    python evaluate_params.py                          # Uses default_params.json
    python evaluate_params.py --params-json '{"k": 150, "seed": -40, ...}'

Requires env vars: DATABASE_URL, DATA_PATH, DATA_PREFIX
"""
import argparse
import json
import os
import random
import string
import sys
import time

import pandas as pd

from elo_run.param_tuning import (
    set_up_elo_model,
    save_evaluation,
    link_function_list,
)
from elo_run.run_model import run_model_one_season
from elo_run.evaluation import evaluate_by_season
from models import engine, MatchPredictions, SEASON, SEASON_START


def run_and_report(params: dict, name: str = None, gender: str = "M") -> pd.DataFrame:
    """Run model season by season with progress logging, print report, save to DB."""
    link_function = link_function_list[params["link"]]
    elo = set_up_elo_model(
        k=params["k"],
        seed=params["seed"],
        link_function=link_function,
        fgp=params["fgp"],
        fgp3=params["fgp3"],
        r=params["reb"],
        rating=params["rating"],
        d=params.get("d", 600.0),
        alpha=params.get("alpha", 0.0),
        to_margin=params.get("to_margin", 0.0),
        off_reb=params.get("off_reb", 0.0),
        def_reb=params.get("def_reb", 0.0),
        massey_rank=params.get("massey_rank", 0.0),
    )

    total_seasons = SEASON - 1 - SEASON_START + 1
    print(f"\n  Running Elo model: {SEASON_START} -> {SEASON - 1} ({total_seasons} seasons)")
    print()

    # Run season by season (mirrors run_system but with logging)
    all_predictions = []
    rating_seeds = None
    run_start = time.time()

    for i, season in enumerate(range(SEASON_START, SEASON), 1):
        season_start = time.time()

        df = run_model_one_season(season, elo_model=elo, rating_seeds=rating_seeds)
        all_predictions.append(df)

        # Extract end-of-season ratings
        last_rating_df = pd.DataFrame()
        for wl in ["W", "L"]:
            wldf = (
                df.groupby(by=f"{wl}TeamID")
                .tail(1)
                .loc[:, ["match_id", f"{wl}TeamID", f"{wl}TeamRatingAfter"]]
                .rename(
                    columns={f"{wl}TeamID": "Team", f"{wl}TeamRatingAfter": "Rating"}
                )
            )
            last_rating_df = pd.concat([last_rating_df, wldf])

        new_ratings = (
            last_rating_df.sort_values(by="match_id")
            .groupby(by="Team")
            .tail(1)
            .set_index("Team")
            .Rating.to_dict()
        )

        if rating_seeds:
            rating_seeds.update(new_ratings)
        else:
            rating_seeds = new_ratings

        elapsed = time.time() - season_start
        n_matches = len(df)
        n_tourney = len(df.query("Stage == 'T'"))
        tourney_str = f"({n_tourney} tournament)" if n_tourney else "(no tournament)"
        print(f"  [{i:>2}/{total_seasons}] {season}: {n_matches:>5} matches {tourney_str:<20} [{elapsed:.1f}s]")

    match_predictions = pd.concat(all_predictions, ignore_index=True)
    total_time = time.time() - run_start
    total_matches = len(match_predictions)
    print(f"\n  Done: {total_matches:,} total matches in {total_time:.1f}s")

    # Evaluate per season
    print("\n  Evaluating tournament performance...\n")
    results = []
    for season in range(SEASON_START, SEASON):
        season_preds = match_predictions.loc[
            match_predictions["Season"] == season, ["PredProbWTeam", "Stage"]
        ].copy()

        tourney = season_preds.query("Stage == 'T'")
        if tourney.empty:
            results.append((season, None, None, 0))
            continue

        loss, correct = evaluate_by_season(season_preds)
        results.append((season, loss, correct, len(tourney)))

    # Print report
    print(f"  Params: k={params['k']}  seed={params['seed']}  link={params['link']}  "
          f"fgp={params['fgp']}  fgp3={params['fgp3']}  reb={params['reb']}  "
          f"rating={params['rating']}")
    print()
    print(f"  {'Season':<8} {'Brier Loss':>12} {'Correct':>10} {'Games':>7}")
    print(f"  {'-'*39}")

    losses = []
    corrects = []
    for season, loss, correct, games in results:
        if loss is None:
            print(f"  {season:<8} {'N/A':>12} {'N/A':>10} {games:>7}")
        else:
            losses.append(loss)
            corrects.append(correct)
            print(f"  {season:<8} {loss:>12.4f} {correct:>9.1%} {games:>7}")

    print(f"  {'-'*39}")
    avg_loss = sum(losses) / len(losses)
    avg_correct = sum(corrects) / len(corrects)
    print(f"  {'AVERAGE':<8} {avg_loss:>12.4f} {avg_correct:>9.1%}")
    print()

    # Save to DB
    print("  Saving evaluation to DB...")
    save_evaluation(
        rating=params["rating"],
        k=params["k"],
        seed=params["seed"],
        function_code=params["link"],
        fgp=params["fgp"],
        fgp3=params["fgp3"],
        r=params["reb"],
        match_predictions=match_predictions,
        name=name,
    )

    if name:
        gender_flag = gender
        print(f"\n  Eval ID: {name}")
        print(f"  Use with: python generate_submission.py --gender {gender_flag} --eval-id {name}")

    print("  Saving match predictions to DB...")
    conn = engine.raw_connection()
    match_predictions.to_sql(
        con=conn, index=False, name=MatchPredictions.__tablename__, if_exists="replace"
    )
    conn.close()
    print("  Done.\n")

    return match_predictions


def main():
    parser = argparse.ArgumentParser(description="Evaluate model params and print report")
    parser.add_argument(
        "--params-json",
        type=str,
        default=None,
        help='JSON string of params, e.g. \'{"k": 150, "seed": -40, ...}\'',
    )
    parser.add_argument(
        "--gender",
        type=str,
        default=None,
        choices=["mens", "womens"],
        help="Which gender params to use from default_params.json",
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Short name for this evaluation run (used as eval-id in generate_submission.py)",
    )
    args = parser.parse_args()

    if args.params_json:
        params = json.loads(args.params_json)
    else:
        with open("default_params.json") as f:
            all_params = json.load(f)

        gender = args.gender
        if not gender:
            prefix = os.environ.get("DATA_PREFIX", "M")
            gender = "womens" if prefix == "W" else "mens"

        params = all_params[gender]

    name = args.name
    if not name:
        name = "eval_" + "".join(random.choices(string.ascii_lowercase + string.digits, k=6))

    print(f"\n{'='*50}")
    print(f"  {gender.upper()} Evaluation")
    print(f"  DB: {os.environ.get('DATABASE_URL', 'not set')}")
    print(f"  ID: {name}")
    print(f"{'='*50}")

    gender_flag = "W" if os.environ.get("DATA_PREFIX", "M") == "W" else "M"
    run_and_report(params, name=name, gender=gender_flag)


if __name__ == "__main__":
    main()
