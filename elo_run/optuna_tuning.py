"""
Bayesian parameter optimization for the Elo model using Optuna.

Usage:
    DATABASE_URL="sqlite:///march_madness_male.db" DATA_PATH="data_male" DATA_PREFIX="M" \
        python -m elo_run.optuna_tuning --gender mens --n-trials 400 --n-jobs 4
"""
import argparse
import datetime
import json
import math
import os
import subprocess
import sys
import threading
import time

import optuna
import pandas as pd

from elo_run.evaluation import evaluate_by_season
from elo_run.param_tuning import (
    set_up_elo_model,
    link_function_list,
    run_system,
)
from models import SEASON, SEASON_START


def objective(trial: optuna.Trial) -> float:
    """Optuna objective: returns mean Brier loss across tournament seasons."""

    # Sample hyperparameters
    k = trial.suggest_int("k", 10, 400)
    seed = trial.suggest_float("seed", -80, 0)
    link = trial.suggest_categorical("link", ["N", "B", "L"])
    fgp = trial.suggest_float("fgp", 500, 8000, log=True)
    fgp3 = trial.suggest_float("fgp3", -100, 8000)
    reb = trial.suggest_float("reb", 5, 600, log=True)
    rating = trial.suggest_float("rating", 0.1, 10, log=True)
    d = trial.suggest_float("d", 200, 1200)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    to_margin = trial.suggest_float("to_margin", 0, 500)
    off_reb = trial.suggest_float("off_reb", 0, 500)
    def_reb = trial.suggest_float("def_reb", 0, 500)
    massey_rank = trial.suggest_float("massey_rank", -50, 0)

    link_function = link_function_list[link]

    elo = set_up_elo_model(
        k=k, seed=seed, link_function=link_function,
        fgp=fgp, fgp3=fgp3, r=reb, rating=rating,
        d=d, alpha=alpha,
        to_margin=to_margin, off_reb=off_reb, def_reb=def_reb,
        massey_rank=massey_rank,
    )

    # Run all seasons and collect tournament Brier losses
    match_predictions = run_system(elo, end_season=SEASON - 1)

    losses = []
    step = 0
    for season in range(SEASON_START, SEASON):
        season_preds = match_predictions.loc[
            match_predictions["Season"] == season, ["PredProbWTeam", "Stage"]
        ].copy()

        tourney = season_preds.query("Stage == 'T'")
        if tourney.empty:
            continue

        loss, _ = evaluate_by_season(season_preds)
        losses.append(loss)
        step += 1

        # Report intermediate value for pruning
        running_mean = sum(losses) / len(losses)
        trial.report(running_mean, step)

        if trial.should_prune():
            raise optuna.TrialPruned()

    if not losses:
        return float("inf")

    return sum(losses) / len(losses)


def _format_trial_row(t):
    """Format a single trial as a leaderboard row string."""
    p = t.params
    return (
        f"  {t.number:>4} {t.value:>10.6f} {p['k']:>5} {p['seed']:>7.1f} {p['link']:>4} "
        f"{p['fgp']:>8.0f} {p['fgp3']:>8.1f} {p['reb']:>7.1f} {p['rating']:>7.2f} "
        f"{p['d']:>7.0f} {p['alpha']:>6.3f} {p['to_margin']:>6.0f} {p['off_reb']:>6.0f} {p['def_reb']:>6.0f} {p['massey_rank']:>6.1f}"
    )


LEADERBOARD_HEADER = (
    f"  {'#':>4} {'Brier':>10} {'k':>5} {'seed':>7} {'link':>4} {'fgp':>8} "
    f"{'fgp3':>8} {'reb':>7} {'rating':>7} {'d':>7} {'alpha':>6} "
    f"{'TO_m':>6} {'OR':>6} {'DR':>6} {'mRank':>6}"
)
LEADERBOARD_SEP = f"  {'-'*105}"


class ProgressCallback:
    """Prints a leaderboard every `every` completed trials."""

    def __init__(self, every: int = 25):
        self.every = every
        self._lock = threading.Lock()
        self._completed = 0
        self._pruned = 0
        self._start = time.time()

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        with self._lock:
            if trial.state == optuna.trial.TrialState.COMPLETE:
                self._completed += 1
            elif trial.state == optuna.trial.TrialState.PRUNED:
                self._pruned += 1
            else:
                return

            total = self._completed + self._pruned
            if total % self.every != 0:
                return

            elapsed = time.time() - self._start
            rate = total / elapsed if elapsed > 0 else 0

            completed_trials = [
                t for t in study.trials
                if t.state == optuna.trial.TrialState.COMPLETE
            ]
            if not completed_trials:
                return

            sorted_trials = sorted(completed_trials, key=lambda t: t.value)
            top5 = sorted_trials[:5]

            print(f"\n  --- After {total} trials ({self._completed} complete, {self._pruned} pruned) "
                  f"[{elapsed:.0f}s, {rate:.1f} trials/s] ---")
            print(f"  Best Brier: {study.best_value:.6f}")
            print(LEADERBOARD_HEADER)
            print(LEADERBOARD_SEP)
            for t in top5:
                print(_format_trial_row(t))
            print()
            sys.stdout.flush()


def run_study(
    gender: str,
    n_trials: int = 400,
    n_jobs: int = 4,
    study_name: str = None,
    log_every: int = 25,
    enqueue: str = None,
):
    """Create and run an Optuna study."""

    if study_name is None:
        study_name = f"elo_{gender}"

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=10,
        n_warmup_steps=3,
    )

    study = optuna.create_study(
        study_name=study_name,
        direction="minimize",
        pruner=pruner,
    )

    # Enqueue seed params so Optuna evaluates them first
    if enqueue:
        with open(enqueue) as f:
            seed_data = json.load(f)
        # Support both raw param files and full output files with nested "params" key
        seed_params = seed_data.get("params", seed_data)
        study.enqueue_trial(seed_params)
        print(f"\n  Enqueued seed params from {enqueue}")

    print(f"\nStarting Optuna study: {study_name}")
    print(f"  Trials: {n_trials}, Jobs: {n_jobs}")
    print(f"  Seasons: {SEASON_START} -> {SEASON - 1}")
    print()

    # Warn if there are uncommitted changes
    try:
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip()
        if dirty:
            print("  WARNING: git working tree is dirty. Results won't map to a clean commit.")
            print(f"  Uncommitted changes:\n{dirty}\n")
            resp = input("  Continue anyway? [y/N] ").strip().lower()
            if resp != "y":
                print("  Aborted. Commit your changes first.")
                return None
    except Exception:
        pass

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    callback = ProgressCallback(every=log_every)

    start = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True, callbacks=[callback])
    elapsed = time.time() - start

    # Print results
    print(f"\n{'='*60}")
    print(f"  OPTUNA RESULTS ({gender.upper()})")
    print(f"{'='*60}")
    print(f"  Completed trials: {len(study.trials)}")
    print(f"  Time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"  Best Brier loss: {study.best_value:.6f}")
    print(f"\n  Best params:")
    for k, v in study.best_params.items():
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")
    print()

    # Run backtest with best params to get per-season breakdown
    best = study.best_params
    params_out = {
        "k": best["k"],
        "rating": best["rating"],
        "seed": best["seed"],
        "link": best["link"],
        "fgp": best["fgp"],
        "fgp3": best["fgp3"],
        "reb": best["reb"],
        "d": best["d"],
        "alpha": best["alpha"],
        "to_margin": best["to_margin"],
        "off_reb": best["off_reb"],
        "def_reb": best["def_reb"],
        "massey_rank": best["massey_rank"],
    }

    print(f"\n  Running backtest with best params...")
    best_elo = set_up_elo_model(
        k=params_out["k"], seed=params_out["seed"],
        link_function=link_function_list[params_out["link"]],
        fgp=params_out["fgp"], fgp3=params_out["fgp3"],
        r=params_out["reb"], rating=params_out["rating"],
        d=params_out["d"], alpha=params_out["alpha"],
        to_margin=params_out["to_margin"], off_reb=params_out["off_reb"],
        def_reb=params_out["def_reb"], massey_rank=params_out["massey_rank"],
    )
    best_predictions = run_system(best_elo, end_season=SEASON - 1)

    backtest = {}
    all_losses = []
    all_correct = []
    for season in range(SEASON_START, SEASON):
        sp = best_predictions.loc[
            best_predictions["Season"] == season, ["PredProbWTeam", "Stage"]
        ].copy()
        tourney = sp.query("Stage == 'T'")
        if tourney.empty:
            continue
        loss, correct = evaluate_by_season(sp)
        all_losses.append(loss)
        all_correct.append(correct)
        backtest[str(season)] = {
            "brier_loss": round(loss, 6),
            "correct_pct": round(correct, 4),
            "tournament_games": len(tourney),
        }

    avg_loss = sum(all_losses) / len(all_losses) if all_losses else None
    avg_correct = sum(all_correct) / len(all_correct) if all_correct else None

    # Get git hash
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        git_hash = "unknown"

    try:
        git_dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode().strip())
    except Exception:
        git_dirty = None

    # Build full output document
    timestamp = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
    output = {
        "params": params_out,
        "backtest": {
            "seasons": backtest,
            "mean_brier_loss": round(avg_loss, 6) if avg_loss else None,
            "mean_correct_pct": round(avg_correct, 4) if avg_correct else None,
            "season_range": f"{SEASON_START}-{SEASON - 1}",
            "num_seasons": len(all_losses),
        },
        "study": {
            "gender": gender,
            "n_trials": len(study.trials),
            "n_completed": len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]),
            "n_pruned": len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]),
            "best_trial_number": study.best_trial.number,
            "duration_seconds": round(elapsed, 1),
            "study_name": study_name,
        },
        "meta": {
            "timestamp": datetime.datetime.now().isoformat(),
            "git_hash": git_hash,
            "git_dirty": git_dirty,
            "database": os.environ.get("DATABASE_URL", "unknown"),
        },
    }

    # Save to candidate_params/
    os.makedirs("candidate_params", exist_ok=True)
    out_path = f"candidate_params/{timestamp}_optuna_best_{gender}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_path}")

    # Print top 10 trials
    print(f"\n  Top 10 trials:")
    print(LEADERBOARD_HEADER)
    print(LEADERBOARD_SEP)

    sorted_trials = sorted(
        [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value,
    )
    for t in sorted_trials[:10]:
        print(_format_trial_row(t))

    # Print backtest summary
    print(f"\n  Backtest per season:")
    print(f"  {'Season':<8} {'Brier':>10} {'Correct':>10} {'Games':>7}")
    print(f"  {'-'*37}")
    for season_str, bt in sorted(backtest.items()):
        print(f"  {season_str:<8} {bt['brier_loss']:>10.4f} {bt['correct_pct']:>9.1%} {bt['tournament_games']:>7}")
    print(f"  {'-'*37}")
    print(f"  {'AVERAGE':<8} {avg_loss:>10.4f} {avg_correct:>9.1%}")

    print()
    return study


def main():
    parser = argparse.ArgumentParser(description="Bayesian parameter optimization with Optuna")
    parser.add_argument(
        "--gender", type=str, required=True, choices=["mens", "womens"],
        help="Which gender to optimize",
    )
    parser.add_argument(
        "--n-trials", type=int, default=400,
        help="Number of Optuna trials (default: 400)",
    )
    parser.add_argument(
        "--n-jobs", type=int, default=4,
        help="Number of parallel trials (default: 4)",
    )
    parser.add_argument(
        "--study-name", type=str, default=None,
        help="Optuna study name (default: elo_<gender>)",
    )
    parser.add_argument(
        "--log-every", type=int, default=25,
        help="Print leaderboard every N trials (default: 25)",
    )
    parser.add_argument(
        "--enqueue", type=str, default=None,
        help="Path to a JSON file with params to evaluate first (seeds the study)",
    )
    args = parser.parse_args()

    run_study(
        gender=args.gender,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,
        study_name=args.study_name,
        log_every=args.log_every,
        enqueue=args.enqueue,
    )


if __name__ == "__main__":
    main()
