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

import numpy as np
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
    off_reb_rate = trial.suggest_float("off_reb_rate", 0, 5000)
    def_reb_rate = trial.suggest_float("def_reb_rate", 0, 5000)
    massey_rank = trial.suggest_float("massey_rank", -50, 0)
    decay = trial.suggest_float("decay", 0.5, 1.0)
    cal_a = trial.suggest_float("cal_a", 0.5, 2.0)
    cal_b = 0.0

    link_function = link_function_list[link]

    elo = set_up_elo_model(
        k=k, seed=seed, link_function=link_function,
        fgp=fgp, fgp3=fgp3, r=reb, rating=rating,
        d=d, alpha=alpha,
        to_margin=to_margin, off_reb_rate=off_reb_rate, def_reb_rate=def_reb_rate,
        massey_rank=massey_rank, decay=decay,
    )

    # Run all seasons and collect tournament Brier losses
    match_predictions = run_system(elo, end_season=SEASON - 1)

    # Apply Platt calibration: calibrated = sigmoid(a * logit(p) + b)
    if cal_a != 1.0 or cal_b != 0.0:
        p = match_predictions["PredProbWTeam"].clip(0.001, 0.999)
        logit_p = np.log(p / (1 - p))
        match_predictions["PredProbWTeam"] = 1 / (1 + np.exp(-(cal_a * logit_p + cal_b)))

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
        f"{p['d']:>7.0f} {p['alpha']:>6.3f} {p['to_margin']:>6.0f} {p['off_reb_rate']:>6.0f} {p['def_reb_rate']:>6.0f} {p['massey_rank']:>6.1f} {p['decay']:>5.2f}"
        f" {p['cal_a']:>5.2f}"
    )


LEADERBOARD_HEADER = (
    f"  {'#':>4} {'Brier':>10} {'k':>5} {'seed':>7} {'link':>4} {'fgp':>8} "
    f"{'fgp3':>8} {'reb':>7} {'rating':>7} {'d':>7} {'alpha':>6} "
    f"{'TO_m':>6} {'ORr':>6} {'DRr':>6} {'mRank':>6} {'decay':>5} {'calA':>5}"
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
        # Fill in defaults for any missing params (e.g. old 9-param files)
        defaults = {"to_margin": 0.0, "off_reb_rate": 0.0, "def_reb_rate": 0.0, "massey_rank": 0.0, "decay": 1.0, "cal_a": 1.0}
        # Fix zero values for log-scale params (Optuna can't enqueue 0 for log distributions)
        log_params_min = {"fgp": 500, "reb": 5, "rating": 0.1}
        for p, minval in log_params_min.items():
            if p in seed_params and seed_params[p] == 0:
                seed_params[p] = minval
        for k, v in defaults.items():
            if k not in seed_params:
                seed_params[k] = v
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
        "off_reb_rate": best["off_reb_rate"],
        "def_reb_rate": best["def_reb_rate"],
        "massey_rank": best["massey_rank"],
        "decay": best["decay"],
        "cal_a": best["cal_a"],
        "cal_b": 0.0,
    }

    print(f"\n  Running backtest with best params...")
    best_elo = set_up_elo_model(
        k=params_out["k"], seed=params_out["seed"],
        link_function=link_function_list[params_out["link"]],
        fgp=params_out["fgp"], fgp3=params_out["fgp3"],
        r=params_out["reb"], rating=params_out["rating"],
        d=params_out["d"], alpha=params_out["alpha"],
        to_margin=params_out["to_margin"], off_reb_rate=params_out["off_reb_rate"],
        def_reb_rate=params_out["def_reb_rate"], massey_rank=params_out["massey_rank"],
        decay=params_out["decay"],
    )
    best_predictions = run_system(best_elo, end_season=SEASON - 1)

    # Apply calibration to backtest predictions
    best_cal_a = params_out["cal_a"]
    best_cal_b = params_out["cal_b"]
    if best_cal_a != 1.0 or best_cal_b != 0.0:
        p = best_predictions["PredProbWTeam"].clip(0.001, 0.999)
        logit_p = np.log(p / (1 - p))
        best_predictions["PredProbWTeam"] = 1 / (1 + np.exp(-(best_cal_a * logit_p + best_cal_b)))

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

    # Run 2026 bracket simulation with best params
    print(f"\n  Running 2026 bracket simulation with best params...")
    try:
        # Write best params to default_params.json temporarily
        try:
            with open("default_params.json") as f:
                original_defaults = f.read()
        except FileNotFoundError:
            original_defaults = None

        with open("default_params.json", "w") as f:
            json.dump({gender: params_out}, f, indent=2)

        result = subprocess.run(
            ["python", "generate_submission.py", "--gender",
             "M" if gender == "mens" else "W",
             "--description", f"optuna_{gender}_{timestamp}"],
            capture_output=True, text=True, timeout=300,
            env={**os.environ},
        )
        # Print only the simulation output (from MONTE CARLO onwards)
        output_lines = result.stdout.split("\n")
        in_sim = False
        in_matchups = False
        for line in output_lines:
            if "FIRST ROUND MATCHUP" in line:
                in_matchups = True
            if "MONTE CARLO" in line:
                in_matchups = False
                in_sim = True
            if in_matchups or in_sim:
                print(line)
        if result.returncode != 0 and not in_sim:
            print(f"  Warning: generate_submission exited with code {result.returncode}")
            if result.stderr:
                print(f"  {result.stderr[:500]}")
    except Exception as e:
        print(f"  Warning: bracket simulation failed: {e}")
    finally:
        # Restore original default_params.json
        if original_defaults is not None:
            with open("default_params.json", "w") as f:
                f.write(original_defaults)

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
