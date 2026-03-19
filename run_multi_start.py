"""
Multi-start optimization: run multiple Optuna studies from different
starting regions of the param space to escape local minima.

Each start point is a hand-picked region representing a different
"theory" about what the model should look like.
"""
import datetime
import json
import os
import sys
import time

import numpy as np
import optuna
import pandas as pd

from elo_run.evaluation import evaluate_by_season
from elo_run.param_tuning import set_up_elo_model, link_function_list, run_system
from models import SEASON, SEASON_START

optuna.logging.set_verbosity(optuna.logging.WARNING)


def objective(trial):
    k = trial.suggest_int("k", 10, 400)
    seed = trial.suggest_float("seed", -80, 0)
    link = trial.suggest_categorical("link", ["N", "B", "L"])
    fgp = trial.suggest_float("fgp", 500, 8000, log=True)
    fgp3 = trial.suggest_float("fgp3", -100, 8000)
    reb = trial.suggest_float("reb", 5, 600, log=True)
    rating = trial.suggest_float("rating", 0.1, 10, log=True)
    d = trial.suggest_float("d", 200, 1400)
    alpha = trial.suggest_float("alpha", 0.0, 1.0)
    to_margin = trial.suggest_float("to_margin", 0, 500)
    off_reb_rate = trial.suggest_float("off_reb_rate", 0, 5000)
    def_reb_rate = trial.suggest_float("def_reb_rate", 0, 5000)
    massey_rank = trial.suggest_float("massey_rank", -50, 0)
    decay = trial.suggest_float("decay", 0.5, 1.0)
    cal_a = trial.suggest_float("cal_a", 0.5, 2.0)

    link_fn = link_function_list[link]
    elo = set_up_elo_model(
        k=k, seed=seed, link_function=link_fn,
        fgp=fgp, fgp3=fgp3, r=reb, rating=rating,
        d=d, alpha=alpha,
        to_margin=to_margin, off_reb_rate=off_reb_rate,
        def_reb_rate=def_reb_rate, massey_rank=massey_rank,
        decay=decay,
    )

    match_predictions = run_system(elo, end_season=SEASON - 1)

    # Apply calibration
    if cal_a != 1.0:
        p = match_predictions["PredProbWTeam"].clip(0.001, 0.999)
        logit_p = np.log(p / (1 - p))
        match_predictions["PredProbWTeam"] = 1 / (1 + np.exp(-(cal_a * logit_p)))

    losses = []
    step = 0
    for season in range(SEASON_START, SEASON):
        sp = match_predictions.loc[
            match_predictions["Season"] == season, ["PredProbWTeam", "Stage"]
        ].copy()
        tourney = sp.query("Stage == 'T'")
        if tourney.empty:
            continue
        loss, _ = evaluate_by_season(sp)
        losses.append(loss)
        step += 1
        trial.report(sum(losses) / len(losses), step)
        if trial.should_prune():
            raise optuna.TrialPruned()

    return sum(losses) / len(losses) if losses else float("inf")


# Different starting theories about what the model should look like
STARTS_MENS = {
    "current_best": {
        "k": 295, "seed": -57, "link": "N", "fgp": 1587, "fgp3": 67,
        "reb": 8, "rating": 0.57, "d": 1199, "alpha": 0.35,
        "to_margin": 0, "off_reb_rate": 0, "def_reb_rate": 0,
        "massey_rank": 0, "decay": 1.0, "cal_a": 1.0,
    },
    "minimal_massey_decay": {
        "k": 35, "seed": -59, "link": "N", "fgp": 500, "fgp3": 0,
        "reb": 5, "rating": 5.9, "d": 1129, "alpha": 0.44,
        "to_margin": 0, "off_reb_rate": 0, "def_reb_rate": 0,
        "massey_rank": -10, "decay": 0.71, "cal_a": 1.0,
    },
    "high_k_logistic": {
        "k": 350, "seed": -40, "link": "L", "fgp": 2000, "fgp3": 500,
        "reb": 20, "rating": 1.0, "d": 800, "alpha": 0.5,
        "to_margin": 100, "off_reb_rate": 500, "def_reb_rate": 500,
        "massey_rank": -20, "decay": 0.8, "cal_a": 1.2,
    },
    "low_k_bilogistic": {
        "k": 20, "seed": -70, "link": "B", "fgp": 3000, "fgp3": 1000,
        "reb": 50, "rating": 3.0, "d": 600, "alpha": 0.2,
        "to_margin": 200, "off_reb_rate": 1000, "def_reb_rate": 1000,
        "massey_rank": -30, "decay": 0.6, "cal_a": 0.8,
    },
    "pure_elo_sharp": {
        "k": 50, "seed": -50, "link": "N", "fgp": 500, "fgp3": 0,
        "reb": 5, "rating": 8.0, "d": 1400, "alpha": 0.0,
        "to_margin": 0, "off_reb_rate": 0, "def_reb_rate": 0,
        "massey_rank": -5, "decay": 0.9, "cal_a": 1.5,
    },
    "win_loss_heavy": {
        "k": 150, "seed": -60, "link": "N", "fgp": 1000, "fgp3": 200,
        "reb": 10, "rating": 2.0, "d": 1000, "alpha": 0.9,
        "to_margin": 50, "off_reb_rate": 200, "def_reb_rate": 200,
        "massey_rank": -15, "decay": 0.75, "cal_a": 1.1,
    },
    "box_score_heavy": {
        "k": 100, "seed": -30, "link": "N", "fgp": 5000, "fgp3": 3000,
        "reb": 5, "rating": 0.3, "d": 900, "alpha": 0.3,
        "to_margin": 300, "off_reb_rate": 3000, "def_reb_rate": 3000,
        "massey_rank": -25, "decay": 0.85, "cal_a": 1.0,
    },
    "aggressive_decay": {
        "k": 80, "seed": -55, "link": "N", "fgp": 1500, "fgp3": 100,
        "reb": 15, "rating": 4.0, "d": 1100, "alpha": 0.4,
        "to_margin": 0, "off_reb_rate": 0, "def_reb_rate": 0,
        "massey_rank": -12, "decay": 0.55, "cal_a": 1.0,
    },
}

STARTS_WOMENS = {
    "current_best": {
        "k": 328, "seed": -78, "link": "L", "fgp": 921, "fgp3": 686,
        "reb": 6, "rating": 0.51, "d": 933, "alpha": 0.0,
        "to_margin": 0, "off_reb_rate": 0, "def_reb_rate": 0,
        "massey_rank": 0, "decay": 1.0, "cal_a": 1.0,
    },
    "high_k_normal": {
        "k": 350, "seed": -60, "link": "N", "fgp": 1500, "fgp3": 500,
        "reb": 20, "rating": 1.0, "d": 1000, "alpha": 0.3,
        "to_margin": 100, "off_reb_rate": 500, "def_reb_rate": 500,
        "massey_rank": 0, "decay": 0.8, "cal_a": 1.0,
    },
    "low_k_margin": {
        "k": 40, "seed": -50, "link": "L", "fgp": 2000, "fgp3": 1000,
        "reb": 10, "rating": 5.0, "d": 700, "alpha": 0.0,
        "to_margin": 200, "off_reb_rate": 1000, "def_reb_rate": 1000,
        "massey_rank": 0, "decay": 0.7, "cal_a": 1.2,
    },
    "bilogistic_decay": {
        "k": 200, "seed": -70, "link": "B", "fgp": 3000, "fgp3": 200,
        "reb": 50, "rating": 2.0, "d": 500, "alpha": 0.5,
        "to_margin": 50, "off_reb_rate": 200, "def_reb_rate": 200,
        "massey_rank": 0, "decay": 0.6, "cal_a": 0.8,
    },
    "pure_elo": {
        "k": 100, "seed": -80, "link": "N", "fgp": 500, "fgp3": 0,
        "reb": 5, "rating": 8.0, "d": 1200, "alpha": 0.4,
        "to_margin": 0, "off_reb_rate": 0, "def_reb_rate": 0,
        "massey_rank": 0, "decay": 0.9, "cal_a": 1.5,
    },
    "box_score_heavy": {
        "k": 250, "seed": -40, "link": "L", "fgp": 5000, "fgp3": 3000,
        "reb": 5, "rating": 0.5, "d": 800, "alpha": 0.1,
        "to_margin": 300, "off_reb_rate": 3000, "def_reb_rate": 3000,
        "massey_rank": 0, "decay": 0.85, "cal_a": 1.0,
    },
}


def run_from_start(name, start_params, n_trials=60):
    study = optuna.create_study(
        study_name=name, direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
    )
    study.enqueue_trial(start_params)
    start = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=False)
    elapsed = time.time() - start

    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    return {
        "name": name,
        "best_brier": study.best_value,
        "best_trial": study.best_trial.number,
        "best_params": study.best_params,
        "n_completed": len(completed),
        "duration_s": round(elapsed, 1),
    }


def main():
    gender = sys.argv[1] if len(sys.argv) > 1 else "mens"
    n_trials = int(sys.argv[2]) if len(sys.argv) > 2 else 60

    starts = STARTS_MENS if gender == "mens" else STARTS_WOMENS

    print(f"\n{'='*70}")
    print(f"  MULTI-START OPTIMIZATION: {gender.upper()}")
    print(f"  {len(starts)} starting points, {n_trials} trials each")
    print(f"{'='*70}\n")

    results = []
    total_start = time.time()

    for i, (name, params) in enumerate(starts.items()):
        print(f"  [{i+1}/{len(starts)}] {name} ...", end=" ", flush=True)
        result = run_from_start(f"{gender}_{name}", params, n_trials=n_trials)
        results.append(result)
        print(f"best={result['best_brier']:.6f}  trial#{result['best_trial']}  [{result['duration_s']:.0f}s]")

    total_elapsed = time.time() - total_start

    # Leaderboard
    sorted_results = sorted(results, key=lambda r: r["best_brier"])
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD ({gender.upper()})")
    print(f"{'='*70}")
    for i, r in enumerate(sorted_results):
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:>2}. {r['name']:<35} {r['best_brier']:.6f}  trial#{r['best_trial']}{marker}")

    print(f"\n  Total: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save
    timestamp = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
    best = sorted_results[0]

    output = {
        "params": best["best_params"],
        "backtest": {"mean_brier_loss": best["best_brier"]},
        "study": {
            "gender": gender,
            "strategy": "multi_start",
            "winning_start": best["name"],
            "n_starts": len(starts),
            "n_trials_per_start": n_trials,
            "duration_seconds": round(total_elapsed, 1),
        },
        "all_results": [
            {"name": r["name"], "brier": r["best_brier"], "trial": r["best_trial"]}
            for r in sorted_results
        ],
    }

    os.makedirs("candidate_params", exist_ok=True)
    out_path = f"candidate_params/{timestamp}_multistart_{gender}.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved to {out_path}")

    # Print best params
    print(f"\n  Best params ({best['name']}):")
    for k, v in sorted(best["best_params"].items()):
        if isinstance(v, float):
            print(f"    {k}: {v:.4f}")
        else:
            print(f"    {k}: {v}")


if __name__ == "__main__":
    main()
