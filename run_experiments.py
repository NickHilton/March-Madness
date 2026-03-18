"""
Systematic experiment runner: test different feature combinations to find
what actually improves the model vs adds noise.

Each experiment defines which params to search and which to fix at 0.
Runs 75 trials per experiment with the old baseline enqueued as trial 0.
"""
import datetime
import json
import os
import sys
import time

import optuna

from elo_run.evaluation import evaluate_by_season
from elo_run.param_tuning import set_up_elo_model, link_function_list, run_system
from models import SEASON, SEASON_START

optuna.logging.set_verbosity(optuna.logging.WARNING)

# Baseline params (old 9-param best)
BASELINE = {
    "k": 295, "seed": -57.41, "link": "N", "fgp": 1587.0, "fgp3": 67.0,
    "reb": 8.1, "rating": 0.57, "d": 1199.3, "alpha": 0.347,
}

# Define experiments: each is a dict of {param_name: search_range or fixed_value}
# If value is a tuple, it's a search range. If a single value, it's fixed.
EXPERIMENTS = {
    # === BASELINES ===
    "01_baseline_9param": {
        # Original 9 params, no new features
    },
    "02_baseline_adjusted_fgp": {
        # Same 9 params but now using adjusted FGP/FGP3 (data changed)
    },

    # === SINGLE FEATURE ADDITIONS ===
    "03_add_decay_only": {
        "decay": (0.5, 1.0),
    },
    "04_add_massey_only": {
        "massey_rank": (-50, 0),
    },
    "05_add_to_margin_only": {
        "to_margin": (0, 500),
    },
    "06_add_off_reb_rate_only": {
        "off_reb_rate": (0, 5000),
    },
    "07_add_def_reb_rate_only": {
        "def_reb_rate": (0, 5000),
    },

    # === FEATURE COMBINATIONS ===
    "08_decay_plus_massey": {
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
    },
    "09_decay_plus_to_margin": {
        "decay": (0.5, 1.0),
        "to_margin": (0, 500),
    },
    "10_massey_plus_to_margin": {
        "massey_rank": (-50, 0),
        "to_margin": (0, 500),
    },
    "11_decay_massey_to_margin": {
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
        "to_margin": (0, 500),
    },
    "12_all_new_features": {
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
        "to_margin": (0, 500),
        "off_reb_rate": (0, 5000),
        "def_reb_rate": (0, 5000),
    },

    # === ARCHITECTURAL CHANGES ===
    "13_drop_fgp3": {
        # Fix FGP3 to 0 - was always weakly predictive
        "fgp3": 0.0,
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
    },
    "14_drop_reb": {
        # Fix reb to 0 - test if rebounds matter with adjusted FGP
        "reb": 0.0,
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
    },
    "15_minimal_rating_massey_decay": {
        # Minimal model: just Elo rating + massey + decay
        "fgp": 0.0, "fgp3": 0.0, "reb": 0.0,
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
    },
    "16_force_normal_link": {
        # Force normal link function (was best in old model)
        "link": "N",
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
        "to_margin": (0, 500),
    },
    "17_force_logistic_link": {
        "link": "L",
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
        "to_margin": (0, 500),
    },
    "18_high_alpha_win_loss": {
        # Force alpha > 0.5 (lean toward win/loss over margin)
        "alpha": (0.5, 1.0),
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
    },
    "19_low_alpha_margin": {
        # Force alpha < 0.3 (lean toward margin-based)
        "alpha": (0.0, 0.3),
        "decay": (0.5, 1.0),
        "massey_rank": (-50, 0),
    },
    "20_kitchen_sink_narrow": {
        # All features but with tighter search ranges around known good values
        "k": (100, 400),
        "seed": (-70, -30),
        "fgp": (500, 3000),
        "fgp3": (0, 500),
        "reb": (2, 50),
        "rating": (0.1, 3.0),
        "d": (800, 1400),
        "alpha": (0.1, 0.6),
        "decay": (0.7, 1.0),
        "massey_rank": (-30, 0),
        "to_margin": (0, 200),
        "off_reb_rate": (0, 2000),
        "def_reb_rate": (0, 2000),
    },
}

# Default search ranges for the 9 base params
DEFAULT_RANGES = {
    "k": ("int", 10, 400),
    "seed": ("float", -80, 0),
    "link": ("cat", ["N", "B", "L"]),
    "fgp": ("log", 500, 8000),
    "fgp3": ("float", -100, 8000),
    "reb": ("log", 5, 600),
    "rating": ("log", 0.1, 10),
    "d": ("float", 200, 1200),
    "alpha": ("float", 0.0, 1.0),
}

# New feature default = fixed at 0
NEW_FEATURES = {
    "to_margin": 0.0,
    "off_reb_rate": 0.0,
    "def_reb_rate": 0.0,
    "massey_rank": 0.0,
    "decay": 1.0,
}


def make_objective(experiment_config):
    """Create an objective function with the given fixed/search params."""

    def objective(trial):
        params = {}

        # Base params: search unless overridden
        for name, (ptype, *args) in DEFAULT_RANGES.items():
            if name in experiment_config:
                val = experiment_config[name]
                if isinstance(val, tuple):
                    # Override search range
                    if ptype == "int":
                        params[name] = trial.suggest_int(name, int(val[0]), int(val[1]))
                    elif ptype == "cat":
                        params[name] = trial.suggest_categorical(name, val)
                    elif ptype == "log":
                        params[name] = trial.suggest_float(name, val[0], val[1], log=True)
                    else:
                        params[name] = trial.suggest_float(name, val[0], val[1])
                elif isinstance(val, str):
                    # Fixed categorical
                    params[name] = val
                else:
                    # Fixed value
                    params[name] = val
            else:
                # Default search range
                if ptype == "int":
                    params[name] = trial.suggest_int(name, args[0], args[1])
                elif ptype == "cat":
                    params[name] = trial.suggest_categorical(name, args[0])
                elif ptype == "log":
                    params[name] = trial.suggest_float(name, args[0], args[1], log=True)
                else:
                    params[name] = trial.suggest_float(name, args[0], args[1])

        # New features: fixed at 0 unless in experiment config
        for name, default in NEW_FEATURES.items():
            if name in experiment_config:
                val = experiment_config[name]
                if isinstance(val, tuple):
                    params[name] = trial.suggest_float(name, val[0], val[1])
                else:
                    params[name] = val
            else:
                params[name] = default

        # Build and run model
        link_fn = link_function_list[params["link"]]
        elo = set_up_elo_model(
            k=params["k"], seed=params["seed"], link_function=link_fn,
            fgp=params["fgp"], fgp3=params["fgp3"], r=params["reb"],
            rating=params["rating"], d=params["d"], alpha=params["alpha"],
            to_margin=params["to_margin"], off_reb_rate=params["off_reb_rate"],
            def_reb_rate=params["def_reb_rate"], massey_rank=params["massey_rank"],
            decay=params["decay"],
        )

        match_predictions = run_system(elo, end_season=SEASON - 1)

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

    return objective


def run_experiment(name, config, n_trials=75):
    """Run a single experiment and return results."""
    study = optuna.create_study(
        study_name=name,
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=8, n_warmup_steps=3),
    )

    # Enqueue baseline
    baseline_trial = dict(BASELINE)
    for feat, default in NEW_FEATURES.items():
        baseline_trial[feat] = default
    # Apply fixed values from config
    for k, v in config.items():
        if not isinstance(v, tuple):
            baseline_trial[k] = v
    study.enqueue_trial(baseline_trial)

    objective = make_objective(config)
    start = time.time()
    study.optimize(objective, n_trials=n_trials, n_jobs=4, show_progress_bar=False)
    elapsed = time.time() - start

    # Get results
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    baseline_brier = completed[0].value if completed else None

    return {
        "name": name,
        "best_brier": study.best_value,
        "baseline_brier": baseline_brier,
        "improvement": baseline_brier - study.best_value if baseline_brier else None,
        "best_trial": study.best_trial.number,
        "n_completed": len(completed),
        "n_pruned": len(pruned),
        "duration_s": round(elapsed, 1),
        "best_params": study.best_params,
        "config": {k: str(v) for k, v in config.items()},
    }


def main():
    n_trials = int(sys.argv[1]) if len(sys.argv) > 1 else 75
    print(f"\n{'='*70}")
    print(f"  EXPERIMENT RUNNER: {len(EXPERIMENTS)} experiments, {n_trials} trials each")
    print(f"{'='*70}\n")

    results = []
    total_start = time.time()

    for i, (name, config) in enumerate(EXPERIMENTS.items()):
        print(f"  [{i+1}/{len(EXPERIMENTS)}] {name} ...", end=" ", flush=True)
        result = run_experiment(name, config, n_trials=n_trials)
        results.append(result)

        imp = result["improvement"]
        imp_str = f"{imp:+.6f}" if imp is not None else "N/A"
        beat = " ***" if imp and imp > 0.001 else ""
        print(f"best={result['best_brier']:.6f}  baseline={result['baseline_brier']:.6f}  "
              f"imp={imp_str}  trial#{result['best_trial']}  "
              f"[{result['duration_s']:.0f}s, {result['n_completed']}c/{result['n_pruned']}p]{beat}")

    total_elapsed = time.time() - total_start

    # Print summary leaderboard
    print(f"\n{'='*70}")
    print(f"  LEADERBOARD (sorted by best Brier loss)")
    print(f"{'='*70}")
    print(f"  {'#':>2} {'Experiment':<35} {'Brier':>10} {'vs Base':>10} {'Trial':>6}")
    print(f"  {'-'*65}")

    sorted_results = sorted(results, key=lambda r: r["best_brier"])
    for i, r in enumerate(sorted_results):
        imp = r["improvement"]
        imp_str = f"{imp:+.6f}" if imp is not None else ""
        marker = " <-- BEST" if i == 0 else ""
        print(f"  {i+1:>2} {r['name']:<35} {r['best_brier']:>10.6f} {imp_str:>10} {r['best_trial']:>6}{marker}")

    print(f"\n  Total time: {total_elapsed:.0f}s ({total_elapsed/60:.1f} min)")

    # Save results
    timestamp = datetime.datetime.now().strftime("%y%m%dT%H%M%S")
    out_path = f"candidate_params/{timestamp}_experiments.json"
    os.makedirs("candidate_params", exist_ok=True)

    # Also save the best experiment's params as a candidate
    best = sorted_results[0]
    output = {
        "leaderboard": [
            {"rank": i+1, "name": r["name"], "brier": r["best_brier"],
             "improvement": r["improvement"], "best_trial": r["best_trial"]}
            for i, r in enumerate(sorted_results)
        ],
        "experiments": results,
        "best_experiment": best["name"],
        "best_params": best["best_params"],
        "best_brier": best["best_brier"],
        "n_trials_per_experiment": n_trials,
        "total_duration_s": round(total_elapsed, 1),
    }

    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  Results saved to {out_path}")

    # Save best params as a standalone candidate
    best_params_path = f"candidate_params/{timestamp}_best_experiment_{best['name']}.json"
    best_out = {"params": best["best_params"]}
    # Add fixed values from the config
    config = EXPERIMENTS[best["name"]]
    for k, v in config.items():
        if not isinstance(v, tuple) and k not in best_out["params"]:
            best_out["params"][k] = v
    for feat, default in NEW_FEATURES.items():
        if feat not in best_out["params"]:
            best_out["params"][feat] = default

    with open(best_params_path, "w") as f:
        json.dump(best_out, f, indent=2)
    print(f"  Best params saved to {best_params_path}")


if __name__ == "__main__":
    main()
