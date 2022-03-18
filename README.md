# March Madness

This project is an attempt to model the infamously unpredictable march madness tournament

## Background

March Madness are two annual end of season college basketball tournament, used to determine the national champions for both Men and Women.

Every year millions around the country fill out brackets to predict the winners of the tournament, everyone employing different ways of picking winners, from form guides and expert opinions to which mascot you prefer. 

The vast nature of data collected during sports games leaves the tounrament perfectly poised to apply some data science and predict results. Kaggle runs a competition ([mens](https://www.kaggle.com/c/mens-march-mania-2022), [womens](https://www.kaggle.com/c/womens-march-mania-2022/leaderboard)) for data scienctists to predict the results of the tournament.

The evaluation used to determine competition winners is the sum of the log loss of predictions vs results. 

For each possible match, you provide a probability of Team A beating Team B, and the score after the result is the log loss from the result [0,1] vs. the prediction. This penalises confident guesses being wrong harshly

This project implements an [elo type system](https://en.wikipedia.org/wiki/Elo_rating_system) most famously used for chess world rankings, to give team's ratings over time and predict tournament results.

It uses SQLAlchemy and sqlite3 to handle the databases - data is downloaded from Kaggle's March Madness competition website, credit to them for the datasets

## Getting Started

1. To get started with the project, clone from github.
2. Download up to date data from the most recent [Kaggle competition](https://www.kaggle.com/competitions) making sure to put men's data in the `data_male` folder and womens in the `data_female` folder

3. Install requirements
```bash
pip install -r requirements/base.txt
```

4. Load data into databases

```bash
# Initialise database
sqlite3 DATABASE

- SET env var DATABASE_URL

# THis will load all initial data and migrate databases to the initial state using SQLAlchemy
python database_scripts/load_all.py
python database_scripts/load_seeds.py
python database_scripts/load_matches_teams_map.py

# Now upgrade database again
alembic upgrade head

Now run the sql in `score_diff.sql` on your database

# Finally run the last python script
python database_scripts/update_all_match_stats.py
```

## Building Models
### Testing Parameters to the ELO Model
To run tests for a grid of parameters, update the parameters you want to test in `elo_run/param_tuning.py` and then run `python elo_run/param_tuning.py`

This will run tests for params specified and save results to the `evaluations` table in the db

### Evaluating param sets
The notebook `notebooks/Evaluate systems.ipynb` allows you to evaluate different sets of params and pick a set for your final model

### Creating submission files for the kaggle competition
The notebook `notebooks/Season-predictions-Updated-Ratings.ipynb` allows you to get a submission file for the Kaggle competition
It uses final ELO ratings and then updates ratings probabilistically based on your predictions for each round before predicting the next round
