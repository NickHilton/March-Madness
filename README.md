# march_madness

This project is an attempt to model the infamously unpredictable march madness tournament

It implements an elo type system most famously used for chess world rankings
https://en.wikipedia.org/wiki/Elo_rating_system

It uses SQLAlchemy and sqlite3 to handle the data - data downloaded from Kaggle's March Madness competition website, credit to them for the datasets

You will find the ORM models in `model_definitions`, the elo implementation in `elo_run` and some database loading scripts in `database_scripts`

I also used this to perform a very similar analysis on the Women's competition. Any comments and suggestions are welcome, particularly with the problem of optimising the parameters

This is an adaption of a larger project I have to predict football scores and build a fantasy football model of which I have written up a report of my analysis here
https://drive.google.com/open?id=0BwOFJMTMSjbyaE9KQ0RlQUxBc28


