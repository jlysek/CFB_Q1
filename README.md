# CFB Quarter 1 Score Predictor

A Bayesian statistical model for predicting college football first quarter exact scores using pregame betting lines (spread and total).

## Features

- Empirical probability distributions from historical college football data
- Bayesian updating using Vegas betting line information
- Football-specific scoring pattern recognition (TDs, FGs, safeties)
- Multinomial logistic regression with hierarchical priors
- Interactive prediction interface
- Comprehensive confidence assessment

## Requirements

- Python 3.8+
- MySQL database with college football data
- Required packages: `pip install -r requirements.txt`

## Database Schema

Requires two MySQL tables:
- `cfb.games`: Game data with pregame_spread, pregame_total
- `cfb.quarter_scoring`: Quarter-by-quarter scoring data

## Usage
```python
python cfb_quarter_predictor.py